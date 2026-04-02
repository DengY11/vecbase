#include "server/subreactor.h"

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "server/commands.h"
#include "server/resp.h"
#include "server/thread_pool.h"
#include "server/util.h"
#include "vecbase/db.h"

namespace vecbase_server {
namespace {

struct ConnToken {
  int fd = -1;
  std::uint64_t id = 0;
};

struct ResponseMsg {
  ConnToken token;
  std::string payload;
  bool close_after = false;
};

} // namespace

struct SubReactor::Impl {
  Impl(std::size_t index, vecbase::DB *db, const ServerConfig *cfg,
       ThreadPool *workers)
      : index_(index), db_(db), cfg_(cfg), workers_(workers) {}

  void Start() { thread_ = std::thread([this] { Loop(); }); }

  void Stop() {
    stopping_.store(true, std::memory_order_relaxed);
    Notify();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void EnqueueNewConn(int fd) {
    {
      std::lock_guard lock(new_conn_mutex_);
      pending_new_.push_back(fd);
    }
    Notify();
  }

private:
  struct Connection {
    ConnToken token;
    std::string in;
    std::size_t in_pos = 0;
    std::string out;
    std::size_t out_pos = 0;
    bool close_after_write = false;
    bool in_flight = false;
    std::deque<std::vector<std::string>> queue;
  };

  void Notify() {
    if (!event_fd_.valid()) {
      return;
    }
    std::uint64_t one = 1;
    const ssize_t n = ::write(event_fd_.get(), &one, sizeof(one));
    (void)n;
  }

  void EnqueueResponse(ResponseMsg msg) {
    {
      std::lock_guard lock(response_mutex_);
      pending_resp_.push_back(std::move(msg));
    }
    Notify();
  }

  void Loop() {
    try {
      event_fd_.reset(::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC));
      if (!event_fd_.valid()) {
        ThrowSys("eventfd");
      }
      epoll_fd_.reset(::epoll_create1(EPOLL_CLOEXEC));
      if (!epoll_fd_.valid()) {
        ThrowSys("epoll_create1");
      }

      epoll_event ev{};
      ev.events = EPOLLIN;
      ev.data.fd = event_fd_.get();
      if (::epoll_ctl(epoll_fd_.get(), EPOLL_CTL_ADD, event_fd_.get(), &ev) <
          0) {
        ThrowSys("epoll_ctl(ADD eventfd)");
      }

      std::vector<epoll_event> events(256);
      while (!stopping_.load(std::memory_order_relaxed)) {
        const int n = ::epoll_wait(epoll_fd_.get(), events.data(),
                                   static_cast<int>(events.size()), 1000);
        if (n < 0) {
          if (errno == EINTR) {
            continue;
          }
          ThrowSys("epoll_wait");
        }
        for (int i = 0; i < n; ++i) {
          const int fd = events[i].data.fd;
          const std::uint32_t current = events[i].events;
          if (fd == event_fd_.get()) {
            DrainEventFd();
            ProcessPending();
            continue;
          }
          auto it = connections_.find(fd);
          if (it == connections_.end()) {
            continue;
          }
          if (current & (EPOLLERR | EPOLLHUP | EPOLLRDHUP)) {
            CloseConn(it->second);
            connections_.erase(it);
            continue;
          }
          if ((current & EPOLLIN) && !HandleRead(it->second)) {
            CloseConn(it->second);
            connections_.erase(it);
            continue;
          }
          if ((current & EPOLLOUT) && !HandleWrite(it->second)) {
            CloseConn(it->second);
            connections_.erase(it);
          }
        }
        ProcessPending();
      }

      for (auto &[fd, conn] : connections_) {
        (void)fd;
        CloseConn(conn);
      }
      connections_.clear();
    } catch (const std::exception &ex) {
      std::cerr << "[subreactor " << index_ << "] fatal: " << ex.what()
                << "\n";
    }
  }

  void DrainEventFd() {
    std::uint64_t value = 0;
    for (;;) {
      const ssize_t n = ::read(event_fd_.get(), &value, sizeof(value));
      if (n == sizeof(value)) {
        continue;
      }
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        return;
      }
      return;
    }
  }

  void ProcessPending() {
    std::deque<int> new_conns;
    {
      std::lock_guard lock(new_conn_mutex_);
      new_conns.swap(pending_new_);
    }
    for (int fd : new_conns) {
      AddConn(fd);
    }

    std::deque<ResponseMsg> responses;
    {
      std::lock_guard lock(response_mutex_);
      responses.swap(pending_resp_);
    }
    for (const ResponseMsg &msg : responses) {
      ApplyResponse(msg);
    }
  }

  void AddConn(int fd) {
    Connection conn;
    conn.token.fd = fd;
    conn.token.id = ++next_conn_id_;

    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLRDHUP;
    ev.data.fd = fd;
    if (::epoll_ctl(epoll_fd_.get(), EPOLL_CTL_ADD, fd, &ev) < 0) {
      ::close(fd);
      return;
    }
    connections_.emplace(fd, std::move(conn));
  }

  void CloseConn(Connection &conn) {
    if (conn.token.fd >= 0) {
      ::epoll_ctl(epoll_fd_.get(), EPOLL_CTL_DEL, conn.token.fd, nullptr);
      ::close(conn.token.fd);
      conn.token.fd = -1;
    }
  }

  void ApplyResponse(const ResponseMsg &msg) {
    auto it = connections_.find(msg.token.fd);
    if (it == connections_.end()) {
      return;
    }
    Connection &conn = it->second;
    if (conn.token.id != msg.token.id) {
      return;
    }
    conn.out.append(msg.payload);
    if (msg.close_after) {
      conn.close_after_write = true;
    }
    conn.in_flight = false;
    UpdateInterest(conn);
    MaybeDispatch(conn);
  }

  void UpdateInterest(const Connection &conn) {
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLRDHUP;
    if (conn.out_pos < conn.out.size()) {
      ev.events |= EPOLLOUT;
    }
    ev.data.fd = conn.token.fd;
    ::epoll_ctl(epoll_fd_.get(), EPOLL_CTL_MOD, conn.token.fd, &ev);
  }

  bool HandleRead(Connection &conn) {
    char buffer[16 * 1024];
    for (;;) {
      const ssize_t n = ::read(conn.token.fd, buffer, sizeof(buffer));
      if (n > 0) {
        conn.in.append(buffer, static_cast<std::size_t>(n));
        continue;
      }
      if (n == 0) {
        return false;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      }
      if (errno == EINTR) {
        continue;
      }
      return false;
    }

    for (;;) {
      std::string_view view(conn.in);
      view.remove_prefix(conn.in_pos);
      std::size_t consumed = 0;
      std::vector<std::string> args;
      std::string error;
      const resp::ParseResult parsed =
          resp::ParseCommand(view, &consumed, &args, &error);
      if (parsed == resp::ParseResult::kNeedMore) {
        break;
      }
      if (parsed == resp::ParseResult::kError) {
        resp::AppendError(conn.out, error.empty() ? "ERR protocol error" : error);
        conn.close_after_write = true;
        UpdateInterest(conn);
        return true;
      }

      conn.in_pos += consumed;
      if (conn.in_pos > 4096 && conn.in_pos * 2 > conn.in.size()) {
        conn.in.erase(0, conn.in_pos);
        conn.in_pos = 0;
      }
      conn.queue.push_back(std::move(args));
    }

    MaybeDispatch(conn);
    return true;
  }

  bool HandleWrite(Connection &conn) {
    while (conn.out_pos < conn.out.size()) {
      const ssize_t n = ::write(conn.token.fd, conn.out.data() + conn.out_pos,
                                conn.out.size() - conn.out_pos);
      if (n > 0) {
        conn.out_pos += static_cast<std::size_t>(n);
        continue;
      }
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        break;
      }
      if (n < 0 && errno == EINTR) {
        continue;
      }
      return false;
    }

    if (conn.out_pos == conn.out.size()) {
      conn.out.clear();
      conn.out_pos = 0;
      UpdateInterest(conn);
      if (conn.close_after_write) {
        return false;
      }
    } else {
      UpdateInterest(conn);
    }
    return true;
  }

  void MaybeDispatch(Connection &conn) {
    if (conn.in_flight || conn.queue.empty() || workers_ == nullptr) {
      return;
    }
    conn.in_flight = true;

    constexpr std::size_t kMaxBatch = 8;
    const std::size_t batch_size =
        std::min<std::size_t>(kMaxBatch, conn.queue.size());
    std::vector<std::vector<std::string>> batch;
    batch.reserve(batch_size);
    for (std::size_t i = 0; i < batch_size; ++i) {
      batch.push_back(std::move(conn.queue.front()));
      conn.queue.pop_front();
    }

    const ConnToken token = conn.token;
    vecbase::DB *db = db_;
    const ServerConfig *cfg = cfg_;
    Impl *self = this;
    workers_->Submit([self, token, db, cfg, batch = std::move(batch)]() mutable {
      ResponseMsg msg;
      msg.token = token;
      for (std::vector<std::string> &command : batch) {
        const CommandResult result = ExecuteCommand(command, db, cfg);
        msg.payload.append(result.payload);
        if (result.close_after) {
          msg.close_after = true;
          break;
        }
      }
      self->EnqueueResponse(std::move(msg));
    });
  }

  std::size_t index_ = 0;
  vecbase::DB *db_ = nullptr;
  const ServerConfig *cfg_ = nullptr;
  ThreadPool *workers_ = nullptr;

  std::atomic_bool stopping_{false};
  std::thread thread_;
  UniqueFd event_fd_;
  UniqueFd epoll_fd_;

  std::mutex new_conn_mutex_;
  std::deque<int> pending_new_;
  std::mutex response_mutex_;
  std::deque<ResponseMsg> pending_resp_;

  std::unordered_map<int, Connection> connections_;
  std::uint64_t next_conn_id_ = 0;
};

SubReactor::SubReactor(std::size_t index, vecbase::DB *db,
                       const ServerConfig *cfg, ThreadPool *workers)
    : impl_(std::make_unique<Impl>(index, db, cfg, workers)) {}

SubReactor::~SubReactor() { Stop(); }

void SubReactor::Start() { impl_->Start(); }

void SubReactor::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

void SubReactor::EnqueueNewConn(int fd) { impl_->EnqueueNewConn(fd); }

} // namespace vecbase_server
