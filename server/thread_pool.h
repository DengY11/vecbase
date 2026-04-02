#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace vecbase_server {

class ThreadPool {
public:
  explicit ThreadPool(std::size_t thread_count) { Start(thread_count); }
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  ~ThreadPool() { Stop(); }

  void Submit(std::function<void()> task) {
    {
      std::lock_guard lock(mutex_);
      if (stopping_) {
        throw std::runtime_error("thread pool is stopping");
      }
      tasks_.push_back(std::move(task));
    }
    cv_.notify_one();
  }

  void Stop() {
    {
      std::lock_guard lock(mutex_);
      if (stopping_) {
        return;
      }
      stopping_ = true;
    }
    cv_.notify_all();
    for (std::thread &thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    threads_.clear();
  }

private:
  void Start(std::size_t thread_count) {
    if (thread_count == 0) {
      thread_count = 1;
    }
    threads_.reserve(thread_count);
    for (std::size_t i = 0; i < thread_count; ++i) {
      threads_.emplace_back([this] { WorkerLoop(); });
    }
  }

  void WorkerLoop() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
        if (stopping_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      try {
        task();
      } catch (...) {
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  bool stopping_ = false;
  std::deque<std::function<void()>> tasks_;
  std::vector<std::thread> threads_;
};

} // namespace vecbase_server
