#include "server/server.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

#include "server/acceptor.h"
#include "server/subreactor.h"
#include "server/thread_pool.h"

namespace vecbase_server {

Server::Server(ServerConfig cfg) : cfg_(std::move(cfg)) {}

Server::~Server() { Stop(); }

void Server::Start() {
  if (started_.exchange(true)) {
    return;
  }

  try {
    vecbase::Options options;
    vecbase::DB *raw_db = nullptr;
    vecbase::Status status = vecbase::DB::Open(options, cfg_.db_path, &raw_db);
    if (!status.ok()) {
      throw std::runtime_error("DB::Open failed: " + status.ToString());
    }
    db_.reset(raw_db);

    const std::size_t hc =
        std::max<std::size_t>(1, std::thread::hardware_concurrency());
    std::size_t sub_n = cfg_.subreactors == 0 ? hc : cfg_.subreactors;
    std::size_t worker_n = cfg_.workers == 0 ? hc : cfg_.workers;
    sub_n = std::max<std::size_t>(1, std::min(sub_n, hc));
    worker_n = std::max<std::size_t>(1, worker_n);
    cfg_.subreactors = sub_n;
    cfg_.workers = worker_n;

    workers_ = std::make_unique<ThreadPool>(worker_n);
    subreactors_.reserve(sub_n);
    std::vector<SubReactor *> raw_subreactors;
    raw_subreactors.reserve(sub_n);
    for (std::size_t i = 0; i < sub_n; ++i) {
      subreactors_.push_back(
          std::make_unique<SubReactor>(i, db_.get(), &cfg_, workers_.get()));
      raw_subreactors.push_back(subreactors_.back().get());
    }
    for (const auto &subreactor : subreactors_) {
      subreactor->Start();
    }

    acceptor_ = std::make_unique<Acceptor>(cfg_.bind, cfg_.port,
                                           std::move(raw_subreactors),
                                           cfg_.log_new_conn);
    acceptor_->Start();

    std::cerr << "vecbase-server listening on " << cfg_.bind << ":" << cfg_.port
              << " (subreactors=" << cfg_.subreactors
              << ", workers=" << cfg_.workers
              << ", db_path=" << cfg_.db_path << ")\n";
  } catch (...) {
    Stop();
    throw;
  }
}

void Server::Stop() {
  if (!started_.exchange(false)) {
    return;
  }
  if (acceptor_) {
    acceptor_->Stop();
  }
  for (const auto &subreactor : subreactors_) {
    if (subreactor) {
      subreactor->Stop();
    }
  }
  if (workers_) {
    workers_->Stop();
  }

  acceptor_.reset();
  subreactors_.clear();
  workers_.reset();
  db_.reset();
}

} // namespace vecbase_server
