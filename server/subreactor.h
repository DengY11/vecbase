#pragma once

#include <cstddef>
#include <memory>

namespace vecbase {
class DB;
}

namespace vecbase_server {

class ThreadPool;
struct ServerConfig;

class SubReactor {
public:
  SubReactor(std::size_t index, vecbase::DB *db, const ServerConfig *cfg,
             ThreadPool *workers);
  SubReactor(const SubReactor &) = delete;
  SubReactor &operator=(const SubReactor &) = delete;
  ~SubReactor();

  void Start();
  void Stop();
  void EnqueueNewConn(int fd);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace vecbase_server
