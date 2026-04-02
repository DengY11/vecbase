#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "vecbase/db.h"

namespace vecbase_server {

class Acceptor;
class SubReactor;
class ThreadPool;

struct ServerConfig {
  std::string bind = "0.0.0.0";
  int port = 6380;
  std::size_t subreactors = 0;
  std::size_t workers = 0;
  std::string db_path = "vecbase-data";
  bool log_new_conn = false;
};

class Server {
public:
  explicit Server(ServerConfig cfg);
  Server(const Server &) = delete;
  Server &operator=(const Server &) = delete;
  ~Server();

  void Start();
  void Stop();

private:
  ServerConfig cfg_;
  std::unique_ptr<vecbase::DB> db_;
  std::unique_ptr<ThreadPool> workers_;
  std::vector<std::unique_ptr<SubReactor>> subreactors_;
  std::unique_ptr<Acceptor> acceptor_;
  std::atomic_bool started_{false};
};

} // namespace vecbase_server
