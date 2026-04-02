#pragma once

#include <memory>
#include <string>
#include <vector>

namespace vecbase_server {

class SubReactor;

class Acceptor {
public:
  Acceptor(std::string bind, int port, std::vector<SubReactor *> subreactors,
           bool log_new_conn);
  Acceptor(const Acceptor &) = delete;
  Acceptor &operator=(const Acceptor &) = delete;
  ~Acceptor();

  void Start();
  void Stop();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace vecbase_server
