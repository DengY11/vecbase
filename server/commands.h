#pragma once

#include <string>
#include <vector>

namespace vecbase {
class DB;
}

namespace vecbase_server {

struct ServerConfig;

struct CommandResult {
  std::string payload;
  bool close_after = false;
};

CommandResult ExecuteCommand(const std::vector<std::string> &args,
                             vecbase::DB *db, const ServerConfig *cfg);

} // namespace vecbase_server
