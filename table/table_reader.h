#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "table/buffer_pool.h"
#include "vecbase/options.h"
#include "vecbase/status.h"

namespace vecbase {

class TableReader {
public:
  explicit TableReader(std::string path);

  Status Read(std::vector<VectorRecord> *vectors,
              std::unordered_map<VectorId, std::string> *payloads);

private:
  std::string path_;
};

} // namespace vecbase
