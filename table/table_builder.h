#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "table/buffer_pool.h"
#include "vecbase/options.h"
#include "vecbase/status.h"

namespace vecbase {

class TableBuilder {
public:
  explicit TableBuilder(std::string path);

  Status Build(const std::vector<VectorRecord> &vectors,
               const std::unordered_map<VectorId, std::string> &payloads);

private:
  std::string path_;
};

} // namespace vecbase
