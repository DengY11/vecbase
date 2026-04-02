#pragma once

#include <string>
#include <unordered_map>

#include "vecbase/options.h"
#include "vecbase/status.h"

namespace vecbase {

class VersionSet {
public:
  explicit VersionSet(std::string manifest_path);

  Status Load(std::unordered_map<std::string, IndexOptions> *indexes) const;
  Status Save(const std::unordered_map<std::string, IndexOptions> &indexes) const;

private:
  std::string manifest_path_;
};

} // namespace vecbase
