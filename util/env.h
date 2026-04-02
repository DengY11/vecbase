#pragma once

#include <string>

namespace vecbase {

class Env {
public:
  virtual ~Env() = default;
  virtual std::string GetWorkingDirectory() const = 0;
};

} // namespace vecbase
