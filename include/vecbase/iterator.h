#pragma once

#include "vecbase/status.h"

namespace vecbase {

class Iterator {
public:
  virtual ~Iterator() = default;

  virtual bool Valid() const = 0;
  virtual void Next() = 0;
  virtual Status status() const = 0;
};

} // namespace vecbase
