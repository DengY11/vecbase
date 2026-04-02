#pragma once

#include <vector>

#include "vecbase/options.h"

namespace vecbase {

float ComputeDistance(MetricType metric, const std::vector<float> &lhs,
                      const std::vector<float> &rhs);

} // namespace vecbase
