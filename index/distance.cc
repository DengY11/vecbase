#include "index/distance.h"

#include <cmath>
#include <stdexcept>

namespace vecbase {

namespace {

float Dot(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    sum += lhs[i] * rhs[i];
  }
  return sum;
}

float Norm(const std::vector<float> &value) {
  return std::sqrt(Dot(value, value));
}

} // namespace

float ComputeDistance(MetricType metric, const std::vector<float> &lhs,
                      const std::vector<float> &rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("vector dimensions must match");
  }

  switch (metric) {
  case MetricType::kL2: {
    float sum = 0.0f;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
      const float diff = lhs[i] - rhs[i];
      sum += diff * diff;
    }
    return sum;
  }
  case MetricType::kInnerProduct:
    return -Dot(lhs, rhs);
  case MetricType::kCosine: {
    const float lhs_norm = Norm(lhs);
    const float rhs_norm = Norm(rhs);
    if (lhs_norm == 0.0f || rhs_norm == 0.0f) {
      return 1.0f;
    }
    return 1.0f - (Dot(lhs, rhs) / (lhs_norm * rhs_norm));
  }
  }

  throw std::invalid_argument("unsupported metric type");
}

} // namespace vecbase
