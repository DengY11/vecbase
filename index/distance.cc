#include "index/distance.h"

#include <cstddef>
#include <cmath>
#include <stdexcept>

#if defined(VECBASE_SIMD_AVX2_FMA)
#include <immintrin.h>
#elif defined(VECBASE_SIMD_NEON)
#include <arm_neon.h>
#endif

namespace vecbase {

namespace {

struct CosineAccumulator {
  float dot = 0.0f;
  float lhs_squared_norm = 0.0f;
  float rhs_squared_norm = 0.0f;
};

float DotScalar(const float *lhs, const float *rhs, std::size_t size) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < size; ++i) {
    sum += lhs[i] * rhs[i];
  }
  return sum;
}

float L2SquaredScalar(const float *lhs, const float *rhs, std::size_t size) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < size; ++i) {
    const float diff = lhs[i] - rhs[i];
    sum += diff * diff;
  }
  return sum;
}

CosineAccumulator CosineAccumulatorScalar(const float *lhs, const float *rhs,
                                          std::size_t size) {
  CosineAccumulator acc;
  for (std::size_t i = 0; i < size; ++i) {
    const float lhs_value = lhs[i];
    const float rhs_value = rhs[i];
    acc.dot += lhs_value * rhs_value;
    acc.lhs_squared_norm += lhs_value * lhs_value;
    acc.rhs_squared_norm += rhs_value * rhs_value;
  }
  return acc;
}

#if defined(VECBASE_SIMD_AVX2_FMA)

float HorizontalAdd(__m256 value) {
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, value);
  float sum = 0.0f;
  for (float lane : lanes) {
    sum += lane;
  }
  return sum;
}

float DotAvx2(const float *lhs, const float *rhs, std::size_t size) {
  __m256 acc = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    const __m256 lhs_values = _mm256_loadu_ps(lhs + i);
    const __m256 rhs_values = _mm256_loadu_ps(rhs + i);
    acc = _mm256_fmadd_ps(lhs_values, rhs_values, acc);
  }

  float sum = HorizontalAdd(acc);
  for (; i < size; ++i) {
    sum += lhs[i] * rhs[i];
  }
  return sum;
}

float L2SquaredAvx2(const float *lhs, const float *rhs, std::size_t size) {
  __m256 acc = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    const __m256 lhs_values = _mm256_loadu_ps(lhs + i);
    const __m256 rhs_values = _mm256_loadu_ps(rhs + i);
    const __m256 diff = _mm256_sub_ps(lhs_values, rhs_values);
    acc = _mm256_fmadd_ps(diff, diff, acc);
  }

  float sum = HorizontalAdd(acc);
  for (; i < size; ++i) {
    const float diff = lhs[i] - rhs[i];
    sum += diff * diff;
  }
  return sum;
}

CosineAccumulator CosineAccumulatorAvx2(const float *lhs, const float *rhs,
                                        std::size_t size) {
  __m256 dot_acc = _mm256_setzero_ps();
  __m256 lhs_norm_acc = _mm256_setzero_ps();
  __m256 rhs_norm_acc = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    const __m256 lhs_values = _mm256_loadu_ps(lhs + i);
    const __m256 rhs_values = _mm256_loadu_ps(rhs + i);
    dot_acc = _mm256_fmadd_ps(lhs_values, rhs_values, dot_acc);
    lhs_norm_acc = _mm256_fmadd_ps(lhs_values, lhs_values, lhs_norm_acc);
    rhs_norm_acc = _mm256_fmadd_ps(rhs_values, rhs_values, rhs_norm_acc);
  }

  CosineAccumulator acc{HorizontalAdd(dot_acc), HorizontalAdd(lhs_norm_acc),
                        HorizontalAdd(rhs_norm_acc)};
  for (; i < size; ++i) {
    const float lhs_value = lhs[i];
    const float rhs_value = rhs[i];
    acc.dot += lhs_value * rhs_value;
    acc.lhs_squared_norm += lhs_value * lhs_value;
    acc.rhs_squared_norm += rhs_value * rhs_value;
  }
  return acc;
}

#elif defined(VECBASE_SIMD_NEON)

float HorizontalAdd(float32x4_t value) {
#if defined(__aarch64__) || defined(_M_ARM64)
  return vaddvq_f32(value);
#else
  const float32x2_t low = vget_low_f32(value);
  const float32x2_t high = vget_high_f32(value);
  const float32x2_t pair_sum = vadd_f32(low, high);
  const float32x2_t reduced = vpadd_f32(pair_sum, pair_sum);
  return vget_lane_f32(reduced, 0);
#endif
}

float32x4_t MultiplyAdd(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
#if defined(__aarch64__) || defined(_M_ARM64)
  return vfmaq_f32(acc, lhs, rhs);
#else
  return vmlaq_f32(acc, lhs, rhs);
#endif
}

float DotNeon(const float *lhs, const float *rhs, std::size_t size) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    const float32x4_t lhs_values = vld1q_f32(lhs + i);
    const float32x4_t rhs_values = vld1q_f32(rhs + i);
    acc = MultiplyAdd(acc, lhs_values, rhs_values);
  }

  float sum = HorizontalAdd(acc);
  for (; i < size; ++i) {
    sum += lhs[i] * rhs[i];
  }
  return sum;
}

float L2SquaredNeon(const float *lhs, const float *rhs, std::size_t size) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    const float32x4_t lhs_values = vld1q_f32(lhs + i);
    const float32x4_t rhs_values = vld1q_f32(rhs + i);
    const float32x4_t diff = vsubq_f32(lhs_values, rhs_values);
    acc = MultiplyAdd(acc, diff, diff);
  }

  float sum = HorizontalAdd(acc);
  for (; i < size; ++i) {
    const float diff = lhs[i] - rhs[i];
    sum += diff * diff;
  }
  return sum;
}

CosineAccumulator CosineAccumulatorNeon(const float *lhs, const float *rhs,
                                        std::size_t size) {
  float32x4_t dot_acc = vdupq_n_f32(0.0f);
  float32x4_t lhs_norm_acc = vdupq_n_f32(0.0f);
  float32x4_t rhs_norm_acc = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    const float32x4_t lhs_values = vld1q_f32(lhs + i);
    const float32x4_t rhs_values = vld1q_f32(rhs + i);
    dot_acc = MultiplyAdd(dot_acc, lhs_values, rhs_values);
    lhs_norm_acc = MultiplyAdd(lhs_norm_acc, lhs_values, lhs_values);
    rhs_norm_acc = MultiplyAdd(rhs_norm_acc, rhs_values, rhs_values);
  }

  CosineAccumulator acc{HorizontalAdd(dot_acc), HorizontalAdd(lhs_norm_acc),
                        HorizontalAdd(rhs_norm_acc)};
  for (; i < size; ++i) {
    const float lhs_value = lhs[i];
    const float rhs_value = rhs[i];
    acc.dot += lhs_value * rhs_value;
    acc.lhs_squared_norm += lhs_value * lhs_value;
    acc.rhs_squared_norm += rhs_value * rhs_value;
  }
  return acc;
}

#endif

float Dot(const float *lhs, const float *rhs, std::size_t size) {
#if defined(VECBASE_SIMD_AVX2_FMA)
  return DotAvx2(lhs, rhs, size);
#elif defined(VECBASE_SIMD_NEON)
  return DotNeon(lhs, rhs, size);
#else
  return DotScalar(lhs, rhs, size);
#endif
}

float L2Squared(const float *lhs, const float *rhs, std::size_t size) {
#if defined(VECBASE_SIMD_AVX2_FMA)
  return L2SquaredAvx2(lhs, rhs, size);
#elif defined(VECBASE_SIMD_NEON)
  return L2SquaredNeon(lhs, rhs, size);
#else
  return L2SquaredScalar(lhs, rhs, size);
#endif
}

CosineAccumulator CosineAccumulate(const float *lhs, const float *rhs,
                                   std::size_t size) {
#if defined(VECBASE_SIMD_AVX2_FMA)
  return CosineAccumulatorAvx2(lhs, rhs, size);
#elif defined(VECBASE_SIMD_NEON)
  return CosineAccumulatorNeon(lhs, rhs, size);
#else
  return CosineAccumulatorScalar(lhs, rhs, size);
#endif
}

} // namespace

float ComputeDistance(MetricType metric, const std::vector<float> &lhs,
                      const std::vector<float> &rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("vector dimensions must match");
  }

  const float *lhs_data = lhs.data();
  const float *rhs_data = rhs.data();
  const std::size_t size = lhs.size();

  switch (metric) {
  case MetricType::kL2:
    return L2Squared(lhs_data, rhs_data, size);
  case MetricType::kInnerProduct:
    return -Dot(lhs_data, rhs_data, size);
  case MetricType::kCosine: {
    const CosineAccumulator acc = CosineAccumulate(lhs_data, rhs_data, size);
    if (acc.lhs_squared_norm == 0.0f || acc.rhs_squared_norm == 0.0f) {
      return 1.0f;
    }
    return 1.0f - (acc.dot / std::sqrt(acc.lhs_squared_norm *
                                       acc.rhs_squared_norm));
  }
  }

  throw std::invalid_argument("unsupported metric type");
}

} // namespace vecbase
