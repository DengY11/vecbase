#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vecbase {

using VectorId = std::uint64_t;

enum class MetricType {
  kL2,
  kInnerProduct,
  kCosine,
};

struct Options {
  bool create_if_missing = true;
  bool error_if_exists = false;
};

struct ReadOptions {
  bool verify_checksums = false;
  bool fill_cache = true;
};

struct WriteOptions {
  bool sync = false;
};

struct IndexOptions {
  std::size_t dimension = 0;
  MetricType metric = MetricType::kL2;
  std::size_t max_degree = 32;
  std::size_t ef_construction = 200;
  bool allow_replace_deleted = true;
};

struct SearchOptions {
  std::string index_name;
  std::size_t top_k = 10;
  std::size_t ef_search = 50;
  bool include_payload = false;
};

struct VectorRecord {
  VectorId id = 0;
  std::vector<float> embedding;
};

struct Record {
  VectorId id = 0;
  std::vector<float> embedding;
  std::string payload;
};

struct SearchResult {
  VectorId id = 0;
  float score = 0.0f;
  std::string payload;
};

struct IndexStats {
  std::size_t dimension = 0;
  std::size_t size = 0;
  std::size_t deleted_count = 0;
  std::size_t graph_edges = 0;
  std::size_t memory_bytes = 0;
};

} // namespace vecbase
