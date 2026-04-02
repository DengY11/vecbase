#include "table/table_builder.h"

namespace vecbase {

namespace {

template <typename T> void AppendPrimitive(std::vector<std::byte> *buffer, T value) {
  const auto *begin = reinterpret_cast<const std::byte *>(&value);
  buffer->insert(buffer->end(), begin, begin + sizeof(T));
}

void AppendString(std::vector<std::byte> *buffer, const std::string &value) {
  AppendPrimitive(buffer, static_cast<std::uint64_t>(value.size()));
  const auto *begin = reinterpret_cast<const std::byte *>(value.data());
  buffer->insert(buffer->end(), begin, begin + value.size());
}

void AppendVector(std::vector<std::byte> *buffer,
                  const std::vector<float> &value) {
  AppendPrimitive(buffer, static_cast<std::uint64_t>(value.size()));
  const auto *begin = reinterpret_cast<const std::byte *>(value.data());
  buffer->insert(buffer->end(), begin, begin + value.size() * sizeof(float));
}

} // namespace

TableBuilder::TableBuilder(std::string path) : path_(std::move(path)) {}

Status TableBuilder::Build(
    const std::vector<VectorRecord> &vectors,
    const std::unordered_map<VectorId, std::string> &payloads) {
  BufferPool pool(path_);
  Status status = pool.Open(true);
  if (!status.ok()) {
    return status;
  }

  std::vector<std::byte> buffer;
  AppendPrimitive(&buffer, static_cast<std::uint64_t>(vectors.size()));
  for (const VectorRecord &record : vectors) {
    AppendPrimitive(&buffer, record.id);
    AppendVector(&buffer, record.embedding);
  }

  AppendPrimitive(&buffer, static_cast<std::uint64_t>(payloads.size()));
  for (const auto &[id, payload] : payloads) {
    AppendPrimitive(&buffer, id);
    AppendString(&buffer, payload);
  }

  return pool.WriteAll(buffer);
}

} // namespace vecbase
