#include "table/table_reader.h"

#include <cstring>

namespace vecbase {

namespace {

template <typename T>
bool ReadPrimitive(const std::vector<std::byte> &buffer, std::size_t *offset,
                   T *value) {
  if (*offset + sizeof(T) > buffer.size()) {
    return false;
  }
  std::memcpy(value, buffer.data() + *offset, sizeof(T));
  *offset += sizeof(T);
  return true;
}

bool ReadString(const std::vector<std::byte> &buffer, std::size_t *offset,
                std::string *value) {
  std::uint64_t size = 0;
  if (!ReadPrimitive(buffer, offset, &size)) {
    return false;
  }
  if (*offset + size > buffer.size()) {
    return false;
  }
  value->assign(reinterpret_cast<const char *>(buffer.data() + *offset), size);
  *offset += static_cast<std::size_t>(size);
  return true;
}

bool ReadVector(const std::vector<std::byte> &buffer, std::size_t *offset,
                std::vector<float> *value) {
  std::uint64_t size = 0;
  if (!ReadPrimitive(buffer, offset, &size)) {
    return false;
  }
  const std::size_t byte_size = static_cast<std::size_t>(size) * sizeof(float);
  if (*offset + byte_size > buffer.size()) {
    return false;
  }
  value->resize(static_cast<std::size_t>(size));
  std::memcpy(value->data(), buffer.data() + *offset, byte_size);
  *offset += byte_size;
  return true;
}

} // namespace

TableReader::TableReader(std::string path) : path_(std::move(path)) {}

Status TableReader::Read(std::vector<VectorRecord> *vectors,
                         std::unordered_map<VectorId, std::string> *payloads) {
  if (vectors == nullptr || payloads == nullptr) {
    return Status::InvalidArgument("table reader outputs must not be null");
  }

  BufferPool pool(path_);
  Status status = pool.Open(false);
  if (!status.ok()) {
    if (status.code() == Status::Code::kNotFound) {
      vectors->clear();
      payloads->clear();
      return Status::OK();
    }
    return status;
  }

  std::vector<std::byte> buffer;
  status = pool.ReadAll(&buffer);
  if (!status.ok()) {
    return status;
  }

  vectors->clear();
  payloads->clear();
  std::size_t offset = 0;

  std::uint64_t vector_count = 0;
  if (!ReadPrimitive(buffer, &offset, &vector_count)) {
    return buffer.empty() ? Status::OK()
                          : Status::IOError("failed to read vector count");
  }
  vectors->reserve(static_cast<std::size_t>(vector_count));
  for (std::uint64_t i = 0; i < vector_count; ++i) {
    VectorRecord record;
    if (!ReadPrimitive(buffer, &offset, &record.id) ||
        !ReadVector(buffer, &offset, &record.embedding)) {
      return Status::IOError("failed to read vector record");
    }
    vectors->push_back(std::move(record));
  }

  std::uint64_t payload_count = 0;
  if (!ReadPrimitive(buffer, &offset, &payload_count)) {
    return Status::IOError("failed to read payload count");
  }
  for (std::uint64_t i = 0; i < payload_count; ++i) {
    VectorId id = 0;
    std::string payload;
    if (!ReadPrimitive(buffer, &offset, &id) ||
        !ReadString(buffer, &offset, &payload)) {
      return Status::IOError("failed to read payload record");
    }
    (*payloads)[id] = std::move(payload);
  }

  return Status::OK();
}

} // namespace vecbase
