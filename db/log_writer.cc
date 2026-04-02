#include "db/log_writer.h"

#include <filesystem>

namespace vecbase {

LogWriter::LogWriter(std::string path) : path_(std::move(path)) {}

Status LogWriter::Open() {
  std::filesystem::create_directories(
      std::filesystem::path(path_).parent_path());
  stream_.open(path_, std::ios::binary | std::ios::app);
  if (!stream_.is_open()) {
    return Status::IOError("failed to open WAL: " + path_);
  }
  return Status::OK();
}

Status LogWriter::AppendCreateIndex(const std::string &index_name,
                                    const IndexOptions &options) {
  const auto type = LogRecordType::kCreateIndex;
  WritePrimitive(type);
  WriteString(index_name);
  WritePrimitive(options.dimension);
  WritePrimitive(options.metric);
  WritePrimitive(options.max_degree);
  WritePrimitive(options.ef_construction);
  WritePrimitive(options.allow_replace_deleted);
  return Flush();
}

Status LogWriter::AppendDropIndex(const std::string &index_name) {
  const auto type = LogRecordType::kDropIndex;
  WritePrimitive(type);
  WriteString(index_name);
  return Flush();
}

Status LogWriter::AppendPut(const std::string &index_name, const Record &record) {
  const auto type = LogRecordType::kPut;
  WritePrimitive(type);
  WriteString(index_name);
  WritePrimitive(record.id);
  WriteVector(record.embedding);
  WriteString(record.payload);
  return Flush();
}

Status LogWriter::AppendDelete(const std::string &index_name, VectorId id) {
  const auto type = LogRecordType::kDelete;
  WritePrimitive(type);
  WriteString(index_name);
  WritePrimitive(id);
  return Flush();
}

Status LogWriter::Reset() {
  stream_.close();
  stream_.open(path_, std::ios::binary | std::ios::trunc);
  if (!stream_.is_open()) {
    return Status::IOError("failed to reset WAL: " + path_);
  }
  return Status::OK();
}

void LogWriter::WriteString(const std::string &value) {
  const std::uint64_t size = value.size();
  WritePrimitive(size);
  stream_.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void LogWriter::WriteVector(const std::vector<float> &value) {
  const std::uint64_t size = value.size();
  WritePrimitive(size);
  stream_.write(reinterpret_cast<const char *>(value.data()),
                static_cast<std::streamsize>(value.size() * sizeof(float)));
}

Status LogWriter::Flush() {
  stream_.flush();
  if (!stream_) {
    return Status::IOError("failed to flush WAL: " + path_);
  }
  return Status::OK();
}

} // namespace vecbase
