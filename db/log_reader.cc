#include "db/log_reader.h"

#include <filesystem>

namespace vecbase {

LogReader::LogReader(std::string path) : path_(std::move(path)) {}

Status LogReader::Open() {
  if (!std::filesystem::exists(path_)) {
    return Status::OK();
  }
  stream_.open(path_, std::ios::binary);
  if (!stream_.is_open()) {
    return Status::IOError("failed to open WAL for reading: " + path_);
  }
  return Status::OK();
}

Status LogReader::ReadAll(std::vector<LogRecord> *records) {
  if (records == nullptr) {
    return Status::InvalidArgument("records output pointer must not be null");
  }
  if (!stream_.is_open()) {
    return Status::OK();
  }

  while (stream_.peek() != EOF) {
    LogRecord record;
    if (!ReadPrimitive(&record.type)) {
      break;
    }
    if (!ReadString(&record.index_name)) {
      return Status::IOError("failed to read WAL index name");
    }

    switch (record.type) {
    case LogRecordType::kCreateIndex:
      if (!ReadPrimitive(&record.index_options.dimension) ||
          !ReadPrimitive(&record.index_options.metric) ||
          !ReadPrimitive(&record.index_options.max_degree) ||
          !ReadPrimitive(&record.index_options.ef_construction) ||
          !ReadPrimitive(&record.index_options.allow_replace_deleted)) {
        return Status::IOError("failed to read create-index WAL record");
      }
      break;
    case LogRecordType::kDropIndex:
      break;
    case LogRecordType::kPut:
      if (!ReadPrimitive(&record.record.id) ||
          !ReadVector(&record.record.embedding) ||
          !ReadString(&record.record.payload)) {
        return Status::IOError("failed to read put WAL record");
      }
      break;
    case LogRecordType::kDelete:
      if (!ReadPrimitive(&record.id)) {
        return Status::IOError("failed to read delete WAL record");
      }
      break;
    }

    records->push_back(std::move(record));
  }

  return Status::OK();
}

bool LogReader::ReadString(std::string *value) {
  std::uint64_t size = 0;
  if (!ReadPrimitive(&size)) {
    return false;
  }
  value->resize(size);
  return static_cast<bool>(
      stream_.read(value->data(), static_cast<std::streamsize>(size)));
}

bool LogReader::ReadVector(std::vector<float> *value) {
  std::uint64_t size = 0;
  if (!ReadPrimitive(&size)) {
    return false;
  }
  value->resize(size);
  return static_cast<bool>(stream_.read(
      reinterpret_cast<char *>(value->data()),
      static_cast<std::streamsize>(size * sizeof(float))));
}

} // namespace vecbase
