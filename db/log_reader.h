#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "db/log_writer.h"

namespace vecbase {

struct LogRecord {
  LogRecordType type = LogRecordType::kPut;
  std::string index_name;
  IndexOptions index_options;
  Record record;
  VectorId id = 0;
};

class LogReader {
public:
  explicit LogReader(std::string path);

  Status Open();
  Status ReadAll(std::vector<LogRecord> *records);

private:
  template <typename T> bool ReadPrimitive(T *value) {
    return static_cast<bool>(
        stream_.read(reinterpret_cast<char *>(value), sizeof(T)));
  }

  bool ReadString(std::string *value);
  bool ReadVector(std::vector<float> *value);

  std::string path_;
  std::ifstream stream_;
};

} // namespace vecbase
