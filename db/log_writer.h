#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "vecbase/options.h"
#include "vecbase/status.h"

namespace vecbase {

enum class LogRecordType : std::uint8_t {
  kCreateIndex = 1,
  kDropIndex = 2,
  kPut = 3,
  kDelete = 4,
};

class LogWriter {
public:
  explicit LogWriter(std::string path);

  Status Open();
  Status AppendCreateIndex(const std::string &index_name,
                           const IndexOptions &options);
  Status AppendDropIndex(const std::string &index_name);
  Status AppendPut(const std::string &index_name, const Record &record);
  Status AppendDelete(const std::string &index_name, VectorId id);
  Status Reset();

private:
  template <typename T> void WritePrimitive(const T &value) {
    stream_.write(reinterpret_cast<const char *>(&value), sizeof(T));
  }

  void WriteString(const std::string &value);
  void WriteVector(const std::vector<float> &value);
  Status Flush();

  std::string path_;
  std::ofstream stream_;
};

} // namespace vecbase
