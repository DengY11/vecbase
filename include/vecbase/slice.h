#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace vecbase {

class Slice {
public:
  Slice() = default;
  Slice(const char *data, std::size_t size) : data_(data), size_(size) {}
  Slice(const std::string &value) : data_(value.data()), size_(value.size()) {}
  Slice(std::string_view value) : data_(value.data()), size_(value.size()) {}

  const char *data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  std::string ToString() const { return std::string(data_, size_); }

private:
  const char *data_ = "";
  std::size_t size_ = 0;
};

} // namespace vecbase
