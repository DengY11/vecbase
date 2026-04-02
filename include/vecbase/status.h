#pragma once

#include <string>
#include <utility>

namespace vecbase {

class Status {
public:
  enum class Code {
    kOk = 0,
    kNotFound,
    kInvalidArgument,
    kAlreadyExists,
    kIOError,
    kNotSupported,
  };

  Status() = default;
  Status(Code code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status OK() { return Status(); }
  static Status NotFound(std::string message) {
    return Status(Code::kNotFound, std::move(message));
  }
  static Status InvalidArgument(std::string message) {
    return Status(Code::kInvalidArgument, std::move(message));
  }
  static Status AlreadyExists(std::string message) {
    return Status(Code::kAlreadyExists, std::move(message));
  }
  static Status IOError(std::string message) {
    return Status(Code::kIOError, std::move(message));
  }
  static Status NotSupported(std::string message) {
    return Status(Code::kNotSupported, std::move(message));
  }

  bool ok() const { return code_ == Code::kOk; }
  Code code() const { return code_; }
  const std::string &message() const { return message_; }

  std::string ToString() const;

private:
  Code code_ = Code::kOk;
  std::string message_;
};

inline std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }
  return message_;
}

} // namespace vecbase
