#include "server/resp.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vecbase_server::resp {
namespace {

std::optional<std::size_t> FindCrlf(std::string_view s, std::size_t from) {
  const std::size_t pos = s.find("\r\n", from);
  if (pos == std::string_view::npos) {
    return std::nullopt;
  }
  return pos;
}

bool ParseInt64(std::string_view s, std::int64_t *out) {
  if (s.empty()) {
    return false;
  }
  std::int64_t value = 0;
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
  if (ec != std::errc() || ptr != s.data() + s.size()) {
    return false;
  }
  *out = value;
  return true;
}

std::vector<std::string> SplitInline(std::string_view line) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
      ++i;
    }
    if (i >= line.size()) {
      break;
    }
    std::size_t j = i;
    while (j < line.size() && line[j] != ' ' && line[j] != '\t') {
      ++j;
    }
    out.emplace_back(line.substr(i, j - i));
    i = j;
  }
  return out;
}

ParseResult ParseBulk(std::string_view input, std::size_t *offset,
                      std::string *out, std::string *out_error) {
  if (*offset >= input.size() || input[*offset] != '$') {
    *out_error = "ERR expected bulk string";
    return ParseResult::kError;
  }
  const auto len_end = FindCrlf(input, *offset);
  if (!len_end) {
    return ParseResult::kNeedMore;
  }
  const std::string_view len_sv =
      input.substr(*offset + 1, *len_end - (*offset + 1));
  std::int64_t len = 0;
  if (!ParseInt64(len_sv, &len) || len < -1 ||
      len > std::numeric_limits<int>::max()) {
    *out_error = "ERR invalid bulk length";
    return ParseResult::kError;
  }
  const std::size_t pos = *len_end + 2;
  if (len == -1) {
    out->clear();
    *offset = pos;
    return ParseResult::kOk;
  }
  const std::size_t ulen = static_cast<std::size_t>(len);
  if (input.size() < pos + ulen + 2) {
    return ParseResult::kNeedMore;
  }
  if (input[pos + ulen] != '\r' || input[pos + ulen + 1] != '\n') {
    *out_error = "ERR bulk string missing CRLF";
    return ParseResult::kError;
  }
  out->assign(input.substr(pos, ulen));
  *offset = pos + ulen + 2;
  return ParseResult::kOk;
}

ParseResult ParseSimple(std::string_view input, std::size_t *offset,
                        std::string *out, std::string *out_error) {
  if (*offset >= input.size() || input[*offset] != '+') {
    *out_error = "ERR expected simple string";
    return ParseResult::kError;
  }
  const auto end = FindCrlf(input, *offset);
  if (!end) {
    return ParseResult::kNeedMore;
  }
  out->assign(input.substr(*offset + 1, *end - (*offset + 1)));
  *offset = *end + 2;
  return ParseResult::kOk;
}

ParseResult ParseIntegerAsString(std::string_view input, std::size_t *offset,
                                 std::string *out, std::string *out_error) {
  if (*offset >= input.size() || input[*offset] != ':') {
    *out_error = "ERR expected integer";
    return ParseResult::kError;
  }
  const auto end = FindCrlf(input, *offset);
  if (!end) {
    return ParseResult::kNeedMore;
  }
  std::int64_t value = 0;
  const std::string_view sv =
      input.substr(*offset + 1, *end - (*offset + 1));
  if (!ParseInt64(sv, &value)) {
    *out_error = "ERR invalid integer";
    return ParseResult::kError;
  }
  *offset = *end + 2;
  *out = std::to_string(value);
  return ParseResult::kOk;
}

ParseResult ParseArray(std::string_view input, std::size_t *consumed,
                       std::vector<std::string> *out_args,
                       std::string *out_error) {
  const auto header_end = FindCrlf(input, 0);
  if (!header_end) {
    return ParseResult::kNeedMore;
  }
  std::int64_t n = 0;
  const std::string_view n_sv = input.substr(1, *header_end - 1);
  if (!ParseInt64(n_sv, &n) || n < 0 || n > 1024 * 1024) {
    *out_error = "ERR invalid multibulk length";
    return ParseResult::kError;
  }

  std::size_t offset = *header_end + 2;
  out_args->clear();
  out_args->reserve(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    if (offset >= input.size()) {
      return ParseResult::kNeedMore;
    }
    std::string item;
    ParseResult result = ParseResult::kError;
    if (input[offset] == '$') {
      result = ParseBulk(input, &offset, &item, out_error);
    } else if (input[offset] == '+') {
      result = ParseSimple(input, &offset, &item, out_error);
    } else if (input[offset] == ':') {
      result = ParseIntegerAsString(input, &offset, &item, out_error);
    } else {
      *out_error = "ERR unsupported RESP type in array";
      return ParseResult::kError;
    }
    if (result != ParseResult::kOk) {
      return result;
    }
    out_args->push_back(std::move(item));
  }
  *consumed = offset;
  return ParseResult::kOk;
}

} // namespace

ParseResult ParseCommand(std::string_view input, std::size_t *consumed,
                         std::vector<std::string> *out_args,
                         std::string *out_error) {
  *consumed = 0;
  out_args->clear();
  out_error->clear();
  if (input.empty()) {
    return ParseResult::kNeedMore;
  }
  if (input[0] == '*') {
    return ParseArray(input, consumed, out_args, out_error);
  }
  const auto end = FindCrlf(input, 0);
  if (!end) {
    return ParseResult::kNeedMore;
  }
  *out_args = SplitInline(input.substr(0, *end));
  if (out_args->empty()) {
    *out_error = "ERR empty command";
    return ParseResult::kError;
  }
  *consumed = *end + 2;
  return ParseResult::kOk;
}

} // namespace vecbase_server::resp
