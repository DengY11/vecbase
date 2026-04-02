#include "server/commands.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "server/resp.h"
#include "server/server.h"
#include "vecbase/db.h"
#include "vecbase/options.h"

namespace vecbase_server {
namespace {

std::string UpperAscii(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    out.push_back(static_cast<char>(std::toupper(c)));
  }
  return out;
}

bool ParseUint64(std::string_view s, std::uint64_t *out) {
  if (s.empty()) {
    return false;
  }
  std::uint64_t value = 0;
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
  if (ec != std::errc() || ptr != s.data() + s.size()) {
    return false;
  }
  *out = value;
  return true;
}

bool ParseSize(std::string_view s, std::size_t *out) {
  std::uint64_t value = 0;
  if (!ParseUint64(s, &value) ||
      value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  *out = static_cast<std::size_t>(value);
  return true;
}

bool ParseFloat(std::string_view s, float *out) {
  if (s.empty()) {
    return false;
  }
  std::string owned(s);
  char *end = nullptr;
  errno = 0;
  const float value = std::strtof(owned.c_str(), &end);
  if (errno != 0 || end != owned.c_str() + owned.size()) {
    return false;
  }
  *out = value;
  return true;
}

bool ParseVectorCsv(std::string_view s, std::vector<float> *out) {
  out->clear();
  if (s.empty()) {
    return true;
  }
  std::size_t start = 0;
  while (start <= s.size()) {
    const std::size_t end = s.find(',', start);
    std::string_view token =
        end == std::string_view::npos ? s.substr(start) : s.substr(start, end - start);
    while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) {
      token.remove_prefix(1);
    }
    while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) {
      token.remove_suffix(1);
    }
    float value = 0.0f;
    if (!ParseFloat(token, &value)) {
      return false;
    }
    out->push_back(value);
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return true;
}

std::string FloatToString(float value) {
  std::ostringstream out;
  out << std::setprecision(6) << value;
  return out.str();
}

vecbase::MetricType ParseMetric(std::string_view s, bool *ok) {
  const std::string metric = UpperAscii(s);
  *ok = true;
  if (metric == "L2") {
    return vecbase::MetricType::kL2;
  }
  if (metric == "IP" || metric == "INNER_PRODUCT" || metric == "INNERPRODUCT") {
    return vecbase::MetricType::kInnerProduct;
  }
  if (metric == "COSINE") {
    return vecbase::MetricType::kCosine;
  }
  *ok = false;
  return vecbase::MetricType::kL2;
}

void AppendStatus(std::string *out, const vecbase::Status &status) {
  if (status.ok()) {
    resp::AppendSimpleString(*out, "OK");
  } else {
    resp::AppendError(*out, "ERR " + status.ToString());
  }
}

void AppendVector(std::string *out, const std::vector<float> &embedding) {
  resp::AppendArrayHeader(*out, embedding.size());
  for (float value : embedding) {
    resp::AppendBulkString(*out, FloatToString(value));
  }
}

CommandResult CmdPing(const std::vector<std::string> &args) {
  CommandResult result;
  if (args.size() == 1) {
    resp::AppendSimpleString(result.payload, "PONG");
  } else {
    resp::AppendBulkString(result.payload, args[1]);
  }
  return result;
}

CommandResult CmdEcho(const std::vector<std::string> &args) {
  CommandResult result;
  if (args.size() != 2) {
    resp::AppendError(result.payload, "ERR wrong number of arguments for 'echo' command");
    return result;
  }
  resp::AppendBulkString(result.payload, args[1]);
  return result;
}

CommandResult CmdQuit() {
  CommandResult result;
  resp::AppendSimpleString(result.payload, "OK");
  result.close_after = true;
  return result;
}

CommandResult CmdHello() {
  CommandResult result;
  resp::AppendArrayHeader(result.payload, 14);
  resp::AppendBulkString(result.payload, "server");
  resp::AppendBulkString(result.payload, "vecbase");
  resp::AppendBulkString(result.payload, "version");
  resp::AppendBulkString(result.payload, "0.1");
  resp::AppendBulkString(result.payload, "proto");
  resp::AppendBulkString(result.payload, "2");
  resp::AppendBulkString(result.payload, "id");
  resp::AppendBulkString(result.payload, "0");
  resp::AppendBulkString(result.payload, "mode");
  resp::AppendBulkString(result.payload, "standalone");
  resp::AppendBulkString(result.payload, "role");
  resp::AppendBulkString(result.payload, "master");
  resp::AppendBulkString(result.payload, "modules");
  resp::AppendArrayHeader(result.payload, 0);
  return result;
}

CommandResult CmdInfo(const ServerConfig *cfg) {
  CommandResult result;
  std::string info;
  info += "# Server\n";
  info += "redis_version:0.1\n";
  info += "vecbase_server:vecbase-server\n";
  info += "bind:" + cfg->bind + "\n";
  info += "port:" + std::to_string(cfg->port) + "\n";
  info += "db_path:" + cfg->db_path + "\n";
  resp::AppendBulkString(result.payload, info);
  return result;
}

CommandResult CmdCommand() {
  CommandResult result;
  resp::AppendArrayHeader(result.payload, 0);
  return result;
}

CommandResult CmdClient() {
  CommandResult result;
  resp::AppendSimpleString(result.payload, "OK");
  return result;
}

CommandResult CmdVCreate(const std::vector<std::string> &args, vecbase::DB *db) {
  CommandResult result;
  if (args.size() < 4 || args.size() > 6) {
    resp::AppendError(result.payload,
                      "ERR usage: VCREATE <index> <dimension> <metric> [max_degree] [ef_construction]");
    return result;
  }
  std::size_t dimension = 0;
  if (!ParseSize(args[2], &dimension) || dimension == 0) {
    resp::AppendError(result.payload, "ERR invalid dimension");
    return result;
  }
  bool metric_ok = false;
  const vecbase::MetricType metric = ParseMetric(args[3], &metric_ok);
  if (!metric_ok) {
    resp::AppendError(result.payload, "ERR invalid metric");
    return result;
  }

  vecbase::IndexOptions options;
  options.dimension = dimension;
  options.metric = metric;
  if (args.size() >= 5 && !ParseSize(args[4], &options.max_degree)) {
    resp::AppendError(result.payload, "ERR invalid max_degree");
    return result;
  }
  if (args.size() >= 6 && !ParseSize(args[5], &options.ef_construction)) {
    resp::AppendError(result.payload, "ERR invalid ef_construction");
    return result;
  }

  AppendStatus(&result.payload, db->CreateIndex({}, args[1], options));
  return result;
}

CommandResult CmdVDrop(const std::vector<std::string> &args, vecbase::DB *db) {
  CommandResult result;
  if (args.size() != 2) {
    resp::AppendError(result.payload, "ERR usage: VDROP <index>");
    return result;
  }
  AppendStatus(&result.payload, db->DropIndex({}, args[1]));
  return result;
}

CommandResult CmdVList(vecbase::DB *db) {
  CommandResult result;
  const std::vector<std::string> indexes = db->ListIndexes();
  resp::AppendArrayHeader(result.payload, indexes.size());
  for (const std::string &index : indexes) {
    resp::AppendBulkString(result.payload, index);
  }
  return result;
}

CommandResult CmdVHasIndex(const std::vector<std::string> &args, vecbase::DB *db) {
  CommandResult result;
  if (args.size() != 2) {
    resp::AppendError(result.payload, "ERR usage: VHASINDEX <index>");
    return result;
  }
  resp::AppendInteger(result.payload, db->HasIndex(args[1]) ? 1 : 0);
  return result;
}

CommandResult CmdVPut(const std::vector<std::string> &args, vecbase::DB *db) {
  CommandResult result;
  if (args.size() < 4 || args.size() > 5) {
    resp::AppendError(result.payload, "ERR usage: VPUT <index> <id> <vector_csv> [payload]");
    return result;
  }
  std::uint64_t id = 0;
  if (!ParseUint64(args[2], &id)) {
    resp::AppendError(result.payload, "ERR invalid id");
    return result;
  }
  std::vector<float> embedding;
  if (!ParseVectorCsv(args[3], &embedding)) {
    resp::AppendError(result.payload, "ERR invalid vector");
    return result;
  }
  vecbase::Record record;
  record.id = id;
  record.embedding = std::move(embedding);
  if (args.size() == 5) {
    record.payload = args[4];
  }
  AppendStatus(&result.payload, db->Put({}, args[1], record));
  return result;
}

CommandResult CmdVGet(const std::vector<std::string> &args, vecbase::DB *db) {
  CommandResult result;
  if (args.size() != 3) {
    resp::AppendError(result.payload, "ERR usage: VGET <index> <id>");
    return result;
  }
  std::uint64_t id = 0;
  if (!ParseUint64(args[2], &id)) {
    resp::AppendError(result.payload, "ERR invalid id");
    return result;
  }
  vecbase::Record record;
  const vecbase::Status status = db->Get({}, args[1], id, &record);
  if (!status.ok()) {
    resp::AppendError(result.payload, "ERR " + status.ToString());
    return result;
  }
  resp::AppendArrayHeader(result.payload, 3);
  resp::AppendInteger(result.payload, static_cast<std::int64_t>(record.id));
  AppendVector(&result.payload, record.embedding);
  resp::AppendBulkString(result.payload, record.payload);
  return result;
}

CommandResult CmdVDel(const std::vector<std::string> &args, vecbase::DB *db) {
  CommandResult result;
  if (args.size() != 3) {
    resp::AppendError(result.payload, "ERR usage: VDEL <index> <id>");
    return result;
  }
  std::uint64_t id = 0;
  if (!ParseUint64(args[2], &id)) {
    resp::AppendError(result.payload, "ERR invalid id");
    return result;
  }
  const vecbase::Status status = db->Delete({}, args[1], id);
  if (!status.ok()) {
    resp::AppendError(result.payload, "ERR " + status.ToString());
    return result;
  }
  resp::AppendInteger(result.payload, 1);
  return result;
}

CommandResult CmdVSearch(const std::vector<std::string> &args, vecbase::DB *db) {
  CommandResult result;
  if (args.size() < 4 || args.size() > 6) {
    resp::AppendError(result.payload,
                      "ERR usage: VSEARCH <index> <vector_csv> <top_k> [ef_search] [WITHPAYLOADS]");
    return result;
  }

  std::vector<float> query;
  if (!ParseVectorCsv(args[2], &query)) {
    resp::AppendError(result.payload, "ERR invalid vector");
    return result;
  }
  vecbase::SearchOptions options;
  options.index_name = args[1];
  if (!ParseSize(args[3], &options.top_k) || options.top_k == 0) {
    resp::AppendError(result.payload, "ERR invalid top_k");
    return result;
  }
  options.ef_search = std::max<std::size_t>(options.top_k, 50);
  if (args.size() >= 5) {
    const std::string fifth = UpperAscii(args[4]);
    if (fifth == "WITHPAYLOADS") {
      options.include_payload = true;
    } else if (!ParseSize(args[4], &options.ef_search)) {
      resp::AppendError(result.payload, "ERR invalid ef_search");
      return result;
    }
  }
  if (args.size() == 6) {
    if (UpperAscii(args[5]) != "WITHPAYLOADS") {
      resp::AppendError(result.payload, "ERR expected WITHPAYLOADS");
      return result;
    }
    options.include_payload = true;
  }

  std::vector<vecbase::SearchResult> results;
  const vecbase::Status status = db->Search({}, options, query, &results);
  if (!status.ok()) {
    resp::AppendError(result.payload, "ERR " + status.ToString());
    return result;
  }

  resp::AppendArrayHeader(result.payload, results.size());
  for (const vecbase::SearchResult &item : results) {
    resp::AppendArrayHeader(result.payload, options.include_payload ? 3 : 2);
    resp::AppendInteger(result.payload, static_cast<std::int64_t>(item.id));
    resp::AppendBulkString(result.payload, FloatToString(item.score));
    if (options.include_payload) {
      resp::AppendBulkString(result.payload, item.payload);
    }
  }
  return result;
}

CommandResult CmdVStats(const std::vector<std::string> &args, vecbase::DB *db) {
  CommandResult result;
  if (args.size() != 2) {
    resp::AppendError(result.payload, "ERR usage: VSTATS <index>");
    return result;
  }
  vecbase::IndexStats stats;
  const vecbase::Status status = db->GetIndexStats(args[1], &stats);
  if (!status.ok()) {
    resp::AppendError(result.payload, "ERR " + status.ToString());
    return result;
  }
  resp::AppendArrayHeader(result.payload, 10);
  resp::AppendBulkString(result.payload, "dimension");
  resp::AppendBulkString(result.payload, std::to_string(stats.dimension));
  resp::AppendBulkString(result.payload, "size");
  resp::AppendBulkString(result.payload, std::to_string(stats.size));
  resp::AppendBulkString(result.payload, "deleted_count");
  resp::AppendBulkString(result.payload, std::to_string(stats.deleted_count));
  resp::AppendBulkString(result.payload, "graph_edges");
  resp::AppendBulkString(result.payload, std::to_string(stats.graph_edges));
  resp::AppendBulkString(result.payload, "memory_bytes");
  resp::AppendBulkString(result.payload, std::to_string(stats.memory_bytes));
  return result;
}

} // namespace

CommandResult ExecuteCommand(const std::vector<std::string> &args, vecbase::DB *db,
                             const ServerConfig *cfg) {
  if (args.empty()) {
    CommandResult result;
    resp::AppendError(result.payload, "ERR empty command");
    return result;
  }

  const std::string command = UpperAscii(args[0]);
  if (command == "PING") {
    return CmdPing(args);
  }
  if (command == "ECHO") {
    return CmdEcho(args);
  }
  if (command == "QUIT") {
    return CmdQuit();
  }
  if (command == "HELLO") {
    return CmdHello();
  }
  if (command == "INFO") {
    return CmdInfo(cfg);
  }
  if (command == "COMMAND") {
    return CmdCommand();
  }
  if (command == "CLIENT") {
    return CmdClient();
  }
  if (command == "VCREATE") {
    return CmdVCreate(args, db);
  }
  if (command == "VDROP") {
    return CmdVDrop(args, db);
  }
  if (command == "VLIST") {
    return CmdVList(db);
  }
  if (command == "VHASINDEX") {
    return CmdVHasIndex(args, db);
  }
  if (command == "VPUT") {
    return CmdVPut(args, db);
  }
  if (command == "VGET") {
    return CmdVGet(args, db);
  }
  if (command == "VDEL") {
    return CmdVDel(args, db);
  }
  if (command == "VSEARCH") {
    return CmdVSearch(args, db);
  }
  if (command == "VSTATS") {
    return CmdVStats(args, db);
  }

  CommandResult result;
  resp::AppendError(result.payload, "ERR unknown command");
  return result;
}

} // namespace vecbase_server
