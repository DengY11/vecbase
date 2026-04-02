#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "dkv/resp_client.h"

namespace {

void PrintUsage(const char *prog) {
  std::cerr << "Usage:\n";
  std::cerr << "  " << prog
            << " [--host <host>] [--port <port>] <command> [arg ...]\n";
}

bool ParsePort(std::string_view s, std::uint16_t *out) {
  try {
    const int value = std::stoi(std::string(s));
    if (value <= 0 || value > 65535) {
      return false;
    }
    *out = static_cast<std::uint16_t>(value);
    return true;
  } catch (...) {
    return false;
  }
}

void PrintValue(const dkv::RespValue &value, int indent = 0) {
  const std::string prefix(static_cast<std::size_t>(indent), ' ');
  switch (value.type) {
  case dkv::RespValue::Type::kSimpleString:
    std::cout << prefix << value.str << "\n";
    break;
  case dkv::RespValue::Type::kError:
    std::cout << prefix << "(error) " << value.str << "\n";
    break;
  case dkv::RespValue::Type::kInteger:
    std::cout << prefix << value.integer << "\n";
    break;
  case dkv::RespValue::Type::kBulkString:
    std::cout << prefix << value.str << "\n";
    break;
  case dkv::RespValue::Type::kNull:
    std::cout << prefix << "(nil)\n";
    break;
  case dkv::RespValue::Type::kArray:
    for (std::size_t i = 0; i < value.array.size(); ++i) {
      std::cout << prefix << (i + 1) << ") ";
      if (value.array[i].IsArray()) {
        std::cout << "\n";
        PrintValue(value.array[i], indent + 2);
      } else {
        PrintValue(value.array[i], 0);
      }
    }
    break;
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string host = "127.0.0.1";
  std::uint16_t port = 6380;
  int command_index = -1;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
    if (arg == "--host") {
      if (i + 1 >= argc) {
        PrintUsage(argv[0]);
        return 1;
      }
      host = argv[++i];
      continue;
    }
    if (arg == "--port") {
      if (i + 1 >= argc || !ParsePort(argv[++i], &port)) {
        PrintUsage(argv[0]);
        return 1;
      }
      continue;
    }
    command_index = i;
    break;
  }

  if (command_index < 0) {
    PrintUsage(argv[0]);
    return 1;
  }

  std::vector<std::string> owned_args;
  std::vector<std::string_view> args;
  for (int i = command_index; i < argc; ++i) {
    owned_args.emplace_back(argv[i]);
  }
  args.reserve(owned_args.size());
  for (const std::string &arg : owned_args) {
    args.push_back(arg);
  }

  try {
    dkv::RespClient client(host, port);
    const dkv::RespValue reply = client.raw().Call(args);
    PrintValue(reply);
  } catch (const std::exception &ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
