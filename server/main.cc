#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

#include "server/server.h"

namespace {

std::atomic_bool g_stop{false};

void OnSignal(int) { g_stop.store(true, std::memory_order_relaxed); }

void PrintUsage(const char *prog) {
  std::cerr << "Usage:\n";
  std::cerr << "  " << prog
            << " [--db-path <path>] [--bind <ip>] [--port <port>] "
               "[--subreactors <n>] [--workers <n>] [--log-new-conn]\n";
}

bool ParseInt(std::string_view s, int *out) {
  try {
    const int value = std::stoi(std::string(s));
    *out = value;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseSize(std::string_view s, std::size_t *out) {
  try {
    *out = static_cast<std::size_t>(std::stoull(std::string(s)));
    return true;
  } catch (...) {
    return false;
  }
}

std::optional<std::string_view> RequireValue(int argc, char **argv, int *i) {
  if (*i + 1 >= argc) {
    return std::nullopt;
  }
  ++(*i);
  return std::string_view(argv[*i]);
}

} // namespace

int main(int argc, char **argv) {
  vecbase_server::ServerConfig config;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
    if (arg == "--db-path") {
      const auto value = RequireValue(argc, argv, &i);
      if (!value) {
        PrintUsage(argv[0]);
        return 1;
      }
      config.db_path = std::string(*value);
      continue;
    }
    if (arg == "--bind") {
      const auto value = RequireValue(argc, argv, &i);
      if (!value) {
        PrintUsage(argv[0]);
        return 1;
      }
      config.bind = std::string(*value);
      continue;
    }
    if (arg == "--port") {
      const auto value = RequireValue(argc, argv, &i);
      if (!value || !ParseInt(*value, &config.port) || config.port <= 0 ||
          config.port > 65535) {
        PrintUsage(argv[0]);
        return 1;
      }
      continue;
    }
    if (arg == "--subreactors") {
      const auto value = RequireValue(argc, argv, &i);
      if (!value || !ParseSize(*value, &config.subreactors)) {
        PrintUsage(argv[0]);
        return 1;
      }
      continue;
    }
    if (arg == "--workers") {
      const auto value = RequireValue(argc, argv, &i);
      if (!value || !ParseSize(*value, &config.workers)) {
        PrintUsage(argv[0]);
        return 1;
      }
      continue;
    }
    if (arg == "--log-new-conn") {
      config.log_new_conn = true;
      continue;
    }

    std::cerr << "unknown argument: " << arg << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);

  try {
    vecbase_server::Server server(config);
    server.Start();
    while (!g_stop.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    server.Stop();
  } catch (const std::exception &ex) {
    std::cerr << "fatal: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
