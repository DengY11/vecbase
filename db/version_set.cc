#include "db/version_set.h"

#include <filesystem>
#include <fstream>

namespace vecbase {

VersionSet::VersionSet(std::string manifest_path)
    : manifest_path_(std::move(manifest_path)) {}

Status
VersionSet::Load(std::unordered_map<std::string, IndexOptions> *indexes) const {
  if (indexes == nullptr) {
    return Status::InvalidArgument("indexes output pointer must not be null");
  }
  indexes->clear();

  if (!std::filesystem::exists(manifest_path_)) {
    return Status::OK();
  }

  std::ifstream input(manifest_path_, std::ios::binary);
  if (!input.is_open()) {
    return Status::IOError("failed to open manifest: " + manifest_path_);
  }

  std::uint64_t count = 0;
  input.read(reinterpret_cast<char *>(&count), sizeof(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    std::uint64_t name_size = 0;
    input.read(reinterpret_cast<char *>(&name_size), sizeof(name_size));
    std::string name(name_size, '\0');
    input.read(name.data(), static_cast<std::streamsize>(name_size));

    IndexOptions options;
    input.read(reinterpret_cast<char *>(&options.dimension),
               sizeof(options.dimension));
    input.read(reinterpret_cast<char *>(&options.metric), sizeof(options.metric));
    input.read(reinterpret_cast<char *>(&options.max_degree),
               sizeof(options.max_degree));
    input.read(reinterpret_cast<char *>(&options.ef_construction),
               sizeof(options.ef_construction));
    input.read(reinterpret_cast<char *>(&options.allow_replace_deleted),
               sizeof(options.allow_replace_deleted));
    (*indexes)[name] = options;
  }

  if (!input) {
    return Status::IOError("failed to parse manifest: " + manifest_path_);
  }
  return Status::OK();
}

Status VersionSet::Save(
    const std::unordered_map<std::string, IndexOptions> &indexes) const {
  std::filesystem::create_directories(
      std::filesystem::path(manifest_path_).parent_path());

  std::ofstream output(manifest_path_, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    return Status::IOError("failed to write manifest: " + manifest_path_);
  }

  const std::uint64_t count = indexes.size();
  output.write(reinterpret_cast<const char *>(&count), sizeof(count));
  for (const auto &[name, options] : indexes) {
    const std::uint64_t name_size = name.size();
    output.write(reinterpret_cast<const char *>(&name_size), sizeof(name_size));
    output.write(name.data(), static_cast<std::streamsize>(name.size()));
    output.write(reinterpret_cast<const char *>(&options.dimension),
                 sizeof(options.dimension));
    output.write(reinterpret_cast<const char *>(&options.metric),
                 sizeof(options.metric));
    output.write(reinterpret_cast<const char *>(&options.max_degree),
                 sizeof(options.max_degree));
    output.write(reinterpret_cast<const char *>(&options.ef_construction),
                 sizeof(options.ef_construction));
    output.write(reinterpret_cast<const char *>(&options.allow_replace_deleted),
                 sizeof(options.allow_replace_deleted));
  }

  if (!output) {
    return Status::IOError("failed to flush manifest: " + manifest_path_);
  }
  return Status::OK();
}

} // namespace vecbase
