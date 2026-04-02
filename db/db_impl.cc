#include "db/db_impl.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string_view>
#include <utility>

#include "index/hnsw_index.h"
#include "index/vector_index.h"

namespace vecbase {

namespace {

std::unique_ptr<VectorIndex> MakeIndex(const IndexOptions &options) {
  return std::make_unique<HnswIndex>(options);
}

Status FromDkvStatus(const dkv::Status &status) {
  using Code = dkv::Status::Code;
  switch (status.code()) {
  case Code::kOk:
    return Status::OK();
  case Code::kNotFound:
    return Status::NotFound(status.message());
  case Code::kIOError:
    return Status::IOError(status.message());
  case Code::kInvalidArgument:
    return Status::InvalidArgument(status.message());
  case Code::kCorruption:
    return Status::IOError(status.message());
  }
  return Status::IOError(status.message());
}

template <typename T>
void AppendPrimitive(std::string *buffer, const T &value) {
  buffer->append(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T>
bool ReadPrimitive(std::string_view data, std::size_t *offset, T *value) {
  if (*offset + sizeof(T) > data.size()) {
    return false;
  }
  std::memcpy(value, data.data() + *offset, sizeof(T));
  *offset += sizeof(T);
  return true;
}

void AppendString(std::string *buffer, const std::string &value) {
  const std::uint64_t size = value.size();
  AppendPrimitive(buffer, size);
  buffer->append(value);
}

bool ReadString(std::string_view data, std::size_t *offset, std::string *value) {
  std::uint64_t size = 0;
  if (!ReadPrimitive(data, offset, &size)) {
    return false;
  }
  if (*offset + size > data.size()) {
    return false;
  }
  value->assign(data.substr(*offset, static_cast<std::size_t>(size)));
  *offset += static_cast<std::size_t>(size);
  return true;
}

void AppendVector(std::string *buffer, const std::vector<float> &value) {
  const std::uint64_t size = value.size();
  AppendPrimitive(buffer, size);
  buffer->append(reinterpret_cast<const char *>(value.data()),
                 value.size() * sizeof(float));
}

bool ReadVector(std::string_view data, std::size_t *offset,
                std::vector<float> *value) {
  std::uint64_t size = 0;
  if (!ReadPrimitive(data, offset, &size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(float);
  if (*offset + bytes > data.size()) {
    return false;
  }
  value->resize(static_cast<std::size_t>(size));
  std::memcpy(value->data(), data.data() + *offset, bytes);
  *offset += bytes;
  return true;
}

std::string SerializeIndexOptions(const IndexOptions &options) {
  std::string buffer;
  AppendPrimitive(&buffer, options.dimension);
  AppendPrimitive(&buffer, options.metric);
  AppendPrimitive(&buffer, options.max_degree);
  AppendPrimitive(&buffer, options.ef_construction);
  AppendPrimitive(&buffer, options.allow_replace_deleted);
  return buffer;
}

bool DeserializeIndexOptions(std::string_view data, IndexOptions *options) {
  std::size_t offset = 0;
  return ReadPrimitive(data, &offset, &options->dimension) &&
         ReadPrimitive(data, &offset, &options->metric) &&
         ReadPrimitive(data, &offset, &options->max_degree) &&
         ReadPrimitive(data, &offset, &options->ef_construction) &&
         ReadPrimitive(data, &offset, &options->allow_replace_deleted);
}

std::string SerializeRecord(const Record &record) {
  std::string buffer;
  AppendPrimitive(&buffer, record.id);
  AppendVector(&buffer, record.embedding);
  AppendString(&buffer, record.payload);
  return buffer;
}

bool DeserializeRecord(std::string_view data, Record *record) {
  std::size_t offset = 0;
  return ReadPrimitive(data, &offset, &record->id) &&
         ReadVector(data, &offset, &record->embedding) &&
         ReadString(data, &offset, &record->payload);
}

std::string EncodeVectorId(VectorId id) {
  std::ostringstream oss;
  oss.width(20);
  oss.fill('0');
  oss << id;
  return oss.str();
}

dkv::WriteOptions ToDkvWriteOptions(const WriteOptions &options) {
  dkv::WriteOptions out;
  out.sync = options.sync;
  return out;
}

} // namespace

Status DB::Open(const Options &options, const std::string &name, DB **db) {
  if (db == nullptr) {
    return Status::InvalidArgument("db output pointer must not be null");
  }

  auto impl = std::make_unique<DBImpl>(options, name);
  Status status = impl->Open();
  if (!status.ok()) {
    return status;
  }

  *db = impl.release();
  return Status::OK();
}

DBImpl::DBImpl(Options options, std::string name)
    : options_(std::move(options)), name_(std::move(name)),
      kv_path_(name_ + "/dkv") {}

DBImpl::~DBImpl() = default;

Status DBImpl::Open() {
  const bool exists = std::filesystem::exists(name_);
  if (!exists && !options_.create_if_missing) {
    return Status::NotFound("database path not found: " + name_);
  }
  if (exists && options_.error_if_exists) {
    return Status::AlreadyExists("database path already exists: " + name_);
  }

  std::filesystem::create_directories(name_);

  dkv::Options dkv_options;
  dkv_options.data_dir = kv_path_;

  dkv::Status dkv_status = dkv::DB::Open(dkv_options, kv_);
  if (!dkv_status.ok()) {
    return FromDkvStatus(dkv_status);
  }

  return LoadFromStorage();
}

Status DBImpl::CreateIndex(const WriteOptions &options,
                           const std::string &index_name,
                           const IndexOptions &index_options) {
  std::lock_guard lock(mutex_);
  if (index_name.empty()) {
    return Status::InvalidArgument("index name must not be empty");
  }
  if (index_options.dimension == 0) {
    return Status::InvalidArgument("index dimension must be greater than zero");
  }
  if (indexes_.find(index_name) != indexes_.end()) {
    return Status::AlreadyExists("index already exists: " + index_name);
  }

  IndexState state;
  state.options = index_options;
  state.index = MakeIndex(index_options);
  Status status = state.index->OpenStorage(IndexStoragePath(index_name));
  if (!status.ok()) {
    return status;
  }

  status = PersistIndexMeta(options, index_name, index_options);
  if (!status.ok()) {
    return status;
  }

  indexes_.emplace(index_name, std::move(state));
  return Status::OK();
}

Status DBImpl::DropIndex(const WriteOptions &options,
                         const std::string &index_name) {
  std::lock_guard lock(mutex_);
  IndexState *state = FindIndex(index_name);
  if (state == nullptr) {
    return Status::NotFound("index not found: " + index_name);
  }

  dkv::WriteBatch batch;
  batch.Delete(IndexMetaKey(index_name));

  auto it = kv_->Scan({}, RecordKeyPrefix(index_name));
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    batch.Delete(std::string(it->key()));
  }

  const dkv::Status dkv_status = kv_->Write(ToDkvWriteOptions(options), batch);
  if (!dkv_status.ok()) {
    return FromDkvStatus(dkv_status);
  }

  indexes_.erase(index_name);
  return Status::OK();
}

bool DBImpl::HasIndex(const std::string &index_name) const {
  std::lock_guard lock(mutex_);
  return indexes_.find(index_name) != indexes_.end();
}

std::vector<std::string> DBImpl::ListIndexes() const {
  std::lock_guard lock(mutex_);
  std::vector<std::string> names;
  names.reserve(indexes_.size());
  for (const auto &[name, _] : indexes_) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

Status DBImpl::Put(const WriteOptions &options, const std::string &index_name,
                   const Record &record) {
  std::lock_guard lock(mutex_);
  IndexState *state = FindIndex(index_name);
  if (state == nullptr) {
    return Status::NotFound("index not found: " + index_name);
  }

  Status status = state->index->Upsert(VectorRecord{record.id, record.embedding});
  if (!status.ok()) {
    return status;
  }
  state->payloads[record.id] = record.payload;
  return PersistRecord(options, index_name, record);
}

Status DBImpl::Delete(const WriteOptions &options, const std::string &index_name,
                      VectorId id) {
  std::lock_guard lock(mutex_);
  IndexState *state = FindIndex(index_name);
  if (state == nullptr) {
    return Status::NotFound("index not found: " + index_name);
  }

  Status status = state->index->Erase(id);
  if (!status.ok()) {
    return status;
  }
  state->payloads.erase(id);
  return DeleteRecord(options, index_name, id);
}

Status DBImpl::Get(const ReadOptions &, const std::string &index_name,
                   VectorId id, Record *record) const {
  if (record == nullptr) {
    return Status::InvalidArgument("record output pointer must not be null");
  }

  std::lock_guard lock(mutex_);
  const IndexState *state = FindIndex(index_name);
  if (state == nullptr) {
    return Status::NotFound("index not found: " + index_name);
  }

  VectorRecord vector_record;
  Status status = state->index->Get(id, &vector_record);
  if (!status.ok()) {
    return status;
  }

  record->id = vector_record.id;
  record->embedding = std::move(vector_record.embedding);
  const auto payload_it = state->payloads.find(id);
  record->payload =
      payload_it == state->payloads.end() ? std::string() : payload_it->second;
  return Status::OK();
}

Status DBImpl::Search(const ReadOptions &, const SearchOptions &search_options,
                      const std::vector<float> &query,
                      std::vector<SearchResult> *results) const {
  if (results == nullptr) {
    return Status::InvalidArgument("results output pointer must not be null");
  }

  std::lock_guard lock(mutex_);
  const IndexState *state = FindIndex(search_options.index_name);
  if (state == nullptr) {
    return Status::NotFound("index not found: " + search_options.index_name);
  }

  Status status = state->index->Search(search_options, query, results);
  if (!status.ok()) {
    return status;
  }

  if (search_options.include_payload) {
    for (SearchResult &result : *results) {
      const auto it = state->payloads.find(result.id);
      if (it != state->payloads.end()) {
        result.payload = it->second;
      }
    }
  }
  return Status::OK();
}

Status DBImpl::GetIndexStats(const std::string &index_name,
                             IndexStats *stats) const {
  if (stats == nullptr) {
    return Status::InvalidArgument("stats output pointer must not be null");
  }

  std::lock_guard lock(mutex_);
  const IndexState *state = FindIndex(index_name);
  if (state == nullptr) {
    return Status::NotFound("index not found: " + index_name);
  }

  *stats = state->index->GetStats();
  return Status::OK();
}

Status DBImpl::LoadFromStorage() {
  indexes_.clear();

  auto meta_iter = kv_->Scan({}, "meta:index:");
  for (meta_iter->SeekToFirst(); meta_iter->Valid(); meta_iter->Next()) {
    const std::string key(meta_iter->key());
    const std::string_view value = meta_iter->value();
    const std::string index_name = key.substr(std::string("meta:index:").size());

    IndexOptions options;
    if (!DeserializeIndexOptions(value, &options)) {
      return Status::IOError("failed to deserialize index options for " +
                             index_name);
    }

    IndexState state;
    state.options = options;
    state.index = MakeIndex(options);
    Status status = state.index->OpenStorage(IndexStoragePath(index_name));
    if (!status.ok()) {
      return status;
    }

    auto record_iter = kv_->Scan({}, RecordKeyPrefix(index_name));
    for (record_iter->SeekToFirst(); record_iter->Valid(); record_iter->Next()) {
      Record record;
      if (!DeserializeRecord(record_iter->value(), &record)) {
        return Status::IOError("failed to deserialize record for " + index_name);
      }
      state.payloads[record.id] = std::move(record.payload);
    }

    indexes_[index_name] = std::move(state);
  }

  return Status::OK();
}

Status DBImpl::PersistIndexMeta(const WriteOptions &options,
                                const std::string &index_name,
                                const IndexOptions &index_options) {
  const dkv::Status dkv_status = kv_->Put(ToDkvWriteOptions(options),
                                          IndexMetaKey(index_name),
                                          SerializeIndexOptions(index_options));
  return FromDkvStatus(dkv_status);
}

Status DBImpl::PersistRecord(const WriteOptions &options,
                             const std::string &index_name,
                             const Record &record) {
  const dkv::Status dkv_status = kv_->Put(
      ToDkvWriteOptions(options), RecordKey(index_name, record.id),
      SerializeRecord(record));
  return FromDkvStatus(dkv_status);
}

Status DBImpl::DeleteRecord(const WriteOptions &options,
                            const std::string &index_name, VectorId id) {
  const dkv::Status dkv_status =
      kv_->Delete(ToDkvWriteOptions(options), RecordKey(index_name, id));
  return FromDkvStatus(dkv_status);
}

std::string DBImpl::IndexMetaKey(const std::string &index_name) const {
  return "meta:index:" + index_name;
}

std::string DBImpl::RecordKeyPrefix(const std::string &index_name) const {
  return "record:" + index_name + ":";
}

std::string DBImpl::RecordKey(const std::string &index_name, VectorId id) const {
  return RecordKeyPrefix(index_name) + EncodeVectorId(id);
}

std::string DBImpl::IndexStoragePath(const std::string &index_name) const {
  return name_ + "/indexes/" + index_name + ".hnsw";
}

DBImpl::IndexState *DBImpl::FindIndex(const std::string &index_name) {
  const auto it = indexes_.find(index_name);
  return it == indexes_.end() ? nullptr : &it->second;
}

const DBImpl::IndexState *DBImpl::FindIndex(const std::string &index_name) const {
  const auto it = indexes_.find(index_name);
  return it == indexes_.end() ? nullptr : &it->second;
}

} // namespace vecbase
