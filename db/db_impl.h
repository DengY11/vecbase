#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "dkv/db.h"
#include "vecbase/db.h"

namespace vecbase {

class VectorIndex;

class DBImpl final : public DB {
public:
  DBImpl(Options options, std::string name);
  ~DBImpl() override;

  Status Open();

  Status CreateIndex(const WriteOptions &options, const std::string &index_name,
                     const IndexOptions &index_options) override;

  Status DropIndex(const WriteOptions &options,
                   const std::string &index_name) override;

  bool HasIndex(const std::string &index_name) const override;
  std::vector<std::string> ListIndexes() const override;

  Status Put(const WriteOptions &options, const std::string &index_name,
             const Record &record) override;

  Status Delete(const WriteOptions &options, const std::string &index_name,
                VectorId id) override;

  Status Get(const ReadOptions &options, const std::string &index_name,
             VectorId id, Record *record) const override;

  Status Search(const ReadOptions &options, const SearchOptions &search_options,
                const std::vector<float> &query,
                std::vector<SearchResult> *results) const override;

  Status GetIndexStats(const std::string &index_name,
                       IndexStats *stats) const override;

private:
  struct IndexState {
    IndexOptions options;
    std::unique_ptr<VectorIndex> index;
    std::unordered_map<VectorId, std::string> payloads;
  };

  Status LoadFromStorage();
  Status PersistIndexMeta(const WriteOptions &options,
                          const std::string &index_name,
                          const IndexOptions &index_options);
  Status PersistRecord(const WriteOptions &options,
                       const std::string &index_name, const Record &record);
  Status DeleteRecord(const WriteOptions &options, const std::string &index_name,
                      VectorId id);

  std::string IndexMetaKey(const std::string &index_name) const;
  std::string RecordKeyPrefix(const std::string &index_name) const;
  std::string RecordKey(const std::string &index_name, VectorId id) const;
  std::string IndexStoragePath(const std::string &index_name) const;

  IndexState *FindIndex(const std::string &index_name);
  const IndexState *FindIndex(const std::string &index_name) const;

  Options options_;
  std::string name_;
  std::string kv_path_;
  std::unique_ptr<dkv::DB> kv_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, IndexState> indexes_;
};

} // namespace vecbase
