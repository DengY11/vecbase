#pragma once

#include <string>
#include <vector>

#include "vecbase/options.h"
#include "vecbase/status.h"

namespace vecbase {

class DB {
public:
  DB() = default;
  virtual ~DB() = default;

  DB(const DB &) = delete;
  DB &operator=(const DB &) = delete;

  static Status Open(const Options &options, const std::string &name, DB **db);

  virtual Status CreateIndex(const WriteOptions &options,
                             const std::string &index_name,
                             const IndexOptions &index_options) = 0;

  virtual Status DropIndex(const WriteOptions &options,
                           const std::string &index_name) = 0;

  virtual bool HasIndex(const std::string &index_name) const = 0;
  virtual std::vector<std::string> ListIndexes() const = 0;

  virtual Status Put(const WriteOptions &options, const std::string &index_name,
                     const Record &record) = 0;

  virtual Status Delete(const WriteOptions &options,
                        const std::string &index_name, VectorId id) = 0;

  virtual Status Get(const ReadOptions &options, const std::string &index_name,
                     VectorId id, Record *record) const = 0;

  virtual Status Search(const ReadOptions &options,
                        const SearchOptions &search_options,
                        const std::vector<float> &query,
                        std::vector<SearchResult> *results) const = 0;

  virtual Status GetIndexStats(const std::string &index_name,
                               IndexStats *stats) const = 0;
};

} // namespace vecbase
