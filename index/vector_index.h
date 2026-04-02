#pragma once

#include <string>
#include <vector>

#include "vecbase/options.h"
#include "vecbase/status.h"

namespace vecbase {

/*each index should implement this interface*/
class VectorIndex {
public:
  virtual ~VectorIndex() = default;

  virtual Status Upsert(const VectorRecord &record) = 0;
  virtual Status Erase(VectorId id) = 0;
  virtual Status Get(VectorId id, VectorRecord *record) const = 0;
  virtual Status OpenStorage(const std::string &path) = 0;
  virtual Status Search(const SearchOptions &options,
                        const std::vector<float> &query,
                        std::vector<SearchResult> *results) const = 0;
  virtual std::vector<VectorRecord> DumpRecords() const = 0;
  virtual Status LoadRecords(const std::vector<VectorRecord> &records) = 0;
  virtual IndexStats GetStats() const = 0;
};

} // namespace vecbase
