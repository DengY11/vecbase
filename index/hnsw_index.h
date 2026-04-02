#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "index/index_format.h"
#include "index/vector_index.h"
#include "table/buffer_pool.h"

namespace vecbase {

class HnswIndex final : public VectorIndex {
public:
  explicit HnswIndex(IndexOptions options);
  ~HnswIndex() override;

  Status OpenStorage(const std::string &path) override;
  Status Upsert(const VectorRecord &record) override;
  Status Erase(VectorId id) override;
  Status Get(VectorId id, VectorRecord *record) const override;
  Status Search(const SearchOptions &options, const std::vector<float> &query,
                std::vector<SearchResult> *results) const override;
  std::vector<VectorRecord> DumpRecords() const override;
  Status LoadRecords(const std::vector<VectorRecord> &records) override;
  IndexStats GetStats() const override;

private:
  struct NodeRef {
    std::uint64_t node_id = 0;
    VectorId vector_id = 0;
    NodeLocation location;
  };

  struct NodeView {
    std::uint64_t node_id = 0;
    VectorId vector_id = 0;
    int level = 0;
    std::vector<float> embedding;
    std::uint32_t adjacency_page_id = kInvalidPageId;
  };

  struct Candidate {
    float distance = 0.0f;
    VectorId id = 0;
  };

  struct MaxDistanceCompare {
    bool operator()(const Candidate &lhs, const Candidate &rhs) const {
      return lhs.distance < rhs.distance;
    }
  };

  struct MinDistanceCompare {
    bool operator()(const Candidate &lhs, const Candidate &rhs) const {
      return lhs.distance > rhs.distance;
    }
  };

  Status InitializeStorage();
  Status LoadDirectory();
  void ClearInMemoryState();
  Status InsertNewRecordUnlocked(const VectorRecord &record);
  Status RebuildUnlocked(const std::vector<VectorRecord> &records);

  std::size_t NodeSlotSize() const;
  std::size_t NodePageCapacity() const;
  std::size_t DirectoryPageCapacity() const;
  std::size_t AdjacencyCapacity(int levels) const;

  Status ReadMetaPage(MetaPage *meta) const;
  Status WriteMetaPage(const MetaPage &meta);
  Status UpdateDirectoryEntry(const NodeRef &ref);
  Status AllocateNodeSlot(NodeLocation *location);
  Status WriteNode(const NodeRef &ref, const VectorRecord &record, int level,
                   std::uint32_t adjacency_page_id);
  Status ReadNode(const NodeRef &ref, NodeView *node) const;
  Status AllocateAdjacencyPage(std::uint64_t node_id, int level_count,
                               std::uint32_t *page_id);
  Status WriteAdjacencyLevels(std::uint32_t page_id,
                              const std::vector<std::vector<VectorId>> &levels);
  Status ReadAdjacencyLevels(std::uint32_t page_id,
                             std::vector<std::vector<VectorId>> *levels) const;

  std::vector<VectorId> ReadNeighborsUnlocked(VectorId id, int layer) const;
  Status WriteNeighborsUnlocked(VectorId id, int layer,
                                const std::vector<VectorId> &neighbors);
  int SampleLevel();
  std::vector<float> ReadEmbeddingUnlocked(VectorId id) const;
  float DistanceToNodeUnlocked(const std::vector<float> &query,
                               VectorId id) const;
  float DistanceBetweenNodesUnlocked(VectorId lhs_id, VectorId rhs_id) const;
  VectorId GreedySearchAtLayerUnlocked(const std::vector<float> &query,
                                       VectorId entry_id, int layer) const;
  std::vector<Candidate>
  SearchLayerUnlocked(const std::vector<float> &query,
                      const std::vector<VectorId> &entry_points, std::size_t ef,
                      int layer) const;
  std::vector<VectorId>
  SelectNeighborsUnlocked(const std::vector<Candidate> &candidates,
                          std::size_t max_neighbors, VectorId exclude_id) const;
  Status AddBidirectionalLinkUnlocked(VectorId source_id, VectorId target_id,
                                      int layer);
  Status PruneNeighborsUnlocked(VectorId node_id, int layer,
                                std::size_t max_neighbors);

  IndexOptions options_;
  std::string storage_path_;
  std::unique_ptr<BufferPool> buffer_pool_;
  std::unordered_map<VectorId, NodeRef> directory_;
  VectorId entry_point_id_ = 0;
  bool has_entry_point_ = false;
  int max_level_ = -1;
  std::size_t edge_count_ = 0;
  std::uint32_t logical_version_ = 0;
  mutable std::shared_mutex mutex_;
  std::mt19937 rng_;
  std::uniform_real_distribution<float> level_distribution_;
};

} // namespace vecbase
