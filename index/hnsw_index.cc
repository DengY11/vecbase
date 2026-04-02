#include "index/hnsw_index.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "index/distance.h"

namespace vecbase {

namespace {

constexpr float kLevelProbability = 0.5f;
constexpr std::size_t kDefaultPageSize = 4096;
constexpr std::size_t kDefaultBufferPoolPages = 128;
constexpr std::size_t kDefaultLruK = 2;

template <typename T> T *As(std::byte *data) {
  return reinterpret_cast<T *>(data);
}

template <typename T> const T *As(const std::byte *data) {
  return reinterpret_cast<const T *>(data);
}

bool SameEmbedding(const std::vector<float> &lhs,
                   const std::vector<float> &rhs) {
  return lhs == rhs;
}

} // namespace

HnswIndex::HnswIndex(IndexOptions options)
    : options_(std::move(options)), rng_(std::random_device{}()),
      level_distribution_(0.0f, 1.0f) {}

HnswIndex::~HnswIndex() {
  if (buffer_pool_ != nullptr) {
    buffer_pool_->Flush();
  }
}

Status HnswIndex::OpenStorage(const std::string &path) {
  std::unique_lock lock(mutex_);
  storage_path_ = path;
  buffer_pool_ = std::make_unique<BufferPool>(
      path, kDefaultPageSize, kDefaultBufferPoolPages, kDefaultLruK);
  Status status = buffer_pool_->Open(true);
  if (!status.ok()) {
    return status;
  }
  if (buffer_pool_->page_count() == 0) {
    return InitializeStorage();
  }
  return LoadDirectory();
}

Status HnswIndex::Upsert(const VectorRecord &record) {
  std::unique_lock lock(mutex_);
  if (buffer_pool_ == nullptr) {
    return Status::IOError("index storage is not open");
  }
  if (record.embedding.size() != options_.dimension) {
    return Status::InvalidArgument(
        "record dimension does not match index dimension");
  }

  const auto found = directory_.find(record.id);
  if (found == directory_.end()) {
    Status status = InsertNewRecordUnlocked(record);
    if (!status.ok()) {
      return status;
    }
    return buffer_pool_->Flush();
  }

  NodeView existing_node;
  Status status = ReadNode(found->second, &existing_node);
  if (!status.ok()) {
    return status;
  }
  if (SameEmbedding(existing_node.embedding, record.embedding)) {
    return Status::OK();
  }

  std::vector<VectorRecord> records = DumpRecords();
  for (VectorRecord &item : records) {
    if (item.id == record.id) {
      item = record;
      status = RebuildUnlocked(records);
      if (!status.ok()) {
        return status;
      }
      return buffer_pool_->Flush();
    }
  }
  return Status::NotFound("vector id not found");
}

Status HnswIndex::Erase(VectorId id) {
  std::unique_lock lock(mutex_);
  if (directory_.find(id) == directory_.end()) {
    return Status::NotFound("vector id not found");
  }

  std::vector<VectorRecord> records = DumpRecords();
  records.erase(std::remove_if(records.begin(), records.end(),
                               [id](const VectorRecord &record) {
                                 return record.id == id;
                               }),
                records.end());
  Status status = RebuildUnlocked(records);
  if (!status.ok()) {
    return status;
  }
  return buffer_pool_->Flush();
}

Status HnswIndex::Get(VectorId id, VectorRecord *record) const {
  if (record == nullptr) {
    return Status::InvalidArgument("record output pointer must not be null");
  }

  std::shared_lock lock(mutex_);
  const auto it = directory_.find(id);
  if (it == directory_.end()) {
    return Status::NotFound("vector id not found");
  }

  NodeView node;
  Status status = ReadNode(it->second, &node);
  if (!status.ok()) {
    return status;
  }
  record->id = node.vector_id;
  record->embedding = std::move(node.embedding);
  return Status::OK();
}

Status HnswIndex::Search(const SearchOptions &options,
                         const std::vector<float> &query,
                         std::vector<SearchResult> *results) const {
  if (results == nullptr) {
    return Status::InvalidArgument("results output pointer must not be null");
  }
  if (query.size() != options_.dimension) {
    return Status::InvalidArgument(
        "query dimension does not match index dimension");
  }

  std::shared_lock lock(mutex_);
  results->clear();
  if (!has_entry_point_) {
    return Status::OK();
  }

  VectorId current_entry = entry_point_id_;
  for (int layer = max_level_; layer > 0; --layer) {
    current_entry = GreedySearchAtLayerUnlocked(query, current_entry, layer);
  }

  const std::vector<Candidate> candidates =
      SearchLayerUnlocked(query, std::vector<VectorId>{current_entry},
                          std::max<std::size_t>(options.ef_search, options.top_k),
                          0);

  for (const Candidate &candidate : candidates) {
    results->push_back(SearchResult{candidate.id, candidate.distance, {}});
    if (results->size() == options.top_k) {
      break;
    }
  }
  return Status::OK();
}

std::vector<VectorRecord> HnswIndex::DumpRecords() const {
  std::vector<VectorRecord> records;
  records.reserve(directory_.size());
  for (const auto &[vector_id, ref] : directory_) {
    (void)vector_id;
    NodeView node;
    if (ReadNode(ref, &node).ok()) {
      records.push_back(VectorRecord{node.vector_id, std::move(node.embedding)});
    }
  }
  return records;
}

Status HnswIndex::LoadRecords(const std::vector<VectorRecord> &records) {
  std::unique_lock lock(mutex_);
  return RebuildUnlocked(records);
}

IndexStats HnswIndex::GetStats() const {
  std::shared_lock lock(mutex_);
  IndexStats stats;
  stats.dimension = options_.dimension;
  stats.size = directory_.size();
  stats.graph_edges = edge_count_;
  stats.memory_bytes =
      directory_.size() * sizeof(NodeRef) + buffer_pool_->page_size();
  return stats;
}

Status HnswIndex::InitializeStorage() {
  directory_.clear();
  edge_count_ = 0;
  entry_point_id_ = 0;
  has_entry_point_ = false;
  max_level_ = -1;
  logical_version_ = 0;

  BufferPool::PageHandle meta_handle;
  Status status = buffer_pool_->NewPage(&meta_handle);
  if (!status.ok()) {
    return status;
  }
  BufferPool::PageHandle dir_handle;
  status = buffer_pool_->NewPage(&dir_handle);
  if (!status.ok()) {
    return status;
  }
  BufferPool::PageHandle free_handle;
  status = buffer_pool_->NewPage(&free_handle);
  if (!status.ok()) {
    return status;
  }

  MetaPage meta{};
  meta.header.type = static_cast<std::uint16_t>(IndexPageType::kMeta);
  meta.header.page_id = meta_handle.page_id;
  meta.page_size = buffer_pool_->page_size();
  meta.dimension = options_.dimension;
  meta.metric = static_cast<std::uint32_t>(options_.metric);
  meta.max_degree = options_.max_degree;
  meta.ef_construction = options_.ef_construction;
  meta.directory_page_id = dir_handle.page_id;
  meta.freelist_page_id = free_handle.page_id;
  meta.next_page_id = free_handle.page_id + 1;
  std::memcpy(meta_handle.data, &meta, sizeof(meta));
  status = buffer_pool_->UnpinPage(meta_handle, true);
  if (!status.ok()) {
    return status;
  }

  DirectoryPage dir{};
  dir.header.type = static_cast<std::uint16_t>(IndexPageType::kDirectory);
  dir.header.page_id = dir_handle.page_id;
  dir.next_page_id = kInvalidPageId;
  dir.capacity = static_cast<std::uint16_t>(DirectoryPageCapacity());
  std::memcpy(dir_handle.data, &dir, sizeof(dir));
  status = buffer_pool_->UnpinPage(dir_handle, true);
  if (!status.ok()) {
    return status;
  }

  FreelistPage free_page{};
  free_page.header.type = static_cast<std::uint16_t>(IndexPageType::kFreelist);
  free_page.header.page_id = free_handle.page_id;
  std::memcpy(free_handle.data, &free_page, sizeof(free_page));
  status = buffer_pool_->UnpinPage(free_handle, true);
  if (!status.ok()) {
    return status;
  }

  logical_version_ = 1;
  status = buffer_pool_->SetCheckpointState(logical_version_, 0);
  if (!status.ok()) {
    return status;
  }
  return buffer_pool_->Flush();
}

Status HnswIndex::LoadDirectory() {
  directory_.clear();
  edge_count_ = 0;

  MetaPage meta{};
  Status status = ReadMetaPage(&meta);
  if (!status.ok()) {
    return status;
  }
  has_entry_point_ = meta.node_count > 0;
  entry_point_id_ = static_cast<VectorId>(meta.entry_node_id);
  max_level_ = meta.max_level;
  logical_version_ = buffer_pool_->checkpoint_state().logical_version;

  std::uint32_t page_id = meta.directory_page_id;
  std::unordered_set<std::uint32_t> visited_pages;
  while (page_id != kInvalidPageId) {
    if (!visited_pages.insert(page_id).second) {
      return Status::IOError("directory page cycle detected during load");
    }
    BufferPool::PageHandle handle;
    status = buffer_pool_->FetchPage(page_id, &handle);
    if (!status.ok()) {
      return status;
    }

    const DirectoryPage *page = As<DirectoryPage>(handle.data);
    const auto *entries = reinterpret_cast<const DirectoryEntry *>(
        handle.data + sizeof(DirectoryPage));
    for (std::uint16_t i = 0; i < page->entry_count; ++i) {
      const DirectoryEntry &entry = entries[i];
      directory_[entry.vector_id] =
          NodeRef{entry.node_id, entry.vector_id,
                  NodeLocation{entry.page_id, entry.slot_id, 0}};

      std::vector<std::vector<VectorId>> levels;
      NodeView node;
      status = ReadNode(directory_.at(entry.vector_id), &node);
      if (!status.ok()) {
        buffer_pool_->UnpinPage(handle, false);
        return status;
      }
      status = ReadAdjacencyLevels(node.adjacency_page_id, &levels);
      if (!status.ok()) {
        buffer_pool_->UnpinPage(handle, false);
        return status;
      }
      for (const auto &neighbors : levels) {
        edge_count_ += neighbors.size();
      }
    }

    page_id = page->next_page_id;
    status = buffer_pool_->UnpinPage(handle, false);
    if (!status.ok()) {
      return status;
    }
  }

  return Status::OK();
}

void HnswIndex::ClearInMemoryState() {
  directory_.clear();
  edge_count_ = 0;
  entry_point_id_ = 0;
  has_entry_point_ = false;
  max_level_ = -1;
}

Status HnswIndex::InsertNewRecordUnlocked(const VectorRecord &record) {
  MetaPage meta{};
  Status status = ReadMetaPage(&meta);
  if (!status.ok()) {
    return status;
  }

  const int node_level = SampleLevel();
  std::uint32_t adjacency_page_id = kInvalidPageId;
  status = AllocateAdjacencyPage(meta.next_node_id, node_level + 1,
                                 &adjacency_page_id);
  if (!status.ok()) {
    return status;
  }

  std::vector<std::vector<VectorId>> empty_levels(
      static_cast<std::size_t>(node_level + 1));
  status = WriteAdjacencyLevels(adjacency_page_id, empty_levels);
  if (!status.ok()) {
    return status;
  }

  NodeLocation location;
  status = AllocateNodeSlot(&location);
  if (!status.ok()) {
    return status;
  }

  NodeRef ref{meta.next_node_id++, record.id, location};
  status = WriteNode(ref, record, node_level, adjacency_page_id);
  if (!status.ok()) {
    return status;
  }
  status = UpdateDirectoryEntry(ref);
  if (!status.ok()) {
    return status;
  }
  directory_[record.id] = ref;

  if (!has_entry_point_) {
    entry_point_id_ = record.id;
    has_entry_point_ = true;
    max_level_ = node_level;
    meta.entry_node_id = record.id;
    meta.max_level = node_level;
    meta.node_count = 1;
    status = WriteMetaPage(meta);
    if (!status.ok()) {
      return status;
    }
    status = buffer_pool_->SetCheckpointState(++logical_version_, 0);
    if (!status.ok()) {
      return status;
    }
    return Status::OK();
  }

  VectorId current_entry = entry_point_id_;
  for (int layer = max_level_; layer > node_level; --layer) {
    current_entry =
        GreedySearchAtLayerUnlocked(record.embedding, current_entry, layer);
  }

  const int base_layer = std::min(node_level, max_level_);
  for (int layer = base_layer; layer >= 0; --layer) {
    const std::vector<Candidate> layer_candidates =
        SearchLayerUnlocked(
            record.embedding, std::vector<VectorId>{current_entry},
            std::max<std::size_t>(options_.ef_construction, options_.max_degree),
            layer);
    const std::vector<VectorId> selected = SelectNeighborsUnlocked(
        layer_candidates, options_.max_degree, record.id);

    status = WriteNeighborsUnlocked(record.id, layer, selected);
    if (!status.ok()) {
      return status;
    }
    for (VectorId neighbor_id : selected) {
      status = AddBidirectionalLinkUnlocked(record.id, neighbor_id, layer);
      if (!status.ok()) {
        return status;
      }
    }

    if (!layer_candidates.empty()) {
      current_entry = layer_candidates.front().id;
    }
  }

  if (node_level > max_level_) {
    entry_point_id_ = record.id;
    max_level_ = node_level;
    meta.entry_node_id = record.id;
    meta.max_level = node_level;
  }
  meta.node_count = directory_.size();
  status = WriteMetaPage(meta);
  if (!status.ok()) {
    return status;
  }
  return buffer_pool_->SetCheckpointState(++logical_version_, 0);
}

Status HnswIndex::RebuildUnlocked(const std::vector<VectorRecord> &records) {
  if (!storage_path_.empty()) {
    buffer_pool_.reset();
    std::filesystem::remove(storage_path_);
    std::filesystem::remove(storage_path_ + ".wal");
    std::filesystem::remove(storage_path_ + ".ckpt");
    buffer_pool_ = std::make_unique<BufferPool>(
        storage_path_, kDefaultPageSize, kDefaultBufferPoolPages, kDefaultLruK);
    Status status = buffer_pool_->Open(true);
    if (!status.ok()) {
      return status;
    }
    status = InitializeStorage();
    if (!status.ok()) {
      return status;
    }
  } else {
    return Status::IOError("index storage is not open");
  }

  for (const VectorRecord &record : records) {
    Status status = InsertNewRecordUnlocked(record);
    if (!status.ok()) {
      return status;
    }
  }
  Status status = buffer_pool_->SetCheckpointState(++logical_version_, 0);
  if (!status.ok()) {
    return status;
  }
  return buffer_pool_->Flush();
}

std::size_t HnswIndex::NodeSlotSize() const {
  return sizeof(NodeSlotHeader) + options_.dimension * sizeof(float);
}

std::size_t HnswIndex::NodePageCapacity() const {
  const std::size_t slot_size = NodeSlotSize();
  std::size_t capacity = 0;
  while (true) {
    const std::size_t next = capacity + 1;
    const std::size_t used = sizeof(NodePage) + next + next * slot_size;
    if (used > buffer_pool_->page_size()) {
      break;
    }
    capacity = next;
  }
  return capacity;
}

std::size_t HnswIndex::DirectoryPageCapacity() const {
  return (buffer_pool_->page_size() - sizeof(DirectoryPage)) /
         sizeof(DirectoryEntry);
}

std::size_t HnswIndex::AdjacencyCapacity(int levels) const {
  const std::size_t header_bytes =
      sizeof(AdjacencyPage) +
      static_cast<std::size_t>(levels) * sizeof(AdjacencyLevelHeader);
  return (buffer_pool_->page_size() - header_bytes) / sizeof(VectorId);
}

Status HnswIndex::ReadMetaPage(MetaPage *meta) const {
  BufferPool::PageHandle handle;
  Status status = buffer_pool_->FetchPage(0, &handle);
  if (!status.ok()) {
    return status;
  }
  std::memcpy(meta, handle.data, sizeof(MetaPage));
  return buffer_pool_->UnpinPage(handle, false);
}

Status HnswIndex::WriteMetaPage(const MetaPage &meta) {
  BufferPool::PageHandle handle;
  Status status = buffer_pool_->FetchPage(0, &handle);
  if (!status.ok()) {
    return status;
  }
  std::memcpy(handle.data, &meta, sizeof(MetaPage));
  return buffer_pool_->UnpinPage(handle, true);
}

Status HnswIndex::UpdateDirectoryEntry(const NodeRef &ref) {
  MetaPage meta{};
  Status status = ReadMetaPage(&meta);
  if (!status.ok()) {
    return status;
  }

  std::uint32_t page_id = meta.directory_page_id;
  std::unordered_set<std::uint32_t> visited_pages;
  while (true) {
    if (!visited_pages.insert(page_id).second) {
      return Status::IOError("directory page cycle detected during update");
    }
    BufferPool::PageHandle handle;
    status = buffer_pool_->FetchPage(page_id, &handle);
    if (!status.ok()) {
      return status;
    }
    auto *page = As<DirectoryPage>(handle.data);
    auto *entries =
        reinterpret_cast<DirectoryEntry *>(handle.data + sizeof(DirectoryPage));

    for (std::uint16_t i = 0; i < page->entry_count; ++i) {
      if (entries[i].vector_id == ref.vector_id) {
        entries[i].node_id = ref.node_id;
        entries[i].page_id = ref.location.page_id;
        entries[i].slot_id = ref.location.slot_id;
        return buffer_pool_->UnpinPage(handle, true);
      }
    }

    if (page->entry_count < page->capacity) {
      DirectoryEntry &entry = entries[page->entry_count++];
      entry.vector_id = ref.vector_id;
      entry.node_id = ref.node_id;
      entry.page_id = ref.location.page_id;
      entry.slot_id = ref.location.slot_id;
      entry.flags = 1;
      return buffer_pool_->UnpinPage(handle, true);
    }

    if (page->next_page_id == kInvalidPageId) {
      BufferPool::PageHandle new_handle;
      status = buffer_pool_->NewPage(&new_handle);
      if (!status.ok()) {
        buffer_pool_->UnpinPage(handle, false);
        return status;
      }
      DirectoryPage new_page{};
      new_page.header.type = static_cast<std::uint16_t>(IndexPageType::kDirectory);
      new_page.header.page_id = new_handle.page_id;
      new_page.next_page_id = kInvalidPageId;
      new_page.capacity = static_cast<std::uint16_t>(DirectoryPageCapacity());
      std::memcpy(new_handle.data, &new_page, sizeof(new_page));
      page->next_page_id = new_handle.page_id;
      status = buffer_pool_->UnpinPage(new_handle, true);
      if (!status.ok()) {
        buffer_pool_->UnpinPage(handle, true);
        return status;
      }
      status = buffer_pool_->UnpinPage(handle, true);
      if (!status.ok()) {
        return status;
      }
      page_id = new_page.header.page_id;
      continue;
    }

    page_id = page->next_page_id;
    status = buffer_pool_->UnpinPage(handle, false);
    if (!status.ok()) {
      return status;
    }
  }
}

Status HnswIndex::AllocateNodeSlot(NodeLocation *location) {
  const std::size_t capacity = NodePageCapacity();
  for (std::uint32_t page_id = 3; page_id < buffer_pool_->page_count(); ++page_id) {
    BufferPool::PageHandle handle;
    Status status = buffer_pool_->FetchPage(page_id, &handle);
    if (!status.ok()) {
      continue;
    }
    auto *header = As<NodePage>(handle.data);
    if (header->header.type != static_cast<std::uint16_t>(IndexPageType::kNode)) {
      buffer_pool_->UnpinPage(handle, false);
      continue;
    }

    auto *used = reinterpret_cast<std::uint8_t *>(handle.data + sizeof(NodePage));
    for (std::size_t slot = 0; slot < capacity; ++slot) {
      if (used[slot] == 0) {
        used[slot] = 1;
        header->used_count += 1;
        *location = NodeLocation{page_id, static_cast<std::uint16_t>(slot), 0};
        return buffer_pool_->UnpinPage(handle, true);
      }
    }
    status = buffer_pool_->UnpinPage(handle, false);
    if (!status.ok()) {
      return status;
    }
  }

  BufferPool::PageHandle handle;
  Status status = buffer_pool_->NewPage(&handle);
  if (!status.ok()) {
    return status;
  }
  NodePage page{};
  page.header.type = static_cast<std::uint16_t>(IndexPageType::kNode);
  page.header.page_id = handle.page_id;
  page.slot_count = static_cast<std::uint16_t>(capacity);
  page.used_count = 1;
  std::memcpy(handle.data, &page, sizeof(page));
  auto *used = reinterpret_cast<std::uint8_t *>(handle.data + sizeof(NodePage));
  std::memset(used, 0, capacity);
  used[0] = 1;
  *location = NodeLocation{handle.page_id, 0, 0};
  return buffer_pool_->UnpinPage(handle, true);
}

Status HnswIndex::WriteNode(const NodeRef &ref, const VectorRecord &record,
                            int level, std::uint32_t adjacency_page_id) {
  BufferPool::PageHandle handle;
  Status status = buffer_pool_->FetchPage(ref.location.page_id, &handle);
  if (!status.ok()) {
    return status;
  }

  const std::size_t slot_size = NodeSlotSize();
  const std::size_t capacity = NodePageCapacity();
  auto *used = reinterpret_cast<std::uint8_t *>(handle.data + sizeof(NodePage));
  auto *slot = handle.data + sizeof(NodePage) + capacity +
               static_cast<std::size_t>(ref.location.slot_id) * slot_size;

  NodeSlotHeader slot_header{};
  slot_header.vector_id = record.id;
  slot_header.node_id = ref.node_id;
  slot_header.adjacency_page_id = adjacency_page_id;
  slot_header.level = static_cast<std::uint16_t>(level);
  std::memcpy(slot, &slot_header, sizeof(slot_header));
  std::memcpy(slot + sizeof(slot_header), record.embedding.data(),
              record.embedding.size() * sizeof(float));
  used[ref.location.slot_id] = 1;
  return buffer_pool_->UnpinPage(handle, true);
}

Status HnswIndex::ReadNode(const NodeRef &ref, NodeView *node) const {
  BufferPool::PageHandle handle;
  Status status = buffer_pool_->FetchPage(ref.location.page_id, &handle);
  if (!status.ok()) {
    return status;
  }

  const std::size_t slot_size = NodeSlotSize();
  const std::size_t capacity = NodePageCapacity();
  const auto *slot = handle.data + sizeof(NodePage) + capacity +
                     static_cast<std::size_t>(ref.location.slot_id) * slot_size;
  NodeSlotHeader slot_header{};
  std::memcpy(&slot_header, slot, sizeof(slot_header));

  node->node_id = slot_header.node_id;
  node->vector_id = slot_header.vector_id;
  node->level = slot_header.level;
  node->adjacency_page_id = slot_header.adjacency_page_id;
  node->embedding.resize(options_.dimension);
  std::memcpy(node->embedding.data(), slot + sizeof(slot_header),
              options_.dimension * sizeof(float));
  return buffer_pool_->UnpinPage(handle, false);
}

Status HnswIndex::AllocateAdjacencyPage(std::uint64_t node_id, int level_count,
                                        std::uint32_t *page_id) {
  BufferPool::PageHandle handle;
  Status status = buffer_pool_->NewPage(&handle);
  if (!status.ok()) {
    return status;
  }

  AdjacencyPage page{};
  page.header.type = static_cast<std::uint16_t>(IndexPageType::kAdjacency);
  page.header.page_id = handle.page_id;
  page.node_id = node_id;
  page.level_count = static_cast<std::uint16_t>(level_count);
  std::memcpy(handle.data, &page, sizeof(page));
  *page_id = handle.page_id;
  return buffer_pool_->UnpinPage(handle, true);
}

Status HnswIndex::WriteAdjacencyLevels(
    std::uint32_t page_id, const std::vector<std::vector<VectorId>> &levels) {
  if (levels.empty()) {
    return Status::InvalidArgument("levels must not be empty");
  }
  std::size_t total_ids = 0;
  for (const auto &level : levels) {
    total_ids += level.size();
  }
  if (total_ids > AdjacencyCapacity(static_cast<int>(levels.size()))) {
    return Status::NotSupported("adjacency list does not fit in a single page");
  }

  BufferPool::PageHandle handle;
  Status status = buffer_pool_->FetchPage(page_id, &handle);
  if (!status.ok()) {
    return status;
  }

  auto *page = As<AdjacencyPage>(handle.data);
  page->level_count = static_cast<std::uint16_t>(levels.size());

  auto *level_headers =
      reinterpret_cast<AdjacencyLevelHeader *>(handle.data + sizeof(AdjacencyPage));
  auto *ids = reinterpret_cast<VectorId *>(handle.data + sizeof(AdjacencyPage) +
                                           levels.size() *
                                               sizeof(AdjacencyLevelHeader));
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < levels.size(); ++i) {
    level_headers[i].count = static_cast<std::uint16_t>(levels[i].size());
    level_headers[i].capacity = level_headers[i].count;
    for (VectorId id : levels[i]) {
      ids[cursor++] = id;
    }
  }

  return buffer_pool_->UnpinPage(handle, true);
}

Status HnswIndex::ReadAdjacencyLevels(
    std::uint32_t page_id, std::vector<std::vector<VectorId>> *levels) const {
  BufferPool::PageHandle handle;
  Status status = buffer_pool_->FetchPage(page_id, &handle);
  if (!status.ok()) {
    return status;
  }

  const auto *page = As<AdjacencyPage>(handle.data);
  const auto *level_headers = reinterpret_cast<const AdjacencyLevelHeader *>(
      handle.data + sizeof(AdjacencyPage));
  const auto *ids = reinterpret_cast<const VectorId *>(
      handle.data + sizeof(AdjacencyPage) +
      page->level_count * sizeof(AdjacencyLevelHeader));

  levels->assign(page->level_count, {});
  std::size_t cursor = 0;
  for (std::uint16_t i = 0; i < page->level_count; ++i) {
    (*levels)[i].reserve(level_headers[i].count);
    for (std::uint16_t j = 0; j < level_headers[i].count; ++j) {
      (*levels)[i].push_back(ids[cursor++]);
    }
  }

  return buffer_pool_->UnpinPage(handle, false);
}

std::vector<VectorId> HnswIndex::ReadNeighborsUnlocked(VectorId id,
                                                       int layer) const {
  const auto it = directory_.find(id);
  if (it == directory_.end()) {
    return {};
  }

  NodeView node;
  if (!ReadNode(it->second, &node).ok() || layer > node.level) {
    return {};
  }

  std::vector<std::vector<VectorId>> levels;
  if (!ReadAdjacencyLevels(node.adjacency_page_id, &levels).ok()) {
    return {};
  }
  return levels[static_cast<std::size_t>(layer)];
}

Status HnswIndex::WriteNeighborsUnlocked(VectorId id, int layer,
                                         const std::vector<VectorId> &neighbors) {
  const auto it = directory_.find(id);
  if (it == directory_.end()) {
    return Status::NotFound("node not found");
  }

  NodeView node;
  Status status = ReadNode(it->second, &node);
  if (!status.ok()) {
    return status;
  }
  if (layer > node.level) {
    return Status::InvalidArgument("layer out of bounds");
  }

  std::vector<std::vector<VectorId>> levels;
  status = ReadAdjacencyLevels(node.adjacency_page_id, &levels);
  if (!status.ok()) {
    return status;
  }
  edge_count_ -= levels[static_cast<std::size_t>(layer)].size();
  levels[static_cast<std::size_t>(layer)] = neighbors;
  edge_count_ += neighbors.size();
  return WriteAdjacencyLevels(node.adjacency_page_id, levels);
}

int HnswIndex::SampleLevel() {
  int level = 0;
  while (level_distribution_(rng_) < kLevelProbability) {
    ++level;
  }
  return level;
}

std::vector<float> HnswIndex::ReadEmbeddingUnlocked(VectorId id) const {
  const auto it = directory_.find(id);
  if (it == directory_.end()) {
    throw std::invalid_argument("node not found");
  }

  NodeView node;
  const Status status = ReadNode(it->second, &node);
  if (!status.ok()) {
    throw std::invalid_argument("node not found");
  }
  return node.embedding;
}

float HnswIndex::DistanceToNodeUnlocked(const std::vector<float> &query,
                                        VectorId id) const {
  return ComputeDistance(options_.metric, query, ReadEmbeddingUnlocked(id));
}

float HnswIndex::DistanceBetweenNodesUnlocked(VectorId lhs_id,
                                              VectorId rhs_id) const {
  return ComputeDistance(options_.metric, ReadEmbeddingUnlocked(lhs_id),
                         ReadEmbeddingUnlocked(rhs_id));
}

VectorId HnswIndex::GreedySearchAtLayerUnlocked(const std::vector<float> &query,
                                                VectorId entry_id,
                                                int layer) const {
  VectorId best_id = entry_id;
  float best_distance = DistanceToNodeUnlocked(query, best_id);

  bool improved = true;
  while (improved) {
    improved = false;
    for (VectorId neighbor_id : ReadNeighborsUnlocked(best_id, layer)) {
      const float distance = DistanceToNodeUnlocked(query, neighbor_id);
      if (distance < best_distance) {
        best_distance = distance;
        best_id = neighbor_id;
        improved = true;
      }
    }
  }
  return best_id;
}

std::vector<HnswIndex::Candidate> HnswIndex::SearchLayerUnlocked(
    const std::vector<float> &query, const std::vector<VectorId> &entry_points,
    std::size_t ef, int layer) const {
  if (entry_points.empty()) {
    return {};
  }

  std::priority_queue<Candidate, std::vector<Candidate>, MinDistanceCompare>
      candidates;
  std::priority_queue<Candidate, std::vector<Candidate>, MaxDistanceCompare>
      top_candidates;
  std::unordered_set<VectorId> visited;

  for (VectorId entry_id : entry_points) {
    const float distance = DistanceToNodeUnlocked(query, entry_id);
    Candidate candidate{distance, entry_id};
    candidates.push(candidate);
    top_candidates.push(candidate);
    visited.insert(entry_id);
  }

  while (!candidates.empty()) {
    const Candidate current = candidates.top();
    candidates.pop();
    if (top_candidates.size() >= ef &&
        current.distance > top_candidates.top().distance) {
      break;
    }

    for (VectorId neighbor_id : ReadNeighborsUnlocked(current.id, layer)) {
      if (!visited.insert(neighbor_id).second) {
        continue;
      }
      const float distance = DistanceToNodeUnlocked(query, neighbor_id);
      Candidate neighbor{distance, neighbor_id};
      if (top_candidates.size() < ef ||
          distance < top_candidates.top().distance) {
        candidates.push(neighbor);
        top_candidates.push(neighbor);
        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
      }
    }
  }

  std::vector<Candidate> ranked;
  ranked.reserve(top_candidates.size());
  while (!top_candidates.empty()) {
    ranked.push_back(top_candidates.top());
    top_candidates.pop();
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const Candidate &lhs, const Candidate &rhs) {
              return lhs.distance < rhs.distance;
            });
  return ranked;
}

std::vector<VectorId> HnswIndex::SelectNeighborsUnlocked(
    const std::vector<Candidate> &candidates, std::size_t max_neighbors,
    VectorId exclude_id) const {
  std::unordered_map<VectorId, std::vector<float>> embeddings;
  embeddings.reserve(candidates.size());
  for (const Candidate &candidate : candidates) {
    if (candidate.id == exclude_id || embeddings.contains(candidate.id)) {
      continue;
    }
    embeddings.emplace(candidate.id, ReadEmbeddingUnlocked(candidate.id));
  }

  std::vector<VectorId> selected;
  selected.reserve(std::min(max_neighbors, candidates.size()));

  for (const Candidate &candidate : candidates) {
    if (candidate.id == exclude_id) {
      continue;
    }
    bool keep = true;
    for (VectorId selected_id : selected) {
      if (ComputeDistance(options_.metric, embeddings.at(candidate.id),
                          embeddings.at(selected_id)) <
          candidate.distance) {
        keep = false;
        break;
      }
    }
    if (!keep) {
      continue;
    }
    selected.push_back(candidate.id);
    if (selected.size() == max_neighbors) {
      return selected;
    }
  }

  for (const Candidate &candidate : candidates) {
    if (candidate.id == exclude_id) {
      continue;
    }
    if (selected.size() == max_neighbors) {
      break;
    }
    if (std::find(selected.begin(), selected.end(), candidate.id) ==
        selected.end()) {
      selected.push_back(candidate.id);
    }
  }
  return selected;
}

Status HnswIndex::AddBidirectionalLinkUnlocked(VectorId source_id,
                                               VectorId target_id, int layer) {
  std::vector<VectorId> neighbors = ReadNeighborsUnlocked(target_id, layer);
  if (std::find(neighbors.begin(), neighbors.end(), source_id) ==
      neighbors.end()) {
    neighbors.push_back(source_id);
  }
  Status status = WriteNeighborsUnlocked(target_id, layer, neighbors);
  if (!status.ok()) {
    return status;
  }
  return PruneNeighborsUnlocked(target_id, layer, options_.max_degree);
}

Status HnswIndex::PruneNeighborsUnlocked(VectorId node_id, int layer,
                                         std::size_t max_neighbors) {
  std::vector<VectorId> neighbors = ReadNeighborsUnlocked(node_id, layer);
  if (neighbors.size() <= max_neighbors) {
    return Status::OK();
  }

  const std::vector<float> node_embedding = ReadEmbeddingUnlocked(node_id);
  std::vector<Candidate> candidates;
  candidates.reserve(neighbors.size());
  for (VectorId neighbor_id : neighbors) {
    candidates.push_back(Candidate{ComputeDistance(
                                       options_.metric, node_embedding,
                                       ReadEmbeddingUnlocked(neighbor_id)),
                                   neighbor_id});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &lhs, const Candidate &rhs) {
              return lhs.distance < rhs.distance;
            });
  return WriteNeighborsUnlocked(
      node_id, layer,
      SelectNeighborsUnlocked(candidates, max_neighbors, node_id));
}

} // namespace vecbase
