#pragma once

#include <cstddef>
#include <cstdint>

namespace vecbase {

constexpr std::uint32_t kIndexMagic = 0x56425848; // VBXH
constexpr std::uint32_t kIndexVersion = 1;
constexpr std::uint32_t kInvalidPageId = 0xffffffffu;

enum class IndexPageType : std::uint16_t {
  kMeta = 1,
  kDirectory = 2,
  kNode = 3,
  kAdjacency = 4,
  kFreelist = 5,
};

struct IndexPageHeader {
  std::uint32_t magic = kIndexMagic;
  std::uint16_t version = kIndexVersion;
  std::uint16_t type = 0;
  std::uint32_t page_id = 0;
};

struct MetaPage {
  IndexPageHeader header;
  std::uint32_t page_size = 0;
  std::uint32_t dimension = 0;
  std::uint32_t metric = 0;
  std::uint32_t max_degree = 0;
  std::uint32_t ef_construction = 0;
  std::uint64_t next_node_id = 1;
  std::uint64_t node_count = 0;
  std::uint64_t entry_node_id = 0;
  std::int32_t max_level = -1;
  std::uint32_t directory_page_id = 0;
  std::uint32_t freelist_page_id = 0;
  std::uint32_t next_page_id = 1;
};

struct DirectoryEntry {
  std::uint64_t vector_id = 0;
  std::uint64_t node_id = 0;
  std::uint32_t page_id = 0;
  std::uint16_t slot_id = 0;
  std::uint16_t flags = 0;
};

struct DirectoryPage {
  IndexPageHeader header;
  std::uint32_t next_page_id = kInvalidPageId;
  std::uint16_t entry_count = 0;
  std::uint16_t capacity = 0;
};

struct NodePage {
  IndexPageHeader header;
  std::uint32_t next_page_id = kInvalidPageId;
  std::uint16_t slot_count = 0;
  std::uint16_t used_count = 0;
};

struct NodeSlotHeader {
  std::uint64_t vector_id = 0;
  std::uint64_t node_id = 0;
  std::uint32_t adjacency_page_id = kInvalidPageId;
  std::uint16_t level = 0;
  std::uint16_t flags = 0;
};

struct AdjacencyPage {
  IndexPageHeader header;
  std::uint32_t next_page_id = kInvalidPageId;
  std::uint64_t node_id = 0;
  std::uint16_t level_count = 0;
  std::uint16_t reserved = 0;
};

struct AdjacencyLevelHeader {
  std::uint16_t count = 0;
  std::uint16_t capacity = 0;
};

struct FreelistPage {
  IndexPageHeader header;
  std::uint32_t next_page_id = kInvalidPageId;
  std::uint16_t page_count = 0;
  std::uint16_t reserved = 0;
};

struct NodeLocation {
  std::uint32_t page_id = kInvalidPageId;
  std::uint16_t slot_id = 0;
  std::uint16_t reserved = 0;
};

} // namespace vecbase
