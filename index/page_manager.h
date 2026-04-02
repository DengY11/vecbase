#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "util/lru_k_replacer.h"

namespace vecbase {

struct PageAllocation {
  void *ptr = nullptr;
  std::size_t size = 0;
  std::size_t alignment = 0;
  std::uint32_t page_id = 0;
  std::size_t offset = 0;

  explicit operator bool() const { return ptr != nullptr; }
};

class PageManager {
public:
  virtual ~PageManager() = default;

  virtual PageAllocation Allocate(std::size_t size, std::size_t alignment) = 0;
  virtual void Deallocate(const PageAllocation &allocation) = 0;
  virtual void Touch(const PageAllocation &allocation) = 0;
  virtual std::size_t page_size() const = 0;
};

class PagedMemoryManager final : public PageManager {
public:
  explicit PagedMemoryManager(std::size_t page_size = 16384,
                              std::size_t lru_k = 2);

  PageAllocation Allocate(std::size_t size, std::size_t alignment) override;
  void Deallocate(const PageAllocation &allocation) override;
  void Touch(const PageAllocation &allocation) override;
  std::size_t page_size() const override;

private:
  struct FreeBlock {
    std::size_t offset = 0;
    std::size_t size = 0;
  };

  struct Page {
    std::uint32_t id = 0;
    std::unique_ptr<std::byte[]> data;
    std::vector<FreeBlock> free_blocks;
    std::size_t live_allocations = 0;
    std::size_t live_bytes = 0;
  };

  static std::size_t AlignUp(std::size_t value, std::size_t alignment);

  Page &GetOrCreateAllocPage(std::size_t size, std::size_t alignment);
  bool TryAllocateFromPage(Page &page, std::size_t size, std::size_t alignment,
                           PageAllocation *allocation);
  void MergeFreeBlocks(Page &page);
  void ResetPage(Page &page);
  Page &GetPage(std::uint32_t page_id);
  const Page &GetPage(std::uint32_t page_id) const;
  std::optional<std::uint32_t> AcquireVictimPage();

  std::size_t page_size_ = 0;
  std::uint32_t next_page_id_ = 0;
  LruKReplacer replacer_;
  std::vector<std::unique_ptr<Page>> pages_;
};

} // namespace vecbase
