#include "index/page_manager.h"

#include <algorithm>
#include <cassert>
#include <new>
#include <stdexcept>

namespace vecbase {

PagedMemoryManager::PagedMemoryManager(std::size_t page_size, std::size_t lru_k)
    : page_size_(page_size), replacer_(lru_k) {
  if (page_size_ == 0) {
    throw std::invalid_argument("page size must be greater than zero");
  }
}

PageAllocation PagedMemoryManager::Allocate(std::size_t size,
                                            std::size_t alignment) {
  if (size == 0) {
    return {};
  }
  if (alignment == 0) {
    alignment = alignof(std::max_align_t);
  }
  if (size > page_size_) {
    throw std::bad_alloc();
  }

  Page &page = GetOrCreateAllocPage(size, alignment);
  PageAllocation allocation;
  if (!TryAllocateFromPage(page, size, alignment, &allocation)) {
    throw std::bad_alloc();
  }

  replacer_.RecordAccess(page.id);
  replacer_.SetEvictable(page.id, false);
  return allocation;
}

void PagedMemoryManager::Deallocate(const PageAllocation &allocation) {
  if (!allocation) {
    return;
  }

  Page &page = GetPage(allocation.page_id);
  page.free_blocks.push_back(FreeBlock{allocation.offset, allocation.size});
  page.live_allocations -= 1;
  page.live_bytes -= allocation.size;
  MergeFreeBlocks(page);

  replacer_.RecordAccess(page.id);
  replacer_.SetEvictable(page.id, page.live_allocations == 0);
}

void PagedMemoryManager::Touch(const PageAllocation &allocation) {
  if (!allocation) {
    return;
  }

  const Page &page = GetPage(allocation.page_id);
  replacer_.RecordAccess(page.id);
  replacer_.SetEvictable(page.id, page.live_allocations == 0);
}

std::size_t PagedMemoryManager::page_size() const { return page_size_; }

std::size_t PagedMemoryManager::AlignUp(std::size_t value,
                                        std::size_t alignment) {
  const std::size_t mask = alignment - 1;
  return (value + mask) & ~mask;
}

PagedMemoryManager::Page &
PagedMemoryManager::GetOrCreateAllocPage(std::size_t size,
                                         std::size_t alignment) {
  for (const auto &page_ptr : pages_) {
    if (page_ptr == nullptr) {
      continue;
    }

    PageAllocation allocation;
    if (TryAllocateFromPage(*page_ptr, size, alignment, &allocation)) {
      Deallocate(allocation);
      return *page_ptr;
    }
  }

  const std::optional<std::uint32_t> victim_page_id = AcquireVictimPage();
  if (victim_page_id.has_value()) {
    Page &victim = GetPage(*victim_page_id);
    ResetPage(victim);
    return victim;
  }

  auto page = std::make_unique<Page>();
  page->id = next_page_id_++;
  page->data = std::make_unique<std::byte[]>(page_size_);
  page->free_blocks.push_back(FreeBlock{0, page_size_});
  replacer_.RecordAccess(page->id);
  replacer_.SetEvictable(page->id, true);

  pages_.push_back(std::move(page));
  return *pages_.back();
}

bool PagedMemoryManager::TryAllocateFromPage(Page &page, std::size_t size,
                                             std::size_t alignment,
                                             PageAllocation *allocation) {
  for (std::size_t i = 0; i < page.free_blocks.size(); ++i) {
    const FreeBlock block = page.free_blocks[i];
    const std::size_t aligned_offset = AlignUp(block.offset, alignment);
    const std::size_t padding = aligned_offset - block.offset;
    if (block.size < padding + size) {
      continue;
    }

    page.free_blocks.erase(page.free_blocks.begin() +
                           static_cast<std::ptrdiff_t>(i));
    if (padding > 0) {
      page.free_blocks.push_back(FreeBlock{block.offset, padding});
    }

    const std::size_t suffix_offset = aligned_offset + size;
    const std::size_t suffix_size = (block.offset + block.size) - suffix_offset;
    if (suffix_size > 0) {
      page.free_blocks.push_back(FreeBlock{suffix_offset, suffix_size});
    }
    MergeFreeBlocks(page);

    page.live_allocations += 1;
    page.live_bytes += size;
    allocation->ptr = page.data.get() + aligned_offset;
    allocation->size = size;
    allocation->alignment = alignment;
    allocation->page_id = page.id;
    allocation->offset = aligned_offset;
    return true;
  }

  return false;
}

void PagedMemoryManager::MergeFreeBlocks(Page &page) {
  std::sort(page.free_blocks.begin(), page.free_blocks.end(),
            [](const FreeBlock &lhs, const FreeBlock &rhs) {
              return lhs.offset < rhs.offset;
            });

  std::vector<FreeBlock> merged;
  for (const FreeBlock &block : page.free_blocks) {
    if (merged.empty() ||
        merged.back().offset + merged.back().size < block.offset) {
      merged.push_back(block);
      continue;
    }

    FreeBlock &tail = merged.back();
    const std::size_t new_end =
        std::max(tail.offset + tail.size, block.offset + block.size);
    tail.size = new_end - tail.offset;
  }
  page.free_blocks = std::move(merged);
}

void PagedMemoryManager::ResetPage(Page &page) {
  page.free_blocks.clear();
  page.free_blocks.push_back(FreeBlock{0, page_size_});
  page.live_allocations = 0;
  page.live_bytes = 0;
  replacer_.RecordAccess(page.id);
  replacer_.SetEvictable(page.id, true);
}

PagedMemoryManager::Page &PagedMemoryManager::GetPage(std::uint32_t page_id) {
  if (page_id >= pages_.size() || pages_[page_id] == nullptr) {
    throw std::out_of_range("page id is out of range");
  }
  return *pages_[page_id];
}

const PagedMemoryManager::Page &
PagedMemoryManager::GetPage(std::uint32_t page_id) const {
  if (page_id >= pages_.size() || pages_[page_id] == nullptr) {
    throw std::out_of_range("page id is out of range");
  }
  return *pages_[page_id];
}

std::optional<std::uint32_t> PagedMemoryManager::AcquireVictimPage() {
  return replacer_.Evict();
}

} // namespace vecbase
