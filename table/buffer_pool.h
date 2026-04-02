#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "util/lru_k_replacer.h"
#include "vecbase/status.h"

namespace vecbase {

class BufferPool {
public:
  struct PageHandle {
    std::uint32_t page_id = 0;
    std::byte *data = nullptr;
    std::size_t size = 0;
    bool dirty = false;

    explicit operator bool() const { return data != nullptr; }
  };

  struct CheckpointState {
    std::uint32_t logical_version = 0;
    std::uint32_t root_page_id = 0;
    std::uint32_t page_count = 0;
    std::uint64_t checkpoint_seq = 0;
  };

  BufferPool(std::string path, std::size_t page_size = 4096,
             std::size_t capacity = 128, std::size_t lru_k = 2);
  ~BufferPool();

  Status Open(bool create_if_missing);
  Status FetchPage(std::uint32_t page_id, PageHandle *handle);
  Status NewPage(PageHandle *handle);
  Status DeletePage(std::uint32_t page_id);
  Status UnpinPage(const PageHandle &handle, bool dirty);
  Status FlushPage(std::uint32_t page_id);
  Status Flush();
  Status SetCheckpointState(std::uint32_t logical_version,
                            std::uint32_t root_page_id);
  CheckpointState checkpoint_state() const;

  std::size_t page_size() const { return page_size_; }
  std::uint32_t page_count() const { return page_count_; }

private:
  static constexpr std::uint32_t kWalMagic = 0x5642574c; // VBWL
  static constexpr std::uint32_t kCheckpointMagic = 0x56424350; // VBCP

  struct WalRecordHeader {
    std::uint32_t magic = kWalMagic;
    std::uint32_t page_id = 0;
    std::uint32_t payload_size = 0;
    std::uint32_t checksum = 0;
  };

  struct CheckpointMeta {
    std::uint32_t magic = kCheckpointMagic;
    std::uint32_t version = 1;
    std::uint32_t page_size = 0;
    std::uint32_t reserved = 0;
    std::uint32_t page_count = 0;
    std::uint32_t logical_version = 0;
    std::uint32_t root_page_id = 0;
    std::uint64_t checkpoint_seq = 0;
  };

  struct Frame {
    std::uint32_t page_id = 0;
    std::vector<std::byte> data;
    bool dirty = false;
    std::size_t pin_count = 0;
  };

  void FlushLoop();
  std::uint32_t ComputeChecksum(const std::vector<std::byte> &data) const;
  void EnqueueDirtyPage(std::uint32_t page_id);
  Status LoadCheckpointMeta(CheckpointMeta *meta);
  Status StoreCheckpointMeta();
  Status OpenWal(bool create_if_missing);
  Status ReplayWal();
  Status AppendWalRecord(const Frame &frame);
  Status ResetWal();
  Status LoadFrame(std::uint32_t page_id, Frame **frame);
  Status AllocateFrame(std::uint32_t page_id, Frame **frame);
  Status EvictOne();
  Status FlushFrame(Frame *frame);
  std::uint64_t OffsetForPage(std::uint32_t page_id) const;

  std::string path_;
  std::string wal_path_;
  std::string checkpoint_path_;
  std::size_t page_size_ = 0;
  std::size_t capacity_ = 0;
  std::uint32_t page_count_ = 0;
  std::uint32_t logical_version_ = 0;
  std::uint32_t root_page_id_ = 0;
  std::uint64_t checkpoint_seq_ = 0;
  LruKReplacer replacer_;
  std::fstream file_;
  std::fstream wal_file_;
  mutable std::recursive_mutex mutex_;
  std::thread flush_thread_;
  std::atomic<bool> stop_flush_{false};
  std::deque<std::uint32_t> dirty_queue_;
  std::unordered_set<std::uint32_t> dirty_set_;
  std::unordered_map<std::uint32_t, std::unique_ptr<Frame>> frames_;
};

} // namespace vecbase
