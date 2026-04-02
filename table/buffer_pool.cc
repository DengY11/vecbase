#include "table/buffer_pool.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>

#include "util/crc32c.h"

namespace vecbase {

BufferPool::BufferPool(std::string path, std::size_t page_size,
                       std::size_t capacity, std::size_t lru_k)
    : path_(std::move(path)), wal_path_(path_ + ".wal"),
      checkpoint_path_(path_ + ".ckpt"), page_size_(page_size),
      capacity_(capacity), replacer_(lru_k) {}

BufferPool::~BufferPool() {
  stop_flush_.store(true);
  if (flush_thread_.joinable()) {
    flush_thread_.join();
  }
  Flush();
  if (wal_file_.is_open()) {
    wal_file_.close();
  }
}

Status BufferPool::Open(bool create_if_missing) {
  const auto file_path = std::filesystem::path(path_);
  if (create_if_missing) {
    std::filesystem::create_directories(file_path.parent_path());
    std::ofstream touch(path_, std::ios::binary | std::ios::app);
    if (!touch.is_open()) {
      return Status::IOError("failed to create buffer pool file: " + path_);
    }
  } else if (!std::filesystem::exists(path_)) {
    return Status::NotFound("buffer pool file not found: " + path_);
  }

  file_.open(path_, std::ios::binary | std::ios::in | std::ios::out);
  if (!file_.is_open()) {
    return Status::IOError("failed to open buffer pool file: " + path_);
  }

  CheckpointMeta checkpoint;
  Status status = LoadCheckpointMeta(&checkpoint);
  if (!status.ok()) {
    return status;
  }
  page_count_ = checkpoint.page_count;
  logical_version_ = checkpoint.logical_version;
  root_page_id_ = checkpoint.root_page_id;
  checkpoint_seq_ = checkpoint.checkpoint_seq;
  if (page_count_ == 0) {
    const std::uint64_t size = std::filesystem::file_size(path_);
    page_count_ =
        static_cast<std::uint32_t>((size + page_size_ - 1) / page_size_);
  }

  status = OpenWal(create_if_missing);
  if (!status.ok()) {
    return status;
  }
  status = ReplayWal();
  if (!status.ok()) {
    return status;
  }

  stop_flush_.store(false);
  flush_thread_ = std::thread(&BufferPool::FlushLoop, this);
  return Status::OK();
}

Status BufferPool::FetchPage(std::uint32_t page_id, PageHandle *handle) {
  std::lock_guard lock(mutex_);
  if (handle == nullptr) {
    return Status::InvalidArgument("page handle must not be null");
  }
  Frame *frame = nullptr;
  Status status = LoadFrame(page_id, &frame);
  if (!status.ok()) {
    return status;
  }

  frame->pin_count += 1;
  replacer_.RecordAccess(page_id);
  replacer_.SetEvictable(page_id, false);
  handle->page_id = page_id;
  handle->data = frame->data.data();
  handle->size = frame->data.size();
  handle->dirty = frame->dirty;
  return Status::OK();
}

Status BufferPool::NewPage(PageHandle *handle) {
  std::lock_guard lock(mutex_);
  if (handle == nullptr) {
    return Status::InvalidArgument("page handle must not be null");
  }

  Frame *frame = nullptr;
  const std::uint32_t page_id = page_count_++;
  Status status = AllocateFrame(page_id, &frame);
  if (!status.ok()) {
    return status;
  }

  std::fill(frame->data.begin(), frame->data.end(), std::byte{0});
  frame->dirty = true;
  frame->pin_count = 1;
  replacer_.RecordAccess(page_id);
  replacer_.SetEvictable(page_id, false);

  handle->page_id = page_id;
  handle->data = frame->data.data();
  handle->size = frame->data.size();
  handle->dirty = true;
  status = AppendWalRecord(*frame);
  if (!status.ok()) {
    return status;
  }
  EnqueueDirtyPage(page_id);
  return Status::OK();
}

Status BufferPool::DeletePage(std::uint32_t page_id) {
  std::lock_guard lock(mutex_);
  const auto it = frames_.find(page_id);
  if (it != frames_.end()) {
    if (it->second->pin_count != 0) {
      return Status::IOError("cannot delete pinned page");
    }
    replacer_.Remove(page_id);
    dirty_set_.erase(page_id);
    frames_.erase(it);
  }
  return Status::OK();
}

Status BufferPool::UnpinPage(const PageHandle &handle, bool dirty) {
  std::lock_guard lock(mutex_);
  const auto it = frames_.find(handle.page_id);
  if (it == frames_.end()) {
    return Status::NotFound("page not found in buffer pool");
  }

  Frame *frame = it->second.get();
  if (frame->pin_count == 0) {
    return Status::InvalidArgument("page is not pinned");
  }

  frame->pin_count -= 1;
  frame->dirty = frame->dirty || dirty;
  if (dirty) {
    Status status = AppendWalRecord(*frame);
    if (!status.ok()) {
      return status;
    }
    EnqueueDirtyPage(handle.page_id);
  }
  if (frame->pin_count == 0) {
    replacer_.SetEvictable(handle.page_id, true);
  }
  return Status::OK();
}

Status BufferPool::FlushPage(std::uint32_t page_id) {
  std::lock_guard lock(mutex_);
  const auto it = frames_.find(page_id);
  if (it == frames_.end()) {
    return Status::NotFound("page not found in buffer pool");
  }
  Status status = FlushFrame(it->second.get());
  if (!status.ok()) {
    return status;
  }
  dirty_set_.erase(page_id);
  return StoreCheckpointMeta();
}

Status BufferPool::Flush() {
  std::lock_guard lock(mutex_);
  if (!file_.is_open()) {
    return Status::OK();
  }
  if (dirty_queue_.empty()) {
    return Status::OK();
  }

  while (!dirty_queue_.empty()) {
    const std::uint32_t page_id = dirty_queue_.front();
    dirty_queue_.pop_front();

    const auto it = frames_.find(page_id);
    if (it == frames_.end()) {
      dirty_set_.erase(page_id);
      continue;
    }

    Status status = FlushFrame(it->second.get());
    if (!status.ok()) {
      return status;
    }
    dirty_set_.erase(page_id);
  }

  file_.flush();
  if (!file_) {
    return Status::IOError("failed to flush buffer pool file");
  }

  Status status = StoreCheckpointMeta();
  if (!status.ok()) {
    return status;
  }
  return ResetWal();
}

Status BufferPool::SetCheckpointState(std::uint32_t logical_version,
                                      std::uint32_t root_page_id) {
  std::lock_guard lock(mutex_);
  logical_version_ = logical_version;
  root_page_id_ = root_page_id;
  return StoreCheckpointMeta();
}

BufferPool::CheckpointState BufferPool::checkpoint_state() const {
  std::lock_guard lock(mutex_);
  return CheckpointState{
      logical_version_,
      root_page_id_,
      page_count_,
      checkpoint_seq_,
  };
}

void BufferPool::FlushLoop() {
  while (!stop_flush_.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    Flush();
  }
}

std::uint32_t
BufferPool::ComputeChecksum(const std::vector<std::byte> &data) const {
  return CRC32C(data);
}

void BufferPool::EnqueueDirtyPage(std::uint32_t page_id) {
  if (dirty_set_.insert(page_id).second) {
    dirty_queue_.push_back(page_id);
  }
}

Status BufferPool::LoadCheckpointMeta(CheckpointMeta *meta) {
  if (meta == nullptr) {
    return Status::InvalidArgument("checkpoint meta must not be null");
  }

  if (!std::filesystem::exists(checkpoint_path_)) {
    meta->page_size = static_cast<std::uint32_t>(page_size_);
    meta->page_count = 0;
    meta->logical_version = 0;
    meta->root_page_id = 0;
    meta->checkpoint_seq = 0;
    return StoreCheckpointMeta();
  }

  std::ifstream input(checkpoint_path_, std::ios::binary);
  if (!input.is_open()) {
    return Status::IOError("failed to open checkpoint metadata");
  }
  input.read(reinterpret_cast<char *>(meta), sizeof(*meta));
  if (!input) {
    return Status::IOError("failed to read checkpoint metadata");
  }
  if (meta->magic != kCheckpointMagic) {
    return Status::IOError("invalid checkpoint metadata magic");
  }
  if (meta->page_size != page_size_) {
    return Status::IOError("checkpoint page size mismatch");
  }
  return Status::OK();
}

Status BufferPool::StoreCheckpointMeta() {
  CheckpointMeta meta;
  meta.page_size = static_cast<std::uint32_t>(page_size_);
  meta.page_count = page_count_;
  meta.logical_version = logical_version_;
  meta.root_page_id = root_page_id_;
  meta.checkpoint_seq = ++checkpoint_seq_;

  std::ofstream output(checkpoint_path_,
                       std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    return Status::IOError("failed to open checkpoint metadata for write");
  }
  output.write(reinterpret_cast<const char *>(&meta), sizeof(meta));
  if (!output) {
    return Status::IOError("failed to write checkpoint metadata");
  }
  output.flush();
  return output ? Status::OK()
                : Status::IOError("failed to flush checkpoint metadata");
}

Status BufferPool::OpenWal(bool create_if_missing) {
  if (create_if_missing) {
    std::ofstream touch(wal_path_, std::ios::binary | std::ios::app);
    if (!touch.is_open()) {
      return Status::IOError("failed to create WAL file: " + wal_path_);
    }
  }

  wal_file_.open(wal_path_, std::ios::binary | std::ios::in | std::ios::out);
  if (!wal_file_.is_open()) {
    return Status::IOError("failed to open WAL file: " + wal_path_);
  }
  wal_file_.seekp(0, std::ios::end);
  return Status::OK();
}

Status BufferPool::ReplayWal() {
  wal_file_.seekg(0, std::ios::beg);
  while (true) {
    WalRecordHeader header;
    wal_file_.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (wal_file_.gcount() == 0) {
      break;
    }
    if (wal_file_.gcount() != static_cast<std::streamsize>(sizeof(header))) {
      return Status::IOError("corrupted page WAL header");
    }
    if (header.magic != kWalMagic) {
      return Status::IOError("invalid page WAL magic");
    }
    if (header.payload_size != page_size_) {
      return Status::IOError("unexpected page WAL payload size");
    }

    std::vector<std::byte> payload(page_size_);
    wal_file_.read(reinterpret_cast<char *>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
    if (!wal_file_) {
      return Status::IOError("corrupted page WAL payload");
    }
    if (ComputeChecksum(payload) != header.checksum) {
      return Status::IOError("page WAL checksum mismatch");
    }

    file_.seekp(static_cast<std::streamoff>(OffsetForPage(header.page_id)));
    file_.write(reinterpret_cast<const char *>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
    if (!file_) {
      return Status::IOError("failed to replay page WAL");
    }
    page_count_ = std::max(page_count_, header.page_id + 1);
  }

  file_.flush();
  wal_file_.clear();
  wal_file_.seekp(0, std::ios::end);
  Status status = StoreCheckpointMeta();
  if (!status.ok()) {
    return status;
  }
  return ResetWal();
}

Status BufferPool::AppendWalRecord(const Frame &frame) {
  WalRecordHeader header;
  header.page_id = frame.page_id;
  header.payload_size = static_cast<std::uint32_t>(frame.data.size());
  header.checksum = ComputeChecksum(frame.data);

  wal_file_.write(reinterpret_cast<const char *>(&header), sizeof(header));
  wal_file_.write(reinterpret_cast<const char *>(frame.data.data()),
                  static_cast<std::streamsize>(frame.data.size()));
  wal_file_.flush();
  return wal_file_ ? Status::OK() : Status::IOError("failed to append page WAL");
}

Status BufferPool::ResetWal() {
  wal_file_.close();
  wal_file_.open(wal_path_, std::ios::binary | std::ios::in | std::ios::out |
                                std::ios::trunc);
  if (!wal_file_.is_open()) {
    return Status::IOError("failed to reset page WAL");
  }
  return Status::OK();
}

Status BufferPool::LoadFrame(std::uint32_t page_id, Frame **frame) {
  const auto found = frames_.find(page_id);
  if (found != frames_.end()) {
    *frame = found->second.get();
    return Status::OK();
  }
  return AllocateFrame(page_id, frame);
}

Status BufferPool::AllocateFrame(std::uint32_t page_id, Frame **frame) {
  if (frames_.size() >= capacity_) {
    Status status = EvictOne();
    if (!status.ok()) {
      return status;
    }
  }

  auto owned = std::make_unique<Frame>();
  owned->page_id = page_id;
  owned->data.resize(page_size_, std::byte{0});

  if (page_id < page_count_) {
    file_.seekg(static_cast<std::streamoff>(OffsetForPage(page_id)));
    file_.read(reinterpret_cast<char *>(owned->data.data()),
               static_cast<std::streamsize>(page_size_));
    file_.clear();
  }

  replacer_.RecordAccess(page_id);
  replacer_.SetEvictable(page_id, true);
  auto [it, _] = frames_.emplace(page_id, std::move(owned));
  *frame = it->second.get();
  return Status::OK();
}

Status BufferPool::EvictOne() {
  const auto victim = replacer_.Evict();
  if (!victim.has_value()) {
    return Status::IOError("no evictable page available");
  }

  const auto it = frames_.find(*victim);
  if (it == frames_.end()) {
    return Status::OK();
  }
  if (it->second->pin_count != 0) {
    return Status::IOError("victim page is still pinned");
  }

  Status status = FlushFrame(it->second.get());
  if (!status.ok()) {
    return status;
  }
  dirty_set_.erase(*victim);
  frames_.erase(it);
  return Status::OK();
}

Status BufferPool::FlushFrame(Frame *frame) {
  if (frame == nullptr || !frame->dirty) {
    return Status::OK();
  }

  file_.seekp(static_cast<std::streamoff>(OffsetForPage(frame->page_id)));
  file_.write(reinterpret_cast<const char *>(frame->data.data()),
              static_cast<std::streamsize>(frame->data.size()));
  if (!file_) {
    return Status::IOError("failed to write page to buffer pool file");
  }
  frame->dirty = false;
  return Status::OK();
}

std::uint64_t BufferPool::OffsetForPage(std::uint32_t page_id) const {
  return static_cast<std::uint64_t>(page_id) * page_size_;
}

} // namespace vecbase
