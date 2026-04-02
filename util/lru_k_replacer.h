#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vecbase {

class LruKReplacer {
public:
  explicit LruKReplacer(std::size_t k);

  void RecordAccess(std::uint32_t frame_id);
  void SetEvictable(std::uint32_t frame_id, bool evictable);
  std::optional<std::uint32_t> Evict();
  void Remove(std::uint32_t frame_id);
  std::size_t Size() const;

private:
  struct FrameState {
    bool evictable = false;
    std::vector<std::size_t> history;
  };

  std::size_t BackwardKDistance(const FrameState &state) const;

  std::size_t current_timestamp_ = 0;
  std::size_t k_ = 0;
  std::unordered_map<std::uint32_t, FrameState> frames_;
  std::size_t evictable_count_ = 0;
};

} // namespace vecbase
