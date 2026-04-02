#include "util/lru_k_replacer.h"

#include <limits>

namespace vecbase {

LruKReplacer::LruKReplacer(std::size_t k) : k_(k == 0 ? 1 : k) {}

void LruKReplacer::RecordAccess(std::uint32_t frame_id) {
  FrameState &state = frames_[frame_id];
  state.history.push_back(++current_timestamp_);
  if (state.history.size() > k_) {
    state.history.erase(state.history.begin());
  }
}

void LruKReplacer::SetEvictable(std::uint32_t frame_id, bool evictable) {
  FrameState &state = frames_[frame_id];
  if (state.evictable == evictable) {
    return;
  }

  state.evictable = evictable;
  if (evictable) {
    ++evictable_count_;
  } else {
    --evictable_count_;
  }
}

std::optional<std::uint32_t> LruKReplacer::Evict() {
  std::optional<std::uint32_t> victim;
  bool victim_has_infinite_distance = false;
  std::size_t best_distance = 0;
  std::size_t oldest_access = 0;

  for (const auto &[frame_id, state] : frames_) {
    if (!state.evictable || state.history.empty()) {
      continue;
    }

    const bool has_infinite_distance = state.history.size() < k_;
    const std::size_t distance = BackwardKDistance(state);
    const std::size_t first_access = state.history.front();

    if (!victim.has_value()) {
      victim = frame_id;
      victim_has_infinite_distance = has_infinite_distance;
      best_distance = distance;
      oldest_access = first_access;
      continue;
    }

    if (has_infinite_distance != victim_has_infinite_distance) {
      if (has_infinite_distance) {
        victim = frame_id;
        victim_has_infinite_distance = true;
        best_distance = distance;
        oldest_access = first_access;
      }
      continue;
    }

    if (distance > best_distance ||
        (distance == best_distance && first_access < oldest_access)) {
      victim = frame_id;
      best_distance = distance;
      oldest_access = first_access;
    }
  }

  if (!victim.has_value()) {
    return std::nullopt;
  }

  Remove(*victim);
  return victim;
}

void LruKReplacer::Remove(std::uint32_t frame_id) {
  const auto it = frames_.find(frame_id);
  if (it == frames_.end()) {
    return;
  }

  if (it->second.evictable) {
    --evictable_count_;
  }
  frames_.erase(it);
}

std::size_t LruKReplacer::Size() const { return evictable_count_; }

std::size_t LruKReplacer::BackwardKDistance(const FrameState &state) const {
  if (state.history.size() < k_) {
    return std::numeric_limits<std::size_t>::max();
  }
  return current_timestamp_ - state.history.front();
}

} // namespace vecbase
