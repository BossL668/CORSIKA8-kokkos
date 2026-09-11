/* Internal host-only scheduling primitives. No particle physics or shared Stack. */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace corsika::accelerator::em::detail {
enum class CooperativeEndpoint : unsigned { Cuda, OpenMP };
enum class CooperativePhase { Ready, Submitted, WaitingForControl, ResultsReady, Committed, Failed };

// Bounded one-in-flight/one-staged state; coordinator is the sole owner.
class CooperativeBatchState {
 public:
  void stage(std::uint64_t id) {
    if (phase_ == CooperativePhase::Failed)
      throw std::logic_error("cannot stage work after cooperative failure");
    if (!id || staged_) throw std::logic_error("invalid or duplicate staged batch");
    if (id <= last_id_) throw std::logic_error("batch identity must increase");
    staged_ = id; last_id_ = id;
  }
  std::uint64_t submit() {
    if (!staged_ || (phase_ != CooperativePhase::Ready && phase_ != CooperativePhase::Committed))
      throw std::logic_error("batch already in flight or no staged input");
    active_ = staged_; staged_ = 0; phase_ = CooperativePhase::Submitted;
    return active_;
  }
  void waiting() { transition(CooperativePhase::Submitted, CooperativePhase::WaitingForControl); }
  void controlReady(bool final) {
    transition(CooperativePhase::WaitingForControl,
               final ? CooperativePhase::ResultsReady : CooperativePhase::Submitted);
  }
  void commit(std::uint64_t id) {
    if (id != active_) throw std::logic_error("wrong batch commit");
    transition(CooperativePhase::ResultsReady, CooperativePhase::Committed);
    active_ = 0;
  }
  void fail() noexcept { phase_ = CooperativePhase::Failed; }
  bool inFlight() const noexcept { return active_ != 0; }
  bool pending() const noexcept { return active_ || staged_; }
  CooperativePhase phase() const noexcept { return phase_; }
 private:
  void transition(CooperativePhase from, CooperativePhase to) {
    if (phase_ != from) throw std::logic_error("invalid cooperative batch transition");
    phase_ = to;
  }
  std::uint64_t active_{}, staged_{}, last_id_{};
  CooperativePhase phase_{CooperativePhase::Ready};
};

struct CooperativeDispatch { CooperativeEndpoint endpoint; std::size_t particles; };

class CooperativeLoadBalancer {
 public:
  explicit CooperativeLoadBalancer(std::size_t gpu_min = 4096) : gpu_min_(gpu_min) {
    if (!gpu_min_) throw std::invalid_argument("GPU minimum batch must be positive");
  }
  void observe(CooperativeEndpoint endpoint, std::size_t work, double total_ms) {
    if (!work || !std::isfinite(total_ms) || total_ms <= 0.)
      throw std::invalid_argument("invalid cooperative timing sample");
    auto const index = static_cast<unsigned>(endpoint);
    if (index >= rates_.size())
      throw std::invalid_argument("invalid cooperative endpoint");
    auto& rate = rates_[index];
    auto const measured = static_cast<double>(work) / total_ms;
    if (!std::isfinite(measured))
      throw std::invalid_argument("nonfinite cooperative throughput");
    rate = rate == 0. ? measured : .8 * rate + .2 * measured;
    if (endpoint == CooperativeEndpoint::OpenMP)
      cpu_batch_ = static_cast<std::size_t>(std::clamp(2. * rate, 256., 2048.));
  }
  CooperativeDispatch choose(std::size_t ready, bool gpu_busy, bool cpu_busy,
                             std::size_t gpu_capacity, std::size_t cpu_capacity,
                             double gpu_backlog_ms = 0., double cpu_backlog_ms = 0.) const {
    if (!std::isfinite(gpu_backlog_ms) || gpu_backlog_ms < 0. ||
        !std::isfinite(cpu_backlog_ms) || cpu_backlog_ms < 0.)
      throw std::invalid_argument("invalid backlog prediction");
    if (!ready) return {CooperativeEndpoint::Cuda, 0};
    // Do not deliberately drain an idle GPU's viable front into CPU work.
    if (!gpu_busy && gpu_capacity && (ready >= gpu_min_ || cpu_busy))
      return {CooperativeEndpoint::Cuda, std::min(ready, gpu_capacity)};
    if (!cpu_busy && cpu_capacity && gpu_busy)
      return {CooperativeEndpoint::OpenMP, std::min({ready, cpu_batch_, cpu_capacity})};
    if (gpu_busy && cpu_busy) return {CooperativeEndpoint::Cuda, 0};
    if (!gpu_busy && gpu_capacity) {
      auto const ncpu = std::min({ready, cpu_batch_, cpu_capacity});
      if (!cpu_busy && ncpu && rates_[0] > 0. && rates_[1] > 0. &&
          cpu_backlog_ms + ncpu / rates_[1] < gpu_backlog_ms + ncpu / rates_[0])
        return {CooperativeEndpoint::OpenMP, ncpu};
      return {CooperativeEndpoint::Cuda, std::min(ready, gpu_capacity)};
    }
    return {CooperativeEndpoint::OpenMP, cpu_busy ? 0 : std::min({ready, cpu_batch_, cpu_capacity})};
  }
  std::size_t cpuBatch() const noexcept { return cpu_batch_; }
 private:
  std::size_t gpu_min_, cpu_batch_{256};
  std::array<double, 2> rates_{};
};

struct CooperativeHistoryRange { std::uint64_t first, limit; };
inline CooperativeHistoryRange checkedCooperativeHistoryRange(std::uint64_t first, std::uint64_t count) {
  if (!first || !count || count > std::numeric_limits<std::uint64_t>::max() - first)
    throw std::overflow_error("cooperative history reservation overflow");
  return {first, first + count};
}

// One allocator belongs to the coordinator, never to individual endpoints.
// Reservations are never recycled, including after a failed or empty batch.
class CooperativeHistoryAllocator {
 public:
  explicit CooperativeHistoryAllocator(std::uint64_t first) : next_(first) {
    if (!first) throw std::invalid_argument("history zero is reserved");
  }
  CooperativeHistoryRange reserve(std::uint64_t count) {
    auto range = checkedCooperativeHistoryRange(next_, count);
    next_ = range.limit;
    return range;
  }
  std::uint64_t next() const noexcept { return next_; }
 private:
  std::uint64_t next_;
};
} // namespace corsika::accelerator::em::detail
