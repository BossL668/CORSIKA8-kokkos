/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#pragma once

#include <corsika/accelerator/em/detail/CooperativeProfileMerge.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

namespace corsika::accelerator::em::detail {

/** Bounded host-only replicas of the existing integer profile ledger.
 *
 * No formula, particle ordering or scale changes. OpenMP workers select a
 * replica; the coordinator combines integers exactly once after all work has
 * finished. Replica collisions remain safe because the original checked
 * atomic operations are used. This is deliberately NOT a per-thread copy of
 * transport workspaces, tables or radio buffers.
 */
class HostProfileShards {
 public:
  using Accumulator = gpu::em::detail::DeviceProfileAccumulator;
  using Counters = gpu::em::detail::DeviceProfileCounters;
  struct alignas(64) Entry {
    Counters counters{};
    Accumulator accumulator{};
  };
  static constexpr std::size_t MaximumBytes = 16 * 1024 * 1024;
  static constexpr std::size_t MaximumShards = 256;

  static std::size_t bytesPerShard(std::size_t bins) {
    constexpr auto columns = gpu::em::detail::DeviceProfileHistogramCount;
    if (bins == 0 || bins > (std::numeric_limits<std::size_t>::max() -
                            sizeof(Entry)) / columns / sizeof(long long))
      throw std::length_error("host profile shard size overflow/empty bins");
    return bins * columns * sizeof(long long) + sizeof(Entry);
  }

  void configure(Accumulator const& base, std::size_t workers,
                 std::size_t available_bytes) {
    if (configured_)
      throw std::logic_error("host profile shards already configured");
    configured_ = true;
    auto const per_shard = bytesPerShard(base.bins);
    auto const allowed = std::min(available_bytes, MaximumBytes) / per_shard;
    // Do not round down to a power of two: e.g. 130 workers sharing 128
    // shards leave two contended pairs whose reduction barrier stalls all
    // workers, even though the budget can easily accommodate 130 replicas.
    auto const count = std::min({workers, allowed, MaximumShards});
    // The canonical accumulator already covers the one-shard case.
    if (workers < 2 || count < 2 || allowed < 2) return;
    count_ = count;
    bins_ = base.bins;
    values_per_shard_ = bins_ * gpu::em::detail::DeviceProfileHistogramCount;
    histograms_ = std::make_unique<long long[]>(count_ * values_per_shard_);
    entries_ = std::make_unique<Entry[]>(count_);
    bytes_ = count_ * per_shard;
    reset(base);
  }

  // Called at beginShower, only with an idle endpoint. Allocation is reused;
  // every bin/counter and the event-specific fixed-point scales are reset.
  void reset(Accumulator const& base) {
    if (!enabled()) return;
    if (base.bins != bins_ || !base.counters ||
        !std::isfinite(base.inverse_weight_scale) ||
        !std::isfinite(base.inverse_energy_scale) ||
        base.inverse_weight_scale <= 0. || base.inverse_energy_scale <= 0.)
      throw std::invalid_argument("host profile shard layout/scale mismatch");
    std::fill_n(histograms_.get(), count_ * values_per_shard_, 0LL);
    base_ = base;
    for (std::size_t i = 0; i < count_; ++i) {
      auto& entry = entries_[i];
      entry.counters = {};
      entry.accumulator = base;
      auto& a = entry.accumulator;
      auto* h = histograms_.get() + i * values_per_shard_;
      a.photons = h;
      a.electrons = h + bins_;
      a.positrons = h + 2 * bins_;
      a.muons_minus = h + 3 * bins_;
      a.muons_plus = h + 4 * bins_;
      a.muon_parent_productions = h + 5 * bins_;
      a.energy_loss = h + 6 * bins_;
      a.muon_energy_loss = h + 7 * bins_;
      a.counters = &entry.counters;
    }
    sealed_ = false;
  }

  bool enabled() const noexcept { return count_ != 0; }
  std::size_t count() const noexcept { return count_; }
  std::size_t bytes() const noexcept { return bytes_; }
  Entry const* entries() const noexcept { return entries_.get(); }

  // Transactional: no partial integer output is published on overflow or a
  // malformed shard. A failed finalization is sealed, not retried as new data.
  void mergeInto(FixedProfileSnapshot& canonical) {
    if (!enabled()) return;
    if (sealed_)
      throw std::logic_error("host profile shards already committed");
    sealed_ = true;
    validateFixedProfile(canonical);
    if (canonical.config.output_bin_count != bins_ ||
        canonical.config.output_bin_width_g_per_cm2 != base_.bin_width_g_per_cm2 ||
        canonical.config.energy_loss_threshold_g_per_cm2 !=
            base_.energy_loss_threshold_g_per_cm2 ||
        canonical.weight_units != base_.inverse_weight_scale ||
        canonical.energy_units != base_.inverse_energy_scale)
      throw std::invalid_argument("host profile shard snapshot mismatch");
    auto merged = canonical;
    std::vector<std::int64_t> row(values_per_shard_);
    FixedAccumulatorLayout layout{canonical.shower_identity + "/host-profile-shards",
                                   canonical.weight_units, values_per_shard_};
    for (std::size_t i = 0; i < count_; ++i) {
      auto const& entry = entries_[i];
      if (entry.counters.fixed_point_overflows || entry.counters.invalid_records)
        throw std::runtime_error("host profile shard invalid/overflowed");
      std::copy_n(histograms_.get() + i * values_per_shard_, values_per_shard_,
                  row.begin());
      mergeFixedAccumulators(merged.histograms, layout, row, layout);
      merged.counters = mergeProfileCounters(merged.counters, entry.counters);
    }
    validateFixedProfile(merged);
    canonical = std::move(merged);
  }

 private:
  std::unique_ptr<long long[]> histograms_;
  std::unique_ptr<Entry[]> entries_;
  Accumulator base_{};
  std::size_t count_{}, bins_{}, values_per_shard_{}, bytes_{};
  bool configured_{}, sealed_{};
};
} // namespace corsika::accelerator::em::detail
