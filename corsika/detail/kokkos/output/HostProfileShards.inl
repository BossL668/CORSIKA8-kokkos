/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  inline std::size_t HostProfileShards::bytesPerShard(std::size_t bins) {
    constexpr auto columns = ProfileHistogramCount;
    if (bins == 0 || bins > (std::numeric_limits<std::size_t>::max() - sizeof(Entry)) /
                                columns / sizeof(long long))
      throw std::length_error("host profile shard size overflow/empty bins");
    return bins * columns * sizeof(long long) + sizeof(Entry);
  }

  inline void HostProfileShards::configure(accumulator_type const& base,
                                           std::size_t workers,
                                           std::size_t available_bytes) {
    if (configured_) throw std::logic_error("host profile shards already configured");
    configured_ = true;
    auto const per_shard = bytesPerShard(base.bins_);
    auto const allowed = std::min(available_bytes, MaximumBytes) / per_shard;
    // Do not round down to a power of two: e.g. 130 workers sharing 128
    // shards leave two contended pairs whose reduction barrier stalls all
    // workers, even though the budget can easily accommodate 130 replicas.
    auto const count = std::min({workers, allowed, MaximumShards});
    // The canonical accumulator already covers the one-shard case.
    if (workers < 2 || count < 2 || allowed < 2) return;
    count_ = count;
    bins_ = base.bins_;
    valuesPerShard_ = bins_ * ProfileHistogramCount;
    histograms_ = std::make_unique<long long[]>(count_ * valuesPerShard_);
    entries_ = std::make_unique<Entry[]>(count_);
    bytes_ = count_ * per_shard;
    reset(base);
  }

  inline void HostProfileShards::reset(accumulator_type const& base) {
    if (!enabled()) return;
    if (base.bins_ != bins_ || !base.counters_ ||
        !std::isfinite(base.inverseWeightScale_) ||
        !std::isfinite(base.inverseEnergyScale_) || base.inverseWeightScale_ <= 0. ||
        base.inverseEnergyScale_ <= 0.)
      throw std::invalid_argument("host profile shard layout/scale mismatch");
    std::fill_n(histograms_.get(), count_ * valuesPerShard_, 0LL);
    base_ = base;
    for (std::size_t i = 0; i < count_; ++i) {
      auto& entry = entries_[i];
      entry.counters_ = {};
      entry.accumulator_ = base;
      auto& a = entry.accumulator_;
      auto* h = histograms_.get() + i * valuesPerShard_;
      a.photons_ = h;
      a.electrons_ = h + bins_;
      a.positrons_ = h + 2 * bins_;
      a.muonsMinus_ = h + 3 * bins_;
      a.muonsPlus_ = h + 4 * bins_;
      a.muonParentProductions_ = h + 5 * bins_;
      a.energyLoss_ = h + 6 * bins_;
      a.muonEnergyLoss_ = h + 7 * bins_;
      a.counters_ = &entry.counters_;
    }
    sealed_ = false;
  }

  inline bool HostProfileShards::enabled() const noexcept { return count_ != 0; }

  inline std::size_t HostProfileShards::count() const noexcept { return count_; }

  inline std::size_t HostProfileShards::bytes() const noexcept { return bytes_; }

  inline HostProfileShards::Entry const* HostProfileShards::entries() const noexcept {
    return entries_.get();
  }

  inline void HostProfileShards::mergeInto(FixedProfileSnapshot& canonical) {
    if (!enabled()) return;
    if (sealed_) throw std::logic_error("host profile shards already committed");
    sealed_ = true;
    validate_fixed_profile(canonical);
    if (canonical.config_.outputBinCount_ != bins_ ||
        canonical.config_.crossingMode_ != base_.crossingMode_ ||
        canonical.config_.outputBinWidthGPerCm2_ != base_.binWidthGPerCm2_ ||
        canonical.config_.energyLossThresholdGPerCm2_ !=
            base_.energyLossThresholdGPerCm2_ ||
        canonical.weightUnits_ != base_.inverseWeightScale_ ||
        canonical.energyUnits_ != base_.inverseEnergyScale_)
      throw std::invalid_argument("host profile shard snapshot mismatch");
    auto merged = canonical;
    std::vector<std::int64_t> row(valuesPerShard_);
    FixedAccumulatorLayout layout{canonical.showerIdentity_ + "/host-profile-shards",
                                  canonical.weightUnits_, valuesPerShard_};
    for (std::size_t i = 0; i < count_; ++i) {
      auto const& entry = entries_[i];
      if (entry.counters_.fixedPointOverflows_ || entry.counters_.invalidRecords_)
        throw std::runtime_error("host profile shard invalid/overflowed");
      std::copy_n(histograms_.get() + i * valuesPerShard_, valuesPerShard_, row.begin());
      merge_fixed_accumulators(merged.histograms_, layout, row, layout);
      merged.counters_ = merge_profile_counters(merged.counters_, entry.counters_);
    }
    validate_fixed_profile(merged);
    canonical = std::move(merged);
  }

} // namespace corsika::kokkos
