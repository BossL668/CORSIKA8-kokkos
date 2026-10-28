/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <corsika/kokkos/output/CooperativeProfileMerge.hpp>

namespace corsika::kokkos {

  /** Own bounded OpenMP histogram shards, with production memory limits.
   * Configure/reset only while idle, and finish all writers before mergeInto().
   * A worker's borrowed accumulator must not outlive or resize this owner.
   */
  class HostProfileShards {
  public:
    using accumulator_type = ProfileAccumulatorView;
    using counters_type = ProfileCounters;
    struct alignas(64) Entry {
      counters_type counters_{};
      accumulator_type accumulator_{};
    };
    static constexpr std::size_t MaximumBytes = 16 * 1024 * 1024;
    static constexpr std::size_t MaximumShards = 256;

    static std::size_t bytesPerShard(std::size_t bins);

    void configure(accumulator_type const& base, std::size_t workers,
                   std::size_t available_bytes);

    // Called at beginShower, only with an idle endpoint. Allocation is reused;
    // every bin/counter and the event-specific fixed-point scales are reset.
    void reset(accumulator_type const& base);

    bool enabled() const noexcept;
    std::size_t count() const noexcept;
    std::size_t bytes() const noexcept;
    Entry const* entries() const noexcept;

    // Transactional: no partial integer output is published on overflow or a
    // malformed shard. A failed finalization is sealed, not retried as new data.
    void mergeInto(FixedProfileSnapshot& canonical);

  private:
    std::unique_ptr<long long[]> histograms_;
    std::unique_ptr<Entry[]> entries_;
    accumulator_type base_{};
    std::size_t count_{}, bins_{}, valuesPerShard_{}, bytes_{};
    bool configured_{}, sealed_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/output/HostProfileShards.inl>
