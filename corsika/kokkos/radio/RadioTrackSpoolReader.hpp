/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/RadioTrackSpoolFormat.hpp>

namespace corsika::kokkos::radio {
  /** Bounded, ordered handoff of already transported CPU tracks.
   * Keeps the production version-1 little-endian format; no host ABI or RNG. */
  class RadioTrackSpoolReader {
    std::ifstream file_;
    std::uint64_t count_{};
    std::uint64_t consumed_{};

  public:
    explicit RadioTrackSpoolReader(std::filesystem::path const& path);
    std::size_t readBatch(std::vector<corsika::radio::RadioTrackSegment>& tracks,
                          std::size_t maximum = 4096);
    std::uint64_t count() const;
    std::uint64_t consumed() const;
    bool finished() const;
  };
} // namespace corsika::kokkos::radio

#include <corsika/detail/kokkos/radio/RadioTrackSpoolReader.inl>
