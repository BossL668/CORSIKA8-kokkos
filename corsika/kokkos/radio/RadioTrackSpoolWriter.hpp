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
  class RadioTrackSpoolWriter final : public corsika::radio::RadioTrackSink {
    std::ofstream file_;
    std::uint64_t count_{};
    bool completed_{};

  public:
    explicit RadioTrackSpoolWriter(std::filesystem::path const& path);
    void append(corsika::radio::RadioTrackSegment const& track) override;
    void complete();
    std::uint64_t count() const;
  };
} // namespace corsika::kokkos::radio

#include <corsika/detail/kokkos/radio/RadioTrackSpoolWriter.inl>
