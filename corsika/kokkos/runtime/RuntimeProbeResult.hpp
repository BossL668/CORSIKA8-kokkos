/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace corsika::kokkos {

  struct PrimitiveProbeResult {
    std::size_t values_{};
    std::uint64_t selected_{};
    std::uint64_t checksum_{};
    bool scanValid_{};
  };

  struct QueueProbeResult {
    std::size_t inputParticles_{};
    std::size_t retainedParticles_{};
    std::uint64_t secondarySlots_{};
    bool stableOrder_{};
    bool roundtripExact_{};
  };

  struct TilingProbeResult {
    std::size_t tracks_{};
    std::size_t observers_{};
    std::uint64_t checksum_{};
    bool exact_{};
  };

} // namespace corsika::kokkos
