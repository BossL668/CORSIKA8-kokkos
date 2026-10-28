/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/data/ParticleState.hpp>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {
  /** Unthinned physical children of the generation-zero projectile.
   * Output writers observe this snapshot before thinning, while transport queues
   * receive only the retained children. The three slots cover the production EM
   * final states; secondaryCount_ specifies valid entries. No queue is owned.
   */
  // Units match ParticleState (GeV, metres, seconds); weights are dimensionless.
  // Preserve field order, widths and defaults. Native layout is not a wire format.
  struct FirstInteractionRecord {
    ParticleState parentAtVertex_{};
    ParticleState secondaries_[3]{};
    std::int32_t processId_{};
    std::uint32_t secondaryCount_{};
  };
  static_assert(std::is_standard_layout_v<FirstInteractionRecord>);
  static_assert(std::is_trivially_copyable_v<FirstInteractionRecord>);
} // namespace corsika::kokkos
