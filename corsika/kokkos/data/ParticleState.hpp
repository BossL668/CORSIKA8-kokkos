/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::kokkos {

  /**
   * Value-only particle data at the host/device transport boundary.
   *
   * Positions and directions use the common coordinate system selected by the
   * conversion boundary. This record stores neither coordinate-system objects nor
   * host pointers, and performs no conversion, normalization or physics operation.
   * Public members deliberately retain aggregate initialization and direct device
   * access. The type is standard-layout and trivially copyable, but its padding and
   * native byte order are not a portable serialization format.
   */
  struct alignas(16) ParticleState {
    std::int32_t pid_{};      ///< Signed PDG identifier, not CORSIKA's Code enum.
    std::int32_t mediumId_{}; ///< Medium/model identifier, not an environment node.
    std::uint32_t generation_{};
    std::uint32_t reserved_{}; ///< Preserved compatibility tags; not unused storage.

    double energyGeV_{};    ///< Total energy, including rest mass, in GeV.
    double positionM_[3]{}; ///< Position in metres in the common coordinate system.
    double direction_[3]{}; ///< Dimensionless direction in that coordinate system.
    double timeS_{};        ///< Time in seconds.
    double weight_{1.};     ///< Statistical particle weight.

    std::uint64_t historyId_{};       ///< Stable particle history; zero is unassigned.
    std::uint64_t parentHistoryId_{}; ///< Zero denotes no parent history.
    std::uint64_t stepId_{};          ///< Step identifier within this history.
    /** Logical tree node, independent of mediumId_. Distinct nodes can share a
     * medium. Minus three denotes legacy/unassigned; minus one denotes outside
     * the exported tree. Copying the record does not assign a node or history.
     */
    std::int32_t environmentNode_{-3};
  };

} // namespace corsika::kokkos
