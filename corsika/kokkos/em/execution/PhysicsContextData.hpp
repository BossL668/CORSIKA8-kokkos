/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/algorithms/EmThinning.hpp>
#include <corsika/kokkos/em/algorithms/InteractionPhysicsView.hpp>
#include <corsika/kokkos/em/models/BremsLpm.hpp>
#include <corsika/kokkos/em/models/MoliereSnapshot.hpp>
#include <corsika/kokkos/em/models/PhotonPairLpm.hpp>
#include <corsika/kokkos/em/tables/MoliereInterpolation.hpp>
#include <corsika/kokkos/environment/EnvironmentBorrowedView.hpp>
#include <corsika/kokkos/environment/TransportGeometryConfig.hpp>
#include <corsika/kokkos/output/ProfileProjectionView.hpp>

#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  /** Shared resident-kernel inputs plus the keyed RNG and thinning state.
   * All referenced tables, environment arrays and profile support must remain
   * device-accessible and alive until the caller finishes using this context.
   * The record contains numerical values and borrowed pointers, never native C8
   * environment objects or managed allocation handles. Geometry configuration
   * retains the independent observation plane and magnetic deflection limit.
   */
  struct PhysicsContextData {
    InteractionPhysicsView physics_{};
    EnvironmentBorrowedView environment_{};
    TransportGeometryConfig geometry_{};
    MoliereSnapshot electronMoliere_{};
    MoliereSnapshot muonMoliere_{};
    MoliereInterpolationView moliereInterpolation_{};
    PhotonPairLpmSnapshot photonPairLpm_{};
    BremsLpmSnapshot bremsLpm_{};
    EmThinningConfig thinning_{};
    ProfileProjectionView profileProjection_{};
    std::uint64_t randomSeed_{};
    std::uint64_t showerId_{};
    std::uint32_t muonMoliereAvailable_{};
  };
  static_assert(std::is_standard_layout_v<PhysicsContextData>);
  static_assert(std::is_trivially_copyable_v<PhysicsContextData>);

} // namespace corsika::kokkos
