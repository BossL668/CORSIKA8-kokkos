/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/InteractionRecord.hpp>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  enum class PhotonTransportLimit : std::int32_t {
    Interaction = 0,
    LayerBoundary = 1,
    ObservationSurface = 2,
    EscapedEnvironment = 3,
    ParticleCut = 4,
    MaterialBoundary = 5,
  };

  /**
   * Value-only result of a photon step; constructing it performs no transport.
   * Energies/deposits are GeV, lengths are metres and grammage is g/cm^2.
   * The interaction stores the vertex state only on the original interaction
   * branch. Layer/node indices and physical medium IDs remain distinct.
   * observationSurfaceReachedBeforeCut_ retains simultaneous observation and cut.
   * Field order, types, defaults and enum values retain the production contract.
   * Native layout is not a portable serialized format.
   */
  struct PhotonTransportRecord {
    InteractionRecord interaction_{};
    ParticleState start_{};
    ParticleState end_{};
    std::uint64_t inputIndex_{};
    PhotonTransportLimit limit_{PhotonTransportLimit::LayerBoundary};
    std::int32_t startLayerIndex_{-1};
    std::int32_t endLayerIndex_{-1};
    double distanceM_{};
    double traversedGrammageGPerCm2_{};
    double startDensityGPerCm3_{};
    double endDensityGPerCm3_{};
    double limitingRadiusM_{};
    double cutDepositedEnergyGeV_{};
    // ObservationPlane precedes ParticleCut in c8_air_shower's scalar
    // process sequence.  A step that reaches the plane after 10 ms must
    // therefore both publish the observation and terminate at ParticleCut.
    std::uint32_t observationSurfaceReachedBeforeCut_{};
  };
  static_assert(std::is_standard_layout_v<PhotonTransportRecord>);
  static_assert(std::is_trivially_copyable_v<PhotonTransportRecord>);

} // namespace corsika::kokkos
