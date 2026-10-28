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

  enum class LeptonTransportLimit : std::int32_t {
    InteractionCandidate = 0,
    ContinuousStep = 1,
    ParticleCut = 2,
    LayerBoundary = 3,
    ObservationSurface = 4,
    EscapedEnvironment = 5,
    MagneticStep = 6,
    DecayCandidate = 7,
    MaterialBoundary = 8,
  };

  /**
   * Value-only result of a charged-lepton step, including magnetic/scattering data.
   * Constructing it performs neither propagation nor sampling. Energies/deposits
   * are GeV, lengths are metres, grammage is g/cm^2 and angles are radians.
   * Continuous and cut deposits remain separate and unweighted; the tagged native
   * muon-pair cut can carry signed kinetic energy in a zero-length record.
   * observationSurfaceReachedBeforeCut_ retains simultaneous observation and cut.
   * Field order, types, defaults and enum values retain the production contract.
   * Native layout is not a portable serialized format.
   */
  struct LeptonTransportRecord {
    InteractionRecord interaction_{};
    ParticleState start_{};
    ParticleState end_{};
    std::uint64_t inputIndex_{};
    LeptonTransportLimit limit_{LeptonTransportLimit::ContinuousStep};
    std::int32_t startLayerIndex_{-1};
    std::int32_t endLayerIndex_{-1};
    double distanceM_{};
    double traversedGrammageGPerCm2_{};
    double continuousStepGrammageGPerCm2_{};
    double startDensityGPerCm3_{};
    double endDensityGPerCm3_{};
    double limitingRadiusM_{};
    double continuousDepositedEnergyGeV_{};
    double cutDepositedEnergyGeV_{};
    std::uint32_t observationSurfaceReachedBeforeCut_{};
    // Two 16-bit flags plus the iteration counter retain the former
    // eight-byte layout while exposing the dominant Moliere work per step.
    std::uint16_t multipleScatteringApplied_{};
    std::uint16_t multipleScatteringStatus_{};
    std::uint32_t multipleScatteringIterations_{};
    double multipleScatteringAngleRad_{};
    double multipleScatteringFirstUniform_{};
    double multipleScatteringSecondUniform_{};
    double multipleScatteringAzimuthUniform_{};
    std::uint32_t magneticBendingApplied_{};
    std::uint32_t magneticStepStatus_{};
    double magneticStepLimitM_{};
    double magneticGyroradiusM_{};
    double magneticBendParameter_{};
    double magneticChordLengthM_{};
  };
  static_assert(std::is_standard_layout_v<LeptonTransportRecord>);
  static_assert(std::is_trivially_copyable_v<LeptonTransportRecord>);

} // namespace corsika::kokkos
