/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/ScalarPhysicalConstants.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos::radio {
  /** Borrowed numerical views for projection. All pointer storage must be
   * accessible in the calling execution space and remain alive until it finishes.
   * Table observers must lie within its valid height bins. Positions are metres
   * in one common frame, times seconds, fields V/m and potentials V s/m. */

  inline constexpr double SpeedOfLightMPerS =
      ::corsika::kokkos::scalar_constants::SpeedOfLightMPerS;
  inline constexpr double VacuumPermittivityFPerM =
      ::corsika::kokkos::scalar_constants::VacuumPermittivityFPerM;
  // Keep the scalar CORSIKA constant_ exactly; do not substitute a newer
  // CODATA rounding here because radio replay uses the scalar result as its
  // oracle.
  inline constexpr double ElementaryChargeC =
      ::corsika::kokkos::scalar_constants::ElementaryChargeC;
  inline constexpr double Pi = 3.141592653589793238462643383279502884;
  inline constexpr double EmConstant =
      1. / (4. * Pi * VacuumPermittivityFPerM * SpeedOfLightMPerS);
  inline constexpr double CoREASApproximationThreshold = 1.e-3;
  inline constexpr double FixedPointHeadroom = 0x1p62;
  inline constexpr double SignedIntegerLimit = 0x1p63;
  inline constexpr long long SignedIntegerMaximum = 9223372036854775807LL;
  inline constexpr long long SignedIntegerMinimum = (-9223372036854775807LL - 1LL);

  struct RadioObserverView {
    double positionM_[3]{};
    double startTimeS_{};
    double durationS_{};
    double sampleRateHz_{};
    double fixedPointScale_{};
    double inverseFixedPointScale_{};
    std::uint64_t numberOfBins_{};
    std::uint64_t waveformOffset_{};
  };

  struct RadioPropagationView {
    double homogeneousRefractiveIndex_{};
    double minimumHeightM_{};
    double maximumHeightM_{};
    double stepM_{};
    double inverseStepPerM_{};
    double slopeRefractivityLower_{};
    double slopeIntegratedRefractivityLower_{};
    double slopeRefractivityUpper_{};
    double slopeIntegratedRefractivityUpper_{};
    double const* refractivity_{};
    double const* integratedRefractivity_{};
    std::size_t tableSize_{};
    std::uint32_t zhsSubtrackRefinement_{1};
  };

  struct RadioWaveformView {
    double* floatingX_{};
    double* floatingY_{};
    double* floatingZ_{};
    long long* fixedX_{};
    long long* fixedY_{};
    long long* fixedZ_{};
  };

  struct RadioCounters {
    unsigned long long coreasContributions_{};
    unsigned long long zhsContributions_{};
    unsigned long long zhsSubtracks_{};
    unsigned long long validTracks_{};
    unsigned long long fixedPointOverflows_{};
    double weightedSegmentCount_{};
    double trackLengthM_{};
    double weightedTrackLengthM_{};
    double electronWeightedTrackLengthM_{};
    double positronWeightedTrackLengthM_{};
    double signedChargeWeightedTrackLengthM_{};
    double energyWeightedTrackLengthGeVM_{};
    double maximumSegmentLengthM_{};
    double weightedDirectionChangeRad_{};
    double weightedDirectionChangeSquaredRad2_{};
    double weightedBetaDeficitTrackLengthM_{};
    double weightedTimeResidualS_{};
    double maximumDirectionChangeRad_{};
    double signedChargeWeightedDirectionChange_[3]{};
    double weightedTrackLengthByKineticEnergyM_[15]{};
  };

  struct RadioVector {
    double x_{};
    double y_{};
    double z_{};
  };

  struct RadioSignalPath {
    double propagationTimeS_{};
    double refractiveIndexSource_{};
    double refractiveIndexDestination_{};
    double distanceM_{};
    RadioVector emit_{};
  };

  struct RadioTrackKinematics {
    RadioVector start_{};
    RadioVector end_{};
    RadioVector displacement_{};
    RadioVector beta_{};
    double startTimeS_{};
    double endTimeS_{};
    double durationS_{};
    double trackLengthM_{};
    double betaModule_{};
    double constant_{};
    std::uint32_t valid_{};
  };

  static_assert(std::is_standard_layout_v<RadioObserverView>);
  static_assert(std::is_trivially_copyable_v<RadioObserverView>);
  static_assert(std::is_standard_layout_v<RadioPropagationView>);
  static_assert(std::is_trivially_copyable_v<RadioPropagationView>);
  static_assert(std::is_standard_layout_v<RadioWaveformView>);
  static_assert(std::is_trivially_copyable_v<RadioWaveformView>);
  static_assert(std::is_standard_layout_v<RadioCounters>);
  static_assert(std::is_trivially_copyable_v<RadioCounters>);
  static_assert(std::is_standard_layout_v<RadioVector>);
  static_assert(std::is_trivially_copyable_v<RadioVector>);
  static_assert(std::is_standard_layout_v<RadioSignalPath>);
  static_assert(std::is_trivially_copyable_v<RadioSignalPath>);
  static_assert(std::is_standard_layout_v<RadioTrackKinematics>);
  static_assert(std::is_trivially_copyable_v<RadioTrackKinematics>);
} // namespace corsika::kokkos::radio
