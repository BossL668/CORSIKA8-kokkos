/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/FrequencyGrid.hpp>
#include <cmath>
#include <stdexcept>

namespace corsika::kokkos::radio::firn {
  /// The source calculation is independent of the selected propagation model.
  enum class EmissionFormalism { FiniteTrackSpectrum, ZHS, CoREAS };

  /** Fourier window and the archived finite-track phase-subdivision tolerances. */
  struct OnlineRadioConfig {
    FrequencyGrid grid_{};
    EmissionFormalism emissionFormalism_{EmissionFormalism::FiniteTrackSpectrum};
    double timeOriginS_{0.0};
    double maximumPhaseCycles_{0.01};
    double maximumLinearizationLengthM_{0.25};
    double maximumRefractiveIndex_{1.78};
    std::size_t maximumSubdivisions_{100000};
    std::size_t maximumResponseQueries_{500000000};
    bool writeReplayTape_{false};
    bool writeBranchComponents_{false};
    bool includeSurfaceReflection_{true};
    double maximumPointCurrentPhaseIndicator_{0.02};
    /// Use the finite-track limit when |1 - n beta.k| is below this value.
    double coreasApproximationThreshold_{.1};

    void validate() const;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/OnlineRadioConfig.inl>
