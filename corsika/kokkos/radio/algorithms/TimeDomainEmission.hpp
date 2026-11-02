/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/ScalarPhysicalConstants.hpp>
#include <corsika/kokkos/radio/RadioProjectionData.hpp>
#include <corsika/framework/core/HostDevice.hpp>

namespace corsika::kokkos::radio::time_domain_detail {
  /// Positive-frequency polarization transfer, including spreading and attenuation.
  template <typename TAmplitude>
  struct TimeDomainRay {
    TAmplitude projectedDisplacement_[3]{};
    double arrivalTimeS_{};
    double retardedDurationS_{};
    int topology_{};
  };

  /** Deposit a ZHS rectangular vector potential, averaged over each sample bin.
   * The receiver accepts (bin, three complex components, topology). The imaginary
   * quadrature retains total-internal-reflection and caustic phase; it must not
   * be discarded before forming the positive-frequency spectrum.
   */
  template <typename TAmplitude, typename TReceiver>
  CORSIKA_HOST_DEVICE inline void deposit_zhs(TimeDomainRay<TAmplitude> const& ray,
                                              double chargeWeight, double intervalS,
                                              std::size_t sampleCount,
                                              TReceiver const& receive);

  /** Deposit the CoREAS start/end fields for one consistently identified ray family.
   * Near the Cherenkov cone, use the midpoint ZHS-like approximation from CoREAS.
   * Endpoint impulses use linear, moment-preserving deposition onto the time grid.
   * Times are relative to the output window. The caller matches branches by topology
   * and subdivides tracks before calling this function.
   * The dense-medium default bounds |1 - n beta.k| below by 0.1 for endpoint
   * evaluation (T-510, Phys. Rev. D 105, 063025, Sec. III.B.2). This is a
   * numerical fallback criterion, not a cut on contributing tracks.
   */
  template <typename TAmplitude, typename TReceiver>
  CORSIKA_HOST_DEVICE inline void deposit_coreas(
      TimeDomainRay<TAmplitude> const& start, TimeDomainRay<TAmplitude> const& end,
      TimeDomainRay<TAmplitude> const& middle, double durationS, double chargeWeight,
      double intervalS, std::size_t sampleCount, TReceiver const& receive,
      double approximationThreshold = .1);
} // namespace corsika::kokkos::radio::time_domain_detail

#include <corsika/detail/kokkos/radio/algorithms/TimeDomainEmission.inl>
