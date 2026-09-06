/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/accelerator/em/common/BremsLpm.hpp>

#define CORSIKA_GPU_EPAIR_LPM_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em {

  enum class EpairLpmStatus : std::uint32_t {
    Success = 0,
    InvalidSnapshot = 1,
    InvalidInput = 2,
    NonFiniteResult = 3,
  };

  struct EpairLpmResult {
    EpairLpmStatus status{EpairLpmStatus::InvalidSnapshot};
    std::uint32_t reserved{};
    double survival_probability{};
  };

  static_assert(std::is_standard_layout_v<EpairLpmResult>);
  static_assert(std::is_trivially_copyable_v<EpairLpmResult>);

  namespace epair_lpm_detail {

    CORSIKA_GPU_EPAIR_LPM_HOST_DEVICE inline bool finite(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_GPU_EPAIR_LPM_HOST_DEVICE inline double squareRoot(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    CORSIKA_GPU_EPAIR_LPM_HOST_DEVICE inline double logarithm(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::log(value);
#else
      return std::log(value);
#endif
    }

    CORSIKA_GPU_EPAIR_LPM_HOST_DEVICE inline double arcTangent(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::atan(value);
#else
      return std::atan(value);
#endif
    }

    CORSIKA_GPU_EPAIR_LPM_HOST_DEVICE inline double power(
        double base, double exponent) {
#if defined(__CUDA_ARCH__)
      return ::pow(base, exponent);
#else
      return std::pow(base, exponent);
#endif
    }

  } // namespace epair_lpm_detail

  /**
   * Direct translation of PROPOSAL 7.6.2
   * EpairLPM::suppression_factor().
   *
   * EpairLPM's private characteristic energy is reconstructed from the same
   * versioned medium/component constants already carried by BremsLpmSnapshot,
   * avoiding another cache-format field that would duplicate its inputs.
   */
  CORSIKA_GPU_EPAIR_LPM_HOST_DEVICE inline EpairLpmResult
  epairLpmSuppressionFactor(
      BremsLpmSnapshot const& snapshot, double energy_MeV,
      double v, double rho_squared,
      double local_mass_density_g_per_cm3) {
    using namespace epair_lpm_detail;
    if (snapshot.component_count == 0 ||
        snapshot.component_count > MaxBremsLpmComponents ||
        !finite(snapshot.baseline_mass_density_g_per_cm3) ||
        !(snapshot.baseline_mass_density_g_per_cm3 > 0.) ||
        !finite(snapshot.molecular_density_per_cm3) ||
        !(snapshot.molecular_density_per_cm3 > 0.) ||
        !finite(snapshot.lepton_mass_MeV) ||
        !(snapshot.lepton_mass_MeV > 0.) ||
        !finite(snapshot.electron_mass_MeV) ||
        !(snapshot.electron_mass_MeV > 0.) ||
        !finite(snapshot.classical_electron_radius_cm) ||
        !(snapshot.classical_electron_radius_cm > 0.) ||
        !finite(snapshot.fine_structure_constant) ||
        !(snapshot.fine_structure_constant > 0.)) {
      return {EpairLpmStatus::InvalidSnapshot, 0, 0.};
    }
    if (!finite(energy_MeV) ||
        !(energy_MeV > snapshot.lepton_mass_MeV) ||
        !finite(v) || !(v > 0.) || !(v < 1.) ||
        !finite(rho_squared) || !(rho_squared >= 0.) ||
        !(rho_squared < 1.) ||
        !finite(local_mass_density_g_per_cm3) ||
        !(local_mass_density_g_per_cm3 > 0.)) {
      return {EpairLpmStatus::InvalidInput, 0, 0.};
    }

    double sum = 0.;
    for (std::uint32_t index = 0;
         index < snapshot.component_count; ++index) {
      auto const& component = snapshot.components[index];
      if (!finite(component.nuclear_charge) ||
          !(component.nuclear_charge > 0.) ||
          !finite(component.radiation_log_constant) ||
          !(component.radiation_log_constant > 0.)) {
        return {EpairLpmStatus::InvalidSnapshot, 0, 0.};
      }
      sum +=
          component.nuclear_charge *
          component.nuclear_charge *
          logarithm(
              3.25 * component.radiation_log_constant *
              power(component.nuclear_charge, -1. / 3.));
    }
    if (!finite(sum) || !(sum > 0.)) {
      return {EpairLpmStatus::InvalidSnapshot, 0, 0.};
    }

    constexpr double Pi =
        3.141592653589793238462643383279502884;
    constexpr double ComputerPrecision = 1.e-10;
    auto characteristic_energy =
        snapshot.lepton_mass_MeV /
        (snapshot.electron_mass_MeV *
         snapshot.classical_electron_radius_cm);
    characteristic_energy *=
        characteristic_energy * characteristic_energy *
        snapshot.fine_structure_constant *
        snapshot.lepton_mass_MeV /
        (2. * Pi * snapshot.molecular_density_per_cm3 *
         sum);
    if (!finite(characteristic_energy) ||
        !(characteristic_energy > 0.)) {
      return {EpairLpmStatus::InvalidSnapshot, 0, 0.};
    }

    auto const density_correction =
        local_mass_density_g_per_cm3 /
        snapshot.baseline_mass_density_g_per_cm3;
    auto const beta = v * v / (2. * (1. - v));
    auto const xi =
        snapshot.lepton_mass_MeV *
        snapshot.lepton_mass_MeV /
        (snapshot.electron_mass_MeV *
         snapshot.electron_mass_MeV) *
        v * v / 4. * (1. - rho_squared) / (1. - v);
    auto const s =
        0.25 *
        squareRoot(
            characteristic_energy /
            (density_correction * energy_MeV * v *
             (1. - rho_squared)));
    if (!finite(beta) || !finite(xi) || !(xi > 0.) ||
        !finite(s) || !(s > 0.)) {
      return {EpairLpmStatus::InvalidInput, 0, 0.};
    }

    auto const s6 = 6. * s;
    auto atan_argument = s6 * (xi + 1.);
    if (atan_argument > 1. / ComputerPrecision) {
      return {EpairLpmStatus::Success, 0, 1.};
    }
    auto const s36 = 36. * s * s;
    auto const d1 = s6 / (s6 + 1.);
    auto const d2 = s36 / (s36 + 1.);
    auto const shifted_atan =
        arcTangent(atan_argument) - Pi / 2.;
    auto const log1 =
        logarithm(
            (s36 * (1. + xi) * (1. + xi) + 1.) /
            (s36 * xi * xi));
    auto const log2 =
        logarithm(
            (s6 * (1. + xi) + 1.) / (s6 * xi));
    auto const a =
        0.5 * d2 * (1. + 2. * d2 * xi) * log1 - d2 +
        6. * d2 * s *
            (1. + (s36 - 1.) / (s36 + 1.) * xi) *
            shifted_atan;
    auto const b =
        d1 * (1. + d1 * xi) * log2 - d1;
    auto const c =
        -d2 * d2 * xi * log1 + d2 -
        d2 * d2 * (s36 - 1.) / (6. * s) * xi *
            shifted_atan;
    auto const d = d1 - d1 * d1 * xi * log2;
    auto const e = -s6 * shifted_atan;
    auto const denominator =
        ((2. + rho_squared) * (1. + beta) +
         xi * (3. + rho_squared)) *
            logarithm(1. + 1. / xi) +
        (1. - rho_squared - beta) / (1. + xi) -
        (3. + rho_squared);
    auto const probability =
        ((1. + beta) *
             (a + (1. + rho_squared) * b) +
         beta * (c + (1. + rho_squared) * d) +
         (1. - rho_squared) * e) /
        denominator;
    if (!finite(probability) || probability < 0.) {
      return {EpairLpmStatus::NonFiniteResult, 0, 0.};
    }
    // The closed form suffers cancellation in the unsuppressed limit and
    // PROPOSAL can return values a few parts in 1e7 above one. Since every
    // admissible open-interval uniform accepts that branch, canonicalize the
    // numerically equivalent neighbourhood on both host and device.
    if (probability > 1. - 1.e-6) {
      return {EpairLpmStatus::Success, 0, 1.};
    }
    return {EpairLpmStatus::Success, 0, probability};
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_EPAIR_LPM_HOST_DEVICE
