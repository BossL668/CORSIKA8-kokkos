/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/accelerator/em/common/tables/PhysicsConstants.hpp>

#define CORSIKA_GPU_LPM_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em {

  inline constexpr std::size_t MaxPhotonPairLpmComponents = 16;

  struct PhotonPairLpmComponentSnapshot {
    std::uint64_t component_hash{};
    double nuclear_charge{};
    double radiation_log_constant{};
  };

  /**
   * Small, pointer-free copy of PROPOSAL::PhotoPairLPM state.
   *
   * All dimensional quantities use the names and units shown in the fields.
   * It is passed by value as a CUDA kernel argument.
   */
  struct PhotonPairLpmSnapshot {
    double baseline_mass_density_g_per_cm3{};
    double molecular_density_per_cm3{};
    double sum_charge{};
    double e_lpm_MeV{};
    double classical_electron_radius_cm{};
    double fine_structure_constant{};
    std::uint32_t component_count{};
    std::uint32_t reserved{};
    PhotonPairLpmComponentSnapshot
        components[MaxPhotonPairLpmComponents]{};
  };

  enum class PhotonPairLpmStatus : std::uint32_t {
    Success = 0,
    ComponentNotFound = 1,
    InvalidSnapshot = 2,
    InvalidInput = 3,
    NonFiniteResult = 4,
  };

  struct PhotonPairLpmResult {
    PhotonPairLpmStatus status{PhotonPairLpmStatus::InvalidSnapshot};
    std::uint32_t reserved{};
    double survival_probability{};
  };

  static_assert(std::is_standard_layout_v<PhotonPairLpmSnapshot>);
  static_assert(std::is_trivially_copyable_v<PhotonPairLpmSnapshot>);
  static_assert(std::is_standard_layout_v<PhotonPairLpmResult>);
  static_assert(std::is_trivially_copyable_v<PhotonPairLpmResult>);

  inline PhotonPairLpmSnapshot makePhotonPairLpmSnapshot(
      tables::PhotonPairLpmMetadata const& source) {
    if (source.components.empty() ||
        source.components.size() > MaxPhotonPairLpmComponents) {
      throw std::invalid_argument(
          "photon-pair LPM component count is unsupported");
    }
    PhotonPairLpmSnapshot result{};
    result.baseline_mass_density_g_per_cm3 =
        source.baseline_mass_density_g_per_cm3;
    result.molecular_density_per_cm3 =
        source.molecular_density_per_cm3;
    result.sum_charge = source.sum_charge;
    result.e_lpm_MeV = source.e_lpm_MeV;
    result.classical_electron_radius_cm =
        source.classical_electron_radius_cm;
    result.fine_structure_constant =
        source.fine_structure_constant;
    result.component_count =
        static_cast<std::uint32_t>(source.components.size());
    for (std::size_t index = 0; index < source.components.size();
         ++index) {
      auto const& component = source.components[index];
      result.components[index] = {
          component.proposal_hash, component.nuclear_charge,
          component.radiation_log_constant};
    }
    return result;
  }

  namespace detail {

    CORSIKA_GPU_LPM_HOST_DEVICE inline bool lpmFinite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_GPU_LPM_HOST_DEVICE inline double lpmSqrt(double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    CORSIKA_GPU_LPM_HOST_DEVICE inline double lpmLog(double value) {
#if defined(__CUDA_ARCH__)
      return ::log(value);
#else
      return std::log(value);
#endif
    }

    CORSIKA_GPU_LPM_HOST_DEVICE inline double lpmExp(double value) {
#if defined(__CUDA_ARCH__)
      return ::exp(value);
#else
      return std::exp(value);
#endif
    }

    CORSIKA_GPU_LPM_HOST_DEVICE inline double lpmPow(
        double base, double exponent) {
#if defined(__CUDA_ARCH__)
      return ::pow(base, exponent);
#else
      return std::pow(base, exponent);
#endif
    }

  } // namespace detail

  /**
   * Byte-for-formula translation of PROPOSAL 7.6.2
   * PhotoPairLPM::suppression_factor().
   *
   * energy_MeV is the incident photon total energy, x is the first secondary
   * electron's energy divided by the pair energy, and local density is the
   * CORSIKA vertex density. No fast-math approximation or probability clamp is
   * applied: CORSIKA accepts the interaction when uniform <= returned factor.
   */
  CORSIKA_GPU_LPM_HOST_DEVICE inline PhotonPairLpmResult
  photonPairLpmSuppressionFactor(
      PhotonPairLpmSnapshot const& snapshot,
      std::uint64_t component_hash, double energy_MeV, double x,
      double local_mass_density_g_per_cm3) {
    using namespace detail;
    if (snapshot.component_count == 0 ||
        snapshot.component_count > MaxPhotonPairLpmComponents ||
        !lpmFinite(snapshot.baseline_mass_density_g_per_cm3) ||
        !(snapshot.baseline_mass_density_g_per_cm3 > 0.) ||
        !lpmFinite(snapshot.molecular_density_per_cm3) ||
        !(snapshot.molecular_density_per_cm3 > 0.) ||
        !lpmFinite(snapshot.sum_charge) ||
        !(snapshot.sum_charge > 0.) ||
        !lpmFinite(snapshot.e_lpm_MeV) ||
        !(snapshot.e_lpm_MeV > 0.) ||
        !lpmFinite(snapshot.classical_electron_radius_cm) ||
        !(snapshot.classical_electron_radius_cm > 0.) ||
        !lpmFinite(snapshot.fine_structure_constant) ||
        !(snapshot.fine_structure_constant > 0.)) {
      return {PhotonPairLpmStatus::InvalidSnapshot, 0, 0.};
    }
    if (!lpmFinite(energy_MeV) || !(energy_MeV > 0.) ||
        !lpmFinite(x) || !(x > 0.) || !(x < 1.) ||
        !lpmFinite(local_mass_density_g_per_cm3) ||
        !(local_mass_density_g_per_cm3 > 0.)) {
      return {PhotonPairLpmStatus::InvalidInput, 0, 0.};
    }

    PhotonPairLpmComponentSnapshot component{};
    bool found = false;
    for (std::uint32_t index = 0;
         index < snapshot.component_count; ++index) {
      if (snapshot.components[index].component_hash ==
          component_hash) {
        component = snapshot.components[index];
        found = true;
        break;
      }
    }
    if (!found) {
      return {PhotonPairLpmStatus::ComponentNotFound, 0, 0.};
    }
    if (!lpmFinite(component.nuclear_charge) ||
        !(component.nuclear_charge > 0.) ||
        !lpmFinite(component.radiation_log_constant) ||
        !(component.radiation_log_constant > 0.)) {
      return {PhotonPairLpmStatus::InvalidSnapshot, 0, 0.};
    }

    constexpr double Pi =
        3.141592653589793238462643383279502884;
    constexpr double SqrtTwo =
        1.414213562373095048801688724209698079;
    constexpr double Fi1 = 1.54954;
    constexpr double G1 = 0.710390;
    constexpr double G2 = 0.904912;

    auto const density_correction =
        local_mass_density_g_per_cm3 /
        snapshot.baseline_mass_density_g_per_cm3;
    auto const z_to_minus_third =
        lpmPow(component.nuclear_charge, -1. / 3.);
    auto s1 =
        1. / (z_to_minus_third *
              component.radiation_log_constant);
    s1 = s1 * s1 * SqrtTwo;

    auto const sp =
        0.125 *
        lpmSqrt(snapshot.e_lpm_MeV /
                (density_correction * energy_MeV * x * (1. - x)));
    auto const h = lpmLog(sp) / lpmLog(s1);
    double xi = 1.;
    if (sp < s1) {
      xi = 2.;
    } else if (sp < 1.) {
      xi =
          1. + h -
          0.08 * (1. - h) *
              (1. - (1. - h) * (1. - h)) / lpmLog(s1);
    }

    auto gamma =
        snapshot.classical_electron_radius_cm /
        (snapshot.fine_structure_constant * x);
    gamma =
        1. +
        4. * Pi * snapshot.sum_charge *
            snapshot.classical_electron_radius_cm * gamma * gamma *
            snapshot.molecular_density_per_cm3 *
            density_correction;
    auto const s = sp / lpmSqrt(xi) * gamma;
    auto const s2 = s * s;

    double fi = 0.;
    if (s < Fi1) {
      fi =
          1. -
          lpmExp(-6. * s * (1. + (3. - Pi) * s) +
                 s2 * s /
                     (0.623 + 0.796 * s + 0.658 * s2));
    } else {
      fi = 1. - 0.012 / (s2 * s2);
    }

    double g = 0.;
    if (s < G1) {
      auto const psi =
          1. -
          lpmExp(
              -4. * s -
              8. * s2 /
                  (1. + 3.936 * s + 4.97 * s2 -
                   0.05 * s2 * s + 7.50 * s2 * s2));
      g = 3. * psi - 2. * fi;
    } else if (s < G2) {
      g = 36. * s2 / (36. * s2 + 1.);
    } else {
      g = 1. - 0.022 / (s2 * s2);
    }

    auto const probability =
        ((xi / 3.) *
         (g / (gamma * gamma) +
          2. * (x * x + (1. - x) * (1. - x)) * fi /
              gamma)) /
        (1. - (4. / 3.) * x * (1. - x));
    if (!lpmFinite(probability) || probability < 0.) {
      return {PhotonPairLpmStatus::NonFiniteResult, 0, 0.};
    }
    return {PhotonPairLpmStatus::Success, 0, probability};
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_LPM_HOST_DEVICE
