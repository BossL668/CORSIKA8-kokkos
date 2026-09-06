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

#define CORSIKA_GPU_BREMS_LPM_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em {

  inline constexpr std::size_t MaxBremsLpmComponents = 16;

  struct BremsLpmComponentSnapshot {
    std::uint64_t component_hash{};
    double nuclear_charge{};
    double atomic_mass_number{};
    double radiation_log_constant{};
  };

  /**
   * Pointer-free copy of the PROPOSAL 7.6.2 BremsLPM state.
   *
   * Energies/masses are MeV, densities use the PROPOSAL cm/g convention
   * encoded in their field names, and the object is passed by value to CUDA
   * kernels.
   */
  struct BremsLpmSnapshot {
    double baseline_mass_density_g_per_cm3{};
    double molecular_density_per_cm3{};
    double sum_charge{};
    double e_lpm_MeV{};
    double lepton_mass_MeV{};
    double electron_mass_MeV{};
    double muon_mass_MeV{};
    double classical_electron_radius_cm{};
    double fine_structure_constant{};
    std::uint32_t component_count{};
    std::uint32_t reserved{};
    BremsLpmComponentSnapshot components[MaxBremsLpmComponents]{};
  };

  /**
   * Device-evaluated, immutable component terms used by every
   * bremsstrahlung LPM query in one shower.
   *
   * These values are prepared once with CUDA's double-precision pow/log
   * implementation so the production kernel retains the scalar device
   * formula's exact constants without recomputing them per vertex.
   */
  struct BremsLpmPreparedComponent {
    std::uint64_t component_hash{};
    double s1{};
    double logarithm_s1{};
  };

  struct BremsLpmPreparedSnapshot {
    std::uint32_t valid{};
    std::uint32_t component_count{};
    BremsLpmPreparedComponent
        components[MaxBremsLpmComponents]{};
  };

  enum class BremsLpmStatus : std::uint32_t {
    Success = 0,
    ComponentNotFound = 1,
    InvalidSnapshot = 2,
    InvalidInput = 3,
    NonFiniteResult = 4,
  };

  struct BremsLpmResult {
    BremsLpmStatus status{BremsLpmStatus::InvalidSnapshot};
    std::uint32_t reserved{};
    double survival_probability{};
  };

  static_assert(std::is_standard_layout_v<BremsLpmSnapshot>);
  static_assert(std::is_trivially_copyable_v<BremsLpmSnapshot>);
  static_assert(
      std::is_standard_layout_v<BremsLpmPreparedSnapshot>);
  static_assert(
      std::is_trivially_copyable_v<
          BremsLpmPreparedSnapshot>);
  static_assert(std::is_standard_layout_v<BremsLpmResult>);
  static_assert(std::is_trivially_copyable_v<BremsLpmResult>);

  inline BremsLpmSnapshot makeBremsLpmSnapshot(
      tables::BremsLpmMetadata const& source) {
    if (source.components.empty() ||
        source.components.size() > MaxBremsLpmComponents) {
      throw std::invalid_argument(
          "bremsstrahlung LPM component count is unsupported");
    }
    BremsLpmSnapshot result{};
    result.baseline_mass_density_g_per_cm3 =
        source.baseline_mass_density_g_per_cm3;
    result.molecular_density_per_cm3 =
        source.molecular_density_per_cm3;
    result.sum_charge = source.sum_charge;
    result.e_lpm_MeV = source.e_lpm_MeV;
    result.lepton_mass_MeV = source.lepton_mass_MeV;
    result.electron_mass_MeV = source.electron_mass_MeV;
    result.muon_mass_MeV = source.muon_mass_MeV;
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
          component.atomic_mass_number,
          component.radiation_log_constant};
    }
    return result;
  }

  namespace brems_lpm_detail {

    CORSIKA_GPU_BREMS_LPM_HOST_DEVICE inline bool finite(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_GPU_BREMS_LPM_HOST_DEVICE inline double squareRoot(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    CORSIKA_GPU_BREMS_LPM_HOST_DEVICE inline double logarithm(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::log(value);
#else
      return std::log(value);
#endif
    }

    CORSIKA_GPU_BREMS_LPM_HOST_DEVICE inline double exponential(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::exp(value);
#else
      return std::exp(value);
#endif
    }

    CORSIKA_GPU_BREMS_LPM_HOST_DEVICE inline double power(
        double base, double exponent) {
#if defined(__CUDA_ARCH__)
      return ::pow(base, exponent);
#else
      return std::pow(base, exponent);
#endif
    }

  } // namespace brems_lpm_detail

  /**
   * Direct translation of PROPOSAL 7.6.2
   * BremsLPM::suppression_factor().
   *
   * energy_MeV is the incident lepton total energy, v is the sampled
   * fractional stochastic energy loss, and local density is evaluated at the
   * interaction vertex. No probability clamp or fast-math approximation is
   * applied; CORSIKA accepts the interaction when uniform <= factor.
   */
  CORSIKA_GPU_BREMS_LPM_HOST_DEVICE inline BremsLpmResult
  bremsLpmSuppressionFactor(
      BremsLpmSnapshot const& snapshot,
      std::uint64_t component_hash, double energy_MeV, double v,
      double local_mass_density_g_per_cm3) {
    using namespace brems_lpm_detail;
    if (snapshot.component_count == 0 ||
        snapshot.component_count > MaxBremsLpmComponents ||
        !finite(snapshot.baseline_mass_density_g_per_cm3) ||
        !(snapshot.baseline_mass_density_g_per_cm3 > 0.) ||
        !finite(snapshot.molecular_density_per_cm3) ||
        !(snapshot.molecular_density_per_cm3 > 0.) ||
        !finite(snapshot.sum_charge) ||
        !(snapshot.sum_charge > 0.) ||
        !finite(snapshot.e_lpm_MeV) ||
        !(snapshot.e_lpm_MeV > 0.) ||
        !finite(snapshot.lepton_mass_MeV) ||
        !(snapshot.lepton_mass_MeV > 0.) ||
        !finite(snapshot.electron_mass_MeV) ||
        !(snapshot.electron_mass_MeV > 0.) ||
        !finite(snapshot.muon_mass_MeV) ||
        !(snapshot.muon_mass_MeV > 0.) ||
        !finite(snapshot.classical_electron_radius_cm) ||
        !(snapshot.classical_electron_radius_cm > 0.) ||
        !finite(snapshot.fine_structure_constant) ||
        !(snapshot.fine_structure_constant > 0.)) {
      return {BremsLpmStatus::InvalidSnapshot, 0, 0.};
    }
    if (!finite(energy_MeV) ||
        !(energy_MeV > snapshot.lepton_mass_MeV) ||
        !finite(v) || !(v > 0.) || !(v < 1.) ||
        !finite(local_mass_density_g_per_cm3) ||
        !(local_mass_density_g_per_cm3 > 0.)) {
      return {BremsLpmStatus::InvalidInput, 0, 0.};
    }

    BremsLpmComponentSnapshot component{};
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
      return {BremsLpmStatus::ComponentNotFound, 0, 0.};
    }
    if (!finite(component.nuclear_charge) ||
        !(component.nuclear_charge > 0.) ||
        !finite(component.atomic_mass_number) ||
        !(component.atomic_mass_number > 0.) ||
        !finite(component.radiation_log_constant) ||
        !(component.radiation_log_constant > 0.)) {
      return {BremsLpmStatus::InvalidSnapshot, 0, 0.};
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
        power(component.nuclear_charge, -1. / 3.);
    auto const d_n =
        1.54 * power(component.atomic_mass_number, 0.27);
    auto const nuclear_auxiliary =
        snapshot.lepton_mass_MeV / snapshot.muon_mass_MeV *
        d_n;
    auto s1 =
        snapshot.electron_mass_MeV /
        (snapshot.lepton_mass_MeV * z_to_minus_third *
         component.radiation_log_constant);
    s1 =
        s1 * s1 *
        (1. + nuclear_auxiliary * nuclear_auxiliary) *
        SqrtTwo;

    auto const sp =
        0.125 *
        squareRoot(
            snapshot.e_lpm_MeV * v /
            (density_correction * energy_MeV * (1. - v)));
    auto const h = logarithm(sp) / logarithm(s1);
    double xi = 1.;
    if (sp < s1) {
      xi = 2.;
    } else if (sp < 1.) {
      xi =
          1. + h -
          0.08 * (1. - h) *
              (1. - (1. - h) * (1. - h)) /
              logarithm(s1);
    }

    auto gamma =
        snapshot.classical_electron_radius_cm *
        snapshot.electron_mass_MeV /
        (snapshot.fine_structure_constant *
         snapshot.lepton_mass_MeV * v);
    gamma =
        1. +
        4. * Pi * snapshot.sum_charge *
            snapshot.classical_electron_radius_cm *
            gamma * gamma *
            snapshot.molecular_density_per_cm3 *
            density_correction;
    auto const s = sp / squareRoot(xi) * gamma;
    auto const s2 = s * s;

    double fi = 0.;
    if (s < Fi1) {
      fi =
          1. -
          exponential(
              -6. * s * (1. + (3. - Pi) * s) +
              s2 * s /
                  (0.623 + 0.796 * s + 0.658 * s2));
    } else {
      fi = 1. - 0.012 / (s2 * s2);
    }

    double g = 0.;
    if (s < G1) {
      auto const psi =
          1. -
          exponential(
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
         (v * v * g / (gamma * gamma) +
          2. * (1. + (1. - v) * (1. - v)) * fi /
              gamma)) /
        ((4. / 3.) * (1. - v) + v * v);
    if (!finite(probability) || probability < 0.) {
      return {BremsLpmStatus::NonFiniteResult, 0, 0.};
    }
    return {BremsLpmStatus::Success, 0, probability};
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_BREMS_LPM_HOST_DEVICE
