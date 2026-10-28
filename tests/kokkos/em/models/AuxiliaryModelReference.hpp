/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Independent fixed-Git oracle, 41cdc543. Only dependencies, namespace and
// host/device annotations are adapted. Formula bodies retain production order.
#include <corsika/framework/core/HostDevice.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace corsika::kokkos::test::auxiliary_reference {
  namespace tables {

    // Matches scalar ContinuousProcess stopping just below ParticleCut.
    inline constexpr double ContinuousCutSafetyFactor = 0.9999;

    /**
     * PROPOSAL PhotoPairLPM parameters frozen into the physics cache.
     *
     * These values are deliberately stored instead of recomputed in CUDA code:
     * e_lpm_MeV depends on the exact PROPOSAL parametrization and a numerical
     * integral over the complete medium. The component data are the two target
     * quantities used by PhotoPairLPM::suppression_factor().
     */
    struct PhotonPairLpmComponent {
      std::uint64_t proposal_hash{};
      double nuclear_charge{};
      double radiation_log_constant{};
    };

    struct PhotonPairLpmMetadata {
      double baseline_mass_density_g_per_cm3{};
      double molecular_density_per_cm3{};
      double sum_charge{};
      double e_lpm_MeV{};
      double classical_electron_radius_cm{};
      double fine_structure_constant{};
      std::vector<PhotonPairLpmComponent> components;
    };

    /**
     * PROPOSAL BremsLPM parameters frozen into the physics cache.
     *
     * BremsLPM::eLpm_ is obtained from a numerical integral over the complete
     * medium and has no public accessor in PROPOSAL 7.6.2.  The remaining
     * fields are kept explicitly so the CUDA implementation is a direct
     * translation of the reference formula rather than a fit.
     */
    struct BremsLpmComponent {
      std::uint64_t proposal_hash{};
      double nuclear_charge{};
      // PROPOSAL::Component::GetAtomicNum(); despite its historical name this
      // is the target atomic mass used in D_n = 1.54 A^0.27.
      double atomic_mass_number{};
      double radiation_log_constant{};
    };

    struct BremsLpmMetadata {
      double baseline_mass_density_g_per_cm3{};
      double molecular_density_per_cm3{};
      double sum_charge{};
      double e_lpm_MeV{};
      double lepton_mass_MeV{};
      double electron_mass_MeV{};
      double muon_mass_MeV{};
      double classical_electron_radius_cm{};
      double fine_structure_constant{};
      std::vector<BremsLpmComponent> components;
    };

    struct MoliereComponentMetadata {
      std::uint64_t proposal_hash{};
      double nuclear_charge{};
      double atomic_mass_number{};
      double atoms_in_molecule{};
    };

    /**
     * Versioned numerical state for PROPOSAL 7.6.2 Moliere scattering.
     *
     * Synthetic/unit-test tables may set enabled=false and leave all remaining
     * fields empty. Production CUDA capability checks must reject such a table
     * before selecting charged transport with multiple scattering.
     */
    struct MoliereMetadata {
      bool enabled{};
      std::string reference_mode;
      double particle_mass_MeV{};
      double electron_mass_MeV{};
      double fine_structure_constant{};
      double avogadro_per_mol{};
      double hbar_MeV_s{};
      double speed_of_light_cm_per_s{};
      double euler_mascheroni{};
      std::vector<MoliereComponentMetadata> components;
      std::vector<double> c1;
      std::vector<double> c2;
      std::vector<double> c2_large;
      std::vector<double> s2_large;
      std::vector<double> C1_large;
    };

  } // namespace tables

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
    PhotonPairLpmComponentSnapshot components[MaxPhotonPairLpmComponents]{};
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
      throw std::invalid_argument("photon-pair LPM component count is unsupported");
    }
    PhotonPairLpmSnapshot result{};
    result.baseline_mass_density_g_per_cm3 = source.baseline_mass_density_g_per_cm3;
    result.molecular_density_per_cm3 = source.molecular_density_per_cm3;
    result.sum_charge = source.sum_charge;
    result.e_lpm_MeV = source.e_lpm_MeV;
    result.classical_electron_radius_cm = source.classical_electron_radius_cm;
    result.fine_structure_constant = source.fine_structure_constant;
    result.component_count = static_cast<std::uint32_t>(source.components.size());
    for (std::size_t index = 0; index < source.components.size(); ++index) {
      auto const& component = source.components[index];
      result.components[index] = {component.proposal_hash, component.nuclear_charge,
                                  component.radiation_log_constant};
    }
    return result;
  }

  namespace detail {

    CORSIKA_HOST_DEVICE inline bool lpmFinite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double lpmSqrt(double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double lpmLog(double value) {
#if defined(__CUDA_ARCH__)
      return ::log(value);
#else
      return std::log(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double lpmExp(double value) {
#if defined(__CUDA_ARCH__)
      return ::exp(value);
#else
      return std::exp(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double lpmPow(double base, double exponent) {
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
  CORSIKA_HOST_DEVICE inline PhotonPairLpmResult photonPairLpmSuppressionFactor(
      PhotonPairLpmSnapshot const& snapshot, std::uint64_t component_hash,
      double energy_MeV, double x, double local_mass_density_g_per_cm3) {
    using namespace detail;
    if (snapshot.component_count == 0 ||
        snapshot.component_count > MaxPhotonPairLpmComponents ||
        !lpmFinite(snapshot.baseline_mass_density_g_per_cm3) ||
        !(snapshot.baseline_mass_density_g_per_cm3 > 0.) ||
        !lpmFinite(snapshot.molecular_density_per_cm3) ||
        !(snapshot.molecular_density_per_cm3 > 0.) || !lpmFinite(snapshot.sum_charge) ||
        !(snapshot.sum_charge > 0.) || !lpmFinite(snapshot.e_lpm_MeV) ||
        !(snapshot.e_lpm_MeV > 0.) || !lpmFinite(snapshot.classical_electron_radius_cm) ||
        !(snapshot.classical_electron_radius_cm > 0.) ||
        !lpmFinite(snapshot.fine_structure_constant) ||
        !(snapshot.fine_structure_constant > 0.)) {
      return {PhotonPairLpmStatus::InvalidSnapshot, 0, 0.};
    }
    if (!lpmFinite(energy_MeV) || !(energy_MeV > 0.) || !lpmFinite(x) || !(x > 0.) ||
        !(x < 1.) || !lpmFinite(local_mass_density_g_per_cm3) ||
        !(local_mass_density_g_per_cm3 > 0.)) {
      return {PhotonPairLpmStatus::InvalidInput, 0, 0.};
    }

    PhotonPairLpmComponentSnapshot component{};
    bool found = false;
    for (std::uint32_t index = 0; index < snapshot.component_count; ++index) {
      if (snapshot.components[index].component_hash == component_hash) {
        component = snapshot.components[index];
        found = true;
        break;
      }
    }
    if (!found) { return {PhotonPairLpmStatus::ComponentNotFound, 0, 0.}; }
    if (!lpmFinite(component.nuclear_charge) || !(component.nuclear_charge > 0.) ||
        !lpmFinite(component.radiation_log_constant) ||
        !(component.radiation_log_constant > 0.)) {
      return {PhotonPairLpmStatus::InvalidSnapshot, 0, 0.};
    }

    constexpr double Pi = 3.141592653589793238462643383279502884;
    constexpr double SqrtTwo = 1.414213562373095048801688724209698079;
    constexpr double Fi1 = 1.54954;
    constexpr double G1 = 0.710390;
    constexpr double G2 = 0.904912;

    auto const density_correction =
        local_mass_density_g_per_cm3 / snapshot.baseline_mass_density_g_per_cm3;
    auto const z_to_minus_third = lpmPow(component.nuclear_charge, -1. / 3.);
    auto s1 = 1. / (z_to_minus_third * component.radiation_log_constant);
    s1 = s1 * s1 * SqrtTwo;

    auto const sp = 0.125 * lpmSqrt(snapshot.e_lpm_MeV /
                                    (density_correction * energy_MeV * x * (1. - x)));
    auto const h = lpmLog(sp) / lpmLog(s1);
    double xi = 1.;
    if (sp < s1) {
      xi = 2.;
    } else if (sp < 1.) {
      xi = 1. + h - 0.08 * (1. - h) * (1. - (1. - h) * (1. - h)) / lpmLog(s1);
    }

    auto gamma =
        snapshot.classical_electron_radius_cm / (snapshot.fine_structure_constant * x);
    gamma = 1. + 4. * Pi * snapshot.sum_charge * snapshot.classical_electron_radius_cm *
                     gamma * gamma * snapshot.molecular_density_per_cm3 *
                     density_correction;
    auto const s = sp / lpmSqrt(xi) * gamma;
    auto const s2 = s * s;

    double fi = 0.;
    if (s < Fi1) {
      fi = 1. - lpmExp(-6. * s * (1. + (3. - Pi) * s) +
                       s2 * s / (0.623 + 0.796 * s + 0.658 * s2));
    } else {
      fi = 1. - 0.012 / (s2 * s2);
    }

    double g = 0.;
    if (s < G1) {
      auto const psi =
          1. -
          lpmExp(-4. * s -
                 8. * s2 / (1. + 3.936 * s + 4.97 * s2 - 0.05 * s2 * s + 7.50 * s2 * s2));
      g = 3. * psi - 2. * fi;
    } else if (s < G2) {
      g = 36. * s2 / (36. * s2 + 1.);
    } else {
      g = 1. - 0.022 / (s2 * s2);
    }

    auto const probability =
        ((xi / 3.) *
         (g / (gamma * gamma) + 2. * (x * x + (1. - x) * (1. - x)) * fi / gamma)) /
        (1. - (4. / 3.) * x * (1. - x));
    if (!lpmFinite(probability) || probability < 0.) {
      return {PhotonPairLpmStatus::NonFiniteResult, 0, 0.};
    }
    return {PhotonPairLpmStatus::Success, 0, probability};
  }

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
    // Exported from the live muon calculator, not
    // inferred by scaling the electron characteristic energy.
    double muon_baseline_mass_density_g_per_cm3{};
    double muon_molecular_density_per_cm3{};
    double muon_sum_charge{};
    double muon_e_lpm_MeV{};
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
    BremsLpmPreparedComponent components[MaxBremsLpmComponents]{};
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
  static_assert(std::is_standard_layout_v<BremsLpmPreparedSnapshot>);
  static_assert(std::is_trivially_copyable_v<BremsLpmPreparedSnapshot>);
  static_assert(std::is_standard_layout_v<BremsLpmResult>);
  static_assert(std::is_trivially_copyable_v<BremsLpmResult>);

  inline BremsLpmSnapshot makeBremsLpmSnapshot(tables::BremsLpmMetadata const& source) {
    if (source.components.empty() || source.components.size() > MaxBremsLpmComponents) {
      throw std::invalid_argument("bremsstrahlung LPM component count is unsupported");
    }
    BremsLpmSnapshot result{};
    result.baseline_mass_density_g_per_cm3 = source.baseline_mass_density_g_per_cm3;
    result.molecular_density_per_cm3 = source.molecular_density_per_cm3;
    result.sum_charge = source.sum_charge;
    result.e_lpm_MeV = source.e_lpm_MeV;
    result.lepton_mass_MeV = source.lepton_mass_MeV;
    result.electron_mass_MeV = source.electron_mass_MeV;
    result.muon_mass_MeV = source.muon_mass_MeV;
    result.classical_electron_radius_cm = source.classical_electron_radius_cm;
    result.fine_structure_constant = source.fine_structure_constant;
    result.component_count = static_cast<std::uint32_t>(source.components.size());
    for (std::size_t index = 0; index < source.components.size(); ++index) {
      auto const& component = source.components[index];
      result.components[index] = {component.proposal_hash, component.nuclear_charge,
                                  component.atomic_mass_number,
                                  component.radiation_log_constant};
    }
    return result;
  }

  namespace brems_lpm_detail {

    CORSIKA_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double squareRoot(double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double logarithm(double value) {
#if defined(__CUDA_ARCH__)
      return ::log(value);
#else
      return std::log(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double exponential(double value) {
#if defined(__CUDA_ARCH__)
      return ::exp(value);
#else
      return std::exp(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double power(double base, double exponent) {
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
  CORSIKA_HOST_DEVICE inline BremsLpmResult bremsLpmSuppressionFactor(
      BremsLpmSnapshot const& snapshot, std::uint64_t component_hash, double energy_MeV,
      double v, double local_mass_density_g_per_cm3) {
    using namespace brems_lpm_detail;
    if (snapshot.component_count == 0 ||
        snapshot.component_count > MaxBremsLpmComponents ||
        !finite(snapshot.baseline_mass_density_g_per_cm3) ||
        !(snapshot.baseline_mass_density_g_per_cm3 > 0.) ||
        !finite(snapshot.molecular_density_per_cm3) ||
        !(snapshot.molecular_density_per_cm3 > 0.) || !finite(snapshot.sum_charge) ||
        !(snapshot.sum_charge > 0.) || !finite(snapshot.e_lpm_MeV) ||
        !(snapshot.e_lpm_MeV > 0.) || !finite(snapshot.lepton_mass_MeV) ||
        !(snapshot.lepton_mass_MeV > 0.) || !finite(snapshot.electron_mass_MeV) ||
        !(snapshot.electron_mass_MeV > 0.) || !finite(snapshot.muon_mass_MeV) ||
        !(snapshot.muon_mass_MeV > 0.) ||
        !finite(snapshot.classical_electron_radius_cm) ||
        !(snapshot.classical_electron_radius_cm > 0.) ||
        !finite(snapshot.fine_structure_constant) ||
        !(snapshot.fine_structure_constant > 0.)) {
      return {BremsLpmStatus::InvalidSnapshot, 0, 0.};
    }
    if (!finite(energy_MeV) || !(energy_MeV > snapshot.lepton_mass_MeV) || !finite(v) ||
        !(v > 0.) || !(v < 1.) || !finite(local_mass_density_g_per_cm3) ||
        !(local_mass_density_g_per_cm3 > 0.)) {
      return {BremsLpmStatus::InvalidInput, 0, 0.};
    }

    BremsLpmComponentSnapshot component{};
    bool found = false;
    for (std::uint32_t index = 0; index < snapshot.component_count; ++index) {
      if (snapshot.components[index].component_hash == component_hash) {
        component = snapshot.components[index];
        found = true;
        break;
      }
    }
    if (!found) { return {BremsLpmStatus::ComponentNotFound, 0, 0.}; }
    if (!finite(component.nuclear_charge) || !(component.nuclear_charge > 0.) ||
        !finite(component.atomic_mass_number) || !(component.atomic_mass_number > 0.) ||
        !finite(component.radiation_log_constant) ||
        !(component.radiation_log_constant > 0.)) {
      return {BremsLpmStatus::InvalidSnapshot, 0, 0.};
    }

    constexpr double Pi = 3.141592653589793238462643383279502884;
    constexpr double SqrtTwo = 1.414213562373095048801688724209698079;
    constexpr double Fi1 = 1.54954;
    constexpr double G1 = 0.710390;
    constexpr double G2 = 0.904912;

    auto const density_correction =
        local_mass_density_g_per_cm3 / snapshot.baseline_mass_density_g_per_cm3;
    auto const z_to_minus_third = power(component.nuclear_charge, -1. / 3.);
    auto const d_n = 1.54 * power(component.atomic_mass_number, 0.27);
    auto const nuclear_auxiliary =
        snapshot.lepton_mass_MeV / snapshot.muon_mass_MeV * d_n;
    auto s1 = snapshot.electron_mass_MeV / (snapshot.lepton_mass_MeV * z_to_minus_third *
                                            component.radiation_log_constant);
    s1 = s1 * s1 * (1. + nuclear_auxiliary * nuclear_auxiliary) * SqrtTwo;

    auto const sp = 0.125 * squareRoot(snapshot.e_lpm_MeV * v /
                                       (density_correction * energy_MeV * (1. - v)));
    auto const h = logarithm(sp) / logarithm(s1);
    double xi = 1.;
    if (sp < s1) {
      xi = 2.;
    } else if (sp < 1.) {
      xi = 1. + h - 0.08 * (1. - h) * (1. - (1. - h) * (1. - h)) / logarithm(s1);
    }

    auto gamma = snapshot.classical_electron_radius_cm * snapshot.electron_mass_MeV /
                 (snapshot.fine_structure_constant * snapshot.lepton_mass_MeV * v);
    gamma = 1. + 4. * Pi * snapshot.sum_charge * snapshot.classical_electron_radius_cm *
                     gamma * gamma * snapshot.molecular_density_per_cm3 *
                     density_correction;
    auto const s = sp / squareRoot(xi) * gamma;
    auto const s2 = s * s;

    double fi = 0.;
    if (s < Fi1) {
      fi = 1. - exponential(-6. * s * (1. + (3. - Pi) * s) +
                            s2 * s / (0.623 + 0.796 * s + 0.658 * s2));
    } else {
      fi = 1. - 0.012 / (s2 * s2);
    }

    double g = 0.;
    if (s < G1) {
      auto const psi = 1. - exponential(-4. * s - 8. * s2 /
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
         (v * v * g / (gamma * gamma) + 2. * (1. + (1. - v) * (1. - v)) * fi / gamma)) /
        ((4. / 3.) * (1. - v) + v * v);
    if (!finite(probability) || probability < 0.) {
      return {BremsLpmStatus::NonFiniteResult, 0, 0.};
    }
    return {BremsLpmStatus::Success, 0, probability};
  }

  inline constexpr std::size_t MaxMoliereComponents = 16;
  inline constexpr std::size_t MoliereSeriesCoefficientCount = 70;
  inline constexpr std::size_t MoliereLargeSeriesCoefficientCount = 13;
  inline constexpr std::size_t MoliereLargeIntegralCoefficientCount = 15;
  struct MoliereComponentSnapshot {
    double nuclear_charge{};
    double weight_zz{};
    // Static screening factors derived once from the versioned medium.
    double chi_0_squared_MeV2{};
    double coulomb_correction{};
  };
  struct MoliereSnapshot {
    double particle_mass_MeV{};
    double electron_mass_MeV{};
    double fine_structure_constant{};
    double avogadro_per_mol{};
    double hbar_MeV_s{};
    double speed_of_light_cm_per_s{};
    double z_squared_over_a_average{};
    double inverse_weight_zz_sum{};
    double euler_mascheroni{};
    std::uint32_t component_count{};
    std::uint32_t maximum_weight_index{};
    MoliereComponentSnapshot components[MaxMoliereComponents]{};
    double c1[MoliereSeriesCoefficientCount]{};
    double c2[MoliereSeriesCoefficientCount]{};
    double c2_large[MoliereLargeSeriesCoefficientCount]{};
    double s2_large[MoliereLargeSeriesCoefficientCount]{};
    double C1_large[MoliereLargeIntegralCoefficientCount]{};
  };
  namespace moliere_detail {
    inline constexpr double Pi = 3.141592653589793238462643383279502884;
    CORSIKA_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline bool validSnapshot(MoliereSnapshot const& snapshot) {
      if (snapshot.component_count == 0 ||
          snapshot.component_count > MaxMoliereComponents ||
          snapshot.maximum_weight_index >= snapshot.component_count ||
          !finite(snapshot.particle_mass_MeV) || !(snapshot.particle_mass_MeV > 0.) ||
          !finite(snapshot.electron_mass_MeV) || !(snapshot.electron_mass_MeV > 0.) ||
          !finite(snapshot.fine_structure_constant) ||
          !(snapshot.fine_structure_constant > 0.) ||
          !finite(snapshot.avogadro_per_mol) || !(snapshot.avogadro_per_mol > 0.) ||
          !finite(snapshot.hbar_MeV_s) || !(snapshot.hbar_MeV_s > 0.) ||
          !finite(snapshot.speed_of_light_cm_per_s) ||
          !(snapshot.speed_of_light_cm_per_s > 0.) ||
          !finite(snapshot.z_squared_over_a_average) ||
          !(snapshot.z_squared_over_a_average > 0.) ||
          !finite(snapshot.inverse_weight_zz_sum) ||
          !(snapshot.inverse_weight_zz_sum > 0.) || !finite(snapshot.euler_mascheroni)) {
        return false;
      }
      for (std::uint32_t index = 0; index < snapshot.component_count; ++index) {
        if (!finite(snapshot.components[index].nuclear_charge) ||
            !(snapshot.components[index].nuclear_charge > 0.) ||
            !finite(snapshot.components[index].weight_zz) ||
            !(snapshot.components[index].weight_zz > 0.) ||
            !finite(snapshot.components[index].chi_0_squared_MeV2) ||
            !(snapshot.components[index].chi_0_squared_MeV2 > 0.) ||
            !finite(snapshot.components[index].coulomb_correction) ||
            !(snapshot.components[index].coulomb_correction > 0.)) {
          return false;
        }
      }
      for (std::size_t index = 0; index < MoliereSeriesCoefficientCount; ++index) {
        if (!finite(snapshot.c1[index]) || !finite(snapshot.c2[index])) { return false; }
      }
      for (std::size_t index = 0; index < MoliereLargeSeriesCoefficientCount; ++index) {
        if (!finite(snapshot.c2_large[index]) || !finite(snapshot.s2_large[index])) {
          return false;
        }
      }
      for (double coefficient : snapshot.C1_large) {
        if (!finite(coefficient)) { return false; }
      }
      return true;
    }
  } // namespace moliere_detail
  inline MoliereSnapshot makeMoliereSnapshot(tables::MoliereMetadata const& source) {
    if (!source.enabled || source.components.empty() ||
        source.components.size() > MaxMoliereComponents ||
        source.c1.size() != MoliereSeriesCoefficientCount ||
        source.c2.size() != MoliereSeriesCoefficientCount ||
        source.c2_large.size() != MoliereLargeSeriesCoefficientCount ||
        source.s2_large.size() != MoliereLargeSeriesCoefficientCount ||
        source.C1_large.size() != MoliereLargeIntegralCoefficientCount) {
      throw std::invalid_argument("Moliere metadata is unavailable or malformed");
    }
    MoliereSnapshot snapshot{};
    snapshot.particle_mass_MeV = source.particle_mass_MeV;
    snapshot.electron_mass_MeV = source.electron_mass_MeV;
    snapshot.fine_structure_constant = source.fine_structure_constant;
    snapshot.avogadro_per_mol = source.avogadro_per_mol;
    snapshot.hbar_MeV_s = source.hbar_MeV_s;
    snapshot.speed_of_light_cm_per_s = source.speed_of_light_cm_per_s;
    snapshot.euler_mascheroni = source.euler_mascheroni;
    snapshot.component_count = static_cast<std::uint32_t>(source.components.size());

    double atomic_mass_sum = 0.;
    for (auto const& component : source.components) {
      atomic_mass_sum += component.atoms_in_molecule * component.atomic_mass_number;
    }
    double mass_weights[MaxMoliereComponents]{};
    double average_atomic_mass = 0.;
    double average_charge_squared = 0.;
    double weight_zz_sum = 0.;
    for (std::size_t index = 0; index < source.components.size(); ++index) {
      auto const& component = source.components[index];
      auto const weight =
          component.atoms_in_molecule * component.atomic_mass_number / atomic_mass_sum;
      mass_weights[index] = weight;
      average_atomic_mass += weight * component.atomic_mass_number;
      auto const weight_zz = weight * component.nuclear_charge * component.nuclear_charge;
      weight_zz_sum += weight_zz;
      average_charge_squared +=
          source.particle_mass_MeV == source.electron_mass_MeV
              ? weight * component.nuclear_charge * (component.nuclear_charge + 1.)
              : weight_zz;
      auto const chi_0 = source.electron_mass_MeV * source.fine_structure_constant *
                         std::pow(component.nuclear_charge * 128. /
                                      (9. * moliere_detail::Pi * moliere_detail::Pi),
                                  1. / 3.);
      snapshot.components[index] = {
          component.nuclear_charge, weight_zz, chi_0 * chi_0,
          3.76 * source.fine_structure_constant * source.fine_structure_constant *
              component.nuclear_charge * component.nuclear_charge};
    }
    for (std::size_t index = 0; index + 1 < source.components.size(); ++index) {
      if (mass_weights[index + 1] > mass_weights[index]) {
        snapshot.maximum_weight_index = static_cast<std::uint32_t>(index + 1);
      }
    }
    snapshot.z_squared_over_a_average = average_charge_squared / average_atomic_mass;
    snapshot.inverse_weight_zz_sum = 1. / weight_zz_sum;
    std::copy(source.c1.begin(), source.c1.end(), snapshot.c1);
    std::copy(source.c2.begin(), source.c2.end(), snapshot.c2);
    std::copy(source.c2_large.begin(), source.c2_large.end(), snapshot.c2_large);
    std::copy(source.s2_large.begin(), source.s2_large.end(), snapshot.s2_large);
    std::copy(source.C1_large.begin(), source.C1_large.end(), snapshot.C1_large);
    if (!moliere_detail::validSnapshot(snapshot)) {
      throw std::invalid_argument("Moliere metadata produced an invalid device snapshot");
    }
    return snapshot;
  }
} // namespace corsika::kokkos::test::auxiliary_reference
