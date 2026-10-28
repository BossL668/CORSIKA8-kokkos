/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  namespace photon_pair_lpm_detail {

    CORSIKA_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double square_root(double value) {
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

  } // namespace photon_pair_lpm_detail

  CORSIKA_HOST_DEVICE inline PhotonPairLpmResult photon_pair_lpm_suppression_factor(
      PhotonPairLpmSnapshot const& snapshot, std::uint64_t component_hash,
      double energy_MeV, double x, double local_mass_density_g_per_cm3) {
    using namespace photon_pair_lpm_detail;
    if (snapshot.componentCount_ == 0 ||
        snapshot.componentCount_ > MaxPhotonPairLpmComponents ||
        !finite(snapshot.baselineMassDensityGPerCm3_) ||
        !(snapshot.baselineMassDensityGPerCm3_ > 0.) ||
        !finite(snapshot.molecularDensityPerCm3_) ||
        !(snapshot.molecularDensityPerCm3_ > 0.) || !finite(snapshot.sumCharge_) ||
        !(snapshot.sumCharge_ > 0.) || !finite(snapshot.eLpmMeV_) ||
        !(snapshot.eLpmMeV_ > 0.) || !finite(snapshot.classicalElectronRadiusCm_) ||
        !(snapshot.classicalElectronRadiusCm_ > 0.) ||
        !finite(snapshot.fineStructureConstant_) ||
        !(snapshot.fineStructureConstant_ > 0.)) {
      return {PhotonPairLpmStatus::InvalidSnapshot, 0, 0.};
    }
    if (!finite(energy_MeV) || !(energy_MeV > 0.) || !finite(x) || !(x > 0.) ||
        !(x < 1.) || !finite(local_mass_density_g_per_cm3) ||
        !(local_mass_density_g_per_cm3 > 0.)) {
      return {PhotonPairLpmStatus::InvalidInput, 0, 0.};
    }

    PhotonPairLpmComponentSnapshot component{};
    bool found = false;
    for (std::uint32_t index = 0; index < snapshot.componentCount_; ++index) {
      if (snapshot.components_[index].componentHash_ == component_hash) {
        component = snapshot.components_[index];
        found = true;
        break;
      }
    }
    if (!found) { return {PhotonPairLpmStatus::ComponentNotFound, 0, 0.}; }
    if (!finite(component.nuclearCharge_) || !(component.nuclearCharge_ > 0.) ||
        !finite(component.radiationLogConstant_) ||
        !(component.radiationLogConstant_ > 0.)) {
      return {PhotonPairLpmStatus::InvalidSnapshot, 0, 0.};
    }

    constexpr double Pi = 3.141592653589793238462643383279502884;
    constexpr double SqrtTwo = 1.414213562373095048801688724209698079;
    constexpr double Fi1 = 1.54954;
    constexpr double G1 = 0.710390;
    constexpr double G2 = 0.904912;

    auto const density_correction =
        local_mass_density_g_per_cm3 / snapshot.baselineMassDensityGPerCm3_;
    auto const z_to_minus_third = power(component.nuclearCharge_, -1. / 3.);
    auto s1 = 1. / (z_to_minus_third * component.radiationLogConstant_);
    s1 = s1 * s1 * SqrtTwo;

    auto const sp = 0.125 * square_root(snapshot.eLpmMeV_ /
                                        (density_correction * energy_MeV * x * (1. - x)));
    auto const h = logarithm(sp) / logarithm(s1);
    double xi = 1.;
    if (sp < s1) {
      xi = 2.;
    } else if (sp < 1.) {
      xi = 1. + h - 0.08 * (1. - h) * (1. - (1. - h) * (1. - h)) / logarithm(s1);
    }

    auto gamma =
        snapshot.classicalElectronRadiusCm_ / (snapshot.fineStructureConstant_ * x);
    gamma = 1. + 4. * Pi * snapshot.sumCharge_ * snapshot.classicalElectronRadiusCm_ *
                     gamma * gamma * snapshot.molecularDensityPerCm3_ *
                     density_correction;
    auto const s = sp / square_root(xi) * gamma;
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
         (g / (gamma * gamma) + 2. * (x * x + (1. - x) * (1. - x)) * fi / gamma)) /
        (1. - (4. / 3.) * x * (1. - x));
    if (!finite(probability) || probability < 0.) {
      return {PhotonPairLpmStatus::NonFiniteResult, 0, 0.};
    }
    return {PhotonPairLpmStatus::Success, 0, probability};
  }

} // namespace corsika::kokkos
