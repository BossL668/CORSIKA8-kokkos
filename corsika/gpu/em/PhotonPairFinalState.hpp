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
#include <corsika/gpu/em/PhotonPairKinematics.hpp>
#include <corsika/gpu/em/PhotonPairLpm.hpp>

#define CORSIKA_GPU_PHOTON_PAIR_FINAL_STATE_HOST_DEVICE \
  C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em {

  enum class PhotonPairFinalStateStatus : std::uint32_t {
    Success = 0,
    ComponentNotFound = 1,
    InvalidSnapshot = 2,
    InvalidInput = 3,
    EnergyBelowAnalyticDomain = 4,
    NegativeDifferentialWeight = 5,
    EnvelopeViolation = 6,
    NonFiniteResult = 7,
  };

  /**
   * Result of one acceptance--rejection trial.
   *
   * A rejected trial is still a successful evaluation: status is Success and
   * accepted is zero.  A non-Success status must be routed to the exact CPU
   * PROPOSAL final-state generator.
   */
  struct PhotonPairFinalStateTrial {
    PhotonPairFinalStateStatus status{
        PhotonPairFinalStateStatus::InvalidSnapshot};
    std::uint32_t accepted{};
    double split_fraction{};
    double differential_weight{};
    double envelope_weight{};
  };

  static_assert(std::is_standard_layout_v<PhotonPairFinalStateTrial>);
  static_assert(std::is_trivially_copyable_v<PhotonPairFinalStateTrial>);

  namespace photon_pair_final_state_detail {

    CORSIKA_GPU_PHOTON_PAIR_FINAL_STATE_HOST_DEVICE inline bool finite(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_GPU_PHOTON_PAIR_FINAL_STATE_HOST_DEVICE inline double
    logarithm(double value) {
#if defined(__CUDA_ARCH__)
      return ::log(value);
#else
      return std::log(value);
#endif
    }

    CORSIKA_GPU_PHOTON_PAIR_FINAL_STATE_HOST_DEVICE inline double power(
        double base, double exponent) {
#if defined(__CUDA_ARCH__)
      return ::pow(base, exponent);
#else
      return std::pow(base, exponent);
#endif
    }

    CORSIKA_GPU_PHOTON_PAIR_FINAL_STATE_HOST_DEVICE inline double maximum(
        double first, double second) {
#if defined(__CUDA_ARCH__)
      return ::fmax(first, second);
#else
      return std::fmax(first, second);
#endif
    }

    CORSIKA_GPU_PHOTON_PAIR_FINAL_STATE_HOST_DEVICE inline bool
    findComponent(
        PhotonPairLpmSnapshot const& snapshot,
        std::uint64_t component_hash,
        PhotonPairLpmComponentSnapshot& component) {
      for (std::uint32_t index = 0;
           index < snapshot.component_count; ++index) {
        if (snapshot.components[index].component_hash ==
            component_hash) {
          component = snapshot.components[index];
          return true;
        }
      }
      return false;
    }

    /**
     * Rho-dependent part of PROPOSAL 7.6.2
     * PhotoPairKochMotz::DifferentialCrossSectionWithoutA().
     *
     * Factors Z(Z+xi), r_e^2 alpha, N_A/A and the sub-50-MeV correction A are
     * independent of rho and cancel in the conditional energy-split CDF.
     */
    CORSIKA_GPU_PHOTON_PAIR_FINAL_STATE_HOST_DEVICE inline double
    kochMotzWeight(
        PhotonPairLpmSnapshot const& snapshot,
        PhotonPairLpmComponentSnapshot const& component,
        double energy_MeV, double rho, double& envelope) {
      auto const charge = component.nuclear_charge;
      auto const z_to_minus_third = power(charge, -1. / 3.);
      auto const log_z = logarithm(charge);
      auto coulomb_auxiliary =
          snapshot.fine_structure_constant * charge;
      coulomb_auxiliary *= coulomb_auxiliary;
      auto coulomb_correction =
          coulomb_auxiliary *
          (1. / (1. + coulomb_auxiliary) + 0.20206 +
           coulomb_auxiliary *
               (-0.0369 +
                coulomb_auxiliary *
                    (0.0083 - 0.002 * coulomb_auxiliary)));
      if (energy_MeV < 50.) {
        coulomb_correction = 0.;
      }
      auto const subtraction =
          (4. / 3.) * log_z + 4. * coulomb_correction;

      auto phi_values = [](double coordinate, double& first,
                           double& second) {
        if (coordinate <= 1.) {
          auto const square = coordinate * coordinate;
          first =
              20.867 - 3.242 * coordinate +
              0.625 * square;
          second =
              20.029 - 1.930 * coordinate -
              0.086 * square;
        } else {
          first =
              21.12 -
              4.184 * logarithm(coordinate + 0.952);
          second = first;
        }
      };
      auto evaluate = [&](double split) {
        auto const denominator =
            energy_MeV * split * (1. - split);
        if (!(denominator > 0.)) {
          return 0.;
        }
        auto const screening_coordinate =
            136. * z_to_minus_third * ElectronMassMeV /
            denominator;
        double phi1 = 0.;
        double phi2 = 0.;
        phi_values(screening_coordinate, phi1, phi2);
        auto const first_shape =
            2. * split * split - 2. * split + 1.;
        auto const second_shape =
            (2. / 3.) * split * (1. - split);
        return first_shape * (phi1 - subtraction) +
               second_shape * (phi2 - subtraction);
      };

      auto weight = evaluate(rho);
      auto const lower = ElectronMassMeV / energy_MeV;
      auto const endpoint_weight = evaluate(lower);
      if (endpoint_weight < 0.) {
        // The uncorrected low-energy Koch--Motz expression can be negative
        // at the phase-space endpoints.  The zero crossing is component
        // dependent: for hydrogen it is about 4.22 MeV, whereas the former
        // fixed 4 MeV boundary was sufficient only for the heavier dry-air
        // components.  PROPOSAL's sub-50-MeV Storm--Israel multiplier is
        // rho-independent and cannot repair a sign-changing conditional
        // density.  Shift by the actual component endpoint value whenever it
        // is negative.  This is continuous at the component-specific zero,
        // preserves symmetry and phase space, and leaves the original
        // expression exactly unchanged everywhere it is non-negative.
        auto const center_weight = evaluate(0.5);
        auto const shift = -endpoint_weight;
        weight += shift;
        envelope = center_weight + shift;
        auto const scale = maximum(
            1., maximum(
                    -endpoint_weight,
                    center_weight < 0.
                        ? -center_weight
                        : center_weight));
        if (!(envelope > 128. * 2.2204460492503131e-16 *
                              scale)) {
          // At the exact threshold all allowed splits collapse to rho=1/2.
          // A uniform normalized coordinate is then physically
          // indistinguishable and avoids cancellation of two nearly equal
          // structure functions.
          weight = 1.;
          envelope = 1.;
        }
        return weight;
      }

      auto const minimum_screening_coordinate =
          136. * z_to_minus_third * ElectronMassMeV /
          (0.25 * energy_MeV);
      double maximum_phi1 = 0.;
      double maximum_phi2 = 0.;
      phi_values(
          minimum_screening_coordinate, maximum_phi1,
          maximum_phi2);
      // x is minimal at rho=1/2, and both Koch--Motz structure
      // functions decrease monotonically with x.  Combining their values at
      // that point with the global polynomial-shape bounds gives a strict,
      // energy-dependent envelope in the unregularized domain.
      envelope =
          maximum(0., maximum_phi1 - subtraction) +
          maximum(0., maximum_phi2 - subtraction) / 6.;
      return weight;
    }

  } // namespace photon_pair_final_state_detail

  /**
   * Evaluate one exact accept/reject trial for the Koch--Motz conditional
   * electron-energy fraction.
   *
   * candidate_uniform chooses rho uniformly inside
   * [m_e/E, 1-m_e/E]. acceptance_uniform tests the analytical envelope.
   * Repeating this operation with independent Random123 draw IDs samples the
   * same underlying differential expression used to construct PROPOSAL's
   * photopair dN/dX interpolant, without a threshold-singular inverse-CDF
   * table.
   */
  CORSIKA_GPU_PHOTON_PAIR_FINAL_STATE_HOST_DEVICE inline
  PhotonPairFinalStateTrial photonPairFinalStateTrial(
      PhotonPairLpmSnapshot const& snapshot,
      std::uint64_t component_hash, double energy_MeV,
      double candidate_uniform, double acceptance_uniform) {
    using namespace photon_pair_final_state_detail;
    if (snapshot.component_count == 0 ||
        snapshot.component_count > MaxPhotonPairLpmComponents ||
        !finite(snapshot.fine_structure_constant) ||
        !(snapshot.fine_structure_constant > 0.)) {
      return {
          PhotonPairFinalStateStatus::InvalidSnapshot, 0, 0., 0., 0.};
    }
    if (!finite(energy_MeV) ||
        !(energy_MeV > PhotonPairThresholdMeV) ||
        !finite(candidate_uniform) ||
        !(candidate_uniform > 0.) ||
        !(candidate_uniform < 1.) ||
        !finite(acceptance_uniform) ||
        !(acceptance_uniform > 0.) ||
        !(acceptance_uniform < 1.)) {
      return {
          PhotonPairFinalStateStatus::InvalidInput, 0, 0., 0., 0.};
    }
    PhotonPairLpmComponentSnapshot component{};
    if (!findComponent(snapshot, component_hash, component)) {
      return {
          PhotonPairFinalStateStatus::ComponentNotFound,
          0, 0., 0., 0.};
    }
    if (!finite(component.nuclear_charge) ||
        !(component.nuclear_charge > 0.)) {
      return {
          PhotonPairFinalStateStatus::InvalidSnapshot, 0, 0., 0., 0.};
    }

    auto const lower = ElectronMassMeV / energy_MeV;
    auto const width = 1. - 2. * lower;
    auto const split = lower + width * candidate_uniform;
    double envelope = 0.;
    auto weight = kochMotzWeight(
        snapshot, component, energy_MeV, split, envelope);
    if (!finite(split) || !finite(weight) || !finite(envelope) ||
        !(width > 0.) || !(envelope > 0.)) {
      return {
          PhotonPairFinalStateStatus::NonFiniteResult,
          0, split, weight, envelope};
    }

    auto constexpr tolerance =
        256. * 2.2204460492503131e-16;
    if (weight < -tolerance * envelope) {
      return {
          PhotonPairFinalStateStatus::NegativeDifferentialWeight,
          0, split, weight, envelope};
    }
    weight = maximum(0., weight);
    if (weight > (1. + tolerance) * envelope) {
      return {
          PhotonPairFinalStateStatus::EnvelopeViolation,
          0, split, weight, envelope};
    }
    auto const accepted =
        acceptance_uniform * envelope <= weight ? 1U : 0U;
    return {
        PhotonPairFinalStateStatus::Success, accepted,
        split, weight, envelope};
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_PHOTON_PAIR_FINAL_STATE_HOST_DEVICE
