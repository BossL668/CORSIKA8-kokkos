/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/ScalarPhysicalConstants.hpp>
#include <corsika/kokkos/data/ParticleState.hpp>
#include <corsika/kokkos/environment/MagneticStep.hpp>

#include <Kokkos_Macros.hpp>

#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  /// Default magnetic deflection parameter in radians, unchanged from production.
  inline constexpr double DefaultMaximumMagneticDeflection = 0.2;

  /**
   * Value-only result of a uniform-field leap-frog advance.
   *
   * Preserves the production field order and defaults. particle_ is a copy of
   * the input with position, direction and time updated on a completed advance;
   * bendParameter_ is signed and dimensionless, and chordLengthM_ is the endpoint
   * separation in metres. Neither is a substitute for the supplied trajectory
   * length parameter. InvalidInput returns the default record. NonFiniteResult
   * can retain positions written before the failure; it is not a rollback to the
   * input. Callers must inspect status_ before consuming the updated particle.
   */
  struct MagneticAdvanceResult {
    MagneticStepStatus status_{MagneticStepStatus::InvalidInput};
    std::uint32_t reserved_{};
    ParticleState particle_{};
    double bendParameter_{};
    double chordLengthM_{};
  };

  static_assert(std::is_standard_layout_v<MagneticAdvanceResult>);
  static_assert(std::is_trivially_copyable_v<MagneticAdvanceResult>);

  /**
   * Compute the production TrackingLeapFrogCurved magnetic step limit.
   *
   * Uses the vector-projected transverse momentum, retaining the 1 eV/c
   * threshold, the 1e9 m straight-track threshold and the expression
   * 2 cos(alpha) sin(alpha) R. Particle coordinates and the field use the same
   * coordinate system. The existing checks require finite E > m > 0, finite
   * positions, direction, charge and field, and a squared direction norm within
   * 1e-12 of unity; no normalization or additional particle-field validation is
   * performed. In particular, time, weight, identifiers and environment-node
   * assignment do not enter this calculation.
   *
   * \param particle Starting particle; energyGeV_ is total energy in GeV.
   * \param massGeV Rest mass in GeV/c^2 in the same scalar energy convention.
   * \param chargeNumber Signed charge in elementary-charge units.
   * \param fieldT Uniform magnetic field in tesla in the common coordinate system.
   * \param maximumDeflection Finite deflection parameter, strictly between zero
   * and pi/2, in radians.
   * \return Magnetic status, trajectory-length limit and gyroradius in metres.
   * A Linear result carries an infinite distance; its gyroradius retains the
   * original sentinel or computed value, including a non-finite computed value.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION MagneticStepLimitResult
  maximum_uniform_magnetic_step(
      ParticleState const& particle, double massGeV, double chargeNumber,
      double const (&fieldT)[3],
      double maximumDeflection = DefaultMaximumMagneticDeflection);

  /**
   * Advance with the production two-half-step leap-frog polynomial.
   *
   * distanceM is the trajectory parameter v*dt, not the endpoint chord. The
   * displacement uses the unnormalised kick for the second half-step; only the
   * final direction is normalised. The same particle and field checks as the
   * step limiter apply, with a finite nonnegative distance. A zero distance
   * copies the particle and returns Success. Positive straight motion returns
   * Linear when the direction-field cross product or the charge is zero.
   *
   * No magnetic limit, boundary intersection or energy loss is computed here.
   * The caller selects the distance and, when required by the transport policy,
   * supplies a zero field for a previously selected Linear trajectory. Existing
   * partial failure outputs and the original validation scope are preserved:
   * status_ is not a general finiteness check for every output particle field.
   *
   * \param start Starting particle with total energy in GeV, position in metres,
   * unit direction and time in seconds in the common coordinate system.
   * \param massGeV Rest mass in GeV/c^2.
   * \param chargeNumber Signed charge in elementary-charge units.
   * \param fieldT Uniform magnetic field in tesla in the common coordinate system.
   * \param distanceM Finite nonnegative trajectory-length parameter in metres.
   * \return Status, copied or advanced particle, signed bend parameter and chord.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION MagneticAdvanceResult
  advance_uniform_magnetic_field(ParticleState const& start, double massGeV,
                                 double chargeNumber, double const (&fieldT)[3],
                                 double distanceM);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/UniformMagneticField.inl>
