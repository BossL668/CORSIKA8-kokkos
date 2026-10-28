/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/data/ParticleState.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#if defined(__CUDACC__)
#include <math_constants.h>
#endif

namespace uniform_magnetic_reference {

  // Frozen source: production 41cdc543429b82a216857ad458f83deb0f210a5f,
  // corsika/accelerator/em/common/UniformMagneticField.hpp.
  // Helpers and the two primitive bodies retain the original expressions and
  // branches. Only namespace, annotation and ParticleState member names change.
  // Constants below are independent literals, not the newly migrated definitions.
  using EmParticleState = corsika::kokkos::ParticleState;
  inline constexpr double SpeedOfLightMPerS = 299792458.;
  inline constexpr double GeVPerCToTeslaMeter = .2997925146834389;
  inline constexpr double DefaultMaximumMagneticDeflection = .2;

  enum class MagneticStepStatus : std::uint32_t {
    Success = 0,
    Linear = 1,
    InvalidInput = 2,
    NonFiniteResult = 3,
  };

  struct MagneticStepLimitResult {
    MagneticStepStatus status{MagneticStepStatus::InvalidInput};
    std::uint32_t reserved{};
    double distance_m{};
    double gyroradius_m{};
  };

  struct MagneticAdvanceResult {
    MagneticStepStatus status{MagneticStepStatus::InvalidInput};
    std::uint32_t reserved{};
    EmParticleState particle{};
    double bend_parameter{};
    double chord_length_m{};
  };

  namespace magnetic_detail {

    KOKKOS_INLINE_FUNCTION bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION double squareRoot(double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION double absolute(double value) {
#if defined(__CUDA_ARCH__)
      return ::fabs(value);
#else
      return std::abs(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION double sine(double value) {
#if defined(__CUDA_ARCH__)
      return ::sin(value);
#else
      return std::sin(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION double cosine(double value) {
#if defined(__CUDA_ARCH__)
      return ::cos(value);
#else
      return std::cos(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION double arcCosine(double value) {
#if defined(__CUDA_ARCH__)
      return ::acos(value);
#else
      return std::acos(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION double cubeRoot(double value) {
#if defined(__CUDA_ARCH__)
      return ::cbrt(value);
#else
      return std::cbrt(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION double dot(double const left[3], double const right[3]) {
      return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    }

    KOKKOS_INLINE_FUNCTION void cross(double const left[3], double const right[3],
                                      double result[3]) {
      result[0] = left[1] * right[2] - left[2] * right[1];
      result[1] = left[2] * right[0] - left[0] * right[2];
      result[2] = left[0] * right[1] - left[1] * right[0];
    }

    KOKKOS_INLINE_FUNCTION bool validParticle(EmParticleState const& particle,
                                              double mass_GeV, double charge_number) {
      if (!finite(particle.energyGeV_) || !(particle.energyGeV_ > mass_GeV) ||
          !finite(mass_GeV) || !(mass_GeV > 0.) || !finite(charge_number)) {
        return false;
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (!finite(particle.positionM_[axis]) || !finite(particle.direction_[axis])) {
          return false;
        }
      }
      auto const norm_squared = dot(particle.direction_, particle.direction_);
      return finite(norm_squared) && absolute(norm_squared - 1.) <= 1.e-12;
    }

    KOKKOS_INLINE_FUNCTION bool validField(double const field_T[3]) {
      return finite(field_T[0]) && finite(field_T[1]) && finite(field_T[2]);
    }

    KOKKOS_INLINE_FUNCTION double infinity() {
#if defined(__CUDA_ARCH__)
      return CUDART_INF;
#else
      return std::numeric_limits<double>::infinity();
#endif
    }

  } // namespace magnetic_detail

  /**
   * Reproduce TrackingLeapFrogCurved's magnetic step limiter.
   *
   * The momentum threshold (1 eV/c), 1e9 m straight-track threshold and
   * 2 cos(a) sin(a) R step formula intentionally match the CPU
   * implementation. Momentum is expressed in GeV/c and the field in Tesla.
   */
  KOKKOS_INLINE_FUNCTION MagneticStepLimitResult maximumUniformMagneticStep(
      EmParticleState const& particle, double mass_GeV, double charge_number,
      double const field_T[3],
      double maximum_deflection = DefaultMaximumMagneticDeflection) {
    using namespace magnetic_detail;
    if (!validParticle(particle, mass_GeV, charge_number) || !validField(field_T) ||
        !finite(maximum_deflection) || !(maximum_deflection > 0.) ||
        !(maximum_deflection < 0.5 * 3.141592653589793)) {
      return {};
    }
    auto const field_squared = dot(field_T, field_T);
    if (charge_number == 0. || field_squared == 0.) {
      return {MagneticStepStatus::Linear, 0, infinity(), infinity()};
    }
    auto const momentum_GeV =
        squareRoot((particle.energyGeV_ - mass_GeV) * (particle.energyGeV_ + mass_GeV));
    // Match C8's vector projection, not sqrt(B^2 - (n.B)^2).
    // The latter subtracts nearly equal squares for parallel tracks and can
    // manufacture p_perp > 1 eV/c, incorrectly selecting a microscopic step.
    double momentum[3];
    for (int axis = 0; axis < 3; ++axis)
      momentum[axis] = momentum_GeV * particle.direction_[axis];
    auto const projection = dot(momentum, field_T) / field_squared;
    double perpendicular[3];
    for (int axis = 0; axis < 3; ++axis)
      perpendicular[axis] = momentum[axis] - projection * field_T[axis];
    auto const perpendicular_momentum_GeV = squareRoot(dot(perpendicular, perpendicular));
    if (perpendicular_momentum_GeV < 1.e-9) {
      return {MagneticStepStatus::Linear, 0, infinity(), infinity()};
    }
    auto const field_magnitude = squareRoot(field_squared);
    auto const gyroradius =
        perpendicular_momentum_GeV /
        (GeVPerCToTeslaMeter * absolute(charge_number) * field_magnitude);
    if (!finite(gyroradius) || gyroradius > 1.e9) {
      return {MagneticStepStatus::Linear, 0, infinity(), gyroradius};
    }
    auto const distance =
        2. * cosine(maximum_deflection) * sine(maximum_deflection) * gyroradius;
    if (!finite(distance) || !(distance > 0.)) {
      return {MagneticStepStatus::NonFiniteResult, 0, 0., gyroradius};
    }
    return {MagneticStepStatus::Success, 0, distance, gyroradius};
  }

  /**
   * Advance one state with the same two-half-step leapfrog polynomial used by
   * CORSIKA 8 TrackingLeapFrogCurved.
   *
   * distance_m is the CPU trajectory length parameter v*dt. The position uses
   * the unnormalised kick for the second half-step; only the final direction
   * is normalised, matching LeapFrogTrajectory.
   */
  KOKKOS_INLINE_FUNCTION MagneticAdvanceResult advanceUniformMagneticField(
      EmParticleState const& start, double mass_GeV, double charge_number,
      double const field_T[3], double distance_m) {
    using namespace magnetic_detail;
    if (!validParticle(start, mass_GeV, charge_number) || !validField(field_T) ||
        !finite(distance_m) || distance_m < 0.) {
      return {};
    }
    MagneticAdvanceResult result{};
    result.particle = start;
    if (distance_m == 0.) {
      result.status = MagneticStepStatus::Success;
      return result;
    }
    auto const momentum_GeV =
        squareRoot((start.energyGeV_ - mass_GeV) * (start.energyGeV_ + mass_GeV));
    auto const coefficient =
        charge_number * GeVPerCToTeslaMeter * distance_m / momentum_GeV;
    double direction_cross_field[3]{};
    cross(start.direction_, field_T, direction_cross_field);
    double kicked_direction[3]{};
    double displacement[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      auto const kick = coefficient * direction_cross_field[axis];
      kicked_direction[axis] = start.direction_[axis] + kick;
      displacement[axis] = distance_m * (start.direction_[axis] + 0.5 * kick);
      result.particle.positionM_[axis] = start.positionM_[axis] + displacement[axis];
    }
    auto const kicked_norm = squareRoot(dot(kicked_direction, kicked_direction));
    auto const chord_length = squareRoot(dot(displacement, displacement));
    auto const beta = momentum_GeV / start.energyGeV_;
    if (!finite(kicked_norm) || !(kicked_norm > 0.) || !finite(chord_length) ||
        !finite(beta) || !(beta > 0.)) {
      result.status = MagneticStepStatus::NonFiniteResult;
      return result;
    }
    for (int axis = 0; axis < 3; ++axis) {
      result.particle.direction_[axis] = kicked_direction[axis] / kicked_norm;
    }
    result.particle.timeS_ = start.timeS_ + distance_m / (beta * SpeedOfLightMPerS);
    result.status =
        dot(direction_cross_field, direction_cross_field) == 0. || charge_number == 0.
            ? MagneticStepStatus::Linear
            : MagneticStepStatus::Success;
    result.bend_parameter = coefficient * squareRoot(dot(field_T, field_T));
    result.chord_length_m = chord_length;
    return result;
  }

} // namespace uniform_magnetic_reference
