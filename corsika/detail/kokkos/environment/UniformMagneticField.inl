/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#include <math_constants.h>
#endif

namespace corsika::kokkos {

  namespace uniform_magnetic_detail {

    KOKKOS_INLINE_FUNCTION bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION double square_root(double value) {
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

    KOKKOS_INLINE_FUNCTION double dot(double const left[3], double const right[3]) {
      return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    }

    KOKKOS_INLINE_FUNCTION void cross(double const left[3], double const right[3],
                                      double result[3]) {
      result[0] = left[1] * right[2] - left[2] * right[1];
      result[1] = left[2] * right[0] - left[0] * right[2];
      result[2] = left[0] * right[1] - left[1] * right[0];
    }

    KOKKOS_INLINE_FUNCTION bool valid_particle(ParticleState const& particle,
                                               double massGeV, double chargeNumber) {
      if (!finite(particle.energyGeV_) || !(particle.energyGeV_ > massGeV) ||
          !finite(massGeV) || !(massGeV > 0.) || !finite(chargeNumber)) {
        return false;
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (!finite(particle.positionM_[axis]) || !finite(particle.direction_[axis])) {
          return false;
        }
      }
      auto const normSquared = dot(particle.direction_, particle.direction_);
      return finite(normSquared) && absolute(normSquared - 1.) <= 1.e-12;
    }

    KOKKOS_INLINE_FUNCTION bool valid_field(double const fieldT[3]) {
      return finite(fieldT[0]) && finite(fieldT[1]) && finite(fieldT[2]);
    }

    KOKKOS_INLINE_FUNCTION double infinity() {
#if defined(__CUDA_ARCH__)
      return CUDART_INF;
#else
      return std::numeric_limits<double>::infinity();
#endif
    }

  } // namespace uniform_magnetic_detail

  KOKKOS_INLINE_FUNCTION MagneticStepLimitResult maximum_uniform_magnetic_step(
      ParticleState const& particle, double massGeV, double chargeNumber,
      double const (&fieldT)[3], double maximumDeflection) {
    using namespace uniform_magnetic_detail;
    if (!valid_particle(particle, massGeV, chargeNumber) || !valid_field(fieldT) ||
        !finite(maximumDeflection) || !(maximumDeflection > 0.) ||
        !(maximumDeflection < 0.5 * 3.141592653589793)) {
      return {};
    }
    auto const fieldSquared = dot(fieldT, fieldT);
    if (chargeNumber == 0. || fieldSquared == 0.) {
      return {MagneticStepStatus::Linear, 0, infinity(), infinity()};
    }
    auto const momentumGeV =
        square_root((particle.energyGeV_ - massGeV) * (particle.energyGeV_ + massGeV));
    // Retain the vector projection, not subtraction of nearly equal squares
    // for a field-parallel particle.
    double momentum[3];
    for (int axis = 0; axis < 3; ++axis) {
      momentum[axis] = momentumGeV * particle.direction_[axis];
    }
    auto const projection = dot(momentum, fieldT) / fieldSquared;
    double perpendicular[3];
    for (int axis = 0; axis < 3; ++axis) {
      perpendicular[axis] = momentum[axis] - projection * fieldT[axis];
    }
    auto const perpendicularMomentumGeV = square_root(dot(perpendicular, perpendicular));
    if (perpendicularMomentumGeV < 1.e-9) {
      return {MagneticStepStatus::Linear, 0, infinity(), infinity()};
    }
    auto const fieldMagnitude = square_root(fieldSquared);
    auto const gyroradius =
        perpendicularMomentumGeV / (scalar_constants::MagneticRigidityGeVPerTeslaMeter *
                                    absolute(chargeNumber) * fieldMagnitude);
    if (!finite(gyroradius) || gyroradius > 1.e9) {
      return {MagneticStepStatus::Linear, 0, infinity(), gyroradius};
    }
    auto const distance =
        2. * cosine(maximumDeflection) * sine(maximumDeflection) * gyroradius;
    if (!finite(distance) || !(distance > 0.)) {
      return {MagneticStepStatus::NonFiniteResult, 0, 0., gyroradius};
    }
    return {MagneticStepStatus::Success, 0, distance, gyroradius};
  }

  KOKKOS_INLINE_FUNCTION MagneticAdvanceResult advance_uniform_magnetic_field(
      ParticleState const& start, double massGeV, double chargeNumber,
      double const (&fieldT)[3], double distanceM) {
    using namespace uniform_magnetic_detail;
    if (!valid_particle(start, massGeV, chargeNumber) || !valid_field(fieldT) ||
        !finite(distanceM) || distanceM < 0.) {
      return {};
    }
    MagneticAdvanceResult result{};
    result.particle_ = start;
    if (distanceM == 0.) {
      result.status_ = MagneticStepStatus::Success;
      return result;
    }
    auto const momentumGeV =
        square_root((start.energyGeV_ - massGeV) * (start.energyGeV_ + massGeV));
    auto const coefficient = chargeNumber *
                             scalar_constants::MagneticRigidityGeVPerTeslaMeter *
                             distanceM / momentumGeV;
    double directionCrossField[3]{};
    cross(start.direction_, fieldT, directionCrossField);
    double kickedDirection[3]{};
    double displacement[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      auto const kick = coefficient * directionCrossField[axis];
      kickedDirection[axis] = start.direction_[axis] + kick;
      displacement[axis] = distanceM * (start.direction_[axis] + 0.5 * kick);
      result.particle_.positionM_[axis] = start.positionM_[axis] + displacement[axis];
    }
    auto const kickedNorm = square_root(dot(kickedDirection, kickedDirection));
    auto const chordLength = square_root(dot(displacement, displacement));
    auto const beta = momentumGeV / start.energyGeV_;
    if (!finite(kickedNorm) || !(kickedNorm > 0.) || !finite(chordLength) ||
        !finite(beta) || !(beta > 0.)) {
      result.status_ = MagneticStepStatus::NonFiniteResult;
      return result;
    }
    for (int axis = 0; axis < 3; ++axis) {
      result.particle_.direction_[axis] = kickedDirection[axis] / kickedNorm;
    }
    result.particle_.timeS_ =
        start.timeS_ + distanceM / (beta * scalar_constants::SpeedOfLightMPerS);
    result.status_ =
        dot(directionCrossField, directionCrossField) == 0. || chargeNumber == 0.
            ? MagneticStepStatus::Linear
            : MagneticStepStatus::Success;
    result.bendParameter_ = coefficient * square_root(dot(fieldT, fieldT));
    result.chordLengthM_ = chordLength;
    return result;
  }

} // namespace corsika::kokkos
