/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <corsika/gpu/em/ObservationPlane.hpp>
#include <corsika/gpu/em/Types.hpp>

#if defined(__CUDACC__)
#include <math_constants.h>
#define CORSIKA_GPU_MAGNETIC_HOST_DEVICE __host__ __device__
#else
#define CORSIKA_GPU_MAGNETIC_HOST_DEVICE
#endif

namespace corsika::gpu::em {

  inline constexpr double GeVPerCToTeslaMeter =
      0.299792458;
  inline constexpr double DefaultMaximumMagneticDeflection = 0.2;

  enum class MagneticStepStatus : std::uint32_t {
    Success = 0,
    Linear = 1,
    InvalidInput = 2,
    NonFiniteResult = 3,
  };

  struct MagneticStepLimitResult {
    MagneticStepStatus status{
        MagneticStepStatus::InvalidInput};
    std::uint32_t reserved{};
    double distance_m{};
    double gyroradius_m{};
  };

  struct MagneticAdvanceResult {
    MagneticStepStatus status{
        MagneticStepStatus::InvalidInput};
    std::uint32_t reserved{};
    EmParticleState particle{};
    double bend_parameter{};
    double chord_length_m{};
  };

  enum class MagneticIntersectionStatus : std::uint32_t {
    Success = 0,
    NoForwardIntersection = 1,
    InvalidInput = 2,
    NonFiniteResult = 3,
  };

  struct MagneticSphereIntersectionResult {
    MagneticIntersectionStatus status{
        MagneticIntersectionStatus::InvalidInput};
    std::uint32_t reserved{};
    double distance_m{};
    double radius_residual_m2{};
  };

  struct MagneticPlaneIntersectionResult {
    MagneticIntersectionStatus status{
        MagneticIntersectionStatus::InvalidInput};
    std::uint32_t reserved{};
    double distance_m{};
    double plane_residual_m{};
  };

  static_assert(std::is_standard_layout_v<MagneticStepLimitResult>);
  static_assert(std::is_trivially_copyable_v<MagneticStepLimitResult>);
  static_assert(std::is_standard_layout_v<MagneticAdvanceResult>);
  static_assert(std::is_trivially_copyable_v<MagneticAdvanceResult>);
  static_assert(
      std::is_standard_layout_v<MagneticSphereIntersectionResult>);
  static_assert(
      std::is_trivially_copyable_v<MagneticSphereIntersectionResult>);
  static_assert(std::is_standard_layout_v<MagneticPlaneIntersectionResult>);
  static_assert(
      std::is_trivially_copyable_v<MagneticPlaneIntersectionResult>);

  namespace magnetic_detail {

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline bool finite(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline double squareRoot(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline double absolute(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::fabs(value);
#else
      return std::abs(value);
#endif
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline double sine(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::sin(value);
#else
      return std::sin(value);
#endif
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline double cosine(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::cos(value);
#else
      return std::cos(value);
#endif
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline double arcCosine(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::acos(value);
#else
      return std::acos(value);
#endif
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline double cubeRoot(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::cbrt(value);
#else
      return std::cbrt(value);
#endif
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline double dot(
        double const left[3], double const right[3]) {
      return left[0] * right[0] +
             left[1] * right[1] +
             left[2] * right[2];
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline void cross(
        double const left[3], double const right[3],
        double result[3]) {
      result[0] =
          left[1] * right[2] - left[2] * right[1];
      result[1] =
          left[2] * right[0] - left[0] * right[2];
      result[2] =
          left[0] * right[1] - left[1] * right[0];
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline bool validParticle(
        EmParticleState const& particle, double mass_GeV,
        double charge_number) {
      if (!finite(particle.energy_GeV) ||
          !(particle.energy_GeV > mass_GeV) ||
          !finite(mass_GeV) || !(mass_GeV > 0.) ||
          !finite(charge_number)) {
        return false;
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (!finite(particle.position_m[axis]) ||
            !finite(particle.direction[axis])) {
          return false;
        }
      }
      auto const norm_squared =
          dot(particle.direction, particle.direction);
      return finite(norm_squared) &&
             absolute(norm_squared - 1.) <= 1.e-12;
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline bool validField(
        double const field_T[3]) {
      return finite(field_T[0]) &&
             finite(field_T[1]) &&
             finite(field_T[2]);
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline double infinity() {
#if defined(__CUDA_ARCH__)
      return CUDART_INF;
#else
      return std::numeric_limits<double>::infinity();
#endif
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline double clampUnit(
        double value) {
      return value < -1. ? -1. : (value > 1. ? 1. : value);
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline void sortSmall(
        double* values, int count) {
      for (int index = 1; index < count; ++index) {
        auto const value = values[index];
        int insertion = index;
        while (insertion > 0 &&
               values[insertion - 1] > value) {
          values[insertion] = values[insertion - 1];
          --insertion;
        }
        values[insertion] = value;
      }
    }

    /**
     * Return the real stationary points of
     * A*t^4 + C*t^2 + D*t + E in the open unit interval.
     */
    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline int
    quarticStationaryPoints(
        double A, double C, double D, double roots[3]) {
      if (!(A > 0.) || !finite(A) || !finite(C) ||
          !finite(D)) {
        return 0;
      }
      // Divide 4*A*t^3 + 2*C*t + D by 4*A.
      auto const p = C / (2. * A);
      auto const q = D / (4. * A);
      auto const half_q = 0.5 * q;
      auto const third_p = p / 3.;
      auto const discriminant =
          half_q * half_q +
          third_p * third_p * third_p;
      int count = 0;
      if (discriminant >= 0.) {
        auto const root_discriminant =
            squareRoot(discriminant);
        auto const root =
            cubeRoot(-half_q + root_discriminant) +
            cubeRoot(-half_q - root_discriminant);
        if (root > 0. && root < 1. && finite(root)) {
          roots[count++] = root;
        }
      } else {
        auto const radius =
            2. * squareRoot(-third_p);
        auto const argument =
            clampUnit(-half_q /
                      squareRoot(
                          -third_p * third_p * third_p));
        auto const phase = arcCosine(argument);
        constexpr double TwoPi =
            6.283185307179586476925286766559;
        for (int branch = 0; branch < 3; ++branch) {
          auto const root =
              radius * cosine(
                           (phase + TwoPi * branch) / 3.);
          if (root > 0. && root < 1. && finite(root)) {
            roots[count++] = root;
          }
        }
      }
      sortSmall(roots, count);
      return count;
    }

    CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline double
    evaluateReducedQuartic(
        double A, double C, double D, double E,
        double parameter) {
      auto const square = parameter * parameter;
      return A * square * square +
             C * square + D * parameter + E;
    }

  } // namespace magnetic_detail

  /**
   * Reproduce TrackingLeapFrogCurved's magnetic step limiter.
   *
   * The momentum threshold (1 eV/c), 1e9 m straight-track threshold and
   * 2 cos(a) sin(a) R step formula intentionally match the CPU
   * implementation. Momentum is expressed in GeV/c and the field in Tesla.
   */
  CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline MagneticStepLimitResult
  maximumUniformMagneticStep(
      EmParticleState const& particle, double mass_GeV,
      double charge_number, double const field_T[3],
      double maximum_deflection =
          DefaultMaximumMagneticDeflection) {
    using namespace magnetic_detail;
    if (!validParticle(particle, mass_GeV, charge_number) ||
        !validField(field_T) ||
        !finite(maximum_deflection) ||
        !(maximum_deflection > 0.) ||
        !(maximum_deflection < 0.5 * 3.141592653589793)) {
      return {};
    }
    auto const field_squared = dot(field_T, field_T);
    if (charge_number == 0. || field_squared == 0.) {
      return {
          MagneticStepStatus::Linear, 0, infinity(), infinity()};
    }
    auto const momentum_GeV =
        squareRoot(
            (particle.energy_GeV - mass_GeV) *
            (particle.energy_GeV + mass_GeV));
    auto const field_parallel =
        dot(particle.direction, field_T);
    auto const perpendicular_field_squared =
        field_squared - field_parallel * field_parallel;
    auto const perpendicular_momentum_GeV =
        momentum_GeV *
        squareRoot(
            perpendicular_field_squared > 0.
                ? perpendicular_field_squared / field_squared
                : 0.);
    if (perpendicular_momentum_GeV < 1.e-9) {
      return {
          MagneticStepStatus::Linear, 0, infinity(), infinity()};
    }
    auto const field_magnitude = squareRoot(field_squared);
    auto const gyroradius =
        perpendicular_momentum_GeV /
        (GeVPerCToTeslaMeter *
         absolute(charge_number) * field_magnitude);
    if (!finite(gyroradius) || gyroradius > 1.e9) {
      return {
          MagneticStepStatus::Linear, 0, infinity(), gyroradius};
    }
    auto const distance =
        2. * cosine(maximum_deflection) *
        sine(maximum_deflection) * gyroradius;
    if (!finite(distance) || !(distance > 0.)) {
      return {
          MagneticStepStatus::NonFiniteResult, 0, 0., gyroradius};
    }
    return {
        MagneticStepStatus::Success, 0, distance, gyroradius};
  }

  /**
   * Advance one state with the same two-half-step leapfrog polynomial used by
   * CORSIKA 8 TrackingLeapFrogCurved.
   *
   * distance_m is the CPU trajectory length parameter v*dt. The position uses
   * the unnormalised kick for the second half-step; only the final direction
   * is normalised, matching LeapFrogTrajectory.
   */
  CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline MagneticAdvanceResult
  advanceUniformMagneticField(
      EmParticleState const& start, double mass_GeV,
      double charge_number, double const field_T[3],
      double distance_m) {
    using namespace magnetic_detail;
    if (!validParticle(start, mass_GeV, charge_number) ||
        !validField(field_T) || !finite(distance_m) ||
        distance_m < 0.) {
      return {};
    }
    MagneticAdvanceResult result{};
    result.particle = start;
    if (distance_m == 0.) {
      result.status = MagneticStepStatus::Success;
      return result;
    }
    auto const momentum_GeV =
        squareRoot(
            (start.energy_GeV - mass_GeV) *
            (start.energy_GeV + mass_GeV));
    auto const coefficient =
        charge_number * GeVPerCToTeslaMeter *
        distance_m / momentum_GeV;
    double direction_cross_field[3]{};
    cross(start.direction, field_T, direction_cross_field);
    double kicked_direction[3]{};
    double displacement[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      auto const kick =
          coefficient * direction_cross_field[axis];
      kicked_direction[axis] =
          start.direction[axis] + kick;
      displacement[axis] =
          distance_m *
          (start.direction[axis] + 0.5 * kick);
      result.particle.position_m[axis] =
          start.position_m[axis] + displacement[axis];
    }
    auto const kicked_norm =
        squareRoot(dot(kicked_direction, kicked_direction));
    auto const chord_length =
        squareRoot(dot(displacement, displacement));
    auto const beta =
        momentum_GeV / start.energy_GeV;
    if (!finite(kicked_norm) || !(kicked_norm > 0.) ||
        !finite(chord_length) || !finite(beta) ||
        !(beta > 0.)) {
      result.status = MagneticStepStatus::NonFiniteResult;
      return result;
    }
    for (int axis = 0; axis < 3; ++axis) {
      result.particle.direction[axis] =
          kicked_direction[axis] / kicked_norm;
    }
    result.particle.time_s =
        start.time_s +
        distance_m / (beta * 299792458.);
    result.status =
        dot(direction_cross_field, direction_cross_field) == 0. ||
                charge_number == 0.
            ? MagneticStepStatus::Linear
            : MagneticStepStatus::Success;
    result.bend_parameter =
        coefficient *
        squareRoot(dot(field_T, field_T));
    result.chord_length_m = chord_length;
    return result;
  }

  /**
   * Find the first crossing of a sphere by one bounded leapfrog trajectory.
   *
   * The leapfrog position is quadratic in its length parameter, so the sphere
   * equation is a quartic with no cubic term. Its cubic derivative partitions
   * [0, max_distance] into monotonic intervals; bisection then finds every
   * possible crossing without relying on spatial sampling. Intersections
   * within 0.1 mm are ignored to match TrackingLeapFrogCurved's volume
   * transition guard.
   */
  CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline
  MagneticSphereIntersectionResult
  intersectUniformMagneticSphere(
      EmParticleState const& start, double mass_GeV,
      double charge_number, double const field_T[3],
      double const sphere_center_m[3], double sphere_radius_m,
      double maximum_distance_m,
      double maximum_deflection =
          DefaultMaximumMagneticDeflection) {
    using namespace magnetic_detail;
    if (!validParticle(start, mass_GeV, charge_number) ||
        !validField(field_T) || !finite(sphere_radius_m) ||
        !(sphere_radius_m > 0.) ||
        !(maximum_distance_m > 0.)) {
      return {};
    }
    double relative[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      if (!finite(sphere_center_m[axis])) {
        return {};
      }
      relative[axis] =
          start.position_m[axis] - sphere_center_m[axis];
    }

    auto const step_limit = maximumUniformMagneticStep(
        start, mass_GeV, charge_number, field_T,
        maximum_deflection);
    if (step_limit.status == MagneticStepStatus::InvalidInput ||
        step_limit.status ==
            MagneticStepStatus::NonFiniteResult) {
      return {};
    }
    // Keep this equal to SphericalAtmosphere's layer-ownership and straight
    // intersection guard. The header is intentionally not included here to
    // keep the magnetic primitive independent of an atmosphere model.
    constexpr double IntersectionGuardM = 1.e-4;
    if (step_limit.status == MagneticStepStatus::Linear) {
      auto const projection =
          dot(relative, start.direction);
      auto const discriminant =
          projection * projection -
          (dot(relative, relative) -
           sphere_radius_m * sphere_radius_m);
      if (!(discriminant > 0.) || !finite(discriminant)) {
        return {
            MagneticIntersectionStatus::NoForwardIntersection,
            0, infinity(), 0.};
      }
      auto const root = squareRoot(discriminant);
      auto const lower = -projection - root;
      auto const upper = -projection + root;
      auto const distance =
          lower > IntersectionGuardM
              ? lower
              : (upper > IntersectionGuardM
                     ? upper
                     : infinity());
      if (!finite(distance) ||
          (finite(maximum_distance_m) &&
           distance > maximum_distance_m)) {
        return {
            MagneticIntersectionStatus::NoForwardIntersection,
            0, infinity(), 0.};
      }
      return {
          MagneticIntersectionStatus::Success, 0, distance, 0.};
    }
    if (!finite(maximum_distance_m)) {
      return {};
    }
    auto const bounded_distance =
        maximum_distance_m < step_limit.distance_m
            ? maximum_distance_m
            : step_limit.distance_m;
    if (!(bounded_distance > IntersectionGuardM) ||
        !finite(bounded_distance)) {
      return {
          MagneticIntersectionStatus::NoForwardIntersection,
          0, infinity(), 0.};
    }

    auto const momentum_GeV =
        squareRoot(
            (start.energy_GeV - mass_GeV) *
            (start.energy_GeV + mass_GeV));
    auto const curvature =
        charge_number * GeVPerCToTeslaMeter /
        momentum_GeV;
    double direction_cross_field[3]{};
    cross(start.direction, field_T, direction_cross_field);
    double quadratic[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      quadratic[axis] =
          0.5 * curvature * direction_cross_field[axis];
    }

    // Scale l = bounded_distance*t so all searched roots are in [0,1].
    auto const distance_squared =
        bounded_distance * bounded_distance;
    auto const A =
        dot(quadratic, quadratic) *
        distance_squared * distance_squared;
    auto const C =
        (dot(start.direction, start.direction) +
         2. * dot(relative, quadratic)) *
        distance_squared;
    auto const D =
        2. * dot(relative, start.direction) *
        bounded_distance;
    auto const E =
        dot(relative, relative) -
        sphere_radius_m * sphere_radius_m;
    if (!finite(A) || !finite(C) || !finite(D) ||
        !finite(E)) {
      return {
          MagneticIntersectionStatus::NonFiniteResult,
          0, 0., 0.};
    }

    double stationary[3]{};
    auto const stationary_count =
        quarticStationaryPoints(A, C, D, stationary);
    double bounds[5]{};
    int bound_count = 0;
    bounds[bound_count++] =
        IntersectionGuardM / bounded_distance;
    for (int index = 0; index < stationary_count; ++index) {
      if (stationary[index] > bounds[0] &&
          stationary[index] < 1.) {
        bounds[bound_count++] = stationary[index];
      }
    }
    bounds[bound_count++] = 1.;
    sortSmall(bounds, bound_count);

    auto left = bounds[0];
    auto left_value =
        evaluateReducedQuartic(A, C, D, E, left);
    if (!finite(left_value)) {
      return {
          MagneticIntersectionStatus::NonFiniteResult,
          0, 0., 0.};
    }
    for (int interval = 1; interval < bound_count;
         ++interval) {
      auto right = bounds[interval];
      auto right_value =
          evaluateReducedQuartic(A, C, D, E, right);
      if (!finite(right_value)) {
        return {
            MagneticIntersectionStatus::NonFiniteResult,
            0, 0., 0.};
      }
      auto const changes_sign =
          (left_value < 0. && right_value > 0.) ||
          (left_value > 0. && right_value < 0.);
      auto const scale =
          absolute(A) + absolute(C) + absolute(D) +
          absolute(E) + sphere_radius_m * sphere_radius_m;
      auto const at_final_endpoint =
          interval + 1 == bound_count &&
          absolute(right_value) <=
              256. * 2.22044604925031308085e-16 *
                  (scale > 1. ? scale : 1.);
      if (changes_sign || at_final_endpoint) {
        if (changes_sign) {
          for (int iteration = 0; iteration < 96;
               ++iteration) {
            auto const middle = 0.5 * (left + right);
            auto const middle_value =
                evaluateReducedQuartic(
                    A, C, D, E, middle);
            if ((left_value < 0. &&
                 middle_value > 0.) ||
                (left_value > 0. &&
                 middle_value < 0.)) {
              right = middle;
              right_value = middle_value;
            } else {
              left = middle;
              left_value = middle_value;
            }
          }
        }
        auto const parameter =
            changes_sign ? 0.5 * (left + right) : right;
        auto const distance =
            bounded_distance * parameter;
        auto const residual =
            evaluateReducedQuartic(
                A, C, D, E, parameter);
        return {
            MagneticIntersectionStatus::Success, 0,
            distance, residual};
      }
      left = right;
      left_value = right_value;
    }
    return {
        MagneticIntersectionStatus::NoForwardIntersection,
        0, infinity(), 0.};
  }

  /**
   * Find the first crossing of the configured observation plane by one
   * bounded leapfrog trajectory.
   *
   * TrackingLeapFrogCurved advances position as
   *   x(l) = x0 + l*u + 0.5*l^2*(q*0.299792458/p)*(u x B).
   * Substitution in n.(x-plane_point)=0 gives the same quadratic solved by
   * the scalar Plane intersection.  Unlike volume boundaries, a terminal
   * plane does not discard positive roots below the 0.1 mm layer guard.
   */
  CORSIKA_GPU_MAGNETIC_HOST_DEVICE inline
  MagneticPlaneIntersectionResult intersectUniformMagneticPlane(
      EmParticleState const& start, double mass_GeV,
      double charge_number, double const field_T[3],
      EnvironmentSnapshot const& environment,
      double maximum_distance_m,
      double maximum_deflection =
          DefaultMaximumMagneticDeflection) {
    using namespace magnetic_detail;
    if (!validParticle(start, mass_GeV, charge_number) ||
        !validField(field_T) ||
        !observation_plane_detail::validObservationPlane(environment) ||
        !(maximum_distance_m > 0.)) {
      return {};
    }
    auto const step_limit = maximumUniformMagneticStep(
        start, mass_GeV, charge_number, field_T,
        maximum_deflection);
    if (step_limit.status == MagneticStepStatus::InvalidInput ||
        step_limit.status == MagneticStepStatus::NonFiniteResult) {
      return {};
    }
    if (step_limit.status == MagneticStepStatus::Linear) {
      auto const intersection = intersectObservationPlaneStraight(
          environment, start.position_m, start.direction,
          maximum_distance_m);
      if (intersection.status == ObservationPlaneStatus::Success) {
        return {MagneticIntersectionStatus::Success, 0,
                intersection.distance_m, intersection.residual_m};
      }
      if (intersection.status ==
          ObservationPlaneStatus::NonFiniteResult) {
        return {MagneticIntersectionStatus::NonFiniteResult, 0, 0., 0.};
      }
      if (intersection.status == ObservationPlaneStatus::InvalidInput) {
        return {};
      }
      return {MagneticIntersectionStatus::NoForwardIntersection, 0,
              infinity(), 0.};
    }
    if (!finite(maximum_distance_m)) {
      return {};
    }
    auto const bounded_distance =
        maximum_distance_m < step_limit.distance_m
            ? maximum_distance_m
            : step_limit.distance_m;
    if (!(bounded_distance > 0.) || !finite(bounded_distance)) {
      return {MagneticIntersectionStatus::NoForwardIntersection, 0,
              infinity(), 0.};
    }

    auto const momentum_GeV = squareRoot(
        (start.energy_GeV - mass_GeV) *
        (start.energy_GeV + mass_GeV));
    auto const curvature =
        charge_number * GeVPerCToTeslaMeter / momentum_GeV;
    double direction_cross_field[3]{};
    cross(start.direction, field_T, direction_cross_field);
    double relative[3]{};
    double quadratic[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      relative[axis] =
          start.position_m[axis] -
          environment.observation_plane_point_m[axis];
      quadratic[axis] =
          0.5 * curvature * direction_cross_field[axis];
    }
    auto const a =
        dot(environment.observation_plane_normal, quadratic);
    auto const b =
        dot(environment.observation_plane_normal, start.direction);
    auto const c =
        dot(environment.observation_plane_normal, relative);
    if (!finite(a) || !finite(b) || !finite(c)) {
      return {MagneticIntersectionStatus::NonFiniteResult, 0, 0., 0.};
    }

    // If the magnetic quadratic is numerically absent, use the exact straight
    // expression rather than dividing by a tiny coefficient.
    auto const coefficient_scale =
        absolute(b) / bounded_distance +
        absolute(c) / (bounded_distance * bounded_distance);
    if (absolute(a) <=
        64. * 2.22044604925031308085e-16 *
            (coefficient_scale > 1.e-300 ? coefficient_scale : 1.e-300)) {
      auto const intersection = intersectObservationPlaneStraight(
          environment, start.position_m, start.direction,
          bounded_distance);
      if (intersection.status == ObservationPlaneStatus::Success) {
        return {MagneticIntersectionStatus::Success, 0,
                intersection.distance_m, intersection.residual_m};
      }
      return {MagneticIntersectionStatus::NoForwardIntersection, 0,
              infinity(), 0.};
    }

    auto discriminant = b * b - 4. * a * c;
    auto const discriminant_scale =
        b * b + absolute(4. * a * c);
    if (discriminant < 0. &&
        discriminant >=
            -64. * 2.22044604925031308085e-16 *
                (discriminant_scale > 1. ? discriminant_scale : 1.)) {
      discriminant = 0.;
    }
    if (discriminant < 0. || !finite(discriminant)) {
      return {MagneticIntersectionStatus::NoForwardIntersection, 0,
              infinity(), 0.};
    }
    auto const root = squareRoot(discriminant);
    auto const q = -0.5 * (b + (b >= 0. ? root : -root));
    auto first = q / a;
    auto second = q != 0. ? c / q : infinity();
    auto distance = infinity();
    if (first > 0. && first <= bounded_distance) {
      distance = first;
    }
    if (second > 0. && second <= bounded_distance &&
        second < distance) {
      distance = second;
    }
    if (!finite(distance)) {
      return {MagneticIntersectionStatus::NoForwardIntersection, 0,
              infinity(), 0.};
    }
    auto const residual = a * distance * distance + b * distance + c;
    if (!finite(residual)) {
      return {MagneticIntersectionStatus::NonFiniteResult, 0, 0., 0.};
    }
    return {MagneticIntersectionStatus::Success, 0, distance, residual};
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_MAGNETIC_HOST_DEVICE
