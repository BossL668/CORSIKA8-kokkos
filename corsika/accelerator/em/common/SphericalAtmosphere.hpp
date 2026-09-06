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
#include <limits>
#include <type_traits>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/accelerator/em/common/ObservationPlane.hpp>
#include <corsika/accelerator/em/common/Types.hpp>

#if defined(__CUDACC__)
#include <math_constants.h>
#endif
#define CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em {

  inline constexpr double SpeedOfLightMPerS = 299792458.;
  // TrackingLeapFrogCurved ignores volume intersections closer than 0.1 mm.
  // Layer ownership and straight intersections must use the same guard;
  // otherwise a state a few micrometres outside a layer can be assigned to
  // that layer while the magnetic solver intentionally skips the near root
  // and selects a remote second sphere crossing.
  inline constexpr double AtmosphereBoundaryGuardM = 1.e-4;

  enum class AtmosphereStatus : std::uint32_t {
    Success = 0,
    InvalidSnapshot = 1,
    NonFiniteInput = 2,
    OutsideEnvironment = 3,
    BelowObservationSurface = 4,
    NoForwardIntersection = 5,
    GrammageOutOfRange = 6,
  };

  struct AtmosphereLayerQuery {
    AtmosphereStatus status{AtmosphereStatus::InvalidSnapshot};
    std::int32_t layer_index{-1};
    double radius_m{};
    double density_g_per_cm3{};
  };

  struct AtmosphereBoundaryQuery {
    AtmosphereStatus status{AtmosphereStatus::InvalidSnapshot};
    std::int32_t layer_index{-1};
    double distance_m{};
    double radius_m{};
  };

  struct AtmosphereGrammageQuery {
    AtmosphereStatus status{AtmosphereStatus::InvalidSnapshot};
    std::uint32_t reserved{};
    double value{};
  };

  static_assert(std::is_standard_layout_v<AtmosphereLayerQuery>);
  static_assert(std::is_trivially_copyable_v<AtmosphereLayerQuery>);
  static_assert(std::is_standard_layout_v<AtmosphereBoundaryQuery>);
  static_assert(std::is_trivially_copyable_v<AtmosphereBoundaryQuery>);
  static_assert(std::is_standard_layout_v<AtmosphereGrammageQuery>);
  static_assert(std::is_trivially_copyable_v<AtmosphereGrammageQuery>);

  namespace atmosphere_detail {

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline double squareRoot(double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline double exponential(double value) {
#if defined(__CUDA_ARCH__)
      return ::exp(value);
#else
      return std::exp(value);
#endif
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline double exponentialMinusOne(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::expm1(value);
#else
      return std::expm1(value);
#endif
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline double logarithmOnePlus(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::log1p(value);
#else
      return std::log1p(value);
#endif
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline double absolute(double value) {
#if defined(__CUDA_ARCH__)
      return ::fabs(value);
#else
      return std::abs(value);
#endif
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline double infinity() {
#if defined(__CUDA_ARCH__)
      return CUDART_INF;
#else
      return std::numeric_limits<double>::infinity();
#endif
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline double dot(
        double const left[3], double const right[3]) {
      return left[0] * right[0] + left[1] * right[1] +
             left[2] * right[2];
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline double radiusVector(
        EnvironmentSnapshot const& environment, double const position[3],
        double radial[3]) {
      for (int axis = 0; axis < 3; ++axis) {
        radial[axis] =
            position[axis] - environment.earth_center_m[axis];
      }
      return squareRoot(dot(radial, radial));
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline bool normalized(
        double const direction[3]) {
      auto const norm_squared = dot(direction, direction);
      return finite(norm_squared) &&
             absolute(norm_squared - 1.) <= 1.e-12;
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline bool validEnvironment(
        EnvironmentSnapshot const& environment) {
      if (environment.number_of_layers == 0 ||
          environment.number_of_layers > MaxAtmosphereLayers ||
          !finite(environment.observation_radius_m) ||
          !(environment.observation_radius_m > 0.) ||
          !observation_plane_detail::validObservationPlane(environment) ||
          !finite(environment.maximum_magnetic_deflection_rad) ||
          !(environment.maximum_magnetic_deflection_rad > 0.) ||
          !(environment.maximum_magnetic_deflection_rad <
            0.5 * 3.141592653589793)) {
        return false;
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (!finite(environment.earth_center_m[axis])) {
          return false;
        }
      }
      double previous_outer = 0.;
      for (std::uint32_t index = 0;
           index < environment.number_of_layers; ++index) {
        auto const& layer = environment.atmosphere_layers[index];
        if (!finite(layer.inner_radius_m) ||
            !finite(layer.outer_radius_m) ||
            !(layer.inner_radius_m > 0.) ||
            !(layer.outer_radius_m > layer.inner_radius_m) ||
            (index != 0 &&
             absolute(layer.inner_radius_m - previous_outer) >
                 1.e-9 * layer.inner_radius_m) ||
            !finite(layer.density_parameter_a) ||
            !(layer.density_parameter_a > 0.)) {
          return false;
        }
        if (layer.density_model == DensityModel::Exponential) {
          if (!finite(layer.density_parameter_b) ||
              !(layer.density_parameter_b > 0.) ||
              !finite(layer.density_parameter_c) ||
              !(layer.density_parameter_c < 0.)) {
            return false;
          }
        } else if (
            layer.density_model != DensityModel::Homogeneous) {
          return false;
        }
        previous_outer = layer.outer_radius_m;
      }
      auto const& first = environment.atmosphere_layers[0];
      return environment.observation_radius_m >=
                 first.inner_radius_m -
                     1.e-9 * first.inner_radius_m &&
             environment.observation_radius_m <
                 first.outer_radius_m;
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline double density(
        AtmosphereLayerSnapshot const& layer, double radius_m) {
      if (layer.density_model == DensityModel::Exponential) {
        return layer.density_parameter_a *
               exponential(
                   (radius_m - layer.density_parameter_b) /
                   layer.density_parameter_c);
      }
      return layer.density_parameter_a;
    }

    CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline double forwardSphereDistance(
        EnvironmentSnapshot const& environment, double const position[3],
        double const direction[3], double sphere_radius_m) {
      double radial[3]{};
      radiusVector(environment, position, radial);
      auto const projection = dot(radial, direction);
      auto const discriminant =
          projection * projection -
          (dot(radial, radial) -
           sphere_radius_m * sphere_radius_m);
      // Match TrackingStraight: a tangent (zero discriminant) is not treated
      // as a volume crossing.
      if (!(discriminant > 0.) || !finite(discriminant)) {
        return infinity();
      }
      auto const root = squareRoot(discriminant);
      auto const lower = -projection - root;
      auto const upper = -projection + root;
      auto const floating_tolerance =
          32. * 2.22044604925031308085e-16 *
          (sphere_radius_m > 1. ? sphere_radius_m : 1.);
      auto const epsilon =
          floating_tolerance > AtmosphereBoundaryGuardM
              ? floating_tolerance
              : AtmosphereBoundaryGuardM;
      if (lower > epsilon) {
        return lower;
      }
      if (upper > epsilon) {
        return upper;
      }
      return infinity();
    }

  } // namespace atmosphere_detail

  CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline AtmosphereLayerQuery
  queryAtmosphereLayer(
      EnvironmentSnapshot const& environment, double const position_m[3],
      double const* direction = nullptr) {
    using namespace atmosphere_detail;
    if (!validEnvironment(environment)) {
      return {AtmosphereStatus::InvalidSnapshot, -1, 0., 0.};
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (!finite(position_m[axis]) ||
          (direction != nullptr && !finite(direction[axis]))) {
        return {AtmosphereStatus::NonFiniteInput, -1, 0., 0.};
      }
    }
    if (direction != nullptr && !normalized(direction)) {
      return {AtmosphereStatus::NonFiniteInput, -1, 0., 0.};
    }
    double radial[3]{};
    auto const radius_m =
        radiusVector(environment, position_m, radial);
    if (!finite(radius_m)) {
      return {AtmosphereStatus::NonFiniteInput, -1, 0., 0.};
    }
    auto const atmosphere_inner_radius_m =
        environment.atmosphere_layers[0].inner_radius_m;
    if (radius_m < atmosphere_inner_radius_m) {
      return {AtmosphereStatus::BelowObservationSurface, -1,
              radius_m, 0.};
    }
    auto const outermost =
        environment.atmosphere_layers[
            environment.number_of_layers - 1]
            .outer_radius_m;
    if (radius_m > outermost) {
      return {AtmosphereStatus::OutsideEnvironment, -1,
              radius_m, 0.};
    }

    auto const radial_direction =
        direction == nullptr || !(radius_m > 0.)
            ? 0.
            : dot(radial, direction) / radius_m;
    for (std::uint32_t index = 0;
         index < environment.number_of_layers; ++index) {
      auto const& layer = environment.atmosphere_layers[index];
      auto const floating_tolerance =
          64. * 2.22044604925031308085e-16 *
          layer.outer_radius_m;
      auto const tolerance =
          floating_tolerance > AtmosphereBoundaryGuardM
              ? floating_tolerance
              : AtmosphereBoundaryGuardM;
      if (direction != nullptr && index > 0 &&
          absolute(radius_m - layer.inner_radius_m) <=
              tolerance) {
        auto const selected =
            radial_direction < 0.
                ? static_cast<std::int32_t>(index - 1)
                : static_cast<std::int32_t>(index);
        auto const& selected_layer =
            environment.atmosphere_layers[selected];
        return {
            AtmosphereStatus::Success, selected, radius_m,
            density(selected_layer, radius_m)};
      }
      if (radius_m >= layer.inner_radius_m - tolerance &&
          (radius_m < layer.outer_radius_m - tolerance ||
           (index + 1 == environment.number_of_layers &&
            radius_m <= layer.outer_radius_m + tolerance))) {
        return {
            AtmosphereStatus::Success,
            static_cast<std::int32_t>(index), radius_m,
            density(layer, radius_m)};
      }
    }
    return {AtmosphereStatus::OutsideEnvironment, -1,
            radius_m, 0.};
  }

  CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline AtmosphereBoundaryQuery
  distanceToAtmosphereBoundary(
      EnvironmentSnapshot const& environment, double const position_m[3],
      double const direction[3]) {
    using namespace atmosphere_detail;
    auto const layer_query =
        queryAtmosphereLayer(environment, position_m, direction);
    if (layer_query.status != AtmosphereStatus::Success) {
      return {layer_query.status, -1, 0., 0.};
    }
    auto const& layer =
        environment.atmosphere_layers[layer_query.layer_index];
    auto const inner_radius_m = layer.inner_radius_m;
    auto const inner_distance = forwardSphereDistance(
        environment, position_m, direction, inner_radius_m);
    auto const outer_distance = forwardSphereDistance(
        environment, position_m, direction, layer.outer_radius_m);
    auto const use_inner = inner_distance < outer_distance;
    auto const distance_m =
        use_inner ? inner_distance : outer_distance;
    if (!finite(distance_m)) {
      return {AtmosphereStatus::NoForwardIntersection,
              layer_query.layer_index, 0., 0.};
    }
    return {AtmosphereStatus::Success, layer_query.layer_index,
            distance_m,
            use_inner ? inner_radius_m
                      : layer.outer_radius_m};
  }

  /**
   * Reproduces SlidingPlanarExponential grammage for one trajectory segment.
   *
   * Density itself is evaluated with the exact spherical radius. Along the
   * segment, the radial axis is frozen at the starting point, matching the
   * current CORSIKA 8 CPU medium implementation.
   */
  CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline AtmosphereGrammageQuery
  atmosphereGrammage(
      EnvironmentSnapshot const& environment, std::int32_t layer_index,
      double const position_m[3], double const direction[3],
      double distance_m) {
    using namespace atmosphere_detail;
    if (!validEnvironment(environment) || layer_index < 0 ||
        layer_index >=
            static_cast<std::int32_t>(
                environment.number_of_layers) ||
        !finite(distance_m) || distance_m < 0. ||
        !normalized(direction)) {
      return {AtmosphereStatus::NonFiniteInput, 0, 0.};
    }
    double radial[3]{};
    auto const radius_m =
        radiusVector(environment, position_m, radial);
    if (!(radius_m > 0.)) {
      return {AtmosphereStatus::NonFiniteInput, 0, 0.};
    }
    auto const& layer = environment.atmosphere_layers[layer_index];
    auto const rho_start = density(layer, radius_m);
    if (!finite(rho_start) || !(rho_start > 0.)) {
      return {AtmosphereStatus::InvalidSnapshot, 0, 0.};
    }
    if (distance_m == 0.) {
      return {AtmosphereStatus::Success, 0, 0.};
    }
    if (layer.density_model == DensityModel::Homogeneous) {
      return {AtmosphereStatus::Success, 0,
              rho_start * distance_m * 100.};
    }
    auto const radial_cosine =
        dot(radial, direction) / radius_m;
    if (absolute(radial_cosine) < 1.e-14) {
      return {AtmosphereStatus::Success, 0,
              rho_start * distance_m * 100.};
    }
    auto const lambda_m = layer.density_parameter_c;
    auto const grammage =
        rho_start * (lambda_m * 100. / radial_cosine) *
        exponentialMinusOne(
            radial_cosine * distance_m / lambda_m);
    if (!finite(grammage) || grammage < 0.) {
      return {AtmosphereStatus::GrammageOutOfRange, 0, 0.};
    }
    return {AtmosphereStatus::Success, 0, grammage};
  }

  CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline AtmosphereGrammageQuery
  atmosphereDistanceFromGrammage(
      EnvironmentSnapshot const& environment, std::int32_t layer_index,
      double const position_m[3], double const direction[3],
      double grammage_g_per_cm2) {
    using namespace atmosphere_detail;
    if (!validEnvironment(environment) || layer_index < 0 ||
        layer_index >=
            static_cast<std::int32_t>(
                environment.number_of_layers) ||
        !finite(grammage_g_per_cm2) ||
        grammage_g_per_cm2 < 0. || !normalized(direction)) {
      return {AtmosphereStatus::NonFiniteInput, 0, 0.};
    }
    double radial[3]{};
    auto const radius_m =
        radiusVector(environment, position_m, radial);
    if (!(radius_m > 0.)) {
      return {AtmosphereStatus::NonFiniteInput, 0, 0.};
    }
    auto const& layer = environment.atmosphere_layers[layer_index];
    auto const rho_start = density(layer, radius_m);
    if (!finite(rho_start) || !(rho_start > 0.)) {
      return {AtmosphereStatus::InvalidSnapshot, 0, 0.};
    }
    if (grammage_g_per_cm2 == 0.) {
      return {AtmosphereStatus::Success, 0, 0.};
    }
    if (layer.density_model == DensityModel::Homogeneous) {
      return {AtmosphereStatus::Success, 0,
              grammage_g_per_cm2 / (rho_start * 100.)};
    }
    auto const radial_cosine =
        dot(radial, direction) / radius_m;
    if (absolute(radial_cosine) < 1.e-14) {
      return {AtmosphereStatus::Success, 0,
              grammage_g_per_cm2 / (rho_start * 100.)};
    }
    auto const lambda_m = layer.density_parameter_c;
    auto const log_argument =
        grammage_g_per_cm2 * radial_cosine /
        (rho_start * lambda_m * 100.);
    if (!(log_argument > -1.)) {
      return {AtmosphereStatus::GrammageOutOfRange, 0,
              infinity()};
    }
    auto const distance_m =
        lambda_m / radial_cosine *
        logarithmOnePlus(log_argument);
    if (!finite(distance_m) || distance_m < 0.) {
      return {AtmosphereStatus::GrammageOutOfRange, 0,
              infinity()};
    }
    return {AtmosphereStatus::Success, 0, distance_m};
  }

  CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE inline void advancePhotonState(
      EmParticleState const& start, double distance_m,
      EmParticleState& end) {
    end = start;
    for (int axis = 0; axis < 3; ++axis) {
      end.position_m[axis] =
          start.position_m[axis] +
          distance_m * start.direction[axis];
    }
    end.time_s =
        start.time_s + distance_m / SpeedOfLightMPerS;
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_ATMOSPHERE_HOST_DEVICE
