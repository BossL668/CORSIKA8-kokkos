/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/kokkos/environment/EnvironmentModel.hpp>

#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  /// Caller-owned transport configuration used by the environment queries.
  struct EnvironmentQueryConfig {
    double maximumMagneticDeflectionRad_{0.2};
  };

  /// Production query states; BelowObservationSurface is unused for a node tree.
  enum class EnvironmentStatus : std::uint32_t {
    Success = 0,
    InvalidSnapshot = 1,
    NonFiniteInput = 2,
    OutsideEnvironment = 3,
    BelowObservationSurface = 4,
    NoForwardIntersection = 5,
    GrammageOutOfRange = 6
  };

  /// Node and local density; the retained radius diagnostic is zero in this domain.
  struct EnvironmentLayerQuery {
    EnvironmentStatus status_{EnvironmentStatus::InvalidSnapshot};
    std::int32_t nodeIndex_{-1};
    double radiusM_{};
    double densityGPerCm3_{};
  };

  /// Straight boundary distance and destination, with the original failure values.
  struct EnvironmentBoundaryQuery {
    EnvironmentStatus status_{EnvironmentStatus::InvalidSnapshot};
    std::int32_t nodeIndex_{-1};
    double distanceM_{};
    double radiusM_{};
    std::int32_t nextNode_{-3};
  };

  /// value_ is grammage in g/cm^2, or trajectory length in metres for the inverse query.
  struct EnvironmentGrammageQuery {
    EnvironmentStatus status_{EnvironmentStatus::InvalidSnapshot};
    std::uint32_t reserved_{};
    double value_{};
  };

  static_assert(std::is_standard_layout_v<EnvironmentLayerQuery>);
  static_assert(std::is_trivially_copyable_v<EnvironmentLayerQuery>);
  static_assert(std::is_standard_layout_v<EnvironmentBoundaryQuery>);
  static_assert(std::is_trivially_copyable_v<EnvironmentBoundaryQuery>);
  static_assert(std::is_standard_layout_v<EnvironmentGrammageQuery>);
  static_assert(std::is_trivially_copyable_v<EnvironmentGrammageQuery>);

  /**
   * Query a node and its local density in an exported environment tree.
   *
   * \param environment Complete read-only exported arrays, with valid topology,
   * supported shapes and model kinds. No per-query scan validates those arrays.
   * \param config Maximum magnetic deflection must be finite and in (0, pi/2).
   * \param positionM Position in metres in the snapshot's common coordinate system.
   * \param direction Optional finite, unit direction in the same system.
   * \param logicalNode -3 requests location from the root. Any other value is an
   * already-assigned node identity and is not replaced by geometric location.
   * \return Original status, logical node and density, including failure values.
   *
   * A node without its own model is outside the supported material environment;
   * no parent properties are inherited. This API covers exported node trees, not
   * the legacy sphere-layer or convex-polyhedron compatibility paths. Callers
   * must explicitly route unsupported environments through their original path.
   */
  template <typename TEnvironmentView>
  [[nodiscard]] CORSIKA_HOST_DEVICE inline EnvironmentLayerQuery query_environment(
      TEnvironmentView const& environment, EnvironmentQueryConfig const& config,
      double const (&positionM)[3], double const* direction = nullptr,
      std::int32_t logicalNode = -3);

  /**
   * Query the same node, then select its nearest ordered straight boundary.
   *
   * positionM, direction and logicalNode follow query_environment(). The unit
   * direction makes the trajectory parameter a length in metres, not time.
   * The independent observation plane does not participate in this search.
   */
  template <typename TEnvironmentView>
  [[nodiscard]] CORSIKA_HOST_DEVICE inline EnvironmentBoundaryQuery
  distance_to_environment_boundary(TEnvironmentView const& environment,
                                   EnvironmentQueryConfig const& config,
                                   double const (&positionM)[3],
                                   double const (&direction)[3],
                                   std::int32_t logicalNode = -3);

  /**
   * Integrate one assigned model along a straight segment, in g/cm^2.
   *
   * nodeIndex selects a model without location or inheritance. positionM is in
   * metres, direction is dimensionless and unit length, and distanceM must be
   * finite and nonnegative. The sliding-planar radial axis is frozen at the
   * starting point. No boundary clipping or additional position check is made.
   * Preserve status and value together: an out-of-range result retains its raw
   * value, which may be nonfinite. A selected physical step needs a finite value.
   */
  template <typename TEnvironmentView>
  [[nodiscard]] CORSIKA_HOST_DEVICE inline EnvironmentGrammageQuery environment_grammage(
      TEnvironmentView const& environment, EnvironmentQueryConfig const& config,
      std::int32_t nodeIndex, double const (&positionM)[3], double const (&direction)[3],
      double distanceM);

  /**
   * Invert the same assigned model's grammage, returning length in metres.
   *
   * Coordinates and direction follow environment_grammage(); grammageGPerCm2
   * must be finite and nonnegative. No boundary truncation is performed. An
   * unreachable finite length returns GrammageOutOfRange and the original raw
   * inverse result, rather than silently substituting zero or a finite limit.
   */
  template <typename TEnvironmentView>
  [[nodiscard]] CORSIKA_HOST_DEVICE inline EnvironmentGrammageQuery environment_arclength(
      TEnvironmentView const& environment, EnvironmentQueryConfig const& config,
      std::int32_t nodeIndex, double const (&positionM)[3], double const (&direction)[3],
      double grammageGPerCm2);

  /**
   * Grammage of a geometry candidate, retaining the production overflow sentinel.
   *
   * Only the original positive, growing exponential conditions permit a finite
   * candidate length with infinite grammage to return Success. Positive infinity
   * is a comparison sentinel, not an acceptable traversed depth for a selected
   * physical step. The ordinary environment_grammage() policy remains unchanged.
   * This node-tree rule is independent of any external-boundary enable flag;
   * legacy non-tree external-boundary options are outside this API's domain.
   */
  template <typename TEnvironmentView>
  [[nodiscard]] CORSIKA_HOST_DEVICE inline EnvironmentGrammageQuery
  geometry_competition_grammage(TEnvironmentView const& environment,
                                EnvironmentQueryConfig const& config,
                                std::int32_t nodeIndex, double const (&positionM)[3],
                                double const (&direction)[3], double distanceM);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/EnvironmentQuery.inl>
