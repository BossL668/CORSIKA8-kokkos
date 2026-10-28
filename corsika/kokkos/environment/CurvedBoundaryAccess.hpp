/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentSearch.hpp>
#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include <corsika/kokkos/environment/GeometryIntersections.hpp>
#include <corsika/kokkos/environment/NextEnvironmentBoundary.hpp>
#include <corsika/kokkos/environment/StraightIntersections.hpp>
#include <corsika/kokkos/utility/StableQuartic.hpp>

#include <cstdint>

namespace corsika::kokkos {

  /**
   * Per-trajectory numerical access to production curved boundary intersections.
   *
   * The caller supplies an already selected magnetic field and trajectory in
   * the exported tree's common coordinate system. This class does not read a
   * material model, calculate the magnetic step limit, convert particle units,
   * advance a particle or perform a containing-node search. Intersections are
   * returned in seconds, preserving the production leap-frog parameterization.
   *
   * Position, velocity, direction, field, speed, momentum and charge must be
   * finite and mutually consistent: speed and momentum are positive, and
   * direction is the unit direction corresponding to velocity. Gyroradius and
   * the linear flag describe the supplied trajectory; the production positive
   * infinity sentinel for a linear limit is retained. No
   * normalization, validation or repair is added by this adapter. The tree's
   * valid-index, supported-shape and memory-accessibility preconditions apply.
   *
   * Construct a fresh local instance for each particle or query, including
   * directly inside a device kernel. This copies the tree view and numerical
   * inputs without allocating storage. Retain owning host views until kernels
   * finish; device-local copies do not own the host lifetime of allocations.
   * Both diagnostic flags accumulate across all intersections made through
   * this instance, even nonwinning candidates. Check them after the complete
   * ordered boundary search; a flag makes that query unusable, not a silent miss.
   */
  template <typename TTreeView>
  class CurvedBoundaryAccess {
  public:
    using node_index_type = typename TTreeView::node_index_type;

    /**
     * Copy a tree view and the supplied production trajectory parameters.
     *
     * @param tree Read-only numerical tree, alive and accessible for the query.
     * @param positionM Position in metres in the common coordinate system.
     * @param velocityMPerS Velocity in metres per second in that system.
     * @param direction Corresponding dimensionless unit direction.
     * @param fieldT Magnetic field in tesla in the same coordinate system.
     * @param speedMPerS Positive velocity magnitude in metres per second.
     * @param momentumKgMPerS Positive momentum magnitude in kilograms metres per second.
     * @param chargeC Signed electric charge in coulombs.
     * @param gyroradiusM Previously computed gyroradius in metres, including the
     *   production positive infinity sentinel for a linear limit.
     * @param linear Whether this query uses the production straight-trajectory branch.
     */
    KOKKOS_INLINE_FUNCTION CurvedBoundaryAccess(
        TTreeView const& tree, double const (&positionM)[3],
        double const (&velocityMPerS)[3], double const (&direction)[3],
        double const (&fieldT)[3], double speedMPerS, double momentumKgMPerS,
        double chargeC, double gyroradiusM, bool linear);

    /// Intersect a valid tree node and accumulate any production diagnostics.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION GeometryIntersections
    intersect(node_index_type node) const;

    /**
     * Intersect one shape, including a separate observation plane outside the tree.
     *
     * The linear flag is used unchanged. Observation-plane callers that require
     * the curved plane solver must bind their query with linear=false, even when
     * the transport limit selected a straight trajectory. Charge zero retains
     * the production straight plane fallback.
     */
    [[nodiscard]] KOKKOS_INLINE_FUNCTION GeometryIntersections
    intersectShape(EnvironmentShape const& shape) const;

    /// Saved speed in metres per second, without recalculating particle kinematics.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION double getSpeedMPerS() const;

    /// Whether any curved intersection encountered an unsupported shape.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION bool hasUnsupportedShape() const;
    /// Whether any sphere calculation produced nonfinite or incomplete roots.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION bool hasIncompleteRoots() const;

    /// Borrow the parent handle, including the outside-environment sentinel.
    KOKKOS_INLINE_FUNCTION node_index_type getParent(node_index_type node) const;
    /// Number of ordered child handles of this node.
    KOKKOS_INLINE_FUNCTION std::uint32_t getChildCount(node_index_type node) const;
    /// Child handle at a valid local offset in the original list order.
    KOKKOS_INLINE_FUNCTION node_index_type getChild(node_index_type node,
                                                    std::uint32_t child) const;
    /// Number of ordered, directed exclusion handles of this node.
    KOKKOS_INLINE_FUNCTION std::uint32_t getExcludedCount(node_index_type node) const;
    /// Exclusion handle at a valid local offset in the original list order.
    KOKKOS_INLINE_FUNCTION node_index_type getExcluded(node_index_type node,
                                                       std::uint32_t excluded) const;

  private:
    TTreeView tree_;
    double positionM_[3];
    double velocityMPerS_[3];
    double direction_[3];
    double fieldT_[3];
    double speedMPerS_;
    double momentumKgMPerS_;
    double chargeC_;
    double gyroradiusM_;
    bool linear_;
    mutable bool unsupportedShape_{};
    mutable bool incompleteRoots_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/CurvedBoundaryAccess.inl>
