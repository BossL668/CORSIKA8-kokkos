/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "CurvedBoundaryReference.hpp"

#include <corsika/kokkos/data/ParticleState.hpp>
#include <corsika/kokkos/environment/MagneticStep.hpp>

namespace lepton_boundary_reference {

  // Independent copies from production 41cdc543429b82a216857ad458f83deb0f210a5f:
  // ScalarPhysicalConstants.hpp and em/common/EnvironmentLeptonBoundary.hpp.
  // Only record/accessor names and the full view's topology argument are adapted.
  // No newly migrated wrapper or scalar constant is used to compute this oracle.
  inline constexpr double SpeedOfLightMPerS = 299792458.;
  inline constexpr double ElementaryChargeC = 1.6021766208e-19;
  inline constexpr double MagneticRigidityGeVPerTeslaMeter = 0.2997925146834389;

  using particle_type = corsika::kokkos::ParticleState;
  using limit_type = corsika::kokkos::MagneticStepLimitResult;
  using tree_type = straight_boundary_reference::Tree;
  using access_type = curved_boundary_reference::Access;

  struct Boundary {
    double distanceM_{};
    std::int32_t nextNode_{-3};
    bool failed_{};
  };

  KOKKOS_INLINE_FUNCTION access_type lepton_access(tree_type tree,
                                                   particle_type const& particle,
                                                   double massGeV, double chargeNumber,
                                                   double const (&fieldT)[3],
                                                   limit_type const& limit) {
    auto const momentumGeV =
        ::sqrt((particle.energyGeV_ - massGeV) * (particle.energyGeV_ + massGeV));
    access_type access;
    static_cast<tree_type&>(access) = tree;
    access.speed_ = SpeedOfLightMPerS * momentumGeV / particle.energyGeV_;
    access.chargeSI_ = chargeNumber * ElementaryChargeC;
    access.momentumSI_ =
        momentumGeV * ElementaryChargeC / MagneticRigidityGeVPerTeslaMeter;
    access.gyroradiusM_ = limit.gyroradiusM_;
    access.linear_ = limit.status_ == corsika::kokkos::MagneticStepStatus::Linear;
    for (int axis = 0; axis < 3; ++axis) {
      access.position_[axis] = particle.positionM_[axis];
      access.direction_[axis] = particle.direction_[axis];
      access.velocity_[axis] = access.speed_ * particle.direction_[axis];
      access.field_[axis] = fieldT[axis];
    }
    return access;
  }

  KOKKOS_INLINE_FUNCTION Boundary lepton_boundary(tree_type tree,
                                                  particle_type const& particle,
                                                  double massGeV, double chargeNumber,
                                                  double const (&fieldT)[3],
                                                  limit_type const& limit) {
    auto const access =
        lepton_access(tree, particle, massGeV, chargeNumber, fieldT, limit);
    auto const result = boundary_reference::next(access, particle.environmentNode_,
                                                 limit.distanceM_ / access.speed_);
    return {result.parameter_ * access.speed_, result.node_,
            access.unsupportedShape_ || access.incompleteRoots_};
  }

  KOKKOS_INLINE_FUNCTION double lepton_observation_distance(
      tree_type tree, particle_type const& particle, double massGeV, double chargeNumber,
      double const (&fieldT)[3], limit_type const& limit, double const (&pointM)[3],
      double const (&normal)[3]) {
    auto access = lepton_access(tree, particle, massGeV, chargeNumber, fieldT, limit);
    access.linear_ = false;
    curved_boundary_reference::shape_type plane;
    plane.kind_ = curved_boundary_reference::shape_type::Kind::SeparationPlane;
    for (int axis = 0; axis < 3; ++axis) {
      plane.centerM_[axis] = pointM[axis];
      plane.normal_[axis] = normal[axis];
    }
    auto const intersection = access.intersectShape(plane);
    return intersection.entry_ > 0. ? intersection.entry_ * access.speed_
                                    : curved_boundary_reference::infinity();
  }

} // namespace lepton_boundary_reference
