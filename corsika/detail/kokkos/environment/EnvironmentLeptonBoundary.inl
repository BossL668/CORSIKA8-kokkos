/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  namespace lepton_boundary_detail {

    template <typename TTreeView>
    KOKKOS_INLINE_FUNCTION CurvedBoundaryAccess<TTreeView> make_access(
        TTreeView const& tree, ParticleState const& particle, double massGeV,
        double chargeNumber, double const (&fieldT)[3],
        MagneticStepLimitResult const& limit, bool linear) {
      auto const momentumGeV =
          ::sqrt((particle.energyGeV_ - massGeV) * (particle.energyGeV_ + massGeV));
      auto const speedMPerS =
          scalar_constants::SpeedOfLightMPerS * momentumGeV / particle.energyGeV_;
      auto const chargeC = chargeNumber * scalar_constants::ElementaryChargeC;
      auto const momentumKgMPerS = momentumGeV * scalar_constants::ElementaryChargeC /
                                   scalar_constants::MagneticRigidityGeVPerTeslaMeter;
      double velocityMPerS[3];
      for (int axis = 0; axis < 3; ++axis) {
        velocityMPerS[axis] = speedMPerS * particle.direction_[axis];
      }
      return CurvedBoundaryAccess<TTreeView>(
          tree, particle.positionM_, velocityMPerS, particle.direction_, fieldT,
          speedMPerS, momentumKgMPerS, chargeC, limit.gyroradiusM_, linear);
    }

  } // namespace lepton_boundary_detail

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION CurvedBoundaryAccess<TTreeView> lepton_access(
      TTreeView const& tree, ParticleState const& particle, double massGeV,
      double chargeNumber, double const (&fieldT)[3],
      MagneticStepLimitResult const& limit) {
    return lepton_boundary_detail::make_access(
        tree, particle, massGeV, chargeNumber, fieldT, limit,
        limit.status_ == MagneticStepStatus::Linear);
  }

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION LeptonBoundary
  lepton_boundary(TTreeView const& tree, ParticleState const& particle, double massGeV,
                  double chargeNumber, double const (&fieldT)[3],
                  MagneticStepLimitResult const& limit) {
    auto const access =
        lepton_access(tree, particle, massGeV, chargeNumber, fieldT, limit);
    auto const result = next_environment_boundary(
        access, particle.environmentNode_, limit.distanceM_ / access.getSpeedMPerS());
    return {result.getParameter() * access.getSpeedMPerS(), result.getNode(),
            access.hasUnsupportedShape() || access.hasIncompleteRoots()};
  }

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION double lepton_observation_distance(
      TTreeView const& tree, ParticleState const& particle, double massGeV,
      double chargeNumber, double const (&fieldT)[3],
      MagneticStepLimitResult const& limit, double const (&pointM)[3],
      double const (&normal)[3]) {
    auto const access = lepton_boundary_detail::make_access(
        tree, particle, massGeV, chargeNumber, fieldT, limit, false);
    EnvironmentShape plane;
    plane.kind_ = EnvironmentShape::Kind::SeparationPlane;
    for (int axis = 0; axis < 3; ++axis) {
      plane.centerM_[axis] = pointM[axis];
      plane.normal_[axis] = normal[axis];
    }
    auto const intersection = access.intersectShape(plane);
    return intersection.getEntry() > 0. ? intersection.getEntry() * access.getSpeedMPerS()
                                        : straight_intersection_detail::infinity();
  }

} // namespace corsika::kokkos
