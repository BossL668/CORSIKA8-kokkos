/* Distributed under the 3-clause BSD license; see LICENSE. */
#pragma once
#include <corsika/accelerator/em/common/EnvironmentCurvedSnapshot.hpp>
#include <corsika/accelerator/em/common/UniformMagneticField.hpp>

namespace corsika::gpu::em::environment_tree {
  struct LeptonBoundary {
    double distance_m{};
    NodeId next_node{-3};
    bool failed{};
  };

  C8_ACCELERATOR_INLINE_FUNCTION inline CurvedAccess leptonAccess(
      FullView tree, EmParticleState const& particle, double mass_GeV,
      double charge_number, double const field_T[3],
      MagneticStepLimitResult const& limit) {
    auto const p_GeV=::sqrt((particle.energy_GeV-mass_GeV)*
                            (particle.energy_GeV+mass_GeV));
    CurvedAccess access;
    static_cast<TreeView&>(access)=tree.topology;
    access.speed=accelerator::scalar_constants::SpeedOfLightMPerS*p_GeV/particle.energy_GeV;
    access.charge_SI=charge_number*accelerator::scalar_constants::ElementaryChargeC;
    access.momentum_SI=p_GeV*accelerator::scalar_constants::ElementaryChargeC/GeVPerCToTeslaMeter;
    access.gyroradius_m=limit.gyroradius_m;
    access.linear=limit.status==MagneticStepStatus::Linear;
    for (int a=0;a<3;++a) {
      access.position[a]=particle.position_m[a];
      access.direction[a]=particle.direction[a];
      access.velocity[a]=access.speed*particle.direction[a];
      access.field[a]=field_T[a];
    }
    return access;
  }

  // C8 searches in time coordinates. The transport step uses the corresponding
  // leapfrog length, not the endpoint chord. Keep the conversion in one place.
  C8_ACCELERATOR_INLINE_FUNCTION inline LeptonBoundary leptonBoundary(
      FullView tree, EmParticleState const& particle, double mass_GeV,
      double charge_number, double const field_T[3],
      MagneticStepLimitResult const& limit) {
    auto const access=leptonAccess(tree,particle,mass_GeV,charge_number,field_T,limit);
    auto const result=geometry_detail::nextEnvironmentBoundary(
        access,particle.environment_node,limit.distance_m/access.speed);
    return {result.distance*access.speed,result.node,
            access.unsupported_shape || access.incomplete_roots};
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double leptonObservationDistance(
      EnvironmentSnapshot const& environment,EmParticleState const& particle,
      double mass_GeV,double charge_number,double const field_T[3],
      MagneticStepLimitResult const& limit) {
    auto access=leptonAccess(environment.node_tree,particle,mass_GeV,charge_number,field_T,limit);
    // Native ObservationPlane calls the tracking class's plane intersection,
    // even if getTrack selected a linear trajectory for this particular step.
    access.linear=false;
    Node plane;
    plane.shape=Shape::SeparationPlane;
    for(int axis=0;axis<3;++axis) {
      plane.center[axis]=environment.observation_plane_point_m[axis];
      plane.axes[0][axis]=environment.observation_plane_normal[axis];
    }
    auto const intersection=access.intersectShape(plane);
    return intersection.entry>0.?intersection.entry*access.speed:infinity();
  }
}
