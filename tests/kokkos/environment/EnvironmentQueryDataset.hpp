/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/kokkos/data/ParticleState.hpp>
#include <corsika/kokkos/environment/EnvironmentNode.hpp>
#include <corsika/kokkos/environment/EnvironmentShape.hpp>
#include <corsika/kokkos/environment/EnvironmentModel.hpp>
#include <corsika/kokkos/environment/EnvironmentComponent.hpp>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace environment_query_test {

  using node_type = corsika::kokkos::EnvironmentNode;
  using shape_type = corsika::kokkos::EnvironmentShape;
  using model_type = corsika::kokkos::EnvironmentModel;
  using component_type = corsika::kokkos::EnvironmentComponent;
  using particle_type = corsika::kokkos::ParticleState;

  enum class Category {
    Ordinary,
    Assigned,
    MissingModel,
    InvalidConfig,
    InvalidDirection,
    NonFinitePosition,
    Zero,
    Negative,
    NonFiniteValue,
    InvalidDensity,
    OverflowSentinel,
    RejectedSentinel,
    Tie,
    Observation,
    Empty,
    Count
  };

  struct Query {
    double positionM_[3]{0., 0., 10.};
    double direction_[3]{0., 0., 1.};
    std::int32_t logicalNode_{1};
    std::int32_t integralNode_{1};
    double lengthM_{5.};
    double grammageGPerCm2_{.5};
    double maximumDeflection_{.2};
    bool includeDirection_{true};
    double pointM_[3]{0., 0., 20.};
    double normal_[3]{0., 0., 1.};
    double maximumObservationM_{std::numeric_limits<double>::infinity()};
    Category category_{Category::Ordinary};
    // All queries have a strict same-device production oracle. Cross-device
    // rounding is tested only for the stated well-conditioned finite subset.
    bool crossDevice_{true};
  };

  struct IntegrationQuery {
    particle_type particle_;
    double massGeV_{.105658};
    double chargeNumber_{1.};
    double lengthM_{25.};
    std::int32_t nativeNode_{-1};
    double nativeDensityGPerCm3_{};
    double nativeFieldT_[3]{};
  };

  struct Dataset {
    std::string name_;
    std::vector<node_type> nodes_;
    std::vector<shape_type> shapes_;
    std::vector<std::int32_t> children_;
    std::vector<std::int32_t> exclusions_;
    std::vector<model_type> models_;
    std::vector<component_type> components_;
    std::vector<Query> queries_;
    std::vector<IntegrationQuery> integration_;
  };

  // Defined in a host-only TU using the actual five-layer native environment,
  // native node/density/field queries and the real EnvironmentSnapshot exporter.
  Dataset atmosphere();

  CORSIKA_HOST_DEVICE inline Query integration_query(IntegrationQuery const& input) {
    Query result;
    for (int axis = 0; axis < 3; ++axis) {
      result.positionM_[axis] = input.particle_.positionM_[axis];
      result.direction_[axis] = input.particle_.direction_[axis];
      result.pointM_[axis] =
          result.positionM_[axis] + 2. * input.lengthM_ * result.direction_[axis];
      result.normal_[axis] = result.direction_[axis];
    }
    result.logicalNode_ = -3;
    result.integralNode_ = input.nativeNode_;
    result.lengthM_ = input.lengthM_;
    return result;
  }

} // namespace environment_query_test
