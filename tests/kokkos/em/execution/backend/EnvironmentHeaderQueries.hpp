/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/environment/EnvironmentBorrowedView.hpp>
#include <corsika/kokkos/environment/EnvironmentQuery.hpp>
#include <corsika/kokkos/environment/ObservationPlane.hpp>

namespace backend_test {
  struct EnvironmentResults {
    corsika::kokkos::EnvironmentLayerQuery layer_;
    corsika::kokkos::EnvironmentBoundaryQuery boundary_;
    corsika::kokkos::EnvironmentGrammageQuery grammage_;
    corsika::kokkos::EnvironmentGrammageQuery length_;
    corsika::kokkos::ObservationPlaneIntersection observation_;
  };

  CORSIKA_HOST_DEVICE inline EnvironmentResults query_header_environment(
      corsika::kokkos::EnvironmentDataView const& data, unsigned caseIndex) {
    namespace cp = corsika::kokkos;
    cp::EnvironmentBorrowedView const view(data);
    double const position[3]{caseIndex == 0 ? 0. : 10., caseIndex == 0 ? 0. : 20.,
                             caseIndex == 0 ? 0. : 30.};
    double const direction[3]{caseIndex == 0 ? 0. : .6, 0., caseIndex == 0 ? -1. : -.8};
    cp::EnvironmentQueryConfig config;
    cp::ObservationPlane plane;
    plane.pointM_[2] = -100.;
    plane.normal_[2] = 1.;
    EnvironmentResults result;
    result.layer_ = cp::query_environment(view, config, position, direction);
    result.boundary_ =
        cp::distance_to_environment_boundary(view, config, position, direction);
    result.grammage_ = cp::environment_grammage(view, config, result.layer_.nodeIndex_,
                                                position, direction, 100.);
    result.length_ =
        cp::environment_arclength(view, config, result.layer_.nodeIndex_, position,
                                  direction, result.grammage_.value_);
    result.observation_ =
        cp::intersect_observation_plane_straight(plane, position, direction);
    return result;
  }

  // Defined in the real execution-space translation unit, nvcc for CUDA.
  void query_execution_environment(corsika::kokkos::EnvironmentDataView const& data,
                                   EnvironmentResults (&results)[2]);
} // namespace backend_test
