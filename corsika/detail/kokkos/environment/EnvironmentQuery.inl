/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentSearch.hpp>
#include <corsika/kokkos/environment/ModelGrammage.hpp>
#include <corsika/kokkos/environment/StraightBoundaryAccess.hpp>

#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#include <math_constants.h>
#endif

namespace corsika::kokkos {

  namespace environment_query_detail {

    CORSIKA_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double absolute(double value) {
#if defined(__CUDA_ARCH__)
      return ::fabs(value);
#else
      return std::abs(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double infinity() {
#if defined(__CUDA_ARCH__)
      return CUDART_INF;
#else
      return std::numeric_limits<double>::infinity();
#endif
    }

    CORSIKA_HOST_DEVICE inline double dot(double const left[3], double const right[3]) {
      return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    }

    CORSIKA_HOST_DEVICE inline bool normalized(double const direction[3]) {
      auto const normSquared = dot(direction, direction);
      return finite(normSquared) && absolute(normSquared - 1.) <= 1.e-12;
    }

    template <typename TEnvironmentView>
    CORSIKA_HOST_DEVICE inline bool valid_environment(
        TEnvironmentView const& environment, EnvironmentQueryConfig const& config) {
      auto const& tree = environment.getTree();
      return tree.getNodeCount() > 0 && tree.hasNodeStorage() &&
             tree.hasAdjacencyStorage() && environment.getModelCount() > 0 &&
             environment.hasModelStorage() &&
             finite(config.maximumMagneticDeflectionRad_) &&
             config.maximumMagneticDeflectionRad_ > 0. &&
             config.maximumMagneticDeflectionRad_ < 0.5 * 3.141592653589793;
    }

  } // namespace environment_query_detail

  template <typename TEnvironmentView>
  CORSIKA_HOST_DEVICE inline EnvironmentLayerQuery query_environment(
      TEnvironmentView const& environment, EnvironmentQueryConfig const& config,
      double const (&positionM)[3], double const* direction, std::int32_t logicalNode) {
    using namespace environment_query_detail;
    if (!valid_environment(environment, config)) {
      return {EnvironmentStatus::InvalidSnapshot, -1, 0., 0.};
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (!finite(positionM[axis]) ||
          (direction != nullptr && !finite(direction[axis]))) {
        return {EnvironmentStatus::NonFiniteInput, -1, 0., 0.};
      }
    }
    if (direction != nullptr && !normalized(direction)) {
      return {EnvironmentStatus::NonFiniteInput, -1, 0., 0.};
    }
    auto const& tree = environment;
    auto const node =
        logicalNode == -3 ? locate_node(tree.getTree(), positionM) : logicalNode;
    auto model = tree.getModel(node);
    if (!model) { return {EnvironmentStatus::OutsideEnvironment, node, 0., 0.}; }
    auto rho = density(*model, positionM);
    if (!finite(rho) || !(rho > 0.)) {
      return {EnvironmentStatus::InvalidSnapshot, node, 0., rho};
    }
    return {EnvironmentStatus::Success, node, 0., rho};
  }

  template <typename TEnvironmentView>
  CORSIKA_HOST_DEVICE inline EnvironmentBoundaryQuery distance_to_environment_boundary(
      TEnvironmentView const& environment, EnvironmentQueryConfig const& config,
      double const (&positionM)[3], double const (&direction)[3],
      std::int32_t logicalNode) {
    using namespace environment_query_detail;
    auto const layerQuery =
        query_environment(environment, config, positionM, direction, logicalNode);
    if (layerQuery.status_ != EnvironmentStatus::Success) {
      return {layerQuery.status_, -1, 0., 0.};
    }
    StraightBoundaryAccess<typename TEnvironmentView::tree_type> access(
        environment.getTree(), positionM, direction);
    auto boundary = next_environment_boundary(access, layerQuery.nodeIndex_, infinity());
    if (!finite(boundary.getParameter())) {
      return {EnvironmentStatus::NoForwardIntersection, layerQuery.nodeIndex_, 0., 0.,
              boundary.getNode()};
    }
    return {EnvironmentStatus::Success, layerQuery.nodeIndex_, boundary.getParameter(),
            0., boundary.getNode()};
  }

  template <typename TEnvironmentView>
  CORSIKA_HOST_DEVICE inline EnvironmentGrammageQuery environment_grammage(
      TEnvironmentView const& environment, EnvironmentQueryConfig const& config,
      std::int32_t nodeIndex, double const (&positionM)[3], double const (&direction)[3],
      double distanceM) {
    using namespace environment_query_detail;
    auto model = environment.getModel(nodeIndex);
    if (!valid_environment(environment, config) || !model || !finite(distanceM) ||
        distanceM < 0. || !normalized(direction)) {
      return {EnvironmentStatus::NonFiniteInput, 0, 0.};
    }
    auto x = grammage(*model, positionM, direction, distanceM);
    return {finite(x) && x >= 0. ? EnvironmentStatus::Success
                                 : EnvironmentStatus::GrammageOutOfRange,
            0, x};
  }

  template <typename TEnvironmentView>
  CORSIKA_HOST_DEVICE inline EnvironmentGrammageQuery environment_arclength(
      TEnvironmentView const& environment, EnvironmentQueryConfig const& config,
      std::int32_t nodeIndex, double const (&positionM)[3], double const (&direction)[3],
      double grammageGPerCm2) {
    using namespace environment_query_detail;
    auto model = environment.getModel(nodeIndex);
    if (!valid_environment(environment, config) || !model || !finite(grammageGPerCm2) ||
        grammageGPerCm2 < 0. || !normalized(direction)) {
      return {EnvironmentStatus::NonFiniteInput, 0, 0.};
    }
    auto length = arclength(*model, positionM, direction, grammageGPerCm2);
    return {finite(length) && length >= 0. ? EnvironmentStatus::Success
                                           : EnvironmentStatus::GrammageOutOfRange,
            0, length};
  }

  template <typename TEnvironmentView>
  CORSIKA_HOST_DEVICE inline EnvironmentGrammageQuery geometry_competition_grammage(
      TEnvironmentView const& environment, EnvironmentQueryConfig const& config,
      std::int32_t nodeIndex, double const (&positionM)[3], double const (&direction)[3],
      double distanceM) {
    auto result = environment_grammage(environment, config, nodeIndex, positionM,
                                       direction, distanceM);
    if (result.status_ == EnvironmentStatus::GrammageOutOfRange &&
        result.value_ == HUGE_VAL && environment_query_detail::finite(distanceM) &&
        distanceM > 0.) {
      auto const* model = environment.getModel(nodeIndex);
      if (model &&
          (model->densityKind_ == EnvironmentModel::Density::SlidingPlanarExponential ||
           model->densityKind_ == EnvironmentModel::Density::FlatExponential)) {
        auto const rho = density(*model, positionM);
        auto const projection =
            model_grammage_detail::projected_axis(*model, positionM, direction);
        auto const growth = projection * model->inverseScaleLengthPerM_;
        // This is only a comparison sentinel for a finite geometry candidate.
        // The selected physical step still needs a finite traversed depth.
        if (environment_query_detail::finite(rho) && rho > 0. &&
            environment_query_detail::finite(growth) && growth > 0. &&
            environment_query_detail::finite(model->scaleLengthM_) &&
            model->scaleLengthM_ * model->inverseScaleLengthPerM_ > 0.) {
          result.status_ = EnvironmentStatus::Success;
        }
      }
    }
    return result;
  }

} // namespace corsika::kokkos
