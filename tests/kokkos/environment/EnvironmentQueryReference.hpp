/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "EnvironmentSearchReference.hpp"
#include "ModelGrammageReference.hpp"
#include "StraightBoundaryReference.hpp"

#include <corsika/kokkos/environment/EnvironmentComponent.hpp>
#include <Kokkos_Macros.hpp>
#include <cmath>
#include <cstdint>
#include <limits>

namespace environment_query_reference {

  // Independent NodeTree-only transcription of production 41cdc543:
  // EnvironmentModelSnapshot.hpp, SphericalAtmosphere.hpp, ObservationPlane.hpp
  // and ExternalTransportBoundary.hpp. Geometry/model primitives below are the
  // previously frozen independent oracles, not the migrated query functions.
  using model_type = corsika::kokkos::EnvironmentModel;
  using component_type = corsika::kokkos::EnvironmentComponent;

  enum class Status : std::uint32_t {
    Success = 0,
    InvalidSnapshot = 1,
    NonFiniteInput = 2,
    OutsideEnvironment = 3,
    BelowObservationSurface = 4,
    NoForwardIntersection = 5,
    GrammageOutOfRange = 6
  };
  enum class ObservationStatus : std::uint32_t {
    Success = 0,
    NoForwardIntersection = 1,
    InvalidInput = 2,
    NonFiniteResult = 3
  };
  struct NodeQuery {
    Status status_{Status::InvalidSnapshot};
    std::int32_t node_{-1};
    double radiusM_{};
    double densityGPerCm3_{};
  };
  struct BoundaryQuery {
    Status status_{Status::InvalidSnapshot};
    std::int32_t node_{-1};
    double distanceM_{};
    double radiusM_{};
    std::int32_t nextNode_{-3};
  };
  struct IntegralQuery {
    Status status_{Status::InvalidSnapshot};
    std::uint32_t reserved_{};
    double value_{};
  };
  struct ObservationQuery {
    ObservationStatus status_{ObservationStatus::InvalidInput};
    std::uint32_t reserved_{};
    double distanceM_{};
    double residualM_{};
  };

  struct FullView {
    environment_search_reference::Tree tree_;
    model_type const* models_{};
    component_type const* composition_{};
    unsigned int modelCount_{};
    unsigned int compositionCount_{};
    unsigned int childCount_{};
    unsigned int exclusionCount_{};
    double maximumDeflection_{.2};

    KOKKOS_INLINE_FUNCTION model_type const* model(std::int32_t node) const {
      if (!tree_.nodes_ || !models_ || node < 0 ||
          static_cast<unsigned int>(node) >= tree_.count_) {
        return nullptr;
      }
      auto index = tree_.nodes_[node].modelIndex_;
      return index >= 0 && static_cast<unsigned int>(index) < modelCount_
                 ? models_ + index
                 : nullptr;
    }
  };

  KOKKOS_INLINE_FUNCTION bool finite(double value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }
  KOKKOS_INLINE_FUNCTION double absolute(double value) {
#if defined(__CUDA_ARCH__)
    return ::fabs(value);
#else
    return std::abs(value);
#endif
  }
  KOKKOS_INLINE_FUNCTION double infinity() {
#if defined(__CUDA_ARCH__)
    return CUDART_INF;
#else
    return std::numeric_limits<double>::infinity();
#endif
  }
  KOKKOS_INLINE_FUNCTION double dot(double const* left, double const* right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
  }
  KOKKOS_INLINE_FUNCTION bool normalized(double const* direction) {
    auto const normSquared = dot(direction, direction);
    return finite(normSquared) && absolute(normSquared - 1.) <= 1.e-12;
  }
  KOKKOS_INLINE_FUNCTION bool valid(FullView const& environment) {
    auto const& tree = environment.tree_;
    return tree.count_ > 0 && tree.nodes_ &&
           (!environment.childCount_ || tree.children_) &&
           (!environment.exclusionCount_ || tree.exclusions_) &&
           environment.modelCount_ > 0 && environment.models_ &&
           finite(environment.maximumDeflection_) &&
           environment.maximumDeflection_ > 0. &&
           environment.maximumDeflection_ < .5 * 3.141592653589793;
  }
  KOKKOS_INLINE_FUNCTION NodeQuery query(FullView const& environment,
                                         double const* position,
                                         double const* direction = nullptr,
                                         std::int32_t logicalNode = -3) {
    if (!valid(environment)) { return {Status::InvalidSnapshot, -1, 0., 0.}; }
    for (int axis = 0; axis < 3; ++axis) {
      if (!finite(position[axis]) || (direction != nullptr && !finite(direction[axis]))) {
        return {Status::NonFiniteInput, -1, 0., 0.};
      }
    }
    if (direction != nullptr && !normalized(direction)) {
      return {Status::NonFiniteInput, -1, 0., 0.};
    }
    auto const node = logicalNode == -3 ? environment_search_reference::locate(
                                              environment.tree_, position)
                                        : logicalNode;
    auto model = environment.model(node);
    if (!model) { return {Status::OutsideEnvironment, node, 0., 0.}; }
    auto rho = model_evaluation_reference::density(*model, position);
    if (!finite(rho) || !(rho > 0.)) { return {Status::InvalidSnapshot, node, 0., rho}; }
    return {Status::Success, node, 0., rho};
  }
  KOKKOS_INLINE_FUNCTION BoundaryQuery boundary(FullView const& environment,
                                                double const* position,
                                                double const* direction,
                                                std::int32_t logicalNode = -3) {
    auto const node = query(environment, position, direction, logicalNode);
    if (node.status_ != Status::Success) { return {node.status_, -1, 0., 0.}; }
    straight_boundary_reference::Tree access;
    access.nodes_ = environment.tree_.nodes_;
    access.shapes_ = environment.tree_.shapes_;
    access.children_ = environment.tree_.children_;
    access.exclusions_ = environment.tree_.exclusions_;
    for (int axis = 0; axis < 3; ++axis) {
      access.position_[axis] = position[axis];
      access.velocity_[axis] = direction[axis];
    }
    auto result = boundary_reference::next(access, node.node_, infinity());
    if (!finite(result.parameter_)) {
      return {Status::NoForwardIntersection, node.node_, 0., 0., result.node_};
    }
    return {Status::Success, node.node_, result.parameter_, 0., result.node_};
  }
  KOKKOS_INLINE_FUNCTION IntegralQuery grammage(FullView const& environment,
                                                std::int32_t node, double const* position,
                                                double const* direction, double length) {
    auto model = environment.model(node);
    if (!valid(environment) || !model || !finite(length) || length < 0. ||
        !normalized(direction)) {
      return {Status::NonFiniteInput, 0, 0.};
    }
    auto value = model_grammage_reference::grammage(*model, position, direction, length);
    return {finite(value) && value >= 0. ? Status::Success : Status::GrammageOutOfRange,
            0, value};
  }
  KOKKOS_INLINE_FUNCTION IntegralQuery arclength(FullView const& environment,
                                                 std::int32_t node,
                                                 double const* position,
                                                 double const* direction,
                                                 double grammageValue) {
    auto model = environment.model(node);
    if (!valid(environment) || !model || !finite(grammageValue) || grammageValue < 0. ||
        !normalized(direction)) {
      return {Status::NonFiniteInput, 0, 0.};
    }
    auto value =
        model_grammage_reference::arclength(*model, position, direction, grammageValue);
    return {finite(value) && value >= 0. ? Status::Success : Status::GrammageOutOfRange,
            0, value};
  }
  KOKKOS_INLINE_FUNCTION IntegralQuery competition(FullView const& environment,
                                                   std::int32_t node,
                                                   double const* position,
                                                   double const* direction,
                                                   double length) {
    auto result = grammage(environment, node, position, direction, length);
    if (result.status_ == Status::GrammageOutOfRange && result.value_ == HUGE_VAL &&
        finite(length) && length > 0.) {
      auto model = environment.model(node);
      if (model &&
          (model->densityKind_ == model_type::Density::SlidingPlanarExponential ||
           model->densityKind_ == model_type::Density::FlatExponential)) {
        auto const rho = model_evaluation_reference::density(*model, position);
        auto const projection =
            model_grammage_reference::projected_axis(*model, position, direction);
        auto const growth = projection * model->inverseScaleLengthPerM_;
        if (finite(rho) && rho > 0. && finite(growth) && growth > 0. &&
            finite(model->scaleLengthM_) &&
            model->scaleLengthM_ * model->inverseScaleLengthPerM_ > 0.) {
          result.status_ = Status::Success;
        }
      }
    }
    return result;
  }
  KOKKOS_INLINE_FUNCTION ObservationQuery observation(double const* planePoint,
                                                      double const* normal,
                                                      double const* position,
                                                      double const* direction,
                                                      double maximumDistance) {
    auto normalSquared = 0.;
    for (int axis = 0; axis < 3; ++axis) {
      if (!finite(planePoint[axis]) || !finite(normal[axis])) { return {}; }
      normalSquared += normal[axis] * normal[axis];
    }
    if (!(finite(normalSquared) && absolute(normalSquared - 1.) <= 1.e-12) ||
        (!finite(maximumDistance) && maximumDistance != infinity())) {
      return {};
    }
    double displacement[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      if (!finite(position[axis]) || !finite(direction[axis])) { return {}; }
      displacement[axis] = planePoint[axis] - position[axis];
    }
    auto const denominator = dot(normal, direction);
    if (denominator == 0.) {
      return {ObservationStatus::NoForwardIntersection, 0, infinity(), 0.};
    }
    auto const distance = dot(normal, displacement) / denominator;
    if (!finite(distance)) { return {ObservationStatus::NonFiniteResult, 0, 0., 0.}; }
    if (!(distance > 0.) || (finite(maximumDistance) && distance > maximumDistance)) {
      return {ObservationStatus::NoForwardIntersection, 0, infinity(), 0.};
    }
    return {ObservationStatus::Success, 0, distance, 0.};
  }

} // namespace environment_query_reference
