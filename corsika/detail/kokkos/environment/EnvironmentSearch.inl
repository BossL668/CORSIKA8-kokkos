/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  namespace environment_search_detail {

    // Preserve the production CUDA dot-product rounding without global flags.
    CORSIKA_HOST_DEVICE inline double product(double first, double second) {
#if defined(__CUDA_ARCH__)
      return __dmul_rn(first, second);
#else
      return first * second;
#endif
    }

    CORSIKA_HOST_DEVICE inline double sum(double first, double second) {
#if defined(__CUDA_ARCH__)
      return __dadd_rn(first, second);
#else
      return first + second;
#endif
    }

    CORSIKA_HOST_DEVICE inline double dot(double const (&first)[3],
                                          double const (&second)[3]) {
      return sum(sum(product(first[0], second[0]), product(first[1], second[1])),
                 product(first[2], second[2]));
    }

  } // namespace environment_search_detail

  CORSIKA_HOST_DEVICE inline bool contains(EnvironmentShape const& shape,
                                           double const (&positionM)[3]) {
    if (shape.kind_ == EnvironmentShape::Kind::Universe) { return true; }
    double delta[3]{positionM[0] - shape.centerM_[0], positionM[1] - shape.centerM_[1],
                    positionM[2] - shape.centerM_[2]};
    if (shape.kind_ == EnvironmentShape::Kind::Sphere) {
      return environment_search_detail::dot(delta, delta) <
             shape.radiusM_ * shape.radiusM_;
    }
    if (shape.kind_ == EnvironmentShape::Kind::SeparationPlane) {
      return environment_search_detail::dot(delta, shape.normal_) <= 0.;
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (!(::fabs(environment_search_detail::dot(delta, shape.axes_[axis])) <
            shape.halfExtentM_[axis])) {
        return false;
      }
    }
    return true;
  }

  template <typename TTreeView>
  CORSIKA_HOST_DEVICE inline std::int32_t locate_node(TTreeView const& tree,
                                                      double const (&positionM)[3],
                                                      std::int32_t start) {
    auto current = start;
    for (std::uint32_t hop = 0; hop < tree.getNodeCount(); ++hop) {
      if (current < 0 || static_cast<std::uint32_t>(current) >= tree.getNodeCount()) {
        return -2;
      }
      if (!contains(tree.getShape(current), positionM)) { return -1; }
      std::int32_t next = -1;
      for (std::uint32_t child = 0; child < tree.getChildCount(current); ++child) {
        auto const childNode = tree.getChild(current, child);
        if (contains(tree.getShape(childNode), positionM)) {
          next = childNode;
          break;
        }
      }
      if (next == -1) {
        for (std::uint32_t excluded = 0; excluded < tree.getExcludedCount(current);
             ++excluded) {
          auto const excludedNode = tree.getExcluded(current, excluded);
          if (contains(tree.getShape(excludedNode), positionM)) {
            next = excludedNode;
            break;
          }
        }
      }
      if (next == -1) { return current; }
      current = next;
    }
    return -2;
  }

} // namespace corsika::kokkos
