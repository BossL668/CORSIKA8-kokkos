/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentNode.hpp>
#include <corsika/kokkos/environment/EnvironmentShape.hpp>

#include <corsika/framework/core/HostDevice.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace environment_search_reference {

  using node_type = corsika::kokkos::EnvironmentNode;
  using shape_type = corsika::kokkos::EnvironmentShape;

  struct Tree {
    node_type const* nodes_{};
    shape_type const* shapes_{};
    std::int32_t const* children_{};
    std::int32_t const* exclusions_{};
    std::uint32_t count_{};
  };

  // Independent transcription of EnvironmentTreeSnapshot.hpp at production
  // 41cdc543429b82a216857ad458f83deb0f210a5f. Only record field names change;
  // no new search, view accessors or arithmetic helpers are used by this oracle.
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

  CORSIKA_HOST_DEVICE inline double dot(double const* first, double const* second) {
    return sum(sum(product(first[0], second[0]), product(first[1], second[1])),
               product(first[2], second[2]));
  }

  CORSIKA_HOST_DEVICE inline bool contains(shape_type const& shape,
                                           double const* position) {
    if (shape.kind_ == shape_type::Kind::Universe) { return true; }
    double delta[3]{position[0] - shape.centerM_[0], position[1] - shape.centerM_[1],
                    position[2] - shape.centerM_[2]};
    if (shape.kind_ == shape_type::Kind::Sphere) {
      return dot(delta, delta) < shape.radiusM_ * shape.radiusM_;
    }
    if (shape.kind_ == shape_type::Kind::SeparationPlane) {
      return dot(delta, shape.normal_) <= 0.;
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (!(::fabs(dot(delta, shape.axes_[axis])) < shape.halfExtentM_[axis])) {
        return false;
      }
    }
    return true;
  }

  CORSIKA_HOST_DEVICE inline std::int32_t locate(Tree const& tree, double const* position,
                                                 std::int32_t start = 0) {
    auto current = start;
    for (unsigned hop = 0; hop < tree.count_; ++hop) {
      if (current < 0 || static_cast<unsigned>(current) >= tree.count_) { return -2; }
      auto const& node = tree.nodes_[current];
      if (!contains(tree.shapes_[current], position)) { return -1; }
      std::int32_t next = -1;
      for (unsigned index = 0; index < node.childCount_; ++index) {
        auto const child = tree.children_[node.childOffset_ + index];
        if (contains(tree.shapes_[child], position)) {
          next = child;
          break;
        }
      }
      if (next == -1) {
        for (unsigned index = 0; index < node.excludedCount_; ++index) {
          auto const excluded = tree.exclusions_[node.excludedOffset_ + index];
          if (contains(tree.shapes_[excluded], position)) {
            next = excluded;
            break;
          }
        }
      }
      if (next == -1) { return current; }
      current = next;
    }
    return -2;
  }

  struct Query {
    double positionM_[3]{};
    std::int32_t start_{};
  };

  struct Dataset {
    char const* name_{};
    std::vector<node_type> nodes_;
    std::vector<shape_type> shapes_;
    std::vector<std::int32_t> children_;
    std::vector<std::int32_t> exclusions_;
    std::vector<Query> queries_;

    Tree tree() const {
      return {nodes_.data(), shapes_.data(), children_.data(), exclusions_.data(),
              static_cast<std::uint32_t>(nodes_.size())};
    }
  };

  inline shape_type sphere(double radius) {
    shape_type result;
    result.kind_ = shape_type::Kind::Sphere;
    result.radiusM_ = radius;
    return result;
  }

  inline shape_type box(double x, double y, double z) {
    shape_type result;
    result.kind_ = shape_type::Kind::Box;
    result.halfExtentM_[0] = x;
    result.halfExtentM_[1] = y;
    result.halfExtentM_[2] = z;
    for (int axis = 0; axis < 3; ++axis) { result.axes_[axis][axis] = 1.; }
    return result;
  }

  inline Dataset branching() {
    Dataset result;
    result.name_ = "ordered children and directed exclusions";
    result.nodes_.resize(7);
    result.shapes_ = {sphere(0.),  sphere(100.), box(10., 10., 10.), sphere(2.),
                      sphere(20.), sphere(6.),   sphere(9.)};
    result.shapes_[0].kind_ = shape_type::Kind::Universe;
    int const parents[]{-1, 0, 1, 2, 1, 0, 0};
    for (int index = 0; index < 7; ++index) {
      result.nodes_[index].parentIndex_ = parents[index];
    }
    result.nodes_[0].childCount_ = 3;
    result.nodes_[1].childOffset_ = 3;
    result.nodes_[1].childCount_ = 2;
    result.nodes_[1].modelIndex_ = 0;
    result.nodes_[2].childOffset_ = 5;
    result.nodes_[2].childCount_ = 1;
    result.nodes_[2].excludedCount_ = 3;
    result.nodes_[5].modelIndex_ = 0;
    result.nodes_[6].excludedOffset_ = 3;
    result.nodes_[6].excludedCount_ = 1;
    result.children_ = {1, 5, 6, 2, 4, 3};
    result.exclusions_ = {5, 6, 5, 5};
    for (int x = -20; x <= 20; ++x) {
      for (double y : {-8.5, 0., 8.5}) {
        for (double z : {-8.5, 0., 8.5}) {
          for (int start : {0, 1, 2, 5, 6}) {
            result.queries_.push_back({{1.25 * x, y, z}, start});
          }
        }
      }
    }
    for (double x : {0., 5., 8., 9.5, 15., 50., 120.}) {
      result.queries_.push_back({{x, 0., 0.}, 0});
    }
    for (int start : {-1, 7, std::numeric_limits<std::int32_t>::max()}) {
      result.queries_.push_back({{0., 0., 0.}, start});
    }
    return result;
  }

  inline std::vector<Dataset> datasets() {
    std::vector<Dataset> result{branching()};
    Dataset deep;
    deep.name_ = "602-node ancestry";
    deep.nodes_.resize(602);
    deep.shapes_.resize(602);
    for (std::int32_t index = 0; index < 602; ++index) {
      deep.nodes_[index].parentIndex_ = index - 1;
      deep.nodes_[index].childOffset_ = index;
      deep.shapes_[index] = sphere(1000. - index);
      if (index < 601) {
        deep.nodes_[index].childCount_ = 1;
        deep.children_.push_back(index + 1);
      }
    }
    deep.shapes_[0].kind_ = shape_type::Kind::Universe;
    deep.queries_ = {{{0., 0., 0.}, 0},
                     {{10000., 0., 0.}, 0},
                     {{10000., 0., 0.}, 1},
                     {{0., 0., 0.}, 600}};
    result.push_back(deep);

    Dataset cycle;
    cycle.name_ = "reachable exclusion cycle";
    cycle.nodes_.resize(3);
    cycle.shapes_ = {sphere(0.), sphere(10.), sphere(10.)};
    cycle.shapes_[0].kind_ = shape_type::Kind::Universe;
    cycle.nodes_[0].childCount_ = 2;
    cycle.nodes_[1].parentIndex_ = cycle.nodes_[2].parentIndex_ = 0;
    cycle.nodes_[1].excludedCount_ = cycle.nodes_[2].excludedCount_ = 1;
    cycle.nodes_[2].excludedOffset_ = 1;
    cycle.children_ = {1, 2};
    cycle.exclusions_ = {2, 1};
    cycle.queries_ = {{{0., 0., 0.}, 0}, {{20., 0., 0.}, 0}, {{20., 0., 0.}, 1}};
    result.push_back(cycle);

    Dataset self;
    self.name_ = "self exclusion";
    self.nodes_.resize(1);
    self.nodes_[0].excludedCount_ = 1;
    self.shapes_ = {sphere(0.)};
    self.shapes_[0].kind_ = shape_type::Kind::Universe;
    self.exclusions_ = {0};
    self.queries_ = {{{0., 0., 0.}, 0}};
    result.push_back(self);

    Dataset empty;
    empty.name_ = "empty view";
    empty.queries_ = {{{0., 0., 0.}, 0}, {{0., 0., 0.}, -1}};
    result.push_back(empty);

    double const infinity = std::numeric_limits<double>::infinity();
    double const nan = std::numeric_limits<double>::quiet_NaN();
    for (auto kind : {shape_type::Kind::Universe, shape_type::Kind::Sphere,
                      shape_type::Kind::Box, shape_type::Kind::SeparationPlane}) {
      Dataset single;
      single.name_ = "primitive boundaries and IEEE inputs";
      single.nodes_.resize(1);
      auto shape = box(1., 2., 3.);
      shape.kind_ = kind;
      shape.radiusM_ = 1.;
      shape.normal_[2] = 1.;
      single.shapes_ = {shape};
      single.queries_ = {{{0., 0., 0.}, 0},
                         {{1., 0., 0.}, 0},
                         {{-1., 0., 0.}, 0},
                         {{0., 2., 0.}, 0},
                         {{0., 0., 3.}, 0},
                         {{std::nextafter(1., 0.), 0., 0.}, 0},
                         {{std::nextafter(1., infinity), 0., 0.}, 0},
                         {{0., 0., -0.}, 0},
                         {{0., 0., -1.e-15}, 0},
                         {{0., 0., 1.e-15}, 0},
                         {{nan, 0., 0.}, 0},
                         {{0., 0., nan}, 0},
                         {{infinity, 0., 0.}, 0},
                         {{0., 0., -infinity}, 0}};
      result.push_back(single);
    }
    // Ordinary non-binary offsets and axes exercise dot-product rounding on
    // the device without placing the cross-device check on an ambiguous edge.
    for (auto kind : {shape_type::Kind::Box, shape_type::Kind::SeparationPlane}) {
      Dataset oriented;
      oriented.name_ = "translated oblique predicates";
      oriented.nodes_.resize(1);
      auto shape = box(1., 2., 3.);
      shape.kind_ = kind;
      shape.centerM_[0] = .11;
      shape.centerM_[1] = -.27;
      shape.centerM_[2] = .43;
      shape.axes_[0][0] = .6;
      shape.axes_[0][1] = .8;
      shape.axes_[1][0] = -.8;
      shape.axes_[1][1] = .6;
      shape.normal_[0] = .2;
      shape.normal_[1] = .3;
      shape.normal_[2] = .4;
      oriented.shapes_ = {shape};
      for (double x : {-2.13, -.57, .23, 1.19, 3.41}) {
        for (double y : {-1.37, .17, 2.71}) {
          oriented.queries_.push_back({{x, y, -.19}, 0});
        }
      }
      result.push_back(oriented);
    }
    return result;
  }

} // namespace environment_search_reference
