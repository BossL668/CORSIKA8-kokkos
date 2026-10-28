/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/geometry/NextEnvironmentBoundary.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>

#include "NextEnvironmentBoundaryReference.hpp"

#include <catch2/catch_all.hpp>

#include <cmath>

namespace {
  void require_exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }

  struct NativeNode {
    boundary_reference::Interval<corsika::TimeType> interval_;
    NativeNode const* parent_{};
    NativeNode const* child_{};
    NativeNode const* excluded_{};
  };

  struct PointerAccess {
    auto intersect(NativeNode const* node) const { return node->interval_; }
    auto getParent(NativeNode const* node) const { return node->parent_; }
    unsigned getChildCount(NativeNode const* node) const { return node->child_ ? 1 : 0; }
    auto getChild(NativeNode const* node, unsigned) const { return node->child_; }
    unsigned getExcludedCount(NativeNode const* node) const {
      return node->excluded_ ? 1 : 0;
    }
    auto getExcluded(NativeNode const* node, unsigned) const { return node->excluded_; }
  };
} // namespace

TEST_CASE("Ordered boundary selection preserves production corner cases",
          "[next_boundary]") {
  for (auto const& input : boundary_reference::ordered_cases()) {
    CAPTURE(input.name_);
    auto const result =
        corsika::next_environment_boundary(input, std::int32_t{0}, input.limit_);
    auto const reference = boundary_reference::next(input, std::int32_t{0}, input.limit_);
    require_exact(result.getParameter(), input.expectedParameter_);
    require_exact(result.getParameter(), reference.parameter_);
    REQUIRE(result.getNode() == input.expectedNode_);
    REQUIRE(result.getNode() == reference.node_);
  }
}

TEST_CASE("Shared boundary selection retains unitful parameters and pointer handles",
          "[next_boundary]") {
  using namespace corsika;
  NativeNode parent{{true, -10_s, 10_s}};
  NativeNode child{{true, 2_s, 4_s}};
  NativeNode excluded{{true, 2_s, 5_s}};
  NativeNode current{{true, -1_s, 2_s}, &parent, &child, &excluded};
  PointerAccess const access;
  auto const tied =
      next_environment_boundary(access, static_cast<NativeNode const*>(&current), 2_s);
  REQUIRE(tied.getParameter() == 2_s);
  REQUIRE(tied.getNode() == &excluded);
  excluded.interval_ = {true, -3_s, 1_s};
  auto const clamped =
      next_environment_boundary(access, static_cast<NativeNode const*>(&current), 2_s);
  REQUIRE(clamped.getParameter() == 0_s);
  REQUIRE(clamped.getNode() == &excluded);
  current.child_ = current.excluded_ = nullptr;
  current.parent_ = nullptr;
  auto const outside =
      next_environment_boundary(access, static_cast<NativeNode const*>(&current), 2_s);
  REQUIRE(outside.getNode() == nullptr);
  REQUIRE(outside.getParameter() == 2_s);
}
