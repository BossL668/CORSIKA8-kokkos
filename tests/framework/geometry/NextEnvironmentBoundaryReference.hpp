/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace boundary_reference {

  template <typename TParameter>
  struct Interval {
    bool valid_{};
    TParameter entry_{};
    TParameter exit_{};
    CORSIKA_HOST_DEVICE bool hasIntersections() const { return valid_; }
    CORSIKA_HOST_DEVICE TParameter getEntry() const { return entry_; }
    CORSIKA_HOST_DEVICE TParameter getExit() const { return exit_; }
  };

  template <typename TParameter, typename TNode>
  struct Result {
    TParameter parameter_;
    TNode node_;
  };

  // Independent production 41cdc543 NextEnvironmentBoundary.hpp transcription.
  // Only accessor/field names change; no shared selection helper is called.
  template <typename TAccess, typename TNode, typename TParameter>
#if !defined(CORSIKA_BOUNDARY_REFERENCE_HOST_ONLY)
  CORSIKA_HOST_DEVICE
#endif
      inline auto
      next(TAccess const& access, TNode current, TParameter limit) {
    Result<TParameter, TNode> result{limit, current};
    TParameter const zero{};
    auto const own = access.intersect(current);
    if (own.hasIntersections() && own.getExit() <= result.parameter_) {
      result.parameter_ = own.getExit();
      result.node_ = access.getParent(current);
    }
    for (unsigned int index = 0; index < access.getChildCount(current); ++index) {
      auto const child = access.getChild(current, index);
      auto const intersections = access.intersect(child);
      if (!intersections.hasIntersections()) { continue; }
      auto const entry = intersections.getEntry();
      auto const exit = intersections.getExit();
      if (exit > zero && entry <= result.parameter_ && -entry < exit) {
        result.parameter_ = entry;
        result.node_ = child;
      }
    }
    for (unsigned int index = 0; index < access.getExcludedCount(current); ++index) {
      auto const excluded = access.getExcluded(current, index);
      auto const intersections = access.intersect(excluded);
      if (!intersections.hasIntersections()) { continue; }
      auto const entry = intersections.getEntry();
      if (intersections.getExit() > zero && entry <= result.parameter_) {
        result.parameter_ = entry;
        result.node_ = excluded;
      }
    }
    if (result.parameter_ < zero) { result.parameter_ = zero; }
    return result;
  }

  struct OrderedCase {
    char const* name_{};
    Interval<double> intervals_[5]{};
    unsigned childCount_{};
    unsigned excludedCount_{};
    double limit_{4.};
    double expectedParameter_{};
    std::int32_t expectedNode_{};

    CORSIKA_HOST_DEVICE Interval<double> intersect(std::int32_t node) const {
      return intervals_[node];
    }
    CORSIKA_HOST_DEVICE std::int32_t getParent(std::int32_t) const { return -1; }
    CORSIKA_HOST_DEVICE unsigned getChildCount(std::int32_t) const { return childCount_; }
    CORSIKA_HOST_DEVICE std::int32_t getChild(std::int32_t, unsigned index) const {
      return static_cast<std::int32_t>(index + 1);
    }
    CORSIKA_HOST_DEVICE unsigned getExcludedCount(std::int32_t) const {
      return excludedCount_;
    }
    CORSIKA_HOST_DEVICE std::int32_t getExcluded(std::int32_t, unsigned index) const {
      return static_cast<std::int32_t>(index + 3);
    }
  };

  inline std::vector<OrderedCase> ordered_cases() {
    std::vector<OrderedCase> result;
    auto append = [&](char const* name, OrderedCase input, double parameter,
                      std::int32_t node) {
      input.name_ = name;
      input.expectedParameter_ = parameter;
      input.expectedNode_ = node;
      result.push_back(input);
    };
    append("no intersections", {}, 4., 0);
    OrderedCase own;
    own.intervals_[0] = {true, -5., 3.};
    append("current exit", own, 3., -1);
    own.intervals_[0].exit_ = 4.;
    append("exit ties limit", own, 4., -1);
    own.intervals_[0].exit_ = 5.;
    append("limit first", own, 4., 0);
    OrderedCase ties;
    ties.intervals_[0] = {true, -10., 2.};
    ties.childCount_ = 2;
    ties.intervals_[1] = ties.intervals_[2] = {true, 2., 10.};
    append("last child wins ties", ties, 2., 2);
    ties.excludedCount_ = 2;
    ties.intervals_[3] = ties.intervals_[4] = {true, 2., 10.};
    append("last exclusion wins all ties", ties, 2., 4);
    ties.intervals_[2].valid_ = ties.intervals_[3].valid_ = ties.intervals_[4].valid_ =
        false;
    append("invalid intervals skipped", ties, 2., 1);
    OrderedCase child;
    child.childCount_ = 1;
    child.intervals_[1] = {true, -3., 2.};
    append("child negative entry guard", child, 4., 0);
    child.intervals_[1].entry_ = -2.;
    append("child guard strict equality", child, 4., 0);
    child.intervals_[1].entry_ = -std::nextafter(2., 0.);
    append("child just inside guard clamps", child, 0., 1);
    child.intervals_[1] = {true, -1., 0.};
    append("child exit zero rejected", child, 4., 0);
    OrderedCase excluded;
    excluded.excludedCount_ = 1;
    excluded.intervals_[3] = {true, -3., 2.};
    append("exclusion has no child guard", excluded, 0., 3);
    excluded.intervals_[3].exit_ = 0.;
    append("excluded exit zero rejected", excluded, 4., 0);
    own.intervals_[0].exit_ = -1.;
    append("current negative exit clamps", own, 0., -1);
    own.intervals_[0].exit_ = -0.;
    append("negative zero retained", own, -0., -1);
    OrderedCase negative;
    negative.limit_ = -2.;
    append("negative limit clamps", negative, 0., 0);
    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const infinity = std::numeric_limits<double>::infinity();
    own.limit_ = nan;
    append("unordered limit retained", own, nan, 0);
    child.intervals_[0] = {true, -1., nan};
    child.intervals_[1] = {true, 2., 3.};
    append("unordered own exit skipped", child, 2., 1);
    child.intervals_[1].entry_ = nan;
    child.excludedCount_ = 1;
    child.intervals_[3] = {true, nan, 3.};
    append("unordered entries skipped", child, 4., 0);
    child.intervals_[1] = child.intervals_[3] = {true, 2., nan};
    append("unordered exits skipped", child, 4., 0);
    OrderedCase unbounded;
    unbounded.limit_ = infinity;
    append("unbounded no intersection", unbounded, infinity, 0);
    unbounded.intervals_[0] = {true, -infinity, infinity};
    append("infinite exit ties infinite limit", unbounded, infinity, -1);
    child = {};
    child.childCount_ = 1;
    child.intervals_[1] = {true, -infinity, infinity};
    append("infinite child fails strict guard", child, 4., 0);
    child.excludedCount_ = 1;
    child.intervals_[3] = child.intervals_[1];
    append("infinite exclusion clamps", child, 0., 3);
    return result;
  }

} // namespace boundary_reference
