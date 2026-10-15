/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once

#include <corsika/accelerator/AcceleratorMacros.hpp>

namespace corsika::geometry_detail {

  template <typename Distance, typename Node> struct NextBoundaryResult {
    Distance distance;
    Node node;
  };

  /**
   * C8 Intersect::nextIntersect's ordered tree search, shared by the native
   * pointer tree and a device snapshot. Access supplies intersections and
   * topology; this routine deliberately contains no shape-specific physics.
   *
   * Preserve <= ties, the child-only -entry < exit guard, excluded-node
   * precedence, and the final negative-distance clamp. Do NOT replace this
   * by a global nearest-surface search: the current logical node matters.
   */
  template <typename Access, typename Node, typename Distance>
  C8_ACCELERATOR_INLINE_FUNCTION inline auto nextEnvironmentBoundary(
      Access const& access, Node current, Distance limit) {
    NextBoundaryResult<Distance, Node> result{limit, current};
    Distance const zero{};
    auto const own = access.intersect(current);
    if (own.hasIntersections() && own.getExit() <= result.distance) {
      result.distance = own.getExit();
      result.node = access.parent(current);
    }
    for (unsigned int i = 0; i < access.childCount(current); ++i) {
      auto const child = access.child(current, i);
      auto const intersections = access.intersect(child);
      if (!intersections.hasIntersections()) continue;
      auto const entry = intersections.getEntry();
      auto const exit = intersections.getExit();
      if (exit > zero && entry <= result.distance && -entry < exit) {
        result.distance = entry;
        result.node = child;
      }
    }
    for (unsigned int i = 0; i < access.excludedCount(current); ++i) {
      auto const excluded = access.excluded(current, i);
      auto const intersections = access.intersect(excluded);
      if (!intersections.hasIntersections()) continue;
      auto const entry = intersections.getEntry();
      if (intersections.getExit() > zero && entry <= result.distance) {
        result.distance = entry;
        result.node = excluded;
      }
    }
    if (result.distance < zero) result.distance = zero;
    return result;
  }
} // namespace corsika::geometry_detail
