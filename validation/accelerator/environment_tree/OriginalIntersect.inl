/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/geometry/Intersections.hpp>

#include <limits>

namespace corsika {

  template <typename TDerived>
  template <typename TParticle>
  inline auto OriginalIntersect<TDerived>::nextIntersect(TParticle const& particle,
                                                 TimeType const step_limit) const {

    typedef typename std::remove_reference<decltype(*particle.getNode())>::type node_type;

    node_type& volumeNode =
        *particle.getNode(); // current "logical" node, from previous tracking step

    CORSIKA_LOG_DEBUG("volumeNode={}, numericallyInside={} ", fmt::ptr(&volumeNode),
                      volumeNode.getVolume().contains(particle.getPosition()));

    // start values:
    TimeType minTime = step_limit;
    node_type* minNode = &volumeNode;

    // determine the first geometric collision with any other Volume boundary

    // first check, where we leave the current volume
    // this assumes our convention, that all Volume primitives must be convex
    // thus, the last entry is always the exit point
    Intersections const time_intersections_curr =
        TDerived::intersect(particle, volumeNode);
    CORSIKA_LOG_TRACE("curr node {}, parent node {}, hasIntersections={} ",
                      fmt::ptr(&volumeNode), fmt::ptr(volumeNode.getParent()),
                      time_intersections_curr.hasIntersections());
    if (time_intersections_curr.hasIntersections()) {
      CORSIKA_LOG_DEBUG(
          "intersection times with currentLogicalVolumeNode: entry={} s and exit={} s",
          time_intersections_curr.getEntry() / 1_s,
          time_intersections_curr.getExit() / 1_s);
      if (time_intersections_curr.getExit() <= minTime) {
        minTime =
            time_intersections_curr.getExit(); // we exit currentLogicalVolumeNode here
        minNode = volumeNode.getParent();
      }
    }

    // where do we collide with any of the next-tree-level volumes
    // entirely contained by currentLogicalVolumeNode
    for (auto const& node : volumeNode.getChildNodes()) {

      Intersections const time_intersections = TDerived::intersect(particle, *node);
      CORSIKA_LOG_DEBUG("search intersection times with child volume {}:",
                        fmt::ptr(node));
      if (!time_intersections.hasIntersections()) { continue; }
      auto const t_entry = time_intersections.getEntry();
      auto const t_exit = time_intersections.getExit();
      CORSIKA_LOG_DEBUG("children t-entry: {}, t-exit: {}, smaller? {} ", t_entry, t_exit,
                        t_entry <= minTime);
      // note, theoretically t can even be smaller than 0 since we
      // KNOW we can't yet be in this volume yet, so we HAVE TO
      // enter it IF exit point is not also in the "past", AND if
      // extry point is [[much much]] closer than exit point
      // (because we might have already numerically "exited" it)!
      if (t_exit > 0_s && t_entry <= minTime &&
          -t_entry < t_exit) { // protection agains numerical problem, when we already
                               // _exited_ before
                               // enter chile volume here
        minTime = t_entry;
        minNode = node.get();
      }
    }

    // these are volumes from the previous tree-level that are cut-out partly from the
    // current volume
    for (node_type* node : volumeNode.getExcludedNodes()) {

      Intersections const time_intersections = TDerived::intersect(particle, *node);
      if (!time_intersections.hasIntersections()) { continue; }
      CORSIKA_LOG_DEBUG(
          "intersection times with exclusion volume {} : enter {} s, exit {} s",
          fmt::ptr(node), time_intersections.getEntry() / 1_s,
          time_intersections.getExit() / 1_s);
      auto const t_entry = time_intersections.getEntry();
      auto const t_exit = time_intersections.getExit();
      CORSIKA_LOG_DEBUG("children t-entry: {}, t-exit: {}, smaller? {} ", t_entry, t_exit,
                        t_entry <= minTime);
      // note, theoretically t can even be smaller than 0 since we
      // KNOW we can't yet be in this volume yet, so we HAVE TO
      // enter it IF exit point is not also in the "past"!
      if (t_exit > 0_s && t_entry <= minTime) { // enter volumen child here
        minTime = t_entry;
        minNode = node;
      }
    }
    CORSIKA_LOG_DEBUG("next time-intersect: {}, node {} ", minTime, fmt::ptr(minNode));
    // this branch cannot be unit-testes. This is malfunction: LCOV_EXCL_START
    if (minTime < 0_s) {
      if (minTime < -1e-8_s) {
        CORSIKA_LOG_DEBUG(
            "There is a very negative time step detected: {}. This is not physical and "
            "may "
            "easily crash subsequent modules. Set to 0_s, but CHECK AND FIX.",
            minTime);
      }
      minTime = 0_s; // LCOV_EXCL_STOP
    }
    return std::make_tuple(minTime, minNode);
  }

} // namespace corsika
