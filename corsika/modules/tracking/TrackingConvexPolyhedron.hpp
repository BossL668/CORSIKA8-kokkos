/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */

#pragma once

#include <corsika/framework/geometry/ConvexPolyhedron.hpp>
#include <corsika/modules/tracking/TrackingStraight.hpp>

namespace corsika::tracking_line {

  /**
   * Opt-in straight tracking for finite convex mesh volumes.
   *
   * Stock TrackingStraight is untouched. The velocity, environment-tree search
   * and trajectory construction retain its semantics. The application must
   * require zero magnetic field or explicitly handle magnetic propagation in a
   * different tracking policy. No particle position is nudged at a boundary.
   */
  class ConvexPolyhedronTracking : public Intersect<ConvexPolyhedronTracking> {
  public:
    template <typename Particle>
    auto getTrack(Particle const& particle) {
      auto [time, node] = this->nextIntersect(particle);
      VelocityVector const velocity =
          particle.getMomentum() / particle.getEnergy() * constants::c;
      return std::make_tuple(
          StraightTrajectory(Line(particle.getPosition(), velocity), time), node);
    }

    template <typename Particle>
    static Intersections intersect(Particle const& particle,
                                   ConvexPolyhedron const& volume) {
      auto const velocity = particle.getMomentum() / particle.getEnergy() * constants::c;
      auto const speed = velocity.getNorm();
      if (!(speed > 0_m / 1_s) || !std::isfinite(speed / (1_m / 1_s))) {
        throw std::runtime_error("ConvexPolyhedronTracking: invalid particle speed");
      }
      auto const interval = volume.intersectRay(particle.getPosition(), velocity / speed);
      if (!interval.intersects) return Intersections();
      return Intersections(TimeType(interval.entry_m * 1_m / speed),
                           TimeType(interval.exit_m * 1_m / speed));
    }

    template <typename Particle, typename Node>
    static Intersections intersect(Particle const& particle, Node const& node) {
      if (auto const* volume = dynamic_cast<ConvexPolyhedron const*>(&node.getVolume())) {
        return intersect(particle, *volume);
      }
      return Tracking::intersect(particle, node);
    }

    template <typename Particle>
    static Intersections intersect(Particle const& particle, Plane const& plane) {
      return Tracking::intersect(particle, plane);
    }
    template <typename Particle>
    static Intersections intersect(Particle const& particle, Sphere const& sphere) {
      return Tracking::intersect(particle, sphere);
    }
    template <typename Particle>
    static Intersections intersect(Particle const& particle, Box const& box) {
      return Tracking::intersect(particle, box);
    }
    template <typename Particle>
    static Intersections intersect(Particle const& particle,
                                   SeparationPlane const& plane) {
      return Tracking::intersect(particle, plane);
    }

    static std::string getName() { return "Tracking-Straight-ConvexPolyhedron"; }
    static std::string getVersion() { return "1.0.0"; }
  };

} // namespace corsika::tracking_line
