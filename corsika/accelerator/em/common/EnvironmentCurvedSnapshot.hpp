/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once
#include <corsika/accelerator/em/common/EnvironmentStableQuartic.hpp>

namespace corsika::gpu::em::environment_tree {
  struct CurvedAccess : TreeView {
    double position[3]{},velocity[3]{},direction[3]{},field[3]{};
    double speed{},momentum_SI{},charge_SI{},gyroradius_m{};
    bool linear{};
    // A caller must check this flag after the shared search. Native C8 also
    // rejects shapes without curved-intersection implementations (e.g. boxes).
    mutable bool unsupported_shape{};
    mutable bool incomplete_roots{};

    C8_ACCELERATOR_INLINE_FUNCTION Intersections intersect(NodeId id) const {
      return intersectShape(nodes[id]);
    }

    // Also used for the separate observation plane: it is a process boundary,
    // not an environment node, but must use TrackingLeapFrogCurved's solver.
    C8_ACCELERATOR_INLINE_FUNCTION Intersections intersectShape(Node const& node) const {
      if (linear) return straightIntersect(node,position,velocity);
      if (node.shape==Shape::Universe) return {};
      double projected[3]{difference(product(direction[1],field[2]),product(direction[2],field[1])),
                          difference(product(direction[2],field[0]),product(direction[0],field[2])),
                          difference(product(direction[0],field[1]),product(direction[1],field[0]))};
      double delta[3]{position[0]-node.center[0],position[1]-node.center[1],position[2]-node.center[2]};
      auto const inf=infinity();
      if (node.shape==Shape::SeparationPlane) {
        if (charge_SI==0.) return straightIntersect(node,position,velocity);
        double const denom=charge_SI*dot(node.axes[0],projected);
        double const p=2*momentum_SI*dot(direction,node.axes[0]);
        double const q=2*momentum_SI*dot(node.axes[0],delta);
        auto const roots=polynomial::quadratic(denom,p,q);
        double first=inf;
        for (unsigned i=0;i<roots.size;++i)
          if (roots.values[i]>=0. && roots.values[i]<first) first=roots.values[i];
        return {true,first/speed,inf};
      }
      if (node.shape!=Shape::Sphere) {unsupported_shape=true;return {};}
      auto const projected2=dot(projected,projected);
      if (projected2==0.) return straightIntersect(node,position,velocity);
      double const deltaNorm=::sqrt(dot(delta,delta));
      double const distance=::fabs(deltaNorm-node.radius);
      if (distance>gyroradius_m) return {};
      if (gyroradius_m>1000*distance) return straightIntersect(node,position,velocity);
      double const k=charge_SI/momentum_SI,denom=4./(projected2*k*k);
      double const b=product(sum(product(dot(projected,delta),k),1),denom);
      double const c=dot(direction,delta)*2*denom;
      double const d=(deltaNorm+node.radius)*(deltaNorm-node.radius)*denom;
      auto const roots=polynomial::stableQuartic(b,c,d);
      bool const inside=contains(node,position);
      double entry=0.,exit=0.;unsigned count=0,entryCount=0,exitCount=0;
      for (unsigned i=0;i<roots.size;++i) {
        double const distance=roots.values[i];
        if (!polynomial::extendedFinite(distance)) {incomplete_roots=true;return {};}
        if (inside) {
          if (distance<1.e-4) {
            if (!entryCount || entry<distance) entry=distance;
            ++entryCount;
          } else {
            if (!exitCount || distance<exit) exit=distance;
            ++exitCount;
          }
          count=(entryCount>0)+(exitCount>0);
        } else {
          if (distance<-1.e-4) continue;
          if (!count) entry=distance;
          else if (distance<entry) {exit=entry;entry=distance;}
          else exit=distance;
          ++count;
        }
      }
      if (!count) return {};
      // Native LengthType defaults an unassigned endpoint to zero in this
      // degenerate case. Never pass that invented endpoint to the graph search;
      // surface a genuinely unresolved intersection to the
      // transport caller as an explicit numerical error, not a silent miss.
      if ((inside && (!entryCount || !exitCount)) || (!inside && count<2)) {
        incomplete_roots=true;return {};
      }
      return {true,entry/speed,exit/speed};
    }
  };
} // namespace corsika::gpu::em::environment_tree
