/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once

#include <corsika/geometry/NextEnvironmentBoundary.hpp>
#include <tuple>
#include <type_traits>

namespace corsika {
  namespace tracking_tree_detail {
    template <typename Tracking, typename Particle> struct NativeAccess {
      using Node = std::remove_reference_t<decltype(*std::declval<Particle>().getNode())>;
      Particle const& particle;
      auto intersect(Node* node) const { return Tracking::intersect(particle,*node); }
      Node* parent(Node* node) const { return node->getParent(); }
      unsigned childCount(Node* node) const { return node->getChildNodes().size(); }
      Node* child(Node* node,unsigned i) const { return node->getChildNodes()[i].get(); }
      unsigned excludedCount(Node* node) const { return node->getExcludedNodes().size(); }
      Node* excluded(Node* node,unsigned i) const { return node->getExcludedNodes()[i]; }
    };
  }
  template <typename TDerived>
  template <typename TParticle>
  inline auto Intersect<TDerived>::nextIntersect(TParticle const& particle,
                                                 TimeType const step_limit) const {
    auto result=geometry_detail::nextEnvironmentBoundary(
        tracking_tree_detail::NativeAccess<TDerived,TParticle>{particle},
        particle.getNode(),step_limit);
    return std::make_tuple(result.distance,result.node);
  }
} // namespace corsika
