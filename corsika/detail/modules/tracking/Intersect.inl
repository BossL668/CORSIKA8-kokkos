/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/NextEnvironmentBoundary.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

namespace corsika {

  namespace tracking_tree_detail {

    template <typename TTracking, typename TParticle>
    class NativeBoundaryAccess {
    public:
      using node_type =
          std::remove_reference_t<decltype(*std::declval<TParticle>().getNode())>;

      explicit NativeBoundaryAccess(TParticle const& particle)
          : particle_(particle) {}

      auto intersect(node_type* node) const {
        return TTracking::intersect(particle_, *node);
      }

      node_type* getParent(node_type* node) const { return node->getParent(); }

      unsigned int getChildCount(node_type* node) const {
        return static_cast<unsigned int>(node->getChildNodes().size());
      }

      node_type* getChild(node_type* node, unsigned int child) const {
        return node->getChildNodes()[child].get();
      }

      unsigned int getExcludedCount(node_type* node) const {
        return static_cast<unsigned int>(node->getExcludedNodes().size());
      }

      node_type* getExcluded(node_type* node, unsigned int excluded) const {
        return node->getExcludedNodes()[excluded];
      }

    private:
      TParticle const& particle_;
    };

  } // namespace tracking_tree_detail

  template <typename TDerived>
  template <typename TParticle>
  inline auto Intersect<TDerived>::nextIntersect(TParticle const& particle,
                                                 TimeType const step_limit) const {
    auto const result = next_environment_boundary(
        tracking_tree_detail::NativeBoundaryAccess<TDerived, TParticle>{particle},
        particle.getNode(), step_limit);
    return std::make_tuple(result.getParameter(), result.getNode());
  }

} // namespace corsika
