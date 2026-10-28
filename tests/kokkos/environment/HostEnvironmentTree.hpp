/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentNode.hpp>
#include <corsika/kokkos/environment/EnvironmentShape.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace environment_test {
  // Keep native C8 units/geometry in a host-compiled test translation unit.
  // The implementation owns the real EnvironmentTreeView<Kokkos::HostSpace>;
  // copies exercise its shared allocations, rather than a substitute tree.
  class HostEnvironmentTree {
  public:
    using node_index_type = std::int32_t;
    HostEnvironmentTree();
    HostEnvironmentTree(std::vector<corsika::kokkos::EnvironmentNode> const&,
                        std::vector<corsika::kokkos::EnvironmentShape> const&,
                        std::vector<node_index_type> const&,
                        std::vector<node_index_type> const&);
    HostEnvironmentTree(HostEnvironmentTree const&);
    ~HostEnvironmentTree();

    std::uint32_t getNodeCount() const;
    node_index_type getParent(node_index_type) const;
    corsika::kokkos::EnvironmentShape const& getShape(node_index_type) const;
    std::uint32_t getChildCount(node_index_type) const;
    node_index_type getChild(node_index_type, std::uint32_t) const;
    std::uint32_t getExcludedCount(node_index_type) const;
    node_index_type getExcluded(node_index_type, std::uint32_t) const;

  private:
    struct State;
    std::unique_ptr<State> state_;
  };

  bool is_initialized();
  bool is_finalized();
} // namespace environment_test
