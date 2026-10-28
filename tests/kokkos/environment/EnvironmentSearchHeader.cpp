/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include <corsika/kokkos/environment/EnvironmentSearch.hpp>

#include <type_traits>
#include <utility>

namespace {
  using tree_type = corsika::kokkos::EnvironmentTreeView<Kokkos::HostSpace>;
  static_assert(std::is_copy_constructible_v<tree_type>);
  static_assert(std::is_same_v<tree_type::node_view_type::value_type,
                               corsika::kokkos::EnvironmentNode const>);
  static_assert(std::is_same_v<tree_type::shape_view_type::value_type,
                               corsika::kokkos::EnvironmentShape const>);
  static_assert(std::is_same_v<decltype(std::declval<tree_type const&>().getNodeCount()),
                               std::uint32_t>);
  static_assert(std::is_same_v<decltype(std::declval<tree_type const&>().getShape(0)),
                               corsika::kokkos::EnvironmentShape const&>);
  using search_type = std::int32_t (*)(tree_type const&, double const (&)[3],
                                       std::int32_t);
  static_assert(
      std::is_same_v<decltype(&corsika::kokkos::locate_node<tree_type>), search_type>);
} // namespace
