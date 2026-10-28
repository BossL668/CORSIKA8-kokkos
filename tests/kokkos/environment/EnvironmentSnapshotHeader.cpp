/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentSnapshot.hpp>

#include <type_traits>
#include <utility>

namespace {

  struct HeaderModel;
  using snapshot_type = corsika::kokkos::EnvironmentSnapshot<HeaderModel>;

  static_assert(std::is_copy_constructible_v<snapshot_type>);
  static_assert(std::is_move_constructible_v<snapshot_type>);
  static_assert(!std::is_default_constructible_v<snapshot_type>);
  static_assert(
      std::is_same_v<decltype(std::declval<snapshot_type const&>().getTopology()),
                     snapshot_type::topology_type const&>);
  static_assert(std::is_same_v<decltype(std::declval<snapshot_type const&>().getShapes()),
                               std::vector<corsika::kokkos::EnvironmentShape> const&>);
  static_assert(std::is_same_v<decltype(std::declval<snapshot_type const&>().getModels()),
                               std::vector<corsika::kokkos::EnvironmentModel> const&>);
  static_assert(
      std::is_same_v<decltype(std::declval<snapshot_type const&>().getComposition()),
                     std::vector<corsika::kokkos::EnvironmentComponent> const&>);
  static_assert(
      std::is_same_v<decltype(std::declval<snapshot_type const&>().getCoordinateSystem()),
                     corsika::CoordinateSystemPtr const&>);

} // namespace
