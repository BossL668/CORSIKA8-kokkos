/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include <corsika/kokkos/environment/EnvironmentSearch.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include "EnvironmentSearchReference.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using tree_type = corsika::kokkos::EnvironmentTreeView<memory_space_type>;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif

  template <typename TValue>
  auto copy_to_device(std::vector<TValue> const& values) {
    Kokkos::View<TValue*, Kokkos::HostSpace> host("search_host_input", values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
      host(index) = values[index];
    }
    Kokkos::View<TValue*, memory_space_type> device("search_device_input", values.size());
    Kokkos::deep_copy(device, host);
    return device;
  }

} // namespace

TEST_CASE("Device environment search agrees exactly with production and host",
          "[environment_search_device]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    for (auto const& input : environment_search_reference::datasets()) {
      CAPTURE(input.name_);
      auto const nodes = copy_to_device(input.nodes_);
      auto const shapes = copy_to_device(input.shapes_);
      auto const children = copy_to_device(input.children_);
      auto const exclusions = copy_to_device(input.exclusions_);
      auto const queries = copy_to_device(input.queries_);
      tree_type const tree(nodes, shapes, children, exclusions);
      environment_search_reference::Tree const reference{
          nodes.data(), shapes.data(), children.data(), exclusions.data(),
          static_cast<std::uint32_t>(input.nodes_.size())};
      int const count = static_cast<int>(input.queries_.size());
      Kokkos::View<std::int32_t * [4], Kokkos::LayoutRight, memory_space_type> results(
          "search_results", count);
      Kokkos::parallel_for(
          "environment_node_search", Kokkos::RangePolicy<execution_space_type>(0, count),
          KOKKOS_LAMBDA(int index) {
            auto const query = queries(index);
            results(index, 0) =
                corsika::kokkos::locate_node(tree, query.positionM_, query.start_);
            results(index, 1) = environment_search_reference::locate(
                reference, query.positionM_, query.start_);
            results(index, 2) = -1;
            results(index, 3) = -1;
            if (query.start_ >= 0 &&
                static_cast<std::uint32_t>(query.start_) < tree.getNodeCount()) {
              results(index, 2) = corsika::kokkos::contains(tree.getShape(query.start_),
                                                            query.positionM_);
              results(index, 3) = environment_search_reference::contains(
                  shapes(query.start_), query.positionM_);
            }
          });
      Kokkos::View<std::int32_t*, memory_space_type> parents("search_parents",
                                                             nodes.extent(0));
      Kokkos::parallel_for(
          "environment_node_parents",
          Kokkos::RangePolicy<execution_space_type>(0, static_cast<int>(nodes.extent(0))),
          KOKKOS_LAMBDA(int index) { parents(index) = tree.getParent(index); });
      Kokkos::fence("environment searches complete");
      auto const hostResults =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
      auto const hostParents =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, parents);
      for (int index = 0; index < count; ++index) {
        auto const& query = input.queries_[index];
        CAPTURE(index, query.start_, query.positionM_[0], query.positionM_[1],
                query.positionM_[2]);
        // Exact same-device production agreement is mandatory. The fixture's
        // axis-aligned strict boundaries also permit exact CPU/GPU node equality.
        REQUIRE(hostResults(index, 0) == hostResults(index, 1));
        REQUIRE(hostResults(index, 2) == hostResults(index, 3));
        REQUIRE(hostResults(index, 0) ==
                environment_search_reference::locate(input.tree(), query.positionM_,
                                                     query.start_));
        if (query.start_ >= 0 &&
            static_cast<std::size_t>(query.start_) < input.shapes_.size()) {
          REQUIRE(hostResults(index, 2) ==
                  environment_search_reference::contains(input.shapes_[query.start_],
                                                         query.positionM_));
        }
      }
      for (std::size_t index = 0; index < input.nodes_.size(); ++index) {
        REQUIRE(hostParents(index) == input.nodes_[index].parentIndex_);
      }
    }
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}
