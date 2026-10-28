/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentNode.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <cstdint>
#include <type_traits>

namespace {
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using node_type = corsika::kokkos::EnvironmentNode;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
} // namespace

TEST_CASE("Topology arrays can be traversed in the configured execution space",
          "[environment_transfer]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    // Synthetic ring of directed exclusions and a star ownership tree. This
    // tests numeric storage/adjacency, not containment or boundary searching.
    int constexpr count = 257;
    Kokkos::View<node_type*, Kokkos::HostSpace> hostNodes("host_nodes", count);
    Kokkos::View<std::int32_t*, Kokkos::HostSpace> hostChildren("host_children",
                                                                count - 1);
    Kokkos::View<std::int32_t*, Kokkos::HostSpace> hostExclusions("host_exclusions",
                                                                  count);
    for (int i = 0; i < count; ++i) {
      hostNodes(i) = {i == 0 ? -1 : 0,
                      i % 7 - 1,
                      static_cast<std::uint32_t>(i == 0 ? 0 : count - 1),
                      static_cast<std::uint32_t>(i == 0 ? count - 1 : 0),
                      static_cast<std::uint32_t>(i),
                      1};
      hostExclusions(i) = (i + 1) % count;
      if (i > 0) hostChildren(i - 1) = i;
    }
    Kokkos::View<node_type*, memory_space_type> nodes("nodes", count);
    Kokkos::View<std::int32_t*, memory_space_type> children("children", count - 1);
    Kokkos::View<std::int32_t*, memory_space_type> exclusions("exclusions", count);
    Kokkos::View<node_type*, memory_space_type> output("output", count);
    // The two-dimensional diagnostics need the same layout on host and CUDA.
    Kokkos::View<std::int32_t * [3], Kokkos::LayoutRight, memory_space_type> checks(
        "checks", count);
    Kokkos::deep_copy(nodes, hostNodes);
    Kokkos::deep_copy(children, hostChildren);
    Kokkos::deep_copy(exclusions, hostExclusions);
    Kokkos::deep_copy(checks, -999);
    Kokkos::parallel_for(
        "topology_adjacency", Kokkos::RangePolicy<execution_space_type>(0, count),
        KOKKOS_LAMBDA(int const i) {
          auto const source = nodes(i);
          output(count - 1 - i) = source;
          std::int32_t childSum = 0;
          for (std::uint32_t j = 0; j < source.childCount_; ++j) {
            childSum += children(source.childOffset_ + j);
          }
          checks(i, 0) = childSum;
          checks(i, 1) = exclusions(source.excludedOffset_);
          checks(i, 2) = node_type{}.parentIndex_;
        });
    Kokkos::fence("topology transfer complete");
    Kokkos::View<node_type*, Kokkos::HostSpace> hostOutput("host_output", count);
    Kokkos::View<std::int32_t * [3], Kokkos::LayoutRight, Kokkos::HostSpace> hostChecks(
        "host_checks", count);
    Kokkos::deep_copy(hostOutput, output);
    Kokkos::deep_copy(hostChecks, checks);
    for (int i = 0; i < count; ++i) {
      CAPTURE(i);
      auto const actual = hostOutput(count - 1 - i);
      auto const expected = hostNodes(i);
      REQUIRE(actual.parentIndex_ == expected.parentIndex_);
      REQUIRE(actual.modelIndex_ == expected.modelIndex_);
      REQUIRE(actual.childOffset_ == expected.childOffset_);
      REQUIRE(actual.childCount_ == expected.childCount_);
      REQUIRE(actual.excludedOffset_ == expected.excludedOffset_);
      REQUIRE(actual.excludedCount_ == expected.excludedCount_);
      REQUIRE(hostChecks(i, 0) == (i == 0 ? count * (count - 1) / 2 : 0));
      REQUIRE(hostChecks(i, 1) == (i + 1) % count);
      REQUIRE(hostChecks(i, 2) == -1);
    }
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}
