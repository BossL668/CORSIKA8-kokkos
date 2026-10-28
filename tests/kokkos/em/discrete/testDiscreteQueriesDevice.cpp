/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "DiscreteChecks.hpp"
#include "../tables/NativeTableFixture.hpp"
#include <corsika/kokkos/em/execution/NativeTableStorage.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <type_traits>

TEST_CASE("Discrete device queries match fixed production", "[discrete_device]") {
  using namespace discrete_test;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
  current::Runtime runtime;
  auto const table = current::test::make_native_tables();
  auto const requests = queries(table);
  current::NativeTableStorage<execution_space_type> storage;
  storage.initialize(table);
  native_query_test::UploadedReference<memory_space_type> oldStorage(table);
  auto const native = storage.getDeviceView();
  auto const old = oldStorage.view();
  auto const input =
      native_query_test::UploadedReference<memory_space_type>::upload(requests);
  Kokkos::View<Outcome * [2], Kokkos::LayoutRight, memory_space_type> output(
      "discrete_outcomes", requests.size());
  Kokkos::parallel_for(
      "discrete_fixed_production",
      Kokkos::RangePolicy<execution_space_type>(0, requests.size()),
      KOKKOS_LAMBDA(std::size_t index) {
        output(index, 0) = evaluate(native, input(index));
        output(index, 1) = expected(old, input(index));
      });
  Kokkos::fence("discrete outcomes ready");
  auto const results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  std::vector<Outcome> outcomes;
  for (std::size_t index = 0; index < requests.size(); ++index) {
    INFO("Same-device discrete query "
         << index << ", operation " << static_cast<unsigned>(requests[index].operation_));
    check(results(index, 0), results(index, 1));
    outcomes.push_back(results(index, 0));
  }
  REQUIRE(storage.getSourceContentHash() == table.contentHash_);
  report("device", table, requests, outcomes);
}
