/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "ContinuousChecks.hpp"
#include <corsika/kokkos/em/execution/NativeTableStorage.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <type_traits>

TEST_CASE("Continuous device queries match fixed production", "[continuous_device]") {
  using namespace continuous_test;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
  current::Runtime runtime;
  auto data = make_dataset();
  populate_records(data);
  current::NativeTableStorage<execution_space_type> storage;
  storage.initialize(data.table_);
  native_query_test::UploadedReference<memory_space_type> oldStorage(data.table_);
  auto const native = storage.getDeviceView();
  auto const old = oldStorage.view();
  auto const input =
      native_query_test::UploadedReference<memory_space_type>::upload(data.queries_);
  Kokkos::View<Outcome * [2], Kokkos::LayoutRight, memory_space_type> output(
      "continuous_outcomes", data.queries_.size());
  Kokkos::parallel_for(
      "continuous_fixed_production",
      Kokkos::RangePolicy<execution_space_type>(0, data.queries_.size()),
      KOKKOS_LAMBDA(std::size_t index) {
        auto const query = input(index);
        output(index, 0) = evaluate(native, query);
        output(index, 1) = expected(old, query);
      });
  Kokkos::fence("continuous outcomes ready");
  auto const results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  std::vector<Outcome> outcomes;
  for (std::size_t index = 0; index < data.queries_.size(); ++index) {
    INFO("Same-device continuous query "
         << index << ", operation "
         << static_cast<unsigned>(data.queries_[index].operation_));
    check(results(index, 0), results(index, 1));
    outcomes.push_back(results(index, 0));
  }
  REQUIRE(storage.getSourceContentHash() == data.table_.contentHash_);
  report("device", data, outcomes);
}
