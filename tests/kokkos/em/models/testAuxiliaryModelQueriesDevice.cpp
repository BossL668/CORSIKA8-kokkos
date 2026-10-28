/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "AuxiliaryModelAssertions.hpp"
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace {
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
  template <typename TValue>
  auto upload(std::vector<TValue> const& values) {
    Kokkos::View<TValue*, memory_space_type> result("auxiliary_inputs", values.size());
    auto host = Kokkos::create_mirror_view(result);
    for (std::size_t index = 0; index < values.size(); ++index)
      host(index) = values[index];
    Kokkos::deep_copy(result, host);
    return result;
  }
} // namespace

TEST_CASE("Actual device auxiliary queries preserve the independent production oracle",
          "[kokkos][auxiliary][device]") {
  namespace fixture = corsika::kokkos::test;
  corsika::kokkos::Runtime runtime;
  auto const data = fixture::make_auxiliary_dataset();
  REQUIRE(data.queries_.size() > 500);
  std::vector<fixture::AuxiliaryReferenceSnapshots> references;
  for (auto const& snapshot : data.snapshots_)
    references.push_back(fixture::to_reference(snapshot));
  auto const current = upload(data.snapshots_);
  auto const old = upload(references);
  auto const inputs = upload(data.queries_);
  Kokkos::View<fixture::AuxiliaryOutcome * [2], Kokkos::LayoutRight, memory_space_type>
      output("auxiliary_outputs", data.queries_.size());
  Kokkos::parallel_for(
      "fixed_auxiliary_oracle",
      Kokkos::RangePolicy<execution_space_type>(0, data.queries_.size()),
      KOKKOS_LAMBDA(std::size_t index) {
        auto const query = inputs(index);
        output(index, 0) =
            fixture::evaluate_auxiliary(current(query.snapshotIndex_), query);
        output(index, 1) = fixture::reference_auxiliary(old(query.snapshotIndex_), query);
      });
  Kokkos::fence("auxiliary queries before host inspection");
  auto const results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  std::size_t nativeCount{};
  fixture::AuxiliaryErrorReport report;
  for (std::size_t index = 0; index < data.queries_.size(); ++index)
    report.observe(data.queries_[index], results(index, 0), results(index, 1));
  report.write("device");
  for (std::size_t index = 0; index < data.queries_.size(); ++index) {
    auto const& query = data.queries_[index];
    INFO("query " << index << " operation " << static_cast<unsigned>(query.operation_)
                  << " snapshot " << query.snapshotIndex_ << " energy "
                  << query.energyMeV_ << " fraction " << query.fraction_ << " density "
                  << query.densityGPerCm3_);
    fixture::require_auxiliary_exact(results(index, 0), results(index, 1));
    if (query.nativeComparable_) {
      fixture::require_auxiliary_native(results(index, 0), query);
      ++nativeCount;
    }
  }
  REQUIRE(nativeCount > 500);
  REQUIRE(report.sameDeviceDifferences_ == 0);
}
