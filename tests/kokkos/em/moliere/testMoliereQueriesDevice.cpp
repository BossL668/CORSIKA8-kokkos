/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "MoliereAssertions.hpp"
#include "MoliereTestUtilities.hpp"
#include <corsika/kokkos/em/execution/MoliereInterpolationStorage.hpp>
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
  template <typename TContainer>
  auto upload(TContainer const& values) {
    using value_type = typename TContainer::value_type;
    Kokkos::View<value_type*, memory_space_type> result("moliere_inputs", values.size());
    auto host = Kokkos::create_mirror_view(result);
    for (std::size_t index = 0; index < values.size(); ++index)
      host(index) = values[index];
    Kokkos::deep_copy(result, host);
    return result;
  }
} // namespace

TEST_CASE("Moliere device storage and queries preserve the device production oracle",
          "[kokkos][moliere][device]") {
  using namespace corsika::kokkos;
  namespace fixture = corsika::kokkos::test;
  using storage_type = MoliereInterpolationStorage<execution_space_type>;
  Runtime runtime;
  auto const data = fixture::make_moliere_dataset();
  storage_type survivor;
  REQUIRE_FALSE(survivor.isInitialized());
  REQUIRE(survivor.getDeviceBytes() == 0);
  REQUIRE_THROWS_AS(survivor.getDeviceView(), std::logic_error);
  {
    storage_type owner;
    REQUIRE_THROWS_AS(owner.initialize(MoliereSnapshot{}), std::invalid_argument);
    REQUIRE_FALSE(owner.isInitialized());
    REQUIRE(owner.getDeviceBytes() == 0);
    owner.initialize(data.snapshots_[0], data.cacheFile_);
    REQUIRE(owner.isInitialized());
    REQUIRE(owner.getDeviceBytes() == storage_type::projectedDeviceBytes());
    REQUIRE_THROWS_AS(owner.initialize(data.snapshots_[0], data.cacheFile_),
                      std::logic_error);
    survivor = owner;
  }
  // The originating owner is gone; this copy still owns both immutable arrays.
  REQUIRE(survivor.isInitialized());
  auto const view = survivor.getDeviceView();
  auto const polynomialValues = data.table_.polynomials_.size() * 4;
  auto const valueCount = polynomialValues + data.table_.initialGuessDelta_.size();
  Kokkos::View<double*, memory_space_type> roundtrip("moliere_roundtrip", valueCount);
  Kokkos::parallel_for(
      "moliere_upload_readback", Kokkos::RangePolicy<execution_space_type>(0, valueCount),
      KOKKOS_LAMBDA(std::size_t index) {
        if (index >= polynomialValues) {
          roundtrip(index) = view.getInitialGuessDelta()[index - polynomialValues];
        } else {
          auto const function = index / (MoliereInterpolationIntervalCount * 4);
          auto const polynomial = index / 4 % MoliereInterpolationIntervalCount;
          auto const component = index % 4;
          auto const values = function == 0
                                  ? view.getF1()
                                  : function == 1 ? view.getF2()
                                                  : function == 2 ? view.getIntegralF1()
                                                                  : view.getIntegralF2();
          roundtrip(index) = values[polynomial].c_[component];
        }
      });
  Kokkos::fence("Moliere roundtrip completion");
  auto const returned =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, roundtrip);
  for (std::size_t index = 0; index < valueCount; ++index) {
    auto const expected = index < polynomialValues
                              ? data.table_.polynomials_[index / 4].c_[index % 4]
                              : data.table_.initialGuessDelta_[index - polynomialValues];
    REQUIRE(fixture::moliere_same_value(returned(index), expected));
  }

  auto const referenceTable = fixture::moliere_reference_table(data.table_);
  auto const oldPolynomials = upload(referenceTable.polynomials);
  auto const oldGuesses = upload(referenceTable.initial_guess_delta);
  auto const referenceView = fixture::moliere_reference::makeMoliereInterpolationView(
      oldPolynomials.data(), oldGuesses.data());
  std::vector<fixture::moliere_reference::MoliereSnapshot> references;
  for (auto const& snapshot : data.snapshots_)
    references.push_back(fixture::moliere_reference_snapshot(snapshot));
  auto const snapshots = upload(data.snapshots_);
  auto const referenceSnapshots = upload(references);
  auto const queries = upload(data.queries_);
  Kokkos::View<fixture::MoliereOutcome * [2], Kokkos::LayoutRight, memory_space_type>
      output("moliere_outputs", data.queries_.size());
  Kokkos::parallel_for(
      "moliere_query_oracle",
      Kokkos::RangePolicy<execution_space_type>(0, data.queries_.size()),
      KOKKOS_LAMBDA(std::size_t index) {
        auto const query = queries(index);
        output(index, 0) = fixture::evaluate_moliere(
            snapshots(query.snapshotIndex_),
            fixture::moliere_view_variant(view, query.viewIndex_), query);
        output(index, 1) = fixture::reference_moliere(
            referenceSnapshots(query.snapshotIndex_),
            fixture::moliere_reference_view_variant(referenceView, query.viewIndex_),
            query);
      });
  Kokkos::fence("Moliere query completion");
  auto const results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  fixture::MoliereErrorReport report;
  for (std::size_t index = 0; index < data.queries_.size(); ++index)
    report.observe(data.queries_[index], results(index, 0), results(index, 1));
  report.write("device");
  for (std::size_t index = 0; index < data.queries_.size(); ++index) {
    auto const& query = data.queries_[index];
    INFO("query " << index << " operation " << static_cast<unsigned>(query.operation_)
                  << " snapshot " << query.snapshotIndex_ << " view "
                  << query.viewIndex_);
    fixture::require_moliere_exact(results(index, 0), results(index, 1));
    if (query.operation_ == fixture::MoliereOperation::SmallCapacity &&
        query.values_[5] == 4.) {
      REQUIRE(index > 0);
      REQUIRE(data.queries_[index - 1].operation_ == fixture::MoliereOperation::Angle);
      fixture::require_moliere_exact(results(index, 0), results(index - 1, 0));
    }
  }
  REQUIRE(report.differences_ == 0);
  REQUIRE(report.total_ > 1000);
  REQUIRE(survivor.getDeviceBytes() == storage_type::projectedDeviceBytes());
}
