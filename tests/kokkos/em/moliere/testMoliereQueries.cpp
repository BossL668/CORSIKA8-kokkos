/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "MoliereAssertions.hpp"
#include "MoliereTestUtilities.hpp"

TEST_CASE("Real Air Moliere queries preserve fixed-production numerical results",
          "[kokkos][moliere][queries]") {
  using namespace corsika::kokkos;
  namespace fixture = corsika::kokkos::test;
  auto const data = fixture::make_moliere_dataset();
  REQUIRE(data.queries_.size() > 1000);
  auto const view = make_moliere_interpolation_view(
      NativeArrayView<MoliereCubicPolynomial>{data.table_.polynomials_},
      NativeArrayView<double>{data.table_.initialGuessDelta_});
  auto const referenceTable = fixture::moliere_reference_table(data.table_);
  auto const referenceView = fixture::moliere_reference::makeMoliereInterpolationView(
      referenceTable.polynomials.data(), referenceTable.initial_guess_delta.data());
  std::vector<fixture::moliere_reference::MoliereSnapshot> snapshots;
  for (auto const& snapshot : data.snapshots_)
    snapshots.push_back(fixture::moliere_reference_snapshot(snapshot));
  std::vector<std::array<fixture::MoliereOutcome, 2>> results;
  fixture::MoliereErrorReport report;
  for (auto const& query : data.queries_) {
    std::array<fixture::MoliereOutcome, 2> result{
        fixture::evaluate_moliere(data.snapshots_[query.snapshotIndex_],
                                  fixture::moliere_view_variant(view, query.viewIndex_),
                                  query),
        fixture::reference_moliere(
            snapshots[query.snapshotIndex_],
            fixture::moliere_reference_view_variant(referenceView, query.viewIndex_),
            query)};
    report.observe(query, result[0], result[1]);
    results.push_back(result);
  }
  report.write("host");
  for (std::size_t index = 0; index < results.size(); ++index) {
    auto const& query = data.queries_[index];
    INFO("query " << index << " operation " << static_cast<unsigned>(query.operation_)
                  << " snapshot " << query.snapshotIndex_ << " view "
                  << query.viewIndex_);
    fixture::require_moliere_exact(results[index][0], results[index][1]);
    if (query.operation_ == fixture::MoliereOperation::SmallCapacity &&
        query.values_[5] == 4.) {
      REQUIRE(index > 0);
      REQUIRE(data.queries_[index - 1].operation_ == fixture::MoliereOperation::Angle);
      fixture::require_moliere_exact(results[index][0], results[index - 1][0]);
    }
  }
  REQUIRE(report.differences_ == 0);
  for (auto const& native : report.native_) REQUIRE(native.count_ == 24);
  REQUIRE(report.angleStatuses_[0] > 0);
  REQUIRE(report.angleStatuses_[1] > 0);
  REQUIRE(report.angleStatuses_[2] > 0);
  REQUIRE(report.angleStatuses_[3] > 0);
  REQUIRE(report.angleStatuses_[5] > 0);
  // Public analytic overload and the exact projected median retain their contracts.
  auto const analytic =
      sample_moliere_scattering_angle_2d(data.snapshots_[0], 1., 1.e3, 900., .5, .5);
  REQUIRE(analytic.status_ == MoliereStatus::Success);
  REQUIRE(analytic.angleRad_ == 0.);
  REQUIRE(analytic.iterations_ == 0);
}
