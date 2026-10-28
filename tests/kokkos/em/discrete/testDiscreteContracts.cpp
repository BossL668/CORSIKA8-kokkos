/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "DiscreteChecks.hpp"

TEST_CASE("Discrete failure and coordinate contracts preserve production", "[discrete]") {
  using namespace discrete_test;
  current::ProposalNativeTableSet const empty;
  native_query_test::HostReference const owner(empty);
  auto const input = queries(empty);
  for (auto const& query : input) {
    check(evaluate({}, query), expected(owner.view(empty), query));
  }
  for (std::uint32_t a : {0u, 1u, 65535u, 65536u, 0xffffffffu})
    for (std::uint32_t b : {0u, 1u, 65535u, 65536u, 0xffffffffu}) {
      auto const packed = current::pack_native_inverse_iterations(a, b);
      REQUIRE(packed == reference::packNativeInverseIterations(a, b));
      current::TableQueryResult result{current::TableLookupStatus::Success, packed, 0.};
      REQUIRE(current::native_newton_iterations(result) == std::min(a, 65535u));
      REQUIRE(current::native_bisection_iterations(result) == std::min(b, 65535u));
    }
}

TEST_CASE("Selection uses ordered column rates, not the distance-rate spline",
          "[discrete]") {
  using namespace discrete_test;
  // Synthetic control fixture, not a cross-section approximation: direct columns
  // with equal nonzero factors make an exactly representable selection boundary.
  current::ProposalNativeTableSet table;
  table.dndxColumns_.resize(3);
  table.selectionColumnIndices_ = {2, 0, 1};
  table.totalRateColumns_.resize(1);
  table.totalRateColumns_[0].pdgId_ = -11;
  auto& spline = table.totalRateColumns_[0].spline_;
  spline.axis_ = {current::NativeAxisType::Linear, 2, 1., 20., 19.};
  spline.coefficientCount_ = 2;
  table.cubicValues_ = {99., 99.};
  table.cubicNodeDerivatives_ = {0., 0.};
  table.bicubicValues_ = {0.};
  table.bicubicDerivativeEnergy_ = {0.};
  table.bicubicDerivativeLoss_ = {0.};
  table.bicubicMixedDerivative_ = {0.};
  for (std::size_t i = 0; i < 3; ++i) {
    auto& column = table.dndxColumns_[i];
    column.pdgId_ = -11;
    column.processId_ = 100 + i;
    column.componentHash_ = 500 + i;
    column.rateModel_ = current::NativeRateModel::AnnihilationHeitler;
    column.particleMassMeV_ = 1.;
    column.directRateFactor_ = i ? 1. : 0.;
  }
  auto compare_table = [&]() {
    auto const view = current::make_proposal_native_host_view(table);
    native_query_test::HostReference const owner(table);
    for (double uniform : {0., std::nextafter(.5, 0.), .5, std::nextafter(.5, 1.),
                           std::nextafter(1., 0.), 1., -1.}) {
      Query const query{Operation::Select, -11, 0, 0, 10., uniform, false};
      check(evaluate(view, query), expected(owner.view(table), query));
    }
  };
  compare_table();
  auto const view = current::make_proposal_native_host_view(table);
  auto const first = current::select_rate_column_by_uniform(view, -11, 10., 0.);
  REQUIRE(first.selected_ == 1);
  REQUIRE(first.processId_ == 102);
  exact(first.residualQuantile_, 1.);
  auto const boundary = current::select_rate_column_by_uniform(view, -11, 10., .5);
  REQUIRE(boundary.selected_ == 1);
  REQUIRE(boundary.processId_ == 101); // Strict subtraction skips zero and equal edges.
  exact(boundary.residualQuantile_, 1.);
  auto const distance = current::query_discrete_total_rate(view, -11, 10.);
  REQUIRE(distance.status_ == current::TableLookupStatus::Success);
  exact(distance.value_, 99.);
  REQUIRE(first.totalRate_ != distance.value_);
  table.selectionColumnIndices_[0] = 3; // In-bounds storage, out-of-range index value.
  compare_table();
  REQUIRE(current::select_rate_column_by_uniform(
              current::make_proposal_native_host_view(table), -11, 10., 0.)
              .status_ == current::TableLookupStatus::InvalidTableView);
  table.selectionColumnIndices_ = {2, 0, 1};
  for (auto& column : table.dndxColumns_) column.directRateFactor_ = 0.;
  compare_table();
  auto const zero = current::select_rate_column_by_uniform(
      current::make_proposal_native_host_view(table), -11, 10., .5);
  REQUIRE(zero.status_ == current::TableLookupStatus::Success);
  REQUIRE(zero.selected_ == 0);
  exact(zero.totalRate_, 0.);
}
