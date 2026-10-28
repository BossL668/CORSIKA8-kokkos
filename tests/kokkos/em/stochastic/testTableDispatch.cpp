/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "StochasticChecks.hpp"
#include "../discrete/DiscreteTestData.hpp"
#include "../tables/NativeTableFixture.hpp"
#include <corsika/kokkos/em/execution/NativeTableStorage.hpp>

namespace stochastic_test {
  struct DispatchQuery {
    current::TableQuery query_{};
    bool invalidView_{};
    double emCut_{.1}, muonCut_{10.};
  };
  CORSIKA_HOST_DEVICE inline Outcome dispatch(current::ProposalNativeDeviceView native,
                                              DispatchQuery const& input) {
    if (input.invalidView_) native = {};
    Outcome result;
    append(result, current::execute_table_query(current::ContinuousPhysicsView(
                                                    native, input.emCut_, input.muonCut_),
                                                input.query_));
    return result;
  }
  CORSIKA_HOST_DEVICE inline Outcome dispatch_reference(
      native_table_reference::ProposalNativeDeviceView native,
      DispatchQuery const& input) {
    if (input.invalidView_) native = {};
    reference::tables::NativePhysicsView view{};
    view.proposal_native = native;
    view.em_transport_cut_MeV = input.emCut_;
    view.muon_transport_cut_MeV = input.muonCut_;
    auto const& q = input.query_;
    reference::tables::TableQuery query{
        static_cast<reference::tables::TableQueryKind>(q.kind_),
        q.pdgId_,
        q.processId_,
        q.componentHash_,
        q.energyMeV_,
        q.quantile_};
    Outcome result;
    append(result, reference::tables::executeTableQuery(view, query));
    return result;
  }
} // namespace stochastic_test

TEST_CASE("All production query kinds route unchanged on host and device",
          "[table_dispatch]") {
  using namespace stochastic_test;
  current::Runtime runtime;
  auto const table = current::test::make_native_tables();
  std::vector<DispatchQuery> queries;
  for (auto const& input : discrete_test::queries(table)) {
    current::TableQueryKind kind;
    if (input.operation_ == discrete_test::Operation::Rate)
      kind = current::TableQueryKind::Rate;
    else if (input.operation_ == discrete_test::Operation::Total)
      kind = current::TableQueryKind::TotalRate;
    else if (input.operation_ == discrete_test::Operation::Loss)
      kind = current::TableQueryKind::LossFraction;
    else
      continue;
    queries.push_back({{kind, input.pid_, input.process_, input.component_, input.energy_,
                        input.argument_},
                       input.invalidView_});
  }
  auto const inf = std::numeric_limits<double>::infinity();
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  for (auto pid : {11, -11, 13, -13, 22, 999})
    for (auto kind : {current::TableQueryKind::ContinuousDedx,
                      current::TableQueryKind::ContinuousRange,
                      current::TableQueryKind::ContinuousEnergy,
                      current::TableQueryKind::EnergyAfterContinuousLoss})
      for (auto energy : {-1., 0., .511, 106., 1.e3, 1.e5, 1.e7, 1.e8, inf, nan})
        for (auto argument : {-1., 0., .1, 1., 100., inf, nan})
          queries.push_back({{kind, pid, 0, 0, energy, argument}, false});
  // Derive meaningful nonzero grammage and range inputs from the independent oracle.
  native_query_test::HostReference hostOld(table);
  auto const old = hostOld.view(table);
  reference::tables::NativePhysicsView oldView{};
  oldView.proposal_native = old;
  oldView.em_transport_cut_MeV = .1;
  oldView.muon_transport_cut_MeV = 10.;
  for (auto pid : {11, -11, 13, -13})
    for (double energy : {1.e3, 1.e5, 1.e7}) {
      auto const range = reference::tables::queryContinuousRange(oldView, pid, energy);
      REQUIRE(range.status == reference::tables::TableLookupStatus::Success);
      for (double fraction : {0., .01, .1, .5, 1., 1.01}) {
        queries.push_back({{current::TableQueryKind::EnergyAfterContinuousLoss, pid, 0, 0,
                            energy, range.value * fraction},
                           false});
        queries.push_back({{current::TableQueryKind::ContinuousEnergy, pid, 0, 0,
                            range.value * fraction, 0.},
                           false});
      }
    }
  for (std::uint32_t kind : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 0xffffffffu})
    for (bool invalid : {false, true})
      queries.push_back(
          {{static_cast<current::TableQueryKind>(kind), 11, 1000000002, 0, 1.e5, .3},
           invalid});
  current::NativeTableStorage<execution_space_type> storage;
  storage.initialize(table);
  native_query_test::UploadedReference<memory_space_type> deviceOld(table);
  auto const native = storage.getDeviceView();
  auto const oldNative = deviceOld.view();
  auto input = upload(queries);
  Kokkos::View<Outcome * [2], Kokkos::LayoutRight, memory_space_type> output(
      "dispatch_results", queries.size());
  Kokkos::parallel_for(
      "query_dispatch", Kokkos::RangePolicy<execution_space_type>(0, queries.size()),
      KOKKOS_LAMBDA(std::size_t index) {
        output(index, 0) = dispatch(native, input(index));
        output(index, 1) = dispatch_reference(oldNative, input(index));
      });
  auto results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  std::uint64_t counts[8]{}, successes[8]{};
  auto const hostNative = current::make_proposal_native_host_view(table);
  for (std::size_t index = 0; index < queries.size(); ++index) {
    INFO("Dispatch " << index << ", kind "
                     << static_cast<unsigned>(queries[index].query_.kind_));
    check(dispatch(hostNative, queries[index]), dispatch_reference(old, queries[index]));
    check(results(index, 0), results(index, 1));
    auto const raw = static_cast<unsigned>(queries[index].query_.kind_);
    auto const kind = raw < 7 ? raw : 7;
    ++counts[kind];
    if (results(index, 0).states_[0] == 0) ++successes[kind];
  }
  for (unsigned kind = 0; kind < 7; ++kind) {
    REQUIRE(counts[kind] > 0);
    REQUIRE(successes[kind] > 0);
  }
  REQUIRE(counts[7] == 4);
  REQUIRE(successes[7] == 0);
  std::ofstream report("dispatch-counts.txt");
  report << "queries " << queries.size() << "\n";
  for (unsigned kind = 0; kind < 8; ++kind)
    report << kind << " " << counts[kind] << " " << successes[kind] << "\n";
}
