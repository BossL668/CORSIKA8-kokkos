/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "SelectionChecks.hpp"
#include "../tables/NativeTableFixture.hpp"
#include <corsika/kokkos/em/execution/NativeTableStorage.hpp>
#include <map>

TEST_CASE("Real native tables retain distance and vertex selection on host and device",
          "[selection_native]") {
  using namespace selection_test;
  current::Runtime runtime;
  auto const table = current::test::make_native_tables();
  auto const native = current::make_proposal_native_host_view(table);
  native_query_test::HostReference hostOld(table);
  auto const oldNative = hostOld.view(table);
  std::vector<Input> cases;
  for (auto const& rateColumn : table.totalRateColumns_) {
    auto const pid = rateColumn.pdgId_;
    auto const low = rateColumn.spline_.axis_.low_;
    auto const high = rateColumn.spline_.axis_.high_;
    for (double energy :
         {low, std::nextafter(low, 0.), std::nextafter(low, high), 1., 10., 100., 1.e3,
          1.e5, 1.e7, 1.e9, high, std::nextafter(high, HUGE_VAL)})
      for (std::uint64_t sample = 0; sample < 64; ++sample) {
        auto q = input(pid, energy);
        q.candidate_.particle_.historyId_ += sample;
        q.seed_ += sample % 7;
        q.shower_ += sample % 3;
        cases.push_back(q);
        if (current::is_charged_lepton_pid(pid)) {
          // Distance record is produced independently, then a prescribed
          // post-transport energy/vertex is supplied: no transport is invented here.
          auto const old = reference::selectDiscreteInteraction(
              old_physics(oldNative, q), continuous_test::convert(q.candidate_.particle_),
              q.candidate_.inputIndex_, q.seed_, q.shower_);
          q.vertex_ = true;
          q.candidate_.interactionVertexReached_ = 1;
          q.candidate_.totalRateCm2PerG_ = old.interaction.total_rate_cm2_per_g;
          q.candidate_.processUniform_ = old.interaction.process_uniform;
          // Keep both accepted and rejected vertices, including changed energy.
          if (sample % 4 == 0) q.candidate_.totalRateCm2PerG_ *= 4.;
          if (sample % 4 == 1) q.candidate_.particle_.energyGeV_ *= .95;
          q.source_ = static_cast<std::uint32_t>(sample % 3);
          cases.push_back(q);
        }
      }
  }
  // Find a deterministic real-table draw for the relatively rare muon pair
  // channel. Candidate construction uses only the independent production oracle.
  for (int pid : {13, -13}) {
    auto q = input(pid, 1.e7, true);
    auto const oldView = old_physics(oldNative, q);
    auto const rate = reference::table::queryTotalRate(oldView, pid, 1.e7);
    REQUIRE(rate.status == reference::table::TableLookupStatus::Success);
    q.candidate_.totalRateCm2PerG_ = rate.value;
    bool found = false;
    for (std::uint64_t trial = 0; trial < 65536 && !found; ++trial) {
      q.seed_ = trial;
      auto const value = reference::selectLeptonVertex(
          oldView, continuous_test::convert(q.candidate_), q.seed_, q.shower_);
      if (value.fallback_flag &&
          value.fallback.process_id == current::ElectronPairProcessId) {
        cases.push_back(q);
        found = true;
      }
    }
    REQUIRE(found);
  }
  std::map<std::int32_t, std::uint64_t> selected, replay;
  std::uint64_t continuations{}, deferred{}, decays{}, cuts{}, failures{}, newton{},
      bisection{};
  for (std::size_t index = 0; index < cases.size(); ++index) {
    auto const& q = cases[index];
    INFO("Host selection case " << index << ", pid " << q.candidate_.particle_.pid_);
    check(evaluate(native, q), expected(oldNative, q));
    if (q.vertex_) {
      auto const value = current::select_lepton_vertex(physics(native, q), q.candidate_,
                                                       q.seed_, q.shower_);
      if (value.interactionFlag_) ++selected[value.record_.processId_];
      if (value.fallback_.reason_ == current::FallbackReason::NativeSelectionReplay &&
          value.fallbackFlag_)
        ++replay[q.candidate_.particle_.pid_];
      continuations += value.continuationFlag_;
      failures += value.fallbackFlag_;
      newton += value.nativeNewtonIterations_;
      bisection += value.nativeBisectionIterations_;
    } else {
      auto const value = current::select_discrete_interaction(
          physics(native, q), q.candidate_.particle_, q.candidate_.inputIndex_, q.seed_,
          q.shower_);
      auto const& r = value.interaction_;
      if (r.deferredPhotonFallbackReason_ >= 0)
        ++deferred;
      else if (!value.fallbackFlag_ && r.status_ == current::InteractionStatus::Selected)
        ++selected[r.processId_];
      if (r.status_ == current::InteractionStatus::ParticleCut) ++cuts;
      if (r.decayUniform_ > 0.) {
        ++decays;
        REQUIRE(r.decayUniform_ < 1.);
        REQUIRE(r.decayDistanceM_ > 0.);
      }
      if (r.status_ == current::InteractionStatus::DistanceSampled) {
        REQUIRE(r.processId_ == 0);
        REQUIRE(r.proposalSelectionUniform_ == 0.);
      }
      failures += value.fallbackFlag_;
      newton += value.nativeNewtonIterations_;
      bisection += value.nativeBisectionIterations_;
    }
  }
  REQUIRE(continuations > 0);
  // Rare photon completions are forced by testSelectionContracts; this finite
  // real-table sample is not required to encounter them at random.
  REQUIRE(decays > 0);
  REQUIRE(cuts > 0);
  REQUIRE(failures > 0);
  REQUIRE(newton > 0);
  REQUIRE(selected[current::BremsProcessId] > 0);
  REQUIRE(selected[current::IonizationProcessId] > 0);
  REQUIRE(selected[current::PhotonPairProcessId] > 0);
  REQUIRE(selected[current::ComptonProcessId] > 0);
  REQUIRE(replay[11] + replay[-11] > 0);
  REQUIRE(replay[13] + replay[-13] > 0);
  current::NativeTableStorage<execution_space_type> storage;
  storage.initialize(table);
  native_query_test::UploadedReference<memory_space_type> deviceOld(table);
  compare_device(storage.getDeviceView(), deviceOld.view(), cases);
  std::ofstream report("selection-native-counts.txt");
  report << "queries " << cases.size() << "\ncontinuations " << continuations
         << "\ndeferred_photon " << deferred << "\nmuon_decay " << decays << "\ncuts "
         << cuts << "\nfallback " << failures << "\nnewton " << newton << "\nbisection "
         << bisection << "\n";
  for (auto const& item : selected)
    report << "selected " << item.first << " " << item.second << "\n";
  for (auto const& item : replay)
    report << "replay_pid " << item.first << " " << item.second << "\n";
}
