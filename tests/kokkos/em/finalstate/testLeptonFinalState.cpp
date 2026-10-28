/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "LeptonFinalStateChecks.hpp"
#ifdef CORSIKA_TEST_LEPTON_NATIVE
#include "../tables/NativeTableFixture.hpp"
#include "../models/AuxiliaryModelFixture.hpp"
#endif

namespace {
  void ensure_runtime() { static corsika::kokkos::Runtime runtime; }
} // namespace

TEST_CASE("Lepton final states keep production routing and invalid-input contracts",
          "[lepton_final_contract]") {
  using namespace lepton_final_test;
  ensure_runtime();
#ifdef CORSIKA_EXPECT_CUDA
  STATIC_REQUIRE(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#endif
#ifdef CORSIKA_EXPECT_OPENMP
  STATIC_REQUIRE(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
  std::vector<Input> cases;
  for (int pid : {11, -11, 13, -13}) {
    for (int process :
         {current::BremsProcessId, current::IonizationProcessId,
          current::AnnihilationProcessId, current::ElectronPairProcessId, -999}) {
      for (double energy : {1.e-7, .001, 1., 100., 1.e8}) {
        for (unsigned variant = 0; variant < 24; ++variant) {
          auto q = input(pid, process, energy);
          q.interaction_.particle_.historyId_ += variant;
          q.interaction_.particle_.generation_ = variant % 2;
          switch (variant) {
            case 1:
              q.interaction_.status_ = current::InteractionStatus::NoDiscreteInteraction;
              break;
            case 2:
              q.interaction_.status_ = current::InteractionStatus::RequiresReselection;
              break;
            case 3:
              q.interaction_.particle_.generation_ = 0xffffffffU;
              break;
            case 4:
              q.interaction_.particle_.stepId_ = 0xffffffffffffffffULL;
              break;
            case 5:
              q.interaction_.energyFraction_ = 0.;
              break;
            case 6:
              q.interaction_.energyFraction_ = 1.;
              break;
            case 7:
              q.interaction_.energyFraction_ = NAN;
              break;
            case 8:
              q.interaction_.particle_.weight_ = -1.;
              break;
            case 9:
              q.interaction_.particle_.energyGeV_ = NAN;
              break;
            case 10:
              q.interaction_.particleMassGeV_ = 0.;
              break;
            case 11:
              q.interaction_.massDensityGPerCm3_ = -1.;
              break;
            case 12:
              q.lpm_.electronMassMeV_ = NAN;
              break;
            case 13:
              q.interaction_.componentHash_ = 0;
              break;
            case 14:
              q.interaction_.particle_.weight_ = 0.;
              break;
            case 15:
              q.thinning_ = {1.e10, 100., 1, 1};
              break;
            case 16:
              q.thinning_ = {1.e10, 3., 1, 1};
              break;
            case 17:
              q.thinning_ = {1.e10, 3., 1, 0};
              break;
            case 18:
              q.thinning_ = {NAN, 3., 1, 0};
              break;
            case 19:
              q.interaction_.particleMassGeV_ *= 1.01;
              break;
            case 20:
              q.usePrepared_ = 1;
              break;
            case 21:
              q.interaction_.status_ = current::InteractionStatus::AtRestAnnihilation;
              break;
            case 22:
              q.interaction_.status_ = current::InteractionStatus::AtRestAnnihilation;
              q.interaction_.particle_.energyGeV_ = current::transport_mass_gev(pid);
              break;
            case 23:
              q.interaction_.interactionVertexReached_ = 0;
              break;
            default:
              break;
          }
          cases.push_back(q);
        }
      }
    }
  }
  unsigned fallback{}, continuation{}, annihilation{}, ionization{}, cpuPair{};
  for (std::size_t i = 0; i < cases.size(); ++i) {
    INFO("lepton contract " << i);
    auto const& q = cases[i];
    check(evaluate(q), expected(q));
    auto c = classify(q);
    fallback += c.fallbackFlag_;
    continuation += c.continuationFlag_;
    annihilation += c.annihilationFlag_;
    ionization += c.ionizationFlag_;
    REQUIRE(c.electronPairFlag_ == 0);
    REQUIRE(c.electronPairRejectionTrials_ == 0);
    if (q.interaction_.processId_ == current::ElectronPairProcessId &&
        q.interaction_.status_ == current::InteractionStatus::Selected) {
      REQUIRE(c.fallbackFlag_ == 1);
      REQUIRE(c.fallback_.reason_ == current::FallbackReason::CpuOnlyProcess);
      ++cpuPair;
    }
  }
  REQUIRE(fallback > 0);
  REQUIRE(continuation > 0);
  REQUIRE(annihilation > 0);
  REQUIRE(ionization > 0);
  REQUIRE(cpuPair > 0);
  compare_device(cases);
  std::ofstream report("lepton-final-contract-counts.txt");
  report << "cases " << cases.size() << '\n'
         << "fallback " << fallback << '\n'
         << "continuation " << continuation << '\n'
         << "annihilation " << annihilation << '\n'
         << "ionization " << ionization << '\n'
         << "cpu_pair " << cpuPair << '\n';
}

TEST_CASE(
    "Stopped annihilation preserves back-to-back photons without enabling the cut option",
    "[lepton_final_at_rest]") {
  using namespace lepton_final_test;
  ensure_runtime();
  std::vector<Input> cases;
  for (unsigned draw = 0; draw < 64; ++draw) {
    auto q = input(-11, current::AnnihilationProcessId, current::transport_mass_gev(-11));
    q.interaction_.status_ = current::InteractionStatus::AtRestAnnihilation;
    q.interaction_.particleMassGeV_ = current::transport_mass_gev(-11);
    q.interaction_.particle_.historyId_ += draw;
    q.thinning_ = {1.e10, 3., 1, 1};
    auto c = classify(q);
    auto state = current::materialize_lepton_final_state(
        q.interaction_, c, q.offset_, q.firstId_, q.lpm_.electronMassMeV_ / 1000.);
    REQUIRE(c.annihilationFlag_ == 1);
    REQUIRE(state.secondaryCount_ == 2);
    REQUIRE(state.error_ == 0);
    auto const& a = state.secondaries_[0];
    auto const& b = state.secondaries_[1];
    REQUIRE(a.pid_ == 22);
    REQUIRE(b.pid_ == 22);
    REQUIRE(a.energyGeV_ == q.interaction_.particle_.energyGeV_);
    REQUIRE(b.energyGeV_ == a.energyGeV_);
    REQUIRE(a.weight_ == q.interaction_.particle_.weight_);
    REQUIRE(b.weight_ == a.weight_);
    double norm{};
    for (unsigned k = 0; k < 3; ++k) {
      REQUIRE(a.direction_[k] == -b.direction_[k]);
      norm += a.direction_[k] * a.direction_[k];
    }
    REQUIRE(std::abs(norm - 1.) < 2.e-15);
    check(evaluate(q), expected(q));
    cases.push_back(q);
  }
  compare_device(cases);
}

#ifdef CORSIKA_TEST_LEPTON_NATIVE
TEST_CASE("Muon auxiliary failures and zero-weight EM output keep production decisions",
          "[lepton_final_auxiliary]") {
  using namespace lepton_final_test;
  ensure_runtime();
  auto const auxiliary = current::test::make_auxiliary_dataset();
  auto const lpm = auxiliary.snapshots_.front().electron_;
  std::vector<Input> cases;
  for (int pid : {13, -13}) {
    for (unsigned variant = 0; variant < 5; ++variant) {
      auto q = input(pid, current::BremsProcessId, 10.);
      q.lpm_ = lpm;
      q.interaction_.particleMassGeV_ = lpm.muonMassMeV_ / 1000.;
      q.interaction_.componentHash_ = lpm.components_[0].componentHash_;
      if (variant == 0) q.lpm_.muonELpmMeV_ = 0.;
      if (variant == 1) q.lpm_.muonELpmMeV_ = NAN;
      if (variant == 2) q.lpm_.muonMassMeV_ = 0.;
      if (variant == 3) q.interaction_.particleMassGeV_ *= 1.01;
      if (variant == 4) q.interaction_.particleMassGeV_ = 0.;
      auto c = classify(q);
      REQUIRE(c.fallbackFlag_ == 1);
      REQUIRE(c.fallback_.reason_ == current::FallbackReason::LpmParametersUnavailable);
      cases.push_back(q);
      q.interaction_.processId_ = current::ElectronPairProcessId;
      c = current::classify_muon_radiative_final_state(
          q.lpm_, q.thinning_, q.interaction_, q.seed_, q.shower_);
      REQUIRE(c.fallbackFlag_ == 1);
      REQUIRE(c.fallback_.reason_ == current::FallbackReason::CpuOnlyProcess);
      cases.push_back(q);
    }
  }
  for (int pid : {11, -11})
    for (unsigned erase : {0U, 1U}) {
      auto q = input(pid, current::IonizationProcessId, 10.);
      q.lpm_ = lpm;
      q.interaction_.particle_.weight_ = 0.;
      q.thinning_ = {1.e10, 100., 1, erase};
      auto c = classify(q);
      auto state = current::materialize_lepton_final_state(
          q.interaction_, c, q.offset_, q.firstId_, lpm.electronMassMeV_ / 1000.);
      REQUIRE(c.recordFlag_ == 1);
      REQUIRE(c.childCount_ == (erase ? 0 : 2));
      REQUIRE(state.secondaryCount_ == c.childCount_);
      REQUIRE(state.hasFirstInteraction_ == 1);
      REQUIRE(state.firstInteraction_.secondaryCount_ == 2);
      cases.push_back(q);
    }
  for (auto const& q : cases) check(evaluate(q), expected(q));
  compare_device(cases);
}

TEST_CASE("Native energy losses and real auxiliary parameters retain lepton final states",
          "[lepton_final_native]") {
  using namespace lepton_final_test;
  ensure_runtime();
  auto const table = current::test::make_native_tables();
  auto const native = current::make_proposal_native_host_view(table);
  auto const auxiliary = current::test::make_auxiliary_dataset();
  auto const lpm = auxiliary.snapshots_.front().electron_;
  std::vector<Input> cases;
  unsigned queries{}, unavailable{};
  for (auto const& column : table.dndxColumns_) {
    auto const pid = column.pdgId_, process = column.processId_;
    if (!current::is_charged_lepton_pid(pid)) continue;
    if (process != current::BremsProcessId && process != current::IonizationProcessId &&
        process != current::AnnihilationProcessId &&
        process != current::ElectronPairProcessId)
      continue;
    for (double energy : {.01, .1, 1., 10., 1.e3, 1.e5})
      for (unsigned draw = 0; draw < 16; ++draw) {
        auto q = input(pid, process, energy);
        q.lpm_ = lpm;
        q.interaction_.componentHash_ = column.componentHash_;
        q.interaction_.particleMassGeV_ = current::is_muon_pid(pid)
                                              ? lpm.muonMassMeV_ / 1000.
                                              : lpm.electronMassMeV_ / 1000.;
        auto const u = (draw + .5) / 16.;
        if (process == current::AnnihilationProcessId)
          q.interaction_.energyFraction_ = 1.;
        else {
          ++queries;
          auto loss = current::query_discrete_loss_fraction(
              native, pid, process, q.interaction_.componentHash_, energy * 1000., u);
          if (loss.status_ != current::TableLookupStatus::Success) {
            ++unavailable;
            continue;
          }
          q.interaction_.energyFraction_ = loss.value_;
        }
        q.seed_ += draw;
        q.interaction_.particle_.historyId_ += draw;
        q.interaction_.particle_.generation_ = draw % 2;
        q.interaction_.lossQuantile_ = u;
        q.interaction_.massDensityGPerCm3_ = draw % 3 ? .0012 : 1.e6;
        if (draw % 4 == 1) q.thinning_ = {1.e10, 100., 1, 1};
        if (draw % 4 == 2) q.thinning_ = {1.e10, 3., 1, 1};
        if (draw % 4 == 3) q.thinning_ = {1.e10, 3., 1, 0};
        cases.push_back(q);
        // Same supplied prepared terms in both ports; preparation kernels
        // are outside this slice. Values exercise the existing overload.
        q.usePrepared_ = 1;
        q.prepared_.valid_ = 1;
        q.prepared_.componentCount_ = lpm.componentCount_;
        for (unsigned k = 0; k < lpm.componentCount_; ++k)
          q.prepared_.components_[k] = {lpm.components_[k].componentHash_, .001,
                                        std::log(.001)};
        cases.push_back(q);
      }
  }
  unsigned suppressed{}, fallback{}, first{}, kept[4]{};
  std::map<int, unsigned> channels, pids;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    INFO("real lepton vertex " << i);
    auto const& q = cases[i];
    check(evaluate(q), expected(q));
    auto c = classify(q);
    auto state = current::materialize_lepton_final_state(
        q.interaction_, c, q.offset_, q.firstId_, q.lpm_.electronMassMeV_ / 1000.);
    suppressed += c.suppressionFlag_;
    fallback += c.fallbackFlag_;
    first += state.hasFirstInteraction_;
    REQUIRE(c.electronPairFlag_ == 0);
    REQUIRE(c.electronPairRejectionTrials_ == 0);
    if (c.recordFlag_) {
      REQUIRE(state.error_ == 0);
      REQUIRE(state.secondaryCount_ == c.childCount_);
      ++channels[c.parameters_.processId_];
      ++pids[q.interaction_.particle_.pid_];
      ++kept[state.secondaryCount_];
      for (unsigned k = 0; k < state.secondaryCount_; ++k) {
        auto const& child = state.secondaries_[k];
        REQUIRE(child.historyId_ == q.firstId_ + q.offset_ + k);
        REQUIRE(child.parentHistoryId_ == q.interaction_.particle_.historyId_);
        REQUIRE(child.environmentNode_ == q.interaction_.particle_.environmentNode_);
        double norm{};
        for (double x : child.direction_) norm += x * x;
        REQUIRE(std::abs(norm - 1.) < 2.e-14);
      }
    }
  }
  std::ofstream report("lepton-final-native-counts.txt");
  report << "cases " << cases.size() << '\n'
         << "queries " << queries << '\n'
         << "unavailable " << unavailable << '\n'
         << "suppressed " << suppressed << '\n'
         << "fallback " << fallback << '\n'
         << "first " << first << '\n';
  for (auto const& p : channels)
    report << "process_" << p.first << ' ' << p.second << '\n';
  for (auto const& p : pids) report << "pid_" << p.first << ' ' << p.second << '\n';
  for (unsigned i = 0; i < 4; ++i) report << "children_" << i << ' ' << kept[i] << '\n';
  report.close();
  REQUIRE(cases.size() > 1000);
  REQUIRE(suppressed > 0);
  REQUIRE(fallback > 0);
  REQUIRE(first > 0);
  for (int process : {current::BremsProcessId, current::IonizationProcessId,
                      current::AnnihilationProcessId}) {
    INFO("reaction coverage " << process);
    REQUIRE(channels[process] > 0);
  }
  for (int pid : {11, -11, 13, -13}) {
    INFO("projectile coverage " << pid);
    REQUIRE(pids[pid] > 0);
  }
  compare_device(cases);
}
#endif
