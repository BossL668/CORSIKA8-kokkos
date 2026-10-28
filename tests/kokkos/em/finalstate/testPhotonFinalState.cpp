/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "PhotonFinalStateChecks.hpp"
#ifdef CORSIKA_TEST_PHOTON_NATIVE
#include "../tables/NativeTableFixture.hpp"
#include "../models/AuxiliaryModelFixture.hpp"
#endif

namespace {
  void ensure_runtime() { static corsika::kokkos::Runtime runtime; }
} // namespace

TEST_CASE("Photon final states retain deferred fallback and local reaction contracts",
          "[photon_final_contract]") {
  using namespace photon_final_test;
  ensure_runtime();
#ifdef CORSIKA_EXPECT_CUDA
  STATIC_REQUIRE(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#endif
#ifdef CORSIKA_EXPECT_OPENMP
  STATIC_REQUIRE(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
  std::vector<Input> cases;
  for (int process : {current::PhotonPairProcessId, current::ComptonProcessId,
                      current::PhotoelectricProcessId, current::PhotonMuonPairProcessId,
                      current::PhotonuclearProcessId, -999}) {
    for (double energy : {1.e-7, .001, 1., 100., 1.e8}) {
      for (unsigned variant = 0; variant < 32; ++variant) {
        auto q = input(process, energy);
        q.interaction_.particle_.historyId_ += variant;
        q.interaction_.particle_.generation_ = variant % 2;
        if (process == current::PhotoelectricProcessId)
          q.interaction_.energyFraction_ = 1.;
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
            q.lpm_.componentCount_ = 0;
            break;
          case 11:
            q.lpm_.componentCount_ = current::MaxPhotonPairLpmComponents + 1;
            break;
          case 12:
            q.lpm_.fineStructureConstant_ = NAN;
            break;
          case 13:
            q.lpm_.components_[0].nuclearCharge_ = -1.;
            break;
          case 14:
            q.interaction_.componentHash_ = 0;
            break;
          case 15:
            q.interaction_.particle_.weight_ = 0.;
            break;
          case 16:
            q.thinning_ = {1.e10, 100., 1, 1};
            break;
          case 17:
            q.thinning_ = {1.e10, 3., 1, 1};
            break;
          case 18:
            q.thinning_ = {1.e10, 3., 1, 0};
            break;
          case 19:
            q.thinning_ = {NAN, 3., 1, 0};
            break;
          default:
            break;
        }
        if (variant >= 20) {
          q.interaction_.deferredPhotonFallbackReason_ = static_cast<std::int32_t>(
              variant % 2 ? current::FallbackReason::CpuOnlyProcess
                          : current::FallbackReason::InvalidFinalState);
          q.interaction_.interactionVertexReached_ = variant % 3 == 0 ? 0 : 1;
          q.interaction_.deferredPhotonTableStatus_ = variant % 4;
        }
        cases.push_back(q);
      }
    }
  }
  std::map<unsigned, unsigned> flags;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    INFO("empty-table contract " << i);
    check(evaluate({}, cases[i]), expected({}, cases[i]));
    auto const& q = cases[i];
    auto c = current::classify_photon_final_state({}, q.lpm_, q.thinning_, q.interaction_,
                                                  q.seed_, q.shower_);
    flags[0] += c.fallbackFlag_;
    flags[1] += c.continuationFlag_;
    flags[2] += c.comptonFlag_;
    flags[3] += c.photoelectricFlag_;
  }
  for (unsigned i = 0; i < 4; ++i) REQUIRE(flags[i] > 0);
  compare_device({}, {}, cases);
  std::ofstream report("photon-final-contract-counts.txt");
  report << "cases " << cases.size() << '\n';
  for (auto const& item : flags)
    report << "flag_" << item.first << ' ' << item.second << '\n';
}

TEST_CASE(
    "Muon-pair materialization preserves native sub-rest-mass cut and energy convention",
    "[photon_final_cut]") {
  using namespace photon_final_test;
  ensure_runtime();
  unsigned cuts{};
  for (double energy : {.05, .2, 1., 1.e5}) {
    for (double fraction : {.001, .1, .5, .9, .999}) {
      auto q = input(current::PhotonMuonPairProcessId, energy);
      current::PhotonFinalStateClassification c{};
      reference::PhotonFinalStateClassification old{};
      c.recordFlag_ = old.record_flag = 1;
      c.childCount_ = old.child_count = 2;
      c.photonMuonPairFlag_ = old.photon_muon_pair_flag = 1;
      c.parameters_.processId_ = old.parameters.process_id =
          current::PhotonMuonPairProcessId;
      c.parameters_.secondaryMassGeV_ = old.parameters.secondary_mass_GeV = .1056583745;
      c.parameters_.splitFraction_ = old.parameters.split_fraction = fraction;
      c.parameters_.azimuthUniform_ = old.parameters.azimuth_uniform = .375;
      c.parameters_.thinningFirstWeight_ = old.parameters.thinning_first_weight = 2.;
      c.parameters_.thinningSecondWeight_ = old.parameters.thinning_second_weight = 2.;
      auto const actual = current::materialize_photon_final_state(q.interaction_, c,
                                                                  q.offset_, q.firstId_);
      auto const expect = reference::materializePhotonFinalState(
          continuous_test::convert(q.interaction_), old, q.offset_, q.firstId_);
      Payload a, b;
      flatten(a, actual);
      flatten(b, expect);
      check(a, b);
      REQUIRE(actual.secondaryCount_ == 2);
      REQUIRE(actual.hasFirstInteraction_ == 1);
      auto const total = actual.secondaries_[0].energyGeV_ * 2. +
                         actual.secondaries_[1].energyGeV_ * 2. -
                         actual.record_.weightedMassConventionCorrectionGeV_;
      REQUIRE(std::abs(total - energy * 2.) <=
              8. * std::numeric_limits<double>::epsilon() * energy * 2.);
      for (unsigned k = 0; k < 2; ++k) {
        auto const& p = actual.secondaries_[k];
        REQUIRE(p.historyId_ == q.firstId_ + q.offset_ + k);
        REQUIRE(p.parentHistoryId_ == q.interaction_.particle_.historyId_);
        if (p.energyGeV_ <= current::transport_mass_gev(p.pid_)) {
          REQUIRE(p.reserved_ == current::NativeMuonPairCutFlag);
          if (p.energyGeV_ > 0.) REQUIRE(current::is_native_muon_pair_cut(p));
          ++cuts;
        } else
          REQUIRE(p.reserved_ == 0);
      }
    }
  }
  REQUIRE(cuts > 0);
}

#ifdef CORSIKA_TEST_PHOTON_NATIVE
TEST_CASE("Real PROPOSAL photon CDFs produce unchanged children and fallback diagnostics",
          "[photon_final_native]") {
  using namespace photon_final_test;
  ensure_runtime();
  auto const table = current::test::make_native_tables();
  auto const native = current::make_proposal_native_host_view(table);
  native_query_test::HostReference hostOld(table);
  auto const oldNative = hostOld.view(table);
  auto const auxiliary = current::test::make_auxiliary_dataset();
  auto const lpm = auxiliary.snapshots_.front().photon_;
  std::vector<Input> cases;
  for (unsigned component = 0; component < lpm.componentCount_; ++component) {
    for (int process : {current::PhotonPairProcessId, current::PhotonMuonPairProcessId,
                        current::ComptonProcessId, current::PhotoelectricProcessId}) {
      for (double energy : {.002, .01, .1, 1., 10., 1.e3, 1.e5}) {
        for (unsigned draw = 0; draw < 32; ++draw) {
          auto q = input(process, energy, lpm.components_[component].componentHash_);
          q.lpm_ = lpm;
          q.seed_ += draw;
          q.shower_ += draw % 3;
          q.interaction_.particle_.historyId_ += draw;
          q.interaction_.particle_.generation_ = draw % 2;
          q.interaction_.energyFraction_ =
              process == current::PhotoelectricProcessId ? 1. : .25;
          q.interaction_.massDensityGPerCm3_ = draw % 3 ? .0012 : 1.e6;
          if (draw % 4 == 1) q.thinning_ = {1.e10, 100., 1, 1};
          if (draw % 4 == 2) q.thinning_ = {1.e10, 3., 1, 1};
          if (draw % 4 == 3) q.thinning_ = {1.e10, 3., 1, 0};
          cases.push_back(q);
        }
      }
    }
  }
  // Retain explicit out-of-table and invalid-model checks with otherwise valid tables.
  for (int process : {current::PhotonPairProcessId, current::PhotonMuonPairProcessId})
    for (unsigned variant = 0; variant < 8; ++variant) {
      auto q = input(process, 1.e3, lpm.components_[0].componentHash_);
      q.lpm_ = lpm;
      if (variant == 0) q.interaction_.componentHash_ = 0;
      if (variant == 1) q.interaction_.particle_.energyGeV_ = 1.e10;
      if (variant == 2) q.lpm_.componentCount_ = 0;
      if (variant == 3) q.interaction_.massDensityGPerCm3_ = -1.;
      if (variant == 4) q.interaction_.particle_.weight_ = NAN;
      if (variant == 5) q.interaction_.particle_.energyGeV_ = 0.;
      if (variant == 6) q.thinning_ = {1.e10, -1., 1, 1};
      if (variant == 7) q.interaction_.particle_.weight_ = 0.;
      cases.push_back(q);
    }
  std::map<int, unsigned> channels;
  unsigned suppressed{}, failures{}, first{}, kept[3]{};
  for (std::size_t i = 0; i < cases.size(); ++i) {
    INFO("real-table photon vertex " << i);
    auto const& q = cases[i];
    check(evaluate(native, q), expected(oldNative, q));
    auto const c = current::classify_photon_final_state(
        current::ContinuousPhysicsView(native, .5, 10.), q.lpm_, q.thinning_,
        q.interaction_, q.seed_, q.shower_);
    auto const m =
        current::materialize_photon_final_state(q.interaction_, c, q.offset_, q.firstId_);
    failures += c.fallbackFlag_;
    suppressed += c.suppressionFlag_;
    first += m.hasFirstInteraction_;
    REQUIRE(m.error_ == 0);
    if (c.recordFlag_) {
      ++channels[c.parameters_.processId_];
      REQUIRE(m.secondaryCount_ == c.childCount_);
      REQUIRE(m.secondaryCount_ <= 2);
      ++kept[m.secondaryCount_];
      for (unsigned k = 0; k < m.secondaryCount_; ++k) {
        auto const& p = m.secondaries_[k];
        REQUIRE(p.historyId_ == q.firstId_ + q.offset_ + k);
        REQUIRE(p.environmentNode_ == q.interaction_.particle_.environmentNode_);
        double norm{};
        for (double x : p.direction_) norm += x * x;
        REQUIRE(std::abs(norm - 1.) < 2.e-14);
      }
    }
    if (c.suppressionFlag_) {
      REQUIRE(m.hasSuppression_ == 1);
      REQUIRE(m.secondaryCount_ == 0);
      REQUIRE(m.suppression_.particle_.stepId_ == q.interaction_.particle_.stepId_ + 1);
    }
  }
  for (int process : {current::PhotonPairProcessId, current::PhotonMuonPairProcessId,
                      current::ComptonProcessId, current::PhotoelectricProcessId})
    REQUIRE(channels[process] > 0);
  REQUIRE(suppressed > 0);
  REQUIRE(failures > 0);
  REQUIRE(first > 0);
  for (unsigned count : kept) REQUIRE(count > 0);
  current::NativeTableStorage<execution_space_type> storage;
  storage.initialize(table);
  native_query_test::UploadedReference<memory_space_type> deviceOld(table);
  compare_device(storage.getDeviceView(), deviceOld.view(), cases);
  std::ofstream report("photon-final-native-counts.txt");
  report << "cases " << cases.size() << "\nsuppressed " << suppressed << "\nfallback "
         << failures << "\nfirst " << first << '\n';
  for (auto const& item : channels)
    report << "process_" << item.first << ' ' << item.second << '\n';
  for (unsigned i = 0; i < 3; ++i) report << "children_" << i << ' ' << kept[i] << '\n';
}
#endif
