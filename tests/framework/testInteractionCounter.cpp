/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/process/InteractionCounter.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>

#include <catch2/catch_all.hpp>

#include <numeric>
#include <algorithm>
#include <iterator>
#include <string>
#include <fstream>
#include <cstdio>

using namespace corsika;

const std::string refDataDir = std::string(REFDATADIR); // from cmake

struct DummyProcess {

  CrossSectionType getCrossSection(Code const, Code const, FourMomentum const&,
                                   FourMomentum const&) {
    return 100_mb;
  }

  template <typename TParticle>
  void doInteraction(TParticle&, Code const, Code const, FourMomentum const&,
                     FourMomentum const&) {}
};

struct DummyOutput {
  /* can do nothing */
};

struct DeferredDummyProcess {
  static constexpr HadronicWorkerModel
      hadronic_worker_model =
          HadronicWorkerModel::Fluka;
  bool called{};

  CrossSectionType getCrossSection(
      Code const, Code const,
      FourMomentum const&,
      FourMomentum const&) {
    return 100_mb;
  }

  template <typename TParticle>
  void doInteraction(
      TParticle&, Code const, Code const,
      FourMomentum const&,
      FourMomentum const&) {
    called = true;
  }
};

TEST_CASE("InteractionCounter captures worker-capable final state",
          "[HadronicInteractionDeferral]") {
  auto const rootCS = get_root_CoordinateSystem();
  DummyOutput output;
  DeferredDummyProcess process;
  InteractionCounter counted{process};
  HadronicRandomKey const key{
      1234, 2, 77, 9, 4, 0};
  HadronicInteractionDeferralContext context{
      key, 19};

  {
    ScopedHadronicInteractionDeferral scope{
        context};
    counted.doInteraction(
        output, Code::Proton, Code::Oxygen,
        {10_GeV,
         {rootCS, {0_GeV, 0_GeV, 9_GeV}}},
        {Oxygen::mass,
         {rootCS, {0_GeV, 0_GeV, 0_GeV}}});
  }

  CHECK_FALSE(process.called);
  CHECK(context.hasPrepared());
  auto prepared = context.takePrepared();
  CHECK(
      prepared.request.model ==
      HadronicWorkerModel::Fluka);
  CHECK(prepared.request.sequence_id == 19);
  CHECK(prepared.request.random_key.history_id == 77);
  CHECK(prepared.request.projectile_pdg == 2212);
  CHECK(
      prepared.request.target_pdg ==
      static_cast<std::int32_t>(
          get_PDG(Code::Oxygen)));
  CHECK(
      prepared.request.projectile_four_momentum_GeV[3] ==
      Catch::Approx(9.));
  CHECK(counted.getCount() == 1);
  REQUIRE(counted.getTimingSamples().size() == 1);
  CHECK(counted.getTimingSamples().front().deferred);
  CHECK(
      counted.getTimingSamples().front()
          .final_state_time_ms == 0.);
  CHECK(
      counted.getEnergyLedgerStatistics()
          .deferred_interactions == 1);
  CHECK(
      counted.getEnergyLedgerStatistics()
          .audited_interactions == 0);
  counted.releaseTimingSamples();
  CHECK(counted.getTimingSamples().empty());
  CHECK(counted.getTimingSamples().capacity() == 0);
  CHECK(counted.getCount() == 1);
  CHECK(counted.getEnergyLedgerStatistics().deferred_interactions == 1);
  auto const accumulated_time = counted.getTotalFinalStateTimeMs();
  counted.doInteraction(
      output, Code::Proton, Code::Oxygen,
      {10_GeV, {rootCS, {0_GeV, 0_GeV, 9_GeV}}},
      {Oxygen::mass, {rootCS, {0_GeV, 0_GeV, 0_GeV}}});
  REQUIRE(counted.getTimingSamples().size() == 1);
  CHECK(counted.getTimingSamples().front().sequence_id == 1);
  CHECK(counted.getCount() == 2);
  CHECK(counted.getTotalFinalStateTimeMs() >= accumulated_time);
}

TEST_CASE("InteractionCounter", "process") {

  logging::set_level(logging::level::info);

  DummyProcess d;
  InteractionCounter countedProcess(d);
  CHECK(countedProcess.getCount() == 0);
  CHECK(countedProcess.getTimingSamples().empty());
  CHECK(countedProcess.getTotalFinalStateTimeMs() == 0.);
  CHECK(
      countedProcess.getEnergyLedgerStatistics()
          .audited_interactions == 0);

  auto const rootCS = get_root_CoordinateSystem();
  DummyOutput output;

  SECTION("cross section pass-through") {
    CHECK(countedProcess.getCrossSection(
              Code::Oxygen, Code::Proton, {10_GeV, {rootCS, {0_eV, 0_eV, 0_eV}}},
              {10_GeV, {rootCS, {0_eV, 0_eV, 0_eV}}}) == 100_mb);
  }

  SECTION("doInteraction nucleus") {
    unsigned short constexpr A = 14, Z = 7;
    Code const pid = get_nucleus_code(A, Z);

    countedProcess.doInteraction(
        output, pid, Code::Oxygen,
        {sqrt(static_pow<2>(105_TeV) + static_pow<2>(get_mass(pid))),
         {rootCS, {105_TeV, 0_GeV, 0_GeV}}},
        {Oxygen::mass, {rootCS, {0_eV, 0_eV, 0_eV}}});
    CHECK(countedProcess.getCount() == 1);
    REQUIRE(countedProcess.getTimingSamples().size() == 1);
    auto const& timing = countedProcess.getTimingSamples().front();
    CHECK(timing.sequence_id == 0);
    CHECK(timing.projectile == pid);
    CHECK(timing.target == Code::Oxygen);
    CHECK(timing.kinetic_energy > 0_GeV);
    CHECK(timing.final_state_time_ms >= 0.);
    CHECK(countedProcess.getTotalFinalStateTimeMs() ==
          Catch::Approx(timing.final_state_time_ms));

    auto const& h = countedProcess.getHistogram().labHist();
    CHECK(h.at(h.axis(0).index(1'000'070'140), h.axis(1).index(1.05e14)) == 1);
    CHECK(std::accumulate(h.cbegin(), h.cend(), 0) == 1);

    auto const& h2 = countedProcess.getHistogram().CMSHist();
    CHECK(h2.at(h2.axis(0).index(1'000'070'140), h2.axis(1).index(1.6e12)) == 1);
    CHECK(std::accumulate(h2.cbegin(), h2.cend(), 0) == 1);

    save_hist(countedProcess.getHistogram().labHist(), "testInteractionCounter_file1.npz",
              true);
    save_hist(countedProcess.getHistogram().CMSHist(), "testInteractionCounter_file2.npz",
              true);

    SECTION("output validation") {
      auto const file = GENERATE(as<std::string>{}, "testInteractionCounter_file1",
                                 "testInteractionCounter_file2");

      CORSIKA_LOG_INFO("{0}.npz vs {1}/{0}_REF.npz", file, refDataDir);

      // compare to binary reference data
      // note that this currenly compares the whole files byte by byte. If the new
      // or the reference file are compressed this would be a false-negative outcome
      // of this test
      std::ifstream file1(file + ".npz");
      std::ifstream file1ref(refDataDir + "/" + file + "_REF.npz");

      std::istreambuf_iterator<char> begin1(file1);
      std::istreambuf_iterator<char> begin1ref(file1ref);

      std::istreambuf_iterator<char> end;

      CHECK(std::equal(begin1, end, begin1ref));
    }
  }

  SECTION("doInteraction Lambda") {
    auto constexpr pid = Code::Lambda;

    countedProcess.doInteraction(
        output, pid, Code::Oxygen,
        {sqrt(static_pow<2>(105_TeV) + static_pow<2>(get_mass(pid))),
         {rootCS, {105_TeV, 0_GeV, 0_GeV}}},
        {Oxygen::mass, {rootCS, {0_eV, 0_eV, 0_eV}}});
    CHECK(countedProcess.getCount() == 1);
    REQUIRE(countedProcess.getTimingSamples().size() == 1);
    CHECK(countedProcess.getTimingSamples().front().projectile ==
          pid);

    auto const& h = countedProcess.getHistogram().labHist();
    CHECK(h.at(h.axis(0).index(3122), h.axis(1).index(1.05e14)) == 1);
    CHECK(std::accumulate(h.cbegin(), h.cend(), 0) == 1);

    auto const& h2 = countedProcess.getHistogram().CMSHist();
    CHECK(h2.at(h2.axis(0).index(3122), h2.axis(1).index(1.6e12)) == 1);
    CHECK(std::accumulate(h2.cbegin(), h2.cend(), 0) == 1);
  }
}
