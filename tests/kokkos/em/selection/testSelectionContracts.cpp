/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "SelectionChecks.hpp"

namespace selection_test {
  // Controlled branch fixture only: not a physical cross-section fit.
  current::ProposalNativeTableSet control(std::int32_t pid, std::int32_t process,
                                          double total = 2., double factor = 1.) {
    current::ProposalNativeTableSet table;
    table.dndxColumns_.resize(1);
    auto& column = table.dndxColumns_[0];
    column.pdgId_ = pid;
    column.processId_ = process;
    column.componentHash_ = 123;
    column.rateModel_ = current::NativeRateModel::AnnihilationHeitler;
    column.particleMassMeV_ = 1.;
    column.directRateFactor_ = factor;
    table.selectionColumnIndices_ = {0};
    table.totalRateColumns_.resize(1);
    table.totalRateColumns_[0].pdgId_ = pid;
    auto& spline = table.totalRateColumns_[0].spline_;
    spline.axis_ = {current::NativeAxisType::Linear, 2, 1., 1.e9, 1.e9 - 1.};
    spline.coefficientCount_ = 2;
    table.cubicValues_ = {total, total};
    table.cubicNodeDerivatives_ = {0., 0.};
    table.bicubicValues_ = {0.};
    table.bicubicDerivativeEnergy_ = {0.};
    table.bicubicDerivativeLoss_ = {0.};
    table.bicubicMixedDerivative_ = {0.};
    return table;
  }
} // namespace selection_test

TEST_CASE("Selection preserves early cuts, failure provenance and device annotations",
          "[selection]") {
  using namespace selection_test;
  current::Runtime runtime;
  std::vector<Input> cases;
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  auto const inf = std::numeric_limits<double>::infinity();
  for (int pid : {11, -11, 13, -13, 22, 999})
    for (bool vertex : {false, true})
      for (double energy : {-1., 0., .05, .511, 105.658, 1.e3, inf, nan})
        for (double time : {0., .01, std::nextafter(.01, 1.), inf, nan}) {
          auto q = input(pid, energy, vertex);
          q.candidate_.particle_.timeS_ = time;
          cases.push_back(q);
        }
  for (int status = 0; status <= 6; ++status)
    for (double uniform : {-1., 0., .5, 1., inf, nan})
      for (double rate : {-1., 0., 1., inf, nan}) {
        auto q = input(-11, 100., true);
        q.candidate_.status_ = static_cast<current::InteractionStatus>(status);
        q.candidate_.processUniform_ = uniform;
        q.candidate_.totalRateCm2PerG_ = rate;
        cases.push_back(q);
      }
  for (int pid : {13, -13, 11, -11})
    for (unsigned tag : {0u, 1u, 2u})
      for (unsigned generation : {0u, 1u})
        for (std::uint64_t step : {0ULL, 1ULL, 0xffffffffffffffffULL})
          for (double factor : {0., .5, 1., std::nextafter(1., 2.)}) {
            auto q = input(pid, 1000. * current::transport_mass_gev(pid) * factor);
            auto& p = q.candidate_.particle_;
            p.reserved_ = tag;
            p.generation_ = generation;
            p.stepId_ = step;
            REQUIRE(current::is_native_muon_pair_cut(p) ==
                    reference::isNativeMuonPairCut(continuous_test::convert(p)));
            for (double offset : {-1.e-14, 0., 1.e-14}) {
              auto const deposit =
                  p.energyGeV_ - current::transport_mass_gev(pid) + offset;
              REQUIRE(
                  current::is_native_muon_pair_cut_deposit(pid, tag, p.energyGeV_,
                                                           deposit) ==
                  reference::isNativeMuonPairCutDeposit(pid, tag, p.energyGeV_, deposit));
            }
            cases.push_back(q);
          }
  current::ProposalNativeTableSet empty;
  for (auto const& q : cases) compare_host(empty, q);
  compare_device({}, {}, cases);
  std::ofstream report("selection-contract-counts.txt");
  report << "early_and_invalid " << cases.size() << "\n";
}

TEST_CASE(
    "Vertex acceptance, continuation and deferred photon completion retain ordering",
    "[selection]") {
  using namespace selection_test;
  auto q = input(11, 1.e3, true);
  auto table = control(11, current::BremsProcessId);
  auto vertex = [&]() {
    compare_host(table, q);
    return current::select_lepton_vertex(
        physics(current::make_proposal_native_host_view(table), q), q.candidate_, q.seed_,
        q.shower_);
  };
  q.candidate_.totalRateCm2PerG_ = 4.;
  q.candidate_.processUniform_ = .5;
  auto rejected = vertex();
  REQUIRE(rejected.continuationFlag_ == 1);
  REQUIRE(rejected.record_.particle_.stepId_ == q.candidate_.particle_.stepId_ + 1);
  REQUIRE(rejected.record_.proposalSelectionUniform_ == 0.);
  q.candidate_.particle_.stepId_ = 0xffffffffffffffffULL;
  REQUIRE(vertex().fallback_.reason_ == current::FallbackReason::InvalidFinalState);
  q.candidate_.particle_.stepId_ = 17;
  q.candidate_.processUniform_ = std::nextafter(.5, 0.);
  REQUIRE(vertex().fallbackFlag_ ==
          1); // accepted, but the controlled fixture has no inverse CDF
  for (unsigned source : {0u, 1u, 2u, 0xffffffffu}) {
    q.source_ = source;
    auto failed = vertex();
    REQUIRE(failed.nativeInverseFailures_ == (source == 1u ? 1u : 0u));
    REQUIRE(failed.fallback_.interactionVertexReached_ == 1);
    REQUIRE(failed.fallback_.selectionUniform_ > 0.);
    REQUIRE(failed.fallback_.lossQuantile_ > 0.);
    REQUIRE(failed.fallback_.randomProcessId_ ==
            current::ProposalSelectionRandomProcessId);
  }
  table = control(11, current::ElectronPairProcessId);
  auto replay = vertex();
  REQUIRE(replay.fallback_.reason_ == current::FallbackReason::NativeSelectionReplay);
  REQUIRE(replay.nativeInverseFailures_ == 0);
  table = control(11, current::BremsProcessId, 2., 0.);
  REQUIRE(vertex().fallback_.reason_ == current::FallbackReason::ZeroTotalRate);
  table = control(11, current::BremsProcessId);
  table.selectionColumnIndices_[0] = 99;
  auto bad = vertex();
  REQUIRE(bad.fallbackFlag_ == 1);
  REQUIRE(bad.fallback_.diagnosticValue0_ == 1.e3);
  REQUIRE(bad.fallback_.diagnosticValue1_ == 4.);
  REQUIRE(bad.fallback_.diagnosticValue2_ > 0.);
  for (auto process : {current::PhotonuclearProcessId, current::PhotonPairProcessId}) {
    table = control(22, process);
    q = input(22, 1.e3);
    compare_host(table, q);
    auto const selected = current::select_discrete_interaction(
        physics(current::make_proposal_native_host_view(table), q),
        q.candidate_.particle_, q.candidate_.inputIndex_, q.seed_, q.shower_);
    REQUIRE(selected.fallbackFlag_ == 0);
    REQUIRE(selected.interaction_.deferredPhotonFallbackReason_ >= 0);
    REQUIRE(selected.interaction_.interactionVertexReached_ == 0);
    REQUIRE(selected.interaction_.interactionGrammageGPerCm2_ > 0.);
    REQUIRE(selected.interaction_.particle_.positionM_[0] ==
            q.candidate_.particle_.positionM_[0]);
  }
}
