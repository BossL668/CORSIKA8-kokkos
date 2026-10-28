/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "TransportChecks.hpp"

void check_transport_defaults() {
  using namespace transport_test;
  check(flatten(current::PhotonTransportRecord{}),
        flatten(reference::PhotonTransportRecord{}));
  check(flatten(current::LeptonTransportRecord{}),
        flatten(reference::LeptonTransportRecord{}));
  Kokkos::View<Outcome * [2], Kokkos::LayoutRight, memory_space_type> output("defaults",
                                                                             2);
  Kokkos::parallel_for(
      "transport_defaults", Kokkos::RangePolicy<execution_space_type>(0, 1),
      KOKKOS_LAMBDA(std::size_t) {
        output(0, 0) = flatten(current::PhotonTransportRecord{});
        output(0, 1) = flatten(reference::PhotonTransportRecord{});
        output(1, 0) = flatten(current::LeptonTransportRecord{});
        output(1, 1) = flatten(reference::LeptonTransportRecord{});
      });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  check(host(0, 0), host(0, 1));
  check(host(1, 0), host(1, 1));
}

void check_populated_transport_records() {
  using namespace transport_test;
  std::vector<current::PhotonTransportRecord> photons;
  std::vector<current::LeptonTransportRecord> leptons;
  for (std::size_t index = 0; index < 128; ++index) {
    photons.push_back(make_photon_payload(index));
    leptons.push_back(make_lepton_payload(index));
  }
  check_roundtrip(photons);
  check_roundtrip(leptons);
  std::ofstream report("transport-roundtrip-counts.txt");
  report << "photon_records " << photons.size() << "\n";
  report << "lepton_records " << leptons.size() << "\n";
}

void check_native_muon_pair_cut_records() {
  using namespace transport_test;
  std::vector<current::InteractionRecord> cases;
  for (int pid : {13, -13})
    for (double fraction : {1.e-12, .01, .25, .5, std::nextafter(1., 0.), 1.})
      for (unsigned generation : {1u, 3u, 0xffffffffu})
        for (int node : {-3, -1, 0, 7}) {
          current::InteractionRecord input;
          auto& p = input.particle_;
          p.pid_ = pid;
          p.energyGeV_ = current::transport_mass_gev(pid) * fraction;
          p.generation_ = generation;
          p.reserved_ = current::NativeMuonPairCutFlag;
          p.environmentNode_ = node;
          p.mediumId_ = 42;
          p.historyId_ = 0xfedcba9876543210ULL + cases.size();
          p.parentHistoryId_ = 0x9876543210abcdefULL;
          p.timeS_ = 2.e-6;
          p.weight_ = 7.;
          p.positionM_[0] = -123.;
          p.positionM_[1] = 45.;
          p.positionM_[2] = 6.e6;
          p.direction_[2] = -1.;
          input.inputIndex_ = 0x8000000000000000ULL + cases.size();
          input.status_ = current::InteractionStatus::ParticleCut;
          input.interactionVertexReached_ = 1;
          input.processId_ = 987;
          input.componentHash_ = 0x123456789abcdef0ULL;
          input.totalRateCm2PerG_ = 2.;
          input.vertexTotalRateCm2PerG_ = 3.;
          input.interactionGrammageGPerCm2_ = HUGE_VAL;
          input.massDensityGPerCm3_ = .0012;
          input.particleMassGeV_ = current::transport_mass_gev(pid);
          input.decayUniform_ = .2;
          input.decayDrawId_ = 19;
          input.energyFraction_ = .4;
          input.distanceUniform_ = .1;
          input.processUniform_ = .3;
          input.lossQuantile_ = .6;
          input.distanceDrawId_ = 31;
          input.processDrawId_ = 37;
          input.lossDrawId_ = 41;
          input.proposalSelectionUniform_ = .7;
          input.processRandomProcessId_ = 43;
          input.proposalSelectionRandomProcessId_ = 47;
          input.proposalSelectionDrawId_ = 53;
          REQUIRE(current::is_native_muon_pair_cut(p));
          auto const before = continuous_test::convert(input);
          auto const result = current::native_muon_pair_cut_record(input);
          check(flatten(result), flatten(reference::nativeMuonPairCutRecord(before)));
          REQUIRE(result.distanceM_ == 0.);
          REQUIRE(result.traversedGrammageGPerCm2_ == 0.);
          REQUIRE(result.cutDepositedEnergyGeV_ ==
                  p.energyGeV_ - current::transport_mass_gev(pid));
          REQUIRE(result.cutDepositedEnergyGeV_ <= 0.);
          REQUIRE(result.start_.stepId_ == 0);
          REQUIRE(result.end_.stepId_ == 1);
          REQUIRE(result.interaction_.interactionVertexReached_ == 0);
          REQUIRE(result.multipleScatteringApplied_ == 0);
          REQUIRE(result.magneticBendingApplied_ == 0);
          REQUIRE(result.startLayerIndex_ == -1);
          REQUIRE(result.endLayerIndex_ == -1);
          cases.push_back(input);
        }
  auto inputs = upload(cases);
  Kokkos::View<current::LeptonTransportRecord*, memory_space_type> output("cut_records",
                                                                          cases.size());
  Kokkos::View<Outcome*, memory_space_type> expected("cut_reference", cases.size());
  Kokkos::parallel_for(
      "muon_cut", Kokkos::RangePolicy<execution_space_type>(0, cases.size()),
      KOKKOS_LAMBDA(std::size_t index) {
        output(index) = current::native_muon_pair_cut_record(inputs(index));
        expected(index) = flatten(
            reference::nativeMuonPairCutRecord(continuous_test::convert(inputs(index))));
      });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  auto old = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected);
  for (std::size_t index = 0; index < cases.size(); ++index) {
    INFO("Native muon cut case " << index);
    check(flatten(host(index)), old(index));
    check(flatten(host(index)),
          flatten(current::native_muon_pair_cut_record(cases[index])));
  }
  std::ofstream report("transport-record-counts.txt");
  report << "zero_length_cut " << cases.size() << "\n";
}

TEST_CASE("Transport records and native cut retain the production contract",
          "[transport]") {
  // All checks share one lifetime: Kokkos cannot reinitialize after finalization.
  corsika::kokkos::Runtime runtime;
  check_transport_defaults();
  check_populated_transport_records();
  check_native_muon_pair_cut_records();
}
