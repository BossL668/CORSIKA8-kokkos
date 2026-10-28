/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "BackendFixture.hpp"
#include "EnvironmentHeaderQueries.hpp"
#include <corsika/kokkos/em/algorithms/TransportMass.hpp>
#include <corsika/kokkos/em/proposal/ProposalFallbackAdapter.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
  using namespace backend_test;
  try {
    require(argc == 3, "backend and cache directory arguments");
    Fixture fixture(argv[2]);
    cp::RuntimeConfig runtime;
    runtime.executionBackend_ = argv[1];
    runtime.threads_ =
        (runtime.executionBackend_ == "cuda" || runtime.executionBackend_ == "hip" ||
         runtime.executionBackend_ == "sycl")
            ? 1
            : 2;
    cp::EmBackend backend(runtime);
    fixture.initialize(backend);
    // Native g++ and the real Kokkos execution space consume the same exported
    // arrays. CUDA builds compile only the numerical kernel TU with nvcc.
    EnvironmentResults deviceQueries[2];
    auto const environmentData = cp::bind_environment_data(*fixture.snapshot_);
    query_execution_environment(environmentData, deviceQueries);
    auto near = [](double a, double b) {
      return std::abs(a - b) <= 1.e-12 * std::max(1., std::max(std::abs(a), std::abs(b)));
    };
    for (unsigned i = 0; i != 2; ++i) {
      auto const host = query_header_environment(environmentData, i);
      auto const& device = deviceQueries[i];
      require(host.layer_.status_ == cp::EnvironmentStatus::Success &&
                  device.layer_.status_ == host.layer_.status_ &&
                  device.layer_.nodeIndex_ == host.layer_.nodeIndex_,
              "host/device node query");
      require(near(host.layer_.densityGPerCm3_, .001225) &&
                  near(device.layer_.densityGPerCm3_, host.layer_.densityGPerCm3_),
              "real air density");
      require(host.boundary_.status_ == cp::EnvironmentStatus::Success &&
                  device.boundary_.status_ == host.boundary_.status_ &&
                  device.boundary_.nextNode_ == host.boundary_.nextNode_ &&
                  near(device.boundary_.distanceM_, host.boundary_.distanceM_),
              "host/device boundary");
      require(host.grammage_.status_ == cp::EnvironmentStatus::Success &&
                  device.grammage_.status_ == host.grammage_.status_ &&
                  near(host.grammage_.value_, 12.25) &&
                  near(device.grammage_.value_, host.grammage_.value_),
              "host/device grammage");
      require(host.length_.status_ == cp::EnvironmentStatus::Success &&
                  device.length_.status_ == host.length_.status_ &&
                  near(host.length_.value_, 100.) &&
                  near(device.length_.value_, host.length_.value_),
              "host/device inverse grammage");
      require(host.observation_.status_ == cp::ObservationPlaneStatus::Success &&
                  device.observation_.status_ == host.observation_.status_ &&
                  near(host.observation_.distanceM_, i == 0 ? 100. : 162.5) &&
                  near(device.observation_.distanceM_, host.observation_.distanceM_),
              "host/device observation boundary");
    }
    auto const tableHash = backend.statistics().nativeTableHash_;
    std::vector<cp::ParticleState> particles;
    for (int pid : {22, 11, -11, 13, -13})
      for (unsigned i = 0; i != 4; ++i)
        particles.push_back(fixture.particle(pid, 100., 100 + particles.size()));
    auto selection = backend.selectInteractionsForValidation(particles);
    require(selection.inputParticles_ == particles.size(), "selection input count");
    require(selection.interactions_.size() + selection.fallbackEvents_.size() ==
                particles.size(),
            "selection partitions every input");
    require(!selection.interactions_.empty(), "native rates sampled");
    auto repeated = backend.selectInteractionsForValidation(particles);
    require(repeated.interactions_.size() == selection.interactions_.size(),
            "repeat selected count");
    for (std::size_t i = 0; i != selection.interactions_.size(); ++i) {
      auto const& a = selection.interactions_[i];
      auto const& b = repeated.interactions_[i];
      require(a.particle_.historyId_ == b.particle_.historyId_ &&
                  a.interactionGrammageGPerCm2_ == b.interactionGrammageGPerCm2_ &&
                  a.processId_ == b.processId_ && a.componentHash_ == b.componentHash_ &&
                  a.energyFraction_ == b.energyFraction_,
              "same input selection is repeatable");
    }
    std::vector<cp::InteractionRecord> photons, leptons;
    for (auto const& entry : selection.interactions_)
      (entry.particle_.pid_ == 22 ? photons : leptons).push_back(entry);
    auto photonSteps = backend.transportPhotonsForValidation(photons);
    auto leptonSteps = backend.transportLeptonsForValidation(leptons);
    require(photonSteps.inputInteractions_ == photons.size(), "photon transport called");
    require(leptonSteps.inputInteractions_ == leptons.size(), "lepton transport called");

    // Real native identities; selected e/mu direct-pair channels must retain
    // their CPU completion payload, rather than silently losing the reaction.
    std::vector<cp::InteractionRecord> pair;
    for (int pid : {11, -11, 13, -13}) {
      auto column = std::find_if(
          fixture.tables_.dndxColumns_.begin(), fixture.tables_.dndxColumns_.end(),
          [pid](auto const& c) {
            return c.pdgId_ == pid && c.processId_ == cp::ElectronPairProcessId;
          });
      require(column != fixture.tables_.dndxColumns_.end(),
              "native direct-pair column exists");
      cp::InteractionRecord entry;
      entry.particle_ = fixture.particle(pid, 100., 1000 + pair.size());
      entry.processId_ = cp::ElectronPairProcessId;
      entry.componentHash_ = column->componentHash_;
      entry.particleMassGeV_ = column->particleMassMeV_ / 1000.;
      entry.interactionVertexReached_ = 1;
      entry.massDensityGPerCm3_ = .001225;
      entry.energyFraction_ = .1;
      entry.lossQuantile_ = .3;
      entry.proposalSelectionUniform_ = .4;
      pair.push_back(entry);
    }
    auto finalStates = backend.generateLeptonFinalStatesForValidation(pair, 10000);
    require(finalStates.fallbackEvents_.size() == pair.size(),
            "all four direct-pair projectiles return CPU requests");
    for (auto const& event : finalStates.fallbackEvents_) {
      require(event.reason_ == cp::FallbackReason::CpuOnlyProcess,
              "CPU-only pair reason");
      require(event.mediumHash_ && event.interactionHash_,
              "backend enriches native provenance");
      require(cp::has_specified_proposal_final_state(event),
              "CPU can complete specified native final state");
    }
    auto next = cp::make_shower_config(fixture.config_);
    next.showerId_ = 24;
    backend.beginShower(next);
    require(backend.statistics().nativeTableHash_ == tableHash,
            "shower reuse preserves table provenance");
    require(backend.pendingPhotonCount() == 0 && backend.pendingLeptonCount() == 0,
            "new shower has empty resident queues");
    std::cout << "real backend stages: inputs=" << particles.size()
              << " selected=" << selection.interactions_.size()
              << " pair_cpu=" << finalStates.fallbackEvents_.size() << '\n';
    return 0;
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
