/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "SessionFixture.hpp"
#include <corsika/kokkos/environment/EnvironmentSnapshotKey.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
  using namespace session_test;

  // Compare actual named results, not struct padding or elapsed-time counters.
  std::string decisions(cp::InteractionBatchResult const& batch) {
    require(batch.inputParticles_ == 5, "five actual projectile inputs");
    require(batch.interactions_.size() + batch.fallbackEvents_.size() == 5,
            "every input has a selected result or CPU fallback");
    require(!batch.interactions_.empty(), "real native table selection executed");
    std::ostringstream out;
    out << std::hexfloat;
    for (auto const& r : batch.interactions_) {
      out << r.particle_.historyId_ << ' ' << r.inputIndex_ << ' ' << r.processId_ << ' '
          << static_cast<unsigned>(r.status_) << ' ' << r.componentHash_ << ' '
          << r.totalRateCm2PerG_ << ' ' << r.interactionGrammageGPerCm2_ << ' '
          << r.energyFraction_ << ' ' << r.distanceUniform_ << ' ' << r.processUniform_
          << ' ' << r.lossQuantile_ << ' ' << r.distanceDrawId_ << ' ' << r.processDrawId_
          << ' ' << r.lossDrawId_ << '\n';
    }
    for (auto const& f : batch.fallbackEvents_)
      out << "fallback " << f.particle_.historyId_ << ' '
          << static_cast<unsigned>(f.reason_) << ' ' << f.mediumHash_ << ' '
          << f.interactionHash_ << '\n';
    return out.str();
  }

  void validate_cold(cp::EmRunSession& session, Fixture const& fixture) {
    require(!session.isInitialized(), "new session has no backend");
    require(!session.getLoadedNativeTable(), "new session has no export");
    rejects<std::logic_error>([&] { session.getBackend(); });
    auto const begin = fixture.begin(session);
    require(!begin.reused_ && begin.backend_ == &session.getBackend(),
            "cold backend identity");
    require(begin.muonTransportAvailable_, "real muon auxiliary data available");
    auto const& stats = begin.backend_->statistics();
    require(stats.showerOrdinal_ == 1 && !stats.reusedForShower_, "cold ordinal");
    require(session.getLoadedNativeTable() && stats.nativeTableNodes_ > 0,
            "real PROPOSAL table uploaded");
    require(stats.nativeTableHash_ == session.getLoadedNativeTable()->contentHash_,
            "actual native export fingerprint");
    require(stats.oneTimeInitializationMs_ >= 0., "initialization timing exists");
    std::cout << "native table nodes=" << stats.nativeTableNodes_
              << " backend=" << stats.acceleratorBackend_
              << " gpu=" << stats.acceleratorGpu_ << '\n';
  }

  void cold_hot(cp::EmRunSession& session, Fixture const& fixture,
                std::filesystem::path const& directory) {
    validate_cold(session, fixture);
    auto* const backend = &session.getBackend();
    auto const* table = session.getLoadedNativeTable();
    auto const tableHash = table->contentHash_;
    auto const staticBytes = backend->statistics().staticHostToDeviceBytes_;
    auto const initialization = backend->statistics().oneTimeInitializationMs_;
    auto const expected =
        decisions(backend->selectInteractionsForValidation(fixture.particles()));
    std::ofstream(directory / "cold-decisions.txt") << expected;
    EnvironmentCopy relocated(fixture.view());
    require(relocated.view().nodes_ != fixture.view().nodes_,
            "different array allocation");
    require(cp::make_environment_snapshot_key(relocated.view()) ==
                cp::make_environment_snapshot_key(fixture.view()),
            "same exported numeric environment");
    for (unsigned ordinal = 2; ordinal <= 4; ++ordinal) {
      // Reusing a production session does not export calculators or revalidate
      // requirements. An empty borrowed view list demonstrates that contract.
      auto ignoredRequirements = fixture.requirements_;
      ignoredRequirements.maximumPrimaryEnergyMeV_ = 1.e99;
      auto const begin =
          session.beginProposalNative(relocated.view(), fixture.geometry_,
                                      fixture.config_, ignoredRequirements, {}, {});
      require(begin.reused_ && begin.backend_ == backend, "hot backend identity");
      require(session.getLoadedNativeTable() == table && table->contentHash_ == tableHash,
              "hot table ownership and bytes retained");
      auto const& stats = backend->statistics();
      require(stats.showerOrdinal_ == ordinal && stats.reusedForShower_, "hot ordinal");
      require(stats.staticHostToDeviceBytes_ == staticBytes &&
                  stats.oneTimeInitializationMs_ == initialization,
              "no repeated static upload");
      require(stats.interactionSelectionBatches_ == 0, "per-shower counter reset");
      auto const actual =
          decisions(backend->selectInteractionsForValidation(fixture.particles()));
      require(actual == expected, "same-input cold and warm selection results");
      std::ofstream(directory / ("warm-" + std::to_string(ordinal) + "-decisions.txt"))
          << actual;
    }
    cp::EmRunSession moved(std::move(session));
    require(&moved.getBackend() == backend && moved.getLoadedNativeTable() == table,
            "move transfers the existing initialized session");
  }

  void changed_environment(cp::EmRunSession& session, Fixture const& fixture) {
    validate_cold(session, fixture);
    auto const before = session.getBackend().statistics().showerOrdinal_;
    auto reject = [&](cp::EnvironmentDataView const& view,
                      cp::TransportGeometryConfig const& geometry) {
      auto const error = rejects<std::invalid_argument>([&] {
        session.beginProposalNative(view, geometry, fixture.config_,
                                    fixture.requirements_, {}, {});
      });
      require(error.find("Environment changed") != std::string::npos,
              "production reuse rejection");
      require(session.getBackend().statistics().showerOrdinal_ == before,
              "rejection precedes beginShower");
    };
    auto copy = EnvironmentCopy(fixture.view());
    require(!copy.models_.empty(), "exported model exists");
    copy.models_[0].referenceDensityGPerCm3_ *= 1.01;
    reject(copy.view(), fixture.geometry_);
    copy = EnvironmentCopy(fixture.view());
    copy.models_[0].magneticFieldT_[0] += 1.e-6;
    reject(copy.view(), fixture.geometry_);
    copy = EnvironmentCopy(fixture.view());
    copy.shapes_.back().radiusM_ += 1.;
    reject(copy.view(), fixture.geometry_);
    auto geometry = fixture.geometry_;
    geometry.observation_.pointM_[2] += 1.;
    reject(fixture.view(), geometry);
    geometry = fixture.geometry_;
    geometry.observation_.normal_[0] += .1;
    reject(fixture.view(), geometry);
    geometry = fixture.geometry_;
    geometry.query_.maximumMagneticDeflectionRad_ += .01;
    reject(fixture.view(), geometry);
    require(fixture.begin(session).reused_, "unchanged environment remains reusable");
  }

  void early_failure(cp::EmRunSession& session, Fixture const& fixture) {
    auto config = fixture.config_;
    config.physicsSource_ = static_cast<cp::PhysicsSource>(0);
    rejects<std::invalid_argument>([&] {
      session.beginProposalNative(fixture.view(), fixture.geometry_, config,
                                  fixture.requirements_, fixture.interactions_,
                                  fixture.continuous_);
    });
    require(!session.isInitialized() && !session.getLoadedNativeTable(),
            "physics rejection before export");
    auto requirements = fixture.requirements_;
    requirements.maximumPrimaryEnergyMeV_ = 1.e99;
    rejects<std::runtime_error>([&] {
      session.beginProposalNative(fixture.view(), fixture.geometry_, fixture.config_,
                                  requirements, fixture.interactions_,
                                  fixture.continuous_);
    });
    require(!session.isInitialized() && session.getLoadedNativeTable(),
            "domain failure leaves only the exported table, as production");
    auto const result = fixture.begin(session);
    require(!result.reused_ && session.isInitialized(),
            "valid retry constructs actual backend");
  }

  void late_failure(cp::EmRunSession& session, Fixture const& fixture) {
    auto config = fixture.config_;
    config.memoryFraction_ = 0.;
    auto const error = rejects<std::invalid_argument>([&] {
      session.beginProposalNative(fixture.view(), fixture.geometry_, config,
                                  fixture.requirements_, fixture.interactions_,
                                  fixture.continuous_);
    });
    require(error.find("memory fraction") != std::string::npos,
            "actual backend initialize failure");
    // This is the original failure-state contract, not a newly proposed recovery.
    require(session.isInitialized() && session.getLoadedNativeTable(),
            "constructed backend remains owned after initialize failure");
    auto const retry = rejects<std::invalid_argument>([&] { fixture.begin(session); });
    require(retry.find("Environment changed") != std::string::npos,
            "uncommitted original environment key prevents in-place retry");
  }
} // namespace

int main(int argc, char** argv) {
  try {
    using namespace session_test;
    require(argc == 4, "scenario, backend and dedicated cache required");
    std::string const scenario = argv[1];
    std::filesystem::path const directory = argv[3];
    std::filesystem::create_directories(directory);
    Fixture fixture(directory);
    cp::RuntimeConfig runtime;
    runtime.executionBackend_ = argv[2];
    runtime.threads_ = 1;
    cp::EmRunSession session(runtime, directory / "auxiliary");
    if (scenario == "cold-hot")
      cold_hot(session, fixture, directory);
    else if (scenario == "environment")
      changed_environment(session, fixture);
    else if (scenario == "early-failure")
      early_failure(session, fixture);
    else if (scenario == "late-failure")
      late_failure(session, fixture);
    else
      throw std::invalid_argument("Unknown session test scenario");
    std::cout << "passed real-session " << scenario << " checks=" << checks << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
