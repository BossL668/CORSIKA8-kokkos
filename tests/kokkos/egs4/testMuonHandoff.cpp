/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/detail/kokkos/egs4/Egs4ProposalMuonBackend.inl>
#include <cstring>
#include <catch2/catch_all.hpp>
namespace {
  using namespace corsika::kokkos;
  using namespace corsika::kokkos::egs4::application;
  void require(bool ok) {
    if (!ok) throw std::runtime_error("muon handoff contract failed");
  }
  template <class TFunction>
  void rejects(TFunction f) {
    bool failed = false;
    try {
      f();
    } catch (std::exception const&) { failed = true; }
    require(failed);
  }
  struct Fixture : IEmBackend {
    LeptonCascadeResult result_;
    BackendStatistics statistics_;
    std::size_t pending_{};
    std::uint64_t firstSeen_{}, limitSeen_{};
    std::size_t wavesSeen_{};
    void beginShower(ShowerConfig const&) override {}
    bool canTransport(ParticleState const&) const override { return true; }
    bool hasProposalTable() const override { return true; }
    std::size_t minimumBatchSize() const override { return 1; }
    std::size_t maximumResidentPhotonBatchSize() const override { return 32; }
    std::size_t maximumResidentLeptonBatchSize() const override { return 32; }
    std::size_t maximumResidentInputBatchSize() const override { return 32; }
    std::size_t pendingPhotonCount() const noexcept override { return pending_; }
    std::size_t pendingLeptonCount() const noexcept override { return 0; }
    PhotonCascadeResult runPhotonWavefront(std::vector<ParticleState> const&,
                                           std::uint64_t, std::size_t,
                                           std::size_t) override {
      throw std::logic_error("must not call photon transport");
    }
    LeptonCascadeResult runLeptonWavefront(std::vector<ParticleState> const&,
                                           std::uint64_t first, std::size_t waves,
                                           std::uint64_t limit, std::size_t) override {
      firstSeen_ = first;
      wavesSeen_ = waves;
      limitSeen_ = limit;
      return result_;
    }
    BackendCapabilities capabilities() const override {
      BackendCapabilities c;
      c.muonTransport_ = true;
      return c;
    }
    BackendStatistics const& statistics() const override { return statistics_; }
    bool gpuProfileEnabled() const noexcept override { return false; }
    ProfileResult downloadProfile() override { return {}; }
    bool gpuRadioEnabled() const noexcept override { return false; }
    corsika::kokkos::radio::RadioWaveforms downloadRadioWaveforms() override {
      return {};
    }
    std::optional<FirstInteractionRecord> downloadFirstInteractionSnapshot() override {
      return {};
    }
  };
} // namespace

TEST_CASE("EGS4 receives untouched muon daughters after one wave", "[egs4]") {
  Fixture backend;
  ProposalMuonBackend adapter(backend);
  ParticleState mu{}, electron{}, positron{}, photon{};
  mu.pid_ = 13;
  mu.historyId_ = 7;
  mu.stepId_ = 42;
  mu.energyGeV_ = 10.;
  mu.weight_ = 2.;
  electron = mu;
  electron.pid_ = 11;
  electron.historyId_ = 100;
  electron.parentHistoryId_ = 7;
  electron.stepId_ = 0;
  electron.energyGeV_ = .01;
  electron.direction_[2] = -1.;
  electron.mediumId_ = 3;
  positron = electron;
  positron.pid_ = -11;
  positron.historyId_ = 101;
  photon = electron;
  photon.pid_ = 22;
  photon.historyId_ = 102;
  require(adapter.canTransport(mu) && !adapter.canTransport(electron) &&
          !adapter.canTransport(photon));
  backend.result_.inputParticles_ = 1;
  backend.result_.wavefronts_ = 1;
  backend.result_.transportRecords_ = 1;
  backend.result_.remainingLeptons_ = {mu, electron};
  backend.result_.cpuSpillParticles_ = {positron};
  backend.result_.generatedPhotons_ = {photon};
  backend.result_.secondaryHistoryIdsUsed_ = 3;
  backend.result_.decayCandidates_ = {mu};
  backend.result_.fallbackEvents_.resize(1);
  auto output = adapter.runLeptonWavefront({mu}, 100, 1024, 10000, 1);
  require(backend.wavesSeen_ == 1 && backend.firstSeen_ == 100 &&
          backend.limitSeen_ == 10000);
  require(output.remainingLeptons_.size() == 1 && output.generatedPhotons_.empty() &&
          output.cpuSpillParticles_.empty());
  require(output.decayCandidates_.size() == 1 && output.fallbackEvents_.size() == 1 &&
          output.secondaryHistoryIdsUsed_ == 3);
  require(!output.completed_ && adapter.muonSteps() == 1 && adapter.emTransfers() == 3);
  auto em = adapter.takeElectromagnetic();
  require(em.size() == 3 && !adapter.hasElectromagnetic());
  require(std::memcmp(&em[0], &photon, sizeof photon) == 0 &&
          std::memcmp(&em[1], &electron, sizeof electron) == 0 &&
          std::memcmp(&em[2], &positron, sizeof positron) == 0);
  rejects([&] { adapter.runLeptonWavefront({electron}, 100, 2, 10000, 1); });
  rejects([&] { adapter.runPhotonWavefront({photon}, 100, 2, 1); });
  backend.pending_ = 1;
  rejects([&] { adapter.runLeptonWavefront({mu}, 100, 2, 10000, 1); });
  backend.pending_ = 0;
  backend.result_.remainingLeptons_.back().stepId_ = 1;
  rejects([&] { adapter.runLeptonWavefront({mu}, 100, 2, 10000, 1); });
}
