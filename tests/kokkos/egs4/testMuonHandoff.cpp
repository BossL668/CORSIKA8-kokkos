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
  void check_particle(ParticleState const& actual, ParticleState const& expected) {
    // ParticleState has tail padding. Its contract preserves named values,
    // not the unspecified bytes between the last member and its alignment.
    auto sameBits = [](double const& first, double const& second) {
      return std::memcmp(&first, &second, sizeof(double)) == 0;
    };
    CHECK(actual.pid_ == expected.pid_);
    CHECK(actual.mediumId_ == expected.mediumId_);
    CHECK(actual.generation_ == expected.generation_);
    CHECK(actual.reserved_ == expected.reserved_);
    CHECK(sameBits(actual.energyGeV_, expected.energyGeV_));
    CHECK(sameBits(actual.timeS_, expected.timeS_));
    CHECK(sameBits(actual.weight_, expected.weight_));
    CHECK(actual.historyId_ == expected.historyId_);
    CHECK(actual.parentHistoryId_ == expected.parentHistoryId_);
    CHECK(actual.stepId_ == expected.stepId_);
    CHECK(actual.environmentNode_ == expected.environmentNode_);
    for (unsigned component = 0; component < 3; ++component) {
      CAPTURE(component);
      CHECK(sameBits(actual.positionM_[component], expected.positionM_[component]));
      CHECK(sameBits(actual.direction_[component], expected.direction_[component]));
    }
  }
  template <class TFunction>
  void rejects(TFunction f) {
    bool failed = false;
    try {
      f();
    } catch (std::exception const&) { failed = true; }
    REQUIRE((failed));
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
  REQUIRE((adapter.canTransport(mu) && !adapter.canTransport(electron) &&
           !adapter.canTransport(photon)));
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
  REQUIRE((backend.wavesSeen_ == 1 && backend.firstSeen_ == 100 &&
           backend.limitSeen_ == 10000));
  REQUIRE((output.remainingLeptons_.size() == 1 && output.generatedPhotons_.empty() &&
           output.cpuSpillParticles_.empty()));
  REQUIRE((output.decayCandidates_.size() == 1 && output.fallbackEvents_.size() == 1 &&
           output.secondaryHistoryIdsUsed_ == 3));
  REQUIRE((!output.completed_ && adapter.muonSteps() == 1 && adapter.emTransfers() == 3));
  auto em = adapter.takeElectromagnetic();
  REQUIRE((em.size() == 3 && !adapter.hasElectromagnetic()));
  check_particle(output.remainingLeptons_.front(), mu);
  check_particle(em[0], photon);
  check_particle(em[1], electron);
  check_particle(em[2], positron);
  rejects([&] { adapter.runLeptonWavefront({electron}, 100, 2, 10000, 1); });
  rejects([&] { adapter.runPhotonWavefront({photon}, 100, 2, 1); });
  backend.pending_ = 1;
  rejects([&] { adapter.runLeptonWavefront({mu}, 100, 2, 10000, 1); });
  backend.pending_ = 0;
  backend.result_.remainingLeptons_.back().stepId_ = 1;
  rejects([&] { adapter.runLeptonWavefront({mu}, 100, 2, 10000, 1); });
}
