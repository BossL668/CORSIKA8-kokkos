/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4MuonAudit.inl>
#include <corsika/kokkos/em/execution/IEmBackend.hpp>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace corsika::kokkos::egs4::application {
  // Host-boundary adapter only: all muon physics remains in the existing C8
  // backend. A single wave is essential: its freshly produced e+/e-/gamma must
  // reach EGS4 BEFORE another PROPOSAL transport step. Cross-species retention
  // must be disabled in the underlying backend's initialization config.
  class ProposalMuonBackend final : public ::corsika::kokkos::IEmBackend {
    using base_type = ::corsika::kokkos::IEmBackend;
    using particle_type = ::corsika::kokkos::ParticleState;
    base_type& backend_;
    MuonAudit audit_;
    std::vector<particle_type> electromagnetic_;
    std::uint64_t waves_{}, steps_{}, transferred_{};
    static bool muon(particle_type const& p) { return p.pid_ == 13 || p.pid_ == -13; }
    static bool em(particle_type const& p) {
      return p.pid_ == 11 || p.pid_ == -11 || p.pid_ == 22;
    }
    void separate(std::vector<particle_type>& particles) {
      std::vector<particle_type> retained;
      retained.reserve(particles.size());
      for (auto const& p : particles) {
        if (muon(p))
          retained.push_back(p);
        else if (em(p)) {
          if (p.stepId_ != 0)
            throw std::logic_error(
                "PROPOSAL transported an EGS4 daughter before handoff");
          audit_.particle("direct_em_handoff", p);
          electromagnetic_.push_back(p);
          ++transferred_;
        } else
          throw std::logic_error("Unexpected species in PROPOSAL muon output");
      }
      particles = std::move(retained);
    }
    void requireEmptyDeviceQueues() const {
      if (backend_.pendingPhotonCount() || backend_.pendingLeptonCount())
        throw std::logic_error("Muon-only adapter requires resident_cross_species=false");
    }

  public:
    explicit ProposalMuonBackend(base_type& backend)
        : backend_(backend) {
      if (!backend.capabilities().muonTransport_)
        throw std::invalid_argument("PROPOSAL muon transport unavailable");
      if (backend.gpuRadioEnabled())
        throw std::invalid_argument("Muon adapter must not own EM radio output");
      if (backend.gpuProfileEnabled())
        throw std::invalid_argument(
            "Muon adapter currently streams full steps to the C8 writers");
      requireEmptyDeviceQueues();
    }
    std::vector<particle_type> takeElectromagnetic() {
      return std::exchange(electromagnetic_, {});
    }
    bool hasElectromagnetic() const { return !electromagnetic_.empty(); }
    std::uint64_t muonWaves() const { return waves_; }
    std::uint64_t muonSteps() const { return steps_; }
    std::uint64_t emTransfers() const { return transferred_; }
    void auditScalarHandoff(particle_type const& p) {
      audit_.particle("scalar_em_handoff", p);
    }
    void beginShower(::corsika::kokkos::ShowerConfig const& c) override {
      if (hasElectromagnetic()) throw std::logic_error("Undrained muon-to-EGS4 handoff");
      backend_.beginShower(c);
      waves_ = steps_ = transferred_ = 0;
      requireEmptyDeviceQueues();
    }
    bool canTransport(particle_type const& p) const override {
      return muon(p) && backend_.canTransport(p);
    }
    bool hasProposalTable() const override { return backend_.hasProposalTable(); }
    // The EM router's small-batch CPU expansion cannot grow a muon population.
    // Disable that scheduling heuristic here, not any physical fallback.
    std::size_t minimumBatchSize() const override { return 1; }
    std::size_t maximumResidentPhotonBatchSize() const override {
      return backend_.maximumResidentPhotonBatchSize();
    }
    std::size_t maximumResidentLeptonBatchSize() const override {
      return backend_.maximumResidentLeptonBatchSize();
    }
    std::size_t maximumResidentInputBatchSize() const override {
      return backend_.maximumResidentInputBatchSize();
    }
    std::size_t pendingPhotonCount() const noexcept override {
      return backend_.pendingPhotonCount();
    }
    std::size_t pendingLeptonCount() const noexcept override {
      return backend_.pendingLeptonCount();
    }
    ::corsika::kokkos::PhotonCascadeResult runPhotonWavefront(
        std::vector<particle_type> const&, std::uint64_t, std::size_t,
        std::size_t) override {
      throw std::logic_error("Photons belong to EGS4, not the PROPOSAL muon backend");
    }
    ::corsika::kokkos::LeptonCascadeResult runLeptonWavefront(
        std::vector<particle_type> const& input, std::uint64_t first, std::size_t waves,
        std::uint64_t limit, std::size_t minimum) override {
      requireEmptyDeviceQueues();
      if (!waves) throw std::invalid_argument("Muon wave count must be positive");
      for (auto const& p : input)
        if (!muon(p))
          throw std::invalid_argument("Only muons may enter PROPOSAL transport");
      for (auto const& p : input) audit_.particle("muon_input", p);
      auto result = backend_.runLeptonWavefront(input, first, 1, limit, minimum);
      audit_.wave(result);
      requireEmptyDeviceQueues();
      if (result.wavefronts_ > 1)
        throw std::logic_error("Muon backend exceeded the single-wave handoff boundary");
      for (auto const& r : result.stepRecords_)
        if (!muon(r.start_))
          throw std::logic_error("Non-muon PROPOSAL step in EGS4 transport");
      separate(result.generatedPhotons_);
      separate(result.remainingLeptons_);
      separate(result.cpuSpillParticles_);
      result.completed_ = result.remainingLeptons_.empty();
      waves_ += result.wavefronts_;
      steps_ += result.transportRecords_;
      return result;
    }
    ::corsika::kokkos::BackendCapabilities capabilities() const override {
      auto c = backend_.capabilities();
      c.photonTransport_ = c.electronTransport_ = c.positronTransport_ = false;
      return c;
    }
    ::corsika::kokkos::BackendStatistics const& statistics() const override {
      return backend_.statistics();
    }
    bool gpuProfileEnabled() const noexcept override {
      return backend_.gpuProfileEnabled();
    }
    ::corsika::kokkos::ProfileResult downloadProfile() override {
      return backend_.downloadProfile();
    }
    bool gpuRadioEnabled() const noexcept override { return false; }
    ::corsika::kokkos::radio::RadioWaveforms downloadRadioWaveforms() override {
      throw std::logic_error("EGS4 owns radio finalization");
    }
    std::optional<::corsika::kokkos::FirstInteractionRecord>
    downloadFirstInteractionSnapshot() override {
      return backend_.downloadFirstInteractionSnapshot();
    }
  };
  // Optional hook lets the frozen CPU-muon configuration use the same native
  // router without constructing any additional PROPOSAL device resources.
  struct NoAcceleratedMuons {
    template <class TP, class TS>
    bool canRoute(TP const&, TS) {
      return false;
    }
    template <class... TArgs>
    void stage(TArgs&&...) {
      throw std::logic_error("Muon router disabled");
    }
    bool pending() const { return false; }
    bool readyForScalarInterleave() const { return false; }
    template <class TS>
    std::size_t advanceOneWavefrontAndReturn(TS&) {
      return 0;
    }
    template <class TH, class TS>
    bool consumeForcedDecay(TH, TS) {
      return false;
    }
    void endOfShower() {}
  };
  template <class TSink>
  struct MuonOutputSink {
    TSink& sink_;
    explicit MuonOutputSink(TSink& output)
        : sink_(output) {}
    template <class T>
    void onStep(T const& r) {
      sink_.onStep(r);
    }
    template <class T>
    void onObservation(T const& r) {
      sink_.onObservation(r);
    }
    template <class T>
    void onRadioTrack(T const& r) {
      sink_.onRadioTrack(r);
    }
    // This adapter has no first-interaction/production diagnostic writer.
    template <class T>
    void onFirstInteraction(T const&) {}
    template <class T>
    void onProjectedStep(T const&) {
      throw std::logic_error("Muon projected output was not enabled");
    }
    template <class T>
    void onGpuProfile(T const&) {
      throw std::logic_error("Muon resident profile was not enabled");
    }
    template <class T>
    void onGpuRadioWaveforms(T const&, std::uint64_t) {
      throw std::logic_error("EGS4 owns radio finalization");
    }
  };
} // namespace corsika::kokkos::egs4::application
