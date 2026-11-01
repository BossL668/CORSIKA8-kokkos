/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::egs4::application {
  // Actual HybridCascade external-router API. The C8 stack owns the global
  // history-ID allocator; Session owns resident native transport/RNG state.
  // HostHandler must explicitly consume transport/decay/selected-nuclear requests. This
  // is not a PROPOSAL fallback and never silently re-routes an EM particle to C8.
  template <class TStack, class TModelProperties, class THostHandler, class TRegionOf,
            class TMuonRouter>
  class Router {
    Session& session_;
    EnvironmentTopology<TModelProperties> const& topology_;
    ::corsika::CoordinateSystemPtr coordinates_;
    THostHandler& host_;
    TRegionOf regionOf_;
    std::size_t batchSize_;
    std::uint64_t childrenPerParticle_;
    std::vector<::corsika::kokkos::ParticleState> staged_;
    RunStatistics statistics_;
    bool ended_{};
    TMuonRouter* muons_{};
    ProposalMuonBackend* muonBackend_{};

  public:
    Router(Session& session, Configuration const& configuration,
           OutputCallbacks const& output, ::corsika::CoordinateSystemPtr coordinates,
           EnvironmentTopology<TModelProperties> const& topology, THostHandler& host,
           TRegionOf region_of, std::size_t batch_size, bool host_reference = false,
           TMuonRouter* muons = nullptr, ProposalMuonBackend* muon_backend = nullptr)
        : session_(session)
        , topology_(topology)
        , coordinates_(std::move(coordinates))
        , host_(host)
        , regionOf_(std::move(region_of))
        , batchSize_(batch_size)
        , childrenPerParticle_(configuration.nativePromptResonanceDecay_
                                   ? 5
                                   : configuration.nativePromptRhoDecay_
                                         ? 4
                                         : configuration.nativePhotonuclearVertices_ ? 3
                                                                                     : 2)
        , muons_(muons)
        , muonBackend_(muon_backend) {
      if (bool(muons_) != bool(muonBackend_))
        throw std::invalid_argument("Muon router/backend must be supplied together");
      if (!coordinates_ || !batch_size ||
          batch_size > std::size_t(configuration.queueCapacity_ / 2))
        throw std::invalid_argument("Invalid native EGS4 router batch/coordinate system");
      if (configuration.resumeRetainedShower_)
        session_.resume(configuration.seed_, configuration.showerId_, output);
      else
        session_.begin(configuration, output, host_reference);
    }
    template <class TParticle>
    bool canRoute(TParticle const& p, std::uint64_t step) const {
      using ::corsika::Code;
      return p.getPID() == Code::Electron || p.getPID() == Code::Positron ||
             p.getPID() == Code::Photon || (muons_ && muons_->canRoute(p, step));
    }
    template <class TParticle>
    void stage(TParticle const& particle, ::corsika::transport::HistoryId id,
               ::corsika::transport::HistoryId parent,
               ::corsika::transport::Generation generation,
               ::corsika::transport::StepId step) {
      if (ended_) throw std::logic_error("Invalid native EGS4 stage");
      if (!::corsika::is_em(particle.getPID())) {
        if (!muons_) throw std::logic_error("Non-EM input without muon router");
        muons_->stage(particle, id, parent, generation, step);
        return;
      }
      auto p = ::corsika::kokkos::particle_transfer::to_device_state(
          particle, coordinates_, id, parent, generation, step, topology_);
      if (muonBackend_) muonBackend_->auditScalarHandoff(p);
      p.mediumId_ = regionOf_(particle.getNode());
      staged_.push_back(p);
    }
    bool pending() const {
      return !ended_ && (!staged_.empty() || session_.activeParticles() != 0 ||
                         (muons_ && muons_->pending()) ||
                         (muonBackend_ && muonBackend_->hasElectromagnetic()));
    }
    template <class TH, class TS>
    bool consumeForcedDecay(TH id, TS step) {
      return muons_ && muons_->consumeForcedDecay(id, step);
    }
    bool readyForScalarInterleave() const {
      // Accumulate scalar-produced muons with the EM front; a muon-only front
      // is flushed when the scalar stack drains, not after every individual root.
      return !ended_ && staged_.size() + session_.activeParticles() >= batchSize_;
    }
    std::size_t advanceOneWavefrontAndReturn(TStack& stack) {
      if (ended_) throw std::logic_error("Native EGS4 router already ended");
      std::size_t returned = 0;
      if (muons_ && muons_->pending())
        returned += muons_->advanceOneWavefrontAndReturn(stack);
      if (muonBackend_)
        for (auto const& p : muonBackend_->takeElectromagnetic()) staged_.push_back(p);
      if (!staged_.empty()) {
        session_.append(staged_);
        staged_.clear();
      }
      auto active = session_.activeParticles();
      if (!active) return returned;
      auto reserved = childrenPerParticle_ * active;
      auto first = stack.reserveTransportHistoryIds(reserved);
      auto wave = session_.advance(first, reserved);
      for (auto const& request : wave.hostRequests_)
        if (!host_.canHandle(request))
          throw std::runtime_error(
              "Native EGS4 host request has no explicit consumer; shower incomplete");
      for (auto const& request : wave.hostRequests_)
        returned += host_.handle(stack, request);
      return returned;
    }
    void endOfShower() {
      if (ended_ || pending()) throw std::logic_error("Native EGS4 router did not drain");
      if (muons_) muons_->endOfShower();
      statistics_ = session_.finishEMQueue();
      ended_ = true;
    }
    RunStatistics const& statistics() const {
      if (!ended_)
        throw std::logic_error("Native EGS4 statistics requested before router end");
      return statistics_;
    }
  };
} // namespace corsika::kokkos::egs4::application
