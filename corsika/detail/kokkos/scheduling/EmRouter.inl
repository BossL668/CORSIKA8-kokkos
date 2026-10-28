/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  template <typename TStack, typename TModelProperties, typename TProposalFallbackHandler,
            typename TOutputSink, typename TBackend>
  class EmRouter {
  public:
    using particle_type = typename TStack::particle_type;

    EmRouter(TBackend& backend, CoordinateSystemPtr coordinateSystem,
             EnvironmentBorrowedView const& environment,
             TransportGeometryConfig const& geometry,
             EnvironmentTopology<TModelProperties> const& topology)
        : backend_(backend)
        , coordinateSystem_(std::move(coordinateSystem))
        , environment_(environment)
        , geometry_(geometry)
        , topology_(topology) {
      if (!coordinateSystem_) {
        throw std::invalid_argument(
            "physical CUDA EM router requires a coordinate system");
      }
      if (!backend_.hasProposalTable()) {
        throw std::invalid_argument(
            "physical CUDA EM router requires an uploaded PROPOSAL table");
      }
      if (!environment_query_detail::valid_environment(environment_, geometry_.query_)) {
        throw std::invalid_argument(
            "physical CUDA EM router requires a valid atmosphere snapshot");
      }
    }

    EmRouter(TBackend& backend, CoordinateSystemPtr coordinateSystem,
             EnvironmentBorrowedView const& environment,
             TransportGeometryConfig const& geometry,
             EnvironmentTopology<TModelProperties> const& topology,
             TProposalFallbackHandler& fallbackHandler)
        : EmRouter(backend, std::move(coordinateSystem), environment, geometry,
                   topology) {
      static_assert(!std::is_same_v<TProposalFallbackHandler, ScalarProposalFallback>,
                    "the scalar fallback marker cannot be supplied as a handler");
      fallbackHandler_ = &fallbackHandler;
    }

    EmRouter(TBackend& backend, CoordinateSystemPtr coordinateSystem,
             EnvironmentBorrowedView const& environment,
             TransportGeometryConfig const& geometry,
             EnvironmentTopology<TModelProperties> const& topology,
             TProposalFallbackHandler& fallbackHandler, TOutputSink& outputSink)
        : EmRouter(backend, std::move(coordinateSystem), environment, geometry, topology,
                   fallbackHandler) {
      static_assert(!std::is_same_v<TOutputSink, NullCorsikaOutputSink>,
                    "the null output marker cannot be supplied as a sink");
      outputSink_ = &outputSink;
    }

    bool canRoute(particle_type const& particle, transport::StepId stepId) {
      ++statistics_.routeAttempts_;
      auto const fallbackKey = std::make_pair(particle.getHistoryId(), stepId);
      if (forcedDecaySteps_.find(fallbackKey) != forcedDecaySteps_.end()) {
        return false;
      }
      auto const expansion = cpuWavefrontExpansionSteps_.find(fallbackKey);
      if (expansion != cpuWavefrontExpansionSteps_.end()) {
        cpuWavefrontExpansionSteps_.erase(expansion);
        ++statistics_.cpuWavefrontExpansionStepsExecuted_;
        return false;
      }
      auto const blocked = cpuFallbackSteps_.find(fallbackKey);
      if (blocked != cpuFallbackSteps_.end()) {
        cpuFallbackSteps_.erase(blocked);
        ++statistics_.cpuFallbackStepsExecuted_;
        return false;
      }
      auto const memorySpill = cpuMemorySpillSteps_.find(fallbackKey);
      if (memorySpill != cpuMemorySpillSteps_.end()) {
        cpuMemorySpillSteps_.erase(memorySpill);
        ++statistics_.cpuMemorySpillStepsExecuted_;
        return false;
      }
      if (!is_em(particle.getPID()) && !is_muon(particle.getPID())) {
        ++statistics_.routeNonEmRejections_;
        return false;
      }
      auto const state = particle_transfer::to_device_state(
          particle, coordinateSystem_, particle.getHistoryId(),
          particle.getParentHistoryId(), particle.getGeneration(), stepId, topology_);
      if (!backend_.canTransport(state)) {
        ++statistics_.routeBackendRejections_;
        return false;
      }
      auto const layer =
          query_environment(environment_, geometry_.query_, state.positionM_,
                            state.direction_, state.environmentNode_);
      if (layer.status_ != EnvironmentStatus::Success)
        ++statistics_.routeEnvironmentRejections_;
      return layer.status_ == EnvironmentStatus::Success;
    }

    bool consumeForcedDecay(transport::HistoryId historyId, transport::StepId stepId) {
      auto const found = forcedDecaySteps_.find(std::make_pair(historyId, stepId));
      if (found == forcedDecaySteps_.end()) { return false; }
      forcedDecaySteps_.erase(found);
      ++statistics_.forcedCpuDecaysExecuted_;
      return true;
    }

    void stage(particle_type const& particle, transport::HistoryId historyId,
               transport::HistoryId parentHistoryId, transport::Generation generation,
               transport::StepId stepId) {
      cooperativeScalarYield_ = false;
      auto state = particle_transfer::to_device_state(particle, coordinateSystem_,
                                                      historyId, parentHistoryId,
                                                      generation, stepId, topology_);
      auto const layer =
          query_environment(environment_, geometry_.query_, state.positionM_,
                            state.direction_, state.environmentNode_);
      if (layer.status_ != EnvironmentStatus::Success) {
        throw std::logic_error(
            "physical CUDA EM router staged a particle outside its environment");
      }
      state.mediumId_ = environment_.getModel(state.environmentNode_)->mediumId_;
      auto const previous = lastCpuExpansionStates_.find(state.historyId_);
      if (previous != lastCpuExpansionStates_.end() &&
          sameTransportState(previous->second, state)) {
        forceSmallBatchToGpu_ = true;
      }
      staged_.push_back(state);
      ++statistics_.particlesStaged_;
    }

    bool preserveLogicalNodes() const { return true; }

    bool pending() const {
      return !staged_.empty() || backend_.pendingPhotonCount() != 0 ||
             backend_.pendingLeptonCount() != 0 || !deferredFallbackEvents_.empty() ||
             forceDeferredProductBatchToGpu_;
    }

    /*
     * Report whether the resident EM front should be advanced before the
     * scalar stack is exhausted.
     *
     * HybridCascade historically drained every runnable scalar history before
     * calling the router. At ultra-high energy this allowed routed gamma/e+/e-
     * states to accumulate without bound in staged_ while the CPU continued a
     * long hadronic front. Interleave only at the configured production batch
     * boundary (or when a device-resident continuation is waiting), preserving
     * the existing small-batch scalar-expansion policy.
     */
    bool readyForScalarInterleave() const noexcept {
      if (backend_.independentSubshowersEnabled()) {
        return !cooperativeScalarYield_ &&
               (forceSmallBatchToGpu_ || staged_.size() >= backend_.minimumBatchSize() ||
                backend_.independentSubshowersReady());
      }
      return forceSmallBatchToGpu_ || backend_.pendingPhotonCount() != 0 ||
             backend_.pendingLeptonCount() != 0 ||
             staged_.size() >= backend_.minimumBatchSize();
    }

    void endOfShower() {
      if (radioFinalized_) {
        throw std::logic_error("CUDA EM router output finalized more than once");
      }
      if (!staged_.empty() || backend_.pendingPhotonCount() != 0 ||
          backend_.pendingLeptonCount() != 0 || !deferredFallbackEvents_.empty() ||
          forceDeferredProductBatchToGpu_) {
        throw std::logic_error(
            "CUDA EM router cannot finalize with an unconsumed host or device wavefront");
      }
      auto const firstInteraction = backend_.downloadFirstInteractionSnapshot();
      if (firstInteraction.has_value()) {
        if constexpr (!std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
          if (outputSink_ == nullptr) {
            throw std::logic_error(
                "CUDA first-interaction output requires a production output sink");
          }
          outputSink_->onFirstInteraction(*firstInteraction);
        }
      }
      if (backend_.gpuProfileEnabled()) {
        if constexpr (!std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
          if (outputSink_ == nullptr) {
            throw std::logic_error(
                "CUDA resident profile requires a production output sink");
          }
          auto const profile = backend_.downloadProfile();
          outputSink_->onGpuProfile(profile);
          statistics_.particlesCut_ += profile.particleCuts_;
          statistics_.depositedEnergyGeV_ += profile.weightedDepositedEnergyGeV_;
          statistics_.mediumRestMassInputGeV_ += profile.weightedMediumRestMassInputGeV_;
          statistics_.cutRestMassEnergyGeV_ += profile.weightedCutRestMassEnergyGeV_;
          statistics_.observedTotalEnergyGeV_ += profile.weightedObservedTotalEnergyGeV_;
          statistics_.escapedTotalEnergyGeV_ += profile.weightedEscapedTotalEnergyGeV_;
          statistics_.unwrittenPhotoelectricBindingEnergyGeV_ +=
              profile.weightedUnwrittenPhotoelectricBindingEnergyGeV_;
          statistics_.observationCutOverlapEnergyGeV_ +=
              profile.weightedObservationCutOverlapEnergyGeV_;
          statistics_.massConventionCorrectionGeV_ +=
              profile.weightedMassConventionCorrectionGeV_;
        } else {
          throw std::logic_error(
              "CUDA resident profile cannot finalize through a null output sink");
        }
      }
      if (backend_.gpuRadioEnabled()) {
        if constexpr (!std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
          if (outputSink_ == nullptr) {
            throw std::logic_error("CUDA radio requires a production output sink");
          }
          auto const waveforms = backend_.downloadRadioWaveforms();
          outputSink_->onGpuRadioWaveforms(waveforms,
                                           backend_.statistics().radio_.leptonTracks_);
        } else {
          throw std::logic_error("CUDA radio cannot finalize through a null output sink");
        }
      }
      radioFinalized_ = true;
    }

    void setRetainRecords(bool retain) noexcept { retainRecords_ = retain; }

    bool retainsRecords() const noexcept { return retainRecords_; }

    void setFailOnUnexpectedFallback(bool fail) noexcept {
      failOnUnexpectedFallback_ = fail;
    }

    bool failsOnUnexpectedFallback() const noexcept { return failOnUnexpectedFallback_; }

    static constexpr bool isPermittedScalarFallbackReason(
        FallbackReason reason) noexcept {
      return reason == FallbackReason::UnsupportedParticle ||
             reason == FallbackReason::UnsupportedMedium ||
             reason == FallbackReason::UnsupportedGeometry;
    }

    std::size_t advanceOneWavefrontAndReturn(TStack& stack) {
      cooperativeScalarYield_ = false;
      auto const initialDevicePending =
          backend_.pendingPhotonCount() + backend_.pendingLeptonCount();
      if (staged_.empty() && initialDevicePending == 0) {
        if (!deferredFallbackEvents_.empty()) { return flushDeferredFallbacks(stack); }
        // A deferred CPU final-state batch can produce only non-EM children.
        // In that case the one-shot instruction to send its EM products back
        // to CUDA has no staged work to consume and must not leak into an
        // unrelated later wavefront.
        forceDeferredProductBatchToGpu_ = false;
        return 0;
      }
      auto const drainingBeforeDeferredFallback = !deferredFallbackEvents_.empty();
      if (staged_.size() < backend_.minimumBatchSize() && !forceSmallBatchToGpu_ &&
          !drainingBeforeDeferredFallback && !forceDeferredProductBatchToGpu_ &&
          initialDevicePending == 0 &&
          consecutiveScalarExpansionRounds_ < MaximumConsecutiveScalarExpansionRounds) {
        std::vector<ParticleState> expansion;
        expansion.swap(staged_);
        lastCpuExpansionStates_.clear();
        for (auto const& particle : expansion) {
          auto const key = std::make_pair(particle.historyId_, particle.stepId_);
          if (!cpuWavefrontExpansionSteps_.insert(key).second) {
            throw std::runtime_error("duplicate CPU wavefront-expansion step");
          }
          particle_transfer::import_particle(stack, particle, coordinateSystem_,
                                             topology_);
          lastCpuExpansionStates_.emplace(particle.historyId_, particle);
        }
        forceSmallBatchToGpu_ = false;
        ++consecutiveScalarExpansionRounds_;
        ++statistics_.smallBatchExpansions_;
        if (drainingBeforeDeferredFallback) {
          ++statistics_.deferredFallbackScalarExpansionRounds_;
        }
        statistics_.particlesReturnedForCpuWavefrontExpansion_ += expansion.size();
        auto const expansionCount = statistics_.smallBatchExpansions_;
        if (expansionCount <= 8 || (expansionCount & (expansionCount - 1)) == 0) {
          CORSIKA_LOG_DEBUG(
              "CUDA EM scalar wavefront expansion {}: {} particles "
              "(threshold {}, first pid/energy/history/step={}/{}/{}/{})",
              expansionCount, expansion.size(), backend_.minimumBatchSize(),
              expansion.front().pid_, expansion.front().energyGeV_,
              expansion.front().historyId_, expansion.front().stepId_);
        }
        return expansion.size();
      }
      if (staged_.size() < backend_.minimumBatchSize()) {
        if (forceSmallBatchToGpu_) {
          ++statistics_.stalledBoundaryGpuFlushes_;
        } else if (drainingBeforeDeferredFallback) {
          ++statistics_.deferredFrontGpuFlushes_;
        } else if (forceDeferredProductBatchToGpu_) {
          ++statistics_.deferredProductGpuFlushes_;
        } else if (initialDevicePending == 0 &&
                   consecutiveScalarExpansionRounds_ >=
                       MaximumConsecutiveScalarExpansionRounds) {
          ++statistics_.scalarExpansionBudgetGpuFlushes_;
          CORSIKA_LOG_DEBUG(
              "CUDA EM forced a {}-particle wavefront after {} "
              "consecutive scalar expansion rounds",
              staged_.size(), consecutiveScalarExpansionRounds_);
        }
      }
      forceSmallBatchToGpu_ = false;
      // Independent endpoints can accept a prefix while the remaining
      // fallback daughters are still on the scalar stack. Do not consume
      // their no-scalar-expansion instruction on that first partial handoff.
      // The empty-front branch above clears it after all work has drained.
      if (!backend_.batchIndependentSpecifiedFallbacks())
        forceDeferredProductBatchToGpu_ = false;
      consecutiveScalarExpansionRounds_ = 0;
      lastCpuExpansionStates_.clear();
      std::vector<ParticleState> current;
      current.swap(staged_);
      auto const inputLimit = backend_.maximumResidentInputBatchSize();
      if (current.size() > inputLimit) {
        staged_.assign(std::make_move_iterator(current.begin() +
                                               static_cast<std::ptrdiff_t>(inputLimit)),
                       std::make_move_iterator(current.end()));
        current.resize(inputLimit);
        ++statistics_.inputBatchSplits_;
      }
      ++statistics_.wavefronts_;

      if (backend_.independentSubshowersEnabled()) {
        return advanceIndependentSubshowers(stack, current);
      }

      std::vector<ParticleState> photons;
      std::vector<ParticleState> leptons;
      photons.reserve(current.size());
      leptons.reserve(current.size());
      for (auto const& particle : current) {
        if (particle.pid_ == static_cast<std::int32_t>(EmPid::Photon)) {
          photons.push_back(particle);
        } else if (is_charged_lepton_pid(particle.pid_)) {
          leptons.push_back(particle);
        } else {
          throw std::logic_error("physical CUDA EM router owns an unsupported particle");
        }
      }

      auto returnedToCpu = std::size_t{0};
      auto const maximumPhotonInput = backend_.maximumResidentPhotonBatchSize();
      auto const pendingPhotonChunk =
          std::min(backend_.pendingPhotonCount(), maximumPhotonInput);
      auto const hostPhotonLimit = maximumPhotonInput - pendingPhotonChunk;
      if (photons.size() > hostPhotonLimit) {
        staged_.insert(
            staged_.end(),
            std::make_move_iterator(photons.begin() +
                                    static_cast<std::ptrdiff_t>(hostPhotonLimit)),
            std::make_move_iterator(photons.end()));
        photons.resize(hostPhotonLimit);
        ++statistics_.inputBatchSplits_;
      }
      auto const photonInputCount = photons.size() + pendingPhotonChunk;
      if (photonInputCount != 0) {
        statistics_.maximumInputBatch_ =
            std::max(statistics_.maximumInputBatch_, photonInputCount);
        constexpr std::size_t ResidentPhotonWavefronts = 16;
        auto const firstHistory = reservePhotonSecondaryRange(stack, photonInputCount,
                                                              ResidentPhotonWavefronts);
        auto const backendStart = std::chrono::steady_clock::now();
        auto result = backend_.runResidentPhotonCascadeForValidation(
            photons, firstHistory, ResidentPhotonWavefronts, backend_.minimumBatchSize());
        auto const backendStop = std::chrono::steady_clock::now();
        statistics_.photonBackendWallTimeMs_ +=
            elapsedMilliseconds(backendStart, backendStop);
        auto const postprocessStart = std::chrono::steady_clock::now();
        returnedToCpu += consumeResidentPhoton(stack, result);
        auto const postprocessStop = std::chrono::steady_clock::now();
        statistics_.photonHostPostprocessTimeMs_ +=
            elapsedMilliseconds(postprocessStart, postprocessStop);
      }

      auto const leptonInputCount = leptons.size() + backend_.pendingLeptonCount();
      if (leptonInputCount != 0) {
        auto const maximumLeptonInput = backend_.maximumResidentLeptonBatchSize();
        auto const pendingLeptons =
            std::min(backend_.pendingLeptonCount(), maximumLeptonInput);
        auto const hostLeptonLimit = maximumLeptonInput - pendingLeptons;
        if (leptons.size() > hostLeptonLimit) {
          staged_.insert(
              staged_.end(),
              std::make_move_iterator(leptons.begin() +
                                      static_cast<std::ptrdiff_t>(hostLeptonLimit)),
              std::make_move_iterator(leptons.end()));
          leptons.resize(hostLeptonLimit);
          ++statistics_.inputBatchSplits_;
        }
        auto const routedLeptonInputCount = leptons.size() + pendingLeptons;
        statistics_.maximumInputBatch_ =
            std::max(statistics_.maximumInputBatch_, routedLeptonInputCount);
        auto const historyRange =
            reserveResidentSecondaryRange(stack, routedLeptonInputCount);
        auto const backendStart = std::chrono::steady_clock::now();
        auto result = backend_.runResidentLeptonCascadeForValidation(
            leptons, historyRange.first_, 1024, historyRange.limit_,
            backend_.minimumBatchSize());
        auto const backendStop = std::chrono::steady_clock::now();
        statistics_.leptonBackendWallTimeMs_ +=
            elapsedMilliseconds(backendStart, backendStop);
        auto const postprocessStart = std::chrono::steady_clock::now();
        returnedToCpu += consumeResidentLepton(stack, result);
        auto const postprocessStop = std::chrono::steady_clock::now();
        statistics_.leptonHostPostprocessTimeMs_ +=
            elapsedMilliseconds(postprocessStart, postprocessStop);
      }
      statistics_.electromagneticParticlesProduced_ += staged_.size();
      if (staged_.empty() && backend_.pendingPhotonCount() == 0 &&
          backend_.pendingLeptonCount() == 0 && !deferredFallbackEvents_.empty()) {
        returnedToCpu += flushDeferredFallbacks(stack);
      }
      return returnedToCpu;
    }

    EmRouterStatistics const& statistics() const { return statistics_; }

    std::vector<ObservationRecord> const& observations() const { return observations_; }

    std::vector<FallbackRequest> const& fallbackEvents() const { return fallbackEvents_; }

    std::size_t deferredFallbackCount() const noexcept {
      return deferredFallbackEvents_.size();
    }

    std::vector<StepRecord> const& stepRecords() const { return stepRecords_; }

    std::vector<InteractionRecord> const& interactionRecords() const {
      return interactionRecords_;
    }

    std::vector<PhotonFinalStateRecord> const& photonFinalStateRecords() const {
      return photonFinalStateRecords_;
    }

    std::vector<LeptonFinalStateRecord> const& leptonFinalStateRecords() const {
      return leptonFinalStateRecords_;
    }

    std::vector<RadioTrackRecord> const& radioTracks() const { return radioTracks_; }

  private:
    std::size_t advanceIndependentSubshowers(TStack& stack,
                                             std::vector<ParticleState> const& current) {
      // Consumers run exclusively on this coordinator. A result can be
      // consumed while the other endpoint is still executing its subshower.
      std::size_t returned = 0;
      SubshowerCallbacks callbacks;
      callbacks.reserveHistories_ = [&](std::uint64_t count) {
        if (!count || count > std::numeric_limits<std::uint64_t>::max() -
                                  statistics_.reservedHistoryIds_)
          throw std::overflow_error("independent subshower history reservation overflow");
        auto first = stack.reserveTransportHistoryIds(count);
        if (!first || count > std::numeric_limits<std::uint64_t>::max() - first)
          throw std::overflow_error("independent subshower history range overflow");
        statistics_.reservedHistoryIds_ += count;
        return first;
      };
      callbacks.photons_ = [&](PhotonCascadeResult&& result, double elapsed) {
        statistics_.maximumInputBatch_ =
            std::max(statistics_.maximumInputBatch_, result.inputParticles_);
        statistics_.photonBackendWallTimeMs_ += elapsed;
        auto start = std::chrono::steady_clock::now();
        returned += consumeResidentPhoton(stack, result);
        statistics_.photonHostPostprocessTimeMs_ +=
            elapsedMilliseconds(start, std::chrono::steady_clock::now());
      };
      callbacks.leptons_ = [&](LeptonCascadeResult&& result, double elapsed) {
        statistics_.maximumInputBatch_ =
            std::max(statistics_.maximumInputBatch_, result.inputParticles_);
        statistics_.leptonBackendWallTimeMs_ += elapsed;
        auto start = std::chrono::steady_clock::now();
        returned += consumeResidentLepton(stack, result);
        statistics_.leptonHostPostprocessTimeMs_ +=
            elapsedMilliseconds(start, std::chrono::steady_clock::now());
      };
      callbacks.yieldToScalar_ = [&] { return returned != 0; };
      auto accepted = backend_.advanceIndependentSubshowers(current, callbacks);
      if (accepted > current.size())
        throw std::logic_error("independent subshower accepted too many particles");
      staged_.insert(staged_.end(), current.begin() + accepted, current.end());
      if (accepted < current.size()) ++statistics_.inputBatchSplits_;
      statistics_.electromagneticParticlesProduced_ += staged_.size();
      cooperativeScalarYield_ = returned != 0;
      return returned;
    }

    std::size_t consumeResidentPhoton(TStack& stack, PhotonCascadeResult& result) {
      auto const firstPhotonStep = stepRecords_.size();
      std::size_t returnedToCpu = 0;
      if (retainRecords_) {
        interactionRecords_.insert(interactionRecords_.end(),
                                   result.interactionRecords_.begin(),
                                   result.interactionRecords_.end());
        photonFinalStateRecords_.insert(photonFinalStateRecords_.end(),
                                        result.finalStateRecords_.begin(),
                                        result.finalStateRecords_.end());
      }
      statistics_.photonsAdvanced_ += result.transportRecords_;
      statistics_.residentPhotonWavefronts_ += result.wavefronts_;
      if (result.belowMinimumBatchCheckpoint_) {
        ++statistics_.belowMinimumBatchCheckpoints_;
      }
      if (result.workspaceLimitCheckpoint_) { ++statistics_.workspaceLimitCheckpoints_; }
      if (!result.projectedStepRecords_.empty()) {
        if (retainRecords_ || !result.stepRecords_.empty()) {
          throw std::logic_error(
              "projected CUDA photon output is only valid for "
              "non-retaining production routing");
        }
        for (auto& record : result.projectedStepRecords_) {
          recordProjectedPhotonStep(record);
        }
        recordProjectedPhotonFinalStateDeposits(result.projectedStepRecords_,
                                                result.finalStateRecords_);
        publishProjectedSteps(result.projectedStepRecords_);
      } else {
        for (auto const& record : result.stepRecords_) { recordPhotonStep(record); }
        recordPhotonFinalStateDeposits(result.finalStateRecords_);
        publishPhotonSteps(firstPhotonStep);
        if (!retainRecords_) {
          stepRecords_.erase(
              stepRecords_.begin() + static_cast<std::ptrdiff_t>(firstPhotonStep),
              stepRecords_.end());
        }
      }
      appendParticles(result.electromagneticSecondaries_);
      appendParticles(result.remainingPhotons_);
      returnedToCpu += returnMemorySpills(stack, result.cpuSpillParticles_);
      returnedToCpu += returnFallbacks(stack, result.fallbackEvents_);
      for (auto const& observation : result.observations_) {
        recordObservation(observation);
      }

      return returnedToCpu;
    }

    std::size_t consumeResidentLepton(TStack& stack, LeptonCascadeResult& result) {
      auto const firstLeptonStep = stepRecords_.size();
      std::size_t returnedToCpu = 0;
      if (retainRecords_) {
        interactionRecords_.insert(interactionRecords_.end(),
                                   result.interactionRecords_.begin(),
                                   result.interactionRecords_.end());
        leptonFinalStateRecords_.insert(leptonFinalStateRecords_.end(),
                                        result.finalStateRecords_.begin(),
                                        result.finalStateRecords_.end());
      }
      statistics_.leptonsAdvanced_ += result.transportRecords_;
      statistics_.residentLeptonWavefronts_ += result.wavefronts_;
      if (result.historyRangeExhausted_) { ++statistics_.historyRangeCheckpoints_; }
      if (result.belowMinimumBatchCheckpoint_) {
        ++statistics_.belowMinimumBatchCheckpoints_;
      }
      if (result.workspaceLimitCheckpoint_) { ++statistics_.workspaceLimitCheckpoints_; }
      if (!result.projectedStepRecords_.empty()) {
        if (retainRecords_ || !result.stepRecords_.empty()) {
          throw std::logic_error(
              "projected CUDA lepton output is only valid for "
              "non-retaining production routing");
        }
        for (auto const& record : result.projectedStepRecords_) {
          recordProjectedLeptonStep(record);
        }
        recordProjectedLeptonFinalStateLedger(result.projectedStepRecords_,
                                              result.finalStateRecords_);
      } else {
        for (auto const& record : result.stepRecords_) {
          recordLeptonStep(record);
          statistics_.depositedEnergyGeV_ +=
              record.continuousDepositedEnergyGeV_ * record.start_.weight_;
          if (record.limit_ == LeptonTransportLimit::ParticleCut) {
            statistics_.depositedEnergyGeV_ +=
                record.cutDepositedEnergyGeV_ * record.start_.weight_;
            ++statistics_.particlesCut_;
          }
        }
        annotateLeptonProcessIds(firstLeptonStep, result.stepRecords_,
                                 result.finalStateRecords_);
        recordLeptonFinalStateLedger(result.stepRecords_, result.finalStateRecords_);
      }
      appendParticles(result.generatedPhotons_);
      appendParticles(result.remainingLeptons_);
      returnedToCpu += returnMemorySpills(stack, result.cpuSpillParticles_);
      returnedToCpu += returnFallbacks(stack, result.fallbackEvents_);
      returnedToCpu += returnForcedDecays(stack, result.decayCandidates_);
      for (auto const& observation : result.observations_) {
        recordObservation(observation);
      }

      return returnedToCpu;
    }

    static double elapsedMilliseconds(std::chrono::steady_clock::time_point start,
                                      std::chrono::steady_clock::time_point stop) {
      return std::chrono::duration<double, std::milli>(stop - start).count();
    }

    static bool sameTransportState(ParticleState const& left,
                                   ParticleState const& right) {
      if (left.pid_ != right.pid_ || left.environmentNode_ != right.environmentNode_ ||
          left.mediumId_ != right.mediumId_ || left.energyGeV_ != right.energyGeV_ ||
          left.timeS_ != right.timeS_ || left.weight_ != right.weight_) {
        return false;
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (left.positionM_[axis] != right.positionM_[axis] ||
            left.direction_[axis] != right.direction_[axis]) {
          return false;
        }
      }
      return true;
    }

    struct ReservedHistoryRange {
      transport::HistoryId first_{};
      transport::HistoryId limit_{};
      std::uint64_t count_{};
    };

    transport::HistoryId reserveSecondaryRange(TStack& stack, std::size_t inputCount) {
      if (inputCount > std::numeric_limits<std::uint64_t>::max() / 2) {
        throw std::overflow_error("physical CUDA EM history reservation overflow");
      }
      auto const count = static_cast<std::uint64_t>(inputCount) * 2;
      auto const first = stack.reserveTransportHistoryIds(count);
      statistics_.reservedHistoryIds_ += count;
      return first;
    }

    transport::HistoryId reservePhotonSecondaryRange(TStack& stack,
                                                     std::size_t inputCount,
                                                     std::size_t maximumWavefronts) {
      if (inputCount == 0 || maximumWavefronts == 0 ||
          inputCount >
              std::numeric_limits<std::uint64_t>::max() / 2 / maximumWavefronts) {
        throw std::overflow_error("resident CUDA photon history reservation overflow");
      }
      // One resident photon produces at most two final-state children in a
      // wavefront and at most one of them can remain a photon. Therefore the
      // active photon count cannot grow, and this is a strict upper bound for
      // the explicit checkpoint interval.
      auto const count = static_cast<std::uint64_t>(inputCount) * 2 * maximumWavefronts;
      auto const first = stack.reserveTransportHistoryIds(count);
      statistics_.reservedHistoryIds_ += count;
      return first;
    }

    ReservedHistoryRange reserveResidentSecondaryRange(TStack& stack,
                                                       std::size_t inputCount) {
      if (inputCount == 0 || inputCount > std::numeric_limits<std::uint64_t>::max() / 3) {
        throw std::overflow_error("resident CUDA EM history reservation overflow");
      }
      auto const minimum = static_cast<std::uint64_t>(inputCount) * 3;
      constexpr std::uint64_t ReservationMultiplier = 64;
      constexpr std::uint64_t MaximumChunk = 1U << 20;
      auto count = minimum;
      if (minimum <= std::numeric_limits<std::uint64_t>::max() / ReservationMultiplier) {
        count = std::min(MaximumChunk, minimum * ReservationMultiplier);
        count = std::max(count, minimum);
      }
      auto const first = stack.reserveTransportHistoryIds(count);
      if (count > std::numeric_limits<transport::HistoryId>::max() - first) {
        throw std::overflow_error("resident CUDA EM history range limit overflow");
      }
      auto const limit = first + count;
      statistics_.reservedHistoryIds_ += count;
      return {first, limit, count};
    }

    bool consumeNativeMuonPairCut(ParticleState const& particle) {
      if (!is_native_muon_pair_cut(particle)) return false;
      InteractionRecord interaction{};
      interaction.particle_ = particle;
      auto const record = native_muon_pair_cut_record(interaction);
      recordLeptonStep(record);
      ++statistics_.particlesCut_;
      ++statistics_.leptonsAdvanced_;
      statistics_.depositedEnergyGeV_ += record.cutDepositedEnergyGeV_ * particle.weight_;
      // A downloaded newborn/spill has not entered the device profile. With
      // device profiling off, recordLeptonStep already books this rest mass.
      if (backend_.gpuProfileEnabled())
        statistics_.cutRestMassEnergyGeV_ +=
            transport_mass_gev(particle.pid_) * particle.weight_;
      return true;
    }

    void appendParticles(std::vector<ParticleState> const& particles) {
      for (auto const& particle : particles)
        if (!consumeNativeMuonPairCut(particle)) staged_.push_back(particle);
    }

    void appendNonPhotonParticles(std::vector<ParticleState> const& particles) {
      for (auto const& particle : particles)
        if (particle.pid_ != static_cast<std::int32_t>(EmPid::Photon) &&
            !consumeNativeMuonPairCut(particle))
          staged_.push_back(particle);
    }

    std::size_t returnFallbacks(TStack& stack,
                                std::vector<FallbackRequest> const& events) {
      auto returnedToScalar = std::size_t{0};
      auto deferredNodeRefreshes = std::size_t{0};
      for (auto const& event : events) {
        if (proposal_fallback_requires_interaction_vertex(event.reason_) &&
            event.interactionVertexReached_ != 1) {
          throw std::runtime_error(
              "specified PROPOSAL fallback has not reached an interaction vertex");
        }
        ++statistics_.cpuFallbacksByProcess_[event.processId_];
        ++statistics_.cpuFallbacksByReason_[static_cast<std::int32_t>(event.reason_)];
        if constexpr (!std::is_same_v<TProposalFallbackHandler, ScalarProposalFallback>) {
          if (fallbackHandler_ != nullptr && fallbackHandler_->canHandle(event)) {
            if (backend_.independentSubshowersEnabled() &&
                !backend_.batchIndependentSpecifiedFallbacks()) {
              auto key =
                  std::make_pair(event.particle_.historyId_, event.particle_.stepId_);
              if (!cpuFallbackSteps_.insert(key).second)
                throw std::runtime_error("duplicate CPU fallback for one transport step");
              auto start = std::chrono::steady_clock::now();
              { fallbackHandler_->handle(stack, event); }
              statistics_.specifiedCpuFallbackTimeMs_ +=
                  elapsedMilliseconds(start, std::chrono::steady_clock::now());
              ++statistics_.specifiedCpuFinalStates_;
              if (retainRecords_) fallbackEvents_.push_back(event);
              // A nonzero return refreshes nodes even if cuts kept no child.
              ++returnedToScalar;
              forceDeferredProductBatchToGpu_ = true;
              continue;
            }
            // A specified PROPOSAL completion is independent of every other
            // particle in the active EM front. The single-endpoint policy
            // parks it until both resident queues and host staging drain;
            // adaptive independent mode also flushes bounded full batches
            // below while the peer remains live. Completing every event
            // here would put one or two children on the scalar stack after
            // every device checkpoint and repeatedly rebuild tiny CUDA
            // wavefronts.  Valid contract-0.18 electron/positron columns have
            // a device inverse CDF (including dry-air argon); the remaining
            // specified completions are chiefly rate-only muon radiative
            // branches and exceptional quantile-bound returns.
            if (backend_.batchIndependentSpecifiedFallbacks()) {
              auto const key =
                  std::make_pair(event.particle_.historyId_, event.particle_.stepId_);
              if (!cpuFallbackSteps_.insert(key).second)
                throw std::runtime_error("duplicate CPU fallback for one transport step");
            }
            deferredFallbackEvents_.push_back(event);
            ++statistics_.deferredCpuFallbacksQueued_;
            statistics_.maximumDeferredCpuFallbackBatch_ =
                std::max(statistics_.maximumDeferredCpuFallbackBatch_,
                         deferredFallbackEvents_.size());
            // Only the coordinator mutates this queue and the scalar Stack.
            // Bound retained completions by 4096 records; flush full batches
            // at this completed-result boundary, without touching or joining
            // an in-flight peer. A final partial batch uses the existing
            // empty-front flush. Every selected process/random state is kept.
            if (backend_.batchIndependentSpecifiedFallbacks() &&
                deferredFallbackEvents_.size() >= 4096)
              deferredNodeRefreshes += flushDeferredFallbacks(stack, true);
            continue;
          }
        }
        CORSIKA_LOG_WARN(
            "CUDA EM fallback returned to scalar transport: "
            "pid={}, energy_GeV={}, "
            "history={}, step={}, process={}, component={}, reason={}, "
            "diagnostic_status={}, diagnostic_values=[{},{},{}], "
            "position_m=[{},{},{}], direction=[{},{},{}]",
            event.particle_.pid_, event.particle_.energyGeV_, event.particle_.historyId_,
            event.particle_.stepId_, event.processId_, event.componentHash_,
            proposal_fallback_reason_name(event.reason_), event.diagnosticStatus_,
            event.diagnosticValue0_, event.diagnosticValue1_, event.diagnosticValue2_,
            event.particle_.positionM_[0], event.particle_.positionM_[1],
            event.particle_.positionM_[2], event.particle_.direction_[0],
            event.particle_.direction_[1], event.particle_.direction_[2]);
        if (failOnUnexpectedFallback_ &&
            !isPermittedScalarFallbackReason(event.reason_)) {
          std::ostringstream message;
          message << "unexpected CUDA EM fallback cannot be retried silently: "
                  << "pid=" << event.particle_.pid_
                  << ", energy_GeV=" << event.particle_.energyGeV_
                  << ", history=" << event.particle_.historyId_
                  << ", step=" << event.particle_.stepId_
                  << ", process=" << event.processId_
                  << ", component=" << event.componentHash_
                  << ", reason=" << proposal_fallback_reason_name(event.reason_)
                  << ", diagnostic_status=" << event.diagnosticStatus_
                  << ", diagnostic_values=[" << event.diagnosticValue0_ << ","
                  << event.diagnosticValue1_ << "," << event.diagnosticValue2_ << "]";
          throw std::runtime_error(message.str());
        }
        auto const key =
            std::make_pair(event.particle_.historyId_, event.particle_.stepId_);
        if (!cpuFallbackSteps_.insert(key).second) {
          throw std::runtime_error("duplicate CPU fallback for one transport step");
        }
        particle_transfer::import_particle(stack, event.particle_, coordinateSystem_,
                                           topology_);
        if (retainRecords_) { fallbackEvents_.push_back(event); }
        ++returnedToScalar;
      }
      statistics_.particlesReturnedForCpuFallback_ += returnedToScalar;
      // Deferred completions are already counted in queued/flushed, not a
      // second time as generic fallback. Their children still need setNodes.
      return returnedToScalar + deferredNodeRefreshes;
    }

    std::size_t flushDeferredFallbacks(TStack& stack,
                                       bool independentCheckpoint = false) {
      if (deferredFallbackEvents_.empty()) { return 0; }
      if constexpr (std::is_same_v<TProposalFallbackHandler, ScalarProposalFallback>) {
        throw std::logic_error(
            "deferred PROPOSAL fallbacks require a CPU final-state handler");
      } else {
        if (fallbackHandler_ == nullptr) {
          throw std::logic_error("deferred PROPOSAL fallback handler is unavailable");
        }
        if (independentCheckpoint && (!backend_.independentSubshowersEnabled() ||
                                      !backend_.batchIndependentSpecifiedFallbacks()))
          throw std::logic_error(
              "active-front fallback flush requires adaptive independent ownership");
        if (!independentCheckpoint &&
            (!staged_.empty() || backend_.pendingPhotonCount() != 0 ||
             backend_.pendingLeptonCount() != 0)) {
          throw std::logic_error(
              "deferred PROPOSAL fallbacks may only flush after the CUDA EM front "
              "drains");
        }

        std::vector<FallbackRequest> batch;
        batch.swap(deferredFallbackEvents_);
        auto const fallbackStart = std::chrono::steady_clock::now();
        for (auto const& event : batch) {
          if (!fallbackHandler_->canHandle(event)) {
            throw std::logic_error(
                "deferred PROPOSAL fallback lost its specified CPU capability");
          }
          { fallbackHandler_->handle(stack, event); }
          if (retainRecords_) { fallbackEvents_.push_back(event); }
          ++statistics_.specifiedCpuFinalStates_;
        }
        auto const fallbackStop = std::chrono::steady_clock::now();
        statistics_.specifiedCpuFallbackTimeMs_ +=
            std::chrono::duration<double, std::milli>(fallbackStop - fallbackStart)
                .count();
        statistics_.deferredCpuFallbacksFlushed_ += batch.size();
        ++statistics_.deferredCpuFallbackFlushes_;

        // The specified CPU final states are generated only after the active
        // CUDA front has drained.  Their EM products therefore arrive as a
        // fresh, often sub-threshold host batch.  Keep those products on the
        // GPU path instead of expanding them for up to eight scalar rounds.
        // pending() retains this one-shot flag until HybridCascade has drained
        // the generated CPU stack and either stages the complete EM product
        // batch or confirms that the batch produced no EM particle.
        forceDeferredProductBatchToGpu_ = true;
        consecutiveScalarExpansionRounds_ = 0;
        lastCpuExpansionStates_.clear();

        // handle() may have generated no surviving secondary after cuts or
        // thinning, but a non-zero return still tells HybridCascade to refresh
        // environment nodes once for the whole CPU batch.
        return batch.size();
      }
    }

    std::size_t returnMemorySpills(TStack& stack,
                                   std::vector<ParticleState> const& particles) {
      std::size_t returned = 0;
      for (auto const& particle : particles) {
        // Native C8 cuts these newborn states before any transport. Do not
        // import them into the CPU stack's ordinary transport entry point.
        if (consumeNativeMuonPairCut(particle)) continue;
        auto const key = std::make_pair(particle.historyId_, particle.stepId_);
        if (!cpuMemorySpillSteps_.insert(key).second) {
          throw std::runtime_error("duplicate CPU memory spill for one transport step");
        }
        particle_transfer::import_particle(stack, particle, coordinateSystem_, topology_);
        ++returned;
      }
      statistics_.particlesReturnedForCpuMemorySpill_ += returned;
      return returned;
    }

    std::size_t returnForcedDecays(TStack& stack,
                                   std::vector<ParticleState> const& particles) {
      for (auto const& particle : particles) {
        auto const key = std::make_pair(particle.historyId_, particle.stepId_);
        if (!forcedDecaySteps_.insert(key).second) {
          throw std::runtime_error("duplicate forced CPU decay for one transport step");
        }
        particle_transfer::import_particle(stack, particle, coordinateSystem_, topology_);
      }
      statistics_.particlesReturnedForCpuDecay_ += particles.size();
      return particles.size();
    }

    void recordObservation(ObservationRecord const& observation) {
      if (retainRecords_) { observations_.push_back(observation); }
      if constexpr (!std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
        if (outputSink_ != nullptr) { outputSink_->onObservation(observation); }
      }
      if (observation.status_ == ObservationStatus::ReachedObservationSurface) {
        ++statistics_.particlesObserved_;
      } else {
        ++statistics_.particlesEscaped_;
      }
      if (!backend_.gpuProfileEnabled()) {
        auto const weighted_total_energy =
            observation.particle_.energyGeV_ * observation.particle_.weight_;
        if (!std::isfinite(weighted_total_energy) || weighted_total_energy < 0.) {
          throw std::runtime_error(
              "CUDA EM observation has invalid weighted total energy");
        }
        if (observation.status_ == ObservationStatus::ReachedObservationSurface) {
          statistics_.observedTotalEnergyGeV_ += weighted_total_energy;
        } else {
          statistics_.escapedTotalEnergyGeV_ += weighted_total_energy;
        }
      }
    }

    static StepRecord makeStepRecord(ParticleState const& start, ParticleState const& end,
                                     std::int32_t processId, double depositedEnergyGeV) {
      StepRecord step{};
      step.historyId_ = start.historyId_;
      step.stepId_ = start.stepId_;
      step.pid_ = start.pid_;
      step.processId_ = processId;
      for (int axis = 0; axis < 3; ++axis) {
        step.startPositionM_[axis] = start.positionM_[axis];
        step.endPositionM_[axis] = end.positionM_[axis];
      }
      step.startTimeS_ = start.timeS_;
      step.endTimeS_ = end.timeS_;
      step.startEnergyGeV_ = start.energyGeV_;
      step.endEnergyGeV_ = end.energyGeV_;
      step.depositedEnergyGeV_ = depositedEnergyGeV;
      step.weight_ = start.weight_;
      return step;
    }

    static double leptonRestMassGeV(std::int32_t pid) {
      if (is_electron_or_positron_pid(pid)) { return TransportElectronMassGeV; }
      if (is_muon_pid(pid)) { return TransportMuonMassGeV; }
      throw std::logic_error("CUDA charged-lepton record has an unsupported PID");
    }

    void recordPhotonStep(PhotonTransportRecord const& record) {
      auto const processId = record.limit_ == PhotonTransportLimit::Interaction
                                 ? record.interaction_.processId_
                                 : 0;
      auto const deposited = record.cutDepositedEnergyGeV_;
      auto step = makeStepRecord(record.start_, record.end_, processId, deposited);
      step.cutDepositedEnergyGeV_ = deposited;
      stepRecords_.push_back(step);
      if (record.limit_ == PhotonTransportLimit::ParticleCut) {
        statistics_.depositedEnergyGeV_ += deposited * record.start_.weight_;
        if (record.observationSurfaceReachedBeforeCut_ != 0U)
          statistics_.observationCutOverlapEnergyGeV_ +=
              deposited * record.start_.weight_;
        ++statistics_.particlesCut_;
      }
    }

    void recordPhotonFinalStateDeposits(
        std::vector<PhotonFinalStateRecord> const& finalStateRecords) {
      for (auto const& record : finalStateRecords) {
        statistics_.massConventionCorrectionGeV_ +=
            record.weightedMassConventionCorrectionGeV_;
        if (record.processId_ != ComptonProcessId &&
            record.processId_ != PhotoelectricProcessId) {
          continue;
        }
        if (!std::isfinite(record.energySplitFraction_) ||
            record.energySplitFraction_ < 0. || record.energySplitFraction_ > 1.) {
          throw std::runtime_error(
              "photoelectric final state has invalid deposit metadata");
        }
        auto const step = std::find_if(
            stepRecords_.rbegin(), stepRecords_.rend(), [&](StepRecord const& candidate) {
              return candidate.historyId_ == record.parentHistoryId_ &&
                     candidate.processId_ == record.processId_;
            });
        if (step == stepRecords_.rend()) {
          throw std::runtime_error(
              "atomic-electron photon final state has no matching transport step");
        }
        statistics_.mediumRestMassInputGeV_ += ElectronMassGeV * step->weight_;
        if (record.processId_ != PhotoelectricProcessId) { continue; }
        auto const deposit = step->endEnergyGeV_ * (1. - record.energySplitFraction_);
        statistics_.unwrittenPhotoelectricBindingEnergyGeV_ += deposit * step->weight_;
      }
    }

    void recordProjectedPhotonStep(ProjectedStepRecord& record) {
      if (record.transportLimit_ ==
          static_cast<std::int32_t>(PhotonTransportLimit::ParticleCut)) {
        statistics_.depositedEnergyGeV_ += record.depositedEnergyGeV_ * record.weight_;
        if (record.observationSurfaceReachedBeforeCut_ != 0U)
          statistics_.observationCutOverlapEnergyGeV_ +=
              record.cutDepositedEnergyGeV_ * record.weight_;
        ++statistics_.particlesCut_;
      }
    }

    void recordProjectedPhotonFinalStateDeposits(
        std::vector<ProjectedStepRecord>& steps,
        std::vector<PhotonFinalStateRecord> const& finalStateRecords) {
      for (auto const& record : finalStateRecords) {
        statistics_.massConventionCorrectionGeV_ +=
            record.weightedMassConventionCorrectionGeV_;
        if (record.processId_ != ComptonProcessId &&
            record.processId_ != PhotoelectricProcessId) {
          continue;
        }
        if (!std::isfinite(record.energySplitFraction_) ||
            record.energySplitFraction_ < 0. || record.energySplitFraction_ > 1.) {
          throw std::runtime_error(
              "photoelectric final state has invalid projected deposit metadata");
        }
        auto const step = std::find_if(
            steps.rbegin(), steps.rend(), [&](ProjectedStepRecord const& candidate) {
              return candidate.historyId_ == record.parentHistoryId_ &&
                     candidate.processId_ == record.processId_;
            });
        if (step == steps.rend()) {
          throw std::runtime_error(
              "atomic-electron photon final state has no matching projected step");
        }
        statistics_.mediumRestMassInputGeV_ += ElectronMassGeV * step->weight_;
        if (record.processId_ != PhotoelectricProcessId) { continue; }
        auto const deposit = step->endEnergyGeV_ * (1. - record.energySplitFraction_);
        statistics_.unwrittenPhotoelectricBindingEnergyGeV_ += deposit * step->weight_;
      }
    }

    void recordLeptonStep(LeptonTransportRecord const& record) {
      auto const deposited =
          record.continuousDepositedEnergyGeV_ + record.cutDepositedEnergyGeV_;
      auto const processId =
          (record.limit_ == LeptonTransportLimit::InteractionCandidate ||
           record.interaction_.status_ == InteractionStatus::AtRestAnnihilation)
              ? record.interaction_.processId_
              : 0;
      auto step = makeStepRecord(record.start_, record.end_, processId, deposited);
      step.cutDepositedEnergyGeV_ = record.cutDepositedEnergyGeV_;
      step.nativeMuonPairCut_ =
          is_native_muon_pair_cut(record.start_) ? NativeMuonPairCutFlag : 0U;
      if (retainRecords_) { stepRecords_.push_back(step); }
      if constexpr (!std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
        if (outputSink_ != nullptr) { outputSink_->onStep(step); }
      }
      // The scalar CORSIKA radio process deliberately ignores muons. Preserve
      // that observable definition even though the charged transport kernel
      // also advances mu-/mu+.
      if (is_electron_or_positron_pid(record.start_.pid_)) {
        RadioTrackRecord radio{};
        radio.step_ = step;
        for (int axis = 0; axis < 3; ++axis) {
          radio.startDirection_[axis] = record.start_.direction_[axis];
          radio.endDirection_[axis] = record.end_.direction_[axis];
        }
        if (retainRecords_) { radioTracks_.push_back(radio); }
        if constexpr (!std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
          if (outputSink_ != nullptr) { outputSink_->onRadioTrack(radio); }
        }
      }
      if (!backend_.gpuProfileEnabled() &&
          record.limit_ == LeptonTransportLimit::ParticleCut) {
        if (record.interaction_.status_ != InteractionStatus::AtRestAnnihilation)
          statistics_.cutRestMassEnergyGeV_ +=
              leptonRestMassGeV(record.start_.pid_) * record.start_.weight_;
        if (record.observationSurfaceReachedBeforeCut_ != 0U)
          statistics_.observationCutOverlapEnergyGeV_ +=
              (record.cutDepositedEnergyGeV_ + leptonRestMassGeV(record.start_.pid_)) *
              record.start_.weight_;
      }
    }

    void recordLeptonFinalStateLedger(
        std::vector<LeptonTransportRecord> const& transports,
        std::vector<LeptonFinalStateRecord> const& finalStates) {
      for (auto const& record : finalStates)
        statistics_.massConventionCorrectionGeV_ +=
            record.weightedMassConventionCorrectionGeV_;
      auto const hasAtomicElectronProcess =
          std::any_of(finalStates.begin(), finalStates.end(),
                      [](LeptonFinalStateRecord const& record) {
                        return record.processId_ == IonizationProcessId ||
                               record.processId_ == AnnihilationProcessId;
                      });
      if (!hasAtomicElectronProcess) { return; }
      std::unordered_map<std::uint64_t, double> weightsByHistory;
      weightsByHistory.reserve(transports.size());
      for (auto const& transport : transports) {
        weightsByHistory.try_emplace(transport.start_.historyId_,
                                     transport.start_.weight_);
      }
      for (auto const& record : finalStates) {
        if (record.processId_ != IonizationProcessId &&
            record.processId_ != AnnihilationProcessId) {
          continue;
        }
        auto const weight = weightsByHistory.find(record.parentHistoryId_);
        if (weight == weightsByHistory.end()) {
          throw std::runtime_error(
              "atomic-electron lepton final state has no matching transport step");
        }
        statistics_.mediumRestMassInputGeV_ +=
            (record.processId_ == AnnihilationProcessId &&
                     record.finalStateDrawId_ == AtRestAnnihilationPolarDrawId
                 ? TransportElectronMassGeV
                 : ElectronMassGeV) *
            weight->second;
      }
    }

    void recordProjectedLeptonFinalStateLedger(
        std::vector<ProjectedStepRecord> const& transports,
        std::vector<LeptonFinalStateRecord> const& finalStates) {
      for (auto const& record : finalStates) {
        statistics_.massConventionCorrectionGeV_ +=
            record.weightedMassConventionCorrectionGeV_;
        if (record.processId_ != IonizationProcessId &&
            record.processId_ != AnnihilationProcessId)
          continue;
        auto const transport =
            std::find_if(transports.rbegin(), transports.rend(),
                         [&](ProjectedStepRecord const& candidate) {
                           return candidate.historyId_ == record.parentHistoryId_;
                         });
        if (transport == transports.rend())
          throw std::runtime_error(
              "atomic-electron lepton final state has no matching projected step");
        statistics_.mediumRestMassInputGeV_ +=
            (record.processId_ == AnnihilationProcessId &&
                     record.finalStateDrawId_ == AtRestAnnihilationPolarDrawId
                 ? TransportElectronMassGeV
                 : ElectronMassGeV) *
            transport->weight_;
      }
    }

    void annotateLeptonProcessIds(
        std::size_t firstStep, std::vector<LeptonTransportRecord> const& transports,
        std::vector<LeptonFinalStateRecord> const& finalStates) {
      if (!retainRecords_) { return; }
      if (stepRecords_.size() - firstStep != transports.size()) {
        throw std::logic_error("retained lepton transport records lost index alignment");
      }
      std::unordered_map<std::uint64_t, std::vector<std::int32_t>> processesByHistory;
      for (auto const& finalState : finalStates) {
        processesByHistory[finalState.parentHistoryId_].push_back(finalState.processId_);
      }
      std::unordered_map<std::uint64_t, std::size_t> nextProcess;
      for (std::size_t index = 0; index < transports.size(); ++index) {
        auto const& transport = transports[index];
        if (transport.limit_ != LeptonTransportLimit::InteractionCandidate &&
            transport.interaction_.status_ != InteractionStatus::AtRestAnnihilation) {
          continue;
        }
        auto const history = transport.start_.historyId_;
        auto const found = processesByHistory.find(history);
        if (found == processesByHistory.end()) { continue; }
        auto& offset = nextProcess[history];
        if (offset >= found->second.size()) { continue; }
        stepRecords_[firstStep + index].processId_ = found->second[offset++];
      }
    }

    void recordProjectedLeptonStep(ProjectedStepRecord const& record) {
      statistics_.depositedEnergyGeV_ += record.depositedEnergyGeV_ * record.weight_;
      if (record.transportLimit_ ==
          static_cast<std::int32_t>(LeptonTransportLimit::ParticleCut)) {
        ++statistics_.particlesCut_;
        if (record.processId_ != AnnihilationProcessId)
          statistics_.cutRestMassEnergyGeV_ +=
              leptonRestMassGeV(record.pid_) * record.weight_;
        if (record.observationSurfaceReachedBeforeCut_ != 0U)
          statistics_.observationCutOverlapEnergyGeV_ +=
              (record.cutDepositedEnergyGeV_ + leptonRestMassGeV(record.pid_)) *
              record.weight_;
      }
      if constexpr (!std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
        if (outputSink_ != nullptr) { outputSink_->onProjectedStep(record); }
      }
    }

    void publishPhotonSteps(std::size_t first) {
      if (first > stepRecords_.size()) {
        throw std::logic_error("invalid CUDA photon output publication range");
      }
      if constexpr (!std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
        if (outputSink_ != nullptr) {
          for (auto index = first; index < stepRecords_.size(); ++index) {
            outputSink_->onStep(stepRecords_[index]);
          }
        }
      }
    }

    void publishProjectedSteps(std::vector<ProjectedStepRecord> const& records) {
      if constexpr (!std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
        if (outputSink_ != nullptr) {
          for (auto const& record : records) { outputSink_->onProjectedStep(record); }
        }
      }
    }

    TBackend& backend_;
    CoordinateSystemPtr coordinateSystem_;
    EnvironmentBorrowedView environment_{};
    TransportGeometryConfig geometry_{};
    EnvironmentTopology<TModelProperties> const& topology_;
    std::vector<ParticleState> staged_{};
    std::set<std::pair<transport::HistoryId, transport::StepId>> cpuFallbackSteps_{};
    std::set<std::pair<transport::HistoryId, transport::StepId>> cpuMemorySpillSteps_{};
    std::set<std::pair<transport::HistoryId, transport::StepId>>
        cpuWavefrontExpansionSteps_{};
    std::set<std::pair<transport::HistoryId, transport::StepId>> forcedDecaySteps_{};
    std::map<transport::HistoryId, ParticleState> lastCpuExpansionStates_{};
    static constexpr std::uint32_t MaximumConsecutiveScalarExpansionRounds = 8;
    std::uint32_t consecutiveScalarExpansionRounds_{};
    bool cooperativeScalarYield_{};
    bool forceSmallBatchToGpu_{};
    bool forceDeferredProductBatchToGpu_{};
    std::vector<ObservationRecord> observations_{};
    std::vector<FallbackRequest> fallbackEvents_{};
    std::vector<FallbackRequest> deferredFallbackEvents_{};
    std::vector<StepRecord> stepRecords_{};
    std::vector<InteractionRecord> interactionRecords_{};
    std::vector<PhotonFinalStateRecord> photonFinalStateRecords_{};
    std::vector<LeptonFinalStateRecord> leptonFinalStateRecords_{};
    std::vector<RadioTrackRecord> radioTracks_{};
    TProposalFallbackHandler* fallbackHandler_{};
    TOutputSink* outputSink_{};
    bool retainRecords_{true};
    bool failOnUnexpectedFallback_{};
    bool radioFinalized_{};
    EmRouterStatistics statistics_{};
  };

} // namespace corsika::kokkos
