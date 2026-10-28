/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  namespace photon_cascade_detail {
    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonTransportStatisticsFunctor<TExecutionSpace>::init(
        value_type& value) const {
      for (std::size_t index = 0; index < PhotonTransportStatisticCount; ++index)
        value.values_[index] = 0;
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonTransportStatisticsFunctor<TExecutionSpace>::join(
        value_type& destination, value_type const& source) const {
      for (std::size_t index = 0; index < PhotonTransportStatisticCount; ++index)
        destination.values_[index] += source.values_[index];
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonTransportStatisticsFunctor<TExecutionSpace>::final(
        value_type& value) const {
      for (std::size_t index = 0; index < PhotonTransportStatisticCount; ++index)
        callStatistics_(PhotonCallTransportStatisticOffset + index) +=
            value.values_[index];
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonTransportStatisticsFunctor<TExecutionSpace>::operator()(
        std::size_t const source, value_type& value) const {

      auto const& selection = selections_(source);
      value.values_[PhotonSelectionNativeNewtonIterations] +=
          selection.nativeNewtonIterations_;
      value.values_[PhotonSelectionNativeBisectionIterations] +=
          selection.nativeBisectionIterations_;
      value.values_[PhotonSelectionNativeInverseFailures] +=
          selection.nativeInverseFailures_;
      if (source >= stepCount_) return;
      auto const limit = packedSteps_(source).limit_;
      value.values_[PhotonTransportInteractionVertices] +=
          limit == PhotonTransportLimit::Interaction;
      value.values_[PhotonTransportLayerBoundaries] +=
          limit == PhotonTransportLimit::LayerBoundary;
      value.values_[PhotonTransportParticleCuts] +=
          limit == PhotonTransportLimit::ParticleCut;
    }
    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonTransportStatisticsProfileFunctor<TExecutionSpace>::init(
        value_type& value) const {
      statistics_.init(value);
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonTransportStatisticsProfileFunctor<TExecutionSpace>::join(
        value_type& destination, value_type const& source) const {
      statistics_.join(destination, source);
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonTransportStatisticsProfileFunctor<TExecutionSpace>::final(
        value_type& value) const {
      statistics_.final(value);
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonTransportStatisticsProfileFunctor<TExecutionSpace>::operator()(
        std::size_t const source, value_type& value) const {
      statistics_(source, value);
      if (source < statistics_.stepCount_)
        accumulate_photon_profile_step<ProfileAtomicOperations>(
            projection_, accumulator_, statistics_.packedSteps_(source));
    }
    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonInteractionStatisticsFunctor<TExecutionSpace>::init(
        value_type& value) const {
      for (std::size_t index = 0; index < PhotonInteractionStatisticCount; ++index)
        value.values_[index] = 0;
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonInteractionStatisticsFunctor<TExecutionSpace>::join(
        value_type& destination, value_type const& source) const {
      for (std::size_t index = 0; index < PhotonInteractionStatisticCount; ++index)
        destination.values_[index] += source.values_[index];
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonInteractionStatisticsFunctor<TExecutionSpace>::final(
        value_type& value) const {
      for (std::size_t index = 0; index < PhotonInteractionStatisticCount; ++index)
        callStatistics_(PhotonCallInteractionStatisticOffset + index) +=
            value.values_[index];
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonInteractionStatisticsFunctor<TExecutionSpace>::operator()(
        std::size_t const interactionIndex, value_type& value) const {

      auto const& final = finalStates_(interactionIndex);
      value.values_[PhotonInteractionLpmSuppressions] += final.suppressionFlag_;
      value.values_[PhotonInteractionGpuFinalStates] += final.recordFlag_;
      if (final.recordFlag_ != 0) {
        value.values_[PhotonInteractionPhysicalSecondaries] += final.childCount_;
        value.values_[PhotonInteractionPairFinalStates] += final.photonPairFlag_;
        value.values_[PhotonInteractionMuonPairFinalStates] += final.photonMuonPairFlag_;
        value.values_[PhotonInteractionComptonFinalStates] += final.comptonFlag_;
        value.values_[PhotonInteractionPhotoelectricFinalStates] +=
            final.photoelectricFlag_;
        auto const status =
            static_cast<EmThinningStatus>(final.parameters_.thinningStatus_);
        value.values_[PhotonInteractionThinningHillasVertices] +=
            status == EmThinningStatus::Hillas;
        value.values_[PhotonInteractionThinningStatisticalVertices] +=
            status == EmThinningStatus::Statistical;
        auto const originalMultiplicity =
            final.parameters_.processId_ == PhotoelectricProcessId ? 1U : 2U;
        if (final.childCount_ <= originalMultiplicity)
          value.values_[PhotonInteractionThinningParticlesDiscarded] +=
              originalMultiplicity - final.childCount_;
      }
      value.values_[PhotonInteractionLpmTrials] +=
          final.photonPairFlag_ + final.suppressionFlag_;
    }
    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonInteractionStatisticsProfileFunctor<TExecutionSpace>::init(
        value_type& value) const {
      statistics_.init(value);
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonInteractionStatisticsProfileFunctor<TExecutionSpace>::join(
        value_type& destination, value_type const& source) const {
      statistics_.join(destination, source);
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonInteractionStatisticsProfileFunctor<TExecutionSpace>::final(
        value_type& value) const {
      statistics_.final(value);
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonInteractionStatisticsProfileFunctor<TExecutionSpace>::operator()(
        std::size_t const interactionIndex, value_type& value) const {
      statistics_(interactionIndex, value);
      if (interactionIndex < recordCount_)
        accumulate_photon_profile_final_state<ProfileAtomicOperations>(
            projection_, accumulator_, steps_.data(), stepCount_,
            records_(interactionIndex));
    }
  } // namespace photon_cascade_detail
  template <class TExecutionSpace>
  PhotonCascadeResult run_photon_cascade(
      typename PhysicsContext<TExecutionSpace>::device_view_type const physicsContext,
      std::vector<ParticleState> const& particles,
      std::uint64_t const firstSecondaryHistoryId, std::size_t const maximumWavefronts,
      std::size_t const minimumResidentBatchSize,
      std::optional<FirstInteractionRecord>& firstInteraction, bool const projectSteps,
      ProfileProjectionView const profileProjection,
      ProfileAccumulatorStorage<TExecutionSpace>* const profileAccumulator,
      PhotonFrontWorkspace<TExecutionSpace>* const workspace,
      PendingParticleQueue<TExecutionSpace>* const pendingPhotons,
      PendingParticleQueue<TExecutionSpace>* const pendingLeptons,
      std::size_t const pendingInputCount, std::size_t const maximumPendingParticles,
      TExecutionSpace const& execution, bool const captureDiagnosticInteractions,
      std::size_t const openmpChunkSize, ResidentMemoryBudgetGate const memoryGate,
      ResidentExecutionWait<TExecutionSpace>* const cooperativeWait) {
    using namespace photon_cascade_detail;

    auto const policy = [openmpChunkSize](TExecutionSpace const& selectedExecution,
                                          std::size_t const begin,
                                          std::size_t const end) {
      if (begin != 0 || end < begin)
        throw std::invalid_argument("resident photon RangePolicy requires a zero origin");
      return make_range_policy(selectedExecution, end, openmpChunkSize);
    };
    PhotonCascadeResult result{};
    if ((pendingPhotons == nullptr && pendingInputCount != 0) ||
        (pendingPhotons != nullptr && pendingInputCount > pendingPhotons->size()))
      throw std::invalid_argument(
          "resident Kokkos photon pending-input count is invalid");
    auto const pendingInput = pendingInputCount;
    if (pendingInput > std::numeric_limits<std::size_t>::max() - particles.size())
      throw std::overflow_error("resident Kokkos photon input size overflow");
    auto const totalInput = pendingInput + particles.size();
    result.inputParticles_ = totalInput;
    if (totalInput == 0) {
      result.completed_ = true;
      return result;
    }
    if (firstSecondaryHistoryId == 0 || maximumWavefronts == 0 ||
        minimumResidentBatchSize == 0)
      throw std::invalid_argument(
          "resident Kokkos photon cascade requires nonzero limits");

    auto const capacity = totalInput;
    if (capacity > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error(
          "resident Kokkos photon input exceeds addressable charged capacity");
    auto const chargedCapacity = capacity * 2;
    PhotonFrontWorkspace<TExecutionSpace> localWorkspace;
    auto& activeWorkspace = workspace != nullptr ? *workspace : localWorkspace;
    auto spillUnstagedInput = [&]() {
      result.workspaceLimitCheckpoint_ = true;
      result.completed_ = true;
      result.cpuSpillParticles_.reserve(totalInput);
      if (pendingInput != 0) {
        auto prefix = pendingPhotons->downloadPrefix(pendingInput, execution);
        result.cpuSpillParticles_.insert(result.cpuSpillParticles_.end(),
                                         std::make_move_iterator(prefix.begin()),
                                         std::make_move_iterator(prefix.end()));
        pendingPhotons->consume(pendingInput);
      }
      result.cpuSpillParticles_.insert(result.cpuSpillParticles_.end(), particles.begin(),
                                       particles.end());
    };
    auto const initialProjection = activeWorkspace.projectedCapacity(capacity);
    if (!memoryGate.permits(activeWorkspace.deviceBytes(), initialProjection,
                            "Kokkos photon resident workspace growth")) {
      spillUnstagedInput();
      return result;
    }
    activeWorkspace.ensureCapacity(capacity, execution);
    auto& queue = activeWorkspace.queue_;
    WavefrontExecutionGuard<TExecutionSpace> queueExecutionGuard(queue, execution);
    auto spillActiveFront = [&]() {
      auto active = queue.download(execution);
      result.cpuSpillParticles_.insert(result.cpuSpillParticles_.end(),
                                       std::make_move_iterator(active.begin()),
                                       std::make_move_iterator(active.end()));
      queue.clear();
      result.workspaceLimitCheckpoint_ = true;
    };
    if (pendingInput != 0) {
      queue.upload(particles, pendingPhotons->view(), pendingPhotons->head(),
                   pendingInput, execution);
      pendingPhotons->consume(pendingInput);
    } else {
      queue.upload(particles, execution);
    }
    auto const& selections = activeWorkspace.selections_;
    auto const& finalStates = activeWorkspace.finalStates_;
    auto const& steps = activeWorkspace.steps_;
    auto const& projectedSteps = activeWorkspace.projectedSteps_;
    auto const& records = activeWorkspace.records_;
    auto const& charged = activeWorkspace.charged_;
    auto const& fallbacks = activeWorkspace.fallbacks_;
    auto const& observations = activeWorkspace.observations_;
    auto const& firstSnapshots = activeWorkspace.firstSnapshots_;
    auto const& firstInteractionCandidates = activeWorkspace.firstInteractionCandidates_;
    auto const& error = activeWorkspace.error_;
    auto const& callStatistics = activeWorkspace.callStatistics_;
    auto const interactionCountDevice = activeWorkspace.interactionCount_;
    auto const scanTotalsDevice = activeWorkspace.scanTotals_;
    auto const frontControlDevice = activeWorkspace.frontControl_;
    auto const hostFrontControl = activeWorkspace.hostFrontControl_;
    Kokkos::deep_copy(execution, callStatistics, std::uint64_t{0});

    auto appendHost = [](auto& destination, auto const& host) {
      auto const count = host.extent(0);
      if (count == 0) return;
      auto const old = destination.size();
      destination.resize(old + count);
      for (std::size_t i = 0; i < count; ++i) destination[old + i] = host(i);
    };

    std::uint64_t nextHistoryId = firstSecondaryHistoryId;
    result.peakResidentPhotons_ = queue.size();
    auto const captureFirstInteraction = !firstInteraction;
    if (captureFirstInteraction)
      Kokkos::deep_copy(execution, firstInteractionCandidates, std::uint32_t{0});
    while (!queue.empty() && result.wavefronts_ < maximumWavefronts) {
      auto const currentCount = queue.size();
      if (result.wavefronts_ != 0 && currentCount < minimumResidentBatchSize) {
        result.belowMinimumBatchCheckpoint_ = true;
        break;
      }
      auto const candidateProjection =
          activeWorkspace.projectedCandidateCapacity(currentCount);
      if (!memoryGate.permits(activeWorkspace.deviceBytes(), candidateProjection,
                              "Kokkos photon final-state workspace growth")) {
        spillActiveFront();
        break;
      }
      activeWorkspace.ensureCandidateCapacity(currentCount, execution);
      auto const bucketingProjection =
          activeWorkspace.projectedBucketingCapacity(currentCount, execution);
      if (!memoryGate.permits(activeWorkspace.deviceBytes(), bucketingProjection,
                              "Kokkos photon bucketing workspace growth")) {
        spillActiveFront();
        break;
      }
      Kokkos::deep_copy(execution, error, std::uint32_t{0});
      auto const bucketed =
          bucket_wavefront(queue.current(), currentCount, activeWorkspace.bucketing_,
                           execution, openmpChunkSize);
      auto const current = bucketed.particles_.rawDeviceView();
      if (bucketed.sorted_) {
        ++result.wavefrontBucketingBatches_;
        result.wavefrontBucketingParticles_ += currentCount;
      } else {
        ++result.wavefrontBucketingSmallBatches_;
        result.wavefrontBucketingSmallParticles_ += currentCount;
      }
      enqueue_photon_front(physicsContext, current, queue.next().rawDeviceView(),
                           activeWorkspace, currentCount, capacity, nextHistoryId,
                           projectSteps, captureFirstInteraction, execution,
                           openmpChunkSize);

      // Scan totals, compacted interaction count and materialization error
      // are unmanaged views onto this single device POD.  Copy it only after
      // all producers in the execution stream have completed; this preserves
      // the original one-D2H/one-fence host checkpoint without launching a
      // one-thread packing kernel for every resident wavefront.
      Kokkos::deep_copy(execution, hostFrontControl, frontControlDevice);
      wait_resident_execution(execution, "download resident Kokkos photon front control",
                              cooperativeWait);
      auto const& frontControl = hostFrontControl();
      auto const& totals = frontControl.totals_.values_;
      if (frontControl.interactionCount_ > currentCount)
        throw std::runtime_error(
            "resident Kokkos photon interaction compaction overflow");
      auto const interactionCount =
          static_cast<std::size_t>(frontControl.interactionCount_);
      auto const selectionFallbackCount = totals[PhotonSelectionFallbackOffset];
      auto const transportFallbackCount = totals[PhotonTransportFallbackOffset];
      auto const finalStateFallbackCount = totals[PhotonFinalStateFallbackOffset];
      auto const totalFallbackCount =
          selectionFallbackCount + transportFallbackCount + finalStateFallbackCount;
      if (selectionFallbackCount > currentCount || totalFallbackCount > currentCount)
        throw std::runtime_error("resident Kokkos photon fallback accounting overflow");
      result.selectedInteractions_ +=
          currentCount - static_cast<std::size_t>(selectionFallbackCount);
      result.interactionBearingWavefronts_ += interactionCount != 0;
      if (totals[PhotonNextOffset] > capacity ||
          totals[PhotonChargedOffset] > chargedCapacity)
        throw std::runtime_error("resident Kokkos photon endpoint capacity exceeded");
      if (frontControl.materializationError_ != 0)
        throw std::runtime_error(
            "resident Kokkos photon final-state materialization failed");

      auto const stepCount = static_cast<std::size_t>(totals[PhotonStepOffset]);
      auto const recordCount = static_cast<std::size_t>(totals[PhotonRecordOffset]);
      if (stepCount > currentCount)
        throw std::runtime_error(
            "resident Kokkos photon packed step count exceeds source count");
      if (recordCount > interactionCount)
        throw std::runtime_error(
            "resident Kokkos photon packed record count exceeds interaction "
            "count");

      if (profileAccumulator != nullptr) {
        auto const deviceProfile = profileAccumulator->deviceView();
        Kokkos::parallel_reduce(
            "c8_kokkos_resident_photon_transport_statistics",
            policy(execution, 0, currentCount),
            ResidentPhotonTransportStatisticsProfileFunctor<TExecutionSpace>{
                {selections, steps, stepCount, callStatistics},
                profileProjection,
                deviceProfile});
        if (interactionCount != 0)
          Kokkos::parallel_reduce(
              "c8_kokkos_resident_photon_interaction_statistics",
              policy(execution, 0, interactionCount),
              ResidentPhotonInteractionStatisticsProfileFunctor<TExecutionSpace>{
                  {finalStates, callStatistics},
                  profileProjection,
                  deviceProfile,
                  steps,
                  stepCount,
                  records,
                  recordCount});
      } else {
        Kokkos::parallel_reduce("c8_kokkos_resident_photon_transport_statistics",
                                policy(execution, 0, currentCount),
                                ResidentPhotonTransportStatisticsFunctor<TExecutionSpace>{
                                    selections, steps, stepCount, callStatistics});
        if (interactionCount != 0)
          Kokkos::parallel_reduce(
              "c8_kokkos_resident_photon_interaction_statistics",
              policy(execution, 0, interactionCount),
              ResidentPhotonInteractionStatisticsFunctor<TExecutionSpace>{
                  finalStates, callStatistics});
      }

      auto const chargedCount = static_cast<std::size_t>(totals[PhotonChargedOffset]);
      auto const fallbackCount = static_cast<std::size_t>(totalFallbackCount);
      auto const observationCount =
          static_cast<std::size_t>(totals[PhotonObservationOffset]);
      auto const copyProjected = projectSteps && stepCount != 0;
      auto const copySteps = (captureDiagnosticInteractions ||
                              (!projectSteps && profileAccumulator == nullptr)) &&
                             stepCount != 0;
      auto const copyRecords = profileAccumulator == nullptr && recordCount != 0;
      bool copyCharged = pendingLeptons == nullptr && chargedCount != 0;
      // Match the native-CUDA resident-output contract: when the complete
      // profile ledger is accumulated on the execution space, final-state
      // records remain resident as well.  Returning them without their
      // matching transport records would make the host router account the
      // same vertex a second time and breaks the history ledger.
      if (pendingLeptons != nullptr) {
        if (!pendingLeptons->tryAppend(charged, chargedCount, maximumPendingParticles,
                                       execution, memoryGate,
                                       "Kokkos pending lepton queue growth"))
          copyCharged = chargedCount != 0;
      }

      activeWorkspace.ensureHostOutputCapacity(
          copyProjected ? stepCount : 0, copySteps ? stepCount : 0,
          copyRecords ? recordCount : 0, copyCharged ? chargedCount : 0, fallbackCount,
          observationCount);

      auto const activeSteps = std::make_pair<std::size_t>(0, copySteps ? stepCount : 0);
      auto const activeProjected =
          std::make_pair<std::size_t>(0, copyProjected ? stepCount : 0);
      auto const activeRecords =
          std::make_pair<std::size_t>(0, copyRecords ? recordCount : 0);
      auto const activeCharged =
          std::make_pair<std::size_t>(0, copyCharged ? chargedCount : 0);
      auto const activeFallbacks = std::make_pair<std::size_t>(0, fallbackCount);
      auto const activeObservations = std::make_pair<std::size_t>(0, observationCount);
      auto projectedSource = Kokkos::subview(projectedSteps, activeProjected);
      auto stepSource = Kokkos::subview(steps, activeSteps);
      auto recordSource = Kokkos::subview(records, activeRecords);
      auto chargedSource = Kokkos::subview(charged, activeCharged);
      auto fallbackSource = Kokkos::subview(fallbacks, activeFallbacks);
      auto observationSource = Kokkos::subview(observations, activeObservations);
      auto hostProjected =
          Kokkos::subview(activeWorkspace.hostProjectedSteps_, activeProjected);
      auto hostSteps = Kokkos::subview(activeWorkspace.hostSteps_, activeSteps);
      auto hostRecords = Kokkos::subview(activeWorkspace.hostRecords_, activeRecords);
      auto hostCharged = Kokkos::subview(activeWorkspace.hostCharged_, activeCharged);
      auto hostFallbacks =
          Kokkos::subview(activeWorkspace.hostFallbacks_, activeFallbacks);
      auto hostObservations =
          Kokkos::subview(activeWorkspace.hostObservations_, activeObservations);

      // Queue every host-visible output transfer first.  One fence then makes
      // all mirrors visible and also completes endpoint materialization in a
      // no-interaction wavefront.  This preserves the source-ordered scans
      // while avoiding one synchronization per output vector.
      if (copyProjected) Kokkos::deep_copy(execution, hostProjected, projectedSource);
      if (copySteps) Kokkos::deep_copy(execution, hostSteps, stepSource);
      if (copyRecords) Kokkos::deep_copy(execution, hostRecords, recordSource);
      if (copyCharged) Kokkos::deep_copy(execution, hostCharged, chargedSource);
      if (fallbackCount != 0) Kokkos::deep_copy(execution, hostFallbacks, fallbackSource);
      if (observationCount != 0)
        Kokkos::deep_copy(execution, hostObservations, observationSource);
      auto const hasHostOutputs = copyProjected || copySteps || copyRecords ||
                                  copyCharged || fallbackCount != 0 ||
                                  observationCount != 0;
      if (hasHostOutputs)
        wait_resident_execution(execution, "download resident Kokkos photon outputs",
                                cooperativeWait);

      if (copyProjected) appendHost(result.projectedStepRecords_, hostProjected);
      auto const firstCopiedStep = result.stepRecords_.size();
      if (copySteps) appendHost(result.stepRecords_, hostSteps);
      if (captureDiagnosticInteractions) {
        for (auto index = firstCopiedStep; index < result.stepRecords_.size(); ++index) {
          auto const& step = result.stepRecords_[index];
          if (step.limit_ == PhotonTransportLimit::Interaction)
            result.interactionRecords_.push_back(step.interaction_);
        }
      }
      if (copyRecords) appendHost(result.finalStateRecords_, hostRecords);
      if (copyCharged) {
        if (pendingLeptons != nullptr)
          appendHost(result.cpuSpillParticles_, hostCharged);
        else
          appendHost(result.electromagneticSecondaries_, hostCharged);
      }
      if (fallbackCount != 0) appendHost(result.fallbackEvents_, hostFallbacks);
      if (observationCount != 0) appendHost(result.observations_, hostObservations);
      result.transportRecords_ += totals[PhotonStepOffset];
      if (totals[PhotonChildOffset] >
          std::numeric_limits<std::uint64_t>::max() - nextHistoryId)
        throw std::overflow_error("resident Kokkos photon history ID overflow");
      nextHistoryId += totals[PhotonChildOffset];
      queue.commitNextAlreadySynchronized(totals[PhotonNextOffset]);
      result.wavefronts_++;
      result.peakResidentPhotons_ = std::max(result.peakResidentPhotons_, queue.size());
    }
    auto const hostCallStatistics = activeWorkspace.hostCallStatistics_;
    auto const hostCandidates = activeWorkspace.hostFirstInteractionCandidates_;
    auto const hostSnapshot = activeWorkspace.hostFirstSnapshots_;
    Kokkos::deep_copy(execution, hostCallStatistics, callStatistics);
    if (captureFirstInteraction) {
      Kokkos::deep_copy(execution, hostCandidates, firstInteractionCandidates);
      // Copying the single snapshot speculatively lets statistics, candidate
      // count and first-interaction state share one call-end synchronization.
      // It is read only when the candidate count confirms initialization.
      Kokkos::deep_copy(execution, hostSnapshot, firstSnapshots);
    }
    wait_resident_execution(execution, "download resident Kokkos photon call results",
                            cooperativeWait);
    queue.markExecutionSynchronized();
    queueExecutionGuard.release();
    result.interactionVertices_ += hostCallStatistics(PhotonCallTransportStatisticOffset +
                                                      PhotonTransportInteractionVertices);
    result.layerBoundaries_ += hostCallStatistics(PhotonCallTransportStatisticOffset +
                                                  PhotonTransportLayerBoundaries);
    result.particleCuts_ += hostCallStatistics(PhotonCallTransportStatisticOffset +
                                               PhotonTransportParticleCuts);
    result.nativeNewtonIterations_ += hostCallStatistics(
        PhotonCallTransportStatisticOffset + PhotonSelectionNativeNewtonIterations);
    result.nativeBisectionIterations_ += hostCallStatistics(
        PhotonCallTransportStatisticOffset + PhotonSelectionNativeBisectionIterations);
    result.nativeInverseFailures_ += hostCallStatistics(
        PhotonCallTransportStatisticOffset + PhotonSelectionNativeInverseFailures);
    result.lpmSuppressions_ += hostCallStatistics(PhotonCallInteractionStatisticOffset +
                                                  PhotonInteractionLpmSuppressions);
    auto& processStatistics = result.processStatistics_;
    processStatistics.gpuFinalStates_ += hostCallStatistics(
        PhotonCallInteractionStatisticOffset + PhotonInteractionGpuFinalStates);
    processStatistics.physicalSecondariesGenerated_ += hostCallStatistics(
        PhotonCallInteractionStatisticOffset + PhotonInteractionPhysicalSecondaries);
    processStatistics.photonPairFinalStates_ += hostCallStatistics(
        PhotonCallInteractionStatisticOffset + PhotonInteractionPairFinalStates);
    processStatistics.photonMuonPairFinalStates_ += hostCallStatistics(
        PhotonCallInteractionStatisticOffset + PhotonInteractionMuonPairFinalStates);
    processStatistics.comptonFinalStates_ += hostCallStatistics(
        PhotonCallInteractionStatisticOffset + PhotonInteractionComptonFinalStates);
    processStatistics.photoelectricFinalStates_ += hostCallStatistics(
        PhotonCallInteractionStatisticOffset + PhotonInteractionPhotoelectricFinalStates);
    processStatistics.photonPairLpmTrials_ += hostCallStatistics(
        PhotonCallInteractionStatisticOffset + PhotonInteractionLpmTrials);
    processStatistics.photonPairLpmSuppressions_ += hostCallStatistics(
        PhotonCallInteractionStatisticOffset + PhotonInteractionLpmSuppressions);
    processStatistics.thinningHillasVertices_ += hostCallStatistics(
        PhotonCallInteractionStatisticOffset + PhotonInteractionThinningHillasVertices);
    processStatistics.thinningStatisticalVertices_ +=
        hostCallStatistics(PhotonCallInteractionStatisticOffset +
                           PhotonInteractionThinningStatisticalVertices);
    processStatistics.thinningParticlesDiscarded_ +=
        hostCallStatistics(PhotonCallInteractionStatisticOffset +
                           PhotonInteractionThinningParticlesDiscarded);
    if (captureFirstInteraction) {
      if (hostCandidates(0) > 1) {
        throw std::runtime_error(
            "resident Kokkos photon transport produced more than one "
            "generation-zero interaction candidate");
      }
      if (hostCandidates(0) == 1) { firstInteraction = hostSnapshot(0); }
    }
    result.completed_ = queue.empty();
    if (!result.completed_)
      result.remainingPhotons_ = queue.download(execution, cooperativeWait);
    return result;
  }

} // namespace corsika::kokkos
