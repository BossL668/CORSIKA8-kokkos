/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#pragma once
#include <corsika/kokkos/em/execution/LeptonCascade.hpp>
// Frozen production control flow. Only types/fields are adapted to the receiving
// interfaces; the production loop is not called through the migrated wrapper.
namespace lepton_cascade_production {
  using namespace corsika::kokkos;
  template <class TDestination, class THostView>
  void append_lepton_host_copy(TDestination& destination, THostView const& host,
                               std::size_t const count) {
    if (count == 0) return;
    auto const old = destination.size();
    destination.resize(old + count);
    for (std::size_t index = 0; index < count; ++index)
      destination[old + index] = host(index);
  }

  template <class TExecutionSpace>
  void enqueue_lepton_accumulation(
      LeptonFrontWorkspace<TExecutionSpace> const& workspace,
      std::size_t const currentCount, ResidentLeptonFrontControl const& control,
      ProfileProjectionView const profileProjection,
      ProfileAccumulatorStorage<TExecutionSpace>* const profileAccumulator,
      radio::RadioAccumulator<TExecutionSpace>* const radioAccumulator,
      TExecutionSpace const& execution, std::size_t const openmpChunkSize) {
    [[maybe_unused]] auto const policy = [openmpChunkSize](TExecutionSpace const& ex,
                                                           std::size_t, std::size_t end) {
      return make_range_policy(ex, end, openmpChunkSize);
    };
    auto const& transports = workspace.transports_;
    auto const& counts = workspace.counts_;
    auto const& callStatistics = workspace.callStatistics_;
    auto const& vertices = workspace.vertices_;
    auto const& finalStates = workspace.finalStates_;
    auto const& steps = workspace.steps_;
    auto const& records = workspace.records_;
    auto const& totals = control.totals_.values_;
    auto const interactionCount = static_cast<std::size_t>(control.interactionCount_);
    auto const residentStepCount = static_cast<std::size_t>(totals[LeptonStepOffset]);
    auto const residentRecordCount = static_cast<std::size_t>(totals[LeptonRecordOffset]);
    if (profileAccumulator != nullptr) {
      auto const deviceProfile = profileAccumulator->deviceView();
      enqueue_lepton_profile_statistics(
          policy(execution, 0, currentCount),
          ResidentLeptonTransportStatisticsProfileFunctor<TExecutionSpace>{
              {transports, counts, callStatistics},
              profileProjection,
              deviceProfile,
              steps,
              residentStepCount},
          profileAccumulator->hostShards());
      if (interactionCount != 0)
        Kokkos::parallel_reduce(
            "c8_kokkos_resident_lepton_interaction_statistics",
            policy(execution, 0, interactionCount),
            ResidentLeptonInteractionStatisticsProfileFunctor<TExecutionSpace>{
                {vertices, finalStates, callStatistics},
                profileProjection,
                deviceProfile,
                steps,
                residentStepCount,
                records,
                residentRecordCount});
    } else {
      Kokkos::parallel_reduce("c8_kokkos_resident_lepton_transport_statistics",
                              policy(execution, 0, currentCount),
                              ResidentLeptonTransportStatisticsFunctor<TExecutionSpace>{
                                  transports, counts, callStatistics});
      if (interactionCount != 0)
        Kokkos::parallel_reduce(
            "c8_kokkos_resident_lepton_interaction_statistics",
            policy(execution, 0, interactionCount),
            ResidentLeptonInteractionStatisticsFunctor<TExecutionSpace>{
                vertices, finalStates, callStatistics});
    }
    if (radioAccumulator != nullptr)
      radioAccumulator->accumulateLeptonTracks(steps, totals[LeptonStepOffset],
                                               execution);
  }

  template <class TExecutionSpace>
  LeptonCascadeResult run_lepton_cascade(
      typename PhysicsContext<TExecutionSpace>::device_view_type const physicsContext,
      double const electronMassGeV, bool const airMoliereFastPath,
      std::vector<ParticleState> const& particles,
      std::uint64_t const firstSecondaryHistoryId, std::size_t const maximumWavefronts,
      std::uint64_t const secondaryHistoryIdLimitExclusive,
      std::size_t const minimumResidentBatchSize,
      std::optional<FirstInteractionRecord>& firstInteraction,
      bool const projectSteps = false, ProfileProjectionView const profileProjection = {},
      ProfileAccumulatorStorage<TExecutionSpace>* const profileAccumulator = nullptr,
      radio::RadioAccumulator<TExecutionSpace>* const radioAccumulator = nullptr,
      LeptonFrontWorkspace<TExecutionSpace>* const workspace = nullptr,
      PendingParticleQueue<TExecutionSpace>* const pendingLeptons = nullptr,
      PendingParticleQueue<TExecutionSpace>* const pendingPhotons = nullptr,
      std::size_t const pendingInputCount = 0,
      std::size_t const maximumPendingParticles = std::numeric_limits<std::size_t>::max(),
      TExecutionSpace const& execution = {},
      bool const captureDiagnosticInteractions = false,
      std::size_t const openmpChunkSize = 0,
      ResidentMemoryBudgetGate const memoryGate = {},
      std::size_t const reservedSourceLimit = 0,
      ResidentExecutionWait<TExecutionSpace>* const cooperativeWait = nullptr) {

    [[maybe_unused]] auto const policy = [openmpChunkSize](
                                             TExecutionSpace const& selectedExecution,
                                             std::size_t const begin,
                                             std::size_t const end) {
      if (begin != 0 || end < begin)
        throw std::invalid_argument("resident lepton RangePolicy requires a zero origin");
      return make_range_policy(selectedExecution, end, openmpChunkSize);
    };
    LeptonCascadeResult result{};
    if ((pendingLeptons == nullptr && pendingInputCount != 0) ||
        (pendingLeptons != nullptr && pendingInputCount > pendingLeptons->size()))
      throw std::invalid_argument(
          "resident Kokkos lepton pending-input count is invalid");
    auto const pendingInput = pendingInputCount;
    if (pendingInput > std::numeric_limits<std::size_t>::max() - particles.size())
      throw std::overflow_error("resident Kokkos lepton input size overflow");
    auto const totalInput = pendingInput + particles.size();
    result.inputParticles_ = totalInput;
    if (totalInput == 0) {
      result.completed_ = true;
      return result;
    }
    if (firstSecondaryHistoryId == 0 || maximumWavefronts == 0 ||
        secondaryHistoryIdLimitExclusive <= firstSecondaryHistoryId ||
        minimumResidentBatchSize == 0)
      throw std::invalid_argument("resident Kokkos lepton cascade requires valid limits");
    if (totalInput > std::numeric_limits<std::size_t>::max() / 3)
      throw std::length_error(
          "resident Kokkos lepton input exceeds addressable capacity");

    // A source can produce at most three charged children.  Keeping three
    // input fronts gives ordinary Epair growth room while retaining a strict,
    // lossless checkpoint if a later front exceeds this per-call workspace.
    auto const capacity = totalInput * 3;
    auto const photonCapacity = capacity > std::numeric_limits<std::size_t>::max() / 2
                                    ? std::numeric_limits<std::size_t>::max()
                                    : capacity * 2;
    LeptonFrontWorkspace<TExecutionSpace> localWorkspace;
    auto& activeWorkspace = workspace != nullptr ? *workspace : localWorkspace;
    auto spillUnstagedInput = [&]() {
      result.workspaceLimitCheckpoint_ = true;
      result.completed_ = true;
      result.cpuSpillParticles_.reserve(totalInput);
      if (pendingInput != 0) {
        auto prefix = pendingLeptons->downloadPrefix(pendingInput, execution);
        result.cpuSpillParticles_.insert(result.cpuSpillParticles_.end(),
                                         std::make_move_iterator(prefix.begin()),
                                         std::make_move_iterator(prefix.end()));
        pendingLeptons->consume(pendingInput);
      }
      result.cpuSpillParticles_.insert(result.cpuSpillParticles_.end(), particles.begin(),
                                       particles.end());
    };
    auto const queueProjection = activeWorkspace.projectedQueueCapacity(capacity);
    if (!memoryGate.permits(activeWorkspace.deviceBytes(), queueProjection,
                            "Kokkos lepton resident queue growth")) {
      spillUnstagedInput();
      return result;
    }
    activeWorkspace.ensureQueueCapacity(capacity, execution);
    auto const initialSourceProjection =
        activeWorkspace.projectedSourceCapacity(totalInput, capacity);
    if (!memoryGate.permits(activeWorkspace.deviceBytes(), initialSourceProjection,
                            "Kokkos lepton source workspace growth")) {
      spillUnstagedInput();
      return result;
    }
    activeWorkspace.ensureSourceCapacity(totalInput, capacity, execution);
    auto const& callStatistics = activeWorkspace.callStatistics_;
    Kokkos::deep_copy(execution, callStatistics, std::uint64_t{0});
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
      queue.upload(particles, pendingLeptons->view(), pendingLeptons->head(),
                   pendingInput, execution);
      pendingLeptons->consume(pendingInput);
    } else {
      queue.upload(particles, execution);
    }
    [[maybe_unused]] auto const& selections = activeWorkspace.selections_;
    [[maybe_unused]] auto const& transports = activeWorkspace.transports_;
    [[maybe_unused]] auto const& transportFallbacks = activeWorkspace.transportFallbacks_;
    auto const& vertices = activeWorkspace.vertices_;
    [[maybe_unused]] auto const& finalStates = activeWorkspace.finalStates_;
    [[maybe_unused]] auto const& counts = activeWorkspace.counts_;
    [[maybe_unused]] auto const& interactionFlags = activeWorkspace.interactionFlags_;
    [[maybe_unused]] auto const& transportStates = activeWorkspace.transportStates_;
    [[maybe_unused]] auto const& interactionSources = activeWorkspace.interactionSources_;
    [[maybe_unused]] auto const& offsets = activeWorkspace.offsets_;
    auto const& steps = activeWorkspace.steps_;
    auto const& projectedSteps = activeWorkspace.projectedSteps_;
    auto const& records = activeWorkspace.records_;
    auto const& photons = activeWorkspace.photons_;
    auto const& fallbacks = activeWorkspace.fallbacks_;
    auto const& observations = activeWorkspace.observations_;
    auto const& decays = activeWorkspace.decays_;
    auto const& firstSnapshots = activeWorkspace.firstSnapshots_;
    auto const& firstInteractionCandidates = activeWorkspace.firstInteractionCandidates_;
    auto const& error = activeWorkspace.error_;
    auto const interactionCountDevice = activeWorkspace.interactionCount_;
    auto const scanTotalsDevice = activeWorkspace.scanTotals_;
    auto const frontControlDevice = activeWorkspace.frontControl_;
    auto const hostFrontControl = activeWorkspace.hostFrontControl_;

    std::uint64_t nextHistoryId = firstSecondaryHistoryId;
    result.peakResidentLeptons_ = queue.size();
    auto const captureFirstInteraction = !firstInteraction.has_value();
    if (captureFirstInteraction)
      Kokkos::deep_copy(execution, firstInteractionCandidates, std::uint32_t{0});
    while (!queue.empty() && result.wavefronts_ < maximumWavefronts) {
      auto const currentCount = queue.size();
      if (reservedSourceLimit != 0 && currentCount > reservedSourceLimit) {
        // The 3x child queue can hold this front, but the budgeted source and
        // radio arenas cannot. Return it through the existing lossless
        // checkpoint for GPU re-batching, before drawing any random numbers.
        result.workspaceLimitCheckpoint_ = true;
        break;
      }
      if (result.wavefronts_ != 0 && currentCount < minimumResidentBatchSize) {
        result.belowMinimumBatchCheckpoint_ = true;
        break;
      }
      auto const remainingHistoryIds = secondaryHistoryIdLimitExclusive - nextHistoryId;
      if (currentCount > remainingHistoryIds / 3) {
        result.historyRangeExhausted_ = true;
        break;
      }
      auto const sourceProjection =
          activeWorkspace.projectedSourceCapacity(currentCount, capacity);
      if (!memoryGate.permits(activeWorkspace.deviceBytes(), sourceProjection,
                              "Kokkos lepton source workspace growth")) {
        spillActiveFront();
        break;
      }
      activeWorkspace.ensureSourceCapacity(currentCount, capacity, execution);
      auto const maximumFrontPhotons = std::min(
          photonCapacity, currentCount > std::numeric_limits<std::size_t>::max() / 2
                              ? std::numeric_limits<std::size_t>::max()
                              : currentCount * 2);
      // Allocate against the source-count upper bounds before the scans.  The
      // scan totals can consequently stay device resident through endpoint
      // materialization instead of fencing merely to size these Views.
      auto const outputProjection = activeWorkspace.projectedOutputCapacity(
          currentCount, currentCount, maximumFrontPhotons, currentCount, currentCount,
          currentCount, projectSteps, capacity, photonCapacity);
      if (!memoryGate.permits(activeWorkspace.deviceBytes(), outputProjection,
                              "Kokkos lepton output workspace growth")) {
        spillActiveFront();
        break;
      }
      activeWorkspace.ensureOutputCapacity(
          currentCount, currentCount, maximumFrontPhotons, currentCount, currentCount,
          currentCount, projectSteps, capacity, photonCapacity, execution);
      auto const bucketingProjection =
          activeWorkspace.projectedBucketingCapacity(currentCount, execution);
      if (!memoryGate.permits(activeWorkspace.deviceBytes(), bucketingProjection,
                              "Kokkos lepton bucketing workspace growth")) {
        spillActiveFront();
        break;
      }
      if (radioAccumulator != nullptr) {
        auto const radioProjection =
            radioAccumulator->projectedTrackCapacity(currentCount);
        if (!memoryGate.permits(radioAccumulator->deviceBytes(), radioProjection,
                                "Kokkos radio track workspace growth")) {
          spillActiveFront();
          break;
        }
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

      enqueue_lepton_front(physicsContext, electronMassGeV, airMoliereFastPath, current,
                           activeWorkspace, currentCount, capacity, photonCapacity,
                           nextHistoryId, secondaryHistoryIdLimitExclusive, projectSteps,
                           captureFirstInteraction, execution, openmpChunkSize);

      // The scans and final-state writers above target disjoint, aligned
      // members of front_control_device through unmanaged Views.  Preserve
      // the original single D2H/fence host checkpoint without launching a
      // one-thread packing kernel for every resident wavefront.
      Kokkos::deep_copy(execution, hostFrontControl, frontControlDevice);
      wait_resident_execution(execution, "download resident Kokkos lepton front control",
                              cooperativeWait);
      auto const& frontControl = hostFrontControl();
      auto const& totals = frontControl.totals_.values_;
      if (frontControl.interactionCount_ > currentCount)
        throw std::runtime_error(
            "resident Kokkos lepton interaction compaction overflow");
      auto const interactionCount =
          static_cast<std::size_t>(frontControl.interactionCount_);
      auto const selectionFallbackCount = totals[LeptonSelectionFallbackOffset];
      auto const transportFallbackCount = totals[LeptonTransportFallbackOffset];
      auto const vertexFallbackCount = totals[LeptonVertexFallbackOffset];
      auto const finalStateFallbackCount = totals[LeptonFinalStateFallbackOffset];
      auto const totalFallbackCount = selectionFallbackCount + transportFallbackCount +
                                      vertexFallbackCount + finalStateFallbackCount;
      auto const vertexInteractionCount = frontControl.vertexInteractionCount_;
      if (selectionFallbackCount > currentCount || totalFallbackCount > currentCount ||
          vertexFallbackCount > interactionCount ||
          vertexInteractionCount > interactionCount - vertexFallbackCount ||
          finalStateFallbackCount > vertexInteractionCount)
        throw std::runtime_error("resident Kokkos lepton stage accounting overflow");
      auto const vertexContinuationCount =
          interactionCount - vertexFallbackCount - vertexInteractionCount;
      if (totals[LeptonNextOffset] > capacity ||
          totals[LeptonPhotonOffset] > photonCapacity) {
        result.workspaceLimitCheckpoint_ = true;
        break;
      }
      if (totals[LeptonChildOffset] > secondaryHistoryIdLimitExclusive - nextHistoryId) {
        result.historyRangeExhausted_ = true;
        break;
      }
      if (frontControl.materializationError_ != 0)
        throw std::runtime_error(
            "resident Kokkos lepton final-state materialization failed");

      if (captureDiagnosticInteractions && interactionCount != 0) {
        auto const activeVertices =
            Kokkos::subview(vertices, std::make_pair<std::size_t>(0, interactionCount));
        auto hostVertices = Kokkos::create_mirror_view(activeVertices);
        Kokkos::deep_copy(execution, hostVertices, activeVertices);
        wait_resident_execution(execution,
                                "download resident Kokkos lepton interaction diagnostics",
                                cooperativeWait);
        for (std::size_t candidate = 0; candidate < interactionCount; ++candidate) {
          auto const& vertex = hostVertices(candidate);
          if (vertex.interactionFlag_ != 0)
            result.interactionRecords_.push_back(vertex.record_);
        }
      }

      result.selectedInteractions_ +=
          currentCount - static_cast<std::size_t>(selectionFallbackCount);
      result.interactionBearingWavefronts_ += interactionCount != 0;
      result.vertexInteractionsSelected_ += vertexInteractionCount;
      result.vertexNoInteractionContinuations_ += vertexContinuationCount;
      result.finalStateBearingWavefronts_ += vertexInteractionCount != 0;

      auto const residentStepCount = static_cast<std::size_t>(totals[LeptonStepOffset]);
      auto const residentRecordCount =
          static_cast<std::size_t>(totals[LeptonRecordOffset]);
      if (residentStepCount > currentCount)
        throw std::runtime_error(
            "resident Kokkos lepton packed step count exceeds source count");
      if (residentRecordCount > interactionCount)
        throw std::runtime_error(
            "resident Kokkos lepton packed record count exceeds interaction "
            "count");

      lepton_cascade_production::enqueue_lepton_accumulation(
          activeWorkspace, currentCount, frontControl, profileProjection,
          profileAccumulator, radioAccumulator, execution, openmpChunkSize);
      auto const projectedCount = static_cast<std::size_t>(
          projectSteps ? totals[LeptonStepOffset] : std::uint64_t{0});
      auto const stepCount = static_cast<std::size_t>(
          !projectSteps && profileAccumulator == nullptr ? totals[LeptonStepOffset]
                                                         : std::uint64_t{0});
      // Keep the same host/device ownership rule as native CUDA.  A
      // device-resident profile accumulator consumes both transport and
      // final-state records; neither half of that ledger may subsequently be
      // replayed by the host router.
      auto const recordCount = static_cast<std::size_t>(
          profileAccumulator == nullptr ? totals[LeptonRecordOffset] : std::uint64_t{0});
      bool photonsSpilledToCpu = false;
      if (pendingPhotons != nullptr) {
        if (!pendingPhotons->tryAppend(photons, totals[LeptonPhotonOffset],
                                       maximumPendingParticles, execution, memoryGate,
                                       "Kokkos pending photon queue growth"))
          photonsSpilledToCpu = true;
      }
      auto const photonCount = static_cast<std::size_t>(
          pendingPhotons == nullptr || photonsSpilledToCpu ? totals[LeptonPhotonOffset]
                                                           : std::uint64_t{0});
      auto const fallbackCount = static_cast<std::size_t>(totalFallbackCount);
      auto const observationCount =
          static_cast<std::size_t>(totals[LeptonObservationOffset]);
      auto const decayCount = static_cast<std::size_t>(totals[LeptonDecayOffset]);

      activeWorkspace.ensureHostOutputCapacity(
          projectedCount, stepCount, recordCount, photonCount, fallbackCount,
          observationCount, decayCount, capacity, photonCapacity);
      auto const activeProjected = std::make_pair<std::size_t>(0, projectedCount);
      auto const activeSteps = std::make_pair<std::size_t>(0, stepCount);
      auto const activeRecords = std::make_pair<std::size_t>(0, recordCount);
      auto const activePhotons = std::make_pair<std::size_t>(0, photonCount);
      auto const activeFallbacks = std::make_pair<std::size_t>(0, fallbackCount);
      auto const activeObservations = std::make_pair<std::size_t>(0, observationCount);
      auto const activeDecays = std::make_pair<std::size_t>(0, decayCount);
      auto const projectedSource = Kokkos::subview(projectedSteps, activeProjected);
      auto const stepSource = Kokkos::subview(steps, activeSteps);
      auto const recordSource = Kokkos::subview(records, activeRecords);
      auto const photonSource = Kokkos::subview(photons, activePhotons);
      auto const fallbackSource = Kokkos::subview(fallbacks, activeFallbacks);
      auto const observationSource = Kokkos::subview(observations, activeObservations);
      auto const decaySource = Kokkos::subview(decays, activeDecays);
      auto const hostProjectedSteps =
          Kokkos::subview(activeWorkspace.hostProjectedSteps_, activeProjected);
      auto const hostSteps = Kokkos::subview(activeWorkspace.hostSteps_, activeSteps);
      auto const hostRecords =
          Kokkos::subview(activeWorkspace.hostRecords_, activeRecords);
      auto const hostPhotons =
          Kokkos::subview(activeWorkspace.hostPhotons_, activePhotons);
      auto const hostFallbacks =
          Kokkos::subview(activeWorkspace.hostFallbacks_, activeFallbacks);
      auto const hostObservations =
          Kokkos::subview(activeWorkspace.hostObservations_, activeObservations);
      auto const hostDecays = Kokkos::subview(activeWorkspace.hostDecays_, activeDecays);

      if (projectedCount != 0)
        Kokkos::deep_copy(execution, hostProjectedSteps, projectedSource);
      if (stepCount != 0) Kokkos::deep_copy(execution, hostSteps, stepSource);
      if (recordCount != 0) Kokkos::deep_copy(execution, hostRecords, recordSource);
      if (photonCount != 0) Kokkos::deep_copy(execution, hostPhotons, photonSource);
      if (fallbackCount != 0) Kokkos::deep_copy(execution, hostFallbacks, fallbackSource);
      if (observationCount != 0)
        Kokkos::deep_copy(execution, hostObservations, observationSource);
      if (decayCount != 0) Kokkos::deep_copy(execution, hostDecays, decaySource);

      auto const hasHostOutputs =
          projectedCount != 0 || stepCount != 0 || recordCount != 0 || photonCount != 0 ||
          fallbackCount != 0 || observationCount != 0 || decayCount != 0;
      if (hasHostOutputs)
        wait_resident_execution(execution, "download resident Kokkos lepton host outputs",
                                cooperativeWait);

      append_lepton_host_copy(result.projectedStepRecords_, hostProjectedSteps,
                              projectedCount);
      append_lepton_host_copy(result.stepRecords_, hostSteps, stepCount);
      append_lepton_host_copy(result.finalStateRecords_, hostRecords, recordCount);
      if (photonsSpilledToCpu)
        append_lepton_host_copy(result.cpuSpillParticles_, hostPhotons, photonCount);
      else if (pendingPhotons == nullptr)
        append_lepton_host_copy(result.generatedPhotons_, hostPhotons, photonCount);
      append_lepton_host_copy(result.fallbackEvents_, hostFallbacks, fallbackCount);
      append_lepton_host_copy(result.observations_, hostObservations, observationCount);
      append_lepton_host_copy(result.decayCandidates_, hostDecays, decayCount);
      result.transportRecords_ += totals[LeptonStepOffset];
      nextHistoryId += totals[LeptonChildOffset];
      queue.commitNextAlreadySynchronized(totals[LeptonNextOffset]);
      result.wavefronts_++;
      result.peakResidentLeptons_ = std::max(result.peakResidentLeptons_, queue.size());
    }

    auto const hostCallStatistics = activeWorkspace.hostCallStatistics_;
    auto const hostFirstInteractionCandidates =
        activeWorkspace.hostFirstInteractionCandidates_;
    auto const hostFirstSnapshots = activeWorkspace.hostFirstSnapshots_;
    Kokkos::deep_copy(execution, hostCallStatistics, callStatistics);
    if (captureFirstInteraction) {
      Kokkos::deep_copy(execution, hostFirstInteractionCandidates,
                        firstInteractionCandidates);
      // Copy the single diagnostic slot together with the candidate count.
      // Its bytes are inspected only when the count proves it was written.
      Kokkos::deep_copy(execution, hostFirstSnapshots, firstSnapshots);
    }
    wait_resident_execution(execution, "download resident Kokkos lepton call diagnostics",
                            cooperativeWait);
    queue.markExecutionSynchronized();
    queueExecutionGuard.release();
    auto const transportStatistic = [&](std::size_t const index) {
      return hostCallStatistics(LeptonCallTransportStatisticBase + index);
    };
    auto const interactionStatistic = [&](std::size_t const index) {
      return hostCallStatistics(LeptonCallInteractionStatisticBase + index);
    };

    result.interactionVertices_ += transportStatistic(LeptonTransportInteractionVertices);
    result.lpmSuppressions_ += interactionStatistic(LeptonInteractionLpmSuppressions);
    auto& processStatistics = result.processStatistics_;
    processStatistics.gpuFinalStates_ +=
        interactionStatistic(LeptonInteractionGpuFinalStates);
    processStatistics.physicalSecondariesGenerated_ +=
        interactionStatistic(LeptonInteractionPhysicalSecondaries);
    processStatistics.bremsFinalStates_ +=
        interactionStatistic(LeptonInteractionBremsFinalStates);
    processStatistics.annihilationFinalStates_ +=
        interactionStatistic(LeptonInteractionAnnihilationFinalStates);
    processStatistics.ionizationFinalStates_ +=
        interactionStatistic(LeptonInteractionIonizationFinalStates);
    processStatistics.electronPairFinalStates_ +=
        interactionStatistic(LeptonInteractionElectronPairFinalStates);
    processStatistics.bremsLpmTrials_ +=
        interactionStatistic(LeptonInteractionBremsLpmTrials);
    processStatistics.bremsLpmSuppressions_ +=
        interactionStatistic(LeptonInteractionBremsLpmSuppressions);
    processStatistics.electronPairLpmTrials_ +=
        interactionStatistic(LeptonInteractionElectronPairLpmTrials);
    processStatistics.electronPairLpmSuppressions_ +=
        interactionStatistic(LeptonInteractionElectronPairLpmSuppressions);
    processStatistics.electronPairRejectionTrials_ +=
        interactionStatistic(LeptonInteractionElectronPairRejectionTrials);
    processStatistics.electronPairZeroWeightSamples_ +=
        interactionStatistic(LeptonInteractionElectronPairZeroWeightSamples);
    processStatistics.electronPairRejectionFallbacks_ +=
        interactionStatistic(LeptonInteractionElectronPairRejectionFallbacks);
    processStatistics.electronPairEnvelopeViolations_ +=
        interactionStatistic(LeptonInteractionElectronPairEnvelopeViolations);
    processStatistics.thinningHillasVertices_ +=
        interactionStatistic(LeptonInteractionThinningHillasVertices);
    processStatistics.thinningStatisticalVertices_ +=
        interactionStatistic(LeptonInteractionThinningStatisticalVertices);
    processStatistics.thinningParticlesDiscarded_ +=
        interactionStatistic(LeptonInteractionThinningParticlesDiscarded);
    processStatistics.moliereTrials_ += transportStatistic(LeptonTransportMoliereTrials);
    processStatistics.moliereDeflections_ +=
        transportStatistic(LeptonTransportMoliereDeflections);
    processStatistics.moliereZeroDeflections_ +=
        transportStatistic(LeptonTransportMoliereZeroDeflections);
    processStatistics.moliereNewtonIterations_ +=
        transportStatistic(LeptonTransportMoliereNewtonIterations);
    processStatistics.moliereMaxNewtonIterations_ =
        std::max(processStatistics.moliereMaxNewtonIterations_,
                 static_cast<std::uint32_t>(
                     transportStatistic(LeptonTransportMoliereMaximumNewtonIterations)));
    for (std::size_t limit = 0; limit < processStatistics.leptonTransportLimits_.size();
         ++limit)
      processStatistics.leptonTransportLimits_[limit] +=
          transportStatistic(LeptonTransportLimitBase + limit);

    result.secondaryHistoryIdsUsed_ = nextHistoryId - firstSecondaryHistoryId;
    result.completed_ = queue.empty();
    if (!result.completed_)
      result.remainingLeptons_ = queue.download(execution, cooperativeWait);
    if (captureFirstInteraction) {
      if (hostFirstInteractionCandidates(0) > 1) {
        throw std::runtime_error(
            "resident Kokkos lepton transport produced more than one "
            "generation-zero interaction candidate");
      }
      if (hostFirstInteractionCandidates(0) == 1)
        firstInteraction = hostFirstSnapshots(0);
    }
    return result;
  }
} // namespace lepton_cascade_production
