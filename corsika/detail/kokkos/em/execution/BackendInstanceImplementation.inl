/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <corsika/kokkos/em/execution/BackendInstance.hpp>
#include <corsika/kokkos/runtime/BackendRuntime.hpp>
#include <corsika/kokkos/runtime/TuningCache.hpp>
#include <corsika/kokkos/em/execution/InteractionSelectionBatch.hpp>
#include <corsika/kokkos/em/execution/PhotonTransportBatch.hpp>
#include <corsika/kokkos/em/execution/LeptonTransportBatch.hpp>
#include <corsika/kokkos/em/execution/LeptonVertexBatch.hpp>
#include <corsika/kokkos/em/execution/PhotonFinalStateBatch.hpp>
#include <corsika/kokkos/em/execution/LeptonFinalStateBatch.hpp>
#include <corsika/kokkos/environment/EnvironmentStorage.hpp>
#include <corsika/kokkos/environment/EnvironmentBorrowedView.hpp>
#include <corsika/kokkos/environment/TransportGeometryConfig.hpp>
#include <corsika/kokkos/em/execution/NativeTableStorage.hpp>
#include <corsika/kokkos/em/execution/MoliereInterpolationStorage.hpp>
#include <corsika/kokkos/output/ProfileProjectionStorage.hpp>
#include <corsika/kokkos/output/ProfileAccumulatorStorage.hpp>
#include <corsika/kokkos/radio/RadioAccumulator.hpp>
#include <corsika/kokkos/em/execution/PhysicsContext.hpp>
#include <corsika/kokkos/em/execution/PhotonCascade.hpp>
#include <corsika/kokkos/em/execution/LeptonCascade.hpp>
#include <corsika/kokkos/em/execution/ResidentCapacity.hpp>
#include <corsika/kokkos/em/proposal/ProposalNativeAux.hpp>
#include <corsika/kokkos/utility/Sha256.hpp>

namespace corsika::kokkos {

  template <class TExecutionSpace>
  class BackendInstanceImplementation final : public BackendInstance {
  public:
    explicit BackendInstanceImplementation(RuntimeConfig const&);
    ~BackendInstanceImplementation() override;
    void initialize(EnvironmentDataView const& environment,
                    TransportGeometryConfig const& geometry,
                    ProposalNativeTableSet const& table,
                    ProposalNativeAuxData const& auxiliary,
                    BackendConfig const& config) override;
    InteractionBatchResult selectInteractionsForValidation(
        std::vector<ParticleState> const& particles) override;
    PhotonTransportBatchResult transportPhotonsForValidation(
        std::vector<InteractionRecord> const& interactions) override;
    LeptonTransportBatchResult transportLeptonsForValidation(
        std::vector<InteractionRecord> const& interactions) override;
    LeptonVertexSelectionBatchResult selectLeptonVerticesForValidation(
        std::vector<InteractionRecord> const& candidates) override;
    PhotonFinalStateBatchResult generatePhotonFinalStatesForValidation(
        std::vector<InteractionRecord> const& interactions,
        std::uint64_t const firstSecondaryHistoryId) override;
    LeptonFinalStateBatchResult generateLeptonFinalStatesForValidation(
        std::vector<InteractionRecord> const& interactions,
        std::uint64_t const firstSecondaryHistoryId) override;
    radio::RadioWaveforms projectRadioForValidation(
        std::vector<LeptonTransportRecord> const& records) override;
    void beginShower(ShowerConfig const& shower) override;
    bool canTransport(ParticleState const& particle) const override;
    bool hasProposalTable() const override;
    std::size_t minimumBatchSize() const override;
    std::size_t maximumResidentPhotonBatchSize() const override;
    std::size_t maximumResidentLeptonBatchSize() const override;
    std::size_t maximumResidentInputBatchSize() const override;
    std::size_t pendingPhotonCount() const noexcept override;
    std::size_t pendingLeptonCount() const noexcept override;
    PhotonCascadeResult runPhotonWavefront(std::vector<ParticleState> const& particles,
                                           std::uint64_t firstSecondaryHistoryId,
                                           std::size_t maximumWavefronts,
                                           std::size_t minimumResidentBatchSize) override;
    LeptonCascadeResult runLeptonWavefront(std::vector<ParticleState> const& particles,
                                           std::uint64_t firstSecondaryHistoryId,
                                           std::size_t maximumWavefronts,
                                           std::uint64_t secondaryHistoryIdLimitExclusive,
                                           std::size_t minimumResidentBatchSize) override;
    BackendCapabilities capabilities() const override;
    BackendStatistics const& statistics() const override;
    bool gpuProfileEnabled() const noexcept override;
    ProfileResult downloadProfile() override;
    bool gpuRadioEnabled() const noexcept override;
    void enqueueRadioTracks(boost::span<LeptonTransportRecord const>) override;
    radio::RadioWaveforms downloadRadioWaveforms() override;
    std::optional<FirstInteractionRecord> downloadFirstInteractionSnapshot() override;
    RuntimeInfo const& runtimeInfo() const noexcept override;
    void setCooperativeProgress(std::function<bool()> progress) override {
      impl_->cooperativeWait_.setProgress(std::move(progress));
    }
    void setCooperativeBlockingWait(bool const enabled) override {
      impl_->requireInitialized();
      if (!impl_->runtime_.getInfo().cooperativeRuntime_ ||
          impl_->statistics_.particlesAdvanced_ != 0 ||
          // Fused resident transport uses these counters, not particles_advanced.
          // They also cover the standalone transport-oracle entry points.
          impl_->statistics_.photonTransportBatches_ != 0 ||
          impl_->statistics_.leptonTransportBatches_ != 0 ||
          !impl_->pendingPhotons_.empty() || !impl_->pendingLeptons_.empty())
        throw std::logic_error(
            "Blocking CUDA waits require an idle cooperative endpoint before transport");
      impl_->cooperativeWait_.setBlocking(enabled);
    }
    CooperativeCudaWaitStatistics cooperativeWaitStatistics() const override {
      auto const& wait = impl_->cooperativeWait_;
      return {wait.blockingEnabled(), wait.blockingCalls(), wait.blockingHostSeconds()};
    }
    void setCooperativePendingInputLimit(std::size_t count) override {
      impl_->cooperativePendingLimit_ = count;
    }
    void prepareCooperativeWorkspace() override {
      impl_->requireInitialized();
      bool const projected =
          impl_->profileProjection_.enabled() && !impl_->profileAccumulator_.enabled();
      reserve_resident_arenas(impl_->maximumResidentBatchSize_, impl_->photonWorkspace_,
                              impl_->leptonWorkspace_, impl_->radioAccumulator_,
                              projected, impl_->execution_);
      impl_->pendingPhotons_.ensureTailCapacity(impl_->maximumPendingParticles_,
                                                impl_->maximumPendingParticles_,
                                                impl_->execution_);
      impl_->pendingLeptons_.ensureTailCapacity(impl_->maximumPendingParticles_,
                                                impl_->maximumPendingParticles_,
                                                impl_->execution_);
      impl_->execution_.fence("prepare bounded cooperative endpoint workspace");
      impl_->refreshDeviceMemoryStatistics();
    }
    std::pair<std::size_t, std::size_t> enableCooperativeHostProfileShards() override {
      impl_->requireInitialized();
      if (!impl_->runtime_.getInfo().cooperativeRuntime_ ||
          !impl_->runtime_.getInfo().openmp_ || impl_->runtime_.getInfo().gpu_ ||
          impl_->statistics_.particlesAdvanced_ != 0)
        throw std::logic_error(
            "host profile shards require an idle cooperative OpenMP endpoint");
      auto& profile = impl_->profileAccumulator_;
      if (!profile.enabled()) return {};
      auto const retained = impl_->retainedDeviceBytes();
      auto const available =
          impl_->memoryBudgetBytes_ == 0
              ? HostProfileShards::MaximumBytes
              : impl_->memoryBudgetBytes_ - std::min(retained, impl_->memoryBudgetBytes_);
      profile.enableCooperativeHostShards(
          static_cast<std::size_t>(impl_->runtime_.getInfo().hostThreads_), available);
      impl_->refreshDeviceMemoryStatistics();
      return {profile.hostShards().count(), profile.hostShards().bytes()};
    }
    std::vector<ParticleState> takeCooperativePending(bool photons,
                                                      std::size_t count) override {
      auto& queue = photons ? impl_->pendingPhotons_ : impl_->pendingLeptons_;
      auto result = queue.downloadPrefix(count, impl_->execution_);
      queue.consume(count);
      return result;
    }
    FixedProfileSnapshot downloadFixedProfile(std::string const& identity,
                                              CooperativeEndpoint endpoint) override {
      auto result =
          impl_->profileAccumulator_.downloadFixed(identity, endpoint, impl_->execution_);
      auto decoded = decode_fixed_profile(result);
      impl_->statistics_.profile_.steps_ = decoded.steps_;
      impl_->statistics_.profile_.depositedSteps_ = decoded.depositedSteps_;
      impl_->statistics_.profile_.fixedPointOverflows_ = decoded.fixedPointOverflows_;
      impl_->statistics_.profile_.invalidRecords_ = decoded.invalidRecords_;
      return result;
    }
    radio::FixedRadioSnapshot downloadFixedRadio(std::string const& identity,
                                                 CooperativeEndpoint endpoint) override {
      auto result =
          impl_->radioAccumulator_.downloadFixed(identity, endpoint, impl_->execution_);
      impl_->statistics_.radio_ = impl_->radioAccumulator_.statistics();
      return result;
    }

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

  template <class TExecutionSpace>
  class BackendInstanceImplementation<TExecutionSpace>::Impl {
  public:
    explicit Impl(RuntimeConfig const& runtimeConfig)
        : runtime_{runtimeConfig} {}

    void requireInitialized() const {
      if (!initialized_) {
        throw std::logic_error("Kokkos EM backend is not initialized");
      }
    }

    ResidentExecutionWait<TExecutionSpace> cooperativeWait_;
    std::size_t cooperativePendingLimit_{std::numeric_limits<std::size_t>::max()};

    std::uint64_t interactionHashForPid(std::int32_t const pid) const {
      auto const found =
          std::find_if(interactionHashes_.begin(), interactionHashes_.end(),
                       [pid](auto const& entry) { return entry.first == pid; });
      return found == interactionHashes_.end() ? std::uint64_t{} : found->second;
    }

    void enrichFallbackEvents(std::vector<FallbackRequest>& events) const {
      for (auto& event : events) {
        event.mediumHash_ = proposalMediumHash_;
        event.interactionHash_ = interactionHashForPid(event.particle_.pid_);
      }
    }

    void refreshCrossSpeciesStorageStatistics() {
      statistics_.crossSpeciesQueueDeviceBytes_ =
          pendingPhotons_.deviceBytes() + pendingLeptons_.deviceBytes();
      statistics_.crossSpeciesQueueCapacityPerPid_ =
          config_.residentCrossSpecies_ ? maximumPendingParticles_ : std::size_t{};
      statistics_.peakPendingPhotons_ =
          std::max(statistics_.peakPendingPhotons_, pendingPhotons_.size());
      statistics_.peakPendingLeptons_ =
          std::max(statistics_.peakPendingLeptons_, pendingLeptons_.size());
    }

    struct StaticDeviceMemoryProjection {
      std::size_t proposalNativeTable_{};
      std::size_t moliereInterpolation_{};
      std::size_t profileProjection_{};
      std::size_t profileAccumulator_{};
      std::size_t radioAccumulator_{};
      std::size_t physicsContext_{};
      std::size_t environmentTree_{};
      MemoryProjection aggregate_{};
    };

    static StaticDeviceMemoryProjection projectStaticDeviceMemory(
        ProposalNativeTableSet const& table, BackendConfig const& config,
        EnvironmentDataView const& environment) {
      StaticDeviceMemoryProjection result{};
      result.proposalNativeTable_ =
          NativeTableStorage<TExecutionSpace>::projectedDeviceBytes(table);
      result.moliereInterpolation_ =
          MoliereInterpolationStorage<TExecutionSpace>::projectedDeviceBytes();
      result.profileProjection_ =
          ProfileProjectionStorage<TExecutionSpace>::projectedDeviceBytes(
              config.profileProjection_);
      result.profileAccumulator_ =
          ProfileAccumulatorStorage<TExecutionSpace>::projectedDeviceBytes(
              config.profileProjection_);
      result.radioAccumulator_ =
          radio::RadioAccumulator<TExecutionSpace>::projectedDeviceBytes(config.radio_);
      result.physicsContext_ = PhysicsContext<TExecutionSpace>::projectedDeviceBytes();
      result.environmentTree_ =
          std::size_t(environment.nodeCount_) * sizeof(EnvironmentNode) +
          std::size_t(environment.nodeCount_) * sizeof(EnvironmentShape) +
          (std::size_t(environment.childCount_) + environment.exclusionCount_) *
              sizeof(std::int32_t) +
          std::size_t(environment.modelCount_) * sizeof(EnvironmentModel) +
          std::size_t(environment.componentCount_) * sizeof(EnvironmentComponent);

      MemoryProjectionBuilder projection{0};
      projection.replace(0, result.proposalNativeTable_);
      projection.replace(0, result.moliereInterpolation_);
      projection.replace(0, result.profileProjection_);
      projection.replace(0, result.profileAccumulator_);
      projection.replace(0, result.radioAccumulator_);
      projection.replace(0, result.physicsContext_);
      projection.replace(0, result.environmentTree_);
      result.aggregate_ = projection.result();
      return result;
    }

    void requireStaticDeviceMemory(StaticDeviceMemoryProjection const& projection) const {
      if (memoryBudgetBytes_ == 0 ||
          projection.aggregate_.transientPeakBytes_ <= memoryBudgetBytes_)
        return;
      throw std::runtime_error(
          "Kokkos initialization preflight requires " +
          std::to_string(projection.aggregate_.transientPeakBytes_) +
          " bytes of device memory (proposal-native=" +
          std::to_string(projection.proposalNativeTable_) +
          ", Moliere=" + std::to_string(projection.moliereInterpolation_) +
          ", profile-projection=" + std::to_string(projection.profileProjection_) +
          ", profile-accumulator=" + std::to_string(projection.profileAccumulator_) +
          ", radio=" + std::to_string(projection.radioAccumulator_) +
          ", context=" + std::to_string(projection.physicsContext_) +
          ", environment-tree=" + std::to_string(projection.environmentTree_) +
          "), exceeding the configured " + std::to_string(memoryBudgetBytes_) +
          " byte device-memory budget before any static upload");
    }

    std::size_t retainedDeviceBytes() const {
      auto bytes = table_.getDeviceBytes();
      auto const append = [&bytes](std::size_t const amount) {
        bytes = checked_memory_add(bytes, amount);
      };
      append(moliereInterpolation_.getDeviceBytes());
      append(physicsContext_.getDeviceBytes());
      append(environmentTree_.getDeviceBytes());
      append(profileProjection_.deviceBytes());
      append(profileAccumulator_.deviceBytes());
      append(radioAccumulator_.deviceBytes());
      append(photonWorkspace_.deviceBytes());
      append(leptonWorkspace_.deviceBytes());
      append(pendingPhotons_.deviceBytes());
      append(pendingLeptons_.deviceBytes());
      return bytes;
    }

    bool permitsResidentGrowth(std::size_t const currentOwnerBytes,
                               MemoryProjection const& projection,
                               char const* const) const {
      if (memoryBudgetBytes_ == 0) return true;
      auto const retained = retainedDeviceBytes();
      if (currentOwnerBytes > retained)
        throw std::logic_error(
            "Kokkos resident-memory projection owner exceeds total storage");
      auto const other = retained - currentOwnerBytes;
      auto const retainedAfter = checked_memory_add(other, projection.retainedBytes_);
      auto const transientPeak =
          checked_memory_add(other, projection.transientPeakBytes_);
      return std::max(retainedAfter, transientPeak) <= memoryBudgetBytes_;
    }

    static bool residentMemoryGateCallback(void* context,
                                           std::size_t const currentOwnerBytes,
                                           MemoryProjection const& projection,
                                           char const* const stage) {
      if (context == nullptr)
        throw std::logic_error("Kokkos resident-memory gate has no backend context");
      return static_cast<Impl*>(context)->permitsResidentGrowth(currentOwnerBytes,
                                                                projection, stage);
    }

    ResidentMemoryBudgetGate residentMemoryGate() {
      return {this, &Impl::residentMemoryGateCallback};
    }

    void refreshDeviceMemoryStatistics() {
      refreshCrossSpeciesStorageStatistics();
      statistics_.physicalWorkspaceBytes_ =
          photonWorkspace_.deviceBytes() + leptonWorkspace_.deviceBytes();
      statistics_.profile_.deviceBytes_ =
          profileProjection_.deviceBytes() + profileAccumulator_.deviceBytes();
      statistics_.radio_.deviceBytes_ = radioAccumulator_.deviceBytes();
      auto const residentBytes = retainedDeviceBytes();
      statistics_.peakDeviceBytes_ =
          std::max(statistics_.peakDeviceBytes_, residentBytes);
      if (memoryBudgetBytes_ != 0 && residentBytes > memoryBudgetBytes_) {
        throw std::runtime_error(
            "Kokkos resident allocation requires " + std::to_string(residentBytes) +
            " bytes, exceeding the configured " + std::to_string(memoryBudgetBytes_) +
            " byte device-memory budget");
      }
    }

    BackendRuntime runtime_;
    RuntimeConfig runtimeConfig_{};
    TuningCacheResult tuning_{};
    TExecutionSpace execution_{};
    NativeTableStorage<TExecutionSpace> table_{};
    MoliereInterpolationStorage<TExecutionSpace> moliereInterpolation_{};
    ProfileProjectionStorage<TExecutionSpace> profileProjection_{};
    ProfileAccumulatorStorage<TExecutionSpace> profileAccumulator_{};
    radio::RadioAccumulator<TExecutionSpace> radioAccumulator_{};
    PhotonFrontWorkspace<TExecutionSpace> photonWorkspace_{};
    LeptonFrontWorkspace<TExecutionSpace> leptonWorkspace_{};
    PhysicsContext<TExecutionSpace> physicsContext_{};
    PendingParticleQueue<TExecutionSpace> pendingPhotons_{"c8_kokkos_pending_photons"};
    PendingParticleQueue<TExecutionSpace> pendingLeptons_{"c8_kokkos_pending_leptons"};
    ProposalNativeTableSet hostTable_{};
    ProposalNativeAuxData auxiliary_{};
    EnvironmentBorrowedView environment_{};
    TransportGeometryConfig geometry_{};
    EnvironmentStorage<TExecutionSpace> environmentTree_{};
    BackendConfig config_{};
    BackendStatistics statistics_{};
    InteractionPhysicsView physics_{};
    std::optional<FirstInteractionRecord> firstInteraction_{};
    std::uint64_t proposalMediumHash_{};
    std::vector<std::pair<std::int32_t, std::uint64_t>> interactionHashes_{};
    bool initialized_{};
    std::size_t maximumResidentBatchSize_{};
    std::size_t maximumPendingParticles_{};
    std::size_t memoryBudgetBytes_{};
    bool automaticGpuCapacity_{};
  };

  template <class TExecutionSpace>
  BackendInstanceImplementation<TExecutionSpace>::BackendInstanceImplementation(
      RuntimeConfig const& config)
      : impl_(std::make_unique<Impl>(config)) {
    impl_->runtimeConfig_ = config;
  }

  template <class TExecutionSpace>
  BackendInstanceImplementation<TExecutionSpace>::~BackendInstanceImplementation() =
      default;

  template <class TExecutionSpace>
  void BackendInstanceImplementation<TExecutionSpace>::initialize(
      EnvironmentDataView const& environment, TransportGeometryConfig const& geometry,
      ProposalNativeTableSet const& table, ProposalNativeAuxData const& auxiliary,
      BackendConfig const& config) {
    Kokkos::Timer initializationTimer;
    if (impl_->initialized_) {
      throw std::logic_error("Kokkos EM backend is already initialized");
    }
    if (config.physicsSource_ != PhysicsSource::ProposalNative) {
      throw std::invalid_argument(
          "Kokkos EM backend supports proposal-native physics only");
    }
    if (!(config.memoryFraction_ > 0.) || !(config.memoryFraction_ <= 1.))
      throw std::invalid_argument(
          "Kokkos device memory fraction must be in the interval (0, 1]");
    validate_proposal_native_table(table);
    if (calculate_proposal_native_hash(table) != table.contentHash_)
      throw std::invalid_argument(
          "PROPOSAL native-table content hash does not match its payload");
    impl_->environment_ = EnvironmentBorrowedView{environment};
    impl_->geometry_ = geometry;
    impl_->config_ = config;
    impl_->hostTable_ = table;
    impl_->proposalMediumHash_ = table.interactionIdentities_.empty()
                                     ? std::uint64_t{}
                                     : table.interactionIdentities_.front().mediumHash_;
    impl_->interactionHashes_.clear();
    impl_->interactionHashes_.reserve(table.interactionIdentities_.size());
    for (auto const& identity : table.interactionIdentities_)
      impl_->interactionHashes_.emplace_back(identity.pdgId_, identity.interactionHash_);
    auto const& runtime = impl_->runtime_.getInfo();
    if (runtime.gpu_) {
      if (runtime.deviceFreeMemoryBytesAtInitialization_ == 0)
        throw std::runtime_error(
            "Kokkos GPU backend cannot establish a device-memory budget");
      impl_->memoryBudgetBytes_ = static_cast<std::size_t>(
          config.memoryFraction_ *
          static_cast<double>(runtime.deviceFreeMemoryBytesAtInitialization_));
      if (impl_->memoryBudgetBytes_ == 0)
        throw std::runtime_error("Kokkos GPU device-memory budget is zero");
    }
    auto const staticMemory = Impl::projectStaticDeviceMemory(table, config, environment);
    impl_->requireStaticDeviceMemory(staticMemory);
    {
      impl_->environmentTree_ = EnvironmentStorage<TExecutionSpace>(environment);
      impl_->environment_ =
          EnvironmentBorrowedView{impl_->environmentTree_.getDataView()};
    }
    TuningKey const tuningKey{
        runtime.backend_,         runtime.deviceName_,      runtime.architecture_,
        runtime.driverVersion_,   runtime.runtimeVersion_,  runtime.kokkosVersion_,
        runtime.compilerVersion_, runtime.projectRevision_, to_hex(table.contentHash_),
        runtime.device_,          runtime.hostThreads_};
    impl_->tuning_ = load_tuning_cache(impl_->runtimeConfig_.tuningCache_, tuningKey,
                                       impl_->runtimeConfig_.requireTuning_);
    if (!impl_->tuning_.matched_) {
      reportUnmatchedTuningCache(runtime.backend_);
      impl_->tuning_.parameters_.batchSize_ = config.minBatchSize_;
      impl_->tuning_.parameters_.chunkSize_ = runtime.openmp_ ? 16 : 1;
      impl_->tuning_.parameters_.teamSize_ = runtime.openmp_ ? 1 : 256;
      impl_->tuning_.parameters_.trackTileSize_ = runtime.openmp_ ? 4 : 8;
      impl_->tuning_.parameters_.observerTileSize_ = runtime.openmp_ ? 4 : 32;
      impl_->tuning_.parameters_.deviceQueues_ = 1;
    }
    impl_->auxiliary_ = auxiliary;
    impl_->table_.initialize(table);
    auto moliereCachePath = auxiliary.cacheFile_;
    if (!moliereCachePath.empty()) moliereCachePath += ".moliere-initial-v1.c8cache";
    impl_->moliereInterpolation_.initialize(auxiliary.electronMoliere_, moliereCachePath);
    impl_->profileProjection_.initialize(config.profileProjection_, impl_->execution_);
    impl_->profileAccumulator_.initialize(config.profileProjection_, impl_->execution_);
    impl_->radioAccumulator_.initialize(config.radio_, impl_->tuning_.parameters_,
                                        impl_->execution_);
    impl_->physics_ = InteractionPhysicsView{
        ContinuousPhysicsView{impl_->table_.getDeviceView(), config.emTransportCutMeV_,
                              config.muonTransportCutMeV_},
        1u};
    impl_->physicsContext_.initialize(
        impl_->physics_, impl_->environment_, impl_->geometry_,
        impl_->auxiliary_.electronMoliere_, impl_->auxiliary_.muonMoliere_,
        impl_->moliereInterpolation_.getDeviceView(),
        impl_->auxiliary_.hasMuonMoliere_ != 0, impl_->auxiliary_.photonPairLpm_,
        impl_->auxiliary_.bremsLpm_, impl_->config_.thinning_,
        impl_->profileProjection_.deviceView(), impl_->config_.randomSeed_,
        impl_->config_.showerId_, impl_->execution_);
    impl_->statistics_.showerOrdinal_ = 1;
    impl_->statistics_.acceleratorBackend_ = runtime.backend_;
    impl_->statistics_.acceleratorDeviceName_ = runtime.deviceName_;
    impl_->statistics_.acceleratorArchitecture_ = runtime.architecture_;
    impl_->statistics_.acceleratorDriverVersion_ = runtime.driverVersion_;
    impl_->statistics_.acceleratorRuntimeVersion_ = runtime.runtimeVersion_;
    impl_->statistics_.acceleratorCompilerVersion_ = runtime.compilerVersion_;
    impl_->statistics_.acceleratorProjectRevision_ = runtime.projectRevision_;
    impl_->statistics_.acceleratorDevice_ = runtime.device_;
    impl_->statistics_.acceleratorConcurrency_ = runtime.concurrency_;
    impl_->statistics_.acceleratorHostThreads_ = runtime.hostThreads_;
    impl_->statistics_.acceleratorOpenmp_ = runtime.openmp_;
    impl_->statistics_.acceleratorGpu_ = runtime.gpu_;
    impl_->statistics_.tuningCacheMatched_ = impl_->tuning_.matched_;
    impl_->statistics_.tuningCacheRequired_ = impl_->runtimeConfig_.requireTuning_;
    impl_->statistics_.tuningCacheHash_ = impl_->tuning_.contentHash_;
    impl_->statistics_.tuningBatchSize_ = impl_->tuning_.parameters_.batchSize_;
    impl_->statistics_.tuningChunkSize_ = impl_->tuning_.parameters_.chunkSize_;
    impl_->statistics_.tuningTeamSize_ = impl_->tuning_.parameters_.teamSize_;
    impl_->statistics_.tuningTrackTileSize_ = impl_->tuning_.parameters_.trackTileSize_;
    impl_->statistics_.tuningObserverTileSize_ =
        impl_->tuning_.parameters_.observerTileSize_;
    impl_->statistics_.tuningDeviceQueues_ = impl_->tuning_.parameters_.deviceQueues_;
    impl_->statistics_.physicsSource_ = PhysicsSource::ProposalNative;
    impl_->statistics_.nativeProposalVersion_ = table.proposalVersion_;
    impl_->statistics_.nativeCubicInterpolationVersion_ =
        table.cubicInterpolationVersion_;
    impl_->statistics_.nativeTableHash_ = table.contentHash_;
    impl_->statistics_.auxiliaryCacheHash_ = auxiliary.contentHash_;
    impl_->statistics_.nativeTableNodes_ =
        table.bicubicValues_.size() + table.cubicValues_.size();
    impl_->statistics_.proposalCacheTableCount_ = table.proposalCacheTableCount_;
    impl_->statistics_.proposalCacheHitCount_ = table.proposalCacheHitCount_;
    impl_->statistics_.proposalCacheAllHit_ = table.proposalCacheAllHit_;
    impl_->statistics_.nativeTableDeviceBytes_ = impl_->table_.getDeviceBytes();
    impl_->statistics_.tableDeviceBytes_ =
        impl_->table_.getDeviceBytes() + impl_->moliereInterpolation_.getDeviceBytes();
    impl_->statistics_.profile_.bins_ = config.profileProjection_.outputBinCount_;
    impl_->statistics_.profile_.deviceBytes_ = impl_->profileProjection_.deviceBytes() +
                                               impl_->profileAccumulator_.deviceBytes();
    impl_->statistics_.profile_.enabled_ = impl_->profileAccumulator_.enabled();
    impl_->statistics_.profile_.deterministic_ = true;
    impl_->statistics_.radio_ = impl_->radioAccumulator_.statistics();
    impl_->statistics_.auxiliaryCacheHit_ = auxiliary.cacheHit_;
    impl_->statistics_.reusedForShower_ = false;
    // OpenMP retains its separately validated capacity policy. GPU automatic
    // capacity is derived below from both arenas and the byte budget; an
    // explicit replay checkpoint keeps the original grow-only behavior.
    impl_->automaticGpuCapacity_ = runtime.gpu_ && config.residentBatchLimit_ == 0;
    // The tuning batch is an execution/workspace sizing hint.  It must not
    // replace the router's minimum resident-wavefront checkpoint: changing
    // that checkpoint changes when histories return to the host stack and can
    // therefore change history-id allocation and the Philox stream.
    if (impl_->config_.residentBatchLimit_ != 0) {
      if (impl_->config_.residentBatchLimit_ < impl_->config_.minBatchSize_) {
        throw std::invalid_argument(
            "configured Kokkos resident batch limit is smaller than the "
            "minimum execution batch");
      }
      impl_->maximumResidentBatchSize_ = impl_->config_.residentBatchLimit_;
    } else if (!runtime.gpu_) {
      impl_->maximumResidentBatchSize_ =
          std::max({std::size_t{65536}, impl_->config_.minBatchSize_,
                    impl_->tuning_.parameters_.batchSize_});
    }
    impl_->statistics_.maximumResidentPhotonBatch_ = impl_->maximumResidentBatchSize_;
    impl_->statistics_.maximumResidentLeptonBatch_ = impl_->maximumResidentBatchSize_;
    // Match native CUDA's conservative policy: at the default 70% device
    // budget, at most 512 MiB in total is devoted to the two cross-species
    // queues.  Smaller configured fractions scale that allowance down;
    // larger fractions leave the cap unchanged so transport workspaces retain
    // headroom.
    constexpr auto MaximumCrossSpeciesBytes = std::size_t{512} * 1024 * 1024;
    auto const scaledCrossSpeciesBytes =
        static_cast<std::size_t>(static_cast<double>(MaximumCrossSpeciesBytes) *
                                 std::min(1.0, config.memoryFraction_ / 0.70));
    impl_->maximumPendingParticles_ =
        config.residentCrossSpecies_
            ? std::max(config.minBatchSize_,
                       scaledCrossSpeciesBytes / (2 * sizeof(ParticleState)))
            : std::size_t{};
    if (impl_->automaticGpuCapacity_) {
      // Reserve at most 1/8 of the budget (512 MiB maximum) for two pending
      // queues, and account for replacement of one full queue during compact.
      impl_->maximumPendingParticles_ =
          config.residentCrossSpecies_
              ? std::min(MaximumCrossSpeciesBytes, impl_->memoryBudgetBytes_ / 8) /
                    (2 * sizeof(ParticleState))
              : 0;
      bool const projectSteps =
          impl_->profileProjection_.enabled() && !impl_->profileAccumulator_.enabled();
      auto const retained = impl_->retainedDeviceBytes();
      auto const plan = select_resident_capacity(
          impl_->memoryBudgetBytes_, config.minBatchSize_,
          impl_->maximumPendingParticles_, [&](std::size_t count) {
            return project_resident_arenas(
                count, retained, impl_->photonWorkspace_, impl_->leptonWorkspace_,
                impl_->radioAccumulator_, projectSteps, impl_->execution_);
          });
      impl_->maximumResidentBatchSize_ = plan.inputParticles_;
      // Pre-reserve once, after the complete projection passed. There is no
      // per-particle allocation and no allocation-driven RNG consumption.
      reserve_resident_arenas(plan.inputParticles_, impl_->photonWorkspace_,
                              impl_->leptonWorkspace_, impl_->radioAccumulator_,
                              projectSteps, impl_->execution_);
      impl_->pendingPhotons_.ensureTailCapacity(plan.pendingParticlesPerSpecies_,
                                                plan.pendingParticlesPerSpecies_,
                                                impl_->execution_);
      impl_->pendingLeptons_.ensureTailCapacity(plan.pendingParticlesPerSpecies_,
                                                plan.pendingParticlesPerSpecies_,
                                                impl_->execution_);
      if (impl_->retainedDeviceBytes() != plan.retainedBytes_)
        throw std::logic_error(
            "Kokkos automatic resident-capacity projection disagrees with allocation");
      impl_->statistics_.automaticCapacityBudgetBytes_ = impl_->memoryBudgetBytes_;
      impl_->statistics_.automaticCapacityPlannedBytes_ = plan.retainedBytes_;
      impl_->statistics_.automaticCapacityPeakBytes_ = plan.transientPeakBytes_;
      impl_->statistics_.automaticCapacityAllocatorReserveBytes_ =
          plan.allocatorReserveBytes_;
      impl_->statistics_.maximumResidentPhotonBatch_ = plan.inputParticles_;
      impl_->statistics_.maximumResidentLeptonBatch_ = plan.inputParticles_;
    }
    impl_->statistics_.crossSpeciesQueueCapacityPerPid_ = impl_->maximumPendingParticles_;
    if (runtime.openmp_ && !runtime.gpu_ && !runtime.cooperativeRuntime_ &&
        impl_->profileAccumulator_.enabled()) {
      // Bounded, host-only copies of the integer profile ledger, not particle
      // queues or physics tables. The resident allocation gate accounts for
      // these bytes during subsequent workspace growth. Cooperative endpoints
      // retain their existing explicit opt-in; GPU paths are unchanged.
      auto const retained = impl_->retainedDeviceBytes();
      auto const available =
          impl_->memoryBudgetBytes_ == 0
              ? HostProfileShards::MaximumBytes
              : impl_->memoryBudgetBytes_ - std::min(retained, impl_->memoryBudgetBytes_);
      impl_->profileAccumulator_.enableHostShards(
          static_cast<std::size_t>(runtime.hostThreads_), available);
    }
    impl_->refreshDeviceMemoryStatistics();
    // Complete asynchronous initialization before stopping the lifecycle
    // timer, otherwise the first shower would inherit part of the setup cost.
    impl_->execution_.fence("complete Kokkos EM backend initialization");
    if (runtime.gpu_) {
      using ::corsika::kokkos::checked_memory_add;
      auto staticUploadBytes = impl_->table_.getDeviceBytes();
      staticUploadBytes = checked_memory_add(
          staticUploadBytes, impl_->moliereInterpolation_.getDeviceBytes());
      staticUploadBytes =
          checked_memory_add(staticUploadBytes, impl_->profileProjection_.deviceBytes());
      staticUploadBytes = checked_memory_add(
          staticUploadBytes, impl_->radioAccumulator_.staticHostToDeviceBytes());
      staticUploadBytes =
          checked_memory_add(staticUploadBytes, impl_->physicsContext_.getDeviceBytes());
      staticUploadBytes =
          checked_memory_add(staticUploadBytes, impl_->environmentTree_.getDeviceBytes());
      impl_->statistics_.staticHostToDeviceBytes_ = staticUploadBytes;
    }
    impl_->statistics_.oneTimeInitializationMs_ = 1000. * initializationTimer.seconds();
    impl_->initialized_ = true;
  }

  template <class TExecutionSpace>
  InteractionBatchResult
  BackendInstanceImplementation<TExecutionSpace>::selectInteractionsForValidation(
      std::vector<ParticleState> const& particles) {
    impl_->requireInitialized();
    auto selected =
        select_interactions_batch(impl_->physics_, particles, impl_->config_.randomSeed_,
                                  impl_->config_.showerId_, impl_->execution_);
    impl_->statistics_.interactionSelectionBatches_++;
    impl_->statistics_.interactionsSelected_ += selected.batch_.interactions_.size();
    impl_->statistics_.proposalFallbacks_ += selected.batch_.fallbackEvents_.size();
    impl_->statistics_.nativeNewtonIterations_ += selected.nativeNewtonIterations_;
    impl_->statistics_.nativeBisectionIterations_ += selected.nativeBisectionIterations_;
    impl_->statistics_.nativeInverseFailures_ += selected.nativeInverseFailures_;
    impl_->enrichFallbackEvents(selected.batch_.fallbackEvents_);
    return std::move(selected.batch_);
  }

  template <class TExecutionSpace>
  PhotonTransportBatchResult
  BackendInstanceImplementation<TExecutionSpace>::transportPhotonsForValidation(
      std::vector<InteractionRecord> const& interactions) {
    impl_->requireInitialized();
    auto result = transport_photons_batch(impl_->environment_, impl_->geometry_,
                                          interactions, impl_->execution_);
    impl_->statistics_.photonTransportBatches_++;
    impl_->statistics_.proposalFallbacks_ += result.fallbackEvents_.size();
    for (auto const& record : result.records_) {
      switch (record.limit_) {
        case PhotonTransportLimit::Interaction:
          impl_->statistics_.photonTransportInteractions_++;
          break;
        case PhotonTransportLimit::LayerBoundary:
        case PhotonTransportLimit::MaterialBoundary:
          impl_->statistics_.photonTransportBoundaries_++;
          break;
        case PhotonTransportLimit::ObservationSurface:
          impl_->statistics_.photonTransportObservations_++;
          break;
        case PhotonTransportLimit::EscapedEnvironment:
          impl_->statistics_.photonTransportEscapes_++;
          break;
        case PhotonTransportLimit::ParticleCut:
          impl_->statistics_.photonTransportCuts_++;
          break;
      }
    }
    impl_->enrichFallbackEvents(result.fallbackEvents_);
    return result;
  }

  template <class TExecutionSpace>
  LeptonTransportBatchResult
  BackendInstanceImplementation<TExecutionSpace>::transportLeptonsForValidation(
      std::vector<InteractionRecord> const& interactions) {
    impl_->requireInitialized();
    auto result = transport_leptons_batch(
        impl_->physics_, impl_->environment_, impl_->geometry_,
        impl_->auxiliary_.electronMoliere_, impl_->auxiliary_.muonMoliere_,
        impl_->moliereInterpolation_.getDeviceView(),
        impl_->auxiliary_.hasMuonMoliere_ != 0, impl_->config_.randomSeed_,
        impl_->config_.showerId_, interactions, impl_->execution_);
    impl_->statistics_.leptonTransportBatches_++;
    impl_->statistics_.proposalFallbacks_ += result.fallbackEvents_.size();
    impl_->enrichFallbackEvents(result.fallbackEvents_);
    return result;
  }

  template <class TExecutionSpace>
  LeptonVertexSelectionBatchResult
  BackendInstanceImplementation<TExecutionSpace>::selectLeptonVerticesForValidation(
      std::vector<InteractionRecord> const& candidates) {
    impl_->requireInitialized();
    auto result = select_lepton_vertices_batch(
        impl_->physics_, candidates, impl_->config_.randomSeed_, impl_->config_.showerId_,
        impl_->execution_);
    impl_->statistics_.leptonVertexSelectionBatches_++;
    impl_->statistics_.proposalFallbacks_ += result.batch_.fallbackEvents_.size();
    impl_->statistics_.nativeNewtonIterations_ += result.nativeNewtonIterations_;
    impl_->statistics_.nativeBisectionIterations_ += result.nativeBisectionIterations_;
    impl_->statistics_.nativeInverseFailures_ += result.nativeInverseFailures_;
    impl_->enrichFallbackEvents(result.batch_.fallbackEvents_);
    return std::move(result.batch_);
  }

  template <class TExecutionSpace>
  PhotonFinalStateBatchResult
  BackendInstanceImplementation<TExecutionSpace>::generatePhotonFinalStatesForValidation(
      std::vector<InteractionRecord> const& interactions,
      std::uint64_t const firstSecondaryHistoryId) {
    impl_->requireInitialized();
    auto result = generate_photon_final_states_batch(
        impl_->physics_, impl_->auxiliary_.photonPairLpm_, impl_->config_.thinning_,
        interactions, impl_->config_.randomSeed_, impl_->config_.showerId_,
        firstSecondaryHistoryId, impl_->execution_);
    impl_->statistics_.finalStateBatches_++;
    impl_->statistics_.gpuFinalStates_ += result.batch_.gpuInteractions_;
    impl_->statistics_.physicalSecondariesGenerated_ += result.batch_.secondaries_.size();
    impl_->statistics_.photonPairFinalStates_ += result.batch_.photonPairInteractions_;
    impl_->statistics_.photonMuonPairFinalStates_ +=
        result.batch_.photonMuonPairInteractions_;
    impl_->statistics_.comptonFinalStates_ += result.batch_.comptonInteractions_;
    impl_->statistics_.photoelectricFinalStates_ +=
        result.batch_.photoelectricInteractions_;
    impl_->statistics_.proposalFallbacks_ += result.batch_.fallbackEvents_.size();
    if (result.firstInteraction_) { impl_->firstInteraction_ = result.firstInteraction_; }
    impl_->enrichFallbackEvents(result.batch_.fallbackEvents_);
    return std::move(result.batch_);
  }

  template <class TExecutionSpace>
  LeptonFinalStateBatchResult
  BackendInstanceImplementation<TExecutionSpace>::generateLeptonFinalStatesForValidation(
      std::vector<InteractionRecord> const& interactions,
      std::uint64_t const firstSecondaryHistoryId) {
    impl_->requireInitialized();
    auto result = generate_lepton_final_states_batch(
        impl_->auxiliary_.bremsLpm_, impl_->config_.thinning_, interactions,
        impl_->config_.randomSeed_, impl_->config_.showerId_, firstSecondaryHistoryId,
        impl_->execution_);
    impl_->statistics_.finalStateBatches_++;
    impl_->statistics_.gpuFinalStates_ += result.batch_.gpuInteractions_;
    impl_->statistics_.physicalSecondariesGenerated_ += result.batch_.secondaries_.size();
    impl_->statistics_.bremsFinalStates_ += result.batch_.bremsInteractions_;
    impl_->statistics_.annihilationFinalStates_ +=
        result.batch_.annihilationInteractions_;
    impl_->statistics_.ionizationFinalStates_ += result.batch_.ionizationInteractions_;
    impl_->statistics_.electronPairFinalStates_ +=
        result.batch_.electronPairInteractions_;
    impl_->statistics_.bremsLpmTrials_ += result.batch_.bremsLpmTrials_;
    impl_->statistics_.bremsLpmSuppressions_ += result.batch_.bremsLpmSuppressions_;
    impl_->statistics_.electronPairLpmTrials_ += result.batch_.electronPairLpmTrials_;
    impl_->statistics_.electronPairLpmSuppressions_ +=
        result.batch_.electronPairLpmSuppressions_;
    impl_->statistics_.electronPairRejectionTrials_ +=
        result.batch_.electronPairRejectionTrials_;
    impl_->statistics_.electronPairZeroWeightSamples_ +=
        result.batch_.electronPairZeroWeightSamples_;
    impl_->statistics_.electronPairRejectionFallbacks_ +=
        result.batch_.electronPairRejectionFallbacks_;
    impl_->statistics_.electronPairEnvelopeViolations_ +=
        result.batch_.electronPairEnvelopeViolations_;
    impl_->statistics_.proposalFallbacks_ += result.batch_.fallbackEvents_.size();
    if (result.firstInteraction_) impl_->firstInteraction_ = result.firstInteraction_;
    impl_->enrichFallbackEvents(result.batch_.fallbackEvents_);
    return std::move(result.batch_);
  }

  template <class TExecutionSpace>
  radio::RadioWaveforms
  BackendInstanceImplementation<TExecutionSpace>::projectRadioForValidation(
      std::vector<LeptonTransportRecord> const& records) {
    impl_->requireInitialized();
    if (!impl_->radioAccumulator_.enabled())
      throw std::logic_error("Kokkos radio validation requested while radio is disabled");
    using RecordView =
        Kokkos::View<LeptonTransportRecord*, typename TExecutionSpace::memory_space>;
    RecordView deviceRecords("c8_kokkos_radio_validation_records", records.size());
    auto mirror = Kokkos::create_mirror_view(deviceRecords);
    for (std::size_t index = 0; index < records.size(); ++index)
      mirror(index) = records[index];
    Kokkos::deep_copy(impl_->execution_, deviceRecords, mirror);
    impl_->radioAccumulator_.reset(impl_->execution_);
    impl_->radioAccumulator_.accumulateLeptonTracks(deviceRecords, records.size(),
                                                    impl_->execution_);
    auto result = impl_->radioAccumulator_.download(impl_->execution_);
    impl_->statistics_.radio_ = impl_->radioAccumulator_.statistics();
    impl_->refreshDeviceMemoryStatistics();
    return result;
  }

  template <class TExecutionSpace>
  void BackendInstanceImplementation<TExecutionSpace>::beginShower(
      ShowerConfig const& shower) {
    impl_->requireInitialized();
    if (!impl_->pendingPhotons_.empty() || !impl_->pendingLeptons_.empty())
      throw std::logic_error(
          "cannot begin a new Kokkos shower with pending cross-species particles");
    impl_->cooperativeWait_.resetStatistics();
    auto const staticStatistics = impl_->statistics_;
    impl_->config_.randomSeed_ = shower.randomSeed_;
    impl_->config_.showerId_ = shower.showerId_;
    impl_->config_.thinning_ = shower.thinning_;
    impl_->physicsContext_.beginShower(shower.thinning_, shower.randomSeed_,
                                       shower.showerId_, impl_->execution_);
    auto const ordinal = impl_->statistics_.showerOrdinal_ + 1;
    auto const staticBytes = impl_->statistics_.nativeTableDeviceBytes_;
    auto const allTableBytes = impl_->statistics_.tableDeviceBytes_;
    auto const tableHash = impl_->statistics_.nativeTableHash_;
    auto const auxiliaryHash = impl_->statistics_.auxiliaryCacheHash_;
    auto const auxiliaryHit = impl_->statistics_.auxiliaryCacheHit_;
    auto const proposal_version = impl_->statistics_.nativeProposalVersion_;
    auto const interpolationVersion = impl_->statistics_.nativeCubicInterpolationVersion_;
    auto const nativeNodes = impl_->statistics_.nativeTableNodes_;
    auto const cacheTables = impl_->statistics_.proposalCacheTableCount_;
    auto const cacheHits = impl_->statistics_.proposalCacheHitCount_;
    auto const cacheAllHit = impl_->statistics_.proposalCacheAllHit_;
    impl_->statistics_ = {};
    impl_->statistics_.showerOrdinal_ = ordinal;
    impl_->statistics_.reusedForShower_ = true;
    impl_->statistics_.physicsSource_ = PhysicsSource::ProposalNative;
    impl_->statistics_.acceleratorBackend_ = staticStatistics.acceleratorBackend_;
    impl_->statistics_.acceleratorDeviceName_ = staticStatistics.acceleratorDeviceName_;
    impl_->statistics_.acceleratorArchitecture_ =
        staticStatistics.acceleratorArchitecture_;
    impl_->statistics_.acceleratorDriverVersion_ =
        staticStatistics.acceleratorDriverVersion_;
    impl_->statistics_.acceleratorRuntimeVersion_ =
        staticStatistics.acceleratorRuntimeVersion_;
    impl_->statistics_.acceleratorCompilerVersion_ =
        staticStatistics.acceleratorCompilerVersion_;
    impl_->statistics_.acceleratorProjectRevision_ =
        staticStatistics.acceleratorProjectRevision_;
    impl_->statistics_.acceleratorDevice_ = staticStatistics.acceleratorDevice_;
    impl_->statistics_.acceleratorConcurrency_ = staticStatistics.acceleratorConcurrency_;
    impl_->statistics_.acceleratorHostThreads_ = staticStatistics.acceleratorHostThreads_;
    impl_->statistics_.acceleratorOpenmp_ = staticStatistics.acceleratorOpenmp_;
    impl_->statistics_.acceleratorGpu_ = staticStatistics.acceleratorGpu_;
    impl_->statistics_.staticHostToDeviceBytes_ =
        staticStatistics.staticHostToDeviceBytes_;
    impl_->statistics_.oneTimeInitializationMs_ =
        staticStatistics.oneTimeInitializationMs_;
    impl_->statistics_.tuningCacheMatched_ = staticStatistics.tuningCacheMatched_;
    impl_->statistics_.tuningCacheRequired_ = staticStatistics.tuningCacheRequired_;
    impl_->statistics_.tuningCacheHash_ = staticStatistics.tuningCacheHash_;
    impl_->statistics_.tuningBatchSize_ = staticStatistics.tuningBatchSize_;
    impl_->statistics_.tuningChunkSize_ = staticStatistics.tuningChunkSize_;
    impl_->statistics_.tuningTeamSize_ = staticStatistics.tuningTeamSize_;
    impl_->statistics_.tuningTrackTileSize_ = staticStatistics.tuningTrackTileSize_;
    impl_->statistics_.tuningObserverTileSize_ = staticStatistics.tuningObserverTileSize_;
    impl_->statistics_.tuningDeviceQueues_ = staticStatistics.tuningDeviceQueues_;
    impl_->statistics_.nativeProposalVersion_ = proposal_version;
    impl_->statistics_.nativeCubicInterpolationVersion_ = interpolationVersion;
    impl_->statistics_.nativeTableNodes_ = nativeNodes;
    impl_->statistics_.proposalCacheTableCount_ = cacheTables;
    impl_->statistics_.proposalCacheHitCount_ = cacheHits;
    impl_->statistics_.proposalCacheAllHit_ = cacheAllHit;
    impl_->statistics_.maximumResidentPhotonBatch_ = impl_->maximumResidentBatchSize_;
    impl_->statistics_.maximumResidentLeptonBatch_ = impl_->maximumResidentBatchSize_;
    impl_->statistics_.automaticCapacityBudgetBytes_ =
        staticStatistics.automaticCapacityBudgetBytes_;
    impl_->statistics_.automaticCapacityPlannedBytes_ =
        staticStatistics.automaticCapacityPlannedBytes_;
    impl_->statistics_.automaticCapacityPeakBytes_ =
        staticStatistics.automaticCapacityPeakBytes_;
    impl_->statistics_.automaticCapacityAllocatorReserveBytes_ =
        staticStatistics.automaticCapacityAllocatorReserveBytes_;
    impl_->statistics_.crossSpeciesQueueCapacityPerPid_ = impl_->maximumPendingParticles_;
    impl_->statistics_.nativeTableHash_ = tableHash;
    impl_->statistics_.auxiliaryCacheHash_ = auxiliaryHash;
    impl_->statistics_.nativeTableDeviceBytes_ = staticBytes;
    impl_->statistics_.tableDeviceBytes_ = allTableBytes;
    impl_->statistics_.auxiliaryCacheHit_ = auxiliaryHit;
    impl_->statistics_.profile_.enabled_ = impl_->profileAccumulator_.enabled();
    impl_->statistics_.profile_.deterministic_ = true;
    impl_->statistics_.profile_.bins_ = impl_->config_.profileProjection_.outputBinCount_;
    impl_->statistics_.profile_.deviceBytes_ = impl_->profileProjection_.deviceBytes() +
                                               impl_->profileAccumulator_.deviceBytes();
    impl_->profileAccumulator_.reset(shower.profileFixedPointWeightLimit_,
                                     shower.profileFixedPointEnergyLimitGeV_,
                                     impl_->execution_);
    impl_->radioAccumulator_.reset(impl_->execution_);
    impl_->pendingPhotons_.clear();
    impl_->pendingLeptons_.clear();
    impl_->firstInteraction_.reset();
    impl_->refreshDeviceMemoryStatistics();
  }

  template <class TExecutionSpace>
  bool BackendInstanceImplementation<TExecutionSpace>::canTransport(
      ParticleState const& particle) const {
    if (!impl_->initialized_) return false;
    if (particle.pid_ == static_cast<std::int32_t>(EmPid::Photon) ||
        is_electron_or_positron_pid(particle.pid_))
      return true;
    if (!is_muon_pid(particle.pid_) || impl_->auxiliary_.hasMuonMoliere_ == 0)
      return false;
    return std::any_of(impl_->hostTable_.interactionIdentities_.begin(),
                       impl_->hostTable_.interactionIdentities_.end(),
                       [&](auto const& entry) { return entry.pdgId_ == particle.pid_; });
  }

  template <class TExecutionSpace>
  bool BackendInstanceImplementation<TExecutionSpace>::hasProposalTable() const {
    return impl_->initialized_;
  }

  template <class TExecutionSpace>
  std::size_t BackendInstanceImplementation<TExecutionSpace>::minimumBatchSize() const {
    impl_->requireInitialized();
    return impl_->config_.minBatchSize_;
  }

  template <class TExecutionSpace>
  std::size_t
  BackendInstanceImplementation<TExecutionSpace>::maximumResidentPhotonBatchSize() const {
    impl_->requireInitialized();
    return impl_->maximumResidentBatchSize_;
  }

  template <class TExecutionSpace>
  std::size_t
  BackendInstanceImplementation<TExecutionSpace>::maximumResidentLeptonBatchSize() const {
    impl_->requireInitialized();
    return impl_->maximumResidentBatchSize_;
  }

  template <class TExecutionSpace>
  std::size_t
  BackendInstanceImplementation<TExecutionSpace>::maximumResidentInputBatchSize() const {
    impl_->requireInitialized();
    return impl_->maximumResidentBatchSize_;
  }

  template <class TExecutionSpace>
  std::size_t BackendInstanceImplementation<TExecutionSpace>::pendingPhotonCount() const
      noexcept {
    return impl_->config_.residentCrossSpecies_ ? impl_->pendingPhotons_.size()
                                                : std::size_t{};
  }

  template <class TExecutionSpace>
  std::size_t BackendInstanceImplementation<TExecutionSpace>::pendingLeptonCount() const
      noexcept {
    return impl_->config_.residentCrossSpecies_ ? impl_->pendingLeptons_.size()
                                                : std::size_t{};
  }

  template <class TExecutionSpace>
  PhotonCascadeResult BackendInstanceImplementation<TExecutionSpace>::runPhotonWavefront(
      std::vector<ParticleState> const& particles, std::uint64_t firstSecondaryHistoryId,
      std::size_t maximumWavefronts, std::size_t minimumResidentBatchSize) {
    using namespace corsika::kokkos;
    impl_->requireInitialized();
    if (particles.size() > impl_->maximumResidentBatchSize_)
      throw std::length_error(
          "Kokkos photon input exceeds the advertised resident batch limit");
    auto const retainCrossSpecies = impl_->config_.residentCrossSpecies_;
    auto const pendingPhotonInput =
        std::min(retainCrossSpecies
                     ? std::min(impl_->pendingPhotons_.size(),
                                particles.size() < impl_->maximumResidentBatchSize_
                                    ? impl_->maximumResidentBatchSize_ - particles.size()
                                    : std::size_t{})
                     : std::size_t{},
                 impl_->cooperativePendingLimit_);
    auto const pendingLeptonsBefore = impl_->pendingLeptons_.size();
    auto resident = run_photon_cascade(
        impl_->physicsContext_.getDeviceView(), particles, firstSecondaryHistoryId,
        maximumWavefronts, minimumResidentBatchSize, impl_->firstInteraction_,
        impl_->profileProjection_.enabled() && !impl_->profileAccumulator_.enabled(),
        impl_->profileProjection_.deviceView(),
        impl_->profileAccumulator_.enabled() ? &impl_->profileAccumulator_ : nullptr,
        &impl_->photonWorkspace_, retainCrossSpecies ? &impl_->pendingPhotons_ : nullptr,
        retainCrossSpecies ? &impl_->pendingLeptons_ : nullptr, pendingPhotonInput,
        impl_->maximumPendingParticles_, impl_->execution_,
        impl_->config_.diagnosticInteractionRecords_,
        impl_->tuning_.parameters_.chunkSize_, impl_->residentMemoryGate(),
        &impl_->cooperativeWait_);
    if (resident.inputParticles_ != pendingPhotonInput + particles.size())
      throw std::logic_error(
          "Kokkos photon pending-input accounting disagrees with the router");
    if (retainCrossSpecies && !resident.electromagneticSecondaries_.empty())
      throw std::logic_error(
          "resident Kokkos photon cascade returned retained leptons twice");
    auto const pendingLeptonsAdded = impl_->pendingLeptons_.size() - pendingLeptonsBefore;
    impl_->statistics_.physicalHostToDeviceBytes_ +=
        particles.size() * sizeof(ParticleState);
    impl_->statistics_.crossSpeciesParticlesKeptOnDevice_ += pendingLeptonsAdded;
    impl_->statistics_.crossSpeciesDeviceToDeviceBytes_ +=
        (pendingPhotonInput + pendingLeptonsAdded) * sizeof(ParticleState);
    if (retainCrossSpecies && !resident.cpuSpillParticles_.empty()) {
      ++impl_->statistics_.crossSpeciesHostSpills_;
      impl_->statistics_.crossSpeciesParticlesSpilledToCpu_ +=
          resident.cpuSpillParticles_.size();
    }
    impl_->refreshDeviceMemoryStatistics();
    impl_->statistics_.physicalPhotonWavefronts_ += resident.wavefronts_;
    impl_->statistics_.wavefrontBucketingBatches_ += resident.wavefrontBucketingBatches_;
    impl_->statistics_.wavefrontBucketingParticles_ +=
        resident.wavefrontBucketingParticles_;
    impl_->statistics_.wavefrontBucketingSmallBatches_ +=
        resident.wavefrontBucketingSmallBatches_;
    impl_->statistics_.wavefrontBucketingSmallParticles_ +=
        resident.wavefrontBucketingSmallParticles_;
    impl_->statistics_.nativeNewtonIterations_ += resident.nativeNewtonIterations_;
    impl_->statistics_.nativeBisectionIterations_ += resident.nativeBisectionIterations_;
    impl_->statistics_.nativeInverseFailures_ += resident.nativeInverseFailures_;
    impl_->statistics_.interactionSelectionBatches_ += resident.wavefronts_;
    impl_->statistics_.photonTransportBatches_ += resident.wavefronts_;
    impl_->statistics_.interactionsSelected_ += resident.selectedInteractions_;
    impl_->statistics_.finalStateBatches_ += resident.interactionBearingWavefronts_;
    impl_->statistics_.photonTransportInteractions_ += resident.interactionVertices_;
    impl_->statistics_.photonTransportBoundaries_ += resident.layerBoundaries_;
    impl_->statistics_.photonTransportCuts_ += resident.particleCuts_;
    impl_->statistics_.photonPairLpmSuppressions_ +=
        resident.processStatistics_.photonPairLpmSuppressions_;
    impl_->statistics_.proposalFallbacks_ += resident.fallbackEvents_.size();
    auto const& process = resident.processStatistics_;
    impl_->statistics_.gpuFinalStates_ += process.gpuFinalStates_;
    impl_->statistics_.physicalSecondariesGenerated_ +=
        process.physicalSecondariesGenerated_;
    impl_->statistics_.photonPairFinalStates_ += process.photonPairFinalStates_;
    impl_->statistics_.photonMuonPairFinalStates_ += process.photonMuonPairFinalStates_;
    impl_->statistics_.comptonFinalStates_ += process.comptonFinalStates_;
    impl_->statistics_.photoelectricFinalStates_ += process.photoelectricFinalStates_;
    impl_->statistics_.photonPairLpmTrials_ += process.photonPairLpmTrials_;
    impl_->statistics_.thinningHillasVertices_ += process.thinningHillasVertices_;
    impl_->statistics_.thinningStatisticalVertices_ +=
        process.thinningStatisticalVertices_;
    impl_->statistics_.thinningParticlesDiscarded_ += process.thinningParticlesDiscarded_;
    for (auto const& observation : resident.observations_) {
      if (observation.status_ == ObservationStatus::ReachedObservationSurface)
        impl_->statistics_.photonTransportObservations_++;
      else
        impl_->statistics_.photonTransportEscapes_++;
    }
    impl_->enrichFallbackEvents(resident.fallbackEvents_);
    return resident;
  }

  template <class TExecutionSpace>
  LeptonCascadeResult BackendInstanceImplementation<TExecutionSpace>::runLeptonWavefront(
      std::vector<ParticleState> const& particles, std::uint64_t firstSecondaryHistoryId,
      std::size_t maximumWavefronts, std::uint64_t secondaryHistoryIdLimitExclusive,
      std::size_t minimumResidentBatchSize) {
    using namespace corsika::kokkos;
    impl_->requireInitialized();
    if (particles.size() > impl_->maximumResidentBatchSize_)
      throw std::length_error(
          "Kokkos lepton input exceeds the advertised resident batch limit");
    auto const retainCrossSpecies = impl_->config_.residentCrossSpecies_;
    auto const pendingLeptonInput =
        std::min(retainCrossSpecies
                     ? std::min(impl_->pendingLeptons_.size(),
                                particles.size() < impl_->maximumResidentBatchSize_
                                    ? impl_->maximumResidentBatchSize_ - particles.size()
                                    : std::size_t{})
                     : std::size_t{},
                 impl_->cooperativePendingLimit_);
    auto const pendingPhotonsBefore = impl_->pendingPhotons_.size();
    auto resident = run_lepton_cascade(
        impl_->physicsContext_.getDeviceView(),
        impl_->auxiliary_.bremsLpm_.leptonMassMeV_ / 1000.,
        impl_->auxiliary_.electronMoliere_.componentCount_ <= 4u, particles,
        firstSecondaryHistoryId, maximumWavefronts, secondaryHistoryIdLimitExclusive,
        minimumResidentBatchSize, impl_->firstInteraction_,
        impl_->profileProjection_.enabled() && !impl_->profileAccumulator_.enabled(),
        impl_->profileProjection_.deviceView(),
        impl_->profileAccumulator_.enabled() ? &impl_->profileAccumulator_ : nullptr,
        impl_->radioAccumulator_.enabled() ? &impl_->radioAccumulator_ : nullptr,
        &impl_->leptonWorkspace_, retainCrossSpecies ? &impl_->pendingLeptons_ : nullptr,
        retainCrossSpecies ? &impl_->pendingPhotons_ : nullptr, pendingLeptonInput,
        impl_->maximumPendingParticles_, impl_->execution_,
        impl_->config_.diagnosticInteractionRecords_,
        impl_->tuning_.parameters_.chunkSize_, impl_->residentMemoryGate(),
        impl_->automaticGpuCapacity_ ? impl_->maximumResidentBatchSize_ : 0,
        &impl_->cooperativeWait_);
    if (resident.inputParticles_ != pendingLeptonInput + particles.size())
      throw std::logic_error(
          "Kokkos lepton pending-input accounting disagrees with the router");
    if (retainCrossSpecies && !resident.generatedPhotons_.empty())
      throw std::logic_error(
          "resident Kokkos lepton cascade returned retained photons twice");
    auto const pendingPhotonsAdded = impl_->pendingPhotons_.size() - pendingPhotonsBefore;
    impl_->statistics_.physicalHostToDeviceBytes_ +=
        particles.size() * sizeof(ParticleState);
    impl_->statistics_.crossSpeciesParticlesKeptOnDevice_ += pendingPhotonsAdded;
    impl_->statistics_.crossSpeciesDeviceToDeviceBytes_ +=
        (pendingLeptonInput + pendingPhotonsAdded) * sizeof(ParticleState);
    if (retainCrossSpecies && !resident.cpuSpillParticles_.empty()) {
      ++impl_->statistics_.crossSpeciesHostSpills_;
      impl_->statistics_.crossSpeciesParticlesSpilledToCpu_ +=
          resident.cpuSpillParticles_.size();
    }
    impl_->refreshDeviceMemoryStatistics();
    impl_->statistics_.physicalLeptonWavefronts_ += resident.wavefronts_;
    impl_->statistics_.wavefrontBucketingBatches_ += resident.wavefrontBucketingBatches_;
    impl_->statistics_.wavefrontBucketingParticles_ +=
        resident.wavefrontBucketingParticles_;
    impl_->statistics_.wavefrontBucketingSmallBatches_ +=
        resident.wavefrontBucketingSmallBatches_;
    impl_->statistics_.wavefrontBucketingSmallParticles_ +=
        resident.wavefrontBucketingSmallParticles_;
    impl_->statistics_.interactionSelectionBatches_ += resident.wavefronts_;
    impl_->statistics_.leptonTransportBatches_ += resident.wavefronts_;
    impl_->statistics_.leptonVertexSelectionBatches_ +=
        resident.interactionBearingWavefronts_;
    impl_->statistics_.leptonVertexInteractionsSelected_ +=
        resident.vertexInteractionsSelected_;
    impl_->statistics_.leptonVertexNoInteractionContinuations_ +=
        resident.vertexNoInteractionContinuations_;
    impl_->statistics_.finalStateBatches_ += resident.finalStateBearingWavefronts_;
    impl_->statistics_.interactionsSelected_ += resident.selectedInteractions_;
    impl_->statistics_.proposalFallbacks_ += resident.fallbackEvents_.size();
    auto const& process = resident.processStatistics_;
    impl_->statistics_.gpuFinalStates_ += process.gpuFinalStates_;
    impl_->statistics_.physicalSecondariesGenerated_ +=
        process.physicalSecondariesGenerated_;
    impl_->statistics_.bremsFinalStates_ += process.bremsFinalStates_;
    impl_->statistics_.annihilationFinalStates_ += process.annihilationFinalStates_;
    impl_->statistics_.ionizationFinalStates_ += process.ionizationFinalStates_;
    impl_->statistics_.electronPairFinalStates_ += process.electronPairFinalStates_;
    impl_->statistics_.bremsLpmTrials_ += process.bremsLpmTrials_;
    impl_->statistics_.bremsLpmSuppressions_ += process.bremsLpmSuppressions_;
    impl_->statistics_.electronPairLpmTrials_ += process.electronPairLpmTrials_;
    impl_->statistics_.electronPairLpmSuppressions_ +=
        process.electronPairLpmSuppressions_;
    impl_->statistics_.electronPairRejectionTrials_ +=
        process.electronPairRejectionTrials_;
    impl_->statistics_.electronPairZeroWeightSamples_ +=
        process.electronPairZeroWeightSamples_;
    impl_->statistics_.electronPairRejectionFallbacks_ +=
        process.electronPairRejectionFallbacks_;
    impl_->statistics_.electronPairEnvelopeViolations_ +=
        process.electronPairEnvelopeViolations_;
    impl_->statistics_.thinningHillasVertices_ += process.thinningHillasVertices_;
    impl_->statistics_.thinningStatisticalVertices_ +=
        process.thinningStatisticalVertices_;
    impl_->statistics_.thinningParticlesDiscarded_ += process.thinningParticlesDiscarded_;
    impl_->statistics_.moliereTrials_ += process.moliereTrials_;
    impl_->statistics_.moliereDeflections_ += process.moliereDeflections_;
    impl_->statistics_.moliereZeroDeflections_ += process.moliereZeroDeflections_;
    impl_->statistics_.moliereNewtonIterations_ += process.moliereNewtonIterations_;
    impl_->statistics_.moliereMaxNewtonIterations_ =
        std::max(impl_->statistics_.moliereMaxNewtonIterations_,
                 process.moliereMaxNewtonIterations_);
    auto const& limits = process.leptonTransportLimits_;
    impl_->statistics_.leptonTransportInteractionCandidates_ += limits[0];
    impl_->statistics_.leptonTransportContinuousSteps_ += limits[1];
    impl_->statistics_.leptonTransportCuts_ += limits[2];
    impl_->statistics_.leptonTransportBoundaries_ += limits[3];
    impl_->statistics_.leptonTransportObservations_ += limits[4];
    impl_->statistics_.leptonTransportEscapes_ += limits[5];
    impl_->statistics_.leptonTransportMagneticSteps_ += limits[6];
    impl_->statistics_.leptonTransportDecayCandidates_ += limits[7];
    impl_->enrichFallbackEvents(resident.fallbackEvents_);
    return resident;
  }

  template <class TExecutionSpace>
  BackendCapabilities BackendInstanceImplementation<TExecutionSpace>::capabilities()
      const {
    auto const ready = impl_->initialized_;
    return {impl_->runtime_.getInfo().kind_,
            ready,
            ready,
            ready,
            ready && impl_->auxiliary_.hasMuonMoliere_ != 0,
            ready && impl_->profileAccumulator_.enabled(),
            ready && impl_->radioAccumulator_.enabled(),
            ready && impl_->radioAccumulator_.enabled(),
            true};
  }

  template <class TExecutionSpace>
  BackendStatistics const& BackendInstanceImplementation<TExecutionSpace>::statistics()
      const {
    impl_->statistics_.radio_ = impl_->radioAccumulator_.statistics();
    return impl_->statistics_;
  }

  template <class TExecutionSpace>
  bool BackendInstanceImplementation<TExecutionSpace>::gpuProfileEnabled() const
      noexcept {
    return impl_->profileAccumulator_.enabled();
  }

  template <class TExecutionSpace>
  ProfileResult BackendInstanceImplementation<TExecutionSpace>::downloadProfile() {
    impl_->requireInitialized();
    auto result = impl_->profileAccumulator_.download(impl_->execution_);
    impl_->statistics_.profile_.steps_ = result.steps_;
    impl_->statistics_.profile_.depositedSteps_ = result.depositedSteps_;
    impl_->statistics_.profile_.fixedPointOverflows_ = result.fixedPointOverflows_;
    impl_->statistics_.profile_.invalidRecords_ = result.invalidRecords_;
    impl_->statistics_.profile_.deviceToHostBytes_ +=
        impl_->profileAccumulator_.deviceBytes() -
        impl_->profileAccumulator_.hostShards().bytes();
    return result;
  }

  template <class TExecutionSpace>
  bool BackendInstanceImplementation<TExecutionSpace>::gpuRadioEnabled() const noexcept {
    return impl_->radioAccumulator_.enabled();
  }

  template <class TExecutionSpace>
  void BackendInstanceImplementation<TExecutionSpace>::enqueueRadioTracks(
      boost::span<LeptonTransportRecord const> records) {
    impl_->requireInitialized();
    impl_->radioAccumulator_.enqueueHostLeptonTracks(records.data(), records.size(),
                                                     impl_->execution_);
  }

  template <class TExecutionSpace>
  radio::RadioWaveforms
  BackendInstanceImplementation<TExecutionSpace>::downloadRadioWaveforms() {
    impl_->requireInitialized();
    auto result = impl_->radioAccumulator_.download(impl_->execution_);
    impl_->statistics_.radio_ = impl_->radioAccumulator_.statistics();
    impl_->refreshDeviceMemoryStatistics();
    return result;
  }

  template <class TExecutionSpace>
  std::optional<FirstInteractionRecord>
  BackendInstanceImplementation<TExecutionSpace>::downloadFirstInteractionSnapshot() {
    return impl_->firstInteraction_;
  }

  template <class TExecutionSpace>
  RuntimeInfo const& BackendInstanceImplementation<TExecutionSpace>::runtimeInfo() const
      noexcept {
    return impl_->runtime_.getInfo();
  }

} // namespace corsika::kokkos
