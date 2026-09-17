/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

// Private template implementation shared by every execution-space instance.
#pragma once

#include <corsika/accelerator/em/detail/KokkosBackendInstance.hpp>
#include <corsika/accelerator/em/KokkosTuningCache.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <corsika/accelerator/em/kokkos/KokkosInteractionSelector.hpp>
#include <corsika/accelerator/em/kokkos/KokkosLeptonFinalState.hpp>
#include <corsika/accelerator/em/kokkos/KokkosLeptonTransport.hpp>
#include <corsika/accelerator/em/kokkos/KokkosLeptonVertexSelector.hpp>
#include <corsika/accelerator/em/kokkos/KokkosMoliereInterpolation.hpp>
#include <corsika/accelerator/em/kokkos/KokkosPhotonFinalState.hpp>
#include <corsika/accelerator/em/kokkos/KokkosPhotonTransport.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileProjection.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentPhotonCascade.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentCapacity.hpp>
#include <corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp>
#include <corsika/accelerator/em/common/tables/Sha256.hpp>

namespace corsika::accelerator::em::detail {

  template <class BackendExecutionSpace>
  class KokkosBackendInstanceImpl final : public KokkosBackendInstance {
   public:
    explicit KokkosBackendInstanceImpl(KokkosRuntimeConfig const&);
    ~KokkosBackendInstanceImpl() override;
    void initialize(
      gpu::em::EnvironmentSnapshot const& environment,
      gpu::em::tables::ProposalNativeTableSet const& table,
      gpu::em::tables::ProposalNativeAuxData const& auxiliary,
      gpu::em::GpuEmConfig const& config) override;
    gpu::em::EmInteractionBatchResult
  selectInteractionsForValidation(
      std::vector<gpu::em::EmParticleState> const& particles) override;
    gpu::em::PhotonTransportBatchResult
  transportPhotonsForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions) override;
    gpu::em::LeptonTransportBatchResult
  transportLeptonsForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions) override;
    gpu::em::LeptonVertexSelectionBatchResult
  selectLeptonVerticesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& candidates) override;
    gpu::em::EmFinalStateBatchResult
  generatePhotonFinalStatesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions,
      std::uint64_t const first_secondary_history_id) override;
    gpu::em::BremsFinalStateBatchResult
  generateLeptonFinalStatesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions,
      std::uint64_t const first_secondary_history_id) override;
    gpu::radio::GpuRadioWaveforms projectRadioForValidation(
      std::vector<gpu::em::LeptonTransportRecord> const& records) override;
    void beginShower(
      AcceleratedEmShowerConfig const& shower) override;
    bool canTransport(
      gpu::em::EmParticleState const& particle) const override;
    bool hasProposalTable() const override;
    std::size_t minimumBatchSize() const override;
    std::size_t maximumResidentPhotonBatchSize() const override;
    std::size_t maximumResidentLeptonBatchSize() const override;
    std::size_t maximumResidentInputBatchSize() const override;
    std::size_t pendingPhotonCount() const noexcept override;
    std::size_t pendingLeptonCount() const noexcept override;
    gpu::em::ResidentPhotonCascadeResult runPhotonWavefront(
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id,
      std::size_t maximum_wavefronts,
      std::size_t minimum_resident_batch_size) override;
    gpu::em::ResidentLeptonCascadeResult runLeptonWavefront(
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id,
      std::size_t maximum_wavefronts,
      std::uint64_t secondary_history_id_limit_exclusive,
      std::size_t minimum_resident_batch_size) override;
    BackendCapabilities capabilities() const override;
    AcceleratedEmStatistics const& statistics() const override;
    bool gpuProfileEnabled() const noexcept override;
    gpu::em::GpuProfileResult downloadProfile() override;
    bool gpuRadioEnabled() const noexcept override;
    gpu::radio::GpuRadioWaveforms downloadRadioWaveforms() override;
    std::optional<gpu::em::GpuFirstInteractionSnapshot>
  downloadFirstInteractionSnapshot() override;
    KokkosRuntimeInfo const& runtimeInfo() const noexcept override;
    void setCooperativeProgress(std::function<bool()> progress) override {
      impl_->cooperative_wait_.setProgress(std::move(progress));
    }
    void setCooperativeBlockingWait(bool const enabled) override {
      impl_->requireInitialized();
      if (!impl_->runtime_.info().cooperative_runtime ||
          impl_->statistics_.particles_advanced != 0 ||
          // Fused resident transport uses these counters, not particles_advanced.
          // They also cover the standalone transport-oracle entry points.
          impl_->statistics_.photon_transport_batches != 0 ||
          impl_->statistics_.lepton_transport_batches != 0 ||
          !impl_->pending_photons_.empty() || !impl_->pending_leptons_.empty())
        throw std::logic_error(
            "Blocking CUDA waits require an idle cooperative endpoint before transport");
      impl_->cooperative_wait_.setBlocking(enabled);
    }
    CooperativeCudaWaitStatistics cooperativeWaitStatistics() const override {
      auto const& wait = impl_->cooperative_wait_;
      return {wait.blockingEnabled(), wait.blockingCalls(), wait.blockingHostSeconds()};
    }
    void setCooperativePendingInputLimit(std::size_t count) override {
      impl_->cooperative_pending_limit_ = count;
    }
    void prepareCooperativeWorkspace() override {
      impl_->requireInitialized();
      bool const projected = impl_->profile_projection_.enabled() &&
                             !impl_->profile_accumulator_.enabled();
      kokkos_detail::reserveResidentArenas(
          impl_->maximum_resident_batch_size_, impl_->photon_workspace_,
          impl_->lepton_workspace_, impl_->radio_accumulator_, projected,
          impl_->execution_);
      impl_->pending_photons_.ensureTailCapacity(
          impl_->maximum_pending_particles_, impl_->maximum_pending_particles_,
          impl_->execution_);
      impl_->pending_leptons_.ensureTailCapacity(
          impl_->maximum_pending_particles_, impl_->maximum_pending_particles_,
          impl_->execution_);
      impl_->execution_.fence("prepare bounded cooperative endpoint workspace");
      impl_->refreshDeviceMemoryStatistics();
    }
    std::pair<std::size_t, std::size_t> enableCooperativeHostProfileShards() override {
      impl_->requireInitialized();
      if (!impl_->runtime_.info().cooperative_runtime ||
          !impl_->runtime_.info().openmp || impl_->runtime_.info().gpu ||
          impl_->statistics_.particles_advanced != 0)
        throw std::logic_error("host profile shards require an idle cooperative OpenMP endpoint");
      auto& profile = impl_->profile_accumulator_;
      if (!profile.enabled()) return {};
      auto const retained = impl_->retainedDeviceBytes();
      auto const available = impl_->memory_budget_bytes_ == 0
          ? HostProfileShards::MaximumBytes
          : impl_->memory_budget_bytes_ - std::min(retained, impl_->memory_budget_bytes_);
      profile.enableCooperativeHostShards(
          static_cast<std::size_t>(impl_->runtime_.info().host_threads), available);
      impl_->refreshDeviceMemoryStatistics();
      return {profile.hostShards().count(), profile.hostShards().bytes()};
    }
    std::vector<gpu::em::EmParticleState> takeCooperativePending(
        bool photons, std::size_t count) override {
      auto& queue = photons ? impl_->pending_photons_ : impl_->pending_leptons_;
      auto result = queue.downloadPrefix(count, impl_->execution_);
      queue.consume(count);
      return result;
    }
    FixedProfileSnapshot downloadFixedProfile(
        std::string const& identity, CooperativeEndpoint endpoint) override {
      auto result = impl_->profile_accumulator_.downloadFixed(
          identity, endpoint, impl_->execution_);
      auto decoded = decodeFixedProfile(result);
      impl_->statistics_.profile.steps = decoded.steps;
      impl_->statistics_.profile.deposited_steps = decoded.deposited_steps;
      impl_->statistics_.profile.fixed_point_overflows = decoded.fixed_point_overflows;
      impl_->statistics_.profile.invalid_records = decoded.invalid_records;
      return result;
    }
    radio::detail::FixedRadioSnapshot downloadFixedRadio(
        std::string const& identity, CooperativeEndpoint endpoint) override {
      auto result = impl_->radio_accumulator_.downloadFixed(
          identity, endpoint, impl_->execution_);
      impl_->statistics_.radio = impl_->radio_accumulator_.statistics();
      return result;
    }
   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

  template <class BackendExecutionSpace>
  class KokkosBackendInstanceImpl<BackendExecutionSpace>::Impl {
  public:
    explicit Impl(KokkosRuntimeConfig const& runtime_config)
        : runtime_{runtime_config} {}

    void requireInitialized() const {
      if (!initialized_) {
        throw std::logic_error("Kokkos EM backend is not initialized");
      }
    }

    kokkos_detail::ResidentExecutionWait<BackendExecutionSpace> cooperative_wait_;
    std::size_t cooperative_pending_limit_{std::numeric_limits<std::size_t>::max()};

    std::uint64_t interactionHashForPid(std::int32_t const pid) const {
      auto const found = std::find_if(
          interaction_hashes_.begin(), interaction_hashes_.end(),
          [pid](auto const& entry) { return entry.first == pid; });
      return found == interaction_hashes_.end() ? std::uint64_t{}
                                                : found->second;
    }

    void enrichFallbackEvents(
        std::vector<gpu::em::ProposalFallbackEvent>& events) const {
      for (auto& event : events) {
        event.medium_hash = proposal_medium_hash_;
        event.interaction_hash = interactionHashForPid(event.particle.pid);
      }
    }

    void refreshCrossSpeciesStorageStatistics() {
      statistics_.cross_species_queue_device_bytes =
          pending_photons_.deviceBytes() + pending_leptons_.deviceBytes();
      statistics_.cross_species_queue_capacity_per_pid =
          config_.resident_cross_species ? maximum_pending_particles_
                                         : std::size_t{};
      statistics_.peak_pending_photons = std::max(
          statistics_.peak_pending_photons, pending_photons_.size());
      statistics_.peak_pending_leptons = std::max(
          statistics_.peak_pending_leptons, pending_leptons_.size());
    }

    struct StaticDeviceMemoryProjection {
      std::size_t proposal_native_table{};
      std::size_t moliere_interpolation{};
      std::size_t profile_projection{};
      std::size_t profile_accumulator{};
      std::size_t radio_accumulator{};
      std::size_t physics_context{};
      kokkos_detail::KokkosMemoryProjection aggregate{};
    };

    static StaticDeviceMemoryProjection projectStaticDeviceMemory(
        gpu::em::tables::ProposalNativeTableSet const& table,
        gpu::em::GpuEmConfig const& config) {
      StaticDeviceMemoryProjection result{};
      result.proposal_native_table =
          kokkos_detail::KokkosProposalNativeTable<
              BackendExecutionSpace>::projectedDeviceBytes(table);
      result.moliere_interpolation =
          kokkos_detail::KokkosMoliereInterpolation<
              BackendExecutionSpace>::projectedDeviceBytes();
      result.profile_projection =
          kokkos_detail::KokkosProfileProjection<
              BackendExecutionSpace>::projectedDeviceBytes(
              config.profile_projection);
      result.profile_accumulator =
          kokkos_detail::KokkosProfileAccumulator<
              BackendExecutionSpace>::projectedDeviceBytes(
              config.profile_projection);
      result.radio_accumulator =
          radio::kokkos_detail::KokkosRadioAccumulator<
              BackendExecutionSpace>::projectedDeviceBytes(config.radio);
      result.physics_context =
          kokkos_detail::KokkosPhysicsContext<
              BackendExecutionSpace>::projectedDeviceBytes();

      kokkos_detail::KokkosMemoryProjectionBuilder projection{0};
      projection.replace(0, result.proposal_native_table);
      projection.replace(0, result.moliere_interpolation);
      projection.replace(0, result.profile_projection);
      projection.replace(0, result.profile_accumulator);
      projection.replace(0, result.radio_accumulator);
      projection.replace(0, result.physics_context);
      result.aggregate = projection.result();
      return result;
    }

    void requireStaticDeviceMemory(
        StaticDeviceMemoryProjection const& projection) const {
      if (memory_budget_bytes_ == 0 ||
          projection.aggregate.transient_peak_bytes <= memory_budget_bytes_)
        return;
      throw std::runtime_error(
          "Kokkos initialization preflight requires " +
          std::to_string(projection.aggregate.transient_peak_bytes) +
          " bytes of device memory (proposal-native=" +
          std::to_string(projection.proposal_native_table) +
          ", Moliere=" +
          std::to_string(projection.moliere_interpolation) +
          ", profile-projection=" +
          std::to_string(projection.profile_projection) +
          ", profile-accumulator=" +
          std::to_string(projection.profile_accumulator) +
          ", radio=" +
          std::to_string(projection.radio_accumulator) +
          ", context=" + std::to_string(projection.physics_context) +
          "), exceeding the configured " +
          std::to_string(memory_budget_bytes_) +
          " byte device-memory budget before any static upload");
    }

    std::size_t retainedDeviceBytes() const {
      auto bytes = table_.deviceBytes();
      auto const append = [&bytes](std::size_t const amount) {
        bytes = kokkos_detail::checkedMemoryAdd(bytes, amount);
      };
      append(moliere_interpolation_.deviceBytes());
      append(physics_context_.deviceBytes());
      append(profile_projection_.deviceBytes());
      append(profile_accumulator_.deviceBytes());
      append(radio_accumulator_.deviceBytes());
      append(photon_workspace_.deviceBytes());
      append(lepton_workspace_.deviceBytes());
      append(pending_photons_.deviceBytes());
      append(pending_leptons_.deviceBytes());
      return bytes;
    }

    bool permitsResidentGrowth(
        std::size_t const current_owner_bytes,
        kokkos_detail::KokkosMemoryProjection const& projection,
        char const* const) const {
      if (memory_budget_bytes_ == 0) return true;
      auto const retained = retainedDeviceBytes();
      if (current_owner_bytes > retained)
        throw std::logic_error(
            "Kokkos resident-memory projection owner exceeds total storage");
      auto const other = retained - current_owner_bytes;
      auto const retained_after = kokkos_detail::checkedMemoryAdd(
          other, projection.retained_bytes);
      auto const transient_peak = kokkos_detail::checkedMemoryAdd(
          other, projection.transient_peak_bytes);
      return std::max(retained_after, transient_peak) <= memory_budget_bytes_;
    }

    static bool residentMemoryGateCallback(
        void* context, std::size_t const current_owner_bytes,
        kokkos_detail::KokkosMemoryProjection const& projection,
        char const* const stage) {
      if (context == nullptr)
        throw std::logic_error(
            "Kokkos resident-memory gate has no backend context");
      return static_cast<Impl*>(context)->permitsResidentGrowth(
          current_owner_bytes, projection, stage);
    }

    kokkos_detail::KokkosResidentMemoryBudgetGate residentMemoryGate() {
      return {this, &Impl::residentMemoryGateCallback};
    }

    void refreshDeviceMemoryStatistics() {
      refreshCrossSpeciesStorageStatistics();
      statistics_.physical_workspace_bytes =
          photon_workspace_.deviceBytes() + lepton_workspace_.deviceBytes();
      statistics_.profile.device_bytes =
          profile_projection_.deviceBytes() +
          profile_accumulator_.deviceBytes();
      statistics_.radio.device_bytes = radio_accumulator_.deviceBytes();
      auto const resident_bytes = retainedDeviceBytes();
      statistics_.peak_device_bytes =
          std::max(statistics_.peak_device_bytes, resident_bytes);
      if (memory_budget_bytes_ != 0 &&
          resident_bytes > memory_budget_bytes_) {
        throw std::runtime_error(
            "Kokkos resident allocation requires " +
            std::to_string(resident_bytes) +
            " bytes, exceeding the configured " +
            std::to_string(memory_budget_bytes_) +
            " byte device-memory budget");
      }
    }

    KokkosRuntime runtime_;
    KokkosRuntimeConfig runtime_config_{};
    KokkosTuningCacheResult tuning_{};
    BackendExecutionSpace execution_{};
    kokkos_detail::KokkosProposalNativeTable<BackendExecutionSpace> table_{};
    kokkos_detail::KokkosMoliereInterpolation<BackendExecutionSpace>
        moliere_interpolation_{};
    kokkos_detail::KokkosProfileProjection<BackendExecutionSpace>
        profile_projection_{};
    kokkos_detail::KokkosProfileAccumulator<BackendExecutionSpace>
        profile_accumulator_{};
    radio::kokkos_detail::KokkosRadioAccumulator<BackendExecutionSpace>
        radio_accumulator_{};
    kokkos_detail::KokkosResidentPhotonWorkspace<BackendExecutionSpace>
        photon_workspace_{};
    kokkos_detail::KokkosResidentLeptonWorkspace<BackendExecutionSpace>
        lepton_workspace_{};
    kokkos_detail::KokkosPhysicsContext<BackendExecutionSpace>
        physics_context_{};
    kokkos_detail::KokkosPendingParticleQueue<BackendExecutionSpace>
        pending_photons_{"c8_kokkos_pending_photons"};
    kokkos_detail::KokkosPendingParticleQueue<BackendExecutionSpace>
        pending_leptons_{"c8_kokkos_pending_leptons"};
    gpu::em::tables::ProposalNativeTableSet host_table_{};
    gpu::em::tables::ProposalNativeAuxData auxiliary_{};
    gpu::em::EnvironmentSnapshot environment_{};
    gpu::em::GpuEmConfig config_{};
    gpu::em::GpuEmStatistics statistics_{};
    gpu::em::tables::NativePhysicsView physics_{};
    std::optional<gpu::em::GpuFirstInteractionSnapshot> first_interaction_{};
    std::uint64_t proposal_medium_hash_{};
    std::vector<std::pair<std::int32_t, std::uint64_t>> interaction_hashes_{};
    bool initialized_{};
    std::size_t maximum_resident_batch_size_{};
    std::size_t maximum_pending_particles_{};
    std::size_t memory_budget_bytes_{};
    bool automatic_gpu_capacity_{};
  };

  template <class BackendExecutionSpace>
  KokkosBackendInstanceImpl<BackendExecutionSpace>::KokkosBackendInstanceImpl(
      KokkosRuntimeConfig const& config)
      : impl_(std::make_unique<Impl>(config)) {
    impl_->runtime_config_ = config;
  }

  template <class BackendExecutionSpace>
  KokkosBackendInstanceImpl<BackendExecutionSpace>::~KokkosBackendInstanceImpl() = default;

  template <class BackendExecutionSpace>
  void KokkosBackendInstanceImpl<BackendExecutionSpace>::initialize(
      gpu::em::EnvironmentSnapshot const& environment,
      gpu::em::tables::ProposalNativeTableSet const& table,
      gpu::em::tables::ProposalNativeAuxData const& auxiliary,
      gpu::em::GpuEmConfig const& config) {
    Kokkos::Timer initialization_timer;
    if (impl_->initialized_) {
      throw std::logic_error("Kokkos EM backend is already initialized");
    }
    if (config.physics_source != gpu::em::GpuPhysicsSource::ProposalNative) {
      throw std::invalid_argument(
          "Kokkos EM backend supports proposal-native physics only");
    }
    if (!(config.memory_fraction > 0.) || !(config.memory_fraction <= 1.))
      throw std::invalid_argument(
          "Kokkos device memory fraction must be in the interval (0, 1]");
    gpu::em::tables::validateProposalNativeTable(table);
    if (gpu::em::tables::calculateProposalNativeHash(table) !=
        table.content_hash)
      throw std::invalid_argument(
          "PROPOSAL native-table content hash does not match its payload");
    impl_->environment_ = environment;
    impl_->config_ = config;
    impl_->host_table_ = table;
    impl_->proposal_medium_hash_ =
        table.interaction_identities.empty()
            ? std::uint64_t{}
            : table.interaction_identities.front().medium_hash;
    impl_->interaction_hashes_.clear();
    impl_->interaction_hashes_.reserve(table.interaction_identities.size());
    for (auto const& identity : table.interaction_identities)
      impl_->interaction_hashes_.emplace_back(
          identity.pdg_id, identity.interaction_hash);
    auto const& runtime = impl_->runtime_.info();
    if (runtime.gpu) {
      if (runtime.device_free_memory_bytes_at_initialization == 0)
        throw std::runtime_error(
            "Kokkos GPU backend cannot establish a device-memory budget");
      impl_->memory_budget_bytes_ = static_cast<std::size_t>(
          config.memory_fraction *
          static_cast<double>(
              runtime.device_free_memory_bytes_at_initialization));
      if (impl_->memory_budget_bytes_ == 0)
        throw std::runtime_error(
            "Kokkos GPU device-memory budget is zero");
    }
    auto const static_memory = Impl::projectStaticDeviceMemory(table, config);
    impl_->requireStaticDeviceMemory(static_memory);
    KokkosTuningKey const tuning_key{
        runtime.backend,
        runtime.device_name,
        runtime.architecture,
        runtime.driver_version,
        runtime.runtime_version,
        runtime.kokkos_version,
        runtime.compiler_version,
        runtime.project_revision,
        gpu::em::tables::toHex(table.content_hash),
        runtime.device,
        runtime.host_threads};
    impl_->tuning_ = loadKokkosTuningCache(
        impl_->runtime_config_.tuning_cache, tuning_key,
        impl_->runtime_config_.require_tuning);
    if (!impl_->tuning_.matched) {
      std::cerr
          << "CORSIKA Kokkos warning: no exactly matching tuning cache was "
             "applied; using conservative "
          << runtime.backend
          << " defaults. Run c8_kokkos_tune for this device, build, thread "
             "count, and proposal-native table hash.\n";
      impl_->tuning_.parameters.batch_size = config.min_batch_size;
      impl_->tuning_.parameters.chunk_size = runtime.openmp ? 16 : 1;
      impl_->tuning_.parameters.team_size = runtime.openmp ? 1 : 256;
      impl_->tuning_.parameters.track_tile_size = runtime.openmp ? 4 : 8;
      impl_->tuning_.parameters.observer_tile_size = runtime.openmp ? 4 : 32;
      impl_->tuning_.parameters.device_queues = 1;
    }
    impl_->auxiliary_ = auxiliary;
    impl_->table_.initialize(table);
    auto moliere_cache_path = auxiliary.cache_file;
    if (!moliere_cache_path.empty())
      moliere_cache_path += ".moliere-initial-v1.c8cache";
    impl_->moliere_interpolation_.initialize(
        auxiliary.electron_moliere, moliere_cache_path);
    impl_->profile_projection_.initialize(
        config.profile_projection, impl_->execution_);
    impl_->profile_accumulator_.initialize(
        config.profile_projection, impl_->execution_);
    impl_->radio_accumulator_.initialize(
        config.radio, impl_->tuning_.parameters, impl_->execution_);
    impl_->physics_ = {};
    impl_->physics_.proposal_native = impl_->table_.deviceView();
    impl_->physics_.physics_source = 1u;
    impl_->physics_.em_transport_cut_MeV = config.em_transport_cut_MeV;
    impl_->physics_.muon_transport_cut_MeV = config.muon_transport_cut_MeV;
    impl_->physics_context_.initialize(
        impl_->physics_, impl_->environment_,
        impl_->auxiliary_.electron_moliere, impl_->auxiliary_.muon_moliere,
        impl_->moliere_interpolation_.deviceView(),
        impl_->auxiliary_.has_muon_moliere != 0,
        impl_->auxiliary_.photon_pair_lpm, impl_->auxiliary_.brems_lpm,
        impl_->config_.thinning, impl_->profile_projection_.deviceView(),
        impl_->config_.random_seed, impl_->config_.shower_id,
        impl_->execution_);
    impl_->statistics_.shower_ordinal = 1;
    impl_->statistics_.accelerator_backend = runtime.backend;
    impl_->statistics_.accelerator_device_name = runtime.device_name;
    impl_->statistics_.accelerator_architecture = runtime.architecture;
    impl_->statistics_.accelerator_driver_version = runtime.driver_version;
    impl_->statistics_.accelerator_runtime_version = runtime.runtime_version;
    impl_->statistics_.accelerator_compiler_version = runtime.compiler_version;
    impl_->statistics_.accelerator_project_revision = runtime.project_revision;
    impl_->statistics_.accelerator_device = runtime.device;
    impl_->statistics_.accelerator_concurrency = runtime.concurrency;
    impl_->statistics_.accelerator_host_threads = runtime.host_threads;
    impl_->statistics_.accelerator_openmp = runtime.openmp;
    impl_->statistics_.accelerator_gpu = runtime.gpu;
    impl_->statistics_.tuning_cache_matched = impl_->tuning_.matched;
    impl_->statistics_.tuning_cache_required =
        impl_->runtime_config_.require_tuning;
    impl_->statistics_.tuning_cache_hash = impl_->tuning_.content_hash;
    impl_->statistics_.tuning_batch_size =
        impl_->tuning_.parameters.batch_size;
    impl_->statistics_.tuning_chunk_size =
        impl_->tuning_.parameters.chunk_size;
    impl_->statistics_.tuning_team_size = impl_->tuning_.parameters.team_size;
    impl_->statistics_.tuning_track_tile_size =
        impl_->tuning_.parameters.track_tile_size;
    impl_->statistics_.tuning_observer_tile_size =
        impl_->tuning_.parameters.observer_tile_size;
    impl_->statistics_.tuning_device_queues =
        impl_->tuning_.parameters.device_queues;
    impl_->statistics_.physics_source = gpu::em::GpuPhysicsSource::ProposalNative;
    impl_->statistics_.native_proposal_version = table.proposal_version;
    impl_->statistics_.native_cubic_interpolation_version =
        table.cubic_interpolation_version;
    impl_->statistics_.native_table_hash = table.content_hash;
    impl_->statistics_.auxiliary_cache_hash = auxiliary.content_hash;
    impl_->statistics_.native_table_nodes =
        table.bicubic_values.size() + table.cubic_values.size();
    impl_->statistics_.proposal_cache_table_count =
        table.proposal_cache_table_count;
    impl_->statistics_.proposal_cache_hit_count =
        table.proposal_cache_hit_count;
    impl_->statistics_.proposal_cache_all_hit =
        table.proposal_cache_all_hit;
    impl_->statistics_.native_table_device_bytes = impl_->table_.deviceBytes();
    impl_->statistics_.table_device_bytes =
        impl_->table_.deviceBytes() +
        impl_->moliere_interpolation_.deviceBytes();
    impl_->statistics_.profile.bins = config.profile_projection.output_bin_count;
    impl_->statistics_.profile.device_bytes =
        impl_->profile_projection_.deviceBytes() +
        impl_->profile_accumulator_.deviceBytes();
    impl_->statistics_.profile.enabled = impl_->profile_accumulator_.enabled();
    impl_->statistics_.profile.deterministic = true;
    impl_->statistics_.radio = impl_->radio_accumulator_.statistics();
    impl_->statistics_.auxiliary_cache_hit = auxiliary.cache_hit;
    impl_->statistics_.reused_for_shower = false;
    // OpenMP retains its separately validated capacity policy. GPU automatic
    // capacity is derived below from both arenas and the byte budget; an
    // explicit replay checkpoint keeps the original grow-only behavior.
    impl_->automatic_gpu_capacity_ =
        runtime.gpu && config.resident_batch_limit == 0;
    // The tuning batch is an execution/workspace sizing hint.  It must not
    // replace the router's minimum resident-wavefront checkpoint: changing
    // that checkpoint changes when histories return to the host stack and can
    // therefore change history-id allocation and the Philox stream.
    if (impl_->config_.resident_batch_limit != 0) {
      if (impl_->config_.resident_batch_limit <
          impl_->config_.min_batch_size) {
        throw std::invalid_argument(
            "configured Kokkos resident batch limit is smaller than the "
            "minimum execution batch");
      }
      impl_->maximum_resident_batch_size_ =
          impl_->config_.resident_batch_limit;
    } else if (!runtime.gpu) {
      impl_->maximum_resident_batch_size_ = std::max(
          {std::size_t{65536}, impl_->config_.min_batch_size,
           impl_->tuning_.parameters.batch_size});
    }
    impl_->statistics_.maximum_resident_photon_batch =
        impl_->maximum_resident_batch_size_;
    impl_->statistics_.maximum_resident_lepton_batch =
        impl_->maximum_resident_batch_size_;
    // Match native CUDA's conservative policy: at the default 70% device
    // budget, at most 512 MiB in total is devoted to the two cross-species
    // queues.  Smaller configured fractions scale that allowance down;
    // larger fractions leave the cap unchanged so transport workspaces retain
    // headroom.
    constexpr auto MaximumCrossSpeciesBytes = std::size_t{512} * 1024 * 1024;
    auto const scaled_cross_species_bytes = static_cast<std::size_t>(
        static_cast<double>(MaximumCrossSpeciesBytes) *
        std::min(1.0, config.memory_fraction / 0.70));
    impl_->maximum_pending_particles_ =
        config.resident_cross_species
            ? std::max(
                  config.min_batch_size,
                  scaled_cross_species_bytes /
                      (2 * sizeof(gpu::em::EmParticleState)))
            : std::size_t{};
    if (impl_->automatic_gpu_capacity_) {
      // Reserve at most 1/8 of the budget (512 MiB maximum) for two pending
      // queues, and account for replacement of one full queue during compact.
      impl_->maximum_pending_particles_ = config.resident_cross_species
          ? std::min(MaximumCrossSpeciesBytes, impl_->memory_budget_bytes_ / 8) /
                (2 * sizeof(gpu::em::EmParticleState))
          : 0;
      bool const project_steps = impl_->profile_projection_.enabled() &&
                                 !impl_->profile_accumulator_.enabled();
      auto const retained = impl_->retainedDeviceBytes();
      auto const plan = kokkos_detail::selectResidentCapacity(
          impl_->memory_budget_bytes_, config.min_batch_size,
          impl_->maximum_pending_particles_, [&](std::size_t count) {
            return kokkos_detail::projectResidentArenas(
                count, retained, impl_->photon_workspace_,
                impl_->lepton_workspace_, impl_->radio_accumulator_,
                project_steps, impl_->execution_);
          });
      impl_->maximum_resident_batch_size_ = plan.input_particles;
      // Pre-reserve once, after the complete projection passed. There is no
      // per-particle allocation and no allocation-driven RNG consumption.
      kokkos_detail::reserveResidentArenas(
          plan.input_particles, impl_->photon_workspace_,
          impl_->lepton_workspace_, impl_->radio_accumulator_,
          project_steps, impl_->execution_);
      impl_->pending_photons_.ensureTailCapacity(
          plan.pending_particles_per_species, plan.pending_particles_per_species,
          impl_->execution_);
      impl_->pending_leptons_.ensureTailCapacity(
          plan.pending_particles_per_species, plan.pending_particles_per_species,
          impl_->execution_);
      if (impl_->retainedDeviceBytes() != plan.retained_bytes)
        throw std::logic_error(
            "Kokkos automatic resident-capacity projection disagrees with allocation");
      impl_->statistics_.automatic_capacity_budget_bytes = impl_->memory_budget_bytes_;
      impl_->statistics_.automatic_capacity_planned_bytes = plan.retained_bytes;
      impl_->statistics_.automatic_capacity_peak_bytes = plan.transient_peak_bytes;
      impl_->statistics_.automatic_capacity_allocator_reserve_bytes =
          plan.allocator_reserve_bytes;
      impl_->statistics_.maximum_resident_photon_batch = plan.input_particles;
      impl_->statistics_.maximum_resident_lepton_batch = plan.input_particles;
    }
    impl_->statistics_.cross_species_queue_capacity_per_pid =
        impl_->maximum_pending_particles_;
    impl_->refreshDeviceMemoryStatistics();
    // Complete asynchronous initialization before stopping the lifecycle
    // timer, otherwise the first shower would inherit part of the setup cost.
    impl_->execution_.fence("complete Kokkos EM backend initialization");
    if (runtime.gpu) {
      using kokkos_detail::checkedMemoryAdd;
      auto static_upload_bytes = impl_->table_.deviceBytes();
      static_upload_bytes = checkedMemoryAdd(
          static_upload_bytes, impl_->moliere_interpolation_.deviceBytes());
      static_upload_bytes = checkedMemoryAdd(
          static_upload_bytes, impl_->profile_projection_.deviceBytes());
      static_upload_bytes = checkedMemoryAdd(
          static_upload_bytes,
          impl_->radio_accumulator_.staticHostToDeviceBytes());
      static_upload_bytes = checkedMemoryAdd(
          static_upload_bytes, impl_->physics_context_.deviceBytes());
      impl_->statistics_.static_host_to_device_bytes = static_upload_bytes;
    }
    impl_->statistics_.one_time_initialization_ms =
        1000. * initialization_timer.seconds();
    impl_->initialized_ = true;
  }

  template <class BackendExecutionSpace>
  gpu::em::EmInteractionBatchResult
  KokkosBackendInstanceImpl<BackendExecutionSpace>::selectInteractionsForValidation(
      std::vector<gpu::em::EmParticleState> const& particles) {
    impl_->requireInitialized();
    auto selected = kokkos_detail::selectInteractions(
        impl_->physics_, particles, impl_->config_.random_seed,
        impl_->config_.shower_id, impl_->execution_);
    impl_->statistics_.interaction_selection_batches++;
    impl_->statistics_.interactions_selected +=
        selected.batch.interactions.size();
    impl_->statistics_.proposal_fallbacks +=
        selected.batch.fallback_events.size();
    impl_->statistics_.native_newton_iterations +=
        selected.native_newton_iterations;
    impl_->statistics_.native_bisection_iterations +=
        selected.native_bisection_iterations;
    impl_->statistics_.native_inverse_failures +=
        selected.native_inverse_failures;
    impl_->enrichFallbackEvents(selected.batch.fallback_events);
    return std::move(selected.batch);
  }

  template <class BackendExecutionSpace>
  gpu::em::PhotonTransportBatchResult
  KokkosBackendInstanceImpl<BackendExecutionSpace>::transportPhotonsForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions) {
    impl_->requireInitialized();
    auto result = kokkos_detail::transportPhotons(
        impl_->environment_, interactions, impl_->execution_);
    impl_->statistics_.photon_transport_batches++;
    impl_->statistics_.proposal_fallbacks += result.fallback_events.size();
    for (auto const& record : result.records) {
      switch (record.limit) {
      case gpu::em::PhotonTransportLimit::Interaction:
        impl_->statistics_.photon_transport_interactions++;
        break;
      case gpu::em::PhotonTransportLimit::LayerBoundary:
        impl_->statistics_.photon_transport_boundaries++;
        break;
      case gpu::em::PhotonTransportLimit::ObservationSurface:
        impl_->statistics_.photon_transport_observations++;
        break;
      case gpu::em::PhotonTransportLimit::EscapedEnvironment:
        impl_->statistics_.photon_transport_escapes++;
        break;
      case gpu::em::PhotonTransportLimit::ParticleCut:
        impl_->statistics_.photon_transport_cuts++;
        break;
      }
    }
    impl_->enrichFallbackEvents(result.fallback_events);
    return result;
  }

  template <class BackendExecutionSpace>
  gpu::em::LeptonTransportBatchResult
  KokkosBackendInstanceImpl<BackendExecutionSpace>::transportLeptonsForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions) {
    impl_->requireInitialized();
    auto result = kokkos_detail::transportLeptons(
        impl_->physics_, impl_->environment_, impl_->auxiliary_.electron_moliere,
        impl_->auxiliary_.muon_moliere,
        impl_->moliere_interpolation_.deviceView(),
        impl_->auxiliary_.has_muon_moliere != 0,
        impl_->config_.random_seed, impl_->config_.shower_id, interactions,
        impl_->execution_);
    impl_->statistics_.lepton_transport_batches++;
    impl_->statistics_.proposal_fallbacks += result.fallback_events.size();
    impl_->enrichFallbackEvents(result.fallback_events);
    return result;
  }

  template <class BackendExecutionSpace>
  gpu::em::LeptonVertexSelectionBatchResult
  KokkosBackendInstanceImpl<BackendExecutionSpace>::selectLeptonVerticesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& candidates) {
    impl_->requireInitialized();
    auto result = kokkos_detail::selectLeptonVertices(
        impl_->physics_, candidates, impl_->config_.random_seed,
        impl_->config_.shower_id, impl_->execution_);
    impl_->statistics_.lepton_vertex_selection_batches++;
    impl_->statistics_.proposal_fallbacks +=
        result.batch.fallback_events.size();
    impl_->statistics_.native_newton_iterations +=
        result.native_newton_iterations;
    impl_->statistics_.native_bisection_iterations +=
        result.native_bisection_iterations;
    impl_->statistics_.native_inverse_failures +=
        result.native_inverse_failures;
    impl_->enrichFallbackEvents(result.batch.fallback_events);
    return std::move(result.batch);
  }

  template <class BackendExecutionSpace>
  gpu::em::EmFinalStateBatchResult
  KokkosBackendInstanceImpl<BackendExecutionSpace>::generatePhotonFinalStatesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions,
      std::uint64_t const first_secondary_history_id) {
    impl_->requireInitialized();
    auto result = kokkos_detail::generatePhotonFinalStates(
        impl_->physics_, impl_->auxiliary_.photon_pair_lpm,
        impl_->config_.thinning, interactions, impl_->config_.random_seed,
        impl_->config_.shower_id, first_secondary_history_id,
        impl_->execution_);
    impl_->statistics_.final_state_batches++;
    impl_->statistics_.gpu_final_states += result.batch.gpu_interactions;
    impl_->statistics_.physical_secondaries_generated +=
        result.batch.secondaries.size();
    impl_->statistics_.photon_pair_final_states +=
        result.batch.photon_pair_interactions;
    impl_->statistics_.compton_final_states +=
        result.batch.compton_interactions;
    impl_->statistics_.photoelectric_final_states +=
        result.batch.photoelectric_interactions;
    impl_->statistics_.proposal_fallbacks += result.batch.fallback_events.size();
    if (result.first_interaction) {
      impl_->first_interaction_ = result.first_interaction;
    }
    impl_->enrichFallbackEvents(result.batch.fallback_events);
    return std::move(result.batch);
  }

  template <class BackendExecutionSpace>
  gpu::em::BremsFinalStateBatchResult
  KokkosBackendInstanceImpl<BackendExecutionSpace>::generateLeptonFinalStatesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions,
      std::uint64_t const first_secondary_history_id) {
    impl_->requireInitialized();
    auto result = kokkos_detail::generateLeptonFinalStates(
        impl_->auxiliary_.brems_lpm, impl_->config_.thinning, interactions,
        impl_->config_.random_seed, impl_->config_.shower_id,
        first_secondary_history_id, impl_->execution_);
    impl_->statistics_.final_state_batches++;
    impl_->statistics_.gpu_final_states += result.batch.gpu_interactions;
    impl_->statistics_.physical_secondaries_generated +=
        result.batch.secondaries.size();
    impl_->statistics_.brems_final_states +=
        result.batch.brems_interactions;
    impl_->statistics_.annihilation_final_states +=
        result.batch.annihilation_interactions;
    impl_->statistics_.ionization_final_states +=
        result.batch.ionization_interactions;
    impl_->statistics_.electron_pair_final_states +=
        result.batch.electron_pair_interactions;
    impl_->statistics_.brems_lpm_trials += result.batch.brems_lpm_trials;
    impl_->statistics_.brems_lpm_suppressions +=
        result.batch.brems_lpm_suppressions;
    impl_->statistics_.electron_pair_lpm_trials +=
        result.batch.electron_pair_lpm_trials;
    impl_->statistics_.electron_pair_lpm_suppressions +=
        result.batch.electron_pair_lpm_suppressions;
    impl_->statistics_.electron_pair_rejection_trials +=
        result.batch.electron_pair_rejection_trials;
    impl_->statistics_.electron_pair_zero_weight_samples +=
        result.batch.electron_pair_zero_weight_samples;
    impl_->statistics_.electron_pair_rejection_fallbacks +=
        result.batch.electron_pair_rejection_fallbacks;
    impl_->statistics_.electron_pair_envelope_violations +=
        result.batch.electron_pair_envelope_violations;
    impl_->statistics_.proposal_fallbacks +=
        result.batch.fallback_events.size();
    if (result.first_interaction) impl_->first_interaction_ = result.first_interaction;
    impl_->enrichFallbackEvents(result.batch.fallback_events);
    return std::move(result.batch);
  }

  template <class BackendExecutionSpace>
  gpu::radio::GpuRadioWaveforms KokkosBackendInstanceImpl<BackendExecutionSpace>::projectRadioForValidation(
      std::vector<gpu::em::LeptonTransportRecord> const& records) {
    impl_->requireInitialized();
    if (!impl_->radio_accumulator_.enabled())
      throw std::logic_error(
          "Kokkos radio validation requested while radio is disabled");
    using RecordView = Kokkos::View<gpu::em::LeptonTransportRecord*,
                                    typename BackendExecutionSpace::memory_space>;
    RecordView device_records("c8_kokkos_radio_validation_records",
                              records.size());
    auto mirror = Kokkos::create_mirror_view(device_records);
    for (std::size_t index = 0; index < records.size(); ++index)
      mirror(index) = records[index];
    Kokkos::deep_copy(impl_->execution_, device_records, mirror);
    impl_->radio_accumulator_.reset(impl_->execution_);
    impl_->radio_accumulator_.accumulateLeptonTracks(
        device_records, records.size(), impl_->execution_);
    auto result = impl_->radio_accumulator_.download(impl_->execution_);
    impl_->statistics_.radio = impl_->radio_accumulator_.statistics();
    impl_->refreshDeviceMemoryStatistics();
    return result;
  }

  template <class BackendExecutionSpace>
  void KokkosBackendInstanceImpl<BackendExecutionSpace>::beginShower(
      AcceleratedEmShowerConfig const& shower) {
    impl_->requireInitialized();
    if (!impl_->pending_photons_.empty() ||
        !impl_->pending_leptons_.empty())
      throw std::logic_error(
          "cannot begin a new Kokkos shower with pending cross-species particles");
    impl_->cooperative_wait_.resetStatistics();
    auto const static_statistics = impl_->statistics_;
    impl_->config_.random_seed = shower.random_seed;
    impl_->config_.shower_id = shower.shower_id;
    impl_->config_.thinning = shower.thinning;
    impl_->physics_context_.beginShower(
        shower.thinning, shower.random_seed, shower.shower_id,
        impl_->execution_);
    auto const ordinal = impl_->statistics_.shower_ordinal + 1;
    auto const static_bytes = impl_->statistics_.native_table_device_bytes;
    auto const all_table_bytes = impl_->statistics_.table_device_bytes;
    auto const table_hash = impl_->statistics_.native_table_hash;
    auto const auxiliary_hash = impl_->statistics_.auxiliary_cache_hash;
    auto const auxiliary_hit = impl_->statistics_.auxiliary_cache_hit;
    auto const proposal_version =
        impl_->statistics_.native_proposal_version;
    auto const interpolation_version =
        impl_->statistics_.native_cubic_interpolation_version;
    auto const native_nodes = impl_->statistics_.native_table_nodes;
    auto const cache_tables =
        impl_->statistics_.proposal_cache_table_count;
    auto const cache_hits =
        impl_->statistics_.proposal_cache_hit_count;
    auto const cache_all_hit =
        impl_->statistics_.proposal_cache_all_hit;
    impl_->statistics_ = {};
    impl_->statistics_.shower_ordinal = ordinal;
    impl_->statistics_.reused_for_shower = true;
    impl_->statistics_.physics_source = gpu::em::GpuPhysicsSource::ProposalNative;
    impl_->statistics_.accelerator_backend =
        static_statistics.accelerator_backend;
    impl_->statistics_.accelerator_device_name =
        static_statistics.accelerator_device_name;
    impl_->statistics_.accelerator_architecture =
        static_statistics.accelerator_architecture;
    impl_->statistics_.accelerator_driver_version =
        static_statistics.accelerator_driver_version;
    impl_->statistics_.accelerator_runtime_version =
        static_statistics.accelerator_runtime_version;
    impl_->statistics_.accelerator_compiler_version =
        static_statistics.accelerator_compiler_version;
    impl_->statistics_.accelerator_project_revision =
        static_statistics.accelerator_project_revision;
    impl_->statistics_.accelerator_device =
        static_statistics.accelerator_device;
    impl_->statistics_.accelerator_concurrency =
        static_statistics.accelerator_concurrency;
    impl_->statistics_.accelerator_host_threads =
        static_statistics.accelerator_host_threads;
    impl_->statistics_.accelerator_openmp =
        static_statistics.accelerator_openmp;
    impl_->statistics_.accelerator_gpu = static_statistics.accelerator_gpu;
    impl_->statistics_.static_host_to_device_bytes =
        static_statistics.static_host_to_device_bytes;
    impl_->statistics_.one_time_initialization_ms =
        static_statistics.one_time_initialization_ms;
    impl_->statistics_.tuning_cache_matched =
        static_statistics.tuning_cache_matched;
    impl_->statistics_.tuning_cache_required =
        static_statistics.tuning_cache_required;
    impl_->statistics_.tuning_cache_hash =
        static_statistics.tuning_cache_hash;
    impl_->statistics_.tuning_batch_size =
        static_statistics.tuning_batch_size;
    impl_->statistics_.tuning_chunk_size =
        static_statistics.tuning_chunk_size;
    impl_->statistics_.tuning_team_size =
        static_statistics.tuning_team_size;
    impl_->statistics_.tuning_track_tile_size =
        static_statistics.tuning_track_tile_size;
    impl_->statistics_.tuning_observer_tile_size =
        static_statistics.tuning_observer_tile_size;
    impl_->statistics_.tuning_device_queues =
        static_statistics.tuning_device_queues;
    impl_->statistics_.native_proposal_version = proposal_version;
    impl_->statistics_.native_cubic_interpolation_version =
        interpolation_version;
    impl_->statistics_.native_table_nodes = native_nodes;
    impl_->statistics_.proposal_cache_table_count = cache_tables;
    impl_->statistics_.proposal_cache_hit_count = cache_hits;
    impl_->statistics_.proposal_cache_all_hit = cache_all_hit;
    impl_->statistics_.maximum_resident_photon_batch =
        impl_->maximum_resident_batch_size_;
    impl_->statistics_.maximum_resident_lepton_batch =
        impl_->maximum_resident_batch_size_;
    impl_->statistics_.automatic_capacity_budget_bytes =
        static_statistics.automatic_capacity_budget_bytes;
    impl_->statistics_.automatic_capacity_planned_bytes =
        static_statistics.automatic_capacity_planned_bytes;
    impl_->statistics_.automatic_capacity_peak_bytes =
        static_statistics.automatic_capacity_peak_bytes;
    impl_->statistics_.automatic_capacity_allocator_reserve_bytes =
        static_statistics.automatic_capacity_allocator_reserve_bytes;
    impl_->statistics_.cross_species_queue_capacity_per_pid =
        impl_->maximum_pending_particles_;
    impl_->statistics_.native_table_hash = table_hash;
    impl_->statistics_.auxiliary_cache_hash = auxiliary_hash;
    impl_->statistics_.native_table_device_bytes = static_bytes;
    impl_->statistics_.table_device_bytes = all_table_bytes;
    impl_->statistics_.auxiliary_cache_hit = auxiliary_hit;
    impl_->statistics_.profile.enabled = impl_->profile_accumulator_.enabled();
    impl_->statistics_.profile.deterministic = true;
    impl_->statistics_.profile.bins =
        impl_->config_.profile_projection.output_bin_count;
    impl_->statistics_.profile.device_bytes =
        impl_->profile_projection_.deviceBytes() +
        impl_->profile_accumulator_.deviceBytes();
    impl_->profile_accumulator_.reset(
        shower.profile_fixed_point_weight_limit,
        shower.profile_fixed_point_energy_limit_GeV, impl_->execution_);
    impl_->radio_accumulator_.reset(impl_->execution_);
    impl_->pending_photons_.clear();
    impl_->pending_leptons_.clear();
    impl_->first_interaction_.reset();
    impl_->refreshDeviceMemoryStatistics();
  }

  template <class BackendExecutionSpace>
  bool KokkosBackendInstanceImpl<BackendExecutionSpace>::canTransport(
      gpu::em::EmParticleState const& particle) const {
    if (!impl_->initialized_) return false;
    if (particle.pid == static_cast<std::int32_t>(gpu::em::EmPid::Photon) ||
        gpu::em::isElectronOrPositronPid(particle.pid))
      return true;
    if (!gpu::em::isMuonPid(particle.pid) ||
        impl_->auxiliary_.has_muon_moliere == 0)
      return false;
    return std::any_of(
        impl_->host_table_.interaction_identities.begin(),
        impl_->host_table_.interaction_identities.end(),
        [&](auto const& entry) { return entry.pdg_id == particle.pid; });
  }

  template <class BackendExecutionSpace>
  bool KokkosBackendInstanceImpl<BackendExecutionSpace>::hasProposalTable() const {
    return impl_->initialized_;
  }

  template <class BackendExecutionSpace>
  std::size_t KokkosBackendInstanceImpl<BackendExecutionSpace>::minimumBatchSize() const {
    impl_->requireInitialized();
    return impl_->config_.min_batch_size;
  }

  template <class BackendExecutionSpace>
  std::size_t KokkosBackendInstanceImpl<BackendExecutionSpace>::maximumResidentPhotonBatchSize() const {
    impl_->requireInitialized();
    return impl_->maximum_resident_batch_size_;
  }

  template <class BackendExecutionSpace>
  std::size_t KokkosBackendInstanceImpl<BackendExecutionSpace>::maximumResidentLeptonBatchSize() const {
    impl_->requireInitialized();
    return impl_->maximum_resident_batch_size_;
  }

  template <class BackendExecutionSpace>
  std::size_t KokkosBackendInstanceImpl<BackendExecutionSpace>::maximumResidentInputBatchSize() const {
    impl_->requireInitialized();
    return impl_->maximum_resident_batch_size_;
  }

  template <class BackendExecutionSpace>
  std::size_t KokkosBackendInstanceImpl<BackendExecutionSpace>::pendingPhotonCount() const noexcept {
    return impl_->config_.resident_cross_species
               ? impl_->pending_photons_.size()
               : std::size_t{};
  }

  template <class BackendExecutionSpace>
  std::size_t KokkosBackendInstanceImpl<BackendExecutionSpace>::pendingLeptonCount() const noexcept {
    return impl_->config_.resident_cross_species
               ? impl_->pending_leptons_.size()
               : std::size_t{};
  }

  template <class BackendExecutionSpace>
  gpu::em::ResidentPhotonCascadeResult KokkosBackendInstanceImpl<BackendExecutionSpace>::runPhotonWavefront(
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id,
      std::size_t maximum_wavefronts,
      std::size_t minimum_resident_batch_size) {
    using namespace gpu::em;
    impl_->requireInitialized();
    if (particles.size() > impl_->maximum_resident_batch_size_)
      throw std::length_error(
          "Kokkos photon input exceeds the advertised resident batch limit");
    auto const retain_cross_species = impl_->config_.resident_cross_species;
    auto const pending_photon_input = std::min(
        retain_cross_species
            ? std::min(
                  impl_->pending_photons_.size(),
                  particles.size() < impl_->maximum_resident_batch_size_
                      ? impl_->maximum_resident_batch_size_ - particles.size()
                      : std::size_t{})
            : std::size_t{}, impl_->cooperative_pending_limit_);
    auto const pending_leptons_before = impl_->pending_leptons_.size();
    auto resident = kokkos_detail::runResidentPhotonCascade(
        impl_->physics_context_.deviceView(), particles,
        first_secondary_history_id, maximum_wavefronts,
        minimum_resident_batch_size, impl_->first_interaction_,
        impl_->profile_projection_.enabled() &&
            !impl_->profile_accumulator_.enabled(),
        impl_->profile_projection_.deviceView(),
        impl_->profile_accumulator_.enabled()
            ? &impl_->profile_accumulator_
            : nullptr,
        &impl_->photon_workspace_,
        retain_cross_species ? &impl_->pending_photons_ : nullptr,
        retain_cross_species ? &impl_->pending_leptons_ : nullptr,
        pending_photon_input, impl_->maximum_pending_particles_,
        impl_->execution_, impl_->config_.diagnostic_interaction_records,
        impl_->tuning_.parameters.chunk_size,
        impl_->residentMemoryGate(), &impl_->cooperative_wait_);
    if (resident.input_particles !=
        pending_photon_input + particles.size())
      throw std::logic_error(
          "Kokkos photon pending-input accounting disagrees with the router");
    if (retain_cross_species &&
        !resident.electromagnetic_secondaries.empty())
      throw std::logic_error(
          "resident Kokkos photon cascade returned retained leptons twice");
    auto const pending_leptons_added =
        impl_->pending_leptons_.size() - pending_leptons_before;
    impl_->statistics_.physical_host_to_device_bytes +=
        particles.size() * sizeof(EmParticleState);
    impl_->statistics_.cross_species_particles_kept_on_device +=
        pending_leptons_added;
    impl_->statistics_.cross_species_device_to_device_bytes +=
        (pending_photon_input + pending_leptons_added) *
        sizeof(EmParticleState);
    if (retain_cross_species && !resident.cpu_spill_particles.empty()) {
      ++impl_->statistics_.cross_species_host_spills;
      impl_->statistics_.cross_species_particles_spilled_to_cpu +=
          resident.cpu_spill_particles.size();
    }
    impl_->refreshDeviceMemoryStatistics();
    impl_->statistics_.physical_photon_wavefronts += resident.wavefronts;
    impl_->statistics_.wavefront_bucketing_batches +=
        resident.wavefront_bucketing_batches;
    impl_->statistics_.wavefront_bucketing_particles +=
        resident.wavefront_bucketing_particles;
    impl_->statistics_.wavefront_bucketing_small_batches +=
        resident.wavefront_bucketing_small_batches;
    impl_->statistics_.wavefront_bucketing_small_particles +=
        resident.wavefront_bucketing_small_particles;
    impl_->statistics_.native_newton_iterations +=
        resident.native_newton_iterations;
    impl_->statistics_.native_bisection_iterations +=
        resident.native_bisection_iterations;
    impl_->statistics_.native_inverse_failures +=
        resident.native_inverse_failures;
    impl_->statistics_.interaction_selection_batches += resident.wavefronts;
    impl_->statistics_.photon_transport_batches += resident.wavefronts;
    impl_->statistics_.interactions_selected += resident.selected_interactions;
    impl_->statistics_.final_state_batches +=
        resident.interaction_bearing_wavefronts;
    impl_->statistics_.photon_transport_interactions +=
        resident.interaction_vertices;
    impl_->statistics_.photon_transport_boundaries +=
        resident.layer_boundaries;
    impl_->statistics_.photon_transport_cuts += resident.particle_cuts;
    impl_->statistics_.photon_pair_lpm_suppressions +=
        resident.process_statistics.photon_pair_lpm_suppressions;
    impl_->statistics_.proposal_fallbacks += resident.fallback_events.size();
    auto const& process = resident.process_statistics;
    impl_->statistics_.gpu_final_states += process.gpu_final_states;
    impl_->statistics_.physical_secondaries_generated +=
        process.physical_secondaries_generated;
    impl_->statistics_.photon_pair_final_states +=
        process.photon_pair_final_states;
    impl_->statistics_.compton_final_states += process.compton_final_states;
    impl_->statistics_.photoelectric_final_states +=
        process.photoelectric_final_states;
    impl_->statistics_.photon_pair_lpm_trials +=
        process.photon_pair_lpm_trials;
    impl_->statistics_.thinning_hillas_vertices +=
        process.thinning_hillas_vertices;
    impl_->statistics_.thinning_statistical_vertices +=
        process.thinning_statistical_vertices;
    impl_->statistics_.thinning_particles_discarded +=
        process.thinning_particles_discarded;
    for (auto const& observation : resident.observations) {
      if (observation.status == ObservationStatus::ReachedObservationSurface)
        impl_->statistics_.photon_transport_observations++;
      else
        impl_->statistics_.photon_transport_escapes++;
    }
    impl_->enrichFallbackEvents(resident.fallback_events);
    return resident;

    // Retained temporarily as a host-visible staging oracle.  Production and
    // validation calls execute the fused resident Kokkos path above.
    if (first_secondary_history_id == 0 || maximum_wavefronts == 0 ||
        minimum_resident_batch_size == 0)
      throw std::invalid_argument(
          "Kokkos photon wavefront requires nonzero history/wavefront limits");
    for (auto const& particle : particles) {
      if (particle.pid != static_cast<std::int32_t>(EmPid::Photon) ||
          !std::isfinite(particle.energy_GeV) || particle.energy_GeV < 0. ||
          !std::isfinite(particle.weight) || particle.weight < 0. ||
          particle.history_id == 0)
        throw std::invalid_argument(
            "Kokkos photon wavefront accepts only valid photons");
    }

    ResidentPhotonCascadeResult result{};
    result.input_particles = particles.size();
    if (particles.empty()) {
      result.completed = true;
      return result;
    }
    std::vector<EmParticleState> current = particles;
    auto next_history_id = first_secondary_history_id;
    result.peak_resident_photons = current.size();
    while (!current.empty() && result.wavefronts < maximum_wavefronts) {
      if (result.wavefronts != 0 &&
          current.size() < minimum_resident_batch_size) {
        result.below_minimum_batch_checkpoint = true;
        break;
      }
      auto selected = selectInteractionsForValidation(current);
      result.selected_interactions += selected.interactions.size();
      auto transported = transportPhotonsForValidation(selected.interactions);
      std::vector<EmInteractionRecord> vertices;
      vertices.reserve(transported.records.size());
      for (auto const& record : transported.records) {
        if (record.limit == PhotonTransportLimit::Interaction)
          vertices.push_back(record.interaction);
      }
      result.interaction_bearing_wavefronts += !vertices.empty();
      auto final_states = generatePhotonFinalStatesForValidation(
          vertices, next_history_id);
      if (final_states.secondaries.size() >
          std::numeric_limits<std::uint64_t>::max() - next_history_id)
        throw std::overflow_error(
            "Kokkos photon secondary history ID overflow");
      next_history_id += final_states.secondaries.size();

      std::vector<std::optional<EmParticleState>> next_by_source(
          current.size());
      std::vector<std::optional<ObservationRecord>> observation_by_source(
          current.size());
      auto claim_next = [&](std::uint64_t source,
                            EmParticleState const& particle) {
        if (source >= next_by_source.size() || next_by_source[source])
          throw std::runtime_error(
              "Kokkos photon endpoint ownership is inconsistent");
        next_by_source[source] = particle;
      };
      for (auto const& record : transported.records) {
        result.transport_records++;
        result.step_records.push_back(record);
        switch (record.limit) {
        case PhotonTransportLimit::Interaction:
          result.interaction_vertices++;
          break;
        case PhotonTransportLimit::LayerBoundary:
          result.layer_boundaries++;
          claim_next(record.input_index, record.end);
          break;
        case PhotonTransportLimit::ObservationSurface:
        case PhotonTransportLimit::EscapedEnvironment: {
          if (record.input_index >= observation_by_source.size() ||
              observation_by_source[record.input_index])
            throw std::runtime_error(
                "Kokkos photon observation ownership is inconsistent");
          ObservationRecord observation{};
          observation.particle = record.end;
          observation.status =
              record.limit == PhotonTransportLimit::ObservationSurface
                  ? ObservationStatus::ReachedObservationSurface
                  : ObservationStatus::EscapedEnvironment;
          observation_by_source[record.input_index] = observation;
          break;
        }
        case PhotonTransportLimit::ParticleCut:
          result.particle_cuts++;
          if (record.observation_surface_reached_before_cut != 0) {
            ObservationRecord observation{};
            observation.particle = record.end;
            observation.status = ObservationStatus::ReachedObservationSurface;
            observation_by_source[record.input_index] = observation;
          }
          break;
        }
      }
      for (auto const& suppression : final_states.lpm_suppressed) {
        claim_next(suppression.input_index, suppression.particle);
        result.lpm_suppressions++;
      }
      for (auto const& continuation : final_states.continuations)
        claim_next(continuation.input_index, continuation.particle);
      for (auto const& record : final_states.final_state_records) {
        result.final_state_records.push_back(record);
        for (std::uint32_t child = 0; child < record.secondary_count; ++child) {
          auto const& particle =
              final_states.secondaries[record.secondary_offset + child];
          if (particle.pid == static_cast<std::int32_t>(EmPid::Photon)) {
            if (record.process_id == ComptonProcessId)
              claim_next(record.input_index, particle);
          } else {
            result.electromagnetic_secondaries.push_back(particle);
          }
        }
      }
      result.fallback_events.insert(result.fallback_events.end(),
                                    selected.fallback_events.begin(),
                                    selected.fallback_events.end());
      result.fallback_events.insert(result.fallback_events.end(),
                                    transported.fallback_events.begin(),
                                    transported.fallback_events.end());
      result.fallback_events.insert(result.fallback_events.end(),
                                    final_states.fallback_events.begin(),
                                    final_states.fallback_events.end());
      std::vector<EmParticleState> next;
      next.reserve(current.size());
      for (std::size_t source = 0; source < current.size(); ++source) {
        if (next_by_source[source]) next.push_back(*next_by_source[source]);
        if (observation_by_source[source])
          result.observations.push_back(*observation_by_source[source]);
      }
      current = std::move(next);
      result.peak_resident_photons =
          std::max(result.peak_resident_photons, current.size());
      result.wavefronts++;
      impl_->statistics_.physical_photon_wavefronts++;
    }
    result.completed = current.empty();
    if (!result.completed) result.remaining_photons = std::move(current);
    return result;
  }

  template <class BackendExecutionSpace>
  gpu::em::ResidentLeptonCascadeResult KokkosBackendInstanceImpl<BackendExecutionSpace>::runLeptonWavefront(
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id,
      std::size_t maximum_wavefronts,
      std::uint64_t secondary_history_id_limit_exclusive,
      std::size_t minimum_resident_batch_size) {
    using namespace gpu::em;
    impl_->requireInitialized();
    if (particles.size() > impl_->maximum_resident_batch_size_)
      throw std::length_error(
          "Kokkos lepton input exceeds the advertised resident batch limit");
    auto const retain_cross_species = impl_->config_.resident_cross_species;
    auto const pending_lepton_input = std::min(
        retain_cross_species
            ? std::min(
                  impl_->pending_leptons_.size(),
                  particles.size() < impl_->maximum_resident_batch_size_
                      ? impl_->maximum_resident_batch_size_ - particles.size()
                      : std::size_t{})
            : std::size_t{}, impl_->cooperative_pending_limit_);
    auto const pending_photons_before = impl_->pending_photons_.size();
    auto resident = kokkos_detail::runResidentLeptonCascade(
        impl_->physics_context_.deviceView(),
        impl_->auxiliary_.brems_lpm.lepton_mass_MeV / 1000.,
        impl_->auxiliary_.electron_moliere.component_count <= 4u, particles,
        first_secondary_history_id, maximum_wavefronts,
        secondary_history_id_limit_exclusive, minimum_resident_batch_size,
        impl_->first_interaction_,
        impl_->profile_projection_.enabled() &&
            !impl_->profile_accumulator_.enabled(),
        impl_->profile_projection_.deviceView(),
        impl_->profile_accumulator_.enabled()
            ? &impl_->profile_accumulator_
            : nullptr,
        impl_->radio_accumulator_.enabled()
            ? &impl_->radio_accumulator_
            : nullptr,
        &impl_->lepton_workspace_,
        retain_cross_species ? &impl_->pending_leptons_ : nullptr,
        retain_cross_species ? &impl_->pending_photons_ : nullptr,
        pending_lepton_input, impl_->maximum_pending_particles_,
        impl_->execution_, impl_->config_.diagnostic_interaction_records,
        impl_->tuning_.parameters.chunk_size,
        impl_->residentMemoryGate(),
        impl_->automatic_gpu_capacity_ ? impl_->maximum_resident_batch_size_ : 0,
        &impl_->cooperative_wait_);
    if (resident.input_particles !=
        pending_lepton_input + particles.size())
      throw std::logic_error(
          "Kokkos lepton pending-input accounting disagrees with the router");
    if (retain_cross_species && !resident.generated_photons.empty())
      throw std::logic_error(
          "resident Kokkos lepton cascade returned retained photons twice");
    auto const pending_photons_added =
        impl_->pending_photons_.size() - pending_photons_before;
    impl_->statistics_.physical_host_to_device_bytes +=
        particles.size() * sizeof(EmParticleState);
    impl_->statistics_.cross_species_particles_kept_on_device +=
        pending_photons_added;
    impl_->statistics_.cross_species_device_to_device_bytes +=
        (pending_lepton_input + pending_photons_added) *
        sizeof(EmParticleState);
    if (retain_cross_species && !resident.cpu_spill_particles.empty()) {
      ++impl_->statistics_.cross_species_host_spills;
      impl_->statistics_.cross_species_particles_spilled_to_cpu +=
          resident.cpu_spill_particles.size();
    }
    impl_->refreshDeviceMemoryStatistics();
    impl_->statistics_.physical_lepton_wavefronts += resident.wavefronts;
    impl_->statistics_.wavefront_bucketing_batches +=
        resident.wavefront_bucketing_batches;
    impl_->statistics_.wavefront_bucketing_particles +=
        resident.wavefront_bucketing_particles;
    impl_->statistics_.wavefront_bucketing_small_batches +=
        resident.wavefront_bucketing_small_batches;
    impl_->statistics_.wavefront_bucketing_small_particles +=
        resident.wavefront_bucketing_small_particles;
    impl_->statistics_.interaction_selection_batches += resident.wavefronts;
    impl_->statistics_.lepton_transport_batches += resident.wavefronts;
    impl_->statistics_.lepton_vertex_selection_batches +=
        resident.interaction_bearing_wavefronts;
    impl_->statistics_.lepton_vertex_interactions_selected +=
        resident.vertex_interactions_selected;
    impl_->statistics_.lepton_vertex_no_interaction_continuations +=
        resident.vertex_no_interaction_continuations;
    impl_->statistics_.final_state_batches +=
        resident.final_state_bearing_wavefronts;
    impl_->statistics_.interactions_selected += resident.selected_interactions;
    impl_->statistics_.proposal_fallbacks += resident.fallback_events.size();
    auto const& process = resident.process_statistics;
    impl_->statistics_.gpu_final_states += process.gpu_final_states;
    impl_->statistics_.physical_secondaries_generated +=
        process.physical_secondaries_generated;
    impl_->statistics_.brems_final_states += process.brems_final_states;
    impl_->statistics_.annihilation_final_states +=
        process.annihilation_final_states;
    impl_->statistics_.ionization_final_states +=
        process.ionization_final_states;
    impl_->statistics_.electron_pair_final_states +=
        process.electron_pair_final_states;
    impl_->statistics_.brems_lpm_trials += process.brems_lpm_trials;
    impl_->statistics_.brems_lpm_suppressions +=
        process.brems_lpm_suppressions;
    impl_->statistics_.electron_pair_lpm_trials +=
        process.electron_pair_lpm_trials;
    impl_->statistics_.electron_pair_lpm_suppressions +=
        process.electron_pair_lpm_suppressions;
    impl_->statistics_.electron_pair_rejection_trials +=
        process.electron_pair_rejection_trials;
    impl_->statistics_.electron_pair_zero_weight_samples +=
        process.electron_pair_zero_weight_samples;
    impl_->statistics_.electron_pair_rejection_fallbacks +=
        process.electron_pair_rejection_fallbacks;
    impl_->statistics_.electron_pair_envelope_violations +=
        process.electron_pair_envelope_violations;
    impl_->statistics_.thinning_hillas_vertices +=
        process.thinning_hillas_vertices;
    impl_->statistics_.thinning_statistical_vertices +=
        process.thinning_statistical_vertices;
    impl_->statistics_.thinning_particles_discarded +=
        process.thinning_particles_discarded;
    impl_->statistics_.moliere_trials += process.moliere_trials;
    impl_->statistics_.moliere_deflections += process.moliere_deflections;
    impl_->statistics_.moliere_zero_deflections +=
        process.moliere_zero_deflections;
    impl_->statistics_.moliere_newton_iterations +=
        process.moliere_newton_iterations;
    impl_->statistics_.moliere_max_newton_iterations = std::max(
        impl_->statistics_.moliere_max_newton_iterations,
        process.moliere_max_newton_iterations);
    auto const& limits = process.lepton_transport_limits;
    impl_->statistics_.lepton_transport_interaction_candidates += limits[0];
    impl_->statistics_.lepton_transport_continuous_steps += limits[1];
    impl_->statistics_.lepton_transport_cuts += limits[2];
    impl_->statistics_.lepton_transport_boundaries += limits[3];
    impl_->statistics_.lepton_transport_observations += limits[4];
    impl_->statistics_.lepton_transport_escapes += limits[5];
    impl_->statistics_.lepton_transport_magnetic_steps += limits[6];
    impl_->statistics_.lepton_transport_decay_candidates += limits[7];
    impl_->enrichFallbackEvents(resident.fallback_events);
    return resident;

    // Retained temporarily as a host-visible staging oracle. Production and
    // validation calls execute the fused resident Kokkos path above.
    if (first_secondary_history_id == 0 || maximum_wavefronts == 0 ||
        secondary_history_id_limit_exclusive <=
            first_secondary_history_id ||
        minimum_resident_batch_size == 0)
      throw std::invalid_argument(
          "Kokkos lepton wavefront requires valid history/wavefront limits");
    for (auto const& particle : particles) {
      if (!isChargedLeptonPid(particle.pid) ||
          !std::isfinite(particle.energy_GeV) ||
          !(particle.energy_GeV > 0.) || !std::isfinite(particle.weight) ||
          particle.weight < 0. || particle.history_id == 0)
        throw std::invalid_argument(
            "Kokkos lepton wavefront accepts only valid charged leptons");
    }

    ResidentLeptonCascadeResult result{};
    result.input_particles = particles.size();
    if (particles.empty()) {
      result.completed = true;
      return result;
    }
    std::vector<EmParticleState> current = particles;
    auto next_history_id = first_secondary_history_id;
    result.peak_resident_leptons = current.size();
    while (!current.empty() && result.wavefronts < maximum_wavefronts) {
      if (result.wavefronts != 0 &&
          current.size() < minimum_resident_batch_size) {
        result.below_minimum_batch_checkpoint = true;
        break;
      }
      auto const remaining_history_ids =
          secondary_history_id_limit_exclusive - next_history_id;
      if (current.size() > remaining_history_ids / 3) {
        result.history_range_exhausted = true;
        break;
      }
      auto selected = selectInteractionsForValidation(current);
      auto transported = transportLeptonsForValidation(selected.interactions);
      std::vector<EmInteractionRecord> candidates;
      candidates.reserve(transported.records.size());
      for (auto const& record : transported.records) {
        if (record.limit == LeptonTransportLimit::InteractionCandidate)
          candidates.push_back(record.interaction);
      }
      auto vertices = selectLeptonVerticesForValidation(candidates);
      auto final_states = generateLeptonFinalStatesForValidation(
          vertices.interactions, next_history_id);
      if (final_states.secondaries.size() >
          secondary_history_id_limit_exclusive - next_history_id) {
        result.history_range_exhausted = true;
        break;
      }
      result.selected_interactions += selected.interactions.size();
      result.interaction_bearing_wavefronts += !candidates.empty();
      result.vertex_interactions_selected += vertices.interactions.size();
      result.vertex_no_interaction_continuations +=
          vertices.continuations.size();
      result.final_state_bearing_wavefronts +=
          !vertices.interactions.empty();
      next_history_id += final_states.secondaries.size();

      struct SourceChildren {
        std::optional<EmParticleState> leptons[3]{};
      };
      std::vector<SourceChildren> next_by_source(current.size());
      std::vector<std::optional<ObservationRecord>> observation_by_source(
          current.size());
      std::vector<std::optional<EmParticleState>> decay_by_source(
          current.size());
      auto claim_lepton = [&](std::uint64_t source,
                              EmParticleState const& particle) {
        if (source >= next_by_source.size())
          throw std::runtime_error("Kokkos lepton source index is invalid");
        auto& slots = next_by_source[source].leptons;
        for (auto& slot : slots) {
          if (!slot) {
            slot = particle;
            return;
          }
        }
        throw std::runtime_error(
            "Kokkos lepton source generated more than three continuations");
      };
      for (auto const& record : transported.records) {
        result.transport_records++;
        result.step_records.push_back(record);
        switch (record.limit) {
        case LeptonTransportLimit::ContinuousStep:
        case LeptonTransportLimit::LayerBoundary:
        case LeptonTransportLimit::MagneticStep:
          claim_lepton(record.input_index, record.end);
          break;
        case LeptonTransportLimit::ObservationSurface:
        case LeptonTransportLimit::EscapedEnvironment: {
          ObservationRecord observation{};
          observation.particle = record.end;
          observation.status =
              record.limit == LeptonTransportLimit::ObservationSurface
                  ? ObservationStatus::ReachedObservationSurface
                  : ObservationStatus::EscapedEnvironment;
          observation_by_source[record.input_index] = observation;
          break;
        }
        case LeptonTransportLimit::ParticleCut:
          if (record.observation_surface_reached_before_cut != 0) {
            ObservationRecord observation{};
            observation.particle = record.end;
            observation.status = ObservationStatus::ReachedObservationSurface;
            observation_by_source[record.input_index] = observation;
          }
          break;
        case LeptonTransportLimit::DecayCandidate:
          decay_by_source[record.input_index] = record.end;
          break;
        case LeptonTransportLimit::InteractionCandidate:
          result.interaction_vertices++;
          break;
        }
      }
      for (auto const& continuation : vertices.continuations)
        claim_lepton(continuation.input_index, continuation.particle);
      for (auto const& continuation : final_states.continuations)
        claim_lepton(continuation.input_index, continuation.particle);
      for (auto const& suppression : final_states.lpm_suppressed) {
        claim_lepton(suppression.input_index, suppression.particle);
        result.lpm_suppressions++;
      }
      for (auto const& record : final_states.final_state_records) {
        result.final_state_records.push_back(record);
        for (std::uint32_t child = 0; child < record.secondary_count; ++child) {
          auto const& particle =
              final_states.secondaries[record.secondary_offset + child];
          if (isChargedLeptonPid(particle.pid))
            claim_lepton(record.input_index, particle);
          else if (particle.pid == static_cast<std::int32_t>(EmPid::Photon))
            result.generated_photons.push_back(particle);
          else
            throw std::runtime_error(
                "Kokkos lepton final state generated an unsupported PID");
        }
      }
      result.fallback_events.insert(result.fallback_events.end(),
                                    selected.fallback_events.begin(),
                                    selected.fallback_events.end());
      result.fallback_events.insert(result.fallback_events.end(),
                                    transported.fallback_events.begin(),
                                    transported.fallback_events.end());
      result.fallback_events.insert(result.fallback_events.end(),
                                    vertices.fallback_events.begin(),
                                    vertices.fallback_events.end());
      result.fallback_events.insert(result.fallback_events.end(),
                                    final_states.fallback_events.begin(),
                                    final_states.fallback_events.end());
      std::vector<EmParticleState> next;
      next.reserve(3 * current.size());
      for (std::size_t source = 0; source < current.size(); ++source) {
        for (auto const& child : next_by_source[source].leptons)
          if (child) next.push_back(*child);
        if (observation_by_source[source])
          result.observations.push_back(*observation_by_source[source]);
        if (decay_by_source[source])
          result.decay_candidates.push_back(*decay_by_source[source]);
      }
      current = std::move(next);
      result.peak_resident_leptons =
          std::max(result.peak_resident_leptons, current.size());
      result.wavefronts++;
      impl_->statistics_.physical_lepton_wavefronts++;
    }
    result.secondary_history_ids_used = next_history_id -
                                        first_secondary_history_id;
    result.completed = current.empty();
    if (!result.completed) result.remaining_leptons = std::move(current);
    return result;
  }

  template <class BackendExecutionSpace>
  BackendCapabilities KokkosBackendInstanceImpl<BackendExecutionSpace>::capabilities() const {
    auto const ready = impl_->initialized_;
    return {impl_->runtime_.info().kind,
            ready,
            ready,
            ready,
            ready && impl_->auxiliary_.has_muon_moliere != 0,
            ready && impl_->profile_accumulator_.enabled(),
            ready && impl_->radio_accumulator_.enabled(),
            ready && impl_->radio_accumulator_.enabled(),
            true};
  }

  template <class BackendExecutionSpace>
  AcceleratedEmStatistics const& KokkosBackendInstanceImpl<BackendExecutionSpace>::statistics() const {
    return impl_->statistics_;
  }

  template <class BackendExecutionSpace>
  bool KokkosBackendInstanceImpl<BackendExecutionSpace>::gpuProfileEnabled() const noexcept {
    return impl_->profile_accumulator_.enabled();
  }

  template <class BackendExecutionSpace>
  gpu::em::GpuProfileResult KokkosBackendInstanceImpl<BackendExecutionSpace>::downloadProfile() {
    impl_->requireInitialized();
    auto result = impl_->profile_accumulator_.download(impl_->execution_);
    impl_->statistics_.profile.steps = result.steps;
    impl_->statistics_.profile.deposited_steps = result.deposited_steps;
    impl_->statistics_.profile.fixed_point_overflows =
        result.fixed_point_overflows;
    impl_->statistics_.profile.invalid_records = result.invalid_records;
    impl_->statistics_.profile.device_to_host_bytes +=
        impl_->profile_accumulator_.deviceBytes();
    return result;
  }

  template <class BackendExecutionSpace>
  bool KokkosBackendInstanceImpl<BackendExecutionSpace>::gpuRadioEnabled() const noexcept {
    return impl_->radio_accumulator_.enabled();
  }

  template <class BackendExecutionSpace>
  gpu::radio::GpuRadioWaveforms KokkosBackendInstanceImpl<BackendExecutionSpace>::downloadRadioWaveforms() {
    impl_->requireInitialized();
    auto result = impl_->radio_accumulator_.download(impl_->execution_);
    impl_->statistics_.radio = impl_->radio_accumulator_.statistics();
    impl_->refreshDeviceMemoryStatistics();
    return result;
  }

  template <class BackendExecutionSpace>
  std::optional<gpu::em::GpuFirstInteractionSnapshot>
  KokkosBackendInstanceImpl<BackendExecutionSpace>::downloadFirstInteractionSnapshot() {
    return impl_->first_interaction_;
  }

  template <class BackendExecutionSpace>
  KokkosRuntimeInfo const& KokkosBackendInstanceImpl<BackendExecutionSpace>::runtimeInfo() const noexcept {
    return impl_->runtime_.info();
  }

} // namespace corsika::accelerator::em::detail
