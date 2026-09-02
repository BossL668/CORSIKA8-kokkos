/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/KokkosEmBackend.hpp>
#include <corsika/accelerator/em/KokkosTuningCache.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
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
#include <corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp>
#include <corsika/gpu/em/tables/Sha256.hpp>

namespace corsika::accelerator::em {

#if defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
  using BackendExecutionSpace = Kokkos::OpenMP;
#elif defined(CORSIKA8_KOKKOS_BACKEND_CUDA)
  using BackendExecutionSpace = Kokkos::Cuda;
#elif defined(CORSIKA8_KOKKOS_BACKEND_HIP)
  using BackendExecutionSpace = Kokkos::HIP;
#elif defined(CORSIKA8_KOKKOS_BACKEND_SYCL)
  using BackendExecutionSpace = Kokkos::SYCL;
#else
#error "Exactly one Kokkos backend must be selected"
#endif

  class KokkosEmBackend::Impl {
  public:
    explicit Impl(KokkosRuntimeConfig const& runtime_config)
        : runtime_{runtime_config} {}

    void requireInitialized() const {
      if (!initialized_) {
        throw std::logic_error("Kokkos EM backend is not initialized");
      }
    }

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
    gpu::em::tables::ProposalNativeTableSet host_table_{};
    gpu::em::tables::ProposalNativeAuxData auxiliary_{};
    gpu::em::EnvironmentSnapshot environment_{};
    gpu::em::GpuEmConfig config_{};
    gpu::em::GpuEmStatistics statistics_{};
    gpu::em::tables::FlatRateTableView physics_{};
    std::optional<gpu::em::GpuFirstInteractionSnapshot> first_interaction_{};
    std::uint64_t proposal_medium_hash_{};
    std::vector<std::pair<std::int32_t, std::uint64_t>> interaction_hashes_{};
    bool initialized_{};
    std::size_t maximum_resident_batch_size_{};
  };

  KokkosEmBackend::KokkosEmBackend(KokkosRuntimeConfig const& config)
      : impl_(std::make_unique<Impl>(config)) {
    impl_->runtime_config_ = config;
  }
  KokkosEmBackend::~KokkosEmBackend() = default;
  KokkosEmBackend::KokkosEmBackend(KokkosEmBackend&&) noexcept = default;
  KokkosEmBackend& KokkosEmBackend::operator=(KokkosEmBackend&&) noexcept =
      default;

  void KokkosEmBackend::initialize(
      gpu::em::EnvironmentSnapshot const& environment,
      gpu::em::tables::ProposalNativeTableSet const& table,
      gpu::em::tables::ProposalNativeAuxData const& auxiliary,
      gpu::em::GpuEmConfig const& config) {
    if (impl_->initialized_) {
      throw std::logic_error("Kokkos EM backend is already initialized");
    }
    if (config.physics_source != gpu::em::GpuPhysicsSource::ProposalNative) {
      throw std::invalid_argument(
          "Kokkos EM backend supports proposal-native physics only");
    }
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
    // This is a routing checkpoint, not a preallocation.  Kokkos Views are
    // still sized to the active wavefront.  A conservative cap prevents one
    // unusually large shower front from exhausting a device before the
    // portable memory-pool implementation lands.
    impl_->maximum_resident_batch_size_ =
        impl_->runtime_.info().gpu ? std::size_t{262144}
                                  : std::size_t{65536};
    // The tuning batch is an execution/workspace sizing hint.  It must not
    // replace the router's minimum resident-wavefront checkpoint: changing
    // that checkpoint changes when histories return to the host stack and can
    // therefore change history-id allocation and the Philox stream.
    impl_->maximum_resident_batch_size_ = std::max(
        {impl_->maximum_resident_batch_size_, impl_->config_.min_batch_size,
         impl_->tuning_.parameters.batch_size});
    impl_->statistics_.maximum_resident_photon_batch =
        impl_->maximum_resident_batch_size_;
    impl_->statistics_.maximum_resident_lepton_batch =
        impl_->maximum_resident_batch_size_;
    impl_->initialized_ = true;
  }

  gpu::em::EmInteractionBatchResult
  KokkosEmBackend::selectInteractionsForValidation(
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

  gpu::em::PhotonTransportBatchResult
  KokkosEmBackend::transportPhotonsForValidation(
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

  gpu::em::LeptonTransportBatchResult
  KokkosEmBackend::transportLeptonsForValidation(
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

  gpu::em::LeptonVertexSelectionBatchResult
  KokkosEmBackend::selectLeptonVerticesForValidation(
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

  gpu::em::EmFinalStateBatchResult
  KokkosEmBackend::generatePhotonFinalStatesForValidation(
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

  gpu::em::BremsFinalStateBatchResult
  KokkosEmBackend::generateLeptonFinalStatesForValidation(
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

  gpu::radio::GpuRadioWaveforms KokkosEmBackend::projectRadioForValidation(
      std::vector<gpu::em::LeptonTransportRecord> const& records) {
    impl_->requireInitialized();
    if (!impl_->radio_accumulator_.enabled())
      throw std::logic_error(
          "Kokkos radio validation requested while radio is disabled");
    using RecordView = Kokkos::View<gpu::em::LeptonTransportRecord*,
                                    BackendExecutionSpace::memory_space>;
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
    return result;
  }

  void KokkosEmBackend::beginShower(
      AcceleratedEmShowerConfig const& shower) {
    impl_->requireInitialized();
    auto const static_statistics = impl_->statistics_;
    impl_->config_.random_seed = shower.random_seed;
    impl_->config_.shower_id = shower.shower_id;
    impl_->config_.thinning = shower.thinning;
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
    impl_->first_interaction_.reset();
  }

  bool KokkosEmBackend::canTransport(
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
  bool KokkosEmBackend::hasProposalTable() const {
    return impl_->initialized_;
  }
  std::size_t KokkosEmBackend::minimumBatchSize() const {
    impl_->requireInitialized();
    return impl_->config_.min_batch_size;
  }
  std::size_t KokkosEmBackend::maximumResidentPhotonBatchSize() const {
    impl_->requireInitialized();
    return impl_->maximum_resident_batch_size_;
  }
  std::size_t KokkosEmBackend::maximumResidentLeptonBatchSize() const {
    impl_->requireInitialized();
    return impl_->maximum_resident_batch_size_;
  }
  std::size_t KokkosEmBackend::maximumResidentInputBatchSize() const {
    impl_->requireInitialized();
    return impl_->maximum_resident_batch_size_;
  }
  std::size_t KokkosEmBackend::pendingPhotonCount() const noexcept { return 0; }
  std::size_t KokkosEmBackend::pendingLeptonCount() const noexcept { return 0; }

  gpu::em::ResidentPhotonCascadeResult KokkosEmBackend::runPhotonWavefront(
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id,
      std::size_t maximum_wavefronts,
      std::size_t minimum_resident_batch_size) {
    using namespace gpu::em;
    impl_->requireInitialized();
    auto resident = kokkos_detail::runResidentPhotonCascade(
        impl_->physics_, impl_->auxiliary_.photon_pair_lpm,
        impl_->config_.thinning, impl_->environment_, particles,
        impl_->config_.random_seed, impl_->config_.shower_id,
        first_secondary_history_id, maximum_wavefronts,
        minimum_resident_batch_size, impl_->first_interaction_,
        impl_->profile_projection_.enabled() &&
            !impl_->profile_accumulator_.enabled(),
        impl_->profile_projection_.deviceView(),
        impl_->profile_accumulator_.enabled()
            ? &impl_->profile_accumulator_
            : nullptr,
        impl_->execution_);
    impl_->statistics_.physical_photon_wavefronts += resident.wavefronts;
    impl_->statistics_.interaction_selection_batches += resident.wavefronts;
    impl_->statistics_.photon_transport_batches += resident.wavefronts;
    impl_->statistics_.interactions_selected += resident.transport_records;
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
      auto transported = transportPhotonsForValidation(selected.interactions);
      std::vector<EmInteractionRecord> vertices;
      vertices.reserve(transported.records.size());
      for (auto const& record : transported.records) {
        if (record.limit == PhotonTransportLimit::Interaction)
          vertices.push_back(record.interaction);
      }
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
  gpu::em::ResidentLeptonCascadeResult KokkosEmBackend::runLeptonWavefront(
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id,
      std::size_t maximum_wavefronts,
      std::uint64_t secondary_history_id_limit_exclusive,
      std::size_t minimum_resident_batch_size) {
    using namespace gpu::em;
    impl_->requireInitialized();
    auto resident = kokkos_detail::runResidentLeptonCascade(
        impl_->physics_, impl_->environment_,
        impl_->auxiliary_.electron_moliere, impl_->auxiliary_.muon_moliere,
        impl_->moliere_interpolation_.deviceView(),
        impl_->auxiliary_.has_muon_moliere != 0,
        impl_->auxiliary_.brems_lpm, impl_->config_.thinning, particles,
        impl_->config_.random_seed, impl_->config_.shower_id,
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
        impl_->execution_);
    impl_->statistics_.physical_lepton_wavefronts += resident.wavefronts;
    impl_->statistics_.interaction_selection_batches += resident.wavefronts;
    impl_->statistics_.lepton_transport_batches += resident.wavefronts;
    impl_->statistics_.lepton_vertex_selection_batches += resident.wavefronts;
    impl_->statistics_.final_state_batches += resident.wavefronts;
    impl_->statistics_.interactions_selected += resident.transport_records;
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
  BackendCapabilities KokkosEmBackend::capabilities() const {
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
  AcceleratedEmStatistics const& KokkosEmBackend::statistics() const {
    return impl_->statistics_;
  }
  bool KokkosEmBackend::gpuProfileEnabled() const noexcept {
    return impl_->profile_accumulator_.enabled();
  }
  gpu::em::GpuProfileResult KokkosEmBackend::downloadProfile() {
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
  bool KokkosEmBackend::gpuRadioEnabled() const noexcept {
    return impl_->radio_accumulator_.enabled();
  }
  gpu::radio::GpuRadioWaveforms KokkosEmBackend::downloadRadioWaveforms() {
    impl_->requireInitialized();
    auto result = impl_->radio_accumulator_.download(impl_->execution_);
    impl_->statistics_.radio = impl_->radio_accumulator_.statistics();
    return result;
  }
  std::optional<gpu::em::GpuFirstInteractionSnapshot>
  KokkosEmBackend::downloadFirstInteractionSnapshot() {
    return impl_->first_interaction_;
  }
  KokkosRuntimeInfo const& KokkosEmBackend::runtimeInfo() const noexcept {
    return impl_->runtime_.info();
  }

} // namespace corsika::accelerator::em
