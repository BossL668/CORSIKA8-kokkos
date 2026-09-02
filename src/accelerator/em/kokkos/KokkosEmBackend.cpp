/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/KokkosEmBackend.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <corsika/accelerator/em/kokkos/KokkosInteractionSelector.hpp>
#include <corsika/accelerator/em/kokkos/KokkosLeptonFinalState.hpp>
#include <corsika/accelerator/em/kokkos/KokkosLeptonTransport.hpp>
#include <corsika/accelerator/em/kokkos/KokkosLeptonVertexSelector.hpp>
#include <corsika/accelerator/em/kokkos/KokkosMoliereInterpolation.hpp>
#include <corsika/accelerator/em/kokkos/KokkosPhotonFinalState.hpp>
#include <corsika/accelerator/em/kokkos/KokkosPhotonTransport.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>

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

    KokkosRuntime runtime_;
    BackendExecutionSpace execution_{};
    kokkos_detail::KokkosProposalNativeTable<BackendExecutionSpace> table_{};
    kokkos_detail::KokkosMoliereInterpolation<BackendExecutionSpace>
        moliere_interpolation_{};
    gpu::em::tables::ProposalNativeTableSet host_table_{};
    gpu::em::tables::ProposalNativeAuxData auxiliary_{};
    gpu::em::EnvironmentSnapshot environment_{};
    gpu::em::GpuEmConfig config_{};
    gpu::em::GpuEmStatistics statistics_{};
    gpu::em::tables::FlatRateTableView physics_{};
    std::optional<gpu::em::GpuFirstInteractionSnapshot> first_interaction_{};
    bool initialized_{};
  };

  KokkosEmBackend::KokkosEmBackend(KokkosRuntimeConfig const& config)
      : impl_(std::make_unique<Impl>(config)) {}
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
    impl_->auxiliary_ = auxiliary;
    impl_->table_.initialize(table);
    auto moliere_cache_path = auxiliary.cache_file;
    if (!moliere_cache_path.empty())
      moliere_cache_path += ".moliere-initial-v1.c8cache";
    impl_->moliere_interpolation_.initialize(
        auxiliary.electron_moliere, moliere_cache_path);
    impl_->physics_ = {};
    impl_->physics_.proposal_native = impl_->table_.deviceView();
    impl_->physics_.physics_source = 1u;
    impl_->physics_.em_transport_cut_MeV = config.em_transport_cut_MeV;
    impl_->physics_.muon_transport_cut_MeV = config.muon_transport_cut_MeV;
    impl_->statistics_.shower_ordinal = 1;
    impl_->statistics_.physics_source = gpu::em::GpuPhysicsSource::ProposalNative;
    impl_->statistics_.native_table_hash = table.content_hash;
    impl_->statistics_.auxiliary_cache_hash = auxiliary.content_hash;
    impl_->statistics_.native_table_nodes =
        table.bicubic_values.size() + table.cubic_values.size();
    impl_->statistics_.native_table_device_bytes = impl_->table_.deviceBytes();
    impl_->statistics_.table_device_bytes =
        impl_->table_.deviceBytes() +
        impl_->moliere_interpolation_.deviceBytes();
    impl_->statistics_.auxiliary_cache_hit = auxiliary.cache_hit;
    impl_->statistics_.reused_for_shower = false;
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
    return std::move(result.batch);
  }

  void KokkosEmBackend::beginShower(
      AcceleratedEmShowerConfig const& shower) {
    impl_->requireInitialized();
    impl_->config_.random_seed = shower.random_seed;
    impl_->config_.shower_id = shower.shower_id;
    impl_->config_.thinning = shower.thinning;
    auto const ordinal = impl_->statistics_.shower_ordinal + 1;
    auto const static_bytes = impl_->statistics_.native_table_device_bytes;
    auto const all_table_bytes = impl_->statistics_.table_device_bytes;
    auto const table_hash = impl_->statistics_.native_table_hash;
    auto const auxiliary_hash = impl_->statistics_.auxiliary_cache_hash;
    auto const auxiliary_hit = impl_->statistics_.auxiliary_cache_hit;
    impl_->statistics_ = {};
    impl_->statistics_.shower_ordinal = ordinal;
    impl_->statistics_.reused_for_shower = true;
    impl_->statistics_.physics_source = gpu::em::GpuPhysicsSource::ProposalNative;
    impl_->statistics_.native_table_hash = table_hash;
    impl_->statistics_.auxiliary_cache_hash = auxiliary_hash;
    impl_->statistics_.native_table_device_bytes = static_bytes;
    impl_->statistics_.table_device_bytes = all_table_bytes;
    impl_->statistics_.auxiliary_cache_hit = auxiliary_hit;
    impl_->first_interaction_.reset();
  }

  bool KokkosEmBackend::canTransport(
      gpu::em::EmParticleState const&) const {
    // The table and selection stages are operational.  Routing remains closed
    // until transport/final-state/profile/radio stages report full capability;
    // this prevents a partially implemented backend from silently changing a
    // production shower.
    return false;
  }
  bool KokkosEmBackend::hasProposalTable() const {
    return impl_->initialized_;
  }
  std::size_t KokkosEmBackend::minimumBatchSize() const {
    impl_->requireInitialized();
    return impl_->config_.min_batch_size;
  }
  std::size_t KokkosEmBackend::maximumResidentPhotonBatchSize() const {
    return 0;
  }
  std::size_t KokkosEmBackend::maximumResidentLeptonBatchSize() const {
    return 0;
  }
  std::size_t KokkosEmBackend::maximumResidentInputBatchSize() const {
    return 0;
  }
  std::size_t KokkosEmBackend::pendingPhotonCount() const noexcept { return 0; }
  std::size_t KokkosEmBackend::pendingLeptonCount() const noexcept { return 0; }

  gpu::em::ResidentPhotonCascadeResult KokkosEmBackend::runPhotonWavefront(
      std::vector<gpu::em::EmParticleState> const&, std::uint64_t,
      std::size_t, std::size_t) {
    throw std::logic_error(
        "Kokkos photon transport is not enabled before its physics gate passes");
  }
  gpu::em::ResidentLeptonCascadeResult KokkosEmBackend::runLeptonWavefront(
      std::vector<gpu::em::EmParticleState> const&, std::uint64_t,
      std::size_t, std::uint64_t, std::size_t) {
    throw std::logic_error(
        "Kokkos lepton transport is not enabled before its physics gate passes");
  }
  BackendCapabilities KokkosEmBackend::capabilities() const {
    return {impl_->runtime_.info().kind, false, false, false, false,
            false, false, false, true};
  }
  AcceleratedEmStatistics const& KokkosEmBackend::statistics() const {
    return impl_->statistics_;
  }
  bool KokkosEmBackend::gpuProfileEnabled() const noexcept { return false; }
  gpu::em::GpuProfileResult KokkosEmBackend::downloadProfile() { return {}; }
  bool KokkosEmBackend::gpuRadioEnabled() const noexcept { return false; }
  gpu::radio::GpuRadioWaveforms KokkosEmBackend::downloadRadioWaveforms() {
    return {};
  }
  std::optional<gpu::em::GpuFirstInteractionSnapshot>
  KokkosEmBackend::downloadFirstInteractionSnapshot() {
    return impl_->first_interaction_;
  }
  KokkosRuntimeInfo const& KokkosEmBackend::runtimeInfo() const noexcept {
    return impl_->runtime_.info();
  }

} // namespace corsika::accelerator::em
