/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/KokkosEmBackend.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
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
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentPhotonCascade.hpp>

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
        resident.lpm_suppressions;
    impl_->statistics_.proposal_fallbacks += resident.fallback_events.size();
    impl_->statistics_.physical_secondaries_generated +=
        resident.electromagnetic_secondaries.size();
    for (auto const& record : resident.final_state_records) {
      impl_->statistics_.gpu_final_states++;
      if (record.process_id == PhotonPairProcessId) {
        impl_->statistics_.photon_pair_final_states++;
        impl_->statistics_.photon_pair_lpm_trials++;
      } else if (record.process_id == ComptonProcessId) {
        impl_->statistics_.compton_final_states++;
      } else if (record.process_id == PhotoelectricProcessId) {
        impl_->statistics_.photoelectric_final_states++;
      }
    }
    impl_->statistics_.photon_pair_lpm_trials += resident.lpm_suppressions;
    for (auto const& observation : resident.observations) {
      if (observation.status == ObservationStatus::ReachedObservationSurface)
        impl_->statistics_.photon_transport_observations++;
      else
        impl_->statistics_.photon_transport_escapes++;
    }
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
        impl_->first_interaction_, impl_->execution_);
    impl_->statistics_.physical_lepton_wavefronts += resident.wavefronts;
    impl_->statistics_.interaction_selection_batches += resident.wavefronts;
    impl_->statistics_.lepton_transport_batches += resident.wavefronts;
    impl_->statistics_.lepton_vertex_selection_batches += resident.wavefronts;
    impl_->statistics_.final_state_batches += resident.wavefronts;
    impl_->statistics_.interactions_selected += resident.transport_records;
    impl_->statistics_.proposal_fallbacks += resident.fallback_events.size();
    for (auto const& step : resident.step_records) {
      if (step.multiple_scattering_applied)
        impl_->statistics_.moliere_deflections++;
      if (step.traversed_grammage_g_per_cm2 > 0.)
        impl_->statistics_.moliere_trials++;
      if (step.multiple_scattering_status ==
          static_cast<std::uint16_t>(MoliereStatus::NoDeflection))
        impl_->statistics_.moliere_zero_deflections++;
      impl_->statistics_.moliere_newton_iterations +=
          step.multiple_scattering_iterations;
      impl_->statistics_.moliere_max_newton_iterations = std::max(
          impl_->statistics_.moliere_max_newton_iterations,
          step.multiple_scattering_iterations);
      switch (step.limit) {
      case LeptonTransportLimit::InteractionCandidate:
        impl_->statistics_.lepton_transport_interaction_candidates++;
        break;
      case LeptonTransportLimit::ContinuousStep:
        impl_->statistics_.lepton_transport_continuous_steps++;
        break;
      case LeptonTransportLimit::ParticleCut:
        impl_->statistics_.lepton_transport_cuts++;
        break;
      case LeptonTransportLimit::LayerBoundary:
        impl_->statistics_.lepton_transport_boundaries++;
        break;
      case LeptonTransportLimit::ObservationSurface:
        impl_->statistics_.lepton_transport_observations++;
        break;
      case LeptonTransportLimit::EscapedEnvironment:
        impl_->statistics_.lepton_transport_escapes++;
        break;
      case LeptonTransportLimit::MagneticStep:
        impl_->statistics_.lepton_transport_magnetic_steps++;
        break;
      case LeptonTransportLimit::DecayCandidate:
        impl_->statistics_.lepton_transport_decay_candidates++;
        break;
      }
    }
    for (auto const& record : resident.final_state_records) {
      impl_->statistics_.gpu_final_states++;
      impl_->statistics_.physical_secondaries_generated +=
          record.secondary_count;
      if (record.process_id == BremsProcessId) {
        impl_->statistics_.brems_final_states++;
        impl_->statistics_.brems_lpm_trials++;
      } else if (record.process_id == AnnihilationProcessId) {
        impl_->statistics_.annihilation_final_states++;
      } else if (record.process_id == IonizationProcessId) {
        impl_->statistics_.ionization_final_states++;
      } else if (record.process_id == ElectronPairProcessId) {
        impl_->statistics_.electron_pair_final_states++;
        impl_->statistics_.electron_pair_lpm_trials++;
      }
    }
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
