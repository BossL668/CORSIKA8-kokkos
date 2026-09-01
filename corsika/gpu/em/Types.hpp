/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <corsika/gpu/em/EmThinning.hpp>
#include <corsika/gpu/radio/Types.hpp>

namespace corsika::gpu::em {

  enum class EmPid : std::int32_t {
    Electron = 11,
    Positron = -11,
    MuonMinus = 13,
    MuonPlus = -13,
    Photon = 22,
  };

  inline constexpr double MuonMassGeV = 0.1056583755;
  // Keep the device transport cut bit-for-bit aligned with the hard-coded
  // scalar ParticleCut condition `timePost > 10_ms`.
  inline constexpr double ParticleCutMaximumTimeS = 10.e-3;

#if defined(__CUDACC__)
#define CORSIKA_GPU_EM_PID_HOST_DEVICE __host__ __device__
#else
#define CORSIKA_GPU_EM_PID_HOST_DEVICE
#endif

  CORSIKA_GPU_EM_PID_HOST_DEVICE inline constexpr bool
  isElectronOrPositronPid(std::int32_t const pid) {
    return pid == static_cast<std::int32_t>(EmPid::Electron) ||
           pid == static_cast<std::int32_t>(EmPid::Positron);
  }

  CORSIKA_GPU_EM_PID_HOST_DEVICE inline constexpr bool
  isMuonPid(std::int32_t const pid) {
    return pid == static_cast<std::int32_t>(EmPid::MuonMinus) ||
           pid == static_cast<std::int32_t>(EmPid::MuonPlus);
  }

  CORSIKA_GPU_EM_PID_HOST_DEVICE inline constexpr bool
  isChargedLeptonPid(std::int32_t const pid) {
    return isElectronOrPositronPid(pid) || isMuonPid(pid);
  }

  CORSIKA_GPU_EM_PID_HOST_DEVICE inline constexpr bool
  exceedsParticleCutTime(double const time_s) {
    return time_s > ParticleCutMaximumTimeS;
  }

#undef CORSIKA_GPU_EM_PID_HOST_DEVICE

  /**
   * Trivially-copyable particle representation shared by host and device code.
   *
   * The units are deliberately not encoded as C++ quantity types here because this object is
   * copied directly to CUDA device memory. The boundary converting between CORSIKA quantities
   * and this representation is responsible for the conversion.
   */
  struct alignas(16) EmParticleState {
    std::int32_t pid{};
    std::int32_t medium_id{};
    std::uint32_t generation{};
    std::uint32_t reserved{};

    double energy_GeV{};
    double position_m[3]{};
    double direction[3]{};
    double time_s{};
    double weight{1.};

    std::uint64_t history_id{};
    std::uint64_t parent_history_id{};
    std::uint64_t step_id{};
  };

  /**
   * The unthinned physical final state of the generation-zero projectile.
   *
   * InteractionWriter runs before EMThinning on the scalar process sequence.
   * The device transport queue, on the other hand, contains only children
   * retained by thinning.  Keeping this one fixed-size snapshot at the final
   * state kernel therefore preserves the scalar writer semantics without
   * downloading every GPU interaction or re-inserting children into Stack.
   * All currently device-native electromagnetic final states have at most
   * three children.
   */
  struct GpuFirstInteractionSnapshot {
    EmParticleState parent_at_vertex{};
    EmParticleState secondaries[3]{};
    std::int32_t process_id{};
    std::uint32_t secondary_count{};
  };

  struct EmStepRecord {
    std::uint64_t history_id{};
    std::uint64_t step_id{};
    std::int32_t pid{};
    std::int32_t process_id{};
    double start_position_m[3]{};
    double end_position_m[3]{};
    double start_time_s{};
    double end_time_s{};
    double start_energy_GeV{};
    double end_energy_GeV{};
    double deposited_energy_GeV{};
    double weight{};
  };

  /**
   * Compact non-radio production record.
   *
   * The CUDA projection kernel evaluates the same tabulated ShowerAxis
   * interpolation as the scalar writer. This removes positions, directions,
   * timing and interaction-selection state from the device-to-host stream
   * when no radio track is required.
   */
  struct ProjectedEmStepRecord {
    std::uint64_t history_id{};
    std::int32_t pid{};
    std::int32_t process_id{};
    std::int32_t transport_limit{};
    std::int32_t reserved{};
    double start_grammage_g_per_cm2{};
    double end_grammage_g_per_cm2{};
    double end_energy_GeV{};
    double deposited_energy_GeV{};
    double weight{};
  };

  enum class ProposalFallbackReason : std::int32_t {
    UnsupportedParticle = 0,
    UnsupportedMedium = 1,
    UnsupportedGeometry = 2,
    ParticleTableMissing = 3,
    ProcessComponentMissing = 4,
    RateEnergyOutOfRange = 5,
    InverseCdfUnavailable = 6,
    LossEnergyOutOfRange = 7,
    LossQuantileOutOfRange = 8,
    InvalidTableQuery = 9,
    DeviceMemorySpill = 10,
    ZeroTotalRate = 11,
    CpuOnlyProcess = 12,
    GpuProcessNotImplemented = 13,
    InvalidFinalState = 14,
    LpmParametersUnavailable = 15,
    InvalidMassDensity = 16,
    ContinuousEnergyOutOfRange = 17,
    ContinuousRangeOutOfRange = 18,
    TransportCutReached = 19,
    MoliereParametersUnavailable = 20,
    MoliereSamplingFailed = 21,
    MagneticTransportFailed = 22,
    MagneticBoundaryFailed = 23,
    AtmosphereGrammageFailed = 24,
    AtmosphereInverseGrammageFailed = 25,
    AtmosphereVertexLookupFailed = 26,
    // The bounded device rejection sampler found a production point above
    // its precomputed Epair envelope.  The interaction type, target and v
    // remain valid, so CPU PROPOSAL may generate only this specified final
    // state.  Keep this separate from InvalidFinalState: the latter is a
    // numerical/physics defect and must not be silently retried.
    EpairRejectionEnvelopeExceeded = 27,
    // The device selected the process/component with the original PROPOSAL
    // uniform, but the conditional interval is too ill-conditioned for the
    // last few host/device interpolation ULPs to guarantee the requested v
    // tolerance.  CPU PROPOSAL replays that same uniform, verifies the
    // selected identity, and completes the final state after the wavefront.
    NativeSelectionReplay = 28,
  };

  struct ProposalFallbackEvent {
    EmParticleState particle{};
    std::uint64_t input_index{};
    std::int32_t process_id{};
    ProposalFallbackReason reason{
        ProposalFallbackReason::InvalidTableQuery};
    // Optional transport diagnostics. These fields are populated only for
    // generic numerical/geometry failures; physics-process fallbacks leave
    // them zero. Keeping the first failing state self-describing avoids
    // silently hiding a rare device transport defect behind a scalar retry.
    std::int32_t diagnostic_status{};
    std::uint32_t diagnostic_reserved{};
    double diagnostic_value0{};
    double diagnostic_value1{};
    double diagnostic_value2{};
    std::uint64_t component_hash{};
    std::uint64_t medium_hash{};
    std::uint64_t interaction_hash{};
    double energy_fraction{};
    double selection_uniform{};
    double loss_quantile{};
    double final_state_uniform{};
    // Counter-key provenance. selection_uniform/random_* identify the inner
    // PROPOSAL SampleLoss draw; outer_acceptance_* identify CORSIKA's
    // independent interaction-process acceptance draw.
    double outer_acceptance_uniform{};
    std::uint32_t random_process_id{};
    std::uint32_t outer_acceptance_random_process_id{};
    std::uint64_t random_draw_id{};
    std::uint64_t outer_acceptance_draw_id{};
    std::uint64_t final_state_draw_id{};
  };

  enum class EmInteractionStatus : std::int32_t {
    Selected = 0,
    NoDiscreteInteraction = 1,
    /**
     * A charged particle reached the sampled interaction grammage after
     * continuous energy loss. Process/component/v must be sampled again at
     * the vertex energy before final-state generation.
     */
    RequiresReselection = 2,
    /**
     * Only the total rate, interaction grammage and process threshold have
     * been sampled. Charged-particle process/component/v selection is
     * deferred until after continuous loss at the vertex.
     */
    DistanceSampled = 3,
    /**
     * The state is already below the configured kinetic-energy cut. The
     * transport kernel emits a zero-length terminal record and deposits the
     * remaining kinetic energy without querying an out-of-domain rate.
     */
    ParticleCut = 4,
  };

  /**
   * Device-selected stochastic interaction before final-state generation.
   *
   * interaction_grammage_g_per_cm2 is exponentially distributed with
   * parameter total_rate_cm2_per_g. The three stored uniforms make the
   * selection independently reproducible on the CPU.
   */
  struct EmInteractionRecord {
    EmParticleState particle{};
    std::uint64_t input_index{};
    std::int32_t process_id{};
    EmInteractionStatus status{EmInteractionStatus::Selected};
    std::uint64_t component_hash{};
    double total_rate_cm2_per_g{};
    double vertex_total_rate_cm2_per_g{};
    double interaction_grammage_g_per_cm2{};
    // Local density at the selected interaction vertex. It is intentionally
    // stored in the record because LPM suppression is evaluated after the
    // final-state energy split has been sampled.
    double mass_density_g_per_cm3{};
    // Rest mass resolved from the versioned continuous-energy table. Keeping
    // it with the selected interaction lets a shared charged-lepton
    // final-state kernel distinguish e± from μ± without hard-coded host
    // dispatch or an additional device-table lookup.
    double particle_mass_GeV{};
    // μ± decay competes with interaction, continuous and geometric limits.
    // The sampled lab-frame distance is retained across the selection and
    // transport kernels so a CPU decay module can be forced at the exact GPU
    // vertex without resampling the already selected lifetime.
    double decay_distance_m{
        std::numeric_limits<double>::infinity()};
    double decay_uniform{};
    std::uint64_t decay_draw_id{};
    double energy_fraction{};
    double distance_uniform{};
    double process_uniform{};
    double loss_quantile{};
    std::uint64_t distance_draw_id{};
    std::uint64_t process_draw_id{};
    std::uint64_t loss_draw_id{};
    // Proposal-native reproduces the second, independent scalar PROPOSAL draw
    // and derives loss_quantile from the selected rate interval. Legacy
    // c8emrt leaves these fields zero and keeps its existing loss draw.
    double proposal_selection_uniform{};
    std::uint32_t process_random_process_id{};
    std::uint32_t proposal_selection_random_process_id{};
    std::uint64_t proposal_selection_draw_id{};
  };

  struct RadioTrackRecord {
    EmStepRecord step{};
    double start_direction[3]{};
    double end_direction[3]{};
  };

  enum class ObservationStatus : std::int32_t {
    ReachedObservationSurface = 0,
    EscapedEnvironment = 1,
  };

  struct ObservationRecord {
    EmParticleState particle{};
    ObservationStatus status{ObservationStatus::ReachedObservationSurface};
    std::int32_t reserved{};
  };

  enum class DensityModel : std::uint32_t {
    Exponential = 0,
    Homogeneous = 1,
    // CORSIKA 7 calls its fifth overburden layer "linear", but the current
    // CORSIKA 8 builder represents it as a homogeneous-density medium.
    Linear = Homogeneous,
  };

  struct AtmosphereLayerSnapshot {
    double inner_radius_m{};
    double outer_radius_m{};
    // Exponential:
    //   rho(r) = a * exp((r - b) / c), with rho in g/cm^3 and radii in m.
    // Homogeneous:
    //   rho(r) = a; b and c are ignored.
    double density_parameter_a{};
    double density_parameter_b{};
    double density_parameter_c{};
    std::int32_t medium_id{};
    DensityModel density_model{DensityModel::Exponential};
  };

  constexpr std::size_t MaxAtmosphereLayers = 5;

  /**
   * Device-facing geometry schema. Phase 3 only validates and stores this object; tracking
   * kernels will consume the fields in a later phase.
   */
  struct EnvironmentSnapshot {
    double earth_center_m[3]{};
    AtmosphereLayerSnapshot atmosphere_layers[MaxAtmosphereLayers]{};
    std::uint32_t number_of_layers{};
    std::uint32_t reserved{};
    double magnetic_field_T[3]{};
    // The atmosphere remains spherical, but c8_air_shower terminates tracks
    // on an independent (locally flat) ObservationPlane.  Keep the former
    // radius as configuration metadata and store the actual plane explicitly.
    double observation_radius_m{};
    double observation_plane_point_m[3]{};
    double observation_plane_normal[3]{};
    double maximum_magnetic_deflection_rad{0.2};
  };

  /**
   * Expected identity of the current v10 file loaded from
   * GpuEmConfig::table_cache. The table reader may upgrade a legacy v9
   * envelope, but newly written descriptors always advertise v10.
   * A zero process_count or all-zero hash disables that individual check;
   * the file envelope, physics metadata and configured tolerance are always
   * validated.
   */
  struct ProposalTableSet {
    std::uint32_t format_version{10};
    std::uint32_t process_count{};
    std::array<std::uint8_t, 32> content_hash{};
  };

  enum class GpuPhysicsSource : std::uint32_t {
    C8EmRt = 0,
    ProposalNative = 1,
  };

  struct GpuEmConfig {
    struct ProfileProjection {
      bool enabled{};
      /**
       * Keep the projected longitudinal profile and energy-loss histograms
       * resident on the device. When enabled, per-step projected records are
       * not copied to the host; only the final O(number-of-bins) histograms
       * are downloaded at end of shower.
       */
      bool accumulate_on_device{};
      double axis_start_position_m[3]{};
      double axis_direction[3]{};
      double axis_step_length_m{};
      std::vector<double> axis_grammage_g_per_cm2{};
      std::size_t output_bin_count{};
      double output_bin_width_g_per_cm2{};
      double energy_loss_threshold_g_per_cm2{1.e-4};
      /**
       * Expected positive ranges for checked deterministic fixed-point sums.
       * Two integer headroom bits are retained; exceeding the representable
       * range is a hard shower error rather than a silent wraparound.
       */
      double fixed_point_weight_limit{};
      double fixed_point_energy_limit_GeV{};
    };

    int device{0};
    std::size_t min_batch_size{4096};
    double memory_fraction{0.70};
    double table_tolerance{1.e-3};
    double em_transport_cut_MeV{};
    // User-facing muon kinetic-energy/ParticleCut threshold.  Native
    // PROPOSAL utilities are shared by electrons and muons, so their scalar
    // transport endpoints must remain independently configurable.
    double muon_transport_cut_MeV{};
    bool deterministic{true};
    /**
     * Record CUDA-event timings at the boundaries of the fused lepton
     * pipeline. This is disabled in production because the extra event
     * records are intended for profiling, not normal shower transport.
     */
    bool detailed_stage_timing{};
    std::uint64_t random_seed{};
    std::uint64_t shower_id{};
    std::filesystem::path table_cache{};
    GpuPhysicsSource physics_source{GpuPhysicsSource::C8EmRt};
    std::filesystem::path auxiliary_cache_directory{};
    EmThinningConfig thinning{};
    bool resident_cross_species{};
    ProfileProjection profile_projection{};
    radio::GpuRadioConfig radio{};
  };

  /**
   * Per-shower state that may change while one initialized CUDA backend keeps
   * its tables, queues, workspaces, profile geometry and radio geometry
   * resident on the device.
   *
   * Profile fixed-point limits depend on the primary energy and therefore
   * belong to the shower lifecycle even though the histogram allocation and
   * ShowerAxis support are invariant for one application run.
   */
  struct GpuEmShowerConfig {
    std::uint64_t random_seed{};
    std::uint64_t shower_id{};
    EmThinningConfig thinning{};
    double profile_fixed_point_weight_limit{};
    double profile_fixed_point_energy_limit_GeV{};
  };

  inline GpuEmShowerConfig makeGpuEmShowerConfig(
      GpuEmConfig const& config) {
    return {
        config.random_seed,
        config.shower_id,
        config.thinning,
        config.profile_projection.fixed_point_weight_limit,
        config.profile_projection.fixed_point_energy_limit_GeV};
  }

  struct GpuProfileResult {
    std::vector<double> photons{};
    std::vector<double> electrons{};
    std::vector<double> positrons{};
    std::vector<double> muons_minus{};
    std::vector<double> muons_plus{};
    /**
     * Muons produced by a muon parent at accepted device-resident discrete
     * interaction vertices.  This is the accelerator counterpart of
     * ProductionProfile::doSecondaries(), not a transported-particle count.
     */
    std::vector<double> muon_parent_productions{};
    std::vector<double> energy_loss_GeV{};
    std::vector<double> muon_energy_loss_GeV{};
    std::uint64_t steps{};
    std::uint64_t deposited_steps{};
    std::uint64_t particle_cuts{};
    std::uint64_t fixed_point_overflows{};
    std::uint64_t invalid_records{};
    double weighted_deposited_energy_GeV{};
    /**
     * Deterministic energy-ledger terms accumulated from the same resident
     * transport/final-state records as the longitudinal profile.
     *
     * Total particle energies include rest mass.  Atomic-electron processes
     * (Compton, photoelectric, ionization and positron annihilation) therefore
     * import one electron rest mass from the medium.  ParticleCut deposits only
     * kinetic energy, so a terminated electron/positron rest mass is retained as
     * an explicit terminal sink instead of being silently hidden.
     */
    double weighted_medium_rest_mass_input_GeV{};
    double weighted_cut_rest_mass_energy_GeV{};
    double weighted_observed_total_energy_GeV{};
    double weighted_escaped_total_energy_GeV{};
  };

  struct GpuProfileStatistics {
    bool enabled{};
    bool deterministic{};
    std::uint64_t steps{};
    std::uint64_t deposited_steps{};
    std::uint64_t fixed_point_overflows{};
    std::uint64_t invalid_records{};
    std::size_t bins{};
    std::size_t device_bytes{};
    std::uint64_t host_to_device_bytes{};
    std::uint64_t device_to_host_bytes{};
    double kernel_time_ms{};
    double transfer_time_ms{};
  };

  /**
   * Detailed timing for synchronous CUDA copies issued by the EM backend.
   *
   * `host_api_time_ms` is measured around the CUDA runtime call and can
   * include time spent waiting for earlier work in the same stream.
   * `device_copy_time_ms` is measured between CUDA events immediately around
   * the copy and therefore excludes work queued before the first event.
   * Their non-negative difference is reported as
   * `host_wait_upper_bound_ms`; it is an upper bound on synchronization
   * wait because it also contains CUDA API and host timer overhead.
   *
   * Event timing is populated only when `GpuEmConfig::detailed_stage_timing`
   * is enabled. The operation counters and host API time are retained in
   * normal production mode without adding a synchronization point.
   */
  struct GpuTransferTimingStatistics {
    bool device_event_timing_enabled{};
    std::uint64_t operations{};
    std::uint64_t host_to_device_operations{};
    std::uint64_t device_to_host_operations{};
    std::uint64_t device_to_device_operations{};
    double host_api_time_ms{};
    double device_copy_time_ms{};
    double host_wait_upper_bound_ms{};
  };

  /** Host time spent in explicit CUDA event synchronization calls. */
  struct GpuSynchronizationTimingStatistics {
    std::uint64_t physical_pipeline_waits{};
    std::uint64_t profile_input_waits{};
    double physical_pipeline_wait_time_ms{};
    double profile_input_wait_time_ms{};
  };

  struct GpuLeptonPipelineTimingStatistics {
    bool enabled{};
    std::uint64_t wavefronts{};
    double selection_ms{};
    double transport_ms{};
    double transport_physics_ms{};
    double moliere_ms{};
    double transport_control_ms{};
    double transport_compaction_ms{};
    double interaction_extraction_ms{};
    double vertex_selection_ms{};
    double final_state_ms{};
    double final_state_classification_ms{};
    double final_state_scan_ms{};
    double final_state_summary_ms{};
    double final_state_write_ms{};
    double endpoint_compaction_ms{};
    double post_endpoint_ms{};
  };

  struct GpuEmStatistics {
    /**
     * Lifecycle metadata. shower_ordinal starts at one for initialize() and
     * increases on every beginShower() call. One-time initialization metrics
     * are reported separately from per-shower transfer and kernel counters.
     */
    std::uint64_t shower_ordinal{};
    bool reused_for_shower{};
    std::uint64_t static_host_to_device_bytes{};
    double one_time_initialization_ms{};
    GpuPhysicsSource physics_source{GpuPhysicsSource::C8EmRt};
    std::string native_proposal_version;
    std::string native_cubic_interpolation_version;
    std::array<std::uint8_t, 32> native_table_hash{};
    std::array<std::uint8_t, 32> auxiliary_cache_hash{};
    std::uint64_t native_table_nodes{};
    std::size_t native_table_device_bytes{};
    std::uint64_t native_newton_iterations{};
    std::uint64_t native_bisection_iterations{};
    std::uint64_t native_inverse_failures{};
    std::uint64_t proposal_cache_table_count{};
    std::uint64_t proposal_cache_hit_count{};
    bool proposal_cache_all_hit{};
    bool auxiliary_cache_hit{};
    std::uint64_t wavefronts{};
    std::uint64_t particles_enqueued{};
    std::uint64_t particles_advanced{};
    std::uint64_t particles_produced{};
    std::uint64_t proposal_fallbacks{};
    std::uint64_t interaction_selection_batches{};
    std::uint64_t interactions_selected{};
    std::uint64_t final_state_batches{};
    std::uint64_t gpu_final_states{};
    std::uint64_t first_interaction_candidates{};
    std::uint64_t physical_secondaries_generated{};
    std::uint64_t photon_pair_lpm_trials{};
    std::uint64_t photon_pair_lpm_suppressions{};
    std::uint64_t photon_pair_final_states{};
    std::uint64_t brems_final_states{};
    std::uint64_t compton_final_states{};
    std::uint64_t photoelectric_final_states{};
    std::uint64_t annihilation_final_states{};
    std::uint64_t ionization_final_states{};
    std::uint64_t electron_pair_final_states{};
    std::uint64_t brems_lpm_trials{};
    std::uint64_t brems_lpm_suppressions{};
    std::uint64_t electron_pair_lpm_trials{};
    std::uint64_t electron_pair_lpm_suppressions{};
    std::uint64_t electron_pair_rejection_trials{};
    std::uint64_t electron_pair_zero_weight_samples{};
    std::uint64_t electron_pair_rejection_fallbacks{};
    std::uint64_t electron_pair_envelope_violations{};
    std::uint64_t thinning_hillas_vertices{};
    std::uint64_t thinning_statistical_vertices{};
    std::uint64_t thinning_particles_discarded{};
    std::uint64_t photon_transport_batches{};
    std::uint64_t photon_transport_interactions{};
    std::uint64_t photon_transport_boundaries{};
    std::uint64_t photon_transport_observations{};
    std::uint64_t photon_transport_escapes{};
    std::uint64_t photon_transport_cuts{};
    std::uint64_t lepton_transport_batches{};
    std::uint64_t lepton_transport_interaction_candidates{};
    std::uint64_t lepton_transport_continuous_steps{};
    std::uint64_t lepton_transport_cuts{};
    std::uint64_t lepton_transport_boundaries{};
    std::uint64_t lepton_transport_observations{};
    std::uint64_t lepton_transport_escapes{};
    std::uint64_t lepton_transport_magnetic_steps{};
    std::uint64_t lepton_transport_decay_candidates{};
    std::uint64_t lepton_vertex_selection_batches{};
    std::uint64_t lepton_vertex_interactions_selected{};
    std::uint64_t lepton_vertex_no_interaction_continuations{};
    std::uint64_t moliere_trials{};
    std::uint64_t moliere_deflections{};
    std::uint64_t moliere_zero_deflections{};
    std::uint64_t moliere_newton_iterations{};
    std::uint32_t moliere_max_newton_iterations{};
    std::uint32_t moliere_statistics_reserved{};
    std::uint64_t physical_photon_wavefronts{};
    std::uint64_t physical_lepton_wavefronts{};
    std::uint64_t photon_selection_transport_summary_fusions{};
    std::uint64_t lepton_selection_transport_summary_fusions{};
    std::uint64_t
        photon_transport_final_state_summary_fusions{};
    std::uint64_t
        lepton_transport_vertex_summary_fusions{};
    std::uint64_t
        lepton_vertex_final_state_summary_fusions{};
    std::uint64_t photon_final_state_endpoint_summary_fusions{};
    std::uint64_t lepton_final_state_endpoint_summary_fusions{};
    std::uint64_t pipeline_host_synchronizations_eliminated{};
    std::uint64_t pipeline_device_to_host_bytes_eliminated{};
    std::uint64_t queue_overflows{};
    std::size_t current_particles{};
    std::size_t peak_particles{};
    std::size_t reserved_particles{};
    std::size_t peak_device_bytes{};
    std::size_t table_device_bytes{};
    std::size_t physical_workspace_bytes{};
    std::size_t maximum_resident_photon_batch{};
    std::size_t maximum_resident_lepton_batch{};
    std::size_t cross_species_queue_device_bytes{};
    std::size_t cross_species_queue_capacity_per_pid{};
    std::size_t peak_pending_photons{};
    std::size_t peak_pending_leptons{};
    std::uint64_t cross_species_particles_kept_on_device{};
    std::uint64_t cross_species_device_to_device_bytes{};
    std::uint64_t cross_species_host_spills{};
    std::uint64_t cross_species_spill_rebalances{};
    std::uint64_t cross_species_particles_spilled_to_cpu{};
    std::uint64_t cross_species_low_energy_ordering_checks{};
    std::uint64_t wavefront_bucketing_batches{};
    std::uint64_t wavefront_bucketing_particles{};
    std::uint64_t wavefront_bucketing_small_batches{};
    std::uint64_t wavefront_bucketing_small_particles{};
    std::uint64_t physical_host_to_device_bytes{};
    std::uint64_t physical_device_to_host_bytes{};
    double kernel_time_ms{};
    /**
     * Legacy host-wall measurement around synchronous CUDA copies. This field
     * is kept for output compatibility; use `transfer_timing` to distinguish
     * actual device copy time from synchronization wait.
     */
    double transfer_time_ms{};
    GpuTransferTimingStatistics transfer_timing{};
    GpuSynchronizationTimingStatistics synchronization_timing{};
    GpuLeptonPipelineTimingStatistics lepton_pipeline_timing{};
    GpuProfileStatistics profile{};
    radio::GpuRadioStatistics radio{};
  };

  struct EmBatchResult {
    std::uint64_t wavefront_index{};
    std::size_t input_particles{};
    std::size_t output_particles{};
    bool queue_overflow{};
    std::vector<EmStepRecord> step_records{};
    std::vector<ProposalFallbackEvent> fallback_events{};
    std::vector<RadioTrackRecord> radio_tracks{};
    std::vector<ObservationRecord> observations{};
  };

  struct EmInteractionBatchResult {
    std::size_t input_particles{};
    std::vector<EmInteractionRecord> interactions{};
    std::vector<ProposalFallbackEvent> fallback_events{};
  };

  struct LeptonVertexSelectionBatchResult {
    std::size_t input_candidates{};
    std::vector<EmInteractionRecord> interactions{};
    std::vector<EmInteractionRecord> continuations{};
    std::vector<ProposalFallbackEvent> fallback_events{};
  };

  enum class PhotonTransportLimit : std::int32_t {
    Interaction = 0,
    LayerBoundary = 1,
    ObservationSurface = 2,
    EscapedEnvironment = 3,
    ParticleCut = 4,
  };

  /**
   * Result of advancing one selected photon to its nearest physical limit.
   *
   * For Interaction, interaction.particle is the state at the vertex and its
   * mass_density_g_per_cm3 is ready for the LPM/final-state kernel. For a
   * boundary, end.step_id has been incremented so the next layer resamples an
   * exponentially distributed interaction grammage with a fresh Philox key.
   */
  struct PhotonTransportRecord {
    EmInteractionRecord interaction{};
    EmParticleState start{};
    EmParticleState end{};
    std::uint64_t input_index{};
    PhotonTransportLimit limit{PhotonTransportLimit::LayerBoundary};
    std::int32_t start_layer_index{-1};
    std::int32_t end_layer_index{-1};
    double distance_m{};
    double traversed_grammage_g_per_cm2{};
    double start_density_g_per_cm3{};
    double end_density_g_per_cm3{};
    double limiting_radius_m{};
    double cut_deposited_energy_GeV{};
    // ObservationPlane precedes ParticleCut in c8_air_shower's scalar
    // process sequence.  A step that reaches the plane after 10 ms must
    // therefore both publish the observation and terminate at ParticleCut.
    std::uint32_t observation_surface_reached_before_cut{};
  };

  struct PhotonTransportBatchResult {
    std::size_t input_interactions{};
    std::vector<PhotonTransportRecord> records{};
    std::vector<ProposalFallbackEvent> fallback_events{};
  };

  enum class LeptonTransportLimit : std::int32_t {
    InteractionCandidate = 0,
    ContinuousStep = 1,
    ParticleCut = 2,
    LayerBoundary = 3,
    ObservationSurface = 4,
    EscapedEnvironment = 5,
    MagneticStep = 6,
    DecayCandidate = 7,
  };

  /**
   * Result of one electron/positron transport step.
   *
   * The current device pipeline supports Moliere multiple scattering but
   * still uses a straight spatial trajectory; uniform-field magnetic bending
   * is integrated in the following stage. An InteractionCandidate always
   * carries RequiresReselection because the projectile energy changed
   * continuously after the interaction distance was sampled.
   */
  struct LeptonTransportRecord {
    EmInteractionRecord interaction{};
    EmParticleState start{};
    EmParticleState end{};
    std::uint64_t input_index{};
    LeptonTransportLimit limit{
        LeptonTransportLimit::ContinuousStep};
    std::int32_t start_layer_index{-1};
    std::int32_t end_layer_index{-1};
    double distance_m{};
    double traversed_grammage_g_per_cm2{};
    double continuous_step_grammage_g_per_cm2{};
    double start_density_g_per_cm3{};
    double end_density_g_per_cm3{};
    double limiting_radius_m{};
    double continuous_deposited_energy_GeV{};
    double cut_deposited_energy_GeV{};
    std::uint32_t observation_surface_reached_before_cut{};
    // Two 16-bit flags plus the iteration counter retain the former
    // eight-byte layout while exposing the dominant Moliere work per step.
    std::uint16_t multiple_scattering_applied{};
    std::uint16_t multiple_scattering_status{};
    std::uint32_t multiple_scattering_iterations{};
    double multiple_scattering_angle_rad{};
    double multiple_scattering_first_uniform{};
    double multiple_scattering_second_uniform{};
    double multiple_scattering_azimuth_uniform{};
    std::uint32_t magnetic_bending_applied{};
    std::uint32_t magnetic_step_status{};
    double magnetic_step_limit_m{};
    double magnetic_gyroradius_m{};
    double magnetic_bend_parameter{};
    double magnetic_chord_length_m{};
  };

  struct LeptonTransportBatchResult {
    std::size_t input_interactions{};
    std::vector<LeptonTransportRecord> records{};
    std::vector<ProposalFallbackEvent> fallback_events{};
  };

  /**
   * One successfully generated photon interaction final state.
   *
   * process_id defines the child multiplicity and process-specific field
   * interpretation. Photon pair and Compton produce two entries; photoelectric
   * absorption produces one electron. For Compton, energy_split_fraction is
   * the transferred fraction v. For photoelectric absorption it is the
   * outgoing electron kinetic energy divided by the parent photon energy,
   * allowing the omitted K-shell binding deposit to be reconstructed.
   */
  struct PhotonFinalStateRecord {
    std::uint64_t input_index{};
    std::uint64_t parent_history_id{};
    std::uint64_t secondary_offset{};
    std::uint32_t secondary_count{};
    std::int32_t process_id{};
    double energy_split_fraction{};
    double split_uniform{};
    double azimuth_uniform{};
    double electron_polar_uniform{};
    double positron_polar_uniform{};
    double lpm_survival_probability{};
    double lpm_uniform{};
    std::uint64_t split_draw_id{};
    std::uint64_t azimuth_draw_id{};
    std::uint64_t electron_polar_draw_id{};
    std::uint64_t positron_polar_draw_id{};
    std::uint64_t lpm_draw_id{};
    std::uint32_t thinning_status{};
    std::uint32_t thinning_keep_mask{0x3U};
    double thinning_first_uniform{};
    double thinning_second_uniform{};
    std::uint64_t thinning_first_draw_id{};
    std::uint64_t thinning_second_draw_id{};
  };

  using PhotonPairFinalStateRecord =
      PhotonFinalStateRecord;

  /**
   * A photon-pair candidate rejected by the LPM acceptance step.
   *
   * The physical state is unchanged. step_id is incremented before the parent
   * is written so the next interaction-distance and LPM draws use a fresh
   * counter-based random-number key.
   */
  struct PhotonPairLpmSuppressionRecord {
    EmParticleState particle{};
    std::uint64_t input_index{};
    std::uint64_t component_hash{};
    double survival_probability{};
    double uniform{};
    std::uint64_t draw_id{};
  };

  /**
   * Accepted PROPOSAL charged-lepton final state.
   *
   * For BremsEGS4Approximation, secondary_offset addresses the surviving
   * lepton followed by the emitted photon and photon_energy_fraction is v.
   * For HeitlerAnnihilation, it addresses the (1-rho) and rho photons and
   * photon_energy_fraction stores rho. For NaivIonization it stores v and the
   * children are the outgoing primary followed by the delta electron.
   * For KelnerKokoulinPetrukhin Epair production it stores v and addresses
   * the surviving primary, e- and e+; final_state_uniform selects |rho|,
   * azimuth_uniform selects its sign, and auxiliary_uniform is PROPOSAL's
   * currently unused pair-direction draw. For the original three processes
   * auxiliary_uniform remains zero.
   */
  struct BremsFinalStateRecord {
    std::uint64_t input_index{};
    std::uint64_t parent_history_id{};
    std::uint64_t secondary_offset{};
    std::uint32_t secondary_count{};
    std::int32_t process_id{};
    double photon_energy_fraction{};
    double final_state_uniform{};
    double azimuth_uniform{};
    double auxiliary_uniform{};
    double lpm_survival_probability{};
    double lpm_uniform{};
    std::uint64_t final_state_draw_id{};
    std::uint64_t azimuth_draw_id{};
    std::uint64_t auxiliary_draw_id{};
    std::uint64_t lpm_draw_id{};
    std::uint32_t thinning_status{};
    std::uint32_t thinning_keep_mask{0x3U};
    double thinning_first_uniform{};
    double thinning_second_uniform{};
    std::uint64_t thinning_first_draw_id{};
    std::uint64_t thinning_second_draw_id{};
  };

  using LeptonFinalStateRecord = BremsFinalStateRecord;

  struct BremsLpmSuppressionRecord {
    EmParticleState particle{};
    std::uint64_t input_index{};
    std::uint64_t component_hash{};
    double survival_probability{};
    double uniform{};
    std::uint64_t draw_id{};
  };

  struct BremsFinalStateBatchResult {
    std::size_t input_interactions{};
    std::size_t gpu_interactions{};
    std::size_t brems_interactions{};
    std::size_t annihilation_interactions{};
    std::size_t ionization_interactions{};
    std::size_t electron_pair_interactions{};
    std::size_t brems_lpm_trials{};
    std::size_t brems_lpm_suppressions{};
    std::size_t electron_pair_lpm_trials{};
    std::size_t electron_pair_lpm_suppressions{};
    std::size_t electron_pair_rejection_trials{};
    std::size_t electron_pair_zero_weight_samples{};
    std::size_t electron_pair_rejection_fallbacks{};
    std::size_t electron_pair_envelope_violations{};
    std::vector<BremsFinalStateRecord> final_state_records{};
    std::vector<EmParticleState> secondaries{};
    std::vector<BremsLpmSuppressionRecord> lpm_suppressed{};
    std::vector<EmInteractionRecord> continuations{};
    std::vector<ProposalFallbackEvent> fallback_events{};
  };

  struct EmFinalStateBatchResult {
    std::size_t input_interactions{};
    std::size_t gpu_interactions{};
    std::size_t photon_pair_interactions{};
    std::size_t compton_interactions{};
    std::size_t photoelectric_interactions{};
    std::vector<PhotonPairFinalStateRecord> final_state_records{};
    std::vector<EmParticleState> secondaries{};
    std::vector<PhotonPairLpmSuppressionRecord> lpm_suppressed{};
    std::vector<EmInteractionRecord> continuations{};
    std::vector<ProposalFallbackEvent> fallback_events{};
  };

  /**
   * Host-visible closure of one physical photon wavefront.
   *
   * Boundary-crossing, LPM-suppressed and Compton-scattered photons are
   * returned in next_photons with stable source-input order. The complete
   * process final state also remains in final_states.secondaries for physics
   * validation; callers must not enqueue its photon entries a second time.
   * Unsupported physics is never hidden: it remains in one of the three
   * explicit fallback collections.
   */
  struct PhotonWavefrontBatchResult {
    std::size_t input_particles{};
    std::vector<PhotonTransportRecord> transport_records{};
    std::vector<EmParticleState> next_photons{};
    std::vector<ObservationRecord> observations{};
    std::vector<ProposalFallbackEvent>
        selection_fallback_events{};
    std::vector<ProposalFallbackEvent>
        transport_fallback_events{};
    EmFinalStateBatchResult final_states{};
  };

  /**
   * Host outputs of a multi-wavefront photon cascade whose boundary/LPM
   * continuation queue stays resident on the GPU.
   *
   * Electrons and positrons are returned as electromagnetic_secondaries until
   * their own transport kernels are implemented. completed=false is allowed
   * when the caller's explicit wavefront limit, minimum resident batch size,
   * or configured workspace limit is reached; in every case
   * remaining_photons is a lossless checkpoint.
   */
  struct ResidentPhotonCascadeResult {
    std::size_t input_particles{};
    std::size_t wavefronts{};
    std::size_t peak_resident_photons{};
    std::size_t transport_records{};
    std::size_t interaction_vertices{};
    std::size_t layer_boundaries{};
    std::size_t particle_cuts{};
    std::size_t lpm_suppressions{};
    bool below_minimum_batch_checkpoint{};
    bool workspace_limit_checkpoint{};
    bool completed{};
    std::vector<EmParticleState> electromagnetic_secondaries{};
    std::vector<EmParticleState> cpu_spill_particles{};
    std::vector<ProposalFallbackEvent> fallback_events{};
    std::vector<ObservationRecord> observations{};
    std::vector<PhotonTransportRecord> step_records{};
    std::vector<ProjectedEmStepRecord> projected_step_records{};
    std::vector<PhotonPairFinalStateRecord> final_state_records{};
    std::vector<EmParticleState> remaining_photons{};
  };

  /**
   * Host outputs of a multi-wavefront charged cascade whose charged
   * continuations stay resident on the GPU.
   *
   * Bremsstrahlung and annihilation photons are returned for the photon
   * scheduler. completed=false is allowed at the explicit wavefront/history
   * limits, once a resident continuation front falls below the configured
   * CUDA batch size, or at the configured workspace limit.
   * remaining_leptons always forms a lossless checkpoint.
   */
  struct ResidentLeptonCascadeResult {
    std::size_t input_particles{};
    std::size_t wavefronts{};
    std::size_t peak_resident_leptons{};
    std::size_t transport_records{};
    std::size_t interaction_vertices{};
    std::size_t lpm_suppressions{};
    std::uint64_t secondary_history_ids_used{};
    bool history_range_exhausted{};
    bool below_minimum_batch_checkpoint{};
    bool workspace_limit_checkpoint{};
    bool completed{};
    std::vector<LeptonTransportRecord> step_records{};
    std::vector<ProjectedEmStepRecord> projected_step_records{};
    std::vector<EmParticleState> generated_photons{};
    std::vector<EmParticleState> cpu_spill_particles{};
    std::vector<ProposalFallbackEvent> fallback_events{};
    std::vector<ObservationRecord> observations{};
    std::vector<EmParticleState> decay_candidates{};
    std::vector<BremsFinalStateRecord> final_state_records{};
    std::vector<EmParticleState> remaining_leptons{};
  };

  static_assert(std::is_standard_layout_v<EmParticleState>);
  static_assert(std::is_trivially_copyable_v<EmParticleState>);
  static_assert(alignof(EmParticleState) == 16);
  static_assert(std::is_standard_layout_v<EmStepRecord>);
  static_assert(std::is_trivially_copyable_v<EmStepRecord>);
  static_assert(std::is_standard_layout_v<ProjectedEmStepRecord>);
  static_assert(std::is_trivially_copyable_v<ProjectedEmStepRecord>);
  static_assert(
      sizeof(ProjectedEmStepRecord) <=
      sizeof(EmInteractionRecord),
      "projected records must fit the reusable interaction arena");
  static_assert(std::is_standard_layout_v<ProposalFallbackEvent>);
  static_assert(std::is_trivially_copyable_v<ProposalFallbackEvent>);
  static_assert(std::is_standard_layout_v<EmInteractionRecord>);
  static_assert(std::is_trivially_copyable_v<EmInteractionRecord>);
  static_assert(std::is_standard_layout_v<GpuFirstInteractionSnapshot>);
  static_assert(std::is_trivially_copyable_v<GpuFirstInteractionSnapshot>);
  static_assert(std::is_standard_layout_v<PhotonPairFinalStateRecord>);
  static_assert(std::is_trivially_copyable_v<PhotonPairFinalStateRecord>);
  static_assert(std::is_standard_layout_v<PhotonPairLpmSuppressionRecord>);
  static_assert(std::is_trivially_copyable_v<PhotonPairLpmSuppressionRecord>);
  static_assert(std::is_standard_layout_v<BremsFinalStateRecord>);
  static_assert(std::is_trivially_copyable_v<BremsFinalStateRecord>);
  static_assert(std::is_standard_layout_v<BremsLpmSuppressionRecord>);
  static_assert(std::is_trivially_copyable_v<BremsLpmSuppressionRecord>);
  static_assert(std::is_standard_layout_v<PhotonTransportRecord>);
  static_assert(std::is_trivially_copyable_v<PhotonTransportRecord>);
  static_assert(std::is_standard_layout_v<LeptonTransportRecord>);
  static_assert(std::is_trivially_copyable_v<LeptonTransportRecord>);
  static_assert(std::is_standard_layout_v<EnvironmentSnapshot>);
  static_assert(std::is_trivially_copyable_v<EnvironmentSnapshot>);

} // namespace corsika::gpu::em
