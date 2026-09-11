/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#pragma once
#include <corsika/accelerator/em/common/Types.hpp>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace corsika::accelerator::em::detail {
template<class T> T cooperativeAdd(T a,T b) {
  if constexpr(std::is_integral_v<T>) {
    if(b>std::numeric_limits<T>::max()-a)
      throw std::overflow_error("cooperative counter overflow");
  }
  T r=a+b;
  if constexpr(std::is_floating_point_v<T>)
    if(!std::isfinite(r)) throw std::overflow_error("nonfinite cooperative diagnostic");
  return r;
}
inline void mergeCooperative(gpu::em::ResidentEmProcessStatistics& a,gpu::em::ResidentEmProcessStatistics b) {
  a.gpu_final_states=cooperativeAdd(a.gpu_final_states,b.gpu_final_states);
  a.physical_secondaries_generated=cooperativeAdd(a.physical_secondaries_generated,b.physical_secondaries_generated);
  a.photon_pair_final_states=cooperativeAdd(a.photon_pair_final_states,b.photon_pair_final_states);
  a.brems_final_states=cooperativeAdd(a.brems_final_states,b.brems_final_states);
  a.compton_final_states=cooperativeAdd(a.compton_final_states,b.compton_final_states);
  a.photoelectric_final_states=cooperativeAdd(a.photoelectric_final_states,b.photoelectric_final_states);
  a.annihilation_final_states=cooperativeAdd(a.annihilation_final_states,b.annihilation_final_states);
  a.ionization_final_states=cooperativeAdd(a.ionization_final_states,b.ionization_final_states);
  a.electron_pair_final_states=cooperativeAdd(a.electron_pair_final_states,b.electron_pair_final_states);
  a.photon_pair_lpm_trials=cooperativeAdd(a.photon_pair_lpm_trials,b.photon_pair_lpm_trials);
  a.photon_pair_lpm_suppressions=cooperativeAdd(a.photon_pair_lpm_suppressions,b.photon_pair_lpm_suppressions);
  a.brems_lpm_trials=cooperativeAdd(a.brems_lpm_trials,b.brems_lpm_trials);
  a.brems_lpm_suppressions=cooperativeAdd(a.brems_lpm_suppressions,b.brems_lpm_suppressions);
  a.electron_pair_lpm_trials=cooperativeAdd(a.electron_pair_lpm_trials,b.electron_pair_lpm_trials);
  a.electron_pair_lpm_suppressions=cooperativeAdd(a.electron_pair_lpm_suppressions,b.electron_pair_lpm_suppressions);
  a.electron_pair_rejection_trials=cooperativeAdd(a.electron_pair_rejection_trials,b.electron_pair_rejection_trials);
  a.electron_pair_zero_weight_samples=cooperativeAdd(a.electron_pair_zero_weight_samples,b.electron_pair_zero_weight_samples);
  a.electron_pair_rejection_fallbacks=cooperativeAdd(a.electron_pair_rejection_fallbacks,b.electron_pair_rejection_fallbacks);
  a.electron_pair_envelope_violations=cooperativeAdd(a.electron_pair_envelope_violations,b.electron_pair_envelope_violations);
  a.thinning_hillas_vertices=cooperativeAdd(a.thinning_hillas_vertices,b.thinning_hillas_vertices);
  a.thinning_statistical_vertices=cooperativeAdd(a.thinning_statistical_vertices,b.thinning_statistical_vertices);
  a.thinning_particles_discarded=cooperativeAdd(a.thinning_particles_discarded,b.thinning_particles_discarded);
  a.moliere_trials=cooperativeAdd(a.moliere_trials,b.moliere_trials);
  a.moliere_deflections=cooperativeAdd(a.moliere_deflections,b.moliere_deflections);
  a.moliere_zero_deflections=cooperativeAdd(a.moliere_zero_deflections,b.moliere_zero_deflections);
  a.moliere_newton_iterations=cooperativeAdd(a.moliere_newton_iterations,b.moliere_newton_iterations);
  a.moliere_max_newton_iterations=std::max(a.moliere_max_newton_iterations,b.moliere_max_newton_iterations);
  for(std::size_t i=0;i<a.lepton_transport_limits.size();++i) a.lepton_transport_limits[i]=cooperativeAdd(a.lepton_transport_limits[i],b.lepton_transport_limits[i]);
}
inline void mergeCooperative(gpu::em::ResidentPhotonCascadeResult& a,gpu::em::ResidentPhotonCascadeResult b) {
  a.input_particles=cooperativeAdd(a.input_particles,b.input_particles);
  a.wavefronts=cooperativeAdd(a.wavefronts,b.wavefronts);
  a.peak_resident_photons=std::max(a.peak_resident_photons,b.peak_resident_photons);
  a.selected_interactions=cooperativeAdd(a.selected_interactions,b.selected_interactions);
  a.interaction_bearing_wavefronts=cooperativeAdd(a.interaction_bearing_wavefronts,b.interaction_bearing_wavefronts);
  a.transport_records=cooperativeAdd(a.transport_records,b.transport_records);
  a.interaction_vertices=cooperativeAdd(a.interaction_vertices,b.interaction_vertices);
  a.layer_boundaries=cooperativeAdd(a.layer_boundaries,b.layer_boundaries);
  a.particle_cuts=cooperativeAdd(a.particle_cuts,b.particle_cuts);
  a.lpm_suppressions=cooperativeAdd(a.lpm_suppressions,b.lpm_suppressions);
  a.wavefront_bucketing_batches=cooperativeAdd(a.wavefront_bucketing_batches,b.wavefront_bucketing_batches);
  a.wavefront_bucketing_particles=cooperativeAdd(a.wavefront_bucketing_particles,b.wavefront_bucketing_particles);
  a.wavefront_bucketing_small_batches=cooperativeAdd(a.wavefront_bucketing_small_batches,b.wavefront_bucketing_small_batches);
  a.wavefront_bucketing_small_particles=cooperativeAdd(a.wavefront_bucketing_small_particles,b.wavefront_bucketing_small_particles);
  a.native_newton_iterations=cooperativeAdd(a.native_newton_iterations,b.native_newton_iterations);
  a.native_bisection_iterations=cooperativeAdd(a.native_bisection_iterations,b.native_bisection_iterations);
  a.native_inverse_failures=cooperativeAdd(a.native_inverse_failures,b.native_inverse_failures);
  mergeCooperative(a.process_statistics,b.process_statistics);
  a.below_minimum_batch_checkpoint=a.below_minimum_batch_checkpoint||b.below_minimum_batch_checkpoint;
  a.workspace_limit_checkpoint=a.workspace_limit_checkpoint||b.workspace_limit_checkpoint;
  a.completed=a.completed&&b.completed;
  a.electromagnetic_secondaries.insert(a.electromagnetic_secondaries.end(),std::make_move_iterator(b.electromagnetic_secondaries.begin()),std::make_move_iterator(b.electromagnetic_secondaries.end()));
  a.cpu_spill_particles.insert(a.cpu_spill_particles.end(),std::make_move_iterator(b.cpu_spill_particles.begin()),std::make_move_iterator(b.cpu_spill_particles.end()));
  a.fallback_events.insert(a.fallback_events.end(),std::make_move_iterator(b.fallback_events.begin()),std::make_move_iterator(b.fallback_events.end()));
  a.observations.insert(a.observations.end(),std::make_move_iterator(b.observations.begin()),std::make_move_iterator(b.observations.end()));
  a.interaction_records.insert(a.interaction_records.end(),std::make_move_iterator(b.interaction_records.begin()),std::make_move_iterator(b.interaction_records.end()));
  a.step_records.insert(a.step_records.end(),std::make_move_iterator(b.step_records.begin()),std::make_move_iterator(b.step_records.end()));
  a.projected_step_records.insert(a.projected_step_records.end(),std::make_move_iterator(b.projected_step_records.begin()),std::make_move_iterator(b.projected_step_records.end()));
  a.final_state_records.insert(a.final_state_records.end(),std::make_move_iterator(b.final_state_records.begin()),std::make_move_iterator(b.final_state_records.end()));
  a.remaining_photons.insert(a.remaining_photons.end(),std::make_move_iterator(b.remaining_photons.begin()),std::make_move_iterator(b.remaining_photons.end()));
}
inline void mergeCooperative(gpu::em::ResidentLeptonCascadeResult& a,gpu::em::ResidentLeptonCascadeResult b) {
  a.input_particles=cooperativeAdd(a.input_particles,b.input_particles);
  a.wavefronts=cooperativeAdd(a.wavefronts,b.wavefronts);
  a.peak_resident_leptons=std::max(a.peak_resident_leptons,b.peak_resident_leptons);
  a.selected_interactions=cooperativeAdd(a.selected_interactions,b.selected_interactions);
  a.interaction_bearing_wavefronts=cooperativeAdd(a.interaction_bearing_wavefronts,b.interaction_bearing_wavefronts);
  a.vertex_interactions_selected=cooperativeAdd(a.vertex_interactions_selected,b.vertex_interactions_selected);
  a.vertex_no_interaction_continuations=cooperativeAdd(a.vertex_no_interaction_continuations,b.vertex_no_interaction_continuations);
  a.final_state_bearing_wavefronts=cooperativeAdd(a.final_state_bearing_wavefronts,b.final_state_bearing_wavefronts);
  a.transport_records=cooperativeAdd(a.transport_records,b.transport_records);
  a.interaction_vertices=cooperativeAdd(a.interaction_vertices,b.interaction_vertices);
  a.lpm_suppressions=cooperativeAdd(a.lpm_suppressions,b.lpm_suppressions);
  a.wavefront_bucketing_batches=cooperativeAdd(a.wavefront_bucketing_batches,b.wavefront_bucketing_batches);
  a.wavefront_bucketing_particles=cooperativeAdd(a.wavefront_bucketing_particles,b.wavefront_bucketing_particles);
  a.wavefront_bucketing_small_batches=cooperativeAdd(a.wavefront_bucketing_small_batches,b.wavefront_bucketing_small_batches);
  a.wavefront_bucketing_small_particles=cooperativeAdd(a.wavefront_bucketing_small_particles,b.wavefront_bucketing_small_particles);
  mergeCooperative(a.process_statistics,b.process_statistics);
  a.secondary_history_ids_used=cooperativeAdd(a.secondary_history_ids_used,b.secondary_history_ids_used);
  a.history_range_exhausted=a.history_range_exhausted||b.history_range_exhausted;
  a.below_minimum_batch_checkpoint=a.below_minimum_batch_checkpoint||b.below_minimum_batch_checkpoint;
  a.workspace_limit_checkpoint=a.workspace_limit_checkpoint||b.workspace_limit_checkpoint;
  a.completed=a.completed&&b.completed;
  a.step_records.insert(a.step_records.end(),std::make_move_iterator(b.step_records.begin()),std::make_move_iterator(b.step_records.end()));
  a.projected_step_records.insert(a.projected_step_records.end(),std::make_move_iterator(b.projected_step_records.begin()),std::make_move_iterator(b.projected_step_records.end()));
  a.generated_photons.insert(a.generated_photons.end(),std::make_move_iterator(b.generated_photons.begin()),std::make_move_iterator(b.generated_photons.end()));
  a.cpu_spill_particles.insert(a.cpu_spill_particles.end(),std::make_move_iterator(b.cpu_spill_particles.begin()),std::make_move_iterator(b.cpu_spill_particles.end()));
  a.fallback_events.insert(a.fallback_events.end(),std::make_move_iterator(b.fallback_events.begin()),std::make_move_iterator(b.fallback_events.end()));
  a.observations.insert(a.observations.end(),std::make_move_iterator(b.observations.begin()),std::make_move_iterator(b.observations.end()));
  a.decay_candidates.insert(a.decay_candidates.end(),std::make_move_iterator(b.decay_candidates.begin()),std::make_move_iterator(b.decay_candidates.end()));
  a.interaction_records.insert(a.interaction_records.end(),std::make_move_iterator(b.interaction_records.begin()),std::make_move_iterator(b.interaction_records.end()));
  a.final_state_records.insert(a.final_state_records.end(),std::make_move_iterator(b.final_state_records.begin()),std::make_move_iterator(b.final_state_records.end()));
  a.remaining_leptons.insert(a.remaining_leptons.end(),std::make_move_iterator(b.remaining_leptons.begin()),std::make_move_iterator(b.remaining_leptons.end()));
}
inline gpu::em::GpuEmStatistics mergeCooperativeStatistics(
    gpu::em::GpuEmStatistics a,gpu::em::GpuEmStatistics const& b) {
  if(a.native_table_hash!=b.native_table_hash || a.auxiliary_cache_hash!=b.auxiliary_cache_hash ||
     a.shower_ordinal!=b.shower_ordinal)
    throw std::logic_error("cooperative endpoints have incompatible physics/shower identity");
  // Identity, GPU capacities/VRAM and tuning describe the CUDA endpoint.
  // Physical counters and phase work are summed; wall time is measured outside.
  a.static_host_to_device_bytes=cooperativeAdd(a.static_host_to_device_bytes,b.static_host_to_device_bytes);
  a.native_newton_iterations=cooperativeAdd(a.native_newton_iterations,b.native_newton_iterations);
  a.native_bisection_iterations=cooperativeAdd(a.native_bisection_iterations,b.native_bisection_iterations);
  a.native_inverse_failures=cooperativeAdd(a.native_inverse_failures,b.native_inverse_failures);
  a.wavefronts=cooperativeAdd(a.wavefronts,b.wavefronts);
  a.particles_enqueued=cooperativeAdd(a.particles_enqueued,b.particles_enqueued);
  a.particles_advanced=cooperativeAdd(a.particles_advanced,b.particles_advanced);
  a.particles_produced=cooperativeAdd(a.particles_produced,b.particles_produced);
  a.proposal_fallbacks=cooperativeAdd(a.proposal_fallbacks,b.proposal_fallbacks);
  a.interaction_selection_batches=cooperativeAdd(a.interaction_selection_batches,b.interaction_selection_batches);
  a.interactions_selected=cooperativeAdd(a.interactions_selected,b.interactions_selected);
  a.final_state_batches=cooperativeAdd(a.final_state_batches,b.final_state_batches);
  a.gpu_final_states=cooperativeAdd(a.gpu_final_states,b.gpu_final_states);
  a.first_interaction_candidates=cooperativeAdd(a.first_interaction_candidates,b.first_interaction_candidates);
  a.physical_secondaries_generated=cooperativeAdd(a.physical_secondaries_generated,b.physical_secondaries_generated);
  a.photon_pair_lpm_trials=cooperativeAdd(a.photon_pair_lpm_trials,b.photon_pair_lpm_trials);
  a.photon_pair_lpm_suppressions=cooperativeAdd(a.photon_pair_lpm_suppressions,b.photon_pair_lpm_suppressions);
  a.photon_pair_final_states=cooperativeAdd(a.photon_pair_final_states,b.photon_pair_final_states);
  a.brems_final_states=cooperativeAdd(a.brems_final_states,b.brems_final_states);
  a.compton_final_states=cooperativeAdd(a.compton_final_states,b.compton_final_states);
  a.photoelectric_final_states=cooperativeAdd(a.photoelectric_final_states,b.photoelectric_final_states);
  a.annihilation_final_states=cooperativeAdd(a.annihilation_final_states,b.annihilation_final_states);
  a.ionization_final_states=cooperativeAdd(a.ionization_final_states,b.ionization_final_states);
  a.electron_pair_final_states=cooperativeAdd(a.electron_pair_final_states,b.electron_pair_final_states);
  a.brems_lpm_trials=cooperativeAdd(a.brems_lpm_trials,b.brems_lpm_trials);
  a.brems_lpm_suppressions=cooperativeAdd(a.brems_lpm_suppressions,b.brems_lpm_suppressions);
  a.electron_pair_lpm_trials=cooperativeAdd(a.electron_pair_lpm_trials,b.electron_pair_lpm_trials);
  a.electron_pair_lpm_suppressions=cooperativeAdd(a.electron_pair_lpm_suppressions,b.electron_pair_lpm_suppressions);
  a.electron_pair_rejection_trials=cooperativeAdd(a.electron_pair_rejection_trials,b.electron_pair_rejection_trials);
  a.electron_pair_zero_weight_samples=cooperativeAdd(a.electron_pair_zero_weight_samples,b.electron_pair_zero_weight_samples);
  a.electron_pair_rejection_fallbacks=cooperativeAdd(a.electron_pair_rejection_fallbacks,b.electron_pair_rejection_fallbacks);
  a.electron_pair_envelope_violations=cooperativeAdd(a.electron_pair_envelope_violations,b.electron_pair_envelope_violations);
  a.thinning_hillas_vertices=cooperativeAdd(a.thinning_hillas_vertices,b.thinning_hillas_vertices);
  a.thinning_statistical_vertices=cooperativeAdd(a.thinning_statistical_vertices,b.thinning_statistical_vertices);
  a.thinning_particles_discarded=cooperativeAdd(a.thinning_particles_discarded,b.thinning_particles_discarded);
  a.photon_transport_batches=cooperativeAdd(a.photon_transport_batches,b.photon_transport_batches);
  a.photon_transport_interactions=cooperativeAdd(a.photon_transport_interactions,b.photon_transport_interactions);
  a.photon_transport_boundaries=cooperativeAdd(a.photon_transport_boundaries,b.photon_transport_boundaries);
  a.photon_transport_observations=cooperativeAdd(a.photon_transport_observations,b.photon_transport_observations);
  a.photon_transport_escapes=cooperativeAdd(a.photon_transport_escapes,b.photon_transport_escapes);
  a.photon_transport_cuts=cooperativeAdd(a.photon_transport_cuts,b.photon_transport_cuts);
  a.lepton_transport_batches=cooperativeAdd(a.lepton_transport_batches,b.lepton_transport_batches);
  a.lepton_transport_interaction_candidates=cooperativeAdd(a.lepton_transport_interaction_candidates,b.lepton_transport_interaction_candidates);
  a.lepton_transport_continuous_steps=cooperativeAdd(a.lepton_transport_continuous_steps,b.lepton_transport_continuous_steps);
  a.lepton_transport_cuts=cooperativeAdd(a.lepton_transport_cuts,b.lepton_transport_cuts);
  a.lepton_transport_boundaries=cooperativeAdd(a.lepton_transport_boundaries,b.lepton_transport_boundaries);
  a.lepton_transport_observations=cooperativeAdd(a.lepton_transport_observations,b.lepton_transport_observations);
  a.lepton_transport_escapes=cooperativeAdd(a.lepton_transport_escapes,b.lepton_transport_escapes);
  a.lepton_transport_magnetic_steps=cooperativeAdd(a.lepton_transport_magnetic_steps,b.lepton_transport_magnetic_steps);
  a.lepton_transport_decay_candidates=cooperativeAdd(a.lepton_transport_decay_candidates,b.lepton_transport_decay_candidates);
  a.lepton_vertex_selection_batches=cooperativeAdd(a.lepton_vertex_selection_batches,b.lepton_vertex_selection_batches);
  a.lepton_vertex_interactions_selected=cooperativeAdd(a.lepton_vertex_interactions_selected,b.lepton_vertex_interactions_selected);
  a.lepton_vertex_no_interaction_continuations=cooperativeAdd(a.lepton_vertex_no_interaction_continuations,b.lepton_vertex_no_interaction_continuations);
  a.moliere_trials=cooperativeAdd(a.moliere_trials,b.moliere_trials);
  a.moliere_deflections=cooperativeAdd(a.moliere_deflections,b.moliere_deflections);
  a.moliere_zero_deflections=cooperativeAdd(a.moliere_zero_deflections,b.moliere_zero_deflections);
  a.moliere_newton_iterations=cooperativeAdd(a.moliere_newton_iterations,b.moliere_newton_iterations);
  a.moliere_max_newton_iterations=std::max(a.moliere_max_newton_iterations,b.moliere_max_newton_iterations);
  a.moliere_statistics_reserved=cooperativeAdd(a.moliere_statistics_reserved,b.moliere_statistics_reserved);
  a.physical_photon_wavefronts=cooperativeAdd(a.physical_photon_wavefronts,b.physical_photon_wavefronts);
  a.physical_lepton_wavefronts=cooperativeAdd(a.physical_lepton_wavefronts,b.physical_lepton_wavefronts);
  a.photon_selection_transport_summary_fusions=cooperativeAdd(a.photon_selection_transport_summary_fusions,b.photon_selection_transport_summary_fusions);
  a.lepton_selection_transport_summary_fusions=cooperativeAdd(a.lepton_selection_transport_summary_fusions,b.lepton_selection_transport_summary_fusions);
  a.photon_transport_final_state_summary_fusions=cooperativeAdd(a.photon_transport_final_state_summary_fusions,b.photon_transport_final_state_summary_fusions);
  a.lepton_transport_vertex_summary_fusions=cooperativeAdd(a.lepton_transport_vertex_summary_fusions,b.lepton_transport_vertex_summary_fusions);
  a.lepton_vertex_final_state_summary_fusions=cooperativeAdd(a.lepton_vertex_final_state_summary_fusions,b.lepton_vertex_final_state_summary_fusions);
  a.photon_final_state_endpoint_summary_fusions=cooperativeAdd(a.photon_final_state_endpoint_summary_fusions,b.photon_final_state_endpoint_summary_fusions);
  a.lepton_final_state_endpoint_summary_fusions=cooperativeAdd(a.lepton_final_state_endpoint_summary_fusions,b.lepton_final_state_endpoint_summary_fusions);
  a.pipeline_host_synchronizations_eliminated=cooperativeAdd(a.pipeline_host_synchronizations_eliminated,b.pipeline_host_synchronizations_eliminated);
  a.pipeline_device_to_host_bytes_eliminated=cooperativeAdd(a.pipeline_device_to_host_bytes_eliminated,b.pipeline_device_to_host_bytes_eliminated);
  a.queue_overflows=cooperativeAdd(a.queue_overflows,b.queue_overflows);
  a.cross_species_particles_kept_on_device=cooperativeAdd(a.cross_species_particles_kept_on_device,b.cross_species_particles_kept_on_device);
  a.cross_species_device_to_device_bytes=cooperativeAdd(a.cross_species_device_to_device_bytes,b.cross_species_device_to_device_bytes);
  a.cross_species_host_spills=cooperativeAdd(a.cross_species_host_spills,b.cross_species_host_spills);
  a.cross_species_spill_rebalances=cooperativeAdd(a.cross_species_spill_rebalances,b.cross_species_spill_rebalances);
  a.cross_species_particles_spilled_to_cpu=cooperativeAdd(a.cross_species_particles_spilled_to_cpu,b.cross_species_particles_spilled_to_cpu);
  a.cross_species_low_energy_ordering_checks=cooperativeAdd(a.cross_species_low_energy_ordering_checks,b.cross_species_low_energy_ordering_checks);
  a.wavefront_bucketing_batches=cooperativeAdd(a.wavefront_bucketing_batches,b.wavefront_bucketing_batches);
  a.wavefront_bucketing_particles=cooperativeAdd(a.wavefront_bucketing_particles,b.wavefront_bucketing_particles);
  a.wavefront_bucketing_small_batches=cooperativeAdd(a.wavefront_bucketing_small_batches,b.wavefront_bucketing_small_batches);
  a.wavefront_bucketing_small_particles=cooperativeAdd(a.wavefront_bucketing_small_particles,b.wavefront_bucketing_small_particles);
  a.physical_host_to_device_bytes=cooperativeAdd(a.physical_host_to_device_bytes,b.physical_host_to_device_bytes);
  a.physical_device_to_host_bytes=cooperativeAdd(a.physical_device_to_host_bytes,b.physical_device_to_host_bytes);
  a.kernel_time_ms=cooperativeAdd(a.kernel_time_ms,b.kernel_time_ms);
  a.transfer_time_ms=cooperativeAdd(a.transfer_time_ms,b.transfer_time_ms);
  a.profile.steps=cooperativeAdd(a.profile.steps,b.profile.steps);
  a.profile.deposited_steps=cooperativeAdd(a.profile.deposited_steps,b.profile.deposited_steps);
  a.profile.fixed_point_overflows=cooperativeAdd(a.profile.fixed_point_overflows,b.profile.fixed_point_overflows);
  a.profile.invalid_records=cooperativeAdd(a.profile.invalid_records,b.profile.invalid_records);
  a.profile.host_to_device_bytes=cooperativeAdd(a.profile.host_to_device_bytes,b.profile.host_to_device_bytes);
  a.profile.device_to_host_bytes=cooperativeAdd(a.profile.device_to_host_bytes,b.profile.device_to_host_bytes);
  a.profile.kernel_time_ms=cooperativeAdd(a.profile.kernel_time_ms,b.profile.kernel_time_ms);
  a.profile.transfer_time_ms=cooperativeAdd(a.profile.transfer_time_ms,b.profile.transfer_time_ms);
  a.radio.lepton_tracks=cooperativeAdd(a.radio.lepton_tracks,b.radio.lepton_tracks);
  a.radio.track_observer_pairs=cooperativeAdd(a.radio.track_observer_pairs,b.radio.track_observer_pairs);
  a.radio.fused_track_observer_pairs=cooperativeAdd(a.radio.fused_track_observer_pairs,b.radio.fused_track_observer_pairs);
  a.radio.coreas_contributions=cooperativeAdd(a.radio.coreas_contributions,b.radio.coreas_contributions);
  a.radio.zhs_contributions=cooperativeAdd(a.radio.zhs_contributions,b.radio.zhs_contributions);
  a.radio.zhs_subtracks=cooperativeAdd(a.radio.zhs_subtracks,b.radio.zhs_subtracks);
  a.radio.fixed_point_overflows=cooperativeAdd(a.radio.fixed_point_overflows,b.radio.fixed_point_overflows);
  a.radio.weighted_segment_count=cooperativeAdd(a.radio.weighted_segment_count,b.radio.weighted_segment_count);
  a.radio.track_length_m=cooperativeAdd(a.radio.track_length_m,b.radio.track_length_m);
  a.radio.weighted_track_length_m=cooperativeAdd(a.radio.weighted_track_length_m,b.radio.weighted_track_length_m);
  a.radio.electron_weighted_track_length_m=cooperativeAdd(a.radio.electron_weighted_track_length_m,b.radio.electron_weighted_track_length_m);
  a.radio.positron_weighted_track_length_m=cooperativeAdd(a.radio.positron_weighted_track_length_m,b.radio.positron_weighted_track_length_m);
  a.radio.signed_charge_weighted_track_length_m=cooperativeAdd(a.radio.signed_charge_weighted_track_length_m,b.radio.signed_charge_weighted_track_length_m);
  a.radio.energy_weighted_track_length_GeV_m=cooperativeAdd(a.radio.energy_weighted_track_length_GeV_m,b.radio.energy_weighted_track_length_GeV_m);
  a.radio.maximum_segment_length_m=std::max(a.radio.maximum_segment_length_m,b.radio.maximum_segment_length_m);
  a.radio.weighted_direction_change_rad=cooperativeAdd(a.radio.weighted_direction_change_rad,b.radio.weighted_direction_change_rad);
  a.radio.weighted_direction_change_squared_rad2=cooperativeAdd(a.radio.weighted_direction_change_squared_rad2,b.radio.weighted_direction_change_squared_rad2);
  a.radio.weighted_beta_deficit_track_length_m=cooperativeAdd(a.radio.weighted_beta_deficit_track_length_m,b.radio.weighted_beta_deficit_track_length_m);
  a.radio.weighted_time_residual_s=cooperativeAdd(a.radio.weighted_time_residual_s,b.radio.weighted_time_residual_s);
  a.radio.maximum_direction_change_rad=std::max(a.radio.maximum_direction_change_rad,b.radio.maximum_direction_change_rad);
  for(std::size_t i=0;i<a.radio.signed_charge_weighted_direction_change.size();++i) a.radio.signed_charge_weighted_direction_change[i]=cooperativeAdd(a.radio.signed_charge_weighted_direction_change[i],b.radio.signed_charge_weighted_direction_change[i]);
  for(std::size_t i=0;i<a.radio.weighted_track_length_by_kinetic_energy_m.size();++i) a.radio.weighted_track_length_by_kinetic_energy_m[i]=cooperativeAdd(a.radio.weighted_track_length_by_kinetic_energy_m[i],b.radio.weighted_track_length_by_kinetic_energy_m[i]);
  a.radio.host_to_device_bytes=cooperativeAdd(a.radio.host_to_device_bytes,b.radio.host_to_device_bytes);
  a.radio.device_to_host_bytes=cooperativeAdd(a.radio.device_to_host_bytes,b.radio.device_to_host_bytes);
  a.radio.device_time_ms=cooperativeAdd(a.radio.device_time_ms,b.radio.device_time_ms);
  a.radio.kernel_time_ms=cooperativeAdd(a.radio.kernel_time_ms,b.radio.kernel_time_ms);
  a.radio.transfer_time_ms=cooperativeAdd(a.radio.transfer_time_ms,b.radio.transfer_time_ms);
  a.radio.input_slot_waits=cooperativeAdd(a.radio.input_slot_waits,b.radio.input_slot_waits);
  a.radio.input_slot_host_wait_time_ms=cooperativeAdd(a.radio.input_slot_host_wait_time_ms,b.radio.input_slot_host_wait_time_ms);
  a.radio.track_precompute_batches=cooperativeAdd(a.radio.track_precompute_batches,b.radio.track_precompute_batches);
  a.radio.track_precomputed_records=cooperativeAdd(a.radio.track_precomputed_records,b.radio.track_precomputed_records);
  a.radio.direct_projection_batches=cooperativeAdd(a.radio.direct_projection_batches,b.radio.direct_projection_batches);
  a.radio.direct_projection_records=cooperativeAdd(a.radio.direct_projection_records,b.radio.direct_projection_records);
  a.radio.projection_tiles=cooperativeAdd(a.radio.projection_tiles,b.radio.projection_tiles);
  a.radio.maximum_track_batch=std::max(a.radio.maximum_track_batch,b.radio.maximum_track_batch);
  a.radio.track_precompute_device_time_ms=cooperativeAdd(a.radio.track_precompute_device_time_ms,b.radio.track_precompute_device_time_ms);
  a.radio.projection_device_time_ms=cooperativeAdd(a.radio.projection_device_time_ms,b.radio.projection_device_time_ms);
  a.transfer_timing.operations=cooperativeAdd(a.transfer_timing.operations,b.transfer_timing.operations);
  a.transfer_timing.host_to_device_operations=cooperativeAdd(a.transfer_timing.host_to_device_operations,b.transfer_timing.host_to_device_operations);
  a.transfer_timing.device_to_host_operations=cooperativeAdd(a.transfer_timing.device_to_host_operations,b.transfer_timing.device_to_host_operations);
  a.transfer_timing.device_to_device_operations=cooperativeAdd(a.transfer_timing.device_to_device_operations,b.transfer_timing.device_to_device_operations);
  a.transfer_timing.host_api_time_ms=cooperativeAdd(a.transfer_timing.host_api_time_ms,b.transfer_timing.host_api_time_ms);
  a.transfer_timing.device_copy_time_ms=cooperativeAdd(a.transfer_timing.device_copy_time_ms,b.transfer_timing.device_copy_time_ms);
  a.transfer_timing.host_wait_upper_bound_ms=cooperativeAdd(a.transfer_timing.host_wait_upper_bound_ms,b.transfer_timing.host_wait_upper_bound_ms);
  a.synchronization_timing.physical_pipeline_waits=cooperativeAdd(a.synchronization_timing.physical_pipeline_waits,b.synchronization_timing.physical_pipeline_waits);
  a.synchronization_timing.profile_input_waits=cooperativeAdd(a.synchronization_timing.profile_input_waits,b.synchronization_timing.profile_input_waits);
  a.synchronization_timing.physical_pipeline_wait_time_ms=cooperativeAdd(a.synchronization_timing.physical_pipeline_wait_time_ms,b.synchronization_timing.physical_pipeline_wait_time_ms);
  a.synchronization_timing.profile_input_wait_time_ms=cooperativeAdd(a.synchronization_timing.profile_input_wait_time_ms,b.synchronization_timing.profile_input_wait_time_ms);
  a.lepton_pipeline_timing.wavefronts=cooperativeAdd(a.lepton_pipeline_timing.wavefronts,b.lepton_pipeline_timing.wavefronts);
  a.lepton_pipeline_timing.selection_ms=cooperativeAdd(a.lepton_pipeline_timing.selection_ms,b.lepton_pipeline_timing.selection_ms);
  a.lepton_pipeline_timing.transport_ms=cooperativeAdd(a.lepton_pipeline_timing.transport_ms,b.lepton_pipeline_timing.transport_ms);
  a.lepton_pipeline_timing.transport_physics_ms=cooperativeAdd(a.lepton_pipeline_timing.transport_physics_ms,b.lepton_pipeline_timing.transport_physics_ms);
  a.lepton_pipeline_timing.moliere_ms=cooperativeAdd(a.lepton_pipeline_timing.moliere_ms,b.lepton_pipeline_timing.moliere_ms);
  a.lepton_pipeline_timing.transport_control_ms=cooperativeAdd(a.lepton_pipeline_timing.transport_control_ms,b.lepton_pipeline_timing.transport_control_ms);
  a.lepton_pipeline_timing.transport_compaction_ms=cooperativeAdd(a.lepton_pipeline_timing.transport_compaction_ms,b.lepton_pipeline_timing.transport_compaction_ms);
  a.lepton_pipeline_timing.interaction_extraction_ms=cooperativeAdd(a.lepton_pipeline_timing.interaction_extraction_ms,b.lepton_pipeline_timing.interaction_extraction_ms);
  a.lepton_pipeline_timing.vertex_selection_ms=cooperativeAdd(a.lepton_pipeline_timing.vertex_selection_ms,b.lepton_pipeline_timing.vertex_selection_ms);
  a.lepton_pipeline_timing.final_state_ms=cooperativeAdd(a.lepton_pipeline_timing.final_state_ms,b.lepton_pipeline_timing.final_state_ms);
  a.lepton_pipeline_timing.final_state_classification_ms=cooperativeAdd(a.lepton_pipeline_timing.final_state_classification_ms,b.lepton_pipeline_timing.final_state_classification_ms);
  a.lepton_pipeline_timing.final_state_scan_ms=cooperativeAdd(a.lepton_pipeline_timing.final_state_scan_ms,b.lepton_pipeline_timing.final_state_scan_ms);
  a.lepton_pipeline_timing.final_state_summary_ms=cooperativeAdd(a.lepton_pipeline_timing.final_state_summary_ms,b.lepton_pipeline_timing.final_state_summary_ms);
  a.lepton_pipeline_timing.final_state_write_ms=cooperativeAdd(a.lepton_pipeline_timing.final_state_write_ms,b.lepton_pipeline_timing.final_state_write_ms);
  a.lepton_pipeline_timing.endpoint_compaction_ms=cooperativeAdd(a.lepton_pipeline_timing.endpoint_compaction_ms,b.lepton_pipeline_timing.endpoint_compaction_ms);
  a.lepton_pipeline_timing.post_endpoint_ms=cooperativeAdd(a.lepton_pipeline_timing.post_endpoint_ms,b.lepton_pipeline_timing.post_endpoint_ms);
  return a;
}
} // namespace corsika::accelerator::em::detail
