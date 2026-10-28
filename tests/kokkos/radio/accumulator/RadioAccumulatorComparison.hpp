/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include "RadioAccumulatorChecks.hpp"
#include "RadioAccumulatorReference.hpp"
namespace radio_accumulator_test {
  namespace rr = ::radio_reference;
  namespace ar = ::radio_accumulator_reference;
  using reference_type = ar::KokkosRadioAccumulator<execution_type>;
  inline rr::RadioObserverSnapshot to_reference(cr::RadioObserverSnapshot const& a) {
    rr::RadioObserverSnapshot b;
    for (unsigned d = 0; d < 3; ++d) b.position_m[d] = a.positionM_[d];
    b.start_time_s = a.startTimeS_;
    b.duration_s = a.durationS_;
    b.sample_rate_Hz = a.sampleRateHz_;
    b.number_of_bins = a.numberOfBins_;
    return b;
  }
  inline rr::GpuRadioConfig to_reference(cr::RadioConfig const& a) {
    rr::GpuRadioConfig b;
    b.enabled = a.enabled_;
    b.coreas_enabled = a.coreasEnabled_;
    b.zhs_enabled = a.zhsEnabled_;
    b.deterministic = a.deterministic_;
    b.fixed_point_field_limit_V_per_m = a.fixedPointFieldLimitVPerM_;
    b.track_diagnostics = a.trackDiagnostics_;
    b.zhs_subtrack_refinement = a.zhsSubtrackRefinement_;
    b.propagation.homogeneous_refractive_index =
        a.propagation_.homogeneousRefractiveIndex_;
    b.propagation.minimum_height_m = a.propagation_.minimumHeightM_;
    b.propagation.maximum_height_m = a.propagation_.maximumHeightM_;
    b.propagation.step_m = a.propagation_.stepM_;
    b.propagation.inverse_step_per_m = a.propagation_.inverseStepPerM_;
    b.propagation.slope_refractivity_lower = a.propagation_.slopeRefractivityLower_;
    b.propagation.slope_integrated_refractivity_lower =
        a.propagation_.slopeIntegratedRefractivityLower_;
    b.propagation.slope_refractivity_upper = a.propagation_.slopeRefractivityUpper_;
    b.propagation.slope_integrated_refractivity_upper =
        a.propagation_.slopeIntegratedRefractivityUpper_;
    b.propagation.refractivity = a.propagation_.refractivity_;
    b.propagation.integrated_refractivity = a.propagation_.integratedRefractivity_;
    for (auto const& o : a.coreasObservers_)
      b.coreas_observers.push_back(to_reference(o));
    for (auto const& o : a.zhsObservers_) b.zhs_observers.push_back(to_reference(o));
    return b;
  }
  inline rr::gpu::em::LeptonTransportRecord to_reference(record_type const& a) {
    rr::gpu::em::LeptonTransportRecord b{};
    b.start.pid = a.start_.pid_;
    b.start.energy_GeV = a.start_.energyGeV_;
    b.start.weight = a.start_.weight_;
    b.start.time_s = a.start_.timeS_;
    for (unsigned d = 0; d < 3; ++d) {
      b.start.position_m[d] = a.start_.positionM_[d];
      b.start.direction[d] = a.start_.direction_[d];
    }
    b.end.pid = a.end_.pid_;
    b.end.energy_GeV = a.end_.energyGeV_;
    b.end.weight = a.end_.weight_;
    b.end.time_s = a.end_.timeS_;
    for (unsigned d = 0; d < 3; ++d) {
      b.end.position_m[d] = a.end_.positionM_[d];
      b.end.direction[d] = a.end_.direction_[d];
    }
    return b;
  }
  inline ar::em::KokkosTuningParameters to_reference(cp::TuningParameters const& a) {
    ar::em::KokkosTuningParameters b;
    b.batch_size = a.batchSize_;
    b.chunk_size = a.chunkSize_;
    b.team_size = a.teamSize_;
    b.track_tile_size = a.trackTileSize_;
    b.observer_tile_size = a.observerTileSize_;
    b.device_queues = a.deviceQueues_;
    return b;
  }
  inline void check_statistics(cr::RadioStatistics const& a,
                               rr::GpuRadioStatistics const& b) {
    INFO("statistics: lepton_tracks");
    CHECK(a.leptonTracks_ == b.lepton_tracks);
    INFO("statistics: track_observer_pairs");
    CHECK(a.trackObserverPairs_ == b.track_observer_pairs);
    INFO("statistics: fused_track_observer_pairs");
    CHECK(a.fusedTrackObserverPairs_ == b.fused_track_observer_pairs);
    INFO("statistics: coreas_contributions");
    CHECK(a.coreasContributions_ == b.coreas_contributions);
    INFO("statistics: zhs_contributions");
    CHECK(a.zhsContributions_ == b.zhs_contributions);
    INFO("statistics: zhs_subtracks");
    CHECK(a.zhsSubtracks_ == b.zhs_subtracks);
    INFO("statistics: fixed_point_overflows");
    CHECK(a.fixedPointOverflows_ == b.fixed_point_overflows);
    INFO("statistics: track_diagnostics_enabled");
    CHECK(a.trackDiagnosticsEnabled_ == b.track_diagnostics_enabled);
    INFO("statistics: weighted_segment_count");
    CHECK(close(a.weightedSegmentCount_, b.weighted_segment_count));
    INFO("statistics: track_length_m");
    CHECK(close(a.trackLengthM_, b.track_length_m));
    INFO("statistics: weighted_track_length_m");
    CHECK(close(a.weightedTrackLengthM_, b.weighted_track_length_m));
    INFO("statistics: electron_weighted_track_length_m");
    CHECK(close(a.electronWeightedTrackLengthM_, b.electron_weighted_track_length_m));
    INFO("statistics: positron_weighted_track_length_m");
    CHECK(close(a.positronWeightedTrackLengthM_, b.positron_weighted_track_length_m));
    INFO("statistics: signed_charge_weighted_track_length_m");
    CHECK(close(a.signedChargeWeightedTrackLengthM_,
                b.signed_charge_weighted_track_length_m));
    INFO("statistics: energy_weighted_track_length_GeV_m");
    CHECK(close(a.energyWeightedTrackLengthGeVM_, b.energy_weighted_track_length_GeV_m));
    INFO("statistics: maximum_segment_length_m");
    CHECK(close(a.maximumSegmentLengthM_, b.maximum_segment_length_m));
    INFO("statistics: weighted_direction_change_rad");
    CHECK(close(a.weightedDirectionChangeRad_, b.weighted_direction_change_rad));
    INFO("statistics: weighted_direction_change_squared_rad2");
    CHECK(close(a.weightedDirectionChangeSquaredRad2_,
                b.weighted_direction_change_squared_rad2));
    INFO("statistics: weighted_beta_deficit_track_length_m");
    CHECK(close(a.weightedBetaDeficitTrackLengthM_,
                b.weighted_beta_deficit_track_length_m));
    INFO("statistics: weighted_time_residual_s");
    CHECK(close(a.weightedTimeResidualS_, b.weighted_time_residual_s));
    INFO("statistics: maximum_direction_change_rad");
    CHECK(close(a.maximumDirectionChangeRad_, b.maximum_direction_change_rad));
    INFO("statistics: signed_charge_weighted_direction_change");
    for (std::size_t i = 0; i < a.signedChargeWeightedDirectionChange_.size(); ++i)
      CHECK(close(a.signedChargeWeightedDirectionChange_[i],
                  b.signed_charge_weighted_direction_change[i]));
    INFO("statistics: weighted_track_length_by_kinetic_energy_m");
    for (std::size_t i = 0; i < a.weightedTrackLengthByKineticEnergyM_.size(); ++i)
      CHECK(close(a.weightedTrackLengthByKineticEnergyM_[i],
                  b.weighted_track_length_by_kinetic_energy_m[i]));
    INFO("statistics: device_bytes");
    CHECK(a.deviceBytes_ == b.device_bytes);
    INFO("statistics: host_to_device_bytes");
    CHECK(a.hostToDeviceBytes_ == b.host_to_device_bytes);
    INFO("statistics: host_tracks_enqueued");
    CHECK(a.hostTracksEnqueued_ == b.host_tracks_enqueued);
    INFO("statistics: host_track_batches");
    CHECK(a.hostTrackBatches_ == b.host_track_batches);
    INFO("statistics: device_to_host_bytes");
    CHECK(a.deviceToHostBytes_ == b.device_to_host_bytes);
    INFO("statistics: device_time_ms");
    CHECK(close(a.deviceTimeMs_, b.device_time_ms));
    INFO("statistics: kernel_time_ms");
    CHECK(close(a.kernelTimeMs_, b.kernel_time_ms));
    INFO("statistics: transfer_time_ms");
    CHECK(close(a.transferTimeMs_, b.transfer_time_ms));
    INFO("statistics: input_slot_waits");
    CHECK(a.inputSlotWaits_ == b.input_slot_waits);
    INFO("statistics: input_slot_host_wait_time_ms");
    CHECK(close(a.inputSlotHostWaitTimeMs_, b.input_slot_host_wait_time_ms));
    INFO("statistics: track_precompute_enabled");
    CHECK(a.trackPrecomputeEnabled_ == b.track_precompute_enabled);
    INFO("statistics: track_tile_size");
    CHECK(a.trackTileSize_ == b.track_tile_size);
    INFO("statistics: observer_tile_size");
    CHECK(a.observerTileSize_ == b.observer_tile_size);
    INFO("statistics: track_precompute_batches");
    CHECK(a.trackPrecomputeBatches_ == b.track_precompute_batches);
    INFO("statistics: track_precomputed_records");
    CHECK(a.trackPrecomputedRecords_ == b.track_precomputed_records);
    INFO("statistics: direct_projection_batches");
    CHECK(a.directProjectionBatches_ == b.direct_projection_batches);
    INFO("statistics: direct_projection_records");
    CHECK(a.directProjectionRecords_ == b.direct_projection_records);
    INFO("statistics: projection_tiles");
    CHECK(a.projectionTiles_ == b.projection_tiles);
    INFO("statistics: track_workspace_bytes");
    CHECK(a.trackWorkspaceBytes_ == b.track_workspace_bytes);
    INFO("statistics: maximum_track_batch");
    CHECK(a.maximumTrackBatch_ == b.maximum_track_batch);
    INFO("statistics: track_precompute_device_time_ms");
    CHECK(close(a.trackPrecomputeDeviceTimeMs_, b.track_precompute_device_time_ms));
    INFO("statistics: projection_device_time_ms");
    CHECK(close(a.projectionDeviceTimeMs_, b.projection_device_time_ms));
  }
  inline void check_snapshot(cr::FixedRadioSnapshot const& a,
                             ar::detail::FixedRadioSnapshot const& b) {
    CHECK(a.showerIdentity_ == b.shower_identity);
    CHECK(static_cast<unsigned>(a.endpoint_) == static_cast<unsigned>(b.endpoint));
    CHECK(ar::detail::sameRadioPhysics(to_reference(a.config_), b.config));
    CHECK(a.coreas_ == b.coreas);
    CHECK(a.zhs_ == b.zhs);
    CHECK(a.counters_.coreasContributions_ == b.counters.coreas_contributions);
    CHECK(a.counters_.zhsContributions_ == b.counters.zhs_contributions);
    CHECK(a.counters_.zhsSubtracks_ == b.counters.zhs_subtracks);
    CHECK(a.counters_.validTracks_ == b.counters.valid_tracks);
    CHECK(a.counters_.fixedPointOverflows_ == b.counters.fixed_point_overflows);
    CHECK(a.counters_.weightedSegmentCount_ == b.counters.weighted_segment_count);
    CHECK(a.counters_.trackLengthM_ == b.counters.track_length_m);
    CHECK(a.counters_.weightedTrackLengthM_ == b.counters.weighted_track_length_m);
    CHECK(a.counters_.electronWeightedTrackLengthM_ ==
          b.counters.electron_weighted_track_length_m);
    CHECK(a.counters_.positronWeightedTrackLengthM_ ==
          b.counters.positron_weighted_track_length_m);
    CHECK(a.counters_.signedChargeWeightedTrackLengthM_ ==
          b.counters.signed_charge_weighted_track_length_m);
    CHECK(a.counters_.energyWeightedTrackLengthGeVM_ ==
          b.counters.energy_weighted_track_length_GeV_m);
    CHECK(a.counters_.maximumSegmentLengthM_ == b.counters.maximum_segment_length_m);
    CHECK(a.counters_.weightedDirectionChangeRad_ ==
          b.counters.weighted_direction_change_rad);
    CHECK(a.counters_.weightedDirectionChangeSquaredRad2_ ==
          b.counters.weighted_direction_change_squared_rad2);
    CHECK(a.counters_.weightedBetaDeficitTrackLengthM_ ==
          b.counters.weighted_beta_deficit_track_length_m);
    CHECK(a.counters_.weightedTimeResidualS_ == b.counters.weighted_time_residual_s);
    CHECK(a.counters_.maximumDirectionChangeRad_ ==
          b.counters.maximum_direction_change_rad);
    for (unsigned i = 0; i < 3; ++i)
      CHECK(a.counters_.signedChargeWeightedDirectionChange_[i] ==
            b.counters.signed_charge_weighted_direction_change[i]);
    for (unsigned i = 0; i < 15; ++i)
      CHECK(a.counters_.weightedTrackLengthByKineticEnergyM_[i] ==
            b.counters.weighted_track_length_by_kinetic_energy_m[i]);
  }
} // namespace radio_accumulator_test
