/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <std::size_t ComponentCapacity>
  KOKKOS_INLINE_FUNCTION bool apply_moliere_scattering(
      MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
      bool const enabled, std::uint64_t const random_seed, std::uint64_t const shower_id,
      LeptonTransportRecord& record, FallbackRequest& fallback) {
    if (!enabled) return true;
    auto const& start = record.start_;
    RandomNumberKey const first_key{random_seed,
                                    shower_id,
                                    start.historyId_,
                                    start.stepId_,
                                    ContinuousScatteringRandomProcessId,
                                    MoliereFirstAngleDrawId};
    RandomNumberKey const second_key{random_seed,
                                     shower_id,
                                     start.historyId_,
                                     start.stepId_,
                                     ContinuousScatteringRandomProcessId,
                                     MoliereSecondAngleDrawId};
    RandomNumberKey const azimuth_key{random_seed,
                                      shower_id,
                                      start.historyId_,
                                      start.stepId_,
                                      ContinuousScatteringRandomProcessId,
                                      MoliereAzimuthDrawId};
    record.multipleScatteringFirstUniform_ = uniform_open01(first_key);
    record.multipleScatteringSecondUniform_ = uniform_open01(second_key);
    record.multipleScatteringAzimuthUniform_ = uniform_open01(azimuth_key);
    auto const scattering =
        sample_moliere_scattering_angle_2d_for_capacity<ComponentCapacity>(
            snapshot, interpolation, record.traversedGrammageGPerCm2_,
            record.start_.energyGeV_ * 1000., record.end_.energyGeV_ * 1000.,
            record.multipleScatteringFirstUniform_,
            record.multipleScatteringSecondUniform_, false);
    record.multipleScatteringStatus_ = static_cast<std::uint16_t>(scattering.status_);
    record.multipleScatteringIterations_ = scattering.iterations_;
    record.multipleScatteringAngleRad_ = scattering.angleRad_;
    if (scattering.status_ == MoliereStatus::NoDeflection) return true;
    if (scattering.status_ != MoliereStatus::Success) {
      fallback = lepton_continuous_detail::make_transport_fallback(
          record.interaction_, FallbackReason::MoliereSamplingFailed);
      fallback.finalStateUniform_ = record.multipleScatteringFirstUniform_;
      fallback.finalStateDrawId_ = MoliereFirstAngleDrawId;
      return false;
    }
    auto const direction = apply_moliere_direction_cpu_step_compatible(
        record.start_.direction_, record.end_.direction_, scattering.angleRad_,
        record.multipleScatteringAzimuthUniform_);
    if (direction.status_ != MoliereStatus::Success) {
      fallback = lepton_continuous_detail::make_transport_fallback(
          record.interaction_, FallbackReason::MoliereSamplingFailed);
      fallback.finalStateUniform_ = record.multipleScatteringAzimuthUniform_;
      fallback.finalStateDrawId_ = MoliereAzimuthDrawId;
      return false;
    }
    for (int axis = 0; axis < 3; ++axis)
      record.end_.direction_[axis] = direction.direction_[axis];
    record.multipleScatteringApplied_ = 1;
    return true;
  }

  template <std::size_t ComponentCapacity>
  KOKKOS_INLINE_FUNCTION std::uint32_t apply_moliere_scattering_stage(
      MoliereSnapshot const& electron_snapshot, MoliereSnapshot const& muon_snapshot,
      MoliereInterpolationView const& interpolation, bool const muon_snapshot_available,
      std::uint64_t const random_seed, std::uint64_t const shower_id,
      LeptonTransportRecord& record, FallbackRequest& fallback,
      std::uint32_t const transport_state) {
    if (transport_state == 2u) return 0u;
    if (transport_state != 0u) return transport_state;
    auto const muon = is_muon_pid(record.start_.pid_);
    if (muon && !muon_snapshot_available) {
      fallback = lepton_continuous_detail::make_transport_fallback(
          record.interaction_, FallbackReason::MoliereParametersUnavailable);
      return 1u;
    }
    auto const& particle_snapshot = muon ? muon_snapshot : electron_snapshot;
    if (!apply_moliere_scattering<ComponentCapacity>(particle_snapshot, interpolation,
                                                     true, random_seed, shower_id, record,
                                                     fallback)) {
      return 1u;
    }
    if (record.limit_ == LeptonTransportLimit::InteractionCandidate ||
        record.limit_ == LeptonTransportLimit::DecayCandidate) {
      record.interaction_.particle_ = record.end_;
    }
    return 0u;
  }

} // namespace corsika::kokkos
