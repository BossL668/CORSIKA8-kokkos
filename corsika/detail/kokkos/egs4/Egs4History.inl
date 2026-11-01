/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4PhotonTransport.inl>

namespace corsika::kokkos::egs4 {
  // Native, per-particle controller. The caller asks geometry for a BoundaryLimit
  // between planning and finishing, then handles observations and returned children.
  // Frame handling is a policy; the default is local flat. This is not yet the
  // production C8 atmosphere/stack adapter.
  enum class PlanAction { error, transport, collision, energy_cut };
  enum class HistoryAction {
    error,
    active,
    boundary,
    escaped,
    replaced,
    needs_host_channel
  };
  enum class HistoryError {
    none,
    invalid_input,
    table_range,
    random_failure,
    transport_failure,
    sampling_failure
  };
  struct HistoryPlan {
    PlanAction action_{PlanAction::error};
    HistoryError error_{HistoryError::invalid_input};
    TrackState particle_;
    ElectronPlan electron_;
    PhotonPlan photon_;
    double proposedCm_{};
  };
  struct HistoryOutcome {
    HistoryAction action_{HistoryAction::error};
    HistoryError error_{HistoryError::invalid_input};
    TrackState particle_;
    TrackState children_[2]{};
    int childCount_{};
    Channel channel_{Channel::invalid};
    BoundaryKind boundary_{BoundaryKind::none};
    bool hasTrack_{};
    Direction transportEndDirection_;
    double geometricStepCm_{}, lossPathCm_{}, timePathCm_{};
    double continuousDepositMeV_{}, vertexDepositMeV_{};
    DepositKind vertexDepositKind_{DepositKind::none};
  };
  KOKKOS_INLINE_FUNCTION bool valid_settings(ModelView model, TransportSettings s) {
    return finite(s.electronCutTotalMeV_) &&
           s.electronCutTotalMeV_ >= model.tables_.medium_.ae_ &&
           s.electronCutTotalMeV_ >=
               model.tables_.medium_.ap_ + model.thresholds_.massMeV_ &&
           finite(s.photonCutMeV_) && s.photonCutMeV_ >= model.tables_.medium_.ap_ &&
           finite(s.stepfc_) && s.stepfc_ > 0. && finite(s.vacuumDistanceCm_) &&
           s.vacuumDistanceCm_ > 0. && finite(s.speedCmS_) && s.speedCmS_ > 0.;
  }
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION HistoryPlan plan_history(ModelView model, TrackState state,
                                                  AirRegion region, MagneticField field,
                                                  TransportSettings settings,
                                                  TRandom& rng,
                                                  bool reuse_query = false) {
    HistoryPlan p;
    p.particle_ = state;
    if (!valid_track(state) || !valid_region(region) ||
        !valid_settings(model, settings) || state.pendingChannel_ != Channel::invalid ||
        (state.pdg_ != 22 &&
         (!valid_field(field) || state.energyMeV_ < model.thresholds_.massMeV_)))
      return p;
    p.error_ = HistoryError::none;
    double cut =
        state.pdg_ == 22 ? settings.photonCutMeV_ : settings.electronCutTotalMeV_;
    if (state.energyMeV_ <= cut) {
      p.action_ = PlanAction::energy_cut;
      return p;
    }
    if (state.pdg_ == 22) {
      auto q = photon_query(model.tables_, state.energyMeV_);
      if (q.status_ != Status::success) {
        p.error_ = HistoryError::table_range;
        return p;
      }
      if (state.photonClock_.initialized_) {
        if (!finite(state.photonClock_.remainingMfp_) ||
            state.photonClock_.remainingMfp_ < 0.) {
          p.error_ = HistoryError::invalid_input;
          return p;
        }
        if (state.photonClock_.remainingMfp_ <= 1.e-10) {
          p.action_ = PlanAction::collision;
          return p;
        }
      } else if (start_photon_clock(q, rng, state.photonClock_) !=
                 ClockStatus::transporting) {
        p.error_ = HistoryError::random_failure;
        return p;
      }
      // A freshly sampled very small path still follows PHOTON's minimum step.
      p.photon_ = prepare_photon_segment(model, state, region, settings,
                                         reuse_query ? &q : nullptr);
      if (p.photon_.status_ != Status::success) {
        p.error_ = HistoryError::transport_failure;
        return p;
      }
      p.proposedCm_ = p.photon_.proposedCm_;
    } else {
      auto q = electron_query(model.tables_, state.energyMeV_, model.thresholds_.massMeV_,
                              state.pdg_ == 11 ? -1 : 1);
      if (q.status_ != Status::success) {
        p.error_ = HistoryError::table_range;
        return p;
      }
      if (state.clock_.initialized_) {
        if (!finite(state.clock_.remainingMfp_) || state.clock_.remainingMfp_ < 0. ||
            !finite(state.clock_.sampledRatePerCm_)) {
          p.error_ = HistoryError::invalid_input;
          return p;
        }
        if (state.clock_.remainingMfp_ < 1.e-10) {
          p.action_ = PlanAction::collision;
          return p;
        }
      } else if (start_electron_clock(q, rng, state.clock_) !=
                 ClockStatus::transporting) {
        p.error_ = HistoryError::random_failure;
        return p;
      }
      p.electron_ = prepare_electron_segment(model, state, region, field, settings,
                                             reuse_query ? &q : nullptr);
      if (p.electron_.status_ != Status::success) {
        p.error_ = HistoryError::transport_failure;
        return p;
      }
      p.proposedCm_ = p.electron_.proposedCm_;
    }
    p.particle_ = state;
    p.action_ = PlanAction::transport;
    return p;
  }
  KOKKOS_INLINE_FUNCTION void apply_local_outcome(HistoryOutcome& r, LocalOutcome local) {
    if (local.secondaries_.status_ != InteractionStatus::success ||
        local.secondaries_.count_ < 0 || local.secondaries_.count_ > 2) {
      r.action_ = HistoryAction::error;
      r.error_ = local.secondaries_.status_ == InteractionStatus::random_failure
                     ? HistoryError::random_failure
                     : HistoryError::sampling_failure;
      return;
    }
    r.vertexDepositMeV_ = local.depositMeV_;
    r.vertexDepositKind_ = local.depositKind_;
    r.childCount_ = local.secondaries_.count_;
    for (int i = 0; i < r.childCount_; ++i) {
      auto child = local.secondaries_.particle_[i];
      TrackState s;
      s.pdg_ = child.pdg_;
      s.energyMeV_ = child.energyMeV_;
      s.positionCm_ = r.particle_.positionCm_;
      s.timeS_ = r.particle_.timeS_;
      s.direction_ = child.direction_;
      r.children_[i] = s; // new histories, no inherited clocks or pending channel
    }
    r.action_ = HistoryAction::replaced;
    r.error_ = HistoryError::none;
  }
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION void finish_energy_cut(ModelView model, HistoryOutcome& r,
                                                TRandom& rng) {
    if (r.particle_.pdg_ == 22) {
      LocalOutcome local;
      local.secondaries_.status_ = InteractionStatus::success;
      local.depositMeV_ = r.particle_.energyMeV_;
      local.depositKind_ = DepositKind::subcut_kinetic;
      apply_local_outcome(r, local);
    } else
      apply_local_outcome(r,
                          stop_lepton(r.particle_.energyMeV_, model.thresholds_.massMeV_,
                                      r.particle_.pdg_, rng));
  }
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION void finish_collision(ModelView model, AirRegion region,
                                               HistoryOutcome& r, TRandom& rng) {
    if (r.particle_.pdg_ != 22) {
      auto q =
          electron_query(model.tables_, r.particle_.energyMeV_,
                         model.thresholds_.massMeV_, r.particle_.pdg_ == 11 ? -1 : 1);
      auto accepted = accept_electron_collision(r.particle_.clock_, q, rng);
      if (accepted == ClockStatus::restart) {
        r.action_ = HistoryAction::active;
        r.error_ = HistoryError::none;
        return;
      }
      if (accepted != ClockStatus::accepted) {
        r.action_ = HistoryAction::error;
        r.error_ = accepted == ClockStatus::random_failure ? HistoryError::random_failure
                                                           : HistoryError::invalid_input;
        return;
      }
    } else
      r.particle_.photonClock_ = {};
    double rho = region.referenceDensityGCm3_ *
                 ::exp(r.particle_.positionCm_.z_ * region.inverseScaleHeightCm_);
    auto vertex = select_and_sample(
        model, {r.particle_.pdg_, r.particle_.energyMeV_, rho, r.particle_.direction_},
        rng);
    r.channel_ = vertex.channel_;
    if (vertex.disposition_ == Disposition::needs_host_channel) {
      r.particle_.pendingChannel_ = vertex.channel_;
      r.action_ = HistoryAction::needs_host_channel;
      r.error_ = HistoryError::none;
    } else if (vertex.disposition_ == Disposition::keep_parent) {
      // Fictitious or LPM-suppressed vertex: retain particle at the new position,
      // but sample a NEW interaction distance on its next native history step.
      r.action_ = HistoryAction::active;
      r.error_ = HistoryError::none;
    } else if (vertex.disposition_ == Disposition::replace_parent)
      apply_local_outcome(r, vertex.local_);
    else {
      r.action_ = HistoryAction::error;
      r.error_ = vertex.local_.secondaries_.status_ == InteractionStatus::random_failure
                     ? HistoryError::random_failure
                     : HistoryError::sampling_failure;
    }
  }
  template <class TRandom, class TFrameStep = FlatFrameStep>
  KOKKOS_INLINE_FUNCTION HistoryOutcome finish_history(ModelView model, HistoryPlan p,
                                                       AirRegion region,
                                                       MagneticField field,
                                                       TransportSettings settings,
                                                       BoundaryLimit limit, TRandom& rng,
                                                       TFrameStep frame = {}) {
    HistoryOutcome r;
    r.particle_ = p.particle_;
    r.error_ = p.error_;
    if (p.action_ == PlanAction::error) return r;
    if (!valid_track(p.particle_) || !valid_region(region) ||
        !valid_settings(model, settings) ||
        p.particle_.pendingChannel_ != Channel::invalid) {
      r.error_ = HistoryError::invalid_input;
      return r;
    }
    if (p.action_ == PlanAction::energy_cut) {
      finish_energy_cut(model, r, rng);
      return r;
    }
    if (p.action_ == PlanAction::collision) {
      finish_collision(model, region, r, rng);
      return r;
    }
    TransportStatus status;
    if (p.particle_.pdg_ == 22) {
      auto segment =
          finish_photon_segment(p.particle_, region, settings, p.photon_, limit, frame);
      status = segment.status_;
      r.particle_ = segment.particle_;
      r.boundary_ = segment.boundary_;
      r.geometricStepCm_ = segment.geometricStepCm_;
      r.timePathCm_ = segment.geometricStepCm_;
      r.transportEndDirection_ = segment.transportEndDirection_;
    } else {
      auto segment = finish_electron_segment(model, p.particle_, region, field, settings,
                                             p.electron_, limit, rng, frame);
      status = segment.status_;
      r.particle_ = segment.particle_;
      r.boundary_ = segment.boundary_;
      r.geometricStepCm_ = segment.geometricStepCm_;
      r.lossPathCm_ = segment.lossPathCm_;
      r.timePathCm_ = segment.timePathCm_;
      r.continuousDepositMeV_ = segment.depositMeV_;
      r.transportEndDirection_ = segment.transportEndDirection_;
    }
    if (status == TransportStatus::invalid || status == TransportStatus::random_failure ||
        status == TransportStatus::sampling_failure) {
      r.error_ = status == TransportStatus::random_failure
                     ? HistoryError::random_failure
                     : HistoryError::transport_failure;
      return r;
    }
    r.hasTrack_ = true;
    r.error_ = HistoryError::none;
    if (status == TransportStatus::energy_cut) {
      finish_energy_cut(model, r, rng);
      return r;
    }
    if (status == TransportStatus::boundary) {
      // Caller first updates geometry/medium or records an observation. A due
      // collision can then be processed at the same point on the next call.
      r.action_ = r.boundary_ == BoundaryKind::escape ? HistoryAction::escaped
                                                      : HistoryAction::boundary;
      return r;
    }
    if (status == TransportStatus::collision_due) {
      finish_collision(model, region, r, rng);
      return r;
    }
    r.action_ = HistoryAction::active;
    return r;
  }
} // namespace corsika::kokkos::egs4
