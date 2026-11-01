/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4CurvedTransport.inl>

namespace corsika::kokkos::egs4 {
  enum class ObservationPreparationStatus { invalid, transport, angular_discard };
  struct ObservationPreparation {
    ObservationPreparationStatus status_{ObservationPreparationStatus::invalid};
    CurvedTrack particle_;
    double stepCm_{};
    bool transportReady_{};
  };
  // C7 ELECTR/PHOTON CURVED pre-transport IDISC=-1 block. This is not a
  // post-step clipping operation. In particular ALTEXP and the sampled loss
  // coefficients remain those of the ORIGINAL proposal. Save the OLD observer
  // endpoint separately for the subsequent observer-speed correction.
  KOKKOS_INLINE_FUNCTION ObservationPreparation
  prepare_observation_step(CurvedTrack previous, CurvedConfig c,
                           double maximum_horizontal_cm, double original_step_cm = 0.) {
    ObservationPreparation r;
    r.particle_ = previous;
    r.stepCm_ = original_step_cm;
    if (!valid_track(previous.particle_) || !finite(c.earthRadiusCm_) ||
        c.earthRadiusCm_ <= 0. || !finite(c.observationHeightCm_) ||
        !finite(c.minimumCosine_) || c.minimumCosine_ < -1. || c.minimumCosine_ > 1. ||
        !finite(maximum_horizontal_cm) || maximum_horizontal_cm <= 0. ||
        !finite(original_step_cm) || original_step_cm < 0.)
      return r;
    auto& p = r.particle_.particle_;
    double lateral =
        ::sqrt(p.positionCm_.x_ * p.positionCm_.x_ + p.positionCm_.y_ * p.positionCm_.y_);
    r.particle_.wa_ = smaller(1., ::cos(lateral / c.earthRadiusCm_));
    double phi = (p.direction_.x_ != 0. || p.direction_.y_ != 0.)
                     ? ::atan2(p.direction_.y_, p.direction_.x_)
                     : 0.;
    double effective = -(::cos(phi) * p.positionCm_.x_ + ::sin(phi) * p.positionCm_.y_);
    if (effective < maximum_horizontal_cm) {
      double sd = ::sin(effective / c.earthRadiusCm_), cd = ::sqrt((1. - sd) * (1. + sd));
      double w = p.direction_.z_ * cd;
      if (::fabs(p.direction_.z_) < 1.)
        w -= ::sqrt((1. - p.direction_.z_) * (1. + p.direction_.z_)) * sd;
      p.direction_.z_ = larger(-1., smaller(1., w));
    }
    if (p.direction_.z_ < c.minimumCosine_) {
      r.status_ = ObservationPreparationStatus::angular_discard;
      return r;
    }
    double apparent_z =
        c.earthRadiusCm_ - (c.earthRadiusCm_ - p.positionCm_.z_) * r.particle_.wa_;
    // Expose C7 ZAP even for an early exit; observer x/y must not be advanced.
    r.particle_.observerCm_.z_ = apparent_z;
    if (c.flatOutput_) {
      if (r.particle_.wa_ != 1.) {
        if (r.particle_.wa_ <= 0.) return r;
        double phi1 = (p.positionCm_.x_ != 0. || p.positionCm_.y_ != 0.)
                          ? ::atan2(p.positionCm_.y_, p.positionCm_.x_)
                          : 0.;
        double st = ::sqrt((1. - r.particle_.wa_) * (1. + r.particle_.wa_));
        double radius = (-apparent_z + c.earthRadiusCm_) * st / r.particle_.wa_;
        p.positionCm_ = {radius * ::cos(phi1), radius * ::sin(phi1), apparent_z};
      }
      r.particle_.wap_ = 0.;
    }
    if (p.direction_.x_ != 0.) {
      double t = p.direction_.y_ / p.direction_.x_;
      p.direction_.x_ =
          ::fabs(p.direction_.z_) < 1.
              ? ::copysign(1., p.direction_.x_) *
                    ::sqrt((1. - p.direction_.z_) * (1. + p.direction_.z_) / (1. + t * t))
              : 0.;
      p.direction_.y_ = t * p.direction_.x_;
    } else if (p.direction_.y_ != 0. && ::fabs(p.direction_.z_) < 1.) {
      p.direction_.y_ = ::copysign(1., p.direction_.y_) *
                        ::sqrt((1. - p.direction_.z_) * (1. + p.direction_.z_));
    } else
      p.direction_.y_ = 0.;
    if (p.direction_.z_ <= 0.) {
      r.status_ = ObservationPreparationStatus::angular_discard;
      return r;
    }
    r.stepCm_ =
        larger(-(p.positionCm_.z_ + c.observationHeightCm_) / p.direction_.z_, .0001);
    r.particle_.detectorLocal_ = true;
    if (!finite(r.stepCm_)) return r;
    // Literal C7 updates W but leaves U=V=0 for an exactly vertical input away
    // from the origin. Do not silently normalize or invent a transverse sign.
    // Expose its arithmetic for the oracle, but reject such an unsupported
    // direction at the transport-controller boundary.
    r.transportReady_ = valid_track(p);
    r.status_ = ObservationPreparationStatus::transport;
    return r;
  }

  enum class DetectorAction { error, observed, stopped, angular_discard };
  struct DetectorHistoryOutcome {
    DetectorAction action_{DetectorAction::error};
    ObservationPreparation preparation_;
    CurvedHistoryOutcome advanced_;
    Point observerStartCm_;
    double discardedEnergyMeV_{}; // C7 angular-loss ledger, NOT continuous deposition
  };
  KOKKOS_INLINE_FUNCTION double termination_energy(TrackState p, double mass_MeV) {
    return p.energyMeV_ + (p.pdg_ == 11 ? -mass_MeV : p.pdg_ == -11 ? mass_MeV : 0.);
  }
  // Finish an already planned normal-air step for which HOWFAR returned IDISC=-1.
  // Keep the original density/loss/scattering coefficients: replanning after the
  // detector-frame conversion changes C7's algorithm. Original C7 performs this
  // correction after its geometry cap, so do not cap it a second time by the old
  // proposal. This entry does not cover atmosphere escape or EM-primary FIXHEI.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION DetectorHistoryOutcome finish_detector_history(
      ModelView model, HistoryPlan plan, CurvedTrack previous, AirRegion air,
      MagneticField field, TransportSettings settings, CurvedConfig config,
      double maximum_horizontal_cm, double howfar_step_cm, TRandom& rng) {
    DetectorHistoryOutcome r;
    r.observerStartCm_ = previous.observerCm_;
    r.advanced_.particle_ = previous;
    if (plan.action_ != PlanAction::transport || !valid_settings(model, settings) ||
        !valid_region(air) || !valid_field(field) || !valid_point(previous.observerCm_))
      return r;
    previous.particle_ = plan.particle_; // include the clock sampled by the proposal
    r.preparation_ =
        prepare_observation_step(previous, config, maximum_horizontal_cm, howfar_step_cm);
    if (r.preparation_.status_ == ObservationPreparationStatus::angular_discard) {
      r.action_ = DetectorAction::angular_discard;
      r.advanced_.particle_ = r.preparation_.particle_;
      auto& h = r.advanced_.history_;
      h.particle_ = r.preparation_.particle_.particle_;
      h.action_ = HistoryAction::replaced;
      h.error_ = HistoryError::none;
      r.discardedEnergyMeV_ = termination_energy(h.particle_, model.thresholds_.massMeV_);
      return r; // label 420/1000: NO stopped-positron photons and NO track
    }
    if (r.preparation_.status_ != ObservationPreparationStatus::transport ||
        !r.preparation_.transportReady_)
      return r;
    auto start = r.preparation_.particle_;
    start.observerCm_ = r.observerStartCm_; // XXXOLD/YYYOLD/ZAPOLD, before preprocessing
    plan.particle_ = start.particle_;
    plan.proposedCm_ = r.preparation_.stepCm_;
    if (start.particle_.pdg_ == 22)
      plan.photon_.proposedCm_ = plan.proposedCm_;
    else
      plan.electron_.proposedCm_ = plan.proposedCm_;
    r.advanced_ =
        finish_curved_history(model, plan, start, air, field, settings,
                              {plan.proposedCm_, BoundaryKind::observation}, config, rng);
    auto& h = r.advanced_.history_;
    if (h.action_ == HistoryAction::error) return r;
    // ELECTR checks the energy cutoff before the post-MSCAT angular cutoff and
    // the observation writer. Its stopped-positron photons therefore survive.
    if (h.action_ == HistoryAction::replaced) {
      r.action_ = DetectorAction::stopped;
      return r;
    }
    if (h.action_ != HistoryAction::boundary || h.boundary_ != BoundaryKind::observation)
      return r;
    if (h.particle_.direction_.z_ < config.minimumCosine_) {
      r.action_ = DetectorAction::angular_discard;
      r.discardedEnergyMeV_ = termination_energy(h.particle_, model.thresholds_.massMeV_);
      h.action_ = HistoryAction::replaced;
      return r; // keep the normally emitted track
    }
    r.action_ = DetectorAction::observed;
    return r;
  }
} // namespace corsika::kokkos::egs4
