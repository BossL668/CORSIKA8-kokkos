/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4Transport.inl>

namespace corsika::kokkos::egs4 {
  struct PhotonPlan {
    Status status_{Status::invalid_input};
    PhotonQuery coefficients_;
    double meanFreePathCm_{}, densityInverse_{}, proposedCm_{};
  };
  // Native PHOTON normal-air transport, no Rayleigh. Flat local EGS frame;
  // geometry must cap this proposal before transport (not a HOWFAR replacement).
  KOKKOS_INLINE_FUNCTION PhotonPlan prepare_photon_segment(
      ModelView model, TrackState state, AirRegion region, TransportSettings settings,
      PhotonQuery const* supplied = nullptr) {
    PhotonPlan p;
    if (!valid_track(state) || state.pdg_ != 22 || !valid_region(region) ||
        !finite(settings.photonCutMeV_) ||
        settings.photonCutMeV_ < model.tables_.medium_.ap_ ||
        state.energyMeV_ <= settings.photonCutMeV_ || !state.photonClock_.initialized_ ||
        !finite(state.photonClock_.remainingMfp_) ||
        state.photonClock_.remainingMfp_ < 0. || !finite(settings.vacuumDistanceCm_) ||
        settings.vacuumDistanceCm_ <= 0.)
      return p;
    p.coefficients_ =
        supplied ? *supplied : photon_query(model.tables_, state.energyMeV_);
    if (p.coefficients_.status_ != Status::success) {
      p.status_ = p.coefficients_.status_;
      return p;
    }
    double rhofi = 1. / (region.referenceDensityGCm3_ / model.tables_.medium_.rho_);
    p.meanFreePathCm_ = p.coefficients_.meanFreePathCm_ * rhofi;
    if (!finite(p.meanFreePathCm_) || p.meanFreePathCm_ <= 0.) return p;
    p.densityInverse_ = ::exp(-state.positionCm_.z_ * region.inverseScaleHeightCm_);
    double step = larger(p.meanFreePathCm_ * state.photonClock_.remainingMfp_, 1.e-3) *
                  p.densityInverse_;
    double disc = state.direction_.z_ * step * region.inverseScaleHeightCm_;
    if (::fabs(disc) < .0000007)
      step *=
          1. - .5 * disc *
                   (1. - .666666666666667 * disc * (1. - .75 * disc * (1. - .8 * disc)));
    else if (disc > -1.)
      step = step * ::log(disc + 1.) / disc;
    else
      step = settings.vacuumDistanceCm_;
    p.proposedCm_ = step;
    if (!finite(p.densityInverse_) || p.densityInverse_ <= 0. || !finite(step) ||
        step <= 0.)
      return p;
    p.status_ = Status::success;
    return p;
  }
  struct PhotonSegment {
    TransportStatus status_{TransportStatus::invalid};
    TrackState particle_;
    BoundaryKind boundary_{BoundaryKind::none};
    double geometricStepCm_{}, opticalPathCm_{};
    Direction transportEndDirection_;
  };
  template <class TFrameStep = FlatFrameStep>
  KOKKOS_INLINE_FUNCTION PhotonSegment
  finish_photon_segment(TrackState state, AirRegion region, TransportSettings settings,
                        PhotonPlan p, BoundaryLimit limit, TFrameStep frame = {}) {
    PhotonSegment r;
    r.particle_ = state;
    if (p.status_ != Status::success || !valid_track(state) || state.pdg_ != 22 ||
        !valid_region(region) || !finite(settings.speedCmS_) ||
        settings.speedCmS_ <= 0. || !finite(limit.distanceCm_) ||
        limit.distanceCm_ <= 0. || !state.photonClock_.initialized_)
      return r;
    double step = smaller(p.proposedCm_, limit.distanceCm_), equivalent = step;
    if (limit.distanceCm_ <= p.proposedCm_) r.boundary_ = limit.kind_;
    double disc = state.direction_.z_ * equivalent * region.inverseScaleHeightCm_;
    if (disc != 0.)
      equivalent = equivalent * (::exp(disc) - 1.) / (disc * p.densityInverse_);
    else
      equivalent /= p.densityInverse_;
    equivalent = larger(equivalent, .0001);
    auto next = state;
    next.positionCm_.x_ += state.direction_.x_ * step;
    next.positionCm_.y_ += state.direction_.y_ * step;
    next.positionCm_.z_ += state.direction_.z_ * step;
    auto transformed = frame(state, next.positionCm_, state.direction_, step, step, 0.,
                             settings.speedCmS_);
    if (!transformed.valid_) return r;
    next.positionCm_ = transformed.positionCm_;
    next.direction_ = transformed.direction_;
    next.timeS_ += transformed.timeIncrementS_;
    // Unlike ELECTR, PHOTON retains its optical depth across a material change.
    // The next proposal obtains the new mean free path, without drawing anew.
    next.photonClock_.remainingMfp_ =
        larger(0., state.photonClock_.remainingMfp_ - equivalent / p.meanFreePathCm_);
    if (!valid_track(next) || !finite(equivalent) ||
        !finite(next.photonClock_.remainingMfp_))
      return r;
    r.particle_ = next;
    r.geometricStepCm_ = step;
    r.opticalPathCm_ = equivalent;
    r.transportEndDirection_ = next.direction_;
    r.status_ = next.photonClock_.remainingMfp_ <= 1.e-10 ? TransportStatus::collision_due
                                                          : TransportStatus::advanced;
    if (r.boundary_ != BoundaryKind::none) r.status_ = TransportStatus::boundary;
    return r;
  }
} // namespace corsika::kokkos::egs4
