/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4Model.inl>

namespace corsika::kokkos::egs4 {
  // Native C7 EGS coordinates and units: x north, y east, z down, cm/MeV/s.
  // The C8 adapter must transform the frame, rather than relabel its axes.
  struct Point {
    double x_{}, y_{}, z_{};
  };
  struct TrackState {
    int pdg_{};
    double energyMeV_{}, timeS_{};
    Point positionCm_;
    Direction direction_;
    ElectronClock clock_;
    PhotonClock photonClock_;
    Channel pendingChannel_{
        Channel::invalid}; // host request must be resolved before further transport
  };
  struct AirRegion {
    double referenceDensityGCm3_{}; // RHOR, NOT density at the current height
    double inverseScaleHeightCm_{}; // HBAROI; rho(z)=RHOR*exp(z*HBAROI)
    double sterncor_{};
  };
  struct MagneticField {
    double bnormMeVCm_{}, sinB_{}, cosB_{1.}, maximumAngleRad_{.2};
  };
  struct TransportSettings {
    double electronCutTotalMeV_{}, photonCutMeV_{};
    double stepfc_{1.}, vacuumDistanceCm_{1.e9};
    double speedCmS_{2.99792458e10};
    bool pairedScatterTrig_{};
  };
  // HATCH's threshold floor is part of the backend, not an optional C8 cut.
  inline TransportSettings transport_settings(ModelView model,
                                              double electron_kinetic_cut,
                                              double photon_cut, double stepfc = 1.) {
    if (!finite(electron_kinetic_cut) || electron_kinetic_cut < 0. ||
        !finite(photon_cut) || photon_cut < 0. || !finite(stepfc) || stepfc <= 0.)
      throw std::runtime_error("Invalid EGS4 transport cut or STEPFC");
    TransportSettings s;
    s.electronCutTotalMeV_ =
        larger(larger(electron_kinetic_cut + model.thresholds_.massMeV_,
                      model.tables_.medium_.ae_),
               model.tables_.medium_.ap_ + model.thresholds_.massMeV_);
    s.photonCutMeV_ = larger(photon_cut, model.tables_.medium_.ap_);
    s.stepfc_ = stepfc;
    return s;
  }
  KOKKOS_INLINE_FUNCTION bool valid_track(TrackState s) {
    return (s.pdg_ == 11 || s.pdg_ == -11 || s.pdg_ == 22) && finite(s.energyMeV_) &&
           s.energyMeV_ > 0. && finite(s.timeS_) && finite(s.positionCm_.x_) &&
           finite(s.positionCm_.y_) && finite(s.positionCm_.z_) &&
           valid_direction(s.direction_);
  }
  KOKKOS_INLINE_FUNCTION bool valid_region(AirRegion r) {
    return finite(r.referenceDensityGCm3_) && r.referenceDensityGCm3_ > 0. &&
           finite(r.inverseScaleHeightCm_) && r.inverseScaleHeightCm_ >= 0. &&
           finite(r.sterncor_);
  }
  KOKKOS_INLINE_FUNCTION bool valid_field(MagneticField b) {
    return finite(b.bnormMeVCm_) && b.bnormMeVCm_ >= 0. && finite(b.sinB_) &&
           finite(b.cosB_) &&
           ::fabs(b.sinB_ * b.sinB_ + b.cosB_ * b.cosB_ - 1.) < 1.e-12 &&
           finite(b.maximumAngleRad_) && b.maximumAngleRad_ > 0. &&
           b.maximumAngleRad_ <= .2;
  }
  struct ElectronPlan {
    Status status_{Status::invalid_input};
    LocalStep local_;
    double scatteringLengthCm_{}, densityInverse_{}, truePhysicalCm_{};
    double unclippedCm_{}, proposedCm_{};
  };
  // Two-phase interface: propose, ask geometry for a cap, then finish. No RNG
  // is consumed by the proposal. A history's initial SIG0 stays in its clock.
  KOKKOS_INLINE_FUNCTION ElectronPlan prepare_electron_segment(
      ModelView model, TrackState state, AirRegion region, MagneticField field,
      TransportSettings settings, ElectronQuery const* supplied = nullptr) {
    ElectronPlan p;
    double m = model.thresholds_.massMeV_;
    if (!valid_track(state) || state.pdg_ == 22 || !valid_region(region) ||
        !valid_field(field) || !state.clock_.initialized_ ||
        settings.electronCutTotalMeV_ < model.tables_.medium_.ae_ ||
        settings.electronCutTotalMeV_ < model.tables_.medium_.ap_ + m)
      return p;
    auto q = supplied ? *supplied
                      : electron_query(model.tables_, state.energyMeV_, m,
                                       state.pdg_ == 11 ? -1 : 1);
    q = with_sampled_rate(q, state.clock_);
    p.local_ = propose_local_step(
        model.tables_, q, state.energyMeV_, m, settings.electronCutTotalMeV_,
        region.referenceDensityGCm3_, state.positionCm_.z_, region.sterncor_,
        settings.stepfc_, state.clock_.remainingMfp_, settings.vacuumDistanceCm_);
    if (p.local_.status_ != Status::success) {
      p.status_ = p.local_.status_;
      return p;
    }
    double beta2 = larger(1.e-8, 1. - m * m / (state.energyMeV_ * state.energyMeV_));
    double beta3 = state.energyMeV_ * beta2 * .094315;
    double rhofi = 1. / (region.referenceDensityGCm3_ / model.tables_.medium_.rho_);
    p.scatteringLengthCm_ =
        model.tables_.medium_.radiationLengthCm_ * beta3 * beta3 * rhofi;
    double ratio = p.local_.trueStepCm_ / p.scatteringLengthCm_;
    p.densityInverse_ = ::exp(-state.positionCm_.z_ * region.inverseScaleHeightCm_);
    double distance = p.local_.projectedStepCm_ * p.densityInverse_;
    double disc = state.direction_.z_ * distance * region.inverseScaleHeightCm_;
    if (::fabs(disc) < .0000007)
      distance *=
          1. - .5 * disc *
                   (1. - .666666666666667 * disc * (1. - .75 * disc * (1. - .8 * disc)));
    else if (disc > -1.)
      distance = distance * ::log(disc + 1.) / disc;
    else
      distance = settings.vacuumDistanceCm_;
    p.unclippedCm_ = distance;
    p.truePhysicalCm_ = distance / (1. - ratio);
    p.proposedCm_ = field.bnormMeVCm_ > 0.
                        ? smaller(distance, (field.maximumAngleRad_ / field.bnormMeVCm_) *
                                                state.energyMeV_)
                        : distance;
    if (!finite(p.densityInverse_) || p.densityInverse_ <= 0. || !finite(p.proposedCm_) ||
        p.proposedCm_ <= 0. || !finite(p.truePhysicalCm_) ||
        !finite(p.scatteringLengthCm_))
      return p;
    p.status_ = Status::success;
    return p;
  }
  KOKKOS_INLINE_FUNCTION double path_correction(double vstp) {
    int i = int(2. + 18.182 * vstp);
    if (i < 1) i = 1;
    if (i > 6) i = 6;
    double c0, c1, c2;
    if (i <= 2) {
      c0 = 1.;
      c1 = .98875;
      c2 = 2.5026;
    } else if (i == 3) {
      c0 = 1.006;
      c1 = .78657;
      c2 = 4.2387;
    } else if (i == 4) {
      c0 = 1.0657;
      c1 = -.25051;
      c2 = 8.7681;
    } else {
      c0 = 1.6971;
      c1 = -7.56;
      c2 = 29.946;
    }
    return c0 + vstp * (c1 + vstp * c2);
  }
  enum class BoundaryKind { none, density_layer, material_change, observation, escape };
  struct BoundaryLimit {
    double distanceCm_{};
    BoundaryKind kind_{BoundaryKind::none};
  };
  enum class TransportStatus {
    invalid,
    advanced,
    collision_due,
    energy_cut,
    boundary,
    random_failure,
    sampling_failure
  };
  struct ElectronSegment {
    TransportStatus status_{TransportStatus::invalid};
    TrackState particle_;
    BoundaryKind boundary_{BoundaryKind::none};
    double geometricStepCm_{}, lossPathCm_{}, timePathCm_{}, depositMeV_{},
        scatteringAngle_{}, bendAngle_{};
    bool pathCorrected_{};
    Direction transportEndDirection_; // before MSCAT rotation; scatter is applied after
                                      // displacement
  };
  struct FrameStepResult {
    bool valid_{};
    Point positionCm_;
    Direction direction_;
    double timeIncrementS_{};
  };
  // The frame policy runs after magnetic displacement and BEFORE the MSCAT
  // azimuth/rotation. In C7 CURVED these operations do not commute. The default
  // policy preserves the previously tested flat arithmetic and random sequence.
  struct FlatFrameStep {
    KOKKOS_INLINE_FUNCTION FrameStepResult operator()(
        TrackState previous, Point advanced, Direction direction, double /*geometric_cm*/,
        double time_path_cm, double mass_MeV, double speed_cm_s) const {
      double time = time_path_cm * (1. / speed_cm_s);
      if (previous.pdg_ != 22)
        time /= ::sqrt((1. - mass_MeV / previous.energyMeV_) *
                       (1. + mass_MeV / previous.energyMeV_));
      return {finite(time) && time >= 0., advanced, direction, time};
    }
  };
  // A non-default policy must explicitly reject unimplemented endpoint cases;
  // they are NOT emitted as successfully timed normal radio tracks.
  // Errors leave the returned particle unchanged. RNG exhaustion is fatal;
  // callers must not substitute a PROPOSAL transport step.
  template <class TRandom, class TFrameStep = FlatFrameStep>
  KOKKOS_INLINE_FUNCTION ElectronSegment
  finish_electron_segment(ModelView model, TrackState state, AirRegion region,
                          MagneticField field, TransportSettings settings, ElectronPlan p,
                          BoundaryLimit limit, TRandom& rng, TFrameStep frame = {}) {
    ElectronSegment r;
    r.particle_ = state;
    if (p.status_ != Status::success || !valid_track(state) || state.pdg_ == 22 ||
        !valid_region(region) || !valid_field(field) || !finite(settings.speedCmS_) ||
        settings.speedCmS_ <= 0. || !finite(limit.distanceCm_) || limit.distanceCm_ <= 0.)
      return r;
    double step = smaller(p.proposedCm_, limit.distanceCm_), tvstep, tvstpc;
    if (limit.distanceCm_ <= p.proposedCm_) r.boundary_ = limit.kind_;
    if (step == p.unclippedCm_) {
      tvstep = p.local_.trueStepCm_;
      tvstpc = p.truePhysicalCm_;
    } else {
      r.pathCorrected_ = true;
      double equivalent = step,
             disc = state.direction_.z_ * equivalent * region.inverseScaleHeightCm_;
      if (disc != 0.)
        equivalent = equivalent * (::exp(disc) - 1.) / (disc * p.densityInverse_);
      else
        equivalent /= p.densityInverse_;
      equivalent = larger(equivalent, .0001);
      double factor = path_correction(equivalent / p.scatteringLengthCm_);
      tvstep = factor * equivalent;
      tvstpc = factor * step;
    }
    int charge = state.pdg_ == 11 ? -1 : 1;
    double alpha = step * charge * field.bnormMeVCm_ / state.energyMeV_;
    tvstpc = tvstpc * (1. + .04166667 * alpha * alpha);
    double deposit = p.local_.lossMeVPerCm_ * tvstep, m = model.thresholds_.massMeV_;
    double new_energy = state.energyMeV_ - deposit;
    if (!finite(tvstep) || tvstep < 0. || !finite(tvstpc) || tvstpc < 0. ||
        !finite(new_energy) || new_energy < m)
      return r;
    double rhofac = region.referenceDensityGCm3_ / model.tables_.medium_.rho_;
    auto scatter =
        sample_scattering(model.tables_.medium_, state.energyMeV_, m, tvstep * rhofac,
                          rng, 1024, settings.pairedScatterTrig_);
    if (scatter.status_ != InteractionStatus::success) {
      r.status_ = scatter.status_ == InteractionStatus::random_failure
                      ? TransportStatus::random_failure
                      : TransportStatus::sampling_failure;
      return r;
    }
    double u0 = state.direction_.x_, v0 = state.direction_.y_, w0 = state.direction_.z_;
    double f = 1. - .5 * alpha * alpha * (1. - .75 * alpha * alpha);
    double fs = (1. - f) * field.sinB_, fc = (1. - f) * field.cosB_, v1 = v0 * alpha * f,
           usw = u0 * field.sinB_ - w0 * field.cosB_;
    Direction bent{u0 - fs * usw + v1 * field.sinB_, f * (v0 - alpha * usw),
                   w0 + fc * usw - v1 * field.cosB_};
    double inv = 1.5 - .5 * (bent.x_ * bent.x_ + bent.y_ * bent.y_ + bent.z_ * bent.z_);
    bent.x_ *= inv;
    bent.y_ *= inv;
    bent.z_ *= inv;
    auto next = state;
    next.positionCm_.x_ += step * .5 * (u0 + bent.x_);
    next.positionCm_.y_ += step * .5 * (v0 + bent.y_);
    next.positionCm_.z_ += step * .5 * (w0 + bent.z_);
    auto transformed =
        frame(state, next.positionCm_, bent, step, tvstpc, m, settings.speedCmS_);
    if (!transformed.valid_) return r;
    next.positionCm_ = transformed.positionCm_;
    bent = transformed.direction_;
    next.timeS_ += transformed.timeIncrementS_;
    double sp, cp;
    // ELECTR always consumes this azimuth, including the NOSCAT branch.
    if (!azimuth(rng, sp, cp)) {
      r.status_ = TransportStatus::random_failure;
      return r;
    }
    next.direction_ = rotate(bent, scatter.sine_, scatter.cosine_, sp, cp);
    next.energyMeV_ = new_energy;
    if (!valid_track(next)) return r;
    if (new_energy <= settings.electronCutTotalMeV_)
      r.status_ = TransportStatus::energy_cut;
    else if (r.boundary_ == BoundaryKind::escape)
      r.status_ = TransportStatus::boundary;
    else if (r.boundary_ == BoundaryKind::material_change) {
      next.clock_ = {};
      r.status_ = TransportStatus::boundary;
    } else {
      auto cs = consume_electron_clock(next.clock_, tvstep, rhofac);
      if (cs == ClockStatus::invalid) return r;
      r.status_ = cs == ClockStatus::collision_due ? TransportStatus::collision_due
                                                   : TransportStatus::advanced;
      if (r.boundary_ != BoundaryKind::none) r.status_ = TransportStatus::boundary;
    }
    r.particle_ = next;
    r.geometricStepCm_ = step;
    r.lossPathCm_ = tvstep;
    r.timePathCm_ = tvstpc;
    r.depositMeV_ = deposit;
    r.scatteringAngle_ = scatter.angle_;
    r.bendAngle_ = alpha;
    r.transportEndDirection_ = bent;
    return r;
  }
} // namespace corsika::kokkos::egs4
