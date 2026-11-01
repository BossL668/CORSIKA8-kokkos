/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/OverheadOptions.inl>
#include <corsika/detail/kokkos/egs4/Egs4C8Adapter.inl>
#include <corsika/detail/kokkos/egs4/Egs4Geometry.inl>
#include <corsika/detail/kokkos/egs4/Egs4Observation.inl>

namespace corsika::kokkos::egs4::c8_adapter {
  // An explicit choice, not a silent reinterpretation of C8's homogeneous fifth
  // layer. EGSIN1 uses four exponential transport regions up to HLAY(6). HOWFAR's
  // overburden still uses the original five-layer atmosphere.
  enum class AirConvention { c7_egs4_four_exponentials };
  struct AirEnvironment {
    unsigned overheadMask_{};
    double fullColumn_[5]{};
    Frame observerFrame_;
    CurvedConfig curved_;
    GeometryConfig geometry_;
    AirRegion regions_[4]; // C7 indices 2..5, highest first
    ::corsika::kokkos::egs4::AtmosphereData source_;
    MagneticField field_;
    std::int32_t mediumId_{};
  };
  KOKKOS_INLINE_FUNCTION double column_above(AirEnvironment const& e, double height_cm) {
    double radius = e.curved_.earthRadiusCm_ * .01 + height_cm * .01, answer = 0.;
    for (unsigned i = 0; i < e.source_.numberOfLayers_; ++i) {
      auto l = e.source_.atmosphereLayers_[i];
      double lo = larger(radius, l.innerRadiusM_), hi = l.outerRadiusM_;
      if (lo >= hi) continue;
      if ((e.overheadMask_ & overhead::column_cache) && radius <= l.innerRadiusM_) {
        answer += e.fullColumn_[i];
        continue;
      }
      if (l.densityModel_ == ::corsika::kokkos::egs4::DensityModel::Homogeneous)
        answer += 100. * l.densityParameterA_ * (hi - lo);
      else
        answer += 100. * l.densityParameterA_ *
                  ::exp((lo - l.densityParameterB_) / l.densityParameterC_) *
                  l.densityParameterC_ * ::expm1((hi - lo) / l.densityParameterC_);
    }
    return answer;
  }
  inline AirEnvironment make_air_environment(
      ::corsika::kokkos::egs4::AtmosphereData const& snapshot, double earth_radius_m,
      double sterncor, AirConvention convention) {
    using namespace ::corsika::kokkos;
    if (convention != AirConvention::c7_egs4_four_exponentials ||
        !finite(earth_radius_m) || earth_radius_m <= 0. || !finite(sterncor) ||
        snapshot.geometry_ != EnvironmentGeometry::SphericalLayers ||
        snapshot.numberOfLayers_ != 5)
      throw std::invalid_argument(
          "Native EGS4 requires an explicit four-exponential C7 air convention and a "
          "five-layer C8 snapshot");
    AirEnvironment e;
    e.source_ = snapshot;
    e.mediumId_ = snapshot.atmosphereLayers_[0].mediumId_;
    for (unsigned i = 0; i < 5; ++i) {
      auto l = snapshot.atmosphereLayers_[i];
      if (!finite(l.innerRadiusM_) || !finite(l.outerRadiusM_) ||
          l.outerRadiusM_ <= l.innerRadiusM_ || !finite(l.densityParameterA_) ||
          l.densityParameterA_ <= 0. || l.mediumId_ != e.mediumId_ ||
          (i && ::fabs(l.innerRadiusM_ -
                       snapshot.atmosphereLayers_[i - 1].outerRadiusM_) > 1.e-7))
        throw std::invalid_argument(
            "Non-contiguous or multi-material native EGS4 atmosphere");
      if (i < 4) {
        if (l.densityModel_ != DensityModel::Exponential ||
            !finite(l.densityParameterB_) || !finite(l.densityParameterC_) ||
            l.densityParameterC_ >= 0.)
          throw std::invalid_argument("EGS4 air regions require exponential density");
        double rho =
            l.densityParameterA_ *
            std::exp((earth_radius_m - l.densityParameterB_) / l.densityParameterC_);
        e.regions_[3 - i] = {rho, -.01 / l.densityParameterC_, sterncor};
        if (!valid_region(e.regions_[3 - i]))
          throw std::invalid_argument("Invalid EGS4 reference density");
      } else if (l.densityModel_ != DensityModel::Homogeneous)
        throw std::invalid_argument(
            "C8 fifth atmosphere layer must be homogeneous for this C7 convention");
    }
    Direction up{snapshot.observationPlaneNormal_[0], snapshot.observationPlaneNormal_[1],
                 snapshot.observationPlaneNormal_[2]};
    Direction center{snapshot.earthCenterM_[0], snapshot.earthCenterM_[1],
                     snapshot.earthCenterM_[2]};
    Direction plane{snapshot.observationPlanePointM_[0],
                    snapshot.observationPlanePointM_[1],
                    snapshot.observationPlanePointM_[2]};
    Direction b{snapshot.magneticFieldT_[0], snapshot.magneticFieldT_[1],
                snapshot.magneticFieldT_[2]};
    if (::fabs(dot(up, up) - 1.) > 1.e-12 || !finite(dot(up, center)) ||
        !finite(dot(up, plane)) || !finite(dot(b, b)))
      throw std::invalid_argument("Invalid native EGS4 observation plane or field");
    auto& f = e.observerFrame_;
    f.z_ = {-up.x_, -up.y_, -up.z_};
    double vertical = dot(b, f.z_);
    Direction horizontal{b.x_ - vertical * f.z_.x_, b.y_ - vertical * f.z_.y_,
                         b.z_ - vertical * f.z_.z_};
    double norm = std::sqrt(dot(horizontal, horizontal)),
           magnitude = std::sqrt(dot(b, b));
    if (norm > 1.e-20)
      f.x_ = {horizontal.x_ / norm, horizontal.y_ / norm, horizontal.z_ / norm};
    else {
      Direction ref = ::fabs(up.x_) < .8 ? Direction{1., 0., 0.} : Direction{0., 1., 0.};
      double projection = dot(ref, f.z_);
      ref = {ref.x_ - projection * f.z_.x_, ref.y_ - projection * f.z_.y_,
             ref.z_ - projection * f.z_.z_};
      double n = std::sqrt(dot(ref, ref));
      f.x_ = {ref.x_ / n, ref.y_ / n, ref.z_ / n};
    }
    f.y_ = cross(f.z_, f.x_);
    f.originM_ = {center.x_ + earth_radius_m * up.x_, center.y_ + earth_radius_m * up.y_,
                  center.z_ + earth_radius_m * up.z_};
    if (!valid_frame(f))
      throw std::invalid_argument("Native EGS4 observer frame is not right handed");
    double height =
        dot({plane.x_ - center.x_, plane.y_ - center.y_, plane.z_ - center.z_}, up) -
        earth_radius_m;
    e.curved_ = {true, earth_radius_m * 100., height * 100., -1.};
    e.geometry_ = {true, true, earth_radius_m * 100., {}, height * 100., 6.e5, 2.e6, 0.};
    e.geometry_.boundCm_[0] =
        (snapshot.atmosphereLayers_[4].outerRadiusM_ - earth_radius_m) * 100.;
    for (int i = 0; i < 4; ++i)
      e.geometry_.boundCm_[i + 1] =
          (snapshot.atmosphereLayers_[3 - i].innerRadiusM_ - earth_radius_m) * 100.;
    e.geometry_.boundCm_[5] = e.geometry_.boundCm_[4] - 1.;
    if (height * 100. < e.geometry_.boundCm_[4] ||
        height * 100. >= e.geometry_.boundCm_[0])
      throw std::invalid_argument("Observation plane outside native EGS4 atmosphere");
    e.overheadMask_ = overhead::options();
    e.curved_.directAzimuth_ = (e.overheadMask_ & overhead::curved_azimuth) != 0;
    for (unsigned i = 0; i < e.source_.numberOfLayers_; ++i) {
      auto l = e.source_.atmosphereLayers_[i];
      double lo = l.innerRadiusM_, hi = l.outerRadiusM_;
      e.fullColumn_[i] =
          l.densityModel_ == ::corsika::kokkos::egs4::DensityModel::Homogeneous
              ? 100. * l.densityParameterA_ * (hi - lo)
              : 100. * l.densityParameterA_ *
                    std::exp((lo - l.densityParameterB_) / l.densityParameterC_) *
                    l.densityParameterC_ * std::expm1((hi - lo) / l.densityParameterC_);
    }
    double column = column_above(e, 0.);
    if (!finite(column) || column <= 0.)
      throw std::invalid_argument("Invalid atmosphere column");
    e.geometry_.horizontalSlopeCmPerGCm2_ = (6.e5 - 2.e6) / column;
    e.field_ = {magnitude * 2.99792458, magnitude > 0. ? vertical / magnitude : 0.,
                magnitude > 0. ? norm / magnitude : 1.,
                snapshot.maximumMagneticDeflectionRad_};
    if (!valid_field(e.field_))
      throw std::invalid_argument("Invalid C7 magnetic step constraint");
    return e;
  }
  KOKKOS_INLINE_FUNCTION int air_region(AirEnvironment const& e, double local_z_cm) {
    double height = -local_z_cm;
    if (height > e.geometry_.boundCm_[0]) return 1;
    for (int i = 1; i < 5; ++i)
      if (height >= e.geometry_.boundCm_[i]) return i + 1;
    return 6;
  }
  // Minimal proper rotation from the observer downward axis to the local radial
  // downward axis. This is an adapter for C8's Cartesian vectors, not a claim that
  // C7's approximate, azimuth-preserving curved transport is an exact rotation.
  KOKKOS_INLINE_FUNCTION Frame radial_frame(Point observer, double earth_radius_cm) {
    double z = earth_radius_cm - observer.z_;
    double radius = ::sqrt(observer.x_ * observer.x_ + observer.y_ * observer.y_ + z * z);
    Direction n{-observer.x_ / radius, -observer.y_ / radius, z / radius};
    double d = 1. + n.z_;
    Frame f;
    f.x_ = {1. - n.x_ * n.x_ / d, -n.x_ * n.y_ / d, -n.x_};
    f.y_ = {-n.x_ * n.y_ / d, 1. - n.y_ * n.y_ / d, -n.y_};
    f.z_ = n;
    return f;
  }
  struct AirRecord {
    particle_state_type metadata_;
    CurvedTrack track_;
    CounterStream random_;
    CounterStream hadronRandom_;
    int region_{};
    bool exportedPosition_{}; // false on import: do not assume Cartesian round-trip is
                              // exact
  };
  KOKKOS_INLINE_FUNCTION bool import_air_record(particle_state_type p,
                                                AirEnvironment const& e,
                                                std::uint64_t seed, std::uint64_t shower,
                                                AirRecord& result) {
    TrackState observer;
    if (p.mediumId_ != e.mediumId_ || !import_particle(p, e.observerFrame_, observer))
      return false;
    Point pos = observer.positionCm_;
    double z = e.curved_.earthRadiusCm_ - pos.z_;
    double lateral = ::sqrt(pos.x_ * pos.x_ + pos.y_ * pos.y_),
           radius = ::sqrt(z * z + lateral * lateral);
    if (z <= 0. || !finite(radius) || radius <= 0.) return false;
    auto tangent = radial_frame(pos, e.curved_.earthRadiusCm_);
    if (!valid_frame(tangent)) return false;
    double arc =
        lateral > 0. ? e.curved_.earthRadiusCm_ * ::atan2(lateral, z) / lateral : 1.;
    AirRecord r;
    r.metadata_ = p;
    r.track_.particle_ = observer;
    r.track_.observerCm_ = pos;
    r.track_.particle_.positionCm_ = {pos.x_ * arc, pos.y_ * arc,
                                      e.curved_.earthRadiusCm_ - radius};
    r.track_.particle_.direction_ = to_local(tangent, observer.direction_);
    r.track_.wa_ = z / radius;
    double dh = -pos.z_ - e.curved_.observationHeightCm_,
           d = ::sqrt(lateral * lateral + dh * dh);
    r.track_.wap_ = d > 0. ? dh / d : 0.;
    r.region_ = air_region(e, r.track_.particle_.positionCm_.z_);
    r.random_.key_ = {seed, shower, p.historyId_, p.stepId_, NativeEgs4RandomDomain, 0};
    r.hadronRandom_.key_ = {
        seed, shower, p.historyId_, p.stepId_, NativeEgs4HadronRandomDomain, 0};
    if (!valid_track(r.track_.particle_)) return false;
    result = r;
    return true;
  }
  KOKKOS_INLINE_FUNCTION bool export_air_particle(CurvedTrack const& t,
                                                  AirEnvironment const& e,
                                                  particle_state_type& p) {
    auto observer = t.particle_;
    observer.positionCm_ = t.observerCm_;
    if (!t.detectorLocal_) {
      auto tangent = radial_frame(t.observerCm_, e.curved_.earthRadiusCm_);
      if (!valid_frame(tangent)) return false;
      observer.direction_ = to_global(tangent, t.particle_.direction_);
    }
    return export_particle(observer, e.observerFrame_, p);
  }
  enum class AirAction {
    error,
    active,
    replaced,
    observed,
    discarded,
    needs_host_channel
  };
  enum class AirDiscard { none, outside_atmosphere, angular, early_observation };
  struct AirOutcome {
    AirAction action_{AirAction::error};
    AirDiscard discard_{AirDiscard::none};
    HistoryOutcome history_;
    CurvedTrack children_[2];
    double discardedEnergyMeV_{};
  };
  struct GuardedFrame {
    CurvedFrameStep frame_;
    GeometryConfig geometry_;
    bool* outside_;
    KOKKOS_INLINE_FUNCTION FrameStepResult operator()(TrackState previous, Point next,
                                                      Direction direction, double step,
                                                      double path, double mass,
                                                      double speed) const {
      // Literal C7 UPWARD guard precedes displacement/normal time/radio output.
      if (-next.z_ >= geometry_.boundCm_[0] - 1. || -next.z_ <= geometry_.boundCm_[5]) {
        *outside_ = true;
        return {};
      }
      return frame_(previous, next, direction, step, path, mass, speed);
    }
  };
  KOKKOS_INLINE_FUNCTION void discard_air(AirOutcome& out, TrackState p, double mass,
                                          AirDiscard why) {
    out.action_ = AirAction::discarded;
    out.discard_ = why;
    out.discardedEnergyMeV_ = termination_energy(p, mass);
    out.history_.action_ = HistoryAction::replaced;
    out.history_.error_ = HistoryError::none;
    out.history_.particle_ = p;
    out.history_.childCount_ = 0;
  }
  // Actual C8-carrier history step in a configured spherical atmosphere. Kept
  // separate from the PROPOSAL router so no table flag or fallback is falsified.
  KOKKOS_INLINE_FUNCTION AirOutcome advance_air_record(ModelView model, AirRecord& record,
                                                       AirEnvironment const& e,
                                                       TransportSettings settings) {
    settings.pairedScatterTrig_ = (e.overheadMask_ & overhead::scatter_pair) != 0;
    PreparedCounterStream step_random(record.random_,
                                      (e.overheadMask_ & overhead::prepared_rng) != 0);
    AirOutcome out;
    auto previous = record.track_;
    out.history_.particle_ = previous.particle_;
    if (record.region_ < 1 || record.region_ > 6 ||
        record.metadata_.stepId_ == ~std::uint64_t{0})
      return out;
    if (record.region_ == 1 || record.region_ == 6) {
      discard_air(out, previous.particle_, model.thresholds_.massMeV_,
                  AirDiscard::outside_atmosphere);
      return out;
    }
    auto region = e.regions_[record.region_ - 2];
    auto plan = plan_history(model, previous.particle_, region, e.field_, settings,
                             step_random, (e.overheadMask_ & overhead::reuse_query) != 0);
    if (plan.action_ == PlanAction::error) {
      out.history_.error_ = plan.error_;
      return out;
    }
    GeometryResult geometry;
    CurvedHistoryOutcome result;
    int next_region = record.region_;
    if (plan.action_ != PlanAction::transport) {
      result = finish_curved_history(model, plan, previous, region, e.field_, settings,
                                     {}, e.curved_, step_random);
    } else {
      geometry = how_far(
          e.geometry_,
          {previous.particle_.positionCm_, previous.particle_.direction_, record.region_,
           previous.wa_, column_above(e, -previous.particle_.positionCm_.z_),
           plan.proposedCm_, 0.});
      if (!geometry.valid_ || geometry.stepCm_ <= 0.) return out;
      if (geometry.discard_ > 0) {
        discard_air(out, previous.particle_, model.thresholds_.massMeV_,
                    AirDiscard::outside_atmosphere);
        return out;
      }
      if (geometry.discard_ == -1) {
        auto detector = finish_detector_history(
            model, plan, previous, region, e.field_, settings, e.curved_,
            e.geometry_.maximumHorizontalCm_, geometry.stepCm_, step_random);
        if (detector.action_ == DetectorAction::error) return out;
        result = detector.advanced_;
        if (detector.action_ == DetectorAction::angular_discard) {
          out.action_ = AirAction::discarded;
          out.discard_ = AirDiscard::angular;
          out.discardedEnergyMeV_ = detector.discardedEnergyMeV_;
        } else if (detector.action_ == DetectorAction::observed)
          out.action_ = AirAction::observed;
      } else {
        bool outside = false;
        result.particle_ = previous;
        GuardedFrame frame{{e.curved_, previous, &result.frame_}, e.geometry_, &outside};
        plan.proposedCm_ = geometry.stepCm_;
        if (plan.particle_.pdg_ == 22)
          plan.photon_.proposedCm_ = geometry.stepCm_;
        else
          plan.electron_.proposedCm_ = geometry.stepCm_;
        // Defer the vertex until after final layer and angular checks, as in C7.
        // A synthetic boundary disposition changes no path arithmetic or RNG.
        result.history_ = finish_history(model, plan, region, e.field_, settings,
                                         {geometry.stepCm_, BoundaryKind::density_layer},
                                         step_random, frame);
        if (outside || result.frame_.status_ == CurvedStatus::angle_cut ||
            result.frame_.status_ == CurvedStatus::observation) {
          discard_air(out, previous.particle_, model.thresholds_.massMeV_,
                      outside ? AirDiscard::outside_atmosphere
                              : result.frame_.status_ == CurvedStatus::angle_cut
                                    ? AirDiscard::angular
                                    : AirDiscard::early_observation);
          return out; // no normally timed segment was emitted
        }
        if (result.history_.action_ == HistoryAction::error) {
          out.history_ = result.history_;
          return out;
        }
        result.frameApplied_ = result.history_.hasTrack_;
        result.particle_.particle_ = result.history_.particle_;
        result.particle_.observerCm_ = result.frame_.observer_;
        result.particle_.wa_ = result.frame_.wa_;
        result.particle_.wap_ = result.frame_.wap_;
        if (result.history_.action_ != HistoryAction::replaced) {
          next_region = geometry.region_;
          double height = -result.history_.particle_.positionCm_.z_;
          if (next_region >= 2 && next_region <= 5 && geometry.discard_ >= 0) {
            if (height < e.geometry_.boundCm_[next_region - 1])
              ++next_region;
            else if (height > e.geometry_.boundCm_[next_region - 2] ||
                     (height == e.geometry_.boundCm_[next_region - 2] &&
                      result.history_.particle_.direction_.z_ <= .003))
              --next_region;
          }
          if (result.history_.particle_.direction_.z_ < e.curved_.minimumCosine_) {
            out.action_ = AirAction::discarded;
            out.discard_ = AirDiscard::angular;
            out.discardedEnergyMeV_ =
                termination_energy(result.history_.particle_, model.thresholds_.massMeV_);
          } else if (next_region <= 1 || next_region >= 6 || geometry.discard_ == -2) {
            out.action_ = AirAction::discarded;
            out.discard_ = AirDiscard::outside_atmosphere;
            out.discardedEnergyMeV_ =
                termination_energy(result.history_.particle_, model.thresholds_.massMeV_);
          } else {
            auto& h = result.history_;
            h.boundary_ = next_region != record.region_ ? BoundaryKind::density_layer
                                                        : BoundaryKind::none;
            bool due = h.particle_.pdg_ == 22
                           ? h.particle_.photonClock_.remainingMfp_ <= 1.e-10
                           : h.particle_.clock_.remainingMfp_ < 1.e-10;
            if (due)
              finish_collision(model, e.regions_[next_region - 2], h, step_random);
            else
              h.action_ = HistoryAction::active;
            result.particle_.particle_ = h.particle_;
          }
        }
      }
    }
    out.history_ = result.history_;
    if (out.history_.action_ == HistoryAction::error) return out;
    if (out.action_ == AirAction::error) {
      auto a = out.history_.action_;
      out.action_ = a == HistoryAction::replaced ? AirAction::replaced
                                                 : a == HistoryAction::needs_host_channel
                                                       ? AirAction::needs_host_channel
                                                       : (a == HistoryAction::active ||
                                                          a == HistoryAction::boundary)
                                                             ? AirAction::active
                                                             : AirAction::error;
    }
    if (out.action_ == AirAction::error) return out;
    for (int j = 0; j < out.history_.childCount_; ++j) {
      out.children_[j] = result.particle_;
      out.children_[j].particle_ = out.history_.children_[j];
    }
    auto metadata = record.metadata_;
    if (!export_air_particle(result.particle_, e, metadata)) {
      out.action_ = AirAction::error;
      return out;
    }
    if (out.history_.hasTrack_) ++metadata.stepId_;
    record.metadata_ = metadata;
    record.exportedPosition_ = true;
    record.track_ = result.particle_;
    record.region_ = next_region;
    return out;
  }
} // namespace corsika::kokkos::egs4::c8_adapter
