/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4Transport.inl>

namespace corsika::kokkos::egs4 {
  // C7 CURVED normal-step rebasing, after the magnetic displacement and BEFORE
  // the MSCAT azimuth rotation. Coordinates are the original EGS surface-arc
  // coordinates (cm), not Cartesian C8 positions. Observer positions are kept
  // separately; using surface arcs as Cartesian radio endpoints is incorrect.
  enum class CurvedStatus { invalid, advanced, angle_cut, observation };
  struct CurvedInput {
    bool photon_{}, detectorLocal_{}, flatOutput_{true};
    Point previousLocal_, previousObserver_, advancedLocal_;
    Direction advancedDirection_;
    double earthRadiusCm_{}, observationHeightCm_{}, minimumCosine_{-1.};
    double geometricStepCm_{}, timePathCm_{}, totalEnergyMeV_{}, massMeV_{.51099895};
    double speedCmS_{2.99792458e10};
    bool directAzimuth_{};
  };
  struct CurvedResult {
    CurvedStatus status_{CurvedStatus::invalid};
    Point local_, observer_;
    Direction direction_;
    double wa_{1.}, wap_{}, timeIncrementS_{}, observerToLocalDistanceRatio_{};
  };
  KOKKOS_INLINE_FUNCTION bool valid_point(Point p) {
    return finite(p.x_) && finite(p.y_) && finite(p.z_);
  }
  KOKKOS_INLINE_FUNCTION CurvedResult rebase_curved_step(CurvedInput q) {
    CurvedResult r;
    r.local_ = q.advancedLocal_;
    r.direction_ = q.advancedDirection_;
    if (!valid_point(q.previousLocal_) || !valid_point(q.previousObserver_) ||
        !valid_point(q.advancedLocal_) || !valid_direction(q.advancedDirection_) ||
        !finite(q.earthRadiusCm_) || q.earthRadiusCm_ <= 0. ||
        q.advancedLocal_.z_ >= q.earthRadiusCm_ || !finite(q.observationHeightCm_) ||
        !finite(q.minimumCosine_) || q.minimumCosine_ < -1. || q.minimumCosine_ > 1. ||
        !finite(q.geometricStepCm_) || q.geometricStepCm_ <= 0. ||
        !finite(q.timePathCm_) || q.timePathCm_ < 0. || !finite(q.speedCmS_) ||
        q.speedCmS_ <= 0. || !finite(q.totalEnergyMeV_) || q.totalEnergyMeV_ <= 0. ||
        (!q.photon_ &&
         (!finite(q.massMeV_) || q.massMeV_ <= 0. || q.totalEnergyMeV_ <= q.massMeV_)))
      return r;
    if (!q.detectorLocal_ || !q.flatOutput_) {
      // ELECTR clamps W before the rebase; PHOTON does not.
      if (!q.photon_) r.direction_.z_ = larger(-1., smaller(1., r.direction_.z_));
      double dx = r.local_.x_ - q.previousLocal_.x_,
             dy = r.local_.y_ - q.previousLocal_.y_;
      double trans2 = larger(dx * dx + dy * dy, q.photon_ ? .00001 : .0001);
      double radial_z = q.earthRadiusCm_ - r.local_.z_;
      double aux = ::sqrt(trans2 + radial_z * radial_z);
      double znew = q.earthRadiusCm_ - aux;
      double sindif = ::sqrt(trans2) / aux, cosdif = smaller(1., radial_z / aux);
      double corr =
          q.earthRadiusCm_ * ::asin(sindif) / ((q.earthRadiusCm_ - znew) * sindif);
      r.local_.x_ = q.previousLocal_.x_ + dx * corr;
      r.local_.y_ = q.previousLocal_.y_ + dy * corr;
      r.local_.z_ = znew;
      double w = r.direction_.z_ * cosdif;
      if (::fabs(r.direction_.z_) < 1.)
        w -= sindif * ::sqrt((1. - r.direction_.z_) * (1. + r.direction_.z_));
      r.direction_.z_ = larger(-1., smaller(1., w));
      if (r.direction_.z_ < q.minimumCosine_) {
        r.status_ = CurvedStatus::angle_cut;
        return r;
      }
      double lateral = ::sqrt(r.local_.x_ * r.local_.x_ + r.local_.y_ * r.local_.y_);
      r.wa_ = smaller(1., ::cos(lateral / q.earthRadiusCm_));
      double radial = q.earthRadiusCm_ - znew;
      r.observer_.z_ = q.earthRadiusCm_ - radial * r.wa_;
      double aux2 = ::sqrt(radial * radial * (1. - r.wa_) * (1. + r.wa_) +
                           (-r.observer_.z_ - q.observationHeightCm_) *
                               (-r.observer_.z_ - q.observationHeightCm_));
      r.wap_ = aux2 > 0. ? -(q.observationHeightCm_ + r.observer_.z_) / aux2 : 0.;
      // Original ELECTR branches to its observation endpoint before normal
      // time/rotation handling. Caller must handle this disposition explicitly.
      if (!q.photon_ && q.flatOutput_ && -r.observer_.z_ <= q.observationHeightCm_) {
        r.status_ = CurvedStatus::observation;
        return r;
      }
      r.wap_ = smaller(1., r.wap_);
      if (r.direction_.x_ != 0.) {
        double tanphi = r.direction_.y_ / r.direction_.x_;
        r.direction_.x_ =
            ::fabs(r.direction_.z_) < 1.
                ? ::copysign(1., r.direction_.x_) *
                      ::sqrt((1. - r.direction_.z_) * (1. + r.direction_.z_) /
                             (1. + tanphi * tanphi))
                : 0.;
        r.direction_.y_ = tanphi * r.direction_.x_;
      } else if (r.direction_.y_ != 0. && ::fabs(r.direction_.z_) < 1.) {
        r.direction_.y_ = ::copysign(1., r.direction_.y_) *
                          ::sqrt((1. - r.direction_.z_) * (1. + r.direction_.z_));
      } else
        r.direction_.y_ = 0.;
      if (r.wa_ != 1.) {
        // The local-to-observer patch is only valid in the detector hemisphere.
        if (r.wa_ <= 0.) return r;
        double cp, sp;
        if (q.directAzimuth_ && lateral > 0.) {
          cp = r.local_.x_ / lateral;
          sp = r.local_.y_ / lateral;
        } else {
          double phi = (r.local_.x_ != 0. || r.local_.y_ != 0.)
                           ? ::atan2(r.local_.y_, r.local_.x_)
                           : 0.;
          cp = ::cos(phi);
          sp = ::sin(phi);
        }
        double radius = ::sqrt((1. - r.wa_) * (1. + r.wa_)) *
                        (q.earthRadiusCm_ - r.observer_.z_) / r.wa_;
        r.observer_.x_ = radius * cp;
        r.observer_.y_ = radius * sp;
      } else {
        r.observer_.x_ = r.local_.x_;
        r.observer_.y_ = r.local_.y_;
      }
    } else
      r.observer_ = r.local_;
    double time = q.timePathCm_ * (1. / q.speedCmS_);
    if (!q.photon_)
      time /= ::sqrt((1. - q.massMeV_ / q.totalEnergyMeV_) *
                     (1. + q.massMeV_ / q.totalEnergyMeV_));
    double ratio = 1.;
    if (q.geometricStepCm_ > 1.e-10) {
      double dx = r.observer_.x_ - q.previousObserver_.x_,
             dy = r.observer_.y_ - q.previousObserver_.y_,
             dz = r.observer_.z_ - q.previousObserver_.z_;
      ratio = ::sqrt(dx * dx + dy * dy + dz * dz) / q.geometricStepCm_;
    }
    r.timeIncrementS_ = time * ratio;
    r.observerToLocalDistanceRatio_ = ratio;
    if (!valid_point(r.local_) || !valid_point(r.observer_) ||
        !valid_direction(r.direction_) || !finite(r.timeIncrementS_) ||
        r.timeIncrementS_ < 0. || !finite(ratio))
      return r;
    r.status_ = CurvedStatus::advanced;
    return r;
  }
} // namespace corsika::kokkos::egs4
