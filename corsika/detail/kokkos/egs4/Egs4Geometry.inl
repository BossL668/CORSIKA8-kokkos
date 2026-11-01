/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4CurvedFrame.inl>

namespace corsika::kokkos::egs4 {
  // Native HOWFAR for the C7 CURVED/UPWARD or flat/UPWARD variants, with one
  // downward-shower observation level. Region numbers retain C7 semantics:
  // 1 above air, 2..5 air, 6 below air. UPWARDOLD is deliberately not selected.
  // The atmospheric thickness at the current height is supplied by the caller.
  struct GeometryConfig {
    bool curved_{true}, flatOutput_{true};
    double earthRadiusCm_{}, boundCm_[6]{}, observationHeightCm_{};
    double minimumHorizontalCm_{}, maximumHorizontalCm_{}, horizontalSlopeCmPerGCm2_{};
  };
  struct GeometryInput {
    Point local_;
    Direction direction_;
    int region_{2};
    double wa_{1.}, thicknessGCm2_{}, requestedCm_{}, nearestCm_{};
  };
  struct GeometryResult {
    bool valid_{}, energyDiscard_{};
    int region_{}, nextObservation_{1}, discard_{};
    double stepCm_{}, nearestCm_{};
  };
  KOKKOS_INLINE_FUNCTION GeometryResult how_far(GeometryConfig c, GeometryInput q) {
    GeometryResult r;
    r.region_ = q.region_;
    r.stepCm_ = q.requestedCm_;
    r.nearestCm_ = q.nearestCm_;
    if (!valid_point(q.local_) || !valid_direction(q.direction_) || q.region_ < 1 ||
        q.region_ > 6 || !finite(q.requestedCm_) || q.requestedCm_ < 0. ||
        !finite(q.nearestCm_) || !finite(q.wa_) || q.wa_ <= 0. || q.wa_ > 1. ||
        !finite(c.earthRadiusCm_) || c.earthRadiusCm_ <= 0. ||
        !finite(c.observationHeightCm_) || !finite(q.thicknessGCm2_) ||
        q.thicknessGCm2_ < 0. || !finite(c.minimumHorizontalCm_) ||
        c.minimumHorizontalCm_ <= 0. || !finite(c.maximumHorizontalCm_) ||
        c.maximumHorizontalCm_ <= 0. || !finite(c.horizontalSlopeCmPerGCm2_))
      return r;
    for (int i = 0; i < 6; ++i)
      if (!finite(c.boundCm_[i]) || (i && c.boundCm_[i] >= c.boundCm_[i - 1])) return r;
    r.valid_ = true;
    if (q.region_ == 6) {
      r.discard_ = 1;
      r.energyDiscard_ = true;
      return r;
    }
    if (q.region_ == 1) {
      if (q.direction_.z_ > 0.) {
        r.stepCm_ = .0001;
        r.region_ = 2;
      } else {
        r.discard_ = 1;
        r.energyDiscard_ = false;
      }
      return r;
    }
    int ir = q.region_;
    double z = q.local_.z_, w = q.direction_.z_, upper = c.boundCm_[ir - 2],
           lower = c.boundCm_[ir - 1];
    double obs = c.observationHeightCm_, obsglob = obs, toaxis = 0., aux3 = w;
    if (c.curved_) {
      double horizontal = larger(
          q.direction_.x_ * q.direction_.x_ + q.direction_.y_ * q.direction_.y_, .001);
      double aux =
          larger(c.minimumHorizontalCm_,
                 c.horizontalSlopeCmPerGCm2_ * q.thicknessGCm2_ + c.maximumHorizontalCm_);
      r.stepCm_ = smaller(r.stepCm_, aux / ::sqrt(horizontal));
      double xnew = q.local_.x_ + q.direction_.x_ * r.stepCm_,
             ynew = q.local_.y_ + q.direction_.y_ * r.stepCm_;
      if (((::fabs(xnew) < ::fabs(q.local_.x_) || xnew * q.local_.x_ <= 0.) &&
           (::fabs(ynew) < ::fabs(q.local_.y_) || ynew * q.local_.y_ <= 0.)) ||
          xnew * xnew + ynew * ynew <
              q.local_.x_ * q.local_.x_ + q.local_.y_ * q.local_.y_)
        toaxis = 1.;
      else
        toaxis = -1.;
      if (c.flatOutput_) obsglob = (c.earthRadiusCm_ + obs) / q.wa_ - c.earthRadiusCm_;
      aux3 = w * q.wa_;
      if (::fabs(w) < 1.)
        aux3 -= toaxis * ::sqrt((1. - w) * (1. + w) * (1. - q.wa_) * (1. + q.wa_));
    }
    bool down =
        c.curved_ ? ((c.flatOutput_ && aux3 > 0. && ir == 5) || w > 0.) : w > .0003;
    if (down) {
      double distance;
      bool reaches = true;
      if (c.curved_) {
        double radius = c.earthRadiusCm_ - z,
               target = larger(obs, lower - .1) + c.earthRadiusCm_;
        double impact = ::fabs(w) < 1. ? radius * ::sqrt((1. - w) * (1. + w)) : 0.;
        double sphere;
        if (target >= impact && w > 0.)
          sphere = radius * w - ::sqrt((target + impact) * (target - impact));
        else {
          sphere = radius * ::fabs(w);
          reaches = false;
        }
        if (c.flatOutput_ && aux3 > 0.) {
          distance = (radius * q.wa_ - c.earthRadiusCm_ - obs) / aux3;
          if (distance <= sphere)
            reaches = true;
          else
            distance = sphere;
        } else
          distance = sphere;
      } else
        distance = (-z - larger(lower, obs)) / w;
      if (distance > r.stepCm_) {
        r.nearestCm_ = smaller(z + upper, -z - larger(lower, obsglob));
        if (c.curved_)
          r.nearestCm_ =
              smaller(r.nearestCm_, (reaches ? distance : r.stepCm_) * ::fabs(w));
      } else {
        r.stepCm_ = larger(distance, .0001);
        if (reaches) {
          if (larger(obsglob, lower) > obsglob) {
            r.region_ = ir + 1;
            if (r.region_ >= 6) {
              r.discard_ = -1;
              r.energyDiscard_ = true;
            }
          } else {
            r.nextObservation_ = 2;
            r.discard_ = -1;
            r.energyDiscard_ = true;
          }
        }
      }
    } else if (c.curved_ || w < -.003) {
      double distance;
      if (c.curved_) {
        if (::fabs(w) <= .003) {
          if (upper > -z)
            distance =
                ::sqrt((2. * c.earthRadiusCm_ + upper + .1 - z) * (upper + .1 + z));
          else {
            // Preserve the original recovery from a stale layer index.
            while (c.boundCm_[r.region_ - 2] <= -z) {
              --r.region_;
              if (r.region_ <= 1) {
                r.discard_ = 1;
                r.energyDiscard_ = false;
                return r;
              }
            }
            double corrected = c.boundCm_[r.region_ - 2];
            distance = ::sqrt((2. * c.earthRadiusCm_ + corrected + .1 - z) *
                              (corrected + .1 + z));
          }
        } else
          distance = ::fabs((-(upper + .1) - z) / w);
      } else
        distance = (-upper - z) / w;
      if (distance > r.stepCm_)
        r.nearestCm_ = smaller(z + upper, -z - larger(lower, obsglob));
      else {
        r.stepCm_ = larger(distance, .0001);
        r.region_ = ir - 1;
        if (r.region_ <= 1) {
          r.discard_ = -2;
          r.energyDiscard_ = false;
        }
      }
    }
    // Flat UPWARD near-horizontal paths retain the requested step and DNEAR.
    if (!finite(r.stepCm_) || r.stepCm_ < 0. || !finite(r.nearestCm_)) r.valid_ = false;
    return r;
  }
} // namespace corsika::kokkos::egs4
