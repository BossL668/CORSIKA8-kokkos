/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline LinearProfile::LinearProfile(double z0M, double z1M, double n0, double n1,
                                      double attenuationPerM)
      : z0M_(z0M)
      , z1M_(z1M)
      , n0_(n0)
      , n1_(n1)
      , attenuationPerM_(attenuationPerM) {
    if (!(z1M_ > z0M_) || !(n0_ >= 1.0) || !(n1_ >= 1.0) || !(attenuationPerM_ >= 0.0)) {
      throw std::runtime_error("invalid linear optical profile");
    }
  }

  inline double LinearProfile::minimumSupportedZ() const { return z0M_; }

  inline double LinearProfile::maximumSupportedZ() const { return z1M_; }

  inline double LinearProfile::refractiveIndexAtZ(double zM) const {
    double const fraction = (zM - z0M_) / (z1M_ - z0M_);
    return n0_ + fraction * (n1_ - n0_);
  }

  inline double LinearProfile::attenuationAtZ(double, double) const {
    return attenuationPerM_;
  }

  inline bool LinearProfile::attenuationIsFrequencyIndependent() const { return true; }

  inline std::string LinearProfile::profileFingerprint() const {
    std::ostringstream stream;
    stream << "linear_z" << std::setprecision(17) << ':' << z0M_ << ':' << z1M_ << ':'
           << n0_ << ':' << n1_ << ':' << attenuationPerM_;
    return stream.str();
  }
} // namespace corsika::kokkos::radio::firn
