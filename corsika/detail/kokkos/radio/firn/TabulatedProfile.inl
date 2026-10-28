/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline TabulatedProfile::TabulatedProfile(std::vector<double> zM,
                                            std::vector<double> refractiveIndex,
                                            std::vector<double> attenuationPerM)
      : zM_(std::move(zM))
      , n_(std::move(refractiveIndex))
      , alpha_(std::move(attenuationPerM)) {
    if (zM_.size() < 2 || n_.size() != zM_.size()) {
      throw std::runtime_error("tabulated profile grids have inconsistent size");
    }
    if (alpha_.empty()) alpha_.assign(zM_.size(), 0.0);
    if (alpha_.size() != zM_.size()) {
      throw std::runtime_error("tabulated attenuation grid has wrong size");
    }
    for (std::size_t index = 0; index < zM_.size(); ++index) {
      if (!std::isfinite(zM_[index]) || !(n_[index] >= 1.0) || !(alpha_[index] >= 0.0) ||
          !std::isfinite(n_[index]) || !std::isfinite(alpha_[index]) ||
          (index && !(zM_[index] > zM_[index - 1]))) {
        throw std::runtime_error("invalid tabulated optical profile");
      }
    }
  }

  inline double TabulatedProfile::minimumSupportedZ() const { return zM_.front(); }

  inline double TabulatedProfile::maximumSupportedZ() const { return zM_.back(); }

  inline double TabulatedProfile::refractiveIndexAtZ(double zM) const {
    return interpolate(n_, zM);
  }

  inline double TabulatedProfile::attenuationAtZ(double zM, double) const {
    return interpolate(alpha_, zM);
  }

  inline bool TabulatedProfile::attenuationIsFrequencyIndependent() const { return true; }

  inline std::string TabulatedProfile::profileFingerprint() const {
    std::ostringstream stream;
    stream << "tabulated_z" << std::setprecision(17);
    for (std::size_t index = 0; index < zM_.size(); ++index) {
      stream << ':' << zM_[index] << ',' << n_[index] << ',' << alpha_[index];
    }
    return stream.str();
  }

  inline double TabulatedProfile::interpolate(std::vector<double> const& values,
                                              double zM) const {
    if (zM <= zM_.front()) return values.front();
    if (zM >= zM_.back()) return values.back();
    auto const upper = std::upper_bound(zM_.begin(), zM_.end(), zM);
    std::size_t const hi = static_cast<std::size_t>(upper - zM_.begin());
    std::size_t const lo = hi - 1;
    double const fraction = (zM - zM_[lo]) / (zM_[hi] - zM_[lo]);
    return values[lo] + fraction * (values[hi] - values[lo]);
  }
} // namespace corsika::kokkos::radio::firn
