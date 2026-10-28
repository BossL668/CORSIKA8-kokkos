/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline UniformProfile::UniformProfile(double refractiveIndex, double attenuationPerM)
      : refractiveIndex_(refractiveIndex)
      , attenuationPerM_(attenuationPerM) {
    validate();
  }

  inline double UniformProfile::refractiveIndexAtZ(double) const {
    return refractiveIndex_;
  }

  inline double UniformProfile::attenuationAtZ(double, double) const {
    return attenuationPerM_;
  }

  inline bool UniformProfile::attenuationIsFrequencyIndependent() const { return true; }

  inline std::string UniformProfile::profileFingerprint() const {
    return canonical("uniform", {refractiveIndex_, attenuationPerM_});
  }

  inline std::string UniformProfile::canonical(std::string const& kind,
                                               std::initializer_list<double> values) {
    std::ostringstream stream;
    stream << kind << std::setprecision(17);
    for (double value : values) stream << ':' << value;
    return stream.str();
  }

  inline void UniformProfile::validate() const {
    if (!(refractiveIndex_ >= 1.0) || !(attenuationPerM_ >= 0.0) ||
        !std::isfinite(refractiveIndex_) || !std::isfinite(attenuationPerM_)) {
      throw std::runtime_error("invalid uniform optical profile");
    }
  }
} // namespace corsika::kokkos::radio::firn
