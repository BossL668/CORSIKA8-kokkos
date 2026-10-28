/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline bool VerticalOpticalProfile::isVerticallyStratified() const { return true; }

  inline bool VerticalOpticalProfile::attenuationIsFrequencyIndependent() const {
    return false;
  }

  inline MediumSample VerticalOpticalProfile::sample(vector_type const& position,
                                                     double frequencyHz) const {
    if (!finite(position) || !std::isfinite(frequencyHz) || frequencyHz < 0.0) {
      throw std::runtime_error("invalid optical-field query");
    }
    if (position[2] < minimumSupportedZ() - 1.0e-10 ||
        position[2] > maximumSupportedZ() + 1.0e-10) {
      throw std::runtime_error("optical-field query lies outside profile bounds");
    }
    MediumSample result{refractiveIndexAtZ(position[2]),
                        attenuationAtZ(position[2], frequencyHz)};
    if (!(result.refractiveIndex_ >= 1.0) || !(result.attenuationPerM_ >= 0.0) ||
        !std::isfinite(result.refractiveIndex_) ||
        !std::isfinite(result.attenuationPerM_)) {
      throw std::runtime_error("optical profile returned a non-physical sample");
    }
    return result;
  }
} // namespace corsika::kokkos::radio::firn
