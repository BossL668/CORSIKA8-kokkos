/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/Types.hpp>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <corsika/kokkos/radio/firn/MediumSample.hpp>
#include <corsika/kokkos/radio/firn/IsotropicOpticalField.hpp>
#include <corsika/kokkos/radio/firn/VerticalOpticalProfile.hpp>
#include <corsika/kokkos/radio/firn/UniformProfile.hpp>
#include <corsika/kokkos/radio/firn/LinearProfile.hpp>
#include <corsika/kokkos/radio/firn/ExponentialProfile.hpp>

namespace corsika::kokkos::radio::firn {
  /** TabulatedProfile from the validated stratified-radio implementation. */
  class TabulatedProfile final : public VerticalOpticalProfile {
  public:
    TabulatedProfile(std::vector<double> zM, std::vector<double> refractiveIndex,
                     std::vector<double> attenuationPerM = {});

    double minimumSupportedZ() const final;
    double maximumSupportedZ() const final;
    double refractiveIndexAtZ(double zM) const final;
    double attenuationAtZ(double zM, double) const final;
    bool attenuationIsFrequencyIndependent() const final;
    std::string profileFingerprint() const final;

  private:
    double interpolate(std::vector<double> const& values, double zM) const;
    std::vector<double> zM_, n_, alpha_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/TabulatedProfile.inl>
