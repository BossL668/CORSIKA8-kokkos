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

namespace corsika::kokkos::radio::firn {
  /** LinearProfile from the validated stratified-radio implementation. */
  class LinearProfile final : public VerticalOpticalProfile {
  public:
    LinearProfile(double z0M, double z1M, double n0, double n1,
                  double attenuationPerM = 0.0);

    double minimumSupportedZ() const final;
    double maximumSupportedZ() const final;
    double refractiveIndexAtZ(double zM) const final;
    double attenuationAtZ(double, double) const final;
    bool attenuationIsFrequencyIndependent() const final;
    std::string profileFingerprint() const final;

  private:
    double z0M_, z1M_, n0_, n1_, attenuationPerM_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/LinearProfile.inl>
