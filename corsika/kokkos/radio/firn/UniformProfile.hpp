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

namespace corsika::kokkos::radio::firn {
  /** UniformProfile from the validated stratified-radio implementation. */
  class UniformProfile final : public VerticalOpticalProfile {
  public:
    UniformProfile(double refractiveIndex, double attenuationPerM = 0.0);

    double refractiveIndexAtZ(double) const final;
    double attenuationAtZ(double, double) const final;
    bool attenuationIsFrequencyIndependent() const final;
    std::string profileFingerprint() const final;

  private:
    static std::string canonical(std::string const& kind,
                                 std::initializer_list<double> values);
    void validate() const;
    double refractiveIndex_;
    double attenuationPerM_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/UniformProfile.inl>
