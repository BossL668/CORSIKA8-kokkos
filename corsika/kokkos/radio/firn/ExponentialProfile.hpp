/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/ExponentialOpticalParameters.hpp>
#include <corsika/kokkos/radio/firn/ExponentialEndpointData.hpp>

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

namespace corsika::kokkos::radio::firn {
  /** ExponentialProfile from the validated stratified-radio implementation. */
  class ExponentialProfile final : public VerticalOpticalProfile {
  public:
    ExponentialProfile(double deepIndex, double deltaIndex, double scaleDepthM,
                       double surfaceZM, double attenuationPerM = 0.0,
                       double minimumZM = -5000.0);

    double minimumSupportedZ() const final;
    double maximumSupportedZ() const final;
    double refractiveIndexAtZ(double zM) const final;
    double attenuationAtZ(double, double) const final;
    bool attenuationIsFrequencyIndependent() const final;
    std::string profileFingerprint() const final;
    ExponentialOpticalParameters parameters() const;

    /// Horizontal distance from an endpoint to an upper turning point.
    double turningLegHorizontal(double p, double endpointZM) const;

    /// Derivative dX/dp of one upper-turning leg, at a fixed endpoint.
    double turningLegHorizontalDerivative(double p, double endpointZM) const;

    /// q is the signed vertical ray cosine at the shallower endpoint.
    ExponentialEndpointData endpointRayData(double q, double deepZM,
                                            double shallowZM) const;

  private:
    double deepIndex_, deltaIndex_, scaleDepthM_, surfaceZM_;
    double attenuationPerM_, minimumZM_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/ExponentialProfile.inl>
