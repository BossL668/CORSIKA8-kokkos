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

namespace corsika::kokkos::radio::firn {
  /** VerticalOpticalProfile from the validated stratified-radio implementation. */
  class VerticalOpticalProfile : public IsotropicOpticalField {
  public:
    bool isVerticallyStratified() const final;
    virtual bool attenuationIsFrequencyIndependent() const;

    MediumSample sample(vector_type const& position, double frequencyHz) const final;

    virtual double refractiveIndexAtZ(double zM) const = 0;
    virtual double attenuationAtZ(double zM, double frequencyHz) const = 0;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/VerticalOpticalProfile.inl>
