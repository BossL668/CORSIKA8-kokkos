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

namespace corsika::kokkos::radio::firn {
  /** IsotropicOpticalField from the validated stratified-radio implementation. */
  class IsotropicOpticalField {
  public:
    virtual ~IsotropicOpticalField() = default;
    virtual MediumSample sample(vector_type const& position,
                                double frequencyHz) const = 0;
    virtual std::string profileFingerprint() const = 0;
    virtual bool isVerticallyStratified() const;
    virtual double minimumSupportedZ() const;
    virtual double maximumSupportedZ() const;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/IsotropicOpticalField.inl>
