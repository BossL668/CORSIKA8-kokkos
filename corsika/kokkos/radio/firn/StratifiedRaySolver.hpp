/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/OpticalField.hpp>
#include <functional>
#include <corsika/detail/kokkos/radio/firn/StratifiedRayCommon.inl>

namespace corsika::kokkos::radio::firn {
  /** StratifiedRaySolver from the validated stratified-radio implementation. */
  class StratifiedRaySolver {
  public:
    explicit StratifiedRaySolver(VerticalOpticalProfile const& profile);

    PathDiagnostics direct(vector_type const& source, vector_type const& observer,
                           double frequencyHz = 0.0) const;

    /** All ordinary upper-turning roots in a monotonic n(z) profile.
     *
     * A fold is identified by a vanishing ray-tube Jacobian and is deliberately
     * omitted: a caller must route that query to a full-wave table.
     */
    std::vector<PathDiagnostics> turningBranches(vector_type const& source,
                                                 vector_type const& observer,
                                                 double frequencyHz = 0.0) const;

    /** Specular path reflected by the horizontal upper boundary of the profile. */
    PathDiagnostics surfaceReflected(vector_type const& source,
                                     vector_type const& observer,
                                     double frequencyHz = 0.0) const;

  private:
    PathDiagnostics nearEndpoint(vector_type const&, vector_type const&, bool) const;
    VerticalOpticalProfile const& profile_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/StratifiedRaySolver.inl>
