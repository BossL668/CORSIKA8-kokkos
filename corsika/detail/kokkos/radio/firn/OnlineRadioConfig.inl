/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline void OnlineRadioConfig::validate() const {
    grid_.validate();
    if (!std::isfinite(timeOriginS_) || !(maximumPhaseCycles_ > 0.0) ||
        !(maximumLinearizationLengthM_ > 0.0) || !(maximumRefractiveIndex_ >= 1.0) ||
        maximumSubdivisions_ == 0 || maximumResponseQueries_ == 0 ||
        !(maximumPointCurrentPhaseIndicator_ > 0.0) ||
        !std::isfinite(maximumPhaseCycles_) ||
        !std::isfinite(maximumLinearizationLengthM_) ||
        !std::isfinite(maximumRefractiveIndex_) ||
        !std::isfinite(maximumPointCurrentPhaseIndicator_)) {
      throw std::runtime_error("invalid online variable-index radio config");
    }
  }
} // namespace corsika::kokkos::radio::firn
