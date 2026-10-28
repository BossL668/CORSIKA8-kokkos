/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline bool IsotropicOpticalField::isVerticallyStratified() const { return false; }

  inline double IsotropicOpticalField::minimumSupportedZ() const { return -1.0e30; }

  inline double IsotropicOpticalField::maximumSupportedZ() const { return 1.0e30; }
} // namespace corsika::kokkos::radio::firn
