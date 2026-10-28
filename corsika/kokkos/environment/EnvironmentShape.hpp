/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::kokkos {

  /**
   * Pointer-free shape parameters expressed in one caller-selected coordinate system.
   *
   * Lengths are in metres. Sphere uses centerM_ and radiusM_; Box uses centerM_,
   * halfExtentM_ and axes_[localAxis][commonComponent]. SeparationPlane uses
   * centerM_ and its outward normal_. Universe needs only its tag. Unused fields
   * are zero when exported. Finite signed dimensions are preserved, not repaired.
   *
   * This transfer record owns values, not native geometry objects. It has no
   * containment or intersection algorithm and no topology or material index.
   * Re-export after changing the source geometry. Native padding and byte order
   * are not a portable serialization format.
   */
  struct EnvironmentShape {
    enum class Kind : std::uint32_t { Invalid, Universe, Sphere, Box, SeparationPlane };

    Kind kind_{Kind::Invalid};
    double centerM_[3]{};
    double radiusM_{};
    double halfExtentM_[3]{};
    double axes_[3][3]{};
    double normal_[3]{};
  };

} // namespace corsika::kokkos
