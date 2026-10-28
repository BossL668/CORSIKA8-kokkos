/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/CoordinateSystem.hpp>
#include <corsika/framework/geometry/IVolume.hpp>
#include <corsika/kokkos/environment/EnvironmentShape.hpp>

namespace corsika::kokkos {

  /**
   * Export native geometry parameters on the host without modifying the volume.
   *
   * Supports the exact native Universe, Sphere, Box and SeparationPlane types.
   * A subclass can override containment, so it is not silently exported as its
   * base shape. Uses native coordinate conversions, with no geometric fitting,
   * extra normalization or change to the signed dimensions. Universe has a
   * dedicated tag rather than exporting its inherited infinite sphere radius.
   *
   * \param volume An existing volume, not a possibly empty tree-node volume.
   * \param coordinateSystem Common frame for the exported points and vectors.
   * \return An independent numeric shape record, suitable for device copies.
   * \throws std::invalid_argument For an unsupported type, a null source or
   * common coordinate system, or non-finite exported geometry parameters.
   * \note Host-only adapter; performs no node search or boundary intersection.
   */
  [[nodiscard]] inline EnvironmentShape export_shape(
      IVolume const& volume, CoordinateSystemPtr const& coordinateSystem);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/ShapeConversion.inl>
