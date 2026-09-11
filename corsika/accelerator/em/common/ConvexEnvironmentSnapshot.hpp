/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */

#pragma once

#include <corsika/accelerator/em/common/SphericalAtmosphere.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>

namespace corsika::gpu::em {

  /**
   * Export one validated, bounded convex volume to the portable EM kernels.
   *
   * TConvexVolume is normally corsika::ConvexPolyhedron. Its exportPlanes()
   * supplies outward unit normals in the same metre-valued coordinate frame
   * used by the router. The original host Environment MUST attach the same
   * composition and mass density to this same volume. No bounding sphere,
   * atmosphere, material file-name inference or implicit CPU fallback is used.
   *
   * This first contract is one homogeneous material and B=0. All outward
   * crossings become EscapedEnvironment observations and terminate transport;
   * there is no continuation into air or refracted radio in this contract.
   */
  template <class TConvexVolume>
  EnvironmentSnapshot makeHomogeneousConvexSnapshot(
      double density_g_per_cm3, TConvexVolume const& volume,
      std::int32_t medium_id = 0,
      std::array<double, 3> const& magnetic_field_T = {0., 0., 0.},
      double boundary_tolerance_m = 1.e-8) {
    auto const& planes = volume.exportPlanes();
    if (planes.size() < 4 || planes.size() > MaxConvexEnvironmentPlanes) {
      throw std::invalid_argument(
          "bounded Kokkos EM geometry requires 4 to 32 convex planes");
    }
    EnvironmentSnapshot snapshot{};
    snapshot.geometry = EnvironmentGeometry::HomogeneousConvexPolyhedron;
    snapshot.number_of_layers = 1;
    snapshot.number_of_convex_planes =
        static_cast<std::uint32_t>(planes.size());
    snapshot.convex_boundary_tolerance_m = boundary_tolerance_m;
    snapshot.atmosphere_layers[0].density_model = DensityModel::Homogeneous;
    snapshot.atmosphere_layers[0].density_parameter_a = density_g_per_cm3;
    snapshot.atmosphere_layers[0].medium_id = medium_id;
    // Plane intersection is disabled by the explicit geometry discriminator.
    // Retain a valid normal for generic diagnostics, without a fictitious
    // observation surface or spherical shell controlling transport.
    snapshot.observation_plane_normal[2] = 1.;
    for (std::size_t face = 0; face < planes.size(); ++face)
      snapshot.convex_planes[face] = planes[face];
    for (std::size_t axis = 0; axis < 3; ++axis)
      snapshot.magnetic_field_T[axis] = magnetic_field_T[axis];
    if (!atmosphere_detail::validEnvironment(snapshot)) {
      throw std::invalid_argument(
          "invalid homogeneous convex snapshot: density/planes/tolerance "
          "must be finite and valid, and magnetic field must be zero");
    }
    return snapshot;
  }

} // namespace corsika::gpu::em
