/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentModel.hpp>
#include <corsika/framework/core/HostDevice.hpp>

namespace corsika::kokkos {

  /** Evaluate the exported ExponentialFirn profile in its common coordinate system.
   * Positions and lengths are metres; densities are g/cm^3 and grammages g/cm^2.
   * These functions require a validated firn record. The model's original
   * infinity sentinels and surface tolerances are retained. Invalid finite
   * queries return NaN on device instead of throwing a host exception; the
   * existing EnvironmentQuery status wrapper then reports the failure.
   */
  CORSIKA_HOST_DEVICE inline double firn_density(EnvironmentModel const& model,
                                                 double const (&positionM)[3]);
  CORSIKA_HOST_DEVICE inline double firn_refractive_index(EnvironmentModel const& model,
                                                          double const (&positionM)[3]);
  CORSIKA_HOST_DEVICE inline double firn_grammage(EnvironmentModel const& model,
                                                  double const (&positionM)[3],
                                                  double const (&direction)[3],
                                                  double lengthM);
  /// The original surface-bounded, 80-step bisection; no new inverse approximation.
  CORSIKA_HOST_DEVICE inline double firn_arclength(EnvironmentModel const& model,
                                                   double const (&positionM)[3],
                                                   double const (&direction)[3],
                                                   double grammageGPerCm2);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/firn/FirnModel.inl>
