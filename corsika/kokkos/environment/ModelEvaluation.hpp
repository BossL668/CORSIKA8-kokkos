/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentModel.hpp>
#include <corsika/kokkos/environment/firn/FirnModel.hpp>

#include <corsika/framework/core/HostDevice.hpp>

namespace corsika::kokkos {

  /**
   * Evaluate the exported density law at one point, on host or device.
   *
   * \param model Valid homogeneous, sliding-planar exponential or flat
   * exponential record produced by export_model(). Invalid/unknown density
   * kinds are outside this function's contract; validate at the export boundary.
   * \param positionM Position in metres in the model's common coordinate system.
   * \return Mass density in g/cm^3, using the fixed production expressions.
   *
   * Homogeneous density does not access positionM. The sliding-planar model
   * uses radial height for this point value; the flat model uses its stored
   * axis without normalization. No clipping, finite-value repair, geometry
   * search or trajectory integration is performed. IEEE exceptional values
   * propagate as in the production evaluator.
   */
  CORSIKA_HOST_DEVICE inline double density(EnvironmentModel const& model,
                                            double const (&positionM)[3]);

  /**
   * Evaluate the exported refractive-index law at one point, on host or device.
   *
   * \param model Record with a uniform, Gladstone-Dale or unavailable index.
   * Gladstone-Dale additionally requires a valid density record as for density().
   * \param positionM Position in metres in the model's common coordinate system.
   * \return Dimensionless refractive index; quiet NaN when unavailable.
   *
   * Uniform index does not evaluate density. Gladstone-Dale retains the separate
   * reference refractivity and inverse-density factors and their production
   * multiplication order. No coordinate conversion or Kokkos runtime startup
   * is performed here. The uniform magnetic field is already available through
   * EnvironmentModel::magneticFieldT_ and needs no pointwise evaluator.
   */
  CORSIKA_HOST_DEVICE inline double refractive_index(EnvironmentModel const& model,
                                                     double const (&positionM)[3]);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/ModelEvaluation.inl>
