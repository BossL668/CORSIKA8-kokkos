/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/ModelEvaluation.hpp>

namespace corsika::kokkos {

  /**
   * Integrate the exported density model along a straight segment.
   *
   * \param model Valid homogeneous, sliding-planar exponential or flat
   * exponential or exponential-firn record produced by export_model().
   * \param positionM Starting position in metres in the model's common system.
   * \param direction Dimensionless direction in the same system; expected to
   * be unit length for lengthM to be a physical arclength. It is not normalized.
   * \param lengthM Segment length in metres, normally nonnegative.
   * \return Grammage in g/cm^2, preserving the production expressions.
   *
   * The sliding-planar model freezes its local radial axis at positionM; this
   * is not an integral of the changing radial density along the segment.
   * The caller supplies a segment in a single model and handles boundaries.
   * For the original atmosphere models, zero length returns zero before
   * evaluating the model. No clipping, model
   * search or finite-value repair is performed; other exceptional arithmetic
   * retains production behavior. Invalid density kinds are outside the contract.
   * Firn retains its archived surface-domain checks and analytic integral;
   * invalid finite segments return NaN instead of a host exception.
   */
  CORSIKA_HOST_DEVICE inline double grammage(EnvironmentModel const& model,
                                             double const (&positionM)[3],
                                             double const (&direction)[3],
                                             double lengthM);

  /**
   * Invert the same single-model straight-segment grammage expression.
   *
   * \param model Valid density record as for grammage().
   * \param positionM Starting position in metres in the model's common system.
   * \param direction Dimensionless direction in the same system; not normalized.
   * \param grammageGPerCm2 Requested grammage in g/cm^2, normally nonnegative.
   * \return Length in metres; positive infinity when the exponential branch's
   * log1p argument is not greater than -1, including an unordered comparison.
   *
   * Atmosphere models use the production analytic inverse. The
   * sliding-planar axis remains fixed at the start, and neither model boundaries
   * nor the observation plane limit this result. Homogeneous density or exactly
   * zero projection uses direct division, with its original IEEE behavior.
   * There is no special zero-grammage shortcut or new input protection.
   * The independent firn branch retains its archived 80-step bisection, including
   * its zero-grammage return and surface-limited upper bound. Requests exceeding
   * the available grammage below the surface return positive infinity.
   */
  CORSIKA_HOST_DEVICE inline double arclength(EnvironmentModel const& model,
                                              double const (&positionM)[3],
                                              double const (&direction)[3],
                                              double grammageGPerCm2);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/ModelGrammage.inl>
