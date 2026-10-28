/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/CoordinateSystem.hpp>
#include <corsika/kokkos/environment/EnvironmentComponent.hpp>
#include <corsika/kokkos/environment/EnvironmentModel.hpp>

#include <vector>

namespace corsika::kokkos {

  /**
   * Convert exact native model parameters on the host and append their composition.
   *
   * Reads the three optional native export interfaces, getMedium(), and
   * getNuclearComposition(). Uses the production unit conversions and C8 coordinate
   * transformations without density sampling, fitting, normalization or physical
   * evaluation. Coordinates are converted on local copies of exported parameters;
   * the source model is unchanged. The native exporters must accurately describe
   * any overridden model behavior, as required by their interface contracts.
   *
   * Preserves finite and non-finite parameter values, signs, raw axes, component
   * ordering and fractions. The destination records are data, not a declaration
   * that a model is physically valid. The explicit interface, enum, frame and
   * array-size checks are initialization-time data-safety checks, not new model
   * algorithms. There is no automatic fallback for unsupported model types.
   *
   * \tparam TModel Polymorphic native properties exposing getMedium() and
   * getNuclearComposition(), as well as the three model-export interfaces.
   * \param model Source model, retained only for the duration of this call.
   * \param coordinateSystem Common frame for the exported point and vectors.
   * \param composition Array to append the native composition entries to. Existing
   * entries are preserved, and its size and contents are unchanged on failure.
   * \return A numeric model record identifying the newly appended component range.
   * \throws std::invalid_argument For a missing exporter, unknown model kind, or
   * null source or destination coordinate system.
   * \throws std::length_error If the resulting component array exceeds the
   * fixed-width index range or the destination vector's maximum size.
   * \note Host-only adapter; only the returned record and component values may be
   * copied to a device. No topology lookup, model evaluation or transport is done.
   */
  template <typename TModel>
  [[nodiscard]] EnvironmentModel export_model(
      TModel const& model, CoordinateSystemPtr const& coordinateSystem,
      std::vector<EnvironmentComponent>& composition);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/ModelConversion.inl>
