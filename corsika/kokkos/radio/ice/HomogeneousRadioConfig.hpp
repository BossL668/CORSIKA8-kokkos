/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/kokkos/radio/RadioConfig.hpp>
#include <corsika/kokkos/radio/ObserverConversion.hpp>
#include <cmath>
#include <stdexcept>

namespace corsika::kokkos::radio {
  /** Configure the existing exact homogeneous propagation kernel and observers.
   * Positions, vectors and observer polarizations use the supplied common frame.
   * The scene adapter must validate that whole rays remain in the uniform domain.
   */
  template <typename TC, typename TZ>
  RadioConfig make_homogeneous_radio_config(double refractiveIndex,
                                            CoordinateSystemPtr const& coordinates,
                                            TC const& coreas, TZ const& zhs,
                                            bool coreasEnabled, bool zhsEnabled);
} // namespace corsika::kokkos::radio

#include <corsika/detail/kokkos/radio/ice/HomogeneousRadioConfig.inl>
