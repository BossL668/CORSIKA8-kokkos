/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika::kokkos::radio {
  template <typename TC, typename TZ>
  RadioConfig make_homogeneous_radio_config(double refractiveIndex,
                                            CoordinateSystemPtr const& coordinates,
                                            TC const& coreas, TZ const& zhs,
                                            bool coreasEnabled, bool zhsEnabled) {
    if (!std::isfinite(refractiveIndex) || refractiveIndex < 1.) {
      throw std::invalid_argument("invalid homogeneous refractive index");
    }
    RadioConfig result;
    result.enabled_ = coreasEnabled || zhsEnabled;
    result.coreasEnabled_ = coreasEnabled;
    result.zhsEnabled_ = zhsEnabled;
    result.propagation_.homogeneousRefractiveIndex_ = refractiveIndex;
    if (coreasEnabled)
      result.coreasObservers_ = snapshot_detail::make_observers(coreas, coordinates);
    if (zhsEnabled)
      result.zhsObservers_ = snapshot_detail::make_observers(zhs, coordinates);
    return result;
  }
} // namespace corsika::kokkos::radio
