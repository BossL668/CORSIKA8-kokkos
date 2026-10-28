/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/modules/radio/propagators/RadioPropagator.hpp>
#include <corsika/framework/core/PhysicalConstants.hpp>
#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace corsika {
  /** Straight propagation through one homogeneous, nonattenuating medium.
   * Preserves the archived ice propagator's endpoint-index check and n R/c delay.
   * The caller must ensure the entire segment lies in that medium; endpoint
   * equality alone does not establish this for a general environment tree.
   */
  template <typename TEnvironment>
  class HomogeneousRadioPropagator final
      : public RadioPropagator<HomogeneousRadioPropagator<TEnvironment>, TEnvironment> {
    using base_type =
        RadioPropagator<HomogeneousRadioPropagator<TEnvironment>, TEnvironment>;
    using signal_path_collection_type = typename base_type::SignalPathCollection;

  public:
    explicit HomogeneousRadioPropagator(TEnvironment const& environment);
    /// Query endpoint indices and return the single direct path, without sampling.
    template <typename TParticle>
    signal_path_collection_type propagate(TParticle const&, Point const&, Point const&);
  };
} // namespace corsika

#include <corsika/detail/modules/radio/propagators/HomogeneousRadioPropagator.inl>
