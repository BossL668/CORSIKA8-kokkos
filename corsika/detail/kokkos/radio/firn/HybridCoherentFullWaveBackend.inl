/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline HybridCoherentFullWaveBackend::HybridCoherentFullWaveBackend(
      std::shared_ptr<TabulatedCoherentFullWaveBackend const> fullWave,
      std::shared_ptr<GreenResponseBackend const> ordinary)
      : fullWave_(std::move(fullWave))
      , ordinary_(std::move(ordinary)) {
    if (!fullWave_ || !ordinary_) {
      throw std::runtime_error("hybrid full-wave backend needs both responses");
    }
  }

  inline GreenResponseBatch HybridCoherentFullWaveBackend::evaluate(
      vector_type const& source, vector_type const& observer,
      FrequencyGrid const& grid) const {
    if (fullWave_->covers(source, observer, grid)) {
      return fullWave_->evaluate(source, observer, grid);
    }
    return ordinary_->evaluate(source, observer, grid);
  }

  inline std::string HybridCoherentFullWaveBackend::name() const {
    return "hybrid_coherent_fullwave";
  }
} // namespace corsika::kokkos::radio::firn
