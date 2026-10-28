/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline AnyGreenBackend::AnyGreenBackend(
      std::shared_ptr<GreenResponseBackend const> backend)
      : backend_(std::move(backend)) {
    if (!backend_) throw std::runtime_error("empty Green backend adapter");
  }

  inline GreenResponseBatch AnyGreenBackend::evaluate(vector_type const& source,
                                                      vector_type const& observer,
                                                      FrequencyGrid const& grid) const {
    return backend_->evaluate(source, observer, grid);
  }

  inline std::string AnyGreenBackend::name() const { return backend_->name(); }
} // namespace corsika::kokkos::radio::firn
