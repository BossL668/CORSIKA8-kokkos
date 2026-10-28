/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/Types.hpp>

namespace corsika::kokkos::radio::firn {
  /** Frequency-domain propagation response, including polarization and coherent branches.
   */
  class GreenResponseBackend {
  public:
    virtual ~GreenResponseBackend() = default;
    virtual GreenResponseBatch evaluate(vector_type const& source,
                                        vector_type const& observer,
                                        FrequencyGrid const& grid) const = 0;
    virtual std::string name() const = 0;
  };
} // namespace corsika::kokkos::radio::firn
