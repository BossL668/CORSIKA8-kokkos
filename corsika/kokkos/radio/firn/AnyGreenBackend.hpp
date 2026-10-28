/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/GreenResponseBackend.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

namespace corsika::kokkos::radio::firn {
  /** Copyable runtime adapter used by the standalone CORSIKA application. */
  class AnyGreenBackend final : public GreenResponseBackend {
  public:
    explicit AnyGreenBackend(std::shared_ptr<GreenResponseBackend const> backend);
    GreenResponseBatch evaluate(vector_type const& source, vector_type const& observer,
                                FrequencyGrid const& grid) const final;
    std::string name() const final;

  private:
    std::shared_ptr<GreenResponseBackend const> backend_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/AnyGreenBackend.inl>
