/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/StratifiedRay.hpp>
#include <corsika/kokkos/radio/firn/Sha256.hpp>
#include <cnpy.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <memory>
#include <corsika/kokkos/radio/firn/GreenResponseBackend.hpp>
#include <corsika/kokkos/radio/firn/UniformAnalyticBackend.hpp>
#include <corsika/detail/kokkos/radio/firn/GreenResponseCommon.inl>

namespace corsika::kokkos::radio::firn {
  /** Historical straight-line approximation with a path-averaged index.
   *
   * This intentionally omits ray bending and ray-tube focusing.  It is kept as
   * a controlled baseline for deciding whether the stratified solution is
   * materially different, not as the production firn solution.
   */
  class StraightAverageIndexBackend final : public GreenResponseBackend {
  public:
    explicit StraightAverageIndexBackend(
        std::shared_ptr<VerticalOpticalProfile const> profile);

    GreenResponseBatch evaluate(vector_type const& source, vector_type const& observer,
                                FrequencyGrid const& grid) const final;

    std::string name() const final;

  private:
    std::shared_ptr<VerticalOpticalProfile const> profile_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/StraightAverageIndexBackend.inl>
