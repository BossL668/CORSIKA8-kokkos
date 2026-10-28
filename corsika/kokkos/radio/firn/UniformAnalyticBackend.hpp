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
#include <corsika/detail/kokkos/radio/firn/GreenResponseCommon.inl>

namespace corsika::kokkos::radio::firn {
  /** UniformAnalyticBackend from the validated stratified-radio implementation. */
  class UniformAnalyticBackend final : public GreenResponseBackend {
  public:
    explicit UniformAnalyticBackend(UniformProfile profile);

    GreenResponseBatch evaluate(vector_type const& source, vector_type const& observer,
                                FrequencyGrid const& grid) const final;
    std::string name() const final;

  private:
    UniformProfile profile_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/UniformAnalyticBackend.inl>
