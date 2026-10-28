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
#include <corsika/kokkos/radio/firn/StraightAverageIndexBackend.hpp>
#include <corsika/kokkos/radio/firn/StratifiedGOBackend.hpp>
#include <corsika/kokkos/radio/firn/StratifiedTurningGOBackend.hpp>
#include <corsika/kokkos/radio/firn/StratifiedSurfaceReflectionBackend.hpp>
#include <corsika/detail/kokkos/radio/firn/GreenResponseCommon.inl>

namespace corsika::kokkos::radio::firn {
  /** Coherent ordinary-GO sum of every supported firn ray family.
   *
   * The returned branches deliberately remain separate.  OnlineRadioProcess
   * applies the finite-track endpoint phase to each branch before summing the
   * complex electric field.  A small turning-branch time separation is treated
   * as a fold/caustic and rejected because ordinary GO is singular there.
   */
  class CoherentStratifiedGOBackend final : public GreenResponseBackend {
  public:
    CoherentStratifiedGOBackend(std::shared_ptr<VerticalOpticalProfile const> profile,
                                double exteriorRefractiveIndex,
                                bool includeSurfaceReflection = true,
                                bool includeTurning = true,
                                double foldGuardTimeS = 2.0e-9);

    GreenResponseBatch evaluate(vector_type const& source, vector_type const& observer,
                                FrequencyGrid const& grid) const final;

    std::string name() const final;

  private:
    GreenResponseBranch goBranch(PathDiagnostics path, FrequencyGrid const& grid) const;

    GreenResponseBranch reflectedBranch(PathDiagnostics path,
                                        FrequencyGrid const& grid) const;

    std::shared_ptr<VerticalOpticalProfile const> profile_;
    StratifiedRaySolver solver_;
    double exteriorIndex_;
    bool includeSurfaceReflection_, includeTurning_;
    double foldGuardTimeS_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/CoherentStratifiedGOBackend.inl>
