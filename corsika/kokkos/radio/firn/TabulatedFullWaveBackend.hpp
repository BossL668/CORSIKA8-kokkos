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
#include <corsika/kokkos/radio/firn/CoherentStratifiedGOBackend.hpp>
#include <corsika/kokkos/radio/firn/AnyGreenBackend.hpp>
#include <corsika/detail/kokkos/radio/firn/GreenResponseCommon.inl>

namespace corsika::kokkos::radio::firn {
  /** Strict, portable response-table reader for precomputed full-wave dyadics.
   *
   * Schema v1 stores global ENU dyadics on (frequency, source_z, horizontal_r).
   * The table is intentionally simple for the first compiled integration.  A
   * phase-reduced interpolation is used so geometric propagation phase is not
   * linearly interpolated as a rapidly wrapping complex number.
   */
  class TabulatedFullWaveBackend final : public GreenResponseBackend {
  public:
    TabulatedFullWaveBackend(std::filesystem::path manifestPath,
                             std::string expectedProfileFingerprint);

    GreenResponseBatch evaluate(vector_type const& source, vector_type const& observer,
                                FrequencyGrid const& grid) const final;
    std::string name() const final;

  private:
    static void requireVector(cnpy::NpyArray const& array, char const* name);
    void load(std::filesystem::path const& npzPath);
    static void increasing(std::vector<double> const& values, char const* name);
    static std::pair<std::size_t, double> bracket(std::vector<double> const& axis,
                                                  double value);
    std::size_t index(std::size_t fi, std::size_t zi, std::size_t ri, std::size_t row,
                      std::size_t col) const;
    double interpolatePath(double z, double radius) const;
    double interpolateSourceIndex(double z) const;
    double interpolateSpreading(double z, double radius) const;
    vector_type emitDirection(vector_type const& source,
                              vector_type const& observer) const;
    static complex_matrix_type rotateFromCylindrical(complex_matrix_type const& local,
                                                     vector_type const& source,
                                                     vector_type const& observer);
    void interpolateDyadic(double frequency, double z, double radius, double restorePath,
                           complex_matrix_type& output) const;

    std::filesystem::path manifestPath_;
    std::string profileFingerprint_;
    double observerZM_{0.0}, observerIndex_{1.0};
    std::vector<double> frequencyHz_, sourceZ_, radiusM_;
    std::vector<double> sourceIndex_, real_, imag_, pathM_, spreadingPerM_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/TabulatedFullWaveBackend.inl>
