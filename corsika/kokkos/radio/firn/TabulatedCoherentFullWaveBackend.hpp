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
#include <corsika/kokkos/radio/firn/TabulatedFullWaveBackend.hpp>
#include <corsika/detail/kokkos/radio/firn/GreenResponseCommon.inl>

namespace corsika::kokkos::radio::firn {
  /** Banded, phase-reduced full-Maxwell total response (schema v2).
   *
   * Each table contains the two independently validated modal sums used by the
   * mountain full-field chain: a matched/direct response and the air-boundary
   * correction.  Frequency interpolation is a not-a-knot cubic represented by
   * its node values and node second derivatives.  The interpolated quantity is
   * f*G with the branch carrier phase removed; this remains smooth through the
   * direct/turning endpoint where ordinary GO is ill-conditioned.
   */
  class TabulatedCoherentFullWaveBackend final : public GreenResponseBackend {
  public:
    TabulatedCoherentFullWaveBackend(std::filesystem::path manifestPath,
                                     std::string expectedProfileFingerprint);

    bool covers(vector_type const& source, vector_type const& observer,
                FrequencyGrid const& grid) const;

    GreenResponseBatch evaluate(vector_type const& source, vector_type const& observer,
                                FrequencyGrid const& grid) const final;

    std::string name() const final;

  private:
    struct BranchBand {
      std::vector<double> real_, imag_, secondReal_, secondImag_;
    };
    struct Band {
      std::vector<double> frequencyHz_;
      BranchBand direct_, boundary_;
    };

    static void requireVector(cnpy::NpyArray const& array, char const* name);

    void load(std::filesystem::path const& npzPath);

    BranchBand loadBranchBand(cnpy::npz_t const& archive, std::string const& prefix,
                              std::string const& branch,
                              std::size_t frequencyCount) const;

    std::vector<double> loadSurface(cnpy::NpyArray const& value, char const* name) const;

    static void increasing(std::vector<double> const& values, char const* name);

    static std::pair<std::size_t, double> bracket(std::vector<double> const& axis,
                                                  double value);

    std::size_t flat(std::size_t fi, std::size_t zi, std::size_t ri, std::size_t row,
                     std::size_t col) const;

    double spatial(std::vector<double> const& values, std::size_t fi, double z,
                   double radius, std::size_t row, std::size_t col) const;

    double surface(std::vector<double> const& values, double z, double radius) const;

    Band const& selectBand(double frequency) const;

    std::complex<double> interpolateReduced(Band const& band, BranchBand const& data,
                                            double frequency, double z, double radius,
                                            std::size_t row, std::size_t col) const;

    vector_type emitDirection(std::vector<double> const& path, vector_type const& source,
                              vector_type const& observer) const;

    static complex_matrix_type rotateFromCylindrical(complex_matrix_type const& local,
                                                     vector_type const& source,
                                                     vector_type const& observer);

    GreenResponseBranch branch(vector_type const& source, vector_type const& observer,
                               double radius, FrequencyGrid const& grid,
                               std::vector<double> const& pathValues, bool direct,
                               char const* topology) const;

    std::filesystem::path manifestPath_;
    std::string profileFingerprint_;
    double observerZM_{0.0}, observerIndex_{1.0};
    std::vector<double> sourceZ_, radiusM_, sourceIndex_;
    std::vector<double> directPathM_, boundaryPathM_;
    std::vector<Band> bands_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/TabulatedCoherentFullWaveBackend.inl>
