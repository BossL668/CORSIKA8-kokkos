/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/ExponentialOpticalParameters.hpp>
#include <corsika/kokkos/radio/firn/OnlineRadioConfig.hpp>
#include <corsika/kokkos/radio/firn/PathDiagnostics.hpp>
#include <array>
#include <cstdint>
#include <memory>
#include <limits>

namespace corsika::kokkos::radio::firn {
  /** Unsubdivided finite charged track in the common local SI coordinates. */
  struct GeometricalOpticsTrack {
    vector_type startM_;
    vector_type endM_;
    double startTimeS_;
    double endTimeS_;
    double chargeWeight_;
  };

  /** Ordinary GO shares the CPU model and its numerical tolerances.
   * The batch bounds the geometry workspace, independently of shower length.
   * The optical profile is exported from the very same CPU profile object.
   */
  struct GeometricalOpticsConfig {
    OnlineRadioConfig radio_{};
    std::size_t batchSegments_{32};
    std::size_t memoryLimitBytes_{512ULL * 1024 * 1024};
    double exteriorRefractiveIndex_{1.};
    double foldGuardTimeS_{2.e-9};
    bool includeSurfaceReflection_{true};
    bool includeTurning_{true};
  };

  struct GeometricalOpticsStatistics {
    std::uint64_t inputSegments_{};
    std::uint64_t subdividedSegments_{};
    std::uint64_t responseQueries_{};
    std::uint64_t branchContributions_{};
    std::uint64_t batches_{};
    std::uint64_t failedQueries_{};
    std::array<std::uint64_t, 7> branchCounts_{};
    std::size_t maximumSubdivisions_{};
    std::size_t allocatedBytes_{};
    double maximumPhaseCycles_{};
    double minimumTravelTimeS_{std::numeric_limits<double>::infinity()};
    double maximumTravelTimeS_{};
    PathDiagnostics worstFermatPath_{};
    double maximumSnellResidual_{};
    double geometryMs_{};
    double spectrumMs_{};
    double transferMs_{};
  };

  struct GeometricalOpticsSpectra {
    using components_type = std::array<std::vector<std::complex<double>>, 3>;
    std::vector<components_type> total_;
    std::vector<std::array<components_type, 7>> branches_;
  };

  /** Batched GO with device ray solving and coherent spectra.
   * Input tracks and the unchanged subdivision controller stay on the host.
   * Geometry and spectra run in the selected Kokkos execution space. Each
   * observer/frequency output owns its sum and visits tracks, children and branches in
   * the CPU order. Runtime acquisition is lazy; an EM runtime is borrowed when available
   * and must outlive this object. Use on the coordinating host thread.
   */
  class IGeometricalOpticsAccumulator {
  public:
    virtual ~IGeometricalOpticsAccumulator() = default;
    virtual void append(GeometricalOpticsTrack const&) = 0;
    virtual void flush() = 0;
    virtual GeometricalOpticsSpectra download() = 0;
    virtual void reset() = 0;
    virtual GeometricalOpticsStatistics const& statistics() const = 0;
    virtual std::string executionSpaceName() const = 0;
    /// The same device ray kernel, exposed for independent geometry validation.
    virtual std::vector<std::vector<PathDiagnostics>> paths(
        std::vector<vector_type> const& sources) = 0;
  };

  std::shared_ptr<IGeometricalOpticsAccumulator> make_geometrical_optics_accumulator(
      ExponentialOpticalParameters profile, std::vector<vector_type> observers,
      GeometricalOpticsConfig config, int numThreads = 0, int deviceId = -1);
} // namespace corsika::kokkos::radio::firn
