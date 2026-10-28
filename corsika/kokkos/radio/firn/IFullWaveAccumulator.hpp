/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/FullWaveChildSegment.hpp>
#include <corsika/kokkos/radio/firn/FullWaveDeviceConfig.hpp>
#include <corsika/kokkos/radio/firn/FullWaveDeviceStatistics.hpp>
#include <corsika/kokkos/radio/firn/FullWaveSpectra.hpp>
#include <filesystem>
#include <memory>
#include <string>

namespace corsika::kokkos::radio::firn {

  /** Ordered schema-2 full-wave projection behind a host-only interface.
   * Kokkos owns device buffers and execution. Each observer/frequency pair owns
   * its spectrum bins and traverses input segments in order; no floating-point
   * atomic sum or implicit out-of-table propagation model is used.
   * Construction validates host table data; the first nonempty flush acquires
   * the runtime and allocates device storage. An existing EM runtime is borrowed
   * and must outlive this accumulator. Otherwise radio owns initialization.
   * Construct, use and destroy the accumulator on the coordinating host thread.
   */
  class IFullWaveAccumulator {
  public:
    virtual ~IFullWaveAccumulator() = default;
    virtual void append(FullWaveChildSegment const&) = 0;
    virtual void flush() = 0;
    virtual FullWaveSpectra download() = 0;
    virtual void reset() = 0;
    virtual double sourceRefractiveIndex(double zM) const = 0;
    virtual bool covers(vector_type const&, vector_type const&) const = 0;
    virtual std::array<double, 2> opticalPathMeters(vector_type const&,
                                                    vector_type const&) const = 0;
    virtual FullWaveDeviceStatistics const& statistics() const = 0;
    virtual std::string executionSpaceName() const = 0;
  };

  /// Create the build-selected Kokkos implementation; an OFF build rejects it.
  std::shared_ptr<IFullWaveAccumulator> make_full_wave_accumulator(
      std::filesystem::path const& manifest, std::string const& profileFingerprint,
      std::vector<vector_type> observers, FullWaveDeviceConfig config, int numThreads = 0,
      int deviceId = -1);

} // namespace corsika::kokkos::radio::firn
