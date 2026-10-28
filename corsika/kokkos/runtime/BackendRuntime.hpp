/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <memory>
#include <string>
#include <corsika/kokkos/runtime/RuntimeConfig.hpp>
#include <corsika/kokkos/runtime/RuntimeInfo.hpp>
#include <corsika/kokkos/runtime/RuntimeProbeResult.hpp>

namespace corsika::kokkos {

  // Select a compiled backend without initializing Kokkos.
  std::string resolve_execution_backend(std::string const& requested);

  /**
   * Process-level Kokkos lifetime, with an explicit internal cooperative lease.
   *
   * Independent builds contain OpenMP or one GPU with a Serial host. The
   * experimental CUDA_OPENMP build normally selects one execution space;
   * ordinary CUDA mode still initializes OpenMP with one thread. An explicitly
   * cooperative owner may instead lend the process lifetime to CUDA/OpenMP
   * instances on the same coordinator thread. All instances and the final
   * lease must be destroyed on that thread, after their views and work finish.
   * This lifetime facility alone does not enable a cooperative air shower.
   */
  class BackendRuntime {
  public:
    explicit BackendRuntime(RuntimeConfig const& = {});
    ~BackendRuntime();

    BackendRuntime(BackendRuntime const&) = delete;
    BackendRuntime& operator=(BackendRuntime const&) = delete;
    BackendRuntime(BackendRuntime&&) noexcept;
    BackendRuntime& operator=(BackendRuntime&&) noexcept;

    RuntimeInfo const& getInfo() const noexcept;
    std::shared_ptr<RuntimeLease> shareCooperativeLifetime() const;
    PrimitiveProbeResult runPrimitiveProbe(std::size_t values,
                                           std::size_t chunkSize = 0) const;
    QueueProbeResult runQueueProbe(std::size_t particles) const;
    TilingProbeResult runTilingProbe(std::size_t tracks, std::size_t observers,
                                     std::size_t teamSize, std::size_t trackTileSize,
                                     std::size_t observerTileSize) const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

} // namespace corsika::kokkos
