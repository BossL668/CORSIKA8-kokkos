/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include <corsika/accelerator/em/AcceleratorKind.hpp>

namespace corsika::accelerator::em {

  struct KokkosRuntimeConfig {
    int device{0};
    int threads{0};
    std::filesystem::path tuning_cache;
    bool require_tuning{};
  };

  struct KokkosRuntimeInfo {
    AcceleratorKind kind{AcceleratorKind::KokkosOpenMP};
    std::string backend;
    std::string kokkos_version;
    std::string device_name;
    std::string architecture;
    std::string driver_version;
    std::string runtime_version;
    std::string compiler_version;
    std::string project_revision;
    int device{};
    int concurrency{};
    int host_threads{1};
    bool gpu{};
    bool openmp{};
  };

  struct KokkosPrimitiveProbeResult {
    std::size_t values{};
    std::uint64_t selected{};
    std::uint64_t checksum{};
    bool scan_valid{};
  };

  struct KokkosQueueProbeResult {
    std::size_t input_particles{};
    std::size_t retained_particles{};
    std::uint64_t secondary_slots{};
    bool stable_order{};
    bool roundtrip_exact{};
  };

  struct KokkosTilingProbeResult {
    std::size_t tracks{};
    std::size_t observers{};
    std::uint64_t checksum{};
    bool exact{};
  };

  /**
   * Process-level Kokkos lifetime for a single, compile-time-selected backend.
   *
   * A CORSIKA build contains either OpenMP or one GPU execution space with a
   * Serial host.  It never initializes OpenMP and a GPU for the same shower.
   */
  class KokkosRuntime {
  public:
    explicit KokkosRuntime(KokkosRuntimeConfig const& = {});
    ~KokkosRuntime();

    KokkosRuntime(KokkosRuntime const&) = delete;
    KokkosRuntime& operator=(KokkosRuntime const&) = delete;
    KokkosRuntime(KokkosRuntime&&) noexcept;
    KokkosRuntime& operator=(KokkosRuntime&&) noexcept;

    KokkosRuntimeInfo const& info() const noexcept;
    KokkosPrimitiveProbeResult runPrimitiveProbe(
        std::size_t values, std::size_t chunk_size = 0) const;
    KokkosQueueProbeResult runQueueProbe(std::size_t particles) const;
    KokkosTilingProbeResult runTilingProbe(
        std::size_t tracks, std::size_t observers, std::size_t team_size,
        std::size_t track_tile_size,
        std::size_t observer_tile_size) const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

} // namespace corsika::accelerator::em
