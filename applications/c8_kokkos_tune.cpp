/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/KokkosTuningCache.hpp>

#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
  template <class Function>
  double medianMilliseconds(Function&& function, int const repetitions) {
    using Clock = std::chrono::steady_clock;
    std::vector<double> times;
    times.reserve(repetitions);
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      auto const start = Clock::now();
      function();
      auto const stop = Clock::now();
      times.push_back(
          std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
  }

  bool validSha256(std::string const& value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char c) {
             return std::isxdigit(c) != 0;
           });
  }
}

int main(int argc, char** argv) {
  try {
    CLI::App app{
        "Tune one mutually exclusive CORSIKA 8 Kokkos execution backend"};
    std::string requested_backend;
    std::string table_hash;
    std::filesystem::path output;
    int threads{};
    int device{};
    int repetitions{5};
    std::size_t radio_tracks{4096};
    std::size_t radio_observers{81};
    app.add_option("--backend", requested_backend,
                   "Required compiled backend: openmp, cuda, hip, or sycl")
        ->required()
        ->check(CLI::IsMember({"openmp", "cuda", "hip", "sycl"}));
    app.add_option("--threads", threads,
                   "OpenMP thread count; GPU builds require zero or one")
        ->check(CLI::NonNegativeNumber);
    app.add_option("--device", device, "GPU device index")
        ->check(CLI::NonNegativeNumber);
    app.add_option("--proposal-table-hash", table_hash,
                   "64-digit proposal-native content hash from a dry run")
        ->required();
    app.add_option("--output", output, "Tuning cache to write")->required();
    app.add_option("--repetitions", repetitions,
                   "Timed repetitions after one warm-up")
        ->check(CLI::Range(3, 31));
    app.add_option("--radio-tracks", radio_tracks)
        ->check(CLI::PositiveNumber);
    app.add_option("--radio-observers", radio_observers)
        ->check(CLI::PositiveNumber);
    CLI11_PARSE(app, argc, argv);

    std::transform(table_hash.begin(), table_hash.end(), table_hash.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (!validSha256(table_hash))
      throw std::invalid_argument(
          "--proposal-table-hash must contain exactly 64 hexadecimal digits");

    corsika::accelerator::em::KokkosRuntimeConfig runtime_config;
    runtime_config.device = device;
    runtime_config.threads = threads;
    runtime_config.execution_backend = requested_backend;
    corsika::accelerator::em::KokkosRuntime runtime{runtime_config};
    auto const& info = runtime.info();
    if (requested_backend != info.backend)
      throw std::runtime_error(
          "requested backend '" + requested_backend +
          "' does not match the compiled Kokkos backend '" + info.backend +
          "'");
    if (info.gpu && threads > 1)
      throw std::invalid_argument(
          "GPU Kokkos tuning uses Serial host scheduling and rejects "
          "--threads > 1");

    std::vector<std::size_t> const batches =
        info.openmp ? std::vector<std::size_t>{256, 512, 1024, 2048}
                    : std::vector<std::size_t>{1024, 2048, 4096, 8192};
    std::vector<std::size_t> const chunks =
        info.openmp ? std::vector<std::size_t>{1, 4, 16, 64}
                    : std::vector<std::size_t>{1};
    double best_queue_score = std::numeric_limits<double>::infinity();
    std::size_t best_batch{};
    std::size_t best_chunk{};
    for (auto const batch : batches) {
      for (auto const chunk : chunks) {
        auto warm = runtime.runPrimitiveProbe(batch, chunk);
        auto queue = runtime.runQueueProbe(batch);
        if (!warm.scan_valid || !queue.stable_order ||
            !queue.roundtrip_exact)
          throw std::runtime_error(
              "Kokkos queue/scan probe changed its deterministic output");
        auto const expected_checksum = warm.checksum;
        auto const score = medianMilliseconds(
            [&] {
              auto const result = runtime.runPrimitiveProbe(batch, chunk);
              if (!result.scan_valid || result.checksum != expected_checksum)
                throw std::runtime_error(
                    "Kokkos scan output hash changed during tuning");
            },
            repetitions) /
            static_cast<double>(batch);
        std::cout << "queue batch=" << batch << " chunk=" << chunk
                  << " median_ms_per_particle=" << score << '\n';
        if (score < best_queue_score) {
          best_queue_score = score;
          best_batch = batch;
          best_chunk = chunk;
        }
      }
    }

    std::vector<std::size_t> const teams =
        info.openmp ? std::vector<std::size_t>{1}
                    : std::vector<std::size_t>{64, 128, 256};
    std::vector<std::size_t> const track_tiles =
        info.openmp ? std::vector<std::size_t>{16, 32, 64}
                    : std::vector<std::size_t>{4, 8, 16};
    std::vector<std::size_t> const observer_tiles =
        info.openmp ? std::vector<std::size_t>{4, 8, 16}
                    : std::vector<std::size_t>{16, 32, 64};
    double best_tiling_score = std::numeric_limits<double>::infinity();
    std::size_t best_team{};
    std::size_t best_track_tile{};
    std::size_t best_observer_tile{};
    for (auto const team : teams) {
      for (auto const track_tile : track_tiles) {
        for (auto const observer_tile : observer_tiles) {
          auto const warm = runtime.runTilingProbe(
              radio_tracks, radio_observers, team, track_tile,
              observer_tile);
          if (!warm.exact)
            throw std::runtime_error(
                "Kokkos tiling probe lost or duplicated work items");
          auto const expected_checksum = warm.checksum;
          auto const score = medianMilliseconds(
              [&] {
                auto const result = runtime.runTilingProbe(
                    radio_tracks, radio_observers, team, track_tile,
                    observer_tile);
                if (!result.exact || result.checksum != expected_checksum)
                  throw std::runtime_error(
                      "Kokkos tiling output hash changed during tuning");
              },
              repetitions) /
              static_cast<double>(radio_tracks * radio_observers);
          std::cout << "radio team=" << team
                    << " track_tile=" << track_tile
                    << " observer_tile=" << observer_tile
                    << " median_ms_per_pair=" << score << '\n';
          if (score < best_tiling_score) {
            best_tiling_score = score;
            best_team = team;
            best_track_tile = track_tile;
            best_observer_tile = observer_tile;
          }
        }
      }
    }

    corsika::accelerator::em::KokkosTuningKey const key{
        info.backend,
        info.device_name,
        info.architecture,
        info.driver_version,
        info.runtime_version,
        info.kokkos_version,
        info.compiler_version,
        info.project_revision,
        table_hash,
        info.device,
        info.host_threads};
    corsika::accelerator::em::KokkosTuningParameters const parameters{
        best_batch, best_chunk, best_team, best_track_tile,
        best_observer_tile, 1};
    corsika::accelerator::em::writeKokkosTuningCache(
        output, key, parameters);
    std::cout << "selected batch=" << best_batch
              << " chunk=" << best_chunk << " team=" << best_team
              << " track_tile=" << best_track_tile
              << " observer_tile=" << best_observer_tile
              << " output=" << output << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "c8_kokkos_tune: " << error.what() << '\n';
    return 2;
  }
}
