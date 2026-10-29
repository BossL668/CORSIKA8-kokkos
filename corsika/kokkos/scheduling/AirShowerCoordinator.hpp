/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace corsika::kokkos::coordinator {
  namespace fs = std::filesystem;
  using json_type = nlohmann::json;

  /** Production static coordinator configuration; CLI defaults are preserved. */
  struct CoordinatorOptions {
    fs::path worker_, output_;
    std::vector<std::string> devices_, physics_;
    std::string emBackend_ = "kokkos-proposal";
    double egs4Stepfc_ = 1.;
    std::uint64_t seed_ = 1;
    unsigned events_ = 1;
    unsigned prefixWorkers_ = 0; // 0: unchanged serial prefix; >0: static CPU subshowers
    double prefixSplitEnergy_ = 0;
    double energy_ = 0, memoryFraction_ = .5, frontierEnergy_ = 0, timeout_ = 7200;
  };

  /// SHA-256 of the closed file; fails on an unreadable input.
  std::string digest(fs::path const& file);
  /// Production seed derivation, independent of worker completion order.
  std::uint32_t worker_seed(std::uint64_t seed, unsigned index);
  /** Assign each unstarted root once to a static shard, ordered by energy/history. */
  json_type partition(fs::path const& source, std::vector<fs::path> const& destinations);
  /// Join compatible closed V3 frontiers, rejecting duplicate histories.
  json_type join_frontiers(std::vector<fs::path> const& sources,
                           fs::path const& destination);
  /// Assign each immutable prefix-radio spool once to an explicit worker.
  json_type partition_radio_tracks(std::vector<fs::path> const& sources,
                                   std::vector<unsigned> const& workers);
  /// Check required outputs, finite arrays and recorded overflow counters.
  void validate_outputs(fs::path const& folder);
  /// For firn, require every assigned prefix segment to survive event initialization.
  void validate_radio_replay(fs::path const& folder, std::uint64_t expected);
  /** Merge closed outputs in input order; sum fields, concatenate ground records. */
  json_type merge(std::vector<fs::path> const& parts, fs::path const& destination);
  /// Publish closed JSON metadata without replacing an existing marker.
  void save(fs::path const& file, json_type const& value);
  /// Install the production SIGINT/SIGTERM cancellation handlers.
  void install_signal_handlers();
  /// Throw when cancellation has been requested.
  void check_interrupt();
  /// Detect coordinator selection before model, RNG or device initialization.
  bool requested(int argc, char** argv);
  /** Execute static prefix/worker stages; children use the configured native binary.
   * Creates a new output directory and complete/incomplete markers. Owns only its
   * child process groups. This function does not initialize a Kokkos context. */
  void run(CoordinatorOptions const& options);
  /** Parse the existing coordinator CLI and return its process exit status.
   * Called by the existing air-shower entry; this library adds no executable. */
  int main_entry(int argc, char** argv);
} // namespace corsika::kokkos::coordinator
