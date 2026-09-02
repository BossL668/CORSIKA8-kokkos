/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/KokkosTuningCache.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace corsika::accelerator::em;
  auto const path = std::filesystem::temp_directory_path() /
                    "c8-kokkos-tuning-cache-test-v1.txt";
  std::filesystem::remove(path);
  KokkosTuningKey const key{
      "openmp", "test-device", "host", "not-applicable", "openmp",
      "4.7.3", "test-compiler", "test-revision",
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      0, 8};
  KokkosTuningParameters const parameters{2048, 16, 1, 32, 8, 1};
  writeKokkosTuningCache(path, key, parameters);
  auto const exact = loadKokkosTuningCache(path, key, true);
  if (!exact.matched || exact.content_hash.size() != 64 ||
      exact.parameters.batch_size != parameters.batch_size ||
      exact.parameters.chunk_size != parameters.chunk_size ||
      exact.parameters.team_size != parameters.team_size ||
      exact.parameters.track_tile_size != parameters.track_tile_size ||
      exact.parameters.observer_tile_size !=
          parameters.observer_tile_size ||
      exact.parameters.device_queues != parameters.device_queues) {
    throw std::runtime_error("exact Kokkos tuning-cache roundtrip failed");
  }
  auto mismatch_key = key;
  mismatch_key.threads = 4;
  if (loadKokkosTuningCache(path, mismatch_key, false).matched)
    throw std::runtime_error("mismatched Kokkos cache was accepted");
  bool rejected = false;
  try {
    static_cast<void>(loadKokkosTuningCache(path, mismatch_key, true));
  } catch (std::runtime_error const&) {
    rejected = true;
  }
  std::filesystem::remove(path);
  if (!rejected)
    throw std::runtime_error("required mismatched Kokkos cache was accepted");
  std::cout << "Kokkos tuning-cache gates passed\n";
}
