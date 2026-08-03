/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/gpu/em/tables/MediumConfig.hpp>
#include <corsika/gpu/em/tables/ProposalMedium.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

  using namespace corsika::gpu::em::tables;

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  template <typename Function>
  void requireThrows(Function function, std::string const& message) {
    ++checks;
    try {
      function();
    } catch (std::exception const&) {
      return;
    }
    throw std::runtime_error(message);
  }

  std::filesystem::path temporaryDirectory() {
    auto const stamp =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();
    auto path = std::filesystem::temp_directory_path() /
                ("c8_gpu_medium_config_" + std::to_string(stamp));
    std::filesystem::create_directories(path);
    return path;
  }

  void testCanonicalDryAir() {
    auto const built_in = standardDryAirMediumConfig();
    auto const source =
        std::filesystem::path(C8_SOURCE_DIR) / "configs" / "media" /
        "air_dry_1_atm.yaml";
    auto const loaded = loadMediumConfig(source);
    require(canonicalMediumYaml(built_in) == canonicalMediumYaml(loaded),
            "repository dry-air YAML differs from the legacy contract");
    require(
        mediumConfigHashHex(loaded) ==
            "054323b00f3ae72fab27ca4a46ee510c5578e65f63fc0eaff41cf93251a857fa",
        "canonical dry-air SHA-256 changed unexpectedly");

    auto reordered = loaded;
    std::reverse(reordered.components.begin(), reordered.components.end());
    require(mediumConfigHash(reordered) == mediumConfigHash(loaded),
            "component order changed the normalized medium hash");

    auto changed = loaded;
    changed.components[0].number_fraction += 1.e-5;
    changed.components[1].number_fraction -= 1.e-5;
    require(mediumConfigHash(changed) != mediumConfigHash(loaded),
            "physical composition change did not change the medium hash");
  }

  void testProposalConstruction() {
    auto const config = standardDryAirMediumConfig();
    auto const medium = makeProposalMedium(config);
    require(medium.GetName() == "air_dry_1_atm",
            "PROPOSAL medium name differs");
    require(medium.GetHash() != 0,
            "PROPOSAL medium hash is empty");
    require(medium.GetComponents().size() == 3,
            "PROPOSAL dry-air component count differs");

    auto const components =
        makeRateTableMediumComponents(config, medium);
    require(components.size() == 3,
            "rate-table component count differs");
    require(components[0].corsika_pid == 1000070140,
            "normalized nitrogen PID differs");
    require(components[1].corsika_pid == 1000080160,
            "normalized oxygen PID differs");
    require(components[2].corsika_pid == 1000180400,
            "normalized argon PID differs");
  }

  void testRoundTripAndRejection() {
    auto const directory = temporaryDirectory();
    auto const canonical = directory / "medium.yaml";
    auto const invalid = directory / "invalid.yaml";
    try {
      auto const source = standardDryAirMediumConfig();
      writeCanonicalMediumYaml(canonical, source);
      require(mediumConfigHash(loadMediumConfig(canonical)) ==
                  mediumConfigHash(source),
              "canonical YAML round trip changed the medium hash");

      {
        std::ofstream output(invalid);
        output << canonicalMediumYaml(source)
               << "unknown_physics_switch: true\n";
      }
      requireThrows([&] { (void)loadMediumConfig(invalid); },
                    "unknown medium YAML key was accepted");

      auto invalid_fraction = source;
      invalid_fraction.components.front().number_fraction += 0.1;
      requireThrows(
          [&] { (void)normalizeMediumConfig(invalid_fraction); },
          "invalid number-fraction sum was accepted");

      auto invalid_pid = source;
      invalid_pid.components.front().corsika_pid = -1;
      requireThrows(
          [&] { (void)normalizeMediumConfig(invalid_pid); },
          "negative medium-component PID was accepted");
    } catch (...) {
      std::filesystem::remove_all(directory);
      throw;
    }
    std::filesystem::remove_all(directory);
  }

} // namespace

int main() {
  try {
    testCanonicalDryAir();
    testProposalConstruction();
    testRoundTripAndRejection();
  } catch (std::exception const& error) {
    std::cerr << "GPU medium-config validation failed after " << checks
              << " checks: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "GPU medium-config validation passed: " << checks
            << " checks\n";
  return EXIT_SUCCESS;
}
