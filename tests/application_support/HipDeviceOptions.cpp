/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "Check.hpp"
#include <corsika/framework/core/Logging.hpp>
#include <corsika/kokkos/application/DeviceCliOptions.hpp>
#include <CLI/CLI.hpp>

namespace application = corsika::kokkos::application;

int main() {
  try {
    CHECK(application::gpu_execution() == "hip");
    CHECK(application::gpu_visibility_variable() == "HIP_VISIBLE_DEVICES");
    char const* arguments[]{"c8_air_shower", "--device", "0,1,2,3,4,5,6,7"};
    auto devices = application::parse_devices(3, arguments);
    CHECK(devices.hasMultipleDevices());
    CHECK(devices.ids_.size() == 8);
    char const* invalidArguments[]{"c8_air_shower", "--device", "GPU-1234"};
    CHECK(outcome([&] { application::parse_devices(3, invalidArguments); }) != "success");
    std::string listing;
    for (int index = 0; index < 8; ++index)
      listing += std::to_string(index) + ", 0000:41:00.0\n";
    CHECK(application::resolve_hip_indices(devices.ids_, listing) == devices.ids_);
    std::vector<std::string> reversed{"7", "0", "4"};
    CHECK(application::resolve_hip_indices(reversed, listing) == reversed);
    for (auto const& invalid :
         std::vector<std::vector<std::string>>{{"8"}, {"-1"}, {"0", "0"}, {"GPU-1234"}})
      CHECK(outcome([&] { (void)application::resolve_hip_indices(invalid, listing); }) !=
            "success");
    CHECK(outcome([&] { (void)application::resolve_hip_indices({"0"}, "invalid\n"); }) !=
          "success");
    CLI::App app;
    application::BackendOptions options;
    application::add_device_cli_options(app, options);
    CHECK(app.help().find("HIP GPU indices") != std::string::npos);
    CHECK(app.help().find("NVIDIA") == std::string::npos);
    // Help and parsing do not initialize HIP or change device visibility.
    EnvironmentValue visibility("HIP_VISIBLE_DEVICES", "prior");
    char const* help[]{"c8_air_shower", "--help", "--device", "invalid"};
    CHECK(application::parse_devices(4, help).ids_.empty());
    CHECK(std::string(std::getenv("HIP_VISIBLE_DEVICES")) == "prior");
    CORSIKA_LOG_INFO("HIP device selection checks: {}", checks);
  } catch (std::exception const& error) {
    CORSIKA_LOG_ERROR("{}", error.what());
    return 1;
  }
}
