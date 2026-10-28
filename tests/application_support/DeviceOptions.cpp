/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "Check.hpp"
#include <corsika/framework/core/Logging.hpp>
#include <spdlog/sinks/ostream_sink.h>
#include <corsika/kokkos/application/DeviceCliOptions.hpp>
#include <CLI/CLI.hpp>
#include "ProductionDeviceCliOptions.hpp"
#include "OptionDefaults.inc"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

namespace candidate = corsika::kokkos::application;
namespace original = production_application;
struct Arguments {
  std::vector<std::string> values;
  std::vector<char*> raw;
  explicit Arguments(std::vector<std::string> args)
      : values(std::move(args)) {
    values.insert(values.begin(), "c8_air_shower");
    for (auto& arg : values) raw.push_back(arg.data());
  }
  int size() const { return static_cast<int>(raw.size()); }
};
void parse_case(std::vector<std::string> const& values) {
  Arguments a(values), b(values);
  candidate::DeviceSelection selected;
  original::gpu_cli::DeviceSelection reference;
  auto result =
      outcome([&] { selected = candidate::parse_devices(a.size(), a.raw.data()); });
  auto expected = outcome(
      [&] { reference = original::gpu_cli::parseDevices(b.size(), b.raw.data()); });
  CHECK(result == expected);
  CHECK(selected.ids_ == reference.ids);
  CHECK(selected.hasMultipleDevices() == reference.multiple());
}
void parser() {
  std::vector<std::vector<std::string>> cases{
      {},
      {"--device", "0"},
      {"--device=2"},
      {"--devices=2"},
      {"--device", "0,1,2,3", "-E", "100"},
      {"--device", "0", "1", "2", "3", "-E", "100"},
      {"--devices=0,1"},
      {"--device=0", "1"},
      {"--device", "GPU-test-2"},
      {"--kokkos-device", "0"},
      {"--help", "--device", "bad"},
      {"--device=bad", "-h"},
      {"--", "--device", "0,1"},
      {"--", "-h", "--device"},
      {"--device", "0", "--", "--device", "1"},
      {"--device", " 0002 "},
      {"--device"},
      {"--device="},
      {"--device", "0,"},
      {"--device", ",0"},
      {"--device", "0,,1"},
      {"--device", "0,00"},
      {"--device", "-1"},
      {"--device", "0.5"},
      {"--device", "0", "--devices", "1"},
      {"--device", "0", "--kokkos-device", "0"},
      {"--kokkos-device=1", "--device=0"},
      {"--device", "GPU-"},
      {"--device", "GPU-x/y"},
      {"--device", "GPU-a_b"},
      {"--device", "2147483647"},
      {"--device", "2147483648"},
      {"--device", "18446744073709551616000"},
      {"--device", "GPU-a-b-C-2"}};
  std::vector<std::string> maximum{"--device"};
  for (unsigned i = 0; i < 255; ++i) maximum.push_back(std::to_string(i));
  cases.push_back(maximum);
  maximum.push_back("255");
  cases.push_back(maximum);
  for (auto const& input : cases) parse_case(input);
  for (auto const& option : {"--device", "--devices"})
    for (auto const& value : {"0", "00", "1", "-1", "GPU-X", "GPU-", "0,1", ",", " "})
      for (auto const& suffix :
           {"", "--help", "--kokkos-device=0", "--", "--devices=3"}) {
        std::vector<std::string> input{option, value};
        if (*suffix) input.emplace_back(suffix);
        parse_case(input);
      }
  CHECK(candidate::parse_devices(0, nullptr).ids_.empty());
  for (auto const& listing :
       {"0, GPU-test-0\n1, GPU-test-1\n", "garbage\n", " 2, GPU-test-2\r\n",
        "0,GPU-A\n0,GPU-B\n", "x,GPU-A\n0,bad\n", "0, GPU-a\n1, GPU-a"}) {
    for (auto const& ids : std::vector<std::vector<std::string>>{
             {}, {"0"}, {"2"}, {"GPU-test-2"}, {"0", "GPU-test-0"}, {"0", "1"}}) {
      std::vector<std::string> a, b;
      auto ca = outcome([&] { a = candidate::resolve_gpu_ids(ids, listing); });
      auto cb = outcome([&] { b = original::gpu_cli::resolveGpuIds(ids, listing); });
      CHECK(ca == cb);
      CHECK(a == b);
    }
  }
}
void cli_case(std::vector<std::string> const& values) {
  Arguments a(values), b(values);
  CLI::App app, reference;
  candidate::BackendOptions options;
  original::air_shower::GpuCliOptions originalOptions;
  double energy = 0., originalEnergy = 0.;
  candidate::add_device_cli_options(app, options);
  original::air_shower::addDeviceCliOptions(reference, originalOptions);
  app.add_option("-E,--energy", energy);
  reference.add_option("-E,--energy", originalEnergy);
  app.add_option("--em-backend", options.emBackend_);
  reference.add_option("--em-backend", originalOptions.em_backend);
  app.add_option("--radio-backend", options.radioBackend_);
  reference.add_option("--radio-backend", originalOptions.radio_backend);
  app.add_option("--kokkos-execution", options.kokkosExecution_);
  reference.add_option("--kokkos-execution", originalOptions.kokkos_execution);
  auto x = outcome([&] { app.parse(a.size(), a.raw.data()); });
  auto y = outcome([&] { reference.parse(b.size(), b.raw.data()); });
  CHECK(x == y);
  CHECK(energy == originalEnergy);
  check_options(options, originalOptions);
  if (x != "success") return;
  candidate::DeviceSelection selection;
  original::gpu_cli::DeviceSelection oldSelection;
  x = outcome([&] { selection = candidate::parse_devices(a.size(), a.raw.data()); });
  y = outcome(
      [&] { oldSelection = original::gpu_cli::parseDevices(b.size(), b.raw.data()); });
  CHECK(x == y);
  if (x != "success") return;
  std::ostringstream aError, bError;
  // Compare the same message payload across the project logger and the
  // frozen reference stderr interface, without weakening the text check.
  auto logger = corsika::get_logger("corsika", true);
  auto const previousSinks = logger->sinks();
  auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(aError);
  sink->set_pattern("%v");
  logger->sinks() = {sink};
  bool prepared = candidate::prepare_device_cli(app, options, selection);
  logger->sinks() = previousSinks;
  auto* previous = std::cerr.rdbuf(bError.rdbuf());
  bool oldPrepared =
      original::air_shower::prepareDeviceCli(reference, originalOptions, oldSelection);
  std::cerr.rdbuf(previous);
  CHECK(prepared == oldPrepared);
  CHECK(aError.str() == bError.str());
  check_options(options, originalOptions);
}
void cli() {
  check_defaults();
  for (auto const& values : std::vector<std::vector<std::string>>{
           {},
           {"--device", "2"},
           {"--devices=1"},
           {"--kokkos-device", "3"},
           {"--device", "0", "1"},
           {"--device", "0,1,2,3", "-E", "100"},
           {"--device", "0", "1", "2", "3", "-E", "100"},
           {"--device=0", "1", "-E", "100"},
           {"--kokkos-device", "-1"},
           {"--device"},
           {"--em-backend", "proposal", "--device", "2"},
           {"--em-backend", "kokkos-egs4", "--kokkos-execution", "openmp", "--device",
            "2"},
           {"--em-backend", "kokkos-proposal", "--radio-backend", "cpu", "--device",
            "2"}})
    cli_case(values);
  for (auto const& em : {"kokkos-proposal", "kokkos-egs4", "proposal", "invalid"})
    for (auto const& backend :
         {"cuda", "openmp", "cuda-openmp", "openmp-cuda", "hip", "sycl", "invalid"})
      cli_case({"--em-backend", em, "--kokkos-execution", backend, "--device", "2"});
  CLI::App help, oldHelp;
  candidate::BackendOptions options;
  original::air_shower::GpuCliOptions oldOptions;
  candidate::add_device_cli_options(help, options);
  original::air_shower::addDeviceCliOptions(oldHelp, oldOptions);
  CHECK(help.help() == oldHelp.help());
  CHECK(help.help().find("--device ") != std::string::npos);
  CHECK(help.help().find("--devices") == std::string::npos);
  CHECK(help.help().find("--kokkos-device") == std::string::npos);
#ifdef CORSIKA_KOKKOS_BACKEND_CUDA
  std::cout << "POLICY=cuda\n";
#elif defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
  std::cout << "POLICY=cuda-openmp\n";
#else
  std::cout << "POLICY=host\n";
#endif
}
void query_case(std::vector<std::string> const& ids, std::string mode) {
  EnvironmentValue fakeMode("C8_HOST_TEST_QUERY_MODE", mode);
  EnvironmentValue visibility("CUDA_VISIBLE_DEVICES", "previous-device");
  auto actual = outcome([&] { candidate::select_single_gpu({ids}); });
  std::string result = std::getenv("CUDA_VISIBLE_DEVICES");
  setenv("CUDA_VISIBLE_DEVICES", "previous-device", 1);
  auto expected = outcome([&] { original::gpu_cli::selectSingleGpu({ids}); });
  CHECK(actual == expected);
  CHECK(result == std::getenv("CUDA_VISIBLE_DEVICES"));
  if (actual != "success") CHECK(result == "previous-device");
}
int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--query-gpu=index,uuid") {
    std::string mode = std::getenv("C8_HOST_TEST_QUERY_MODE")
                           ? std::getenv("C8_HOST_TEST_QUERY_MODE")
                           : "valid";
    if (mode == "timeout") std::this_thread::sleep_for(std::chrono::seconds(30));
    if (mode == "failure") return 17;
    if (mode == "malformed") {
      std::cout << "not a gpu listing\n";
      return 0;
    }
    std::cout << "0, GPU-test-0\n1, GPU-test-1\n2, GPU-test-2\n3, GPU-test-3\n";
    return 0;
  }
  try {
    TempDirectory directory;
    std::filesystem::create_symlink(std::filesystem::canonical("/proc/self/exe"),
                                    directory.path / "nvidia-smi");
    EnvironmentValue path("PATH", directory.path.string() + ":" +
                                      (std::getenv("PATH") ? std::getenv("PATH") : ""));
    EnvironmentValue fakeMode("C8_HOST_TEST_QUERY_MODE", "valid");
    EnvironmentValue visibility("CUDA_VISIBLE_DEVICES", "prior");
    parser();
    cli();
    query_case({}, "valid");
    query_case({"0", "1"}, "valid");
    query_case({"2"}, "valid");
    query_case({"GPU-test-1"}, "valid");
    query_case({"8"}, "valid");
    query_case({"0"}, "malformed");
    query_case({"0"}, "failure");
#ifdef TEST_DEVICE_QUERY_TIMEOUT
    query_case({"0"}, "timeout");
#endif
    std::cout << "device-options CHECKS=" << checks << '\n';
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
