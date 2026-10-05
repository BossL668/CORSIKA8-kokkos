#include "../../applications/detail/air_shower_kokkos/DeviceCliOptions.hpp"
#include <filesystem>
#include <iostream>

using namespace corsika::applications;
static void check(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
static gpu_cli::DeviceSelection parse(std::vector<std::string> args, bool full = true) {
  args.insert(args.begin(), "c8_air_shower");
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  auto selection = gpu_cli::parseDevices(argv.size(), argv.data());
  if (full) {
    CLI::App app;
    air_shower::GpuCliOptions options;
    air_shower::addDeviceCliOptions(app, options);
    double energy = 0.;
    app.add_option("-E,--energy", energy);
    app.parse(argv.size(), argv.data());
  }
  return selection;
}
int main(int argc, char** argv) {
  // The query subprocess uses this fixture, never real GPU hardware.
  if (argc > 1 && std::string(argv[1]) == "--query-gpu=index,uuid") {
    std::cout << "0, GPU-test-0\n1, GPU-test-1\n2, GPU-test-2\n3, GPU-test-3\n";
    return 0;
  }
  try {
    check(parse({}).ids.empty(), "default path changed");
    check(parse({"--device", "0"}).ids == std::vector<std::string>{"0"}, "single device");
    check(!parse({"--devices=2"}).multiple(), "alias must use the direct single-device path");
    check(parse({"--device", "0,1,2,3", "-E", "100"}).ids.size() == 4, "CSV list");
    check(parse({"--device", "0", "1", "2", "3", "-E", "100"}).ids.size() == 4, "space list");
    check(parse({"--devices=0,1"}).multiple(), "compatibility alias");
    check(parse({"--device=0", "1"}).multiple(), "equals/space list");
    check(parse({"--device", "GPU-test-2"}).ids[0] == "GPU-test-2", "UUID");
    check(parse({"--kokkos-device", "0"}).ids.empty(), "internal worker ordinal changed");
    check(parse({"--help", "--device", "bad"}, false).ids.empty(), "help must not dispatch");
    check(parse({"--", "--device", "0,1"}, false).ids.empty(), "end of options");
    for (auto args : std::vector<std::vector<std::string>>{
        {"--device"}, {"--device="}, {"--device", "0,"}, {"--device", ",0"},
        {"--device", "0,,1"}, {"--device", "0,00"}, {"--device", "-1"},
        {"--device", "0.5"}, {"--device", "0", "--devices", "1"},
        {"--device", "0", "--kokkos-device", "0"},
        {"--kokkos-device=1", "--device=0"}}) {
      bool failed = false;
      try { parse(args); } catch (std::exception const&) { failed = true; }
      check(failed, "invalid selection accepted");
    }
    bool duplicate = false;
    try { gpu_cli::resolveGpuIds({"0", "GPU-test-0"}, "0, GPU-test-0\n"); }
    catch (std::exception const&) { duplicate = true; }
    check(duplicate, "physical aliases must not duplicate a GPU");
    char temporary[] = "/tmp/c8-device-cli-XXXXXX";
    auto folder = mkdtemp(temporary);
    check(folder, "temporary directory");
    namespace fs = std::filesystem;
    fs::create_symlink(fs::canonical("/proc/self/exe"), fs::path(folder) / "nvidia-smi");
    auto path = std::string(folder) + ":" + (std::getenv("PATH") ? std::getenv("PATH") : "");
    setenv("PATH", path.c_str(), 1);
    gpu_cli::selectSingleGpu(parse({"--device", "2"}));
    check(std::string(std::getenv("CUDA_VISIBLE_DEVICES")) == "GPU-test-2", "physical index was not resolved to UUID");
    gpu_cli::selectSingleGpu(parse({"--device", "GPU-test-1"}));
    check(std::string(std::getenv("CUDA_VISIBLE_DEVICES")) == "GPU-test-1", "single UUID");
    CLI::App help; air_shower::GpuCliOptions options;
    air_shower::addDeviceCliOptions(help, options);
    check(help.help().find("--device ") != std::string::npos, "canonical help option missing");
    check(help.help().find("--devices") == std::string::npos, "legacy alias clutters help");
    check(help.help().find("--kokkos-device") == std::string::npos, "legacy ordinal clutters help");
    std::cout << "GPU CLI parsing, routing and UUID selection passed\n";
  } catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
