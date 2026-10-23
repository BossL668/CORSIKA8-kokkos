#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace c8::multigpu {
namespace fs = std::filesystem;
using Json = nlohmann::json;

struct Options {
  fs::path worker, output;
  std::vector<std::string> devices, physics;
  std::string emBackend = "kokkos-proposal";
  double egs4Stepfc = 1.;
  std::uint64_t seed = 1;
  unsigned events = 1;
  unsigned prefixWorkers = 0; // 0: unchanged serial prefix; >0: static CPU subshowers
  double prefixSplitEnergy = 0;
  double energy = 0, memoryFraction = .5, frontierEnergy = 0, timeout = 7200;
};

std::string digest(fs::path const& file);
std::uint32_t workerSeed(std::uint64_t seed, unsigned index);
Json partition(fs::path const& source, std::vector<fs::path> const& destinations);
Json joinFrontiers(std::vector<fs::path> const& sources, fs::path const& destination);
Json partitionRadioTracks(std::vector<fs::path> const& sources,
                          std::vector<unsigned> const& workers);
void validateOutputs(fs::path const& folder);
Json merge(std::vector<fs::path> const& parts, fs::path const& destination);
void save(fs::path const& file, Json const& value);
void installSignalHandlers();
void checkInterrupt();
bool requested(int argc, char** argv);
void run(Options const& options);
int mainEntry(int argc, char** argv);
} // namespace c8::multigpu
