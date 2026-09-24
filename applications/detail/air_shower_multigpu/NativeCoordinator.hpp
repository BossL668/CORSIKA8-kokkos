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
  std::uint64_t seed = 1;
  unsigned events = 1;
  double energy = 0, memoryFraction = .5, frontierEnergy = 0, timeout = 7200;
};

std::string digest(fs::path const& file);
std::uint32_t workerSeed(std::uint64_t seed, unsigned index);
Json partition(fs::path const& source, std::vector<fs::path> const& destinations);
void validateOutputs(fs::path const& folder);
Json merge(std::vector<fs::path> const& parts, fs::path const& destination);
void save(fs::path const& file, Json const& value);
void installSignalHandlers();
void checkInterrupt();
bool requested(int argc, char** argv);
void run(Options const& options);
int mainEntry(int argc, char** argv);
} // namespace c8::multigpu
