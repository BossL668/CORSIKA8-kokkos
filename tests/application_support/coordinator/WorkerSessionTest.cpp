/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/application/cascade/WorkerSession.hpp>
#include <corsika/modules/Random.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <array>
#include <algorithm>
#include <vector>

namespace {
  using json_type = nlohmann::json;
  using session_type = corsika::kokkos::application::WorkerSession;
  corsika::rng_function_type modelRandom;

  void require(bool condition, char const* message) {
    if (!condition) throw std::runtime_error(message);
  }
  void set_model_random(corsika::rng_function_type value) {
    modelRandom = std::move(value);
  }

  struct Backend {
    unsigned assignments_ = 0;
    void assignWorkerInput(std::string frontier, unsigned id,
                           std::vector<std::string> radio) {
      require(frontier == "frozen-input" && id == 19 && radio.size() == 1,
              "Request input changed");
      ++assignments_;
    }
  };
  struct Application {
    std::uint64_t seed_ = 0;
    void setWorkerSeed(std::uint64_t value) { seed_ = value; }
  };
  struct Output {
    unsigned opens_ = 0;
    unsigned closes_ = 0;
    unsigned summaries_ = 0;
    void endOfLibrary() { ++closes_; }
    void flushLibrarySummary() { ++summaries_; }
    void reopenLibrary(std::string const& folder, long) {
      require(std::filesystem::path(folder).filename() == "shower", "Wrong library path");
      ++opens_;
    }
  };
  struct Counter {
    unsigned events_ = 1;
    void resetHistogram() { events_ = 0; }
  };
  struct ParticleData {
    std::array<bool, 3> decay_{{false, true, false}};
    int nextId(int id) const { return id < 2 ? id + 1 : 0; }
    bool mayDecay(int id) const { return decay_.at(id); }
    void mayDecay(int id, bool value) { decay_.at(id) = value; }
  };
  struct Decay {
    ParticleData particleData;
  };

  void channel(std::filesystem::path const& path, char const* requests) {
    std::filesystem::create_directory(path);
    setenv("C8_WORKER_SESSION", path.c_str(), 1);
    setenv("C8_WORKER_EVENTS", requests, 1);
    setenv("C8_WORKER_TIMEOUT", "0.03", 1);
  }

  void check_continuation(std::filesystem::path const& root) {
    channel(root, "4");
    auto& manager = corsika::RNGManager<>::getInstance();
    manager.setSeed(17);
    manager.registerRandomStream("resident-test");
    auto& stream = manager.getRandomStream("resident-test");
    stream();
    corsika::connect_random_stream(stream, set_model_random);
    auto reference = stream;
    auto copiedReference = stream;
    std::uniform_real_distribution<double> distribution;
    double initial[5];
    modelRandom(initial, 5);
    for (auto value : initial)
      require(value == distribution(copiedReference), "Model initialization differs");
    Decay decay;
    Counter high, low;
    session_type session;
    session.ready(decay, high, low);
    require(session.isActive(), "Resident channel inactive");
    Backend backend;
    Application application;
    Output output;
    long seed = 17;
    long const otherSeed = (1L << 35) + 23;
    for (unsigned request = 0; request != 4; ++request) {
      auto const folder = root / std::to_string(request);
      std::filesystem::create_directory(folder);
      std::ofstream file(root / ("request_" + std::to_string(request) + ".json"));
      file << json_type{{"request", request},
                        {"skip", request == 1},
                        {"seed", request == 2 ? otherSeed : 17},
                        {"folder", folder.string()},
                        {"frontier", "frozen-input"},
                        {"worker_id", 19},
                        {"radio_inputs", {"radio-input"}}};
    }
    std::vector<std::uint64_t> firstWords;
    for (auto expected : {17L, otherSeed, 17L}) {
      require(session.next(backend, application, output, seed), "Request lost");
      require(
          seed == expected && application.seed_ == static_cast<std::uint64_t>(expected),
          "Device seed changed");
      require(manager.getSeed() == 17, "Native CPU stream was reseeded");
      auto const first = stream();
      firstWords.push_back(first);
      require(first == reference(), "CPU stream rewound");
      for (unsigned draw = 0; draw != 10; ++draw)
        require(stream() == reference(), "CPU sequence differs");
      double values[10];
      modelRandom(values, 10);
      for (auto value : values)
        require(value == distribution(copiedReference), "Copied model stream rewound");
      require(high.events_ == 0 && low.events_ == 0, "Histogram not reset");
      require(decay.particleData.mayDecay(1) && !decay.particleData.mayDecay(2),
              "Decay flags not restored");
      high.events_ = low.events_ = 4;
      decay.particleData.mayDecay(1, false);
      decay.particleData.mayDecay(2, true);
    }
    require(!session.next(backend, application, output, seed), "Extra request");
    session.close(output);
    require(output.opens_ == 3 && output.closes_ == 4 && output.summaries_ == 3,
            "Output lifecycle differs");
    require(backend.assignments_ == 3 && firstWords.front() != firstWords.back(),
            "Request was replayed");
    for (unsigned request = 0; request != 4; ++request) {
      std::ifstream file(root / ("done_" + std::to_string(request) + ".json"));
      auto const done = json_type::parse(file);
      require(done["pid"] == getpid(), "Worker changed");
      require(done["skipped"].get<bool>() == (request == 1),
              "Wrong skip acknowledgement");
      if (request != 1)
        require(done["cpu_rng_policy"] == "continue-worker-streams", "Wrong RNG policy");
    }
    std::ifstream ready(root / "READY.json");
    require(json_type::parse(ready)["gpu_initialization_policy"] ==
                "first-request-then-reuse",
            "Early GPU initialization enabled");
  }

  void check_failure(std::filesystem::path const& root, bool malformed) {
    channel(root, "1");
    session_type session;
    Backend backend;
    Application application;
    Output output;
    long seed = 17;
    if (malformed) {
      std::ofstream file(root / "request_0.json");
      file << json_type{{"request", 99}, {"skip", true}};
    }
    bool failed = false;
    try {
      session.next(backend, application, output, seed);
    } catch (std::runtime_error const&) { failed = true; }
    require(failed, "Malformed or missing request was accepted");
    require(output.opens_ == 0 && backend.assignments_ == 0,
            "Failed request reached backend");
  }
} // namespace

int main() {
  auto const originalDirectory = std::filesystem::current_path();
  std::array<char, 40> pattern{};
  std::string const prefix = "/tmp/c8-worker-session-XXXXXX";
  std::copy(prefix.begin(), prefix.end(), pattern.begin());
  auto const directory = mkdtemp(pattern.data());
  if (!directory) return 1;
  std::filesystem::path const root(directory);
  try {
    unsetenv("C8_WORKER_SESSION");
    session_type direct;
    require(!direct.isActive(), "Direct execution became resident");
    check_continuation(root / "sequence");
    check_failure(root / "invalid", true);
    check_failure(root / "timeout", false);
    std::filesystem::current_path(originalDirectory);
    std::filesystem::remove_all(root);
    CORSIKA_LOG_INFO("Resident worker lifecycle and native stream continuation passed");
    return 0;
  } catch (std::exception const& error) {
    std::filesystem::current_path(originalDirectory);
    std::filesystem::remove_all(root);
    CORSIKA_LOG_ERROR("{}", error.what());
    return 1;
  }
}
