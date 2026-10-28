/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "Check.hpp"
#include <corsika/output/SimulationTiming.hpp>
#include <corsika/output/OutputManager.hpp>
#include "ProductionSimulationTiming.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <thread>

void compare(YAML::Node const& candidate, YAML::Node const& reference) {
  CHECK(candidate.Type() == reference.Type());
  if (candidate.IsNull()) return;
  CHECK(candidate.IsMap());
  CHECK(candidate.size() == reference.size());
  for (auto const& entry : reference) {
    auto key = entry.first.as<std::string>();
    auto a = candidate[key], b = entry.second;
    CHECK(a.size() == b.size());
    for (auto const& field : b) {
      auto name = field.first.as<std::string>();
      CHECK(bool(a[name]));
      if (name == "wall_time_ms") {
        CHECK(std::isfinite(a[name].as<double>()));
        CHECK(a[name].as<double>() >= 0.);
        CHECK(std::isfinite(field.second.as<double>()));
        CHECK(field.second.as<double>() >= 0.);
      } else
        CHECK(a[name].as<std::string>() == field.second.as<std::string>());
    }
  }
}
struct DelayedOutput final : corsika::BaseOutput {
  void startOfLibrary(boost::filesystem::path const&) final { setInit(true); }
  void endOfShower(unsigned int const) final {
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
  }
  void endOfLibrary() final { setInit(false); }
  YAML::Node getConfig() const final { return YAML::Node(); }
};
template <class Timer>
YAML::Node managed(std::filesystem::path const& root) {
  Timer timer(YAML::Node(YAML::NodeType::Map));
  DelayedOutput physics;
  {
    corsika::OutputManager output(root.string(), 17, "host-timing-test", false);
    output.add("simulation_timing", timer);
    output.add("physics", physics);
    output.startOfLibrary();
    for (unsigned i = 0; i < 3; ++i) {
      output.startOfShower();
      output.endOfShower();
    }
    output.endOfLibrary();
  }
  auto summary = YAML::LoadFile((root / "simulation_timing/summary.yaml").string());
  CHECK(summary.size() == 3);
  for (auto const& entry : summary) {
    CHECK(entry.second["status"].as<std::string>() == "closed");
    CHECK(entry.second["wall_time_ms"].as<double>() >= 4.);
  }
  return summary;
}
int main() {
  try {
    TempDirectory directory;
    auto aDir = directory.path / "candidate", bDir = directory.path / "reference";
    std::filesystem::create_directory(aDir);
    std::filesystem::create_directory(bDir);
    YAML::Node config;
    config["em_backend"] = "proposal";
    config["clock"] = "steady_clock";
    corsika::SimulationTiming a(config);
    corsika::ProductionSimulationTiming b(YAML::Clone(config));
    CHECK(a.isInit() == b.isInit());
    CHECK(YAML::Dump(a.getConfig()) == YAML::Dump(b.getConfig()));
    compare(a.getSummary(), b.getSummary());
    a.startOfLibrary(aDir.string());
    b.startOfLibrary(bDir.string());
    CHECK(a.isInit());
    CHECK(b.isInit());
    for (unsigned id = 0; id < 12; ++id) {
      a.startOfShower(id);
      b.startOfShower(id);
      compare(a.getSummary(), b.getSummary());
      CHECK(a.getSummary()["shower_" + std::to_string(id)]["status"].as<std::string>() ==
            "in_progress");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      a.endOfShower(id);
      b.endOfShower(id);
      compare(a.getSummary(), b.getSummary());
    }
    a.startOfShower(12);
    b.startOfShower(12);
    a.endOfLibrary();
    b.endOfLibrary();
    CHECK(!a.isInit());
    CHECK(!b.isInit());
    compare(a.getSummary(), b.getSummary());
    CHECK(a.getSummary()["shower_12"]["status"].as<std::string>() ==
          "library_ended_during_shower");
    a.writeSummary((aDir / "summary.yaml").string());
    b.writeSummary((bDir / "summary.yaml").string());
    compare(YAML::LoadFile((aDir / "summary.yaml").string()),
            YAML::LoadFile((bDir / "summary.yaml").string()));
    CHECK(!std::filesystem::exists(aDir / "summary.yaml.tmp"));
    a.startOfLibrary(aDir.string());
    b.startOfLibrary(bDir.string());
    compare(a.getSummary(), b.getSummary());
    a.endOfShower(7);
    b.endOfShower(7);
    compare(a.getSummary(), b.getSummary());
    CHECK(a.getSummary()["shower_7"]["status"].as<std::string>() ==
          "invalid_callback_order");
    a.startOfShower(8);
    b.startOfShower(8);
    a.endOfShower(9);
    b.endOfShower(9);
    compare(a.getSummary(), b.getSummary());
    CHECK(a.getSummary()["shower_9"]["status"].as<std::string>() ==
          "invalid_callback_order");
    CHECK(outcome([&] { a.startOfShower(3); }) == outcome([&] { b.startOfShower(3); }));
    a.endOfLibrary();
    b.endOfLibrary();
    compare(a.getSummary(), b.getSummary());
    auto missing = (directory.path / "missing/subdirectory").string();
    CHECK(outcome([&] { a.startOfLibrary(missing); }) ==
          outcome([&] { b.startOfLibrary(missing); }));
    compare(managed<corsika::SimulationTiming>(directory.path / "managed-candidate"),
            managed<corsika::ProductionSimulationTiming>(directory.path /
                                                         "managed-reference"));
    std::cout << "simulation-timing CHECKS=" << checks << '\n';
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
