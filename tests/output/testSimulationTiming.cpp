/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <catch2/catch_all.hpp>

#include <chrono>
#include <thread>

#include <corsika/output/SimulationTiming.hpp>

using namespace corsika;

TEST_CASE("SimulationTiming") {
  YAML::Node configuration;
  configuration["em_backend"] = "proposal";
  SimulationTiming timing{configuration};

  timing.startOfLibrary(".");
  CHECK(timing.isInit());
  CHECK(timing.getConfig()["em_backend"].as<std::string>() ==
        "proposal");

  timing.startOfShower(0);
  auto in_progress = timing.getSummary()["shower_0"];
  CHECK_FALSE(in_progress["closed"].as<bool>());
  CHECK(in_progress["status"].as<std::string>() ==
        "in_progress");
  std::this_thread::sleep_for(
      std::chrono::milliseconds(1));
  timing.endOfShower(0);

  auto closed = timing.getSummary()["shower_0"];
  CHECK(closed["closed"].as<bool>());
  CHECK(closed["status"].as<std::string>() == "closed");
  CHECK(closed["wall_time_ms"].as<double>() > 0.);

  timing.startOfShower(1);
  timing.endOfLibrary();
  CHECK_FALSE(timing.isInit());
  auto incomplete = timing.getSummary()["shower_1"];
  CHECK_FALSE(incomplete["closed"].as<bool>());
  CHECK(incomplete["status"].as<std::string>() ==
        "library_ended_during_shower");
  CHECK(incomplete["wall_time_ms"].as<double>() >= 0.);

  SimulationTiming invalid_order{configuration};
  invalid_order.startOfLibrary(".");
  invalid_order.endOfShower(7);
  auto invalid =
      invalid_order.getSummary()["shower_7"];
  CHECK_FALSE(invalid["closed"].as<bool>());
  CHECK(invalid["status"].as<std::string>() ==
        "invalid_callback_order");
}
