/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <boost/filesystem/path.hpp>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <limits>
#include <string>
#include <utility>

#include <corsika/output/BaseOutput.hpp>

namespace corsika {

  /**
   * Backend-neutral per-shower wall timer.
   *
   * OutputManager invokes this participant from the same start/end callbacks
   * for scalar Cascade and HybridCascade. Use an OutputManager key that sorts
   * after the physics writer keys so the end timestamp includes their
   * end-of-shower flushes.
   *
   * "closed" only means that OutputManager paired the callbacks. Accelerator
   * physics completeness remains the responsibility of the backend-specific
   * status output (for example GpuEmRunOutput).
   */
  class SimulationTiming final : public BaseOutput {
  public:
    explicit SimulationTiming(YAML::Node configuration)
        : configuration_(std::move(configuration)) {}

    void startOfLibrary(
        boost::filesystem::path const&) final {
      setInit(true);
    }

    void startOfShower(
        unsigned int const shower_id) final {
      auto shower =
          summary_["shower_" +
                   std::to_string(shower_id)];
      shower["closed"] = false;
      shower["status"] = "in_progress";
      active_shower_ = shower_id;
      start_ = Clock::now();
      running_ = true;
    }

    void endOfShower(
        unsigned int const shower_id) final {
      auto shower =
          summary_["shower_" +
                   std::to_string(shower_id)];
      if (!running_ || shower_id != active_shower_) {
        shower["closed"] = false;
        shower["status"] = "invalid_callback_order";
        running_ = false;
        return;
      }
      shower["wall_time_ms"] =
          std::chrono::duration<double, std::milli>(
              Clock::now() - start_)
              .count();
      shower["closed"] = true;
      shower["status"] = "closed";
      running_ = false;
    }

    void endOfLibrary() final {
      if (running_ &&
          active_shower_ != InvalidShower) {
        auto shower =
            summary_["shower_" +
                     std::to_string(active_shower_)];
        shower["closed"] = false;
        shower["status"] =
            "library_ended_during_shower";
        shower["wall_time_ms"] =
            std::chrono::duration<double, std::milli>(
                Clock::now() - start_)
                .count();
        running_ = false;
      }
      setInit(false);
    }

    YAML::Node getConfig() const final {
      return configuration_;
    }

    YAML::Node getSummary() const final {
      return summary_;
    }

  private:
    using Clock = std::chrono::steady_clock;
    static constexpr unsigned int InvalidShower =
        std::numeric_limits<unsigned int>::max();

    YAML::Node configuration_{};
    YAML::Node summary_{};
    Clock::time_point start_{};
    unsigned int active_shower_{InvalidShower};
    bool running_{};
  };

} // namespace corsika
