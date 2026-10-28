/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

namespace corsika {
  inline SimulationTiming::SimulationTiming(YAML::Node configuration)
      : configuration_(std::move(configuration)) {}
  inline void SimulationTiming::startOfLibrary(boost::filesystem::path const& directory) {
    summary_.open(directory);
    setInit(true);
  }
  inline void SimulationTiming::startOfShower(unsigned int const showerId) {
    auto shower = summary_.event(showerId);
    shower["closed"] = false;
    shower["status"] = "in_progress";
    activeShower_ = showerId;
    start_ = Clock::now();
    running_ = true;
  }
  inline void SimulationTiming::endOfShower(unsigned int const showerId) {
    auto shower = summary_.event(showerId);
    if (!running_ || showerId != activeShower_) {
      shower["closed"] = false;
      shower["status"] = "invalid_callback_order";
      running_ = false;
      summary_.flush();
      return;
    }
    shower["wall_time_ms"] =
        std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    shower["closed"] = true;
    shower["status"] = "closed";
    running_ = false;
    summary_.flush();
  }
  inline void SimulationTiming::endOfLibrary() {
    if (running_ && activeShower_ != InvalidShower) {
      auto shower = summary_.event(activeShower_);
      shower["closed"] = false;
      shower["status"] = "library_ended_during_shower";
      shower["wall_time_ms"] =
          std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
      running_ = false;
      summary_.flush();
    }
    setInit(false);
  }
  inline YAML::Node SimulationTiming::getConfig() const { return configuration_; }
  inline YAML::Node SimulationTiming::getSummary() const { return summary_.snapshot(); }
  inline void SimulationTiming::writeSummary(boost::filesystem::path const& path) const {
    summary_.writeSummary(path);
  }
} // namespace corsika
