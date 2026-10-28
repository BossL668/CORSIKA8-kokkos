/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/output/BaseOutput.hpp>
#include <corsika/output/ShowerSummary.hpp>
#include <boost/filesystem/path.hpp>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <limits>
#include <string>
#include <utility>
namespace corsika {
  /** Per-shower native wall timer; a closed output does not imply physics completion.
   * Register after physics writers in OutputManager's ordered name map to include
   * their end-of-shower flushes, as in the production simulation_timing entry.
   */
  class SimulationTiming final : public BaseOutput {
  public:
    explicit SimulationTiming(YAML::Node configuration);
    void startOfLibrary(boost::filesystem::path const& directory) final;
    void startOfShower(unsigned int const showerId) final;
    void endOfShower(unsigned int const showerId) final;
    void endOfLibrary() final;
    YAML::Node getConfig() const final;
    YAML::Node getSummary() const final;
    void writeSummary(boost::filesystem::path const& path) const final;

  private:
    using Clock = std::chrono::steady_clock;
    static constexpr unsigned int InvalidShower =
        std::numeric_limits<unsigned int>::max();
    YAML::Node configuration_{};
    mutable ShowerSummary summary_;
    Clock::time_point start_{};
    unsigned int activeShower_ = InvalidShower;
    bool running_{};
  };
} // namespace corsika
#include <corsika/detail/output/SimulationTiming.inl>
