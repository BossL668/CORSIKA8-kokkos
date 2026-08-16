/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <boost/filesystem/path.hpp>
#include <yaml-cpp/yaml.h>

#include <corsika/output/BaseOutput.hpp>

#include <cstdint>
#include <string>
#include <utility>

namespace corsika {

  /** Lightweight per-shower YAML output for scalar whole-event ledgers. */
  class EnergyLedgerRunOutput final : public BaseOutput {
  public:
    explicit EnergyLedgerRunOutput(YAML::Node configuration)
        : configuration_(std::move(configuration)) {}

    void startOfLibrary(boost::filesystem::path const&) final {
      setInit(true);
    }

    void startOfShower(unsigned int const shower_id) final {
      auto shower = summary_["shower_" + std::to_string(shower_id)];
      shower["complete"] = false;
      shower["status"] = "in_progress";
      active_shower_ = shower_id;
      shower_recorded_ = false;
    }

    void recordComplete(unsigned int shower_id, YAML::Node metadata) {
      auto shower = summary_["shower_" + std::to_string(shower_id)];
      shower["complete"] = true;
      shower["status"] = "complete";
      shower["statistics"] = std::move(metadata);
      active_shower_ = shower_id;
      shower_recorded_ = true;
    }

    void endOfShower(unsigned int const shower_id) final {
      active_shower_ = shower_id;
    }

    void endOfLibrary() final {
      if (active_shower_ != InvalidShower && !shower_recorded_) {
        auto shower = summary_["shower_" + std::to_string(active_shower_)];
        shower["complete"] = false;
        shower["status"] = "incomplete";
        shower["failure_reason"] =
            "scalar library ended without an energy-ledger record";
      }
      setInit(false);
    }

    YAML::Node getConfig() const final { return configuration_; }
    YAML::Node getSummary() const final { return summary_; }

  private:
    static constexpr unsigned int InvalidShower =
        static_cast<unsigned int>(-1);
    YAML::Node configuration_{};
    YAML::Node summary_{};
    unsigned int active_shower_{InvalidShower};
    bool shower_recorded_{};
  };

} // namespace corsika
