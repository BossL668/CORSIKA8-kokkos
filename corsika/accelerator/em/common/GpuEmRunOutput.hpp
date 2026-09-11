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
#include <corsika/output/ShowerSummary.hpp>
#include <corsika/accelerator/ScalarPhysicalConstants.hpp>

#include <cstdint>
#include <string>
#include <utility>

namespace corsika::gpu::em {

  /**
   * Small OutputManager participant carrying CUDA-EM provenance and per-shower
   * completion state.
   *
   * Physics writers own the large tabular products. This output intentionally
   * writes only YAML so a failed CUDA shower is still self-describing and can
   * never be mistaken for a complete production event.
   */
  class GpuEmRunOutput final : public BaseOutput {
  public:
    explicit GpuEmRunOutput(YAML::Node configuration)
        : configuration_(std::move(configuration)) {
      configuration_["scalar_transport_constants_version"] = accelerator::scalar_constants::Version;
      configuration_["magnetic_rigidity_GeV_per_T_m"] =
          accelerator::scalar_constants::MagneticRigidityGeVPerTeslaMeter;
    }

    void startOfLibrary(
        boost::filesystem::path const& directory) final {
      summary_.open(directory);
      setInit(true);
    }

    void startOfShower(
        unsigned int const shower_id) final {
      auto shower =
          summary_.event(shower_id);
      shower["complete"] = false;
      shower["status"] = "in_progress";
      active_shower_ = shower_id;
      shower_recorded_ = false;
    }

    void recordComplete(
        unsigned int shower_id,
        YAML::Node metadata) {
      auto shower =
          summary_.event(shower_id);
      shower["complete"] = true;
      shower["status"] = "complete";
      shower["statistics"] = std::move(metadata);
      active_shower_ = shower_id;
      shower_recorded_ = true;
      summary_.flush();
    }

    void recordIncomplete(
        unsigned int shower_id,
        std::string const& reason) {
      auto shower =
          summary_.event(shower_id);
      shower["complete"] = false;
      shower["status"] = "incomplete";
      shower["failure_reason"] = reason;
      active_shower_ = shower_id;
      shower_recorded_ = true;
      summary_.flush();
    }

    void recordIncomplete(
        unsigned int shower_id,
        std::string const& reason,
        YAML::Node metadata) {
      auto shower =
          summary_.event(shower_id);
      shower["complete"] = false;
      shower["status"] = "incomplete";
      shower["failure_reason"] = reason;
      shower["statistics"] = std::move(metadata);
      active_shower_ = shower_id;
      shower_recorded_ = true;
      summary_.flush();
    }

    void endOfShower(
        unsigned int const shower_id) final {
      // HybridCascade closes OutputManager before the application has access
      // to its final backend/router counters. Keep the in-progress marker
      // until recordComplete() is called immediately after run().
      active_shower_ = shower_id;
    }

    void endOfLibrary() final {
      if (active_shower_ != InvalidShower &&
          !shower_recorded_) {
        recordIncomplete(
            active_shower_,
            "CUDA EM library ended without a completion record");
      }
      setInit(false);
    }

    YAML::Node getConfig() const final {
      return configuration_;
    }

    YAML::Node getSummary() const final {
      return summary_.snapshot();
    }

    void writeSummary(boost::filesystem::path const& path) const final {
      if (active_shower_ != InvalidShower && !shower_recorded_) {
        auto shower = summary_.event(active_shower_);
        shower["complete"] = false;
        shower["status"] = "incomplete";
        shower["failure_reason"] = "CUDA EM library ended without a completion record";
      }
      summary_.writeSummary(path);
    }

  private:
    static constexpr unsigned int InvalidShower =
        static_cast<unsigned int>(-1);

    YAML::Node configuration_{};
    mutable ShowerSummary summary_;
    unsigned int active_shower_{InvalidShower};
    bool shower_recorded_{};
  };

} // namespace corsika::gpu::em
