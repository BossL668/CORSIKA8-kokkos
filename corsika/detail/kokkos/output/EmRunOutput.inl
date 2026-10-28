/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  inline EmRunOutput::EmRunOutput(YAML::Node configuration)
      : configuration_(std::move(configuration)) {
    configuration_["scalar_transport_constants_version"] = scalar_constants::Version;
    configuration_["magnetic_rigidity_GeV_per_T_m"] =
        scalar_constants::MagneticRigidityGeVPerTeslaMeter;
  }

  inline void EmRunOutput::startOfLibrary(boost::filesystem::path const& directory) {
    summary_.open(directory);
    setInit(true);
  }

  inline void EmRunOutput::startOfShower(unsigned int const shower_id) {
    auto shower = summary_.event(shower_id);
    shower["complete"] = false;
    shower["status"] = "in_progress";
    activeShower_ = shower_id;
    showerRecorded_ = false;
  }

  inline void EmRunOutput::recordComplete(unsigned int shower_id, YAML::Node metadata) {
    auto shower = summary_.event(shower_id);
    shower["complete"] = true;
    shower["status"] = "complete";
    shower["statistics"] = std::move(metadata);
    activeShower_ = shower_id;
    showerRecorded_ = true;
    summary_.flush();
  }

  inline void EmRunOutput::recordIncomplete(unsigned int shower_id,
                                            std::string const& reason) {
    auto shower = summary_.event(shower_id);
    shower["complete"] = false;
    shower["status"] = "incomplete";
    shower["failure_reason"] = reason;
    activeShower_ = shower_id;
    showerRecorded_ = true;
    summary_.flush();
  }

  inline void EmRunOutput::recordIncomplete(unsigned int shower_id,
                                            std::string const& reason,
                                            YAML::Node metadata) {
    auto shower = summary_.event(shower_id);
    shower["complete"] = false;
    shower["status"] = "incomplete";
    shower["failure_reason"] = reason;
    shower["statistics"] = std::move(metadata);
    activeShower_ = shower_id;
    showerRecorded_ = true;
    summary_.flush();
  }

  inline void EmRunOutput::endOfShower(unsigned int const shower_id) {
    // HybridCascade closes OutputManager before the application has access
    // to its final backend/router counters. Keep the in-progress marker
    // until recordComplete() is called immediately after run().
    activeShower_ = shower_id;
  }

  inline void EmRunOutput::endOfLibrary() {
    if (activeShower_ != InvalidShower && !showerRecorded_) {
      recordIncomplete(activeShower_,
                       "CUDA EM library ended without a completion record");
    }
    setInit(false);
  }

  inline YAML::Node EmRunOutput::getConfig() const { return configuration_; }

  inline YAML::Node EmRunOutput::getSummary() const { return summary_.snapshot(); }

  inline void EmRunOutput::writeSummary(boost::filesystem::path const& path) const {
    if (activeShower_ != InvalidShower && !showerRecorded_) {
      auto shower = summary_.event(activeShower_);
      shower["complete"] = false;
      shower["status"] = "incomplete";
      shower["failure_reason"] = "CUDA EM library ended without a completion record";
    }
    summary_.writeSummary(path);
  }
} // namespace corsika::kokkos
