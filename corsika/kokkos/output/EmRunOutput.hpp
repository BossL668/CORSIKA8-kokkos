/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/output/BaseOutput.hpp>
#include <corsika/output/ShowerSummary.hpp>
#include <corsika/kokkos/ScalarPhysicalConstants.hpp>
#include <boost/filesystem/path.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <utility>

namespace corsika::kokkos {
  /** Production completion/provenance ledger, including late router completion. */
  class EmRunOutput final : public BaseOutput {
  public:
    explicit EmRunOutput(YAML::Node configuration);

    void startOfLibrary(boost::filesystem::path const& directory) final;

    void startOfShower(unsigned int const shower_id) final;

    void recordComplete(unsigned int shower_id, YAML::Node metadata);

    void recordIncomplete(unsigned int shower_id, std::string const& reason);

    void recordIncomplete(unsigned int shower_id, std::string const& reason,
                          YAML::Node metadata);

    void endOfShower(unsigned int const shower_id) final;

    void endOfLibrary() final;

    YAML::Node getConfig() const final;

    YAML::Node getSummary() const final;

    void writeSummary(boost::filesystem::path const& path) const final;

  private:
    static constexpr unsigned int InvalidShower = static_cast<unsigned int>(-1);

    YAML::Node configuration_{};
    mutable ShowerSummary summary_;
    unsigned int activeShower_{InvalidShower};
    bool showerRecorded_{};
  };
} // namespace corsika::kokkos

#include <corsika/detail/kokkos/output/EmRunOutput.inl>
