/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <yaml-cpp/yaml.h>
#include <corsika/output/ShowerSummary.hpp>

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/modules/ObservationPlane.hpp>

namespace corsika {

  template <typename TTracking, typename TOutput>
  class PrimaryWriter : public BaseOutput {

  public:
    PrimaryWriter(ObservationPlane<TTracking, TOutput> const& obsPlane);
    ~PrimaryWriter();

    void recordPrimary(
        std::tuple<Code, HEPEnergyType, DirectionVector, Point, TimeType> const&);

    /**
     * Called at the start of each library.
     */
    void startOfLibrary(boost::filesystem::path const& directory) final override;

    /**
     * Called at the end of each shower.
     */
    void endOfShower(unsigned int const showerId) final override;

    /**
     * Called at the end of each library.
     */
    void endOfLibrary() final override;

    YAML::Node getConfig() const final override;
    YAML::Node getSummary() const final override;

    void writeSummary(boost::filesystem::path const& path) const final {
      output_.writeSummary(path);
    }

  private:
    ObservationPlane<TTracking, TOutput> const obsPlane_;
    bool writtenThisShower_;
    unsigned int showerId_;
    ShowerSummary output_;
  };

} // namespace corsika

#include <corsika/detail/modules/writers/PrimaryWriter.inl>
