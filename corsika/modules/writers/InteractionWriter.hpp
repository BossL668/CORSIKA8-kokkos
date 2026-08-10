/*
 * (c) Copyright 2024 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/process/SecondariesProcess.hpp>

#include <corsika/media/ShowerAxis.hpp>

#include <corsika/modules/ObservationPlane.hpp>
#include <corsika/modules/writers/FirstInteractionSnapshot.hpp>

#include <corsika/output/BaseOutput.hpp>
#include <corsika/output/ParquetStreamer.hpp>

namespace corsika {

  template <typename TTracking, typename TOutput>
  class InteractionWriter
      : public SecondariesProcess<InteractionWriter<TTracking, TOutput>>,
        public BaseOutput {

  public:
    /**
     *
     * @param axis - axis used for calculating slant depth of interaction
     * @param obsPlane - used to define the location of the particles
     */
    InteractionWriter(ShowerAxis const& axis,
                      ObservationPlane<TTracking, TOutput> const& obsPlane);

    /**
     *
     * @tparam TStackView
     */
    template <typename TStackView>
    void doSecondaries(TStackView&);

    /**
     * Record a pre-thinning first-interaction snapshot produced outside the
     * scalar Stack path. Returns false when an earlier CPU/GPU interaction
     * has already claimed the shower's single output record.
     */
    bool recordFirstInteraction(FirstInteractionSnapshot const&);

    void startOfLibrary(boost::filesystem::path const&) override;
    void endOfLibrary() override;

    void startOfShower(unsigned int const) override;
    void endOfShower(unsigned int const) override;

    YAML::Node getConfig() const override;
    YAML::Node getSummary() const override;

    auto getInteractionCounter() { return interactionCounter_; }

  private:
    void writeFirstInteraction(FirstInteractionSnapshot const&);

    ObservationPlane<TTracking, TOutput> const obsPlane_;
    ShowerAxis const& showerAxis_;

    unsigned int interactionCounter_;
    unsigned int showerId_;

    ParquetStreamer output_;
    YAML::Node summary_;

  }; // namespace corsika

} // namespace corsika

#include <corsika/detail/modules/writers/InteractionWriter.inl>
