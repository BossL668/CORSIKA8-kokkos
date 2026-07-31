/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/output/BaseOutput.hpp>
#include <corsika/output/ParquetStreamer.hpp>
#include <corsika/framework/process/ContinuousProcess.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupTrajectory.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>

namespace corsika {

  /**
   * The base interface for radio emission processes.
   *
   * TRadioImpl is the concrete implementation of the radio algorithm.
   * TObserverCollection is the detector instance that stores observers
   * and is responsible for managing the output writing.
   */
  template <typename TObserverCollection, typename TRadioImpl, typename TPropagator>
  class RadioProcess : public ContinuousProcess<
                           RadioProcess<TObserverCollection, TRadioImpl, TPropagator>>,
                       public BaseOutput {

    /*
     * A collection of filter objects for deciding on valid particles and tracks.
     */
    // std::vector<std::function<bool(ParticleType&, TrackType const&)>> filters_;

    /**
     * Get a reference to the underlying radio implementation.
     */
    TRadioImpl& implementation();

    /**
     *  Get a const reference to the underlying implementation.
     */
    TRadioImpl const& implementation() const;

  protected:
    TObserverCollection& observers_; ///< The radio observers we store into.
    TPropagator propagator_;         ///< The propagator implementation.
    unsigned int showerId_{0};       ///< The current event ID.
    ParquetStreamer output_;         //!< The parquet streamer for this process.
    std::uint64_t diagnosticSegmentCount_{0};
    double diagnosticWeightedSegmentCount_{0.};
    double diagnosticTrackLengthM_{0.};
    double diagnosticWeightedTrackLengthM_{0.};
    double diagnosticElectronWeightedTrackLengthM_{0.};
    double diagnosticPositronWeightedTrackLengthM_{0.};
    double diagnosticSignedChargeTrackLengthM_{0.};
    double diagnosticEnergyWeightedTrackLengthGeVM_{0.};
    double diagnosticMaximumSegmentLengthM_{0.};
    double diagnosticWeightedDirectionChangeRad_{0.};
    double diagnosticWeightedDirectionChangeSquaredRad2_{0.};
    double diagnosticWeightedBetaDeficitTrackLengthM_{0.};
    double diagnosticWeightedTimeResidualS_{0.};
    double diagnosticMaximumDirectionChangeRad_{0.};
    std::array<double, 3>
        diagnosticSignedChargeWeightedDirectionChange_{};
    std::array<double, 15> diagnosticEnergyBinnedTrackLengthM_{};
    YAML::Node diagnosticSummary_;

    inline static constexpr std::array<double, 15>
        diagnosticEnergyUpperEdgesGeV_{
            1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2,
            5.e-2, 1.e-1, 2.e-1, 5.e-1, 1.,
            2., 5., 10., 100., std::numeric_limits<double>::infinity()};

  public:
    using axistype = std::vector<long double>;
    /**
     * Construct a new RadioProcess.
     */
    RadioProcess(TObserverCollection& observers, TPropagator& propagator);

    /**
     * Perform the continuous process (radio emission).
     *
     * This handles filtering individual particle tracks
     * before passing them to `Simulate`.`
     *
     * @param particle    The current particle.
     * @param track       The current track.
     */
    template <typename Particle>
    ProcessReturn doContinuous(Step<Particle> const& step, bool const);

    /**
     * Return the maximum step length for this particle and track.
     *
     * This must be provided by the TRadioImpl.
     *
     * @param particle    The current particle.
     * @param track       The current track.
     *
     * @returns The maximum length of this track.
     */
    template <typename Particle, typename Track>
    LengthType getMaxStepLength(Particle const& vParticle, Track const& vTrack) const;

    /**
     * Called at the start of each library.
     */
    void startOfLibrary(boost::filesystem::path const& directory) final override;

    /**
     * Called at the end of each shower.
     */
    virtual void endOfShower(unsigned int const) final override;

    /**
     * Called at the end of each library.
     *
     */
    void endOfLibrary() final override;

    TObserverCollection& observerCollection() noexcept {
      return observers_;
    }

    TObserverCollection const& observerCollection() const noexcept {
      return observers_;
    }

    /**
     * Get the configuration of this output.
     */
    YAML::Node getConfig() const final;

    /**
     * Per-shower track diagnostics used to separate transport differences from
     * CoREAS/ZHS projection differences.
     */
    YAML::Node getSummary() const final;

  }; // END: class RadioProcess

} // namespace corsika

#include <corsika/detail/modules/radio/RadioProcess.inl>
