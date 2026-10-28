/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/Fft.hpp>
#include <corsika/kokkos/radio/firn/IFullWaveAccumulator.hpp>
#include <corsika/kokkos/radio/firn/IGeometricalOpticsAccumulator.hpp>
#include <corsika/kokkos/radio/firn/GreenResponse.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/process/ContinuousProcess.hpp>
#include <corsika/output/BaseOutput.hpp>
#include <corsika/output/ParquetStreamer.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <corsika/kokkos/radio/firn/RadioObserver.hpp>
#include <corsika/kokkos/radio/firn/OnlineRadioConfig.hpp>

namespace corsika::kokkos::radio::firn {
  /** Accumulate finite-track emission with the selected stratified Green response.
   * The caller supplies local coordinates with z measured from the profile surface;
   * all C8 points are transformed explicitly, regardless of their stored frame. */
  template <typename TBackend>
  class VariableIndexRadioProcess final
      : public ContinuousProcess<VariableIndexRadioProcess<TBackend>>,
        public BaseOutput {
  public:
    VariableIndexRadioProcess(
        std::vector<RadioObserver> observers, TBackend backend, OnlineRadioConfig config,
        CoordinateSystemPtr radioCoordinates,
        std::shared_ptr<IFullWaveAccumulator> accumulator = nullptr,
        std::shared_ptr<IGeometricalOpticsAccumulator> goAccumulator = nullptr);

    template <typename TParticle>
    ProcessReturn doContinuous(Step<TParticle> const& step, bool const);

    template <typename TParticle, typename TTrack>
    LengthType getMaxStepLength(TParticle const&, TTrack const&) const;

    void startOfLibrary(boost::filesystem::path const& directory) final;

    void startOfShower(unsigned int showerId) final;

    void endOfShower(unsigned int showerId) final;

    void endOfLibrary() final;

    YAML::Node getConfig() const final;

    YAML::Node getSummary() const final;

    std::size_t residentSpectrumBytes() const;

  private:
    template <typename TPoint>
    vector_type pointCoordinates(TPoint const& point) const;

    void resetEvent();
    ProcessReturn accumulateFullWave(vector_type const& start, vector_type const& end,
                                     double startTimeS, double endTimeS, double weight,
                                     Code pid);
    void materializeFullWaveSpectra();
    std::shared_ptr<IFullWaveAccumulator> accumulator_;
    std::shared_ptr<IGeometricalOpticsAccumulator> goAccumulator_;
    void materializeGeometricalOpticsSpectra();

    void validateResponse(GreenResponseBatch const& response) const;

    void recordBranch(PathDiagnostics const& path);

    static constexpr std::size_t branchFamilyCount_ = 7;

    static std::size_t branchFamilyIndex(std::string const& topology);

    static char const* branchFamilyName(std::size_t index);

    std::vector<RadioObserver> observers_;
    TBackend backend_;
    CoordinateSystemPtr radioCoordinates_;
    OnlineRadioConfig config_;
    std::vector<std::array<std::vector<std::complex<double>>, 3>> spectra_;
    std::vector<
        std::array<std::array<std::vector<std::complex<double>>, 3>, branchFamilyCount_>>
        branchSpectra_;
    ParquetStreamer waveformOutput_, spectrumOutput_, replayOutput_;
    ParquetStreamer branchWaveformOutput_, branchSpectrumOutput_;
    unsigned int showerId_{0};
    std::uint64_t inputSegments_{0}, subdividedSegments_{0};
    std::uint64_t skippedDegenerateSegments_{0}, responseQueries_{0};
    std::uint64_t branchContributionCount_{0};
    std::size_t maximumSubdivisionsSeen_{0};
    double maximumPhaseCyclesSeen_{0.0};
    double maximumPointCurrentPhaseIndicatorSeen_{0.0};
    double minimumBranchTravelTimeS_{std::numeric_limits<double>::infinity()};
    double maximumBranchTravelTimeS_{0.0};
    double maximumBranchFermatResidual_{0.0};
    double maximumBranchSnellResidual_{0.0};
    std::string worstFermatTopology_{"none"};
    double worstFermatHorizontalResidualM_{0.0};
    double worstFermatInvariantP_{0.0};
    std::map<std::string, std::uint64_t> branchTopologyCounts_;
    YAML::Node diagnosticSummary_;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/VariableIndexRadioProcess.inl>
