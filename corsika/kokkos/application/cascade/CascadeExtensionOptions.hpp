/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/application/cascade/CascadeSupport.hpp>
#include <corsika/kokkos/radio/firn/IFullWaveAccumulator.hpp>
#include <corsika/kokkos/radio/firn/IGeometricalOpticsAccumulator.hpp>

namespace corsika::kokkos::application {
  /** Optional application extension. No native configuration defaults are replaced. */
  class CascadeExtension {
    CLI::App& app_;
    BackendOptions options_;
    CascadeWorkerOptions workers_;
    CascadeRuntime runtime_;
    RunEnvironmentConfig environment_;

  public:
    explicit CascadeExtension(CLI::App&, CascadeScene = CascadeScene::Atmosphere);
    bool prepare(DeviceSelection const&, char const* executable);
    void configure(RunEnvironmentConfig const&);
    void registerOutput(OutputManager&);
    /// Bind strict full-wave device radio without exposing runtime details in the app.
    std::shared_ptr<radio::firn::IFullWaveAccumulator> firnRadioAccumulator(
        std::string const& model, std::filesystem::path const& manifest,
        std::string const& fingerprint, std::vector<radio::firn::vector_type> observers,
        radio::firn::FullWaveDeviceConfig config);
    /// Bind the existing coherent GO model to the selected execution space.
    std::shared_ptr<radio::firn::IGeometricalOpticsAccumulator>
    firnGeometricalOpticsAccumulator(std::string const& model,
                                     radio::firn::ExponentialOpticalParameters profile,
                                     std::vector<radio::firn::vector_type> observers,
                                     radio::firn::OnlineRadioConfig config);
    double radioWindowDurationNs() const { return options_.radioWindowDurationNs_; }
    double radioSamplingRateGHz() const { return options_.radioSamplingRateGHz_; }
    double radioPretriggerNs() const { return options_.radioPretriggerNs_; }

    template <typename TE, typename TL, typename TH>
    auto interaction(TE const&, TL&, TH&, HEPEnergyType);
    template <typename TModel>
    auto interactionCounter(TModel&);
    template <typename TAxis, typename TDepth>
    auto energyLossWriter(TAxis const&, TDepth const&);
    template <typename TAxis, typename TDepth>
    auto longitudinalWriter(TAxis const&, TDepth const&);

    template <typename TE, typename TC, typename TP, typename TS, typename TStep,
              typename TA, typename TX, typename TW, typename TLW, typename TPW,
              typename TObs, typename TIW, typename TDC, typename TDZ, typename TCoreas,
              typename TZhs, typename TEM, typename TProposal, typename TN, typename TH,
              typename TD, typename TCont, typename TL, typename TProd, typename TCut,
              typename THC, typename TLC, typename TOutput>
    auto bind(TE&, TC const&, TP const&, TS const&, TStep const&, TA const&, TX const&,
              std::uint64_t, HEPEnergyType, Code, HEPEnergyType, HEPEnergyType, bool, TW&,
              TLW&, TPW&, TObs&, TIW&, TDC&, TDZ&, TCoreas&, TZhs&, TEM&, TProposal&, TN&,
              TH&, TD&, TCont&, TL&, TProd&, TCut&, THC&, TLC&, TOutput&);
  };
} // namespace corsika::kokkos::application
