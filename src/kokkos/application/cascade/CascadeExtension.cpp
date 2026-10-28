/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/application/cascade/CascadeExtensionOptions.hpp>
#include <stdexcept>

namespace corsika::kokkos::application {
  CascadeExtension::CascadeExtension(CLI::App& app, CascadeScene scene)
      : app_(app) {
    options_.hostRadioAdapter_ = scene == CascadeScene::Firn;
    add_backend_cli_options(app_, options_);
    workers_.addOptions(app_, options_);
  }

  bool CascadeExtension::prepare(DeviceSelection const& devices, char const* executable) {
    return workers_.prepare(app_, options_, devices, app_["--nevent"]->as<int>(),
                            app_.count("--energy_range") != 0,
                            app_["--force-interaction"]->as<bool>(),
                            app_["--force-decay"]->as<bool>(), executable);
  }

  void CascadeExtension::configure(RunEnvironmentConfig const& environment) {
    if (options_.hostRadioAdapter_ != (environment.scene_ == CascadeScene::Firn)) {
      throw std::invalid_argument("radio adapter does not match the configured scene");
    }
    if (environment.scene_ != CascadeScene::Atmosphere &&
        (options_.emBackend_ == "kokkos-egs4" || !options_.cudaReplayTapeOut_.empty())) {
      throw std::invalid_argument(
          "ice/firn currently support PROPOSAL transport and standard output; "
          "the EGS4 and atmospheric diagnostic replay adapters are not enabled");
    }
    if (environment.scene_ == CascadeScene::Firn &&
        (!options_.radioTrackInputs_.empty() || !workers_.radioOutput_.empty())) {
      throw std::invalid_argument(
          "firn deferred track replay and multi-process radio merging are not connected");
    }
    environment_ = environment;
    runtime_.configure(options_, environment_);
  }

  void CascadeExtension::registerOutput(OutputManager& output) {
    runtime_.registerOutput(output);
  }
  std::shared_ptr<radio::firn::IFullWaveAccumulator>
  CascadeExtension::firnRadioAccumulator(std::string const& model,
                                         std::filesystem::path const& manifest,
                                         std::string const& fingerprint,
                                         std::vector<radio::firn::vector_type> observers,
                                         radio::firn::FullWaveDeviceConfig config) {
    if (environment_.scene_ != CascadeScene::Firn)
      throw std::invalid_argument("full-wave adapter requires a firn scene");
    if (options_.radioBackend_ == "cpu" || model == "coherent-go") return nullptr;
    if (model != "full-wave")
      throw std::invalid_argument(
          "Kokkos firn radio supports coherent-go or full-wave; "
          "implicit hybrid GO fallback is not supported");
    return radio::firn::make_full_wave_accumulator(
        manifest, fingerprint, std::move(observers), config, options_.kokkosNumThreads_);
  }
  std::shared_ptr<radio::firn::IGeometricalOpticsAccumulator>
  CascadeExtension::firnGeometricalOpticsAccumulator(
      std::string const& model, radio::firn::ExponentialOpticalParameters profile,
      std::vector<radio::firn::vector_type> observers,
      radio::firn::OnlineRadioConfig config) {
    if (environment_.scene_ != CascadeScene::Firn)
      throw std::invalid_argument("GO adapter requires a firn scene");
    if (options_.radioBackend_ == "cpu" || model != "coherent-go") return nullptr;
    radio::firn::GeometricalOpticsConfig deviceConfig;
    deviceConfig.radio_ = config;
    return radio::firn::make_geometrical_optics_accumulator(
        profile, std::move(observers), deviceConfig, options_.kokkosNumThreads_);
  }
} // namespace corsika::kokkos::application
