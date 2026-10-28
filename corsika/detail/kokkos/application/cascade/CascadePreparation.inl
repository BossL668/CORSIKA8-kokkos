/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::application {
  template <typename TRunSession, typename TEnvironment, typename TInjectionPosition,
            typename TSurface, typename TPropagationStep, typename TDetectorCoREAS,
            typename TDetectorZHS, typename TShowerAxis, typename TDepthStep,
            typename TEnergyLossWriter, typename TLongitudinalWriter>
  auto prepare_cascade(TRunSession const& session, BackendOptions const& options,
                       CascadeEventConfig const& event, TEnvironment& environment,
                       CoordinateSystemPtr const& rootCoordinates,
                       TInjectionPosition const& injectionPosition,
                       TSurface const& surface, TPropagationStep const& propagationStep,
                       TDetectorCoREAS& coreasDetector, TDetectorZHS& zhsDetector,
                       TShowerAxis const& showerAxis, TDepthStep const& depthStep,
                       double observationHeightM, double showerCoreXM,
                       double showerCoreYM,
                       std::array<double, 3> const& /* magneticFieldT */,
                       TEnergyLossWriter& energyLossWriter,
                       TLongitudinalWriter& longitudinalWriter,
                       ObservationPlane const* observationPlane) {
    using namespace corsika::kokkos;
    using namespace corsika::units::si;

    using model_properties_type = typename TEnvironment::BaseNodeType::IModelProperties;
    PreparedCascade<model_properties_type> prepared;
    prepared.proposalStochasticCut_ =
        proposal::optimized_proposal_energy_cut(event.productionThreshold_);
    prepared.physicsRequirements_ = {(options.gpuPhysicsSource_ == "proposal-native"
                                          ? event.configuredPrimaryEnergyMax_
                                          : event.primaryTotalEnergy_) /
                                         1_MeV,
                                     event.emCut_ / 1_MeV, event.muonCut_ / 1_MeV,
                                     prepared.proposalStochasticCut_ / 1_MeV};
    {
      auto exported = prepare_environment_snapshot(environment, rootCoordinates);
      if (event.scene_ == CascadeScene::Atmosphere) {
        atmosphere::require_air_materials(*exported);
      } else if (event.scene_ == CascadeScene::Firn) {
        firn::require_firn(*exported);
      } else {
        ice::require_homogeneous_ice(*exported);
      }
      prepared.environmentStorage_ = exported;
      prepared.environmentData_ = bind_environment_data(*prepared.environmentStorage_);
      prepared.geometry_.query_.maximumMagneticDeflectionRad_ =
          event.maximumMagneticDeflectionRad_;
      auto const& topology = exported->getTopology();
      CORSIKA_LOG_INFO(
          "C8 environment snapshot: {} nodes, {} models, {} child links, {} exclusions",
          topology.getNodes().size(), exported->getModels().size(),
          topology.getChildren().size(), topology.getExclusions().size());
    }
    prepared.geometry_.observation_ =
        observationPlane ? *observationPlane
                         : atmosphere::horizontal_observation_plane(
                               showerCoreXM, showerCoreYM, observationHeightM);
    if (!observation_plane_detail::valid_observation_plane(
            prepared.geometry_.observation_)) {
      throw std::invalid_argument("invalid GPU observation plane point or normal");
    }

    auto& config = prepared.backendConfig_;
    config.device_ = options.emBackend_ == "kokkos-proposal" ? options.kokkosDevice_
                                                             : options.gpuDevice_;
    config.minBatchSize_ = options.gpuMinBatch_;
    config.residentBatchLimit_ = options.gpuResidentBatchLimit_;
    config.memoryFraction_ = options.gpuMemoryFraction_;
    config.emTransportCutMeV_ = event.emCut_ / 1_MeV;
    config.muonTransportCutMeV_ = event.muonCut_ / 1_MeV;
    config.physicsSource_ = PhysicsSource::ProposalNative;
    config.auxiliaryCacheDirectory_ = options.gpuAuxCacheDir_;
    config.deterministic_ = options.gpuDeterministic_;
    config.detailedStageTiming_ = options.gpuDetailedStageTiming_;
    config.diagnosticInteractionRecords_ = corsika::ReplayTrace::instance().enabled();
    config.randomSeed_ = event.seed_;
    config.showerId_ = event.showerOrdinal_;
    config.thinning_.enabled_ = event.emThinningFraction_ > 0. ? 1 : 0;
    config.thinning_.thresholdGeV_ =
        event.emThinningFraction_ * event.primaryTotalEnergy_ / 1_GeV;
    config.thinning_.maximumWeight_ = event.maximumWeight_;
    config.thinning_.eraseZeroWeight_ = event.keepZeroWeightParticles_ ? 0 : 1;
    config.residentCrossSpecies_ = options.gpuResidentCrossSpecies_;

    prepared.radioEnabled_ =
        event.scene_ != CascadeScene::Firn && options.radioBackend_ == "kokkos" &&
        options.emBackend_ == "kokkos-proposal" && coreasDetector.size() != 0;
    if (prepared.radioEnabled_ && !session.hasBackend()) {
      if (event.scene_ == CascadeScene::Atmosphere) {
        config.radio_ = corsika::kokkos::radio::atmosphere::make_radio_config(
            environment, injectionPosition, surface, propagationStep, coreasDetector,
            zhsDetector, true, event.radioZhsEnabled_);
      } else {
        config.radio_ = corsika::kokkos::radio::make_homogeneous_radio_config(
            ice::require_homogeneous_ice(*prepared.environmentStorage_), rootCoordinates,
            coreasDetector, zhsDetector, true, event.radioZhsEnabled_);
      }
      config.radio_.deterministic_ = config.deterministic_;
      config.radio_.fixedPointFieldLimitVPerM_ = options.gpuRadioFieldLimit_;
      config.radio_.trackDiagnostics_ = options.gpuRadioTrackDiagnostics_;
    }

    if ((coreasDetector.size() == 0 || prepared.radioEnabled_) &&
        !options.gpuFullStepRecords_) {
      auto& projection = config.profileProjection_;
      projection.crossingMode_ = longitudinalWriter.getCrossingMode();
      auto const primaryEnergyGeV = event.primaryTotalEnergy_ / 1_GeV;
      auto const emCutGeV = event.emCut_ / 1_GeV;
      projection.fixedPointWeightLimit_ = 2. * primaryEnergyGeV / emCutGeV;
      // Gross atomic-target rest-mass entries can exceed the primary energy
      // (they cancel against cut electron rest masses). A stopped 0.7 MeV e+
      // followed by six Compton vertices is a concrete example. Keep ample
      // low-energy integer range; this changes neither deposits nor allocation
      // size, and leaves all >=0.5 GeV production scales unchanged.
      projection.fixedPointEnergyLimitGeV_ = std::max(2. * primaryEnergyGeV, 1.);
      if (!session.hasBackend()) {
        projection.enabled_ = true;
        auto const axisStart = showerAxis.getStart().getCoordinates(rootCoordinates);
        auto const axisDirection =
            showerAxis.getDirection().getComponents(rootCoordinates);
        for (int axis = 0; axis < 3; ++axis) {
          projection.axisStartPositionM_[axis] = axisStart[axis] / 1_m;
          projection.axisDirection_[axis] = axisDirection[axis].magnitude();
        }
        projection.axisStepLengthM_ = showerAxis.getSteplength() / 1_m;
        auto const& support = showerAxis.getGrammageSupport();
        projection.axisGrammageGPerCm2_.reserve(support.size());
        for (auto const grammage : support) {
          projection.axisGrammageGPerCm2_.push_back(grammage / (1_g / square(1_cm)));
        }
        if (energyLossWriter.GetNBins() != longitudinalWriter.getNBins()) {
          throw std::runtime_error(
              "CUDA resident profile requires identical energy-loss and "
              "longitudinal bin counts");
        }
        projection.accumulateOnDevice_ = true;
        projection.outputBinCount_ = energyLossWriter.GetNBins();
        projection.outputBinWidthGPerCm2_ = depthStep / (1_g / square(1_cm));
        projection.energyLossThresholdGPerCm2_ = 1.e-4;
      }
    }
    return prepared;
  }
} // namespace corsika::kokkos::application
