/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::application {
  namespace egs4_api = ::corsika::kokkos::egs4::application;
  namespace egs4_detail {
    // Preserve all non-EM cuts. EGS4, not the scalar cut, owns EM termination
    // (including positrons already below the transport cut at their birth).
    template <class TCut>
    struct NonEmCut : ContinuousProcess<NonEmCut<TCut>>,
                      SecondariesProcess<NonEmCut<TCut>> {
      TCut& original_;
      explicit NonEmCut(TCut& c)
          : original_(c) {}
      template <class TP, class TT>
      LengthType getMaxStepLength(TP const&, TT const&) const {
        return std::numeric_limits<double>::infinity() * 1_m;
      }
      template <class TP>
      ProcessReturn doContinuous(Step<TP>& s, bool limited) {
        if (is_em(s.getParticlePre().getPID()))
          throw std::logic_error("EGS4 EM reached scalar transport");
        return original_.doContinuous(s, limited);
      }
      template <class TP>
      bool arrival(TP& p, egs4_api::HostRequest const&) {
        if (is_em(p.getPID())) return true;
        StraightTrajectory track(Line(p.getPosition(), p.getDirection() * constants::c),
                                 0_s);
        Step step(p, track);
        auto result = original_.doContinuous(step, false);
        if (result != ProcessReturn::Ok && result != ProcessReturn::ParticleAbsorbed)
          throw std::runtime_error("Unexpected non-EM arrival cut");
        return result == ProcessReturn::Ok;
      }
      template <class TV>
      void doSecondaries(TV& v) {
        for (auto p = v.begin(); p != v.end(); ++p)
          if (!arrival(p, {})) p.erase();
      }
    };
    class Report final : public BaseOutput {
      YAML::Node config_;
      mutable ShowerSummary summary_;

    public:
      explicit Report(BackendOptions const& o) {
        config_["em_backend"] = o.emBackend_;
        config_["stepfc"] = o.egs4Stepfc_;
        config_["table_sha256"] = egs4_api::embedded_table_sha256();
        config_["table_storage"] = "embedded at build time";
        config_["non_em"] = "shared C8 models; PROPOSAL muons";
        config_["native_em_production_profile_complete"] = false;
      }
      void startOfLibrary(boost::filesystem::path const& p) final {
        summary_.open(p);
        setInit(true);
      }
      void startOfShower(unsigned i) final {
        summary_.event(i)["status"] = "in_progress";
        summary_.flush();
      }
      void endOfShower(unsigned) final { summary_.flush(); }
      void endOfLibrary() final {
        summary_.flush();
        setInit(false);
      }
      YAML::Node getConfig() const final { return config_; }
      YAML::Node getSummary() const final { return summary_.snapshot(); }
      void writeSummary(boost::filesystem::path const& p) const final {
        summary_.writeSummary(p);
      }
      void failed(unsigned i, std::string const& message) {
        summary_.event(i)["status"] = "incomplete";
        summary_.event(i)["reason"] = message;
        summary_.flush();
      }
      template <class TTiming>
      void completed(unsigned i, egs4_api::RunStatistics const& s, TTiming const& t,
                     double discarded, bool prefix = false) {
        for (auto const& entry : t.scalarStepsByPdg_)
          if (entry.first == 11 || entry.first == -11 || entry.first == 22)
            throw std::runtime_error("EM escaped EGS4 router");
        auto n = summary_.event(i);
        n["status"] = "completed";
        n["prefix_capture"] = prefix;
        n["scalar_em_steps"] = 0;
        n["native_steps"] = s.steps_;
        n["execution_space"] = s.executionSpace_;
        n["radio_execution_space"] = s.radioExecutionSpace_;
        n["radio_tracks"] = s.radioTracks_;
        n["radio_kernel_ms"] = s.radioKernelMs_;
        n["radio_projection_batches"] = s.radioProjectionBatches_;
        n["radio_fixed_point_overflows"] = s.radio_.fixedPointOverflows_;
        n["profile_execution_space"] = s.profileExecutionSpace_;
        n["profile_fixed_point_overflows"] = s.profileFixedPointOverflows_;
        n["profile_invalid_records"] = s.profileInvalidRecords_;
        n["discarded_available_energy_GeV"] = discarded;
        summary_.flush();
      }
    };
  } // namespace egs4_detail

  // Uses the SAME application-owned environment, models, cuts, observers, writers
  // and event settings as PROPOSAL. Only the EM router/termination is replaced.
  // No Fe fixture, duplicate CLI, private atmosphere or private antenna defaults.
  class Egs4RunSession {
    EmRunSession muons_;
    std::unique_ptr<egs4_api::Session> native_; // dies before muons_ / Kokkos
    egs4_detail::Report report_;
    int capacity_{};
    double previousThreshold_{-1.}, previousWeight_{-1.};
    static RuntimeConfig runtime(BackendOptions const& o) {
      RuntimeConfig c;
      c.device_ = o.kokkosDevice_;
      c.threads_ = o.kokkosNumThreads_;
      c.executionBackend_ = o.kokkosExecution_;
      return c;
    }

  public:
    explicit Egs4RunSession(BackendOptions const& o)
        : muons_(runtime(o), o.gpuAuxCacheDir_)
        , report_(o) {}
    template <class TOutput>
    void registerOutput(TOutput& o) {
      o.add("native_egs4", report_);
    }
    template <class TEvent, class TContext>
    void run(BackendOptions const& options, TEvent const& event, TContext const& c) {
      auto const& g = c.geometry_;
      auto const& o = c.outputs_;
      auto const& p = c.processes_;
      auto const& e = c.execution_;
      auto const& d = c.diagnostics_;
      using stack_type = std::remove_reference_t<decltype(e.stack_)>;
      using environment_type = std::remove_reference_t<decltype(g.environment_)>;
      using tracking_type = std::remove_reference_t<decltype(e.tracking_)>;
      try {
        if (event.scene_ != CascadeScene::Atmosphere)
          throw std::invalid_argument("EGS4 AIR-NTP does not support ice or firn scenes");
        auto snapshot = prepare_environment_snapshot(g.environment_, g.coordinates_);
        auto treeData = bind_environment_data(*snapshot);
        auto const& topology = snapshot->getTopology();
        using model_properties_type =
            typename environment_type::BaseNodeType::IModelProperties;
        TransportGeometryConfig geometry;
        geometry.query_.maximumMagneticDeflectionRad_ =
            event.maximumMagneticDeflectionRad_;
        geometry.observation_ =
            export_observation_plane(o.observation_.getPlane(), g.coordinates_);

        egs4_detail::NonEmCut<std::remove_reference_t<decltype(p.cut_)>> cut(p.cut_);
        auto sequence = make_sequence(
            p.stackInspector_, p.neutrinoPrimary_, p.hadronSequence_, p.decaySequence_,
            p.emCascade_, p.productionProcess_, p.emContinuous_, p.coreas_, p.zhs_,
            p.longitudinalProcess_, o.observation_, o.interactions_, p.thinning_, cut);
        if (!frontier::activeFrontierOutput.empty()) {
          // Never advance prefix EM with PROPOSAL before handing it to EGS4.
          frontier::CaptureRouter capture(frontier::activeFrontierOutput, g.coordinates_,
                                          std::numeric_limits<double>::max(), *snapshot);
          HybridCascade<tracking_type, decltype(sequence),
                        std::remove_reference_t<decltype(e.output_)>, stack_type,
                        decltype(capture)>
              cascade(g.environment_, e.tracking_, sequence, e.output_, e.stack_,
                      capture);
          e.configurePrimary_(cascade);
          cascade.run();
          report_.completed(event.outputShowerId_, {}, cascade.timingStatistics(), 0.,
                            true);
          return;
        }
        auto environment = ::corsika::kokkos::egs4::export_atmosphere(
            treeData, geometry, constants::EarthRadius::Mean / 1_m);
        double const threshold =
            event.emThinningFraction_ * event.primaryTotalEnergy_ / 1_GeV;
        BackendConfig m;
        m.minBatchSize_ = 64;
        m.residentBatchLimit_ = 4096;
        m.memoryFraction_ = .05;
        m.randomSeed_ = event.seed_;
        m.showerId_ = event.outputShowerId_;
        m.deterministic_ = options.gpuDeterministic_;
        m.detailedStageTiming_ = options.gpuDetailedStageTiming_;
        m.emTransportCutMeV_ = event.emCut_ / 1_MeV;
        m.muonTransportCutMeV_ = event.muonCut_ / 1_MeV;
        m.thinning_ = {threshold, event.maximumWeight_, threshold > 0.,
                       !event.keepZeroWeightParticles_};
        m.residentCrossSpecies_ = false;
        auto muon = muons_.beginProposalNative(
            treeData, geometry, m,
            {event.configuredPrimaryEnergyMax_ / 1_MeV, event.emCut_ / 1_MeV,
             event.muonCut_ / 1_MeV,
             proposal::optimized_proposal_energy_cut(event.productionThreshold_) / 1_MeV},
            p.emCascade_.nativeCalculatorViews(),
            p.emContinuousProposal_.nativeCalculatorViews());
        if (!muon.muonTransportAvailable_)
          throw std::runtime_error("PROPOSAL muon backend unavailable");
        if (!native_) native_ = egs4_api::make_embedded_session();
        egs4_api::Configuration cfg;
        cfg.environment_ = environment;
        cfg.earthRadiusM_ = constants::EarthRadius::Mean / 1_m;
        cfg.seed_ = event.seed_;
        cfg.showerId_ = event.outputShowerId_;
        cfg.stepfc_ = options.egs4Stepfc_;
        cfg.sterncor_ = native_->sternheimerCorrection();
        cfg.maximumWaves_ = std::numeric_limits<int>::max();
        cfg.reuseQueueWorkspace_ = true;
        if (options.radioBackend_ == "kokkos") {
          cfg.radio_ = radio::atmosphere::make_radio_config(
              g.environment_, g.injectionPosition_, g.surface_, g.propagationStep_,
              o.coreasDetector_, o.zhsDetector_, true, event.radioZhsEnabled_);
          cfg.radio_.fixedPointFieldLimitVPerM_ = options.gpuRadioFieldLimit_;
          cfg.radio_.deterministic_ = options.gpuDeterministic_;
          cfg.radio_.trackDiagnostics_ = options.gpuRadioTrackDiagnostics_;
        }
        if (!capacity_) {
          auto execution = resolve_execution_backend(options.kokkosExecution_);
          capacity_ =
              (execution == "cuda" || execution == "hip")
                  ? native_->gpuMemoryBudget(options.gpuMemoryFraction_, cfg.radio_)
                        .queueCapacity_
                  : 1048576;
          if (options.gpuResidentBatchLimit_) {
            if (options.gpuResidentBatchLimit_ > std::size_t(capacity_))
              throw std::length_error(
                  "Resident batch limit exceeds the EGS4 queue memory budget");
            capacity_ = int(options.gpuResidentBatchLimit_);
          }
        }
        cfg.queueCapacity_ = capacity_;
        cfg.thinning_ = m.thinning_;
        auto mass = [](int id) {
          return get_mass(convert_from_PDG(static_cast<PDGCode>(id))) / 1_MeV;
        };
        cfg.electronMassMeV_ = mass(11);
        cfg.electronTotalCutMeV_ = mass(11) + event.emCut_ / 1_MeV;
        cfg.photonCutMeV_ = event.emCut_ / 1_MeV;
        cfg.muonMassMeV_ = mass(13);
        cfg.chargedPionMassMeV_ = mass(211);
        cfg.neutralPionMassMeV_ = mass(111);
        cfg.protonMassMeV_ = mass(2212);
        cfg.neutronMassMeV_ = mass(2112);
        cfg.phiMassMeV_ = mass(333);
        cfg.omegaMassMeV_ = mass(223);
        cfg.rhoMassMeV_ = mass(113);
        cfg.chargedKaonMassMeV_ = mass(321);
        cfg.longKaonMassMeV_ = mass(130);
        cfg.shortKaonMassMeV_ = mass(310);
        cfg.etaMassMeV_ = mass(221);
        cfg.resolveRareVertices_ = cfg.nativePhotonuclearVertices_ =
            cfg.nativePromptRhoDecay_ = cfg.nativePromptResonanceDecay_ = true;
        auto& projection = cfg.profile_;
        projection.enabled_ = projection.accumulateOnDevice_ = true;
        projection.crossingMode_ = o.longitudinal_.getCrossingMode();
        projection.outputBinCount_ = o.energyLoss_.GetNBins();
        if (projection.outputBinCount_ != o.longitudinal_.getNBins())
          throw std::runtime_error("EGS4 output grid mismatch");
        projection.outputBinWidthGPerCm2_ = g.depthStep_ / (1_g / square(1_cm));
        projection.fixedPointWeightLimit_ =
            2. * event.configuredPrimaryEnergyMax_ / event.emCut_;
        projection.fixedPointEnergyLimitGeV_ =
            std::max(2. * event.configuredPrimaryEnergyMax_ / 1_GeV, 1.);
        auto start = g.showerAxis_.getStart().getCoordinates(g.coordinates_);
        auto direction = g.showerAxis_.getDirection().getComponents(g.coordinates_);
        for (int j = 0; j < 3; ++j) {
          projection.axisStartPositionM_[j] = start[j] / 1_m;
          projection.axisDirection_[j] = direction[j].magnitude();
        }
        projection.axisStepLengthM_ = g.showerAxis_.getSteplength() / 1_m;
        for (auto x : g.showerAxis_.getGrammageSupport())
          projection.axisGrammageGPerCm2_.push_back(x / (1_g / square(1_cm)));
        cfg.retainAcrossShowers_ = true;
        cfg.resumeRetainedShower_ = native_->hasRetainedShower() &&
                                    threshold == previousThreshold_ &&
                                    event.maximumWeight_ == previousWeight_;
        CorsikaOutputSink sink(g.coordinates_, o.energyLoss_, o.longitudinal_,
                               o.production_, o.observation_, o.interactions_, p.coreas_,
                               p.zhs_, true, cfg.radio_.enabled_);
        double discarded = 0.;
        egs4_api::OutputCallbacks callbacks;
        callbacks.step_ = [&](auto const& r) { sink.onStep(r); };
        callbacks.profile_ = [&](auto const& r) { sink.onGpuProfile(r); };
        callbacks.radio_ = [&](auto const& r) { sink.onRadioTrack(r); };
        callbacks.radioWaveforms_ = [&](auto const& r, std::uint64_t n) {
          sink.onGpuRadioWaveforms(r, n);
        };
        callbacks.observation_ = [&](auto const& r) { sink.onObservation(r); };
        callbacks.discarded_ = [&](double x, unsigned) { discarded += x; };
        callbacks.promptDecay_ = [](auto const&) {};
        auto secondaries = make_sequence(p.productionProcess_, p.thinning_, cut);
        auto arrival = [&](auto& particle, auto const& request) {
          return cut.arrival(particle, request);
        };
        proposal::HadronicPhotonModel photons(d.photoLow_, d.photoHigh_,
                                              event.highEnergyHadronicThreshold_);
        egs4_api::HostHandler<stack_type, environment_type, decltype(photons),
                              decltype(secondaries), decltype(arrival)>
            host(g.environment_, topology, g.coordinates_, photons, secondaries, arrival);
        using fallback_type =
            ProposalCpuFallbackHandler<stack_type,
                                       std::remove_reference_t<decltype(p.emCascade_)>,
                                       decltype(sequence), environment_type>;
        fallback_type fallback(p.emCascade_, sequence, g.environment_, topology,
                               g.coordinates_, event.seed_, event.outputShowerId_);
        egs4_api::MuonOutputSink<decltype(sink)> muon_sink(sink);
        egs4_api::ProposalMuonBackend muon_adapter(*muon.backend_);
        using muon_router_type =
            EmRouter<stack_type, model_properties_type, fallback_type,
                     decltype(muon_sink), egs4_api::ProposalMuonBackend>;
        muon_router_type muon_router(muon_adapter, g.coordinates_,
                                     EnvironmentBorrowedView(treeData), geometry,
                                     topology, fallback, muon_sink);
        muon_router.setRetainRecords(false);
        auto region = [&](auto const* node) {
          if (!node || !node->hasModelProperties())
            throw std::runtime_error("EGS4 outside air");
          return environment.atmosphereLayers_[0].mediumId_;
        };
        using router_type =
            egs4_api::Router<stack_type, model_properties_type, decltype(host),
                             decltype(region), muon_router_type>;
        auto batch = std::min(options.gpuMinBatch_, std::size_t(capacity_ / 2));
        frontier::BufferedFrontierRouter<router_type, model_properties_type> router(
            frontier::ForwardRouterArguments{}, g.coordinates_, batch, *native_, cfg,
            callbacks, g.coordinates_, topology, host, region, batch,
            resolve_execution_backend(options.kokkosExecution_) == "openmp", &muon_router,
            &muon_adapter);
        HybridCascade<tracking_type, decltype(sequence),
                      std::remove_reference_t<decltype(e.output_)>, stack_type,
                      decltype(router)>
            cascade(g.environment_, e.tracking_, sequence, e.output_, e.stack_, router);
        router.setFrontierEnvironment(*snapshot);
        e.configurePrimary_(cascade);
        cascade.run();
        if (!e.stack_.isEmpty() || router.pending())
          throw std::runtime_error("EGS4 queues did not drain");
        report_.completed(event.outputShowerId_, router.statistics(),
                          cascade.timingStatistics(), discarded);
        previousThreshold_ = threshold;
        previousWeight_ = event.maximumWeight_;
      } catch (std::exception const& error) {
        report_.failed(event.outputShowerId_, error.what());
        throw;
      }
    }
  };
} // namespace corsika::kokkos::application
