/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::application {
  template <typename TD, typename TG, typename TM>
  CascadeEvent<TD, TG, TM>::CascadeEvent(TD& driver, BackendOptions const& options,
                                         CascadeWorkerOptions const& workers,
                                         TG const& geometry, TM monitors,
                                         std::uint64_t ordinal, int events,
                                         double thinning)
      : driver_(driver)
      , options_(options)
      , workers_(workers)
      , geometry_(geometry)
      , monitors_(monitors)
      , ordinal_(ordinal)
      , thinningFraction_(thinning)
      , radio_(driver.captureRadio())
      , before_([&]() {
        CORSIKA_LOG_INFO("Shower {} / {} ", ordinal, events);
        ReplayTrace::instance().beginShower(ordinal - 1);
        diagnostics::DecisionReplayTape::instance().beginShower(ordinal - 1);
        auto const low = monitors_.photoLow_.statistics();
        auto const high = monitors_.photoHigh_.statistics();
        return driver_.captureDiagnostics(high, low);
      }())
      , exceptions_(std::uncaught_exceptions()) {}

  template <typename TD, typename TG, typename TM>
  CascadeEvent<TD, TG, TM>::~CascadeEvent() {
    if (std::uncaught_exceptions() == exceptions_) {
      monitors_.low_.releaseTimingSamples();
      monitors_.high_.releaseTimingSamples();
    }
  }

  template <typename TD, typename TG, typename TM>
  template <typename TStack, typename TPrimary>
  void CascadeEvent<TD, TG, TM>::addPrimary(TStack& stack, TPrimary const& primary) {
    input_.emplace(workers_, options_, geometry_.environment_, geometry_.coordinates_,
                   stack, primary);
  }

  template <typename TD, typename TG, typename TM>
  template <typename TT, typename TS, typename TO, typename TK, typename TI, typename TH>
  auto CascadeEvent<TD, TG, TM>::cascade(TT& tracking, TS& sequence, TO& output,
                                         TK& stack, TI& inspector, TH& thinning,
                                         HEPEnergyType energy, double weight) {
    if (thinningFraction_ > 0. && weight <= 1.) {
      CORSIKA_LOG_WARN(
          "EM thinning is configured with threshold={} GeV, but "
          "maximum weight={} is not above the unit initial particle "
          "weight. EMThinning's parentWeight >= maxWeight guard "
          "prevents thinning from starting. Specify --max-weight > 1 "
          "or use a larger emthin*primary-energy product if thinning "
          "is intended.",
          thinningFraction_ * energy / 1_GeV, weight);
    }
    if (!input_)
      throw std::logic_error("Primary input must precede cascade construction");
    auto run = driver_.cascade(tracking, sequence, output, stack, inspector, thinning,
                               *input_, ordinal_, energy, weight, before_, configure_);
    return CascadeAdapter{[this, run = std::move(run)](auto& force) mutable {
      force(run);
      run.run();
      radio_.complete();
    }};
  }

  template <typename TG, typename TO, typename TM, typename TD>
  template <typename TOutput>
  CascadeModule<TG, TO, TM, TD>::CascadeModule(CascadeRuntime& runtime,
                                               BackendOptions const& options,
                                               CascadeWorkerOptions const& workers,
                                               CascadeEventConfig defaults, TG geometry,
                                               TO outputs, TM models, TD monitors,
                                               TOutput& output, int events)
      : options_(options)
      , workers_(workers)
      , geometry_(geometry)
      , monitors_(monitors)
      , defaults_(defaults)
      , events_(events)
      , timing_([&]() {
        YAML::Node config;
        config["em_backend"] = options.emBackend_;
        config["gpu_physics_source"] = options.emBackend_ == "kokkos-egs4"
                                           ? "c7-egs4-embedded"
                                           : options.gpuPhysicsSource_;
        config["radio_backend"] = options.radioBackend_;
        config["clock"] = "steady_clock";
        config["scope"] = "OutputManager startOfShower through endOfShower";
        return config;
      }()) {
    output.add("simulation_timing", timing_);
    if (!options.cudaReplayTapeOut_.empty()) {
      auto const snapshot = diagnostics::make_replay_radio_snapshot(
          geometry.environment_, geometry.injectionPosition_, geometry.surface_,
          geometry.propagationStep_, outputs.coreasDetector_, outputs.zhsDetector_);
      diagnostics::DecisionReplayTape::instance().open(options.cudaReplayTapeOut_,
                                                       snapshot, defaults.seed_, events);
    }
    driver_.emplace(runtime, options, workers, defaults, geometry, outputs, models,
                    monitors);
    driver_->registerBackendOutput(output);
  }

  template <typename TG, typename TO, typename TM, typename TD>
  CascadeModule<TG, TO, TM, TD>::~CascadeModule() {
    ReplayTrace::instance().close();
    diagnostics::DecisionReplayTape::instance().close();
  }

  template <typename TG, typename TO, typename TM, typename TD>
  auto CascadeModule<TG, TO, TM, TD>::shower(std::uint64_t ordinal) {
    return CascadeEvent<driver_type, TG, TD>{
        *driver_,  options_, workers_, geometry_,
        monitors_, ordinal,  events_,  defaults_.emThinningFraction_};
  }

  template <typename TE, typename TL, typename TH>
  auto CascadeExtension::interaction(TE const& env, TL& low, TH& high,
                                     HEPEnergyType threshold) {
    return proposal::AirShowerInteraction{env, low, high, threshold};
  }

  template <typename TM>
  auto CascadeExtension::interactionCounter(TM& model) {
#ifdef WITH_FLUKA
    return make_fluka_interaction_counter(model);
#else
    return InteractionCounter{model};
#endif
  }

  template <typename TA, typename TX>
  auto CascadeExtension::energyLossWriter(TA const& axis, TX const& depth) {
    return EnergyLossWriter{axis, depth, 0.0001_g / square(1_cm),
                            ShortTrackDepositMode::LowerGrammage};
  }

  template <typename TA, typename TX>
  auto CascadeExtension::longitudinalWriter(TA const& axis, TX const& depth) {
    auto const crossings = app_["--profile-crossings"]->as<std::string>();
    auto const mode = crossings == "original-c8"
                          ? ProfileCrossingMode::OriginalC8
                          : crossings == "forward" ? ProfileCrossingMode::Forward
                                                   : ProfileCrossingMode::Both;
    return LongitudinalWriter{axis, depth, mode};
  }

  template <typename TE, typename TC, typename TP, typename TS, typename TStep,
            typename TA, typename TX, typename TW, typename TLW, typename TPW,
            typename TObs, typename TIW, typename TDC, typename TDZ, typename TCoreas,
            typename TZhs, typename TEM, typename TProposal, typename TN, typename TH,
            typename TD, typename TCont, typename TL, typename TProd, typename TCut,
            typename THC, typename TLC, typename TOutput>
  auto CascadeExtension::bind(
      TE& env, TC const& coordinates, TP const& injection, TS const& surface,
      TStep const& step, TA const& axis, TX const& depth, std::uint64_t seed,
      HEPEnergyType maximumEnergy, Code primary, HEPEnergyType productionThreshold,
      HEPEnergyType hadronicThreshold, bool multithin, TW& energyLoss, TLW& profile,
      TPW& productionProfile, TObs& observation, TIW& interactions, TDC& coreasDetector,
      TDZ& zhsDetector, TCoreas& coreas, TZhs& zhs, TEM& em, TProposal& proposal,
      TN& neutrino, TH& hadrons, TD& decays, TCont& continuous, TL& longitudinal,
      TProd& production, TCut& cut, THC& highCounter, TLC& lowCounter, TOutput& output) {
    auto defaults =
        make_cascade_defaults(app_, seed, maximumEnergy, primary, cut,
                              productionThreshold, hadronicThreshold, multithin);
    defaults.scene_ = environment_.scene_;
    auto geometry = make_cascade_geometry(env, coordinates, injection, surface, step,
                                          axis, depth, environment_);
    auto outputs =
        CascadeOutputs{energyLoss,   profile,        productionProfile, observation,
                       interactions, coreasDetector, zhsDetector};
    auto models =
        CascadeModels{coreas, zhs,        em,           proposal,   neutrino, hadrons,
                      decays, continuous, longitudinal, production, cut};
    using pool_type = std::remove_pointer_t<decltype(runtime_.hadronicPool())>;
    boost::optional<pool_type&> pool;
    if (auto* nativePool = runtime_.hadronicPool()) pool = *nativePool;
    auto monitors = CascadeMonitors{
        em.highEnergyModel(), em.lowEnergyModel(), highCounter, lowCounter, pool,
        em.fallbackModel()};
    return CascadeModule<decltype(geometry), decltype(outputs), decltype(models),
                         decltype(monitors)>{
        runtime_, options_, workers_, defaults, geometry,
        outputs,  models,   monitors, output,   app_["--nevent"]->as<int>()};
  }
} // namespace corsika::kokkos::application
