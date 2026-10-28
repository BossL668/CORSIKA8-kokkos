/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::application {

  template <typename TMedium>
  template <typename TStack, typename TPrimary>
  CascadeParticleInput<TMedium>::CascadeParticleInput(
      CascadeWorkerOptions const& workers, BackendOptions const& options,
      [[maybe_unused]] Environment<TMedium>& env,
      [[maybe_unused]] CoordinateSystemPtr const& rootCS, TStack& stack,
      TPrimary const& primaryProperties)
      : workers_(workers)
      , options_(options) {
#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
    if ((!workers_.frontierIn_.empty() || !workers_.frontierOut_.empty() ||
         !workers_.cpuInput_.empty() || !workers_.cpuOutput_.empty()) &&
        options_.emBackend_ != "kokkos-egs4") {
      environment_ =
          std::make_unique<EnvironmentSnapshot<TMedium>>(*env.getUniverse(), rootCS);
      tree_ = bind_environment_data(*environment_);
    }
    if (!workers_.cpuInput_.empty()) {
      frontier::import_cpu_frontier(stack, rootCS, workers_.cpuInput_, workers_.workerId_,
                                    *environment_);
    } else if (!workers_.frontierIn_.empty()) {
      if (workers_.legacyImport_) {
        frontier::import_frontier(stack, rootCS, workers_.frontierIn_, workers_.workerId_,
                                  *environment_);
      } else {
        source_ = std::make_unique<frontier::FrontierInput>(workers_.frontierIn_);
        source_->setEnvironmentTree(tree_);
        source_->reserveNamespace(stack, workers_.workerId_);
      }
    } else
#endif
      stack.addParticle(primaryProperties);
#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
    scope_.emplace(source_.get(), workers_.frontierOut_);
#endif
  }

  template <typename TMedium>
  template <typename TTracking, typename TSequence, typename TOutput, typename TStack,
            typename TConfigure>
  bool CascadeParticleInput<TMedium>::runPrefix(
      [[maybe_unused]] Environment<TMedium>& env,
      [[maybe_unused]] CoordinateSystemPtr const& rootCS,
      [[maybe_unused]] TTracking& tracking, [[maybe_unused]] TSequence& sequence,
      [[maybe_unused]] TOutput& output, [[maybe_unused]] TStack& stack,
      [[maybe_unused]] TConfigure& configure) {
#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
    if (!workers_.cpuOutput_.empty()) {
      frontier::CpuBootstrapRouter router(workers_.cpuOutput_, rootCS, *environment_,
                                          workers_.cpuMaximumEnergy_);
      HybridCascade<TTracking, TSequence, TOutput, TStack, decltype(router)> cascade(
          env, tracking, sequence, output, stack, router);
      cascade.run();
      return true;
    } else if (!workers_.frontierOut_.empty() && options_.emBackend_ != "kokkos-egs4") {
      frontier::CaptureRouter router(workers_.frontierOut_, rootCS,
                                     workers_.frontierMaximumEnergy_, *environment_,
                                     workers_.historyLimit_);
      HybridCascade<TTracking, TSequence, TOutput, TStack, decltype(router)> cascade(
          env, tracking, sequence, output, stack, router);
      configure(cascade);
      cascade.run();
      return true;
    }
#endif
    return false;
  }

  template <typename TCoreas, typename TZhs>
  CascadeRadioCapture<TCoreas, TZhs>::CascadeRadioCapture(
      [[maybe_unused]] CascadeWorkerOptions const& workers,
      [[maybe_unused]] TCoreas& coreas, [[maybe_unused]] TZhs& zhs,
      [[maybe_unused]] CoordinateSystemPtr const& rootCS) {
#if defined(CORSIKA_KOKKOS_STATIC_MULTIGPU) && defined(CORSIKA8_WITH_KOKKOS_EM)
    if (!workers.radioOutput_.empty()) {
      spool_ = std::make_unique<radio::RadioTrackSpoolWriter>(workers.radioOutput_);
      sink_.emplace(coreas, zhs, *spool_, rootCS);
    }
#endif
  }

  template <typename TCoreas, typename TZhs>
  void CascadeRadioCapture<TCoreas, TZhs>::complete() {
#if defined(CORSIKA_KOKKOS_STATIC_MULTIGPU) && defined(CORSIKA8_WITH_KOKKOS_EM)
    if (spool_) {
      spool_->complete();
      sink_.reset();
      CORSIKA_LOG_INFO("Deferred {} CPU-prefix radio segments to GPU workers",
                       spool_->count());
    }
#endif
  }

  template <typename TG, typename TO, typename TM, typename TD>
  CascadeDriver<TG, TO, TM, TD>::CascadeDriver(
      CascadeRuntime& runtime, BackendOptions const& options,
      CascadeWorkerOptions const& workers, [[maybe_unused]] CascadeEventConfig defaults,
      TG geometry, [[maybe_unused]] TO outputs, TM models, TD monitors)
      : runtime_(runtime)
      , options_(options)
      , workers_(workers)
      , geometry_(geometry)
      , models_(models)
      , monitors_(monitors)
#ifdef CORSIKA8_WITH_KOKKOS_EM
      , accelerated_(runtime.session(), options, defaults, geometry, outputs, models,
                     monitors)
#endif
  {
  }

  template <typename TG, typename TO, typename TM, typename TD>
  template <typename TOutput>
  void CascadeDriver<TG, TO, TM, TD>::registerBackendOutput([
      [maybe_unused]] TOutput& output) {
#ifdef CORSIKA8_WITH_KOKKOS_EM
    accelerated_.registerBackendOutput(output);
#endif
  }

  template <typename TG, typename TO, typename TM, typename TD>
  auto CascadeDriver<TG, TO, TM, TD>::captureRadio() {
    return CascadeRadioCapture{workers_, models_.coreas_, models_.zhs_,
                               geometry_.coordinates_};
  }

  template <typename TG, typename TO, typename TM, typename TD>
  template <typename TH, typename TL>
  auto CascadeDriver<TG, TO, TM, TD>::captureDiagnostics(TH const& high,
                                                         TL const& low) const {
#ifdef CORSIKA8_WITH_KOKKOS_EM
    return accelerated_.captureDiagnostics(high, low);
#else
    return std::make_pair(high, low);
#endif
  }

  template <typename TG, typename TO, typename TM, typename TD>
  template <typename TStack, typename TPrimary>
  auto CascadeDriver<TG, TO, TM, TD>::prepareParticles(TStack& stack,
                                                       TPrimary const& primary) {
    return CascadeParticleInput{
        workers_, options_, geometry_.environment_, geometry_.coordinates_,
        stack,    primary};
  }

  template <typename TG, typename TO, typename TM, typename TD>
  template <typename TTracking, typename TSequence, typename TOutput, typename TStack,
            typename TInspector, typename TThinning, typename TInput, typename TBefore,
            typename TConfigure>
  auto CascadeDriver<TG, TO, TM, TD>::cascade(TTracking& tracking, TSequence& sequence,
                                              TOutput& output, TStack& stack,
                                              TInspector& inspector, TThinning& thinning,
                                              TInput& input, std::uint64_t ordinal,
                                              HEPEnergyType energy, double maximumWeight,
                                              TBefore const& before,
                                              TConfigure& configurePrimary) {
    auto const output_shower_id = static_cast<unsigned int>(output.getEventId());
    return CascadeAdapter{[this, &tracking, &sequence, &output, &stack, &inspector,
                           &thinning, &input, ordinal, energy, maximumWeight, &before,
                           &configurePrimary, output_shower_id](auto& force) {
      auto configure = [&](auto& cascade) {
        configurePrimary(cascade);
        force(cascade);
      };
      if (input.runPrefix(geometry_.environment_, geometry_.coordinates_, tracking,
                          sequence, output, stack, configure))
        return;
      if (options_.emBackend_ == "proposal") {
        Cascade EAS(geometry_.environment_, tracking, sequence, output, stack);
        configure(EAS);
        EAS.run();
        auto const photo_hadronic_after = monitors_.photoHigh_.statistics();
        auto const photo_hadronic_le_after = monitors_.photoLow_.statistics();
#ifdef CORSIKA8_WITH_KOKKOS_EM
        auto const& photo_hadronic_before = before.photoHigh_;
        auto const& photo_hadronic_le_before = before.photoLow_;
#else
        auto const& photo_hadronic_before = before.first;
        auto const& photo_hadronic_le_before = before.second;
#endif
        CORSIKA_LOG_INFO(
            "PROPOSAL photo-hadronic generators: SOPHIA={}, threshold fallback={}, "
            "HE preferred={}, HE fallback={}, QGSJet-II initialized={}",
            photo_hadronic_le_after.preferredInteractions_ -
                photo_hadronic_le_before.preferredInteractions_,
            photo_hadronic_le_after.fallbackInteractions_ -
                photo_hadronic_le_before.fallbackInteractions_,
            photo_hadronic_after.preferredInteractions_ -
                photo_hadronic_before.preferredInteractions_,
            photo_hadronic_after.fallbackInteractions_ -
                photo_hadronic_before.fallbackInteractions_,
            monitors_.fallback_.initialized());
      } else {
#ifdef CORSIKA8_WITH_KOKKOS_EM
        try {
          auto EAS =
              accelerated_.cascade(tracking, sequence, output, stack, inspector, thinning,
                                   ordinal, energy, maximumWeight, before);
          configure(EAS);
          EAS.run();
        } catch (std::exception const& error) {
          runtime_.recordIncomplete(output_shower_id, error.what());
          if (output.showerInProgress()) { output.endOfShower(); }
          output.endOfLibrary();
          CORSIKA_LOG_CRITICAL(
              "Accelerated EM shower {} aborted; output is marked incomplete: {}",
              ordinal, error.what());
          throw CascadeAborted{};
        }
#endif
      }
    }};
  }
} // namespace corsika::kokkos::application
