/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::application {

  template <typename TGeometry, typename TOutputs, typename TModels, typename TMonitors>
  CascadeApplication<TGeometry, TOutputs, TModels, TMonitors>::CascadeApplication(
      CascadeRunSession& session, BackendOptions const& options,
      CascadeEventConfig defaults, TGeometry geometry, TOutputs outputs, TModels models,
      TMonitors monitors)
      : session_(session)
      , options_(options)
      , defaults_(std::move(defaults))
      , geometry_(geometry)
      , outputs_(outputs)
      , models_(models)
      , monitors_(monitors) {
#ifdef CORSIKA8_WITH_NATIVE_EGS4
    if (options.emBackend_ == "kokkos-egs4")
      egs4_ = std::make_unique<Egs4RunSession>(options);
#endif
  }

  template <typename TGeometry, typename TOutputs, typename TModels, typename TMonitors>
  template <class TOutput>
  void CascadeApplication<TGeometry, TOutputs, TModels, TMonitors>::registerBackendOutput(
      [[maybe_unused]] TOutput& output) {
#ifdef CORSIKA8_WITH_NATIVE_EGS4
    if (egs4_) egs4_->registerOutput(output);
#endif
  }

  template <typename TGeometry, typename TOutputs, typename TModels, typename TMonitors>
  template <typename TH, typename TL>
  auto CascadeApplication<TGeometry, TOutputs, TModels, TMonitors>::captureDiagnostics(
      TH const& high, TL const& low) const {
    return monitors_.capture(high, low);
  }

  template <typename TGeometry, typename TOutputs, typename TModels, typename TMonitors>
  template <typename TTracking, typename TSequence, typename TOutput, typename TStack,
            typename TInspector, typename TThinning, typename TBefore>
  auto CascadeApplication<TGeometry, TOutputs, TModels, TMonitors>::cascade(
      TTracking& tracking, TSequence& sequence, TOutput& output, TStack& stack,
      TInspector& inspector, TThinning& thinning, std::uint64_t ordinal,
      HEPEnergyType energy, double maximum_weight, TBefore const& before) {
    auto event = defaults_;
    event.showerOrdinal_ = ordinal;
    event.outputShowerId_ = static_cast<unsigned int>(output.getEventId());
    event.primaryTotalEnergy_ = energy;
    event.maximumWeight_ = maximum_weight;
    event.thinningCanActivateFromUnitWeight_ =
        event.emThinningFraction_ > 0. && maximum_weight > 1.;
    return CascadeAdapter{[this, event, before, &tracking, &sequence, &output, &stack,
                           &inspector, &thinning](auto& configure) {
      auto const& m = models_;
      auto const& d = monitors_;
      CascadeContext const context{
          geometry_, outputs_,
          CascadeProcesses{m.coreas_, m.zhs_, m.emCascade_, m.emContinuousProposal_,
                           inspector, m.neutrinoPrimary_, m.hadronSequence_,
                           m.decaySequence_, m.emContinuous_, m.longitudinal_,
                           m.production_, thinning, m.cut_},
          CascadeExecution{sequence, tracking, stack, output, d.pool_, configure},
          CascadeDiagnostics{d.photoHigh_, d.photoLow_, before.photoHigh_,
                             before.photoLow_, d.high_, d.low_, before.highInteractions_,
                             before.lowInteractions_, before.highTimings_,
                             before.lowTimings_, before.pool_, d.fallback_}};
#ifdef CORSIKA8_WITH_NATIVE_EGS4
      if (egs4_)
        egs4_->run(options_, event, context);
      else
#endif
        run_cascade(session_, options_, event, context);
    }};
  }
} // namespace corsika::kokkos::application
