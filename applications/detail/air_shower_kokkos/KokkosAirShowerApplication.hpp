/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include "KokkosAirShowerBindings.hpp"
#include "KokkosAirShowerRunner.hpp"
#include "KokkosRunSession.hpp"
#include <CLI/App.hpp>
#ifdef CORSIKA8_WITH_NATIVE_EGS4
#include "Egs4AirShowerRunner.hpp"
#endif

namespace corsika::applications::air_shower {

  // Read application options once. Cuts/thresholds are the actual objects used
  // by the native sequence, rather than a second set of independent defaults.
  template <typename Cut>
  KokkosEventConfig makeKokkosAirShowerDefaults(
      CLI::App const& app, std::uint64_t seed, HEPEnergyType maximum_energy,
      Code primary, Cut const& cut, HEPEnergyType production_threshold,
      HEPEnergyType hadronic_threshold, bool keep_zero_weight) {
    KokkosEventConfig result;
    result.seed = seed;
    result.configured_primary_energy_max = maximum_energy;
    result.em_cut = cut.getElectronKineticECut();
    result.muon_cut = cut.getMuonKineticECut();
    result.production_threshold = production_threshold;
    result.high_energy_hadronic_threshold = hadronic_threshold;
    result.em_thinning_fraction = app["--emthin"]->as<double>();
    result.automatic_maximum_weight = app["--max-weight"]->as<double>() <= 0.;
    result.keep_zero_weight_particles = keep_zero_weight;
    result.beam_code = primary;
    result.high_energy_hadronic_model = app["--hadronModel"]->as<std::string>();
    result.maximum_magnetic_deflection_rad = app["--max-deflection-angle"]->as<double>();
    return result;
  }

  template <typename E, typename C, typename P, typename S, typename Step,
            typename Axis, typename Depth>
  auto makeKokkosAirShowerGeometry(E& environment, C const& coordinates,
      P const& injection, S const& surface, Step const& step, Axis const& axis,
      Depth const& depth, AcceleratedRunEnvironmentConfig const& run) {
    return KokkosAirShowerGeometry{
        environment, coordinates, injection, surface, step, axis, depth,
        run.observation_plane_point_m[2], run.observation_plane_point_m[0],
        run.observation_plane_point_m[1], run.magnetic_field_T};
  }

  /**
   * Bind the application once; create a small stack-local cascade per shower.
   * This is an assembly adapter, not a second transport implementation. It
   * forwards to the established runner (including the multi-GPU frontier hooks).
   * All models/writers/session must outlive this adapter; each returned cascade
   * must run before its sequence, inspector, thinning and stack leave scope.
   */
  template <typename Geometry, typename Outputs, typename Models, typename Monitors>
  class KokkosAirShowerApplication {
    KokkosRunSession& session_;
    GpuCliOptions const& options_;
    KokkosEventConfig defaults_;
    Geometry geometry_;
    Outputs outputs_;
    Models models_;
    Monitors monitors_;
#ifdef CORSIKA8_WITH_NATIVE_EGS4
    std::unique_ptr<Egs4RunSession> egs4_;
#endif
  public:
    KokkosAirShowerApplication(KokkosRunSession& session, GpuCliOptions const& options,
        KokkosEventConfig defaults, Geometry geometry, Outputs outputs,
        Models models, Monitors monitors)
        : session_(session), options_(options), defaults_(std::move(defaults)),
          geometry_(geometry), outputs_(outputs), models_(models), monitors_(monitors) {
#ifdef CORSIKA8_WITH_NATIVE_EGS4
      if(options.em_backend=="kokkos-egs4")egs4_=std::make_unique<Egs4RunSession>(options);
#endif
    }

    template<class Output>void registerBackendOutput(Output& output) {
#ifdef CORSIKA8_WITH_NATIVE_EGS4
      if(egs4_)egs4_->registerOutput(output);
#endif
    }

    template <typename H, typename L>
    auto captureDiagnostics(H const& high, L const& low) const {
      return monitors_.capture(high, low);
    }

    template <typename Tracking, typename Sequence, typename Output, typename Stack,
              typename Inspector, typename Thinning, typename Before>
    auto cascade(Tracking& tracking, Sequence& sequence, Output& output, Stack& stack,
        Inspector& inspector, Thinning& thinning, std::uint64_t ordinal,
        HEPEnergyType energy, double maximum_weight, Before const& before) {
      auto event = defaults_;
      event.shower_ordinal = ordinal;
      event.output_shower_id = static_cast<unsigned int>(output.getEventId());
      event.primary_total_energy = energy;
      event.maximum_weight = maximum_weight;
      event.thinning_can_activate_from_unit_weight =
          event.em_thinning_fraction > 0. && maximum_weight > 1.;
      return KokkosCascade{
          [this, event, before, &tracking, &sequence, &output, &stack, &inspector,
           &thinning](auto& configure) {
            auto const& m = models_;
            auto const& d = monitors_;
            KokkosAirShowerContext const context{
                geometry_, outputs_,
                KokkosAirShowerProcesses{
                    m.coreas, m.zhs, m.em_cascade, m.em_continuous_proposal,
                    inspector, m.neutrino_primary, m.hadron_sequence, m.decay_sequence,
                    m.em_continuous, m.longitudinal, m.production, thinning, m.cut},
                KokkosAirShowerExecution{
                    sequence, tracking, stack, output, d.pool, configure},
                KokkosAirShowerDiagnostics{
                    d.photo_high, d.photo_low, before.photo_high, before.photo_low,
                    d.high, d.low, before.high_interactions, before.low_interactions,
                    before.high_timings, before.low_timings, before.pool, d.fallback}};
#ifdef CORSIKA8_WITH_NATIVE_EGS4
            if(egs4_)egs4_->run(options_,event,context);
            else
#endif
              runKokkosAirShower(session_, options_, event, context);
          }};
    }
  };
  template <typename G, typename O, typename M, typename D>
  KokkosAirShowerApplication(KokkosRunSession&, GpuCliOptions const&, KokkosEventConfig,
      G, O, M, D) -> KokkosAirShowerApplication<G, O, M, D>;
}
