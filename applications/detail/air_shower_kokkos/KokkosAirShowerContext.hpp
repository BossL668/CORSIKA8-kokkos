/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace corsika::applications::air_shower {

  // Non-owning, per-event bindings to application-owned objects. Every referred
  // object must outlive the run call. No model, writer, stack or callback is
  // copied. Only scalar configuration and the magnetic-field array are owned.
  // Diagnostics bind the snapshots already captured at the event boundary;
  // constructing a context never resamples counters or consumes random numbers.
  // C++17 deduction guides keep concrete process types out of the main program.

  template <typename Environment, typename Coordinates, typename Position,
      typename Surface, typename Step, typename Axis, typename Depth>
  struct KokkosAirShowerGeometry {
    Environment& environment;
    Coordinates const& coordinates;
    Position const& injection_position;
    Surface const& surface;
    Step const& propagation_step;
    Axis const& shower_axis;
    Depth const& depth_step;
    double observation_height_m;
    double core_x_m;
    double core_y_m;
    std::array<double, 3> magnetic_field_T;
  };

  template <typename Environment, typename Coordinates, typename Position,
      typename Surface, typename Step, typename Axis, typename Depth>
  KokkosAirShowerGeometry(Environment&, Coordinates const&, Position const&,
      Surface const&, Step const&, Axis const&, Depth const&, double, double, double,
      std::array<double, 3>)
      -> KokkosAirShowerGeometry<Environment, Coordinates, Position, Surface, Step, Axis,
      Depth>;

  template <typename EnergyLoss, typename Longitudinal, typename Production,
      typename Observation, typename Interactions, typename CoreasDetector,
      typename ZhsDetector>
  struct KokkosAirShowerOutputs {
    EnergyLoss& energy_loss;
    Longitudinal& longitudinal;
    Production& production;
    Observation& observation;
    Interactions& interactions;
    CoreasDetector& coreas_detector;
    ZhsDetector& zhs_detector;
  };

  template <typename EnergyLoss, typename Longitudinal, typename Production,
      typename Observation, typename Interactions, typename CoreasDetector,
      typename ZhsDetector>
  KokkosAirShowerOutputs(EnergyLoss&, Longitudinal&, Production&, Observation&,
      Interactions&, CoreasDetector&, ZhsDetector&)
      -> KokkosAirShowerOutputs<EnergyLoss, Longitudinal, Production, Observation,
      Interactions, CoreasDetector, ZhsDetector>;

  template <typename Coreas, typename Zhs, typename EmCascade,
      typename EmContinuousProposal, typename Inspector, typename Neutrino,
      typename Hadron, typename Decay, typename EmContinuous, typename Longitudinal,
      typename Production, typename Thinning, typename Cut>
  struct KokkosAirShowerProcesses {
    Coreas& coreas;
    Zhs& zhs;
    EmCascade& em_cascade;
    EmContinuousProposal& em_continuous_proposal;
    Inspector& stack_inspector;
    Neutrino& neutrino_primary;
    Hadron& hadron_sequence;
    Decay& decay_sequence;
    EmContinuous& em_continuous;
    Longitudinal& longitudinal_process;
    Production& production_process;
    Thinning& thinning;
    Cut& cut;
  };

  template <typename Coreas, typename Zhs, typename EmCascade,
      typename EmContinuousProposal, typename Inspector, typename Neutrino,
      typename Hadron, typename Decay, typename EmContinuous, typename Longitudinal,
      typename Production, typename Thinning, typename Cut>
  KokkosAirShowerProcesses(Coreas&, Zhs&, EmCascade&, EmContinuousProposal&, Inspector&,
      Neutrino&, Hadron&, Decay&, EmContinuous&, Longitudinal&, Production&, Thinning&,
      Cut&)
      -> KokkosAirShowerProcesses<Coreas, Zhs, EmCascade, EmContinuousProposal,
      Inspector, Neutrino, Hadron, Decay, EmContinuous, Longitudinal, Production,
      Thinning, Cut>;

  template <typename Sequence, typename Tracking, typename Stack, typename Output,
      typename Pool, typename ConfigurePrimary>
  struct KokkosAirShowerExecution {
    Sequence& sequence;
    Tracking& tracking;
    Stack& stack;
    Output& output;
    Pool* hadronic_pool;
    ConfigurePrimary& configure_primary;
  };

  template <typename Sequence, typename Tracking, typename Stack, typename Output,
      typename Pool, typename ConfigurePrimary>
  KokkosAirShowerExecution(Sequence&, Tracking&, Stack&, Output&, Pool*, ConfigurePrimary&)
      -> KokkosAirShowerExecution<Sequence, Tracking, Stack, Output, Pool,
      ConfigurePrimary>;

  template <typename PhotoHigh, typename PhotoLow, typename PhotoHighStats,
      typename PhotoLowStats, typename HighCounter, typename LowCounter,
      typename PoolStats, typename PhotoFallback>
  struct KokkosAirShowerDiagnostics {
    PhotoHigh& photo_high;
    PhotoLow& photo_low;
    PhotoHighStats const& photo_high_before;
    PhotoLowStats const& photo_low_before;
    HighCounter& high_counter;
    LowCounter& low_counter;
    std::uint64_t high_interactions_before;
    std::uint64_t low_interactions_before;
    std::size_t high_timings_before;
    std::size_t low_timings_before;
    PoolStats const& pool_before;
    PhotoFallback const& photo_fallback;
  };

  template <typename PhotoHigh, typename PhotoLow, typename PhotoHighStats,
      typename PhotoLowStats, typename HighCounter, typename LowCounter,
      typename PoolStats, typename PhotoFallback>
  KokkosAirShowerDiagnostics(PhotoHigh&, PhotoLow&, PhotoHighStats const&,
      PhotoLowStats const&, HighCounter&, LowCounter&, std::uint64_t, std::uint64_t,
      std::size_t, std::size_t, PoolStats const&, PhotoFallback const&)
      -> KokkosAirShowerDiagnostics<PhotoHigh, PhotoLow, PhotoHighStats, PhotoLowStats,
      HighCounter, LowCounter, PoolStats, PhotoFallback>;

  template <typename Geometry, typename Outputs, typename Processes,
            typename Execution, typename Diagnostics>
  struct KokkosAirShowerContext {
    Geometry geometry;
    Outputs outputs;
    Processes processes;
    Execution execution;
    Diagnostics diagnostics;
  };

  template <typename G, typename O, typename P, typename E, typename D>
  KokkosAirShowerContext(G, O, P, E, D) -> KokkosAirShowerContext<G, O, P, E, D>;

  namespace detail {
    // The single, tested mapping to the established positional interface.
    // Keep the legacy computation unchanged, including its exception behavior.
    template <typename Context, typename Function>
    decltype(auto) withKokkosAirShowerArguments(Context const& context,
                                               Function&& function) {
      auto const& g = context.geometry;
      auto const& o = context.outputs;
      auto const& p = context.processes;
      auto const& e = context.execution;
      auto const& d = context.diagnostics;
      return std::forward<Function>(function)(
          g.environment, g.coordinates, g.injection_position, g.surface,
          g.propagation_step, o.coreas_detector, o.zhs_detector, g.shower_axis,
          g.depth_step, g.observation_height_m, g.core_x_m, g.core_y_m,
          g.magnetic_field_T, o.energy_loss, o.longitudinal, o.production,
          o.observation, o.interactions, p.coreas, p.zhs, p.em_cascade,
          p.em_continuous_proposal, p.stack_inspector, p.neutrino_primary,
          p.hadron_sequence, p.decay_sequence, p.em_continuous,
          p.longitudinal_process, p.production_process, p.thinning, p.cut,
          e.sequence, e.tracking, e.stack, e.output, e.hadronic_pool,
          e.configure_primary, d.photo_high, d.photo_low, d.photo_high_before,
          d.photo_low_before, d.high_counter, d.low_counter,
          d.high_interactions_before, d.low_interactions_before,
          d.high_timings_before, d.low_timings_before, d.pool_before,
          d.photo_fallback);
    }
  } // namespace detail
} // namespace corsika::applications::air_shower
