/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace corsika::applications::air_shower {

  // Run-lifetime roles, bound once. Models are never copied or reconstructed.
  // The event-lifetime inspector and thinning process are supplied separately.
  template <typename C, typename Z, typename E, typename P, typename N, typename H,
            typename D, typename Continuous, typename L, typename Production, typename Cut>
  struct KokkosAirShowerModels {
    C& coreas;
    Z& zhs;
    E& em_cascade;
    P& em_continuous_proposal;
    N& neutrino_primary;
    H& hadron_sequence;
    D& decay_sequence;
    Continuous& em_continuous;
    L& longitudinal;
    Production& production;
    Cut& cut;
  };
  template <typename C, typename Z, typename E, typename P, typename N, typename H,
            typename D, typename Continuous, typename L, typename Production, typename Cut>
  KokkosAirShowerModels(C&, Z&, E&, P&, N&, H&, D&, Continuous&, L&, Production&, Cut&)
      -> KokkosAirShowerModels<C, Z, E, P, N, H, D, Continuous, L, Production, Cut>;

  template <typename High, typename Low, typename Pool>
  struct KokkosAirShowerBefore {
    High photo_high;
    Low photo_low;
    std::uint64_t low_interactions;
    std::uint64_t high_interactions;
    std::size_t low_timings;
    std::size_t high_timings;
    Pool pool;
  };

  template <typename PhotoHigh, typename PhotoLow, typename High, typename Low,
            typename Pool, typename Fallback>
  struct KokkosAirShowerMonitors {
    PhotoHigh& photo_high;
    PhotoLow& photo_low;
    High& high;
    Low& low;
    Pool* pool;
    Fallback const& fallback;

    template <typename H, typename L>
    auto capture(H const& photo_high_before, L const& photo_low_before) const {
      using PoolStats = std::decay_t<decltype(pool->statistics())>;
      // Braced initialization preserves the old low/high read order. Snapshots
      // are values: later counter changes cannot mutate the event boundary.
      return KokkosAirShowerBefore<H, L, PoolStats>{
          photo_high_before, photo_low_before, low.getCount(), high.getCount(),
          low.getTimingSamples().size(), high.getTimingSamples().size(),
          pool ? pool->statistics() : PoolStats{}};
    }
  };
  template <typename PhotoHigh, typename PhotoLow, typename High, typename Low,
            typename Pool, typename Fallback>
  KokkosAirShowerMonitors(PhotoHigh&, PhotoLow&, High&, Low&, Pool*, Fallback const&)
      -> KokkosAirShowerMonitors<PhotoHigh, PhotoLow, High, Low, Pool, Fallback>;

  // Stack-local facade with the same force/run vocabulary as Cascade. The
  // callback closes over references to event objects, whose scope contains it.
  template <typename Run>
  class KokkosCascade {
    Run run_;
    bool force_interaction_ = false;
    bool force_decay_ = false;
  public:
    explicit KokkosCascade(Run run) : run_(run) {}
    void forceInteraction() { force_interaction_ = true; }
    void forceDecay() { force_decay_ = true; }
    void run() {
      auto configure = [this](auto& cascade) {
        if (force_interaction_) cascade.forceInteraction();
        if (force_decay_) cascade.forceDecay();
      };
      run_(configure);
    }
  };
  template <typename Run> KokkosCascade(Run) -> KokkosCascade<Run>;
}
