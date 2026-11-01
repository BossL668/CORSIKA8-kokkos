/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/detail/kokkos/egs4/Egs4Interactions.inl>

namespace corsika::kokkos::egs4 {
  // Physical channels are explicit. In particular, the low-rate hadronic and
  // muon-pair branches must be handled by an adapter, never silently rerouted
  // to a PROPOSAL e/gamma reaction when the EGS4 backend is selected.
  enum class Channel {
    invalid,
    continue_transport,
    bremsstrahlung,
    moller,
    bhabha,
    annihilation,
    electronuclear,
    pair_production,
    compton,
    muon_pair,
    photonuclear,
    photoelectric
  };
  struct ChannelThresholds {
    double massMeV_{};
    double electronuclearMeV_{}; // C7 MAX(ELCUT(4)*1.e3, PITHR)
    double photonuclearMeV_{};   // C7 PITHR
    double muonPairMeV_{};       // C7 RMMUT4 + ELCUT(2)*1.e3
  };
  KOKKOS_INLINE_FUNCTION bool valid_thresholds(ChannelThresholds t) {
    return finite(t.massMeV_) && t.massMeV_ > 0. && finite(t.electronuclearMeV_) &&
           t.electronuclearMeV_ >= 0. && finite(t.photonuclearMeV_) &&
           t.photonuclearMeV_ >= 0. && finite(t.muonPairMeV_) && t.muonPairMeV_ >= 0.;
  }
  // Preserve the order and equality convention of ELECTR's final branch.
  // `ae` is the PEGS discrete-secondary threshold, not the transport cut.
  KOKKOS_INLINE_FUNCTION Channel electron_channel(ElectronQuery q, int charge,
                                                  double total, double ae,
                                                  ChannelThresholds t, double u) {
    if (q.status_ != Status::success || !finite(total) || total <= t.massMeV_ ||
        !finite(ae) || ae <= t.massMeV_ || !valid_thresholds(t) || !finite(u) || u < 0. ||
        u >= 1. || (charge != -1 && charge != 1))
      return Channel::invalid;
    for (int i = 0; i < (charge < 0 ? 2 : 3); ++i)
      if (!finite(q.branches_[i])) return Channel::invalid;
    // Do not clamp/re-normalize HATCH's interpolated branching coefficients.
    if (charge < 0) {
      if (u >= q.branches_[1]) return Channel::bremsstrahlung;
      if (u >= q.branches_[0]) {
        double thmoll = (ae - t.massMeV_) * 2. + t.massMeV_;
        if (total <= thmoll)
          return q.branches_[0] <= 0. ? Channel::continue_transport
                                      : Channel::bremsstrahlung;
        return Channel::moller;
      }
    } else {
      if (u >= q.branches_[2]) return Channel::bremsstrahlung;
      if (u >= q.branches_[1]) return Channel::bhabha;
      if (u >= q.branches_[0]) return Channel::annihilation;
    }
    return total < t.electronuclearMeV_ ? Channel::bremsstrahlung
                                        : Channel::electronuclear;
  }
  KOKKOS_INLINE_FUNCTION Channel photon_channel(PhotonQuery q, double energy,
                                                ChannelThresholds t, double u) {
    if (q.status_ != Status::success || !finite(energy) || energy <= 0. ||
        !valid_thresholds(t) || !finite(u) || u < 0. || u >= 1.)
      return Channel::invalid;
    for (int i = 0; i < 4; ++i)
      if (!finite(q.branches_[i])) return Channel::invalid;
    if (u >= q.branches_[3] && energy > 2. * t.massMeV_) return Channel::pair_production;
    if (u >= q.branches_[2]) return Channel::compton;
    if (u <= q.branches_[0] && energy > t.muonPairMeV_) return Channel::muon_pair;
    if (u <= q.branches_[1] && energy > t.photonuclearMeV_) return Channel::photonuclear;
    return Channel::photoelectric;
  }

  struct ElectronClock {
    double remainingMfp_{}, sampledRatePerCm_{};
    bool initialized_{};
  };
  struct PhotonClock {
    double remainingMfp_{};
    bool initialized_{};
  };
  enum class ClockStatus {
    invalid,
    transporting,
    collision_due,
    restart,
    accepted,
    random_failure
  };
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION ClockStatus start_photon_clock(PhotonQuery q, TRandom& rng,
                                                        PhotonClock& clock) {
    clock = {};
    if (q.status_ != Status::success || !finite(q.meanFreePathCm_) ||
        q.meanFreePathCm_ <= 0.)
      return ClockStatus::invalid;
    double u;
    if (!uniform(rng, u) || u == 0.) return ClockStatus::random_failure;
    clock.remainingMfp_ = larger(-::log(u), 1.e-14);
    clock.initialized_ = true;
    return ClockStatus::transporting;
  }
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION ClockStatus start_electron_clock(ElectronQuery q, TRandom& rng,
                                                          ElectronClock& clock) {
    clock = {};
    if (q.status_ != Status::success || !finite(q.ratePerCm_))
      return ClockStatus::invalid;
    double u;
    // RMMARD's production uniforms are nonzero. A zero supplied by an adapter
    // is an explicit error here, not a changed free-path distribution.
    if (!uniform(rng, u) || u == 0.) return ClockStatus::random_failure;
    clock.remainingMfp_ = larger(-::log(u), 1.e-14);
    clock.sampledRatePerCm_ = q.ratePerCm_;
    clock.initialized_ = true;
    return ClockStatus::transporting;
  }
  // TVSTEP is C7's density/path-corrected length in the region-reference medium.
  // It is NOT the geometric segment length and NOT an arbitrary C8 grammage.
  KOKKOS_INLINE_FUNCTION ClockStatus consume_electron_clock(ElectronClock& clock,
                                                            double tvstep,
                                                            double rhofac) {
    if (!clock.initialized_ || !finite(clock.remainingMfp_) ||
        !finite(clock.sampledRatePerCm_) || !finite(tvstep) || tvstep < 0. ||
        !finite(rhofac) || rhofac <= 0.)
      return ClockStatus::invalid;
    double sig = clock.sampledRatePerCm_ * rhofac;
    clock.remainingMfp_ = larger(0., clock.remainingMfp_ - tvstep * sig);
    return clock.remainingMfp_ >= 1.e-10 ? ClockStatus::transporting
                                         : ClockStatus::collision_due;
  }
  // The initial sigma is held fixed while continuous energy loss changes the
  // current energy. This is C7's fictitious-sigma check, not per-step resampling.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION ClockStatus accept_electron_collision(ElectronClock& clock,
                                                               ElectronQuery final,
                                                               TRandom& rng) {
    if (!clock.initialized_ || clock.remainingMfp_ >= 1.e-10 ||
        clock.sampledRatePerCm_ <= 0. || final.status_ != Status::success ||
        !finite(final.ratePerCm_))
      return ClockStatus::invalid;
    double u;
    if (!uniform(rng, u)) return ClockStatus::random_failure;
    auto status = u > final.ratePerCm_ / clock.sampledRatePerCm_ ? ClockStatus::restart
                                                                 : ClockStatus::accepted;
    clock.initialized_ = false; // caller must begin a new history after either branch
    return status;
  }
  // Use up-to-date dE/dx/TMXS but the originally sampled sigma for the local
  // step proposal. Merely querying the current sigma here changes C7's scheme.
  KOKKOS_INLINE_FUNCTION ElectronQuery with_sampled_rate(ElectronQuery current,
                                                         ElectronClock clock) {
    if (!clock.initialized_ || !finite(clock.sampledRatePerCm_))
      current.status_ = Status::invalid_input;
    else
      current.ratePerCm_ = clock.sampledRatePerCm_;
    return current;
  }
} // namespace corsika::kokkos::egs4
