/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4Interactions.inl>

namespace corsika::kokkos::egs4 {
  // Native C7 photonuclear arithmetic. These are pre-cut final-state kernels,
  // NOT a SOPHIA replacement or a complete nuclear transport handler.
  struct PhotonuclearMasses {
    double protonMeV_{}, neutronMeV_{}, neutralPionMeV_{}, chargedPionMeV_{};
    double phiMeV_{}, omegaMeV_{}, rhoMeV_{};
  };
  KOKKOS_INLINE_FUNCTION bool valid_photonuclear_masses(PhotonuclearMasses m) {
    return finite(m.protonMeV_) && finite(m.neutronMeV_) && finite(m.neutralPionMeV_) &&
           finite(m.chargedPionMeV_) && m.protonMeV_ > 0. && m.neutronMeV_ > 0. &&
           m.neutralPionMeV_ > 0. && m.chargedPionMeV_ > 0.;
  }
  // Immutable after preparation: C7's first-call SAVE/COMMON table becomes
  // an explicit value copied into device memory. Source DATA C(12)=.1 GeV,
  // C(34)=20, DX=.5. No shared mutation in the sampling kernel.
  struct TransverseMomentumTable {
    double cumulative_[51]{}, envelope_[51]{}, peakGeV_{};
    int intervals_{};
  };
  KOKKOS_INLINE_FUNCTION TransverseMomentumTable
  make_transverse_momentum_table(double peak_GeV = .1, double cutoff = 20.) {
    TransverseMomentumTable t;
    if (!finite(peak_GeV) || peak_GeV <= 0. || !finite(cutoff) || cutoff < .5 ||
        cutoff > 25.)
      return t;
    t.peakGeV_ = peak_GeV;
    t.intervals_ = int(cutoff / .5);
    t.envelope_[0] = .5 * ::exp(.5);
    for (int i = 1; i <= t.intervals_; ++i) {
      double x = i * .5;
      if (x < 1.) x += .5;
      t.envelope_[i] = x * ::exp(1. - x);
      t.cumulative_[i] = t.cumulative_[i - 1] + t.envelope_[i - 1];
    }
    double inverse = 1. / t.cumulative_[t.intervals_];
    for (int i = 1; i <= t.intervals_; ++i) t.cumulative_[i] *= inverse;
    return t;
  }
  struct TransverseMomentum {
    InteractionStatus status_{InteractionStatus::invalid_input};
    double GeV_{};
    int trials_{};
  };
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION TransverseMomentum sample_transverse_momentum(
      TransverseMomentumTable const& t, TRandom& rng, int max_trials = 4096) {
    TransverseMomentum result;
    if (t.intervals_ < 1 || t.intervals_ > 50 || !finite(t.peakGeV_) ||
        t.peakGeV_ <= 0. || max_trials < 1)
      return result;
    for (int trial = 1; trial <= max_trials; ++trial) {
      result.trials_ = trial;
      double u, v;
      if (!uniform(rng, u) || !uniform(rng, v)) {
        result.status_ = InteractionStatus::random_failure;
        return result;
      }
      int i = 1;
      while (i < t.intervals_ && t.cumulative_[i] < u) ++i;
      double width = t.cumulative_[i] - t.cumulative_[i - 1];
      if (!finite(width) || width <= 0. || t.cumulative_[i] < u ||
          !finite(t.envelope_[i - 1]) || t.envelope_[i - 1] <= 0.)
        return result;
      double x = ((u - t.cumulative_[i - 1]) / width + i - 1) * .5;
      if (v * t.envelope_[i - 1] > x * ::exp(1. - x)) continue;
      result.GeV_ = x * t.peakGeV_;
      result.status_ = InteractionStatus::success;
      return result;
    }
    result.status_ = InteractionStatus::trial_limit;
    return result;
  }
  struct SinglePionOutcome {
    Secondaries secondaries_; // pion first, recoil second; NOT energy-sorted
    int targetPdg_{};
    double targetRestEnergyMeV_{}; // included in outgoing total energy
  };
  // Separate providers preserve the C7 stream distinction: PIGEN1 and UPHI
  // use stream 2; PTRANS uses stream 1. A production per-history policy may
  // supply explicit substreams. There is no mutable process-global RNG here.
  template <class TEmRandom, class THadronRandom>
  KOKKOS_INLINE_FUNCTION SinglePionOutcome
  sample_single_pion(double energy, Direction direction, PhotonuclearMasses masses,
                     TransverseMomentumTable const& pt_table, TEmRandom& em_rng,
                     THadronRandom& hadron_rng, int max_trials = 4096) {
    SinglePionOutcome out;
    auto& r = out.secondaries_;
    if (!finite(energy) || energy <= 0. || !valid_direction(direction) ||
        !valid_photonuclear_masses(masses) || max_trials < 1)
      return out;
    double u, v;
    if (!uniform(em_rng, u) || !uniform(em_rng, v)) {
      r.status_ = InteractionStatus::random_failure;
      return out;
    }
    bool proton = u <= .49923, exchange = v <= .3333333;
    out.targetPdg_ = proton ? 2212 : 2112;
    double m2 = proton ? masses.protonMeV_ : masses.neutronMeV_;
    double m3 = exchange ? masses.chargedPionMeV_ : masses.neutralPionMeV_;
    double m4 = (proton != exchange) ? masses.protonMeV_ : masses.neutronMeV_;
    out.targetRestEnergyMeV_ = m2;
    int pion = exchange ? (proton ? 211 : -211) : 111,
        recoil = (proton != exchange) ? 2212 : 2112;
    double m2i = 1. / m2, w0 = energy + m2, w0i = 1. / w0,
           w0s = ::sqrt(m2 * (m2 + 2. * energy)), w0si = 1. / w0s;
    double threshold = .5 * ((m3 + m4) * (m3 + m4) - m2 * m2) * m2i;
    if (energy <= threshold)
      return out; // forbidden channel, never invent an alternate reaction
    double beta = energy * w0i, gamma = w0 * w0si,
           ed = .5 * ((m3 - m4) * (m3 - m4) - m2 * m2) * m2i;
    double esq = ::sqrt((energy - threshold) * (energy - ed)), bratio = energy / esq;
    double g3 = w0i * bratio * (energy - threshold + m3 * m2i * (m3 + m4));
    double e3cm = g3 * m2 * gamma / bratio, p3cm = m2 * w0si * esq;
    double b3sq = p3cm * p3cm / (p3cm * p3cm + m3 * m3), b3 = ::sqrt(b3sq), ct = 0.;
    if (energy <= 1300.) {
      bool accepted = false;
      for (int trial = 1; trial <= max_trials; ++trial) {
        r.trials_ = trial;
        if (!uniform(em_rng, u) || !uniform(em_rng, v)) {
          r.status_ = InteractionStatus::random_failure;
          return out;
        }
        ct = exchange ? 1. / b3 - 1. / (u * 2. * b3sq / (1. - b3sq) + b3 / (1. + b3))
                      : 2. * u - 1.;
        double probability, scale;
        if (energy <= 900.) {
          probability = exchange ? 1. + ct * (-1.8 + ct * (.65 + ct * (.34 - .18 * ct)))
                                 : 1. - .6 * ct * ct;
          scale = exchange ? 2.5 : 1.;
        } else {
          probability =
              exchange
                  ? 1. +
                        ct *
                            (-2.18 +
                             ct * (7.20 +
                                   ct * (-2.55 +
                                         ct * (-15.39 +
                                               ct * (6.36 + ct * (13.80 - ct * 8.235))))))
                  : 1. + 6. * ct * ct - 5. * ((ct * ct) * (ct * ct));
          scale = exchange ? 13.2 : 2.8;
        }
        if (v * scale <= probability) {
          accepted = true;
          break;
        }
      }
      if (!accepted) {
        r.status_ = InteractionStatus::trial_limit;
        return out;
      }
    } else {
      auto pt = sample_transverse_momentum(pt_table, hadron_rng, max_trials);
      r.trials_ = pt.trials_;
      if (pt.status_ != InteractionStatus::success) {
        r.status_ = pt.status_;
        return out;
      }
      double momentum = 1.e3 * pt.GeV_;
      ct = ::sqrt(larger(0., (p3cm - momentum) * (p3cm + momentum))) / p3cm;
    }
    double e3 = gamma * (e3cm + beta * p3cm * ct);
    double p3 = ::sqrt(larger(0., (e3 - m3) * (e3 + m3)));
    double c3 = p3 > 0.
                    ? (m4 * m4 - m2 * m2 - m3 * m3 + 2. * e3 * w0 - 2. * energy * m2) /
                          (2. * energy * p3)
                    : 1.;
    double s3 = ::sqrt(larger(0., (1. - c3) * (1. + c3))), sp, cp;
    if (!azimuth(em_rng, sp, cp)) {
      r.status_ = InteractionStatus::random_failure;
      return out;
    }
    double e4 = w0 - e3, p4sq = (e4 - m4) * (e4 + m4);
    if (!finite(p4sq) || p4sq <= 0.) return out;
    double p4 = ::sqrt(p4sq),
           c4 = (m3 * m3 - m2 * m2 - m4 * m4 + 2. * e4 * w0 - 2. * energy * m2) /
                (2. * energy * p4);
    double s4 = -::sqrt(larger(0., (1. - c4) * (1. + c4)));
    r.particle_[0] = {e3, rotate(direction, s3, c3, sp, cp), pion};
    r.particle_[1] = {e4, rotate(direction, s4, c4, sp, cp), recoil};
    for (auto const& p : r.particle_)
      if (!finite(p.energyMeV_) || !valid_direction(p.direction_)) return out;
    r.count_ = 2;
    r.status_ = InteractionStatus::success;
    return out;
  }

  enum class PhotonuclearBranch {
    single_pion = 1,
    double_pion = 2,
    vector_meson = 3,
    many_hadrons = 4
  };
  struct PhotonuclearOutcome {
    InteractionStatus status_{InteractionStatus::invalid_input};
    PhotonuclearBranch branch_{PhotonuclearBranch::single_pion};
    Secondary particle_[3]{};
    int count_{}, trials_{}, targetPdg_{};
    double targetRestEnergyMeV_{};
  };
  // PIGEN2: original Dalitz rejection followed by the C7 transverse-momentum
  // prescription and CM->lab rotations. Return both pions and the recoil,
  // including subcut recoils: cut/deposit/stack bookkeeping is the caller's job.
  template <class TEmRandom, class THadronRandom>
  KOKKOS_INLINE_FUNCTION PhotonuclearOutcome
  sample_double_pion(double energy, Direction direction, PhotonuclearMasses masses,
                     TransverseMomentumTable const& pt_table, TEmRandom& em_rng,
                     THadronRandom& hadron_rng, int max_trials = 4096) {
    PhotonuclearOutcome out;
    out.branch_ = PhotonuclearBranch::double_pion;
    if (!finite(energy) || energy <= 0. || !valid_direction(direction) ||
        !valid_photonuclear_masses(masses) || max_trials < 1)
      return out;
    double u, v;
    if (!uniform(em_rng, u) || !uniform(em_rng, v)) {
      out.status_ = InteractionStatus::random_failure;
      return out;
    }
    bool proton = u <= .49923;
    out.targetPdg_ = proton ? 2212 : 2112;
    int ids[3];
    if (v <= .3) {
      ids[0] = ids[1] = 111;
      ids[2] = out.targetPdg_;
    } else if (v <= .6) {
      ids[0] = 211;
      ids[1] = -211;
      ids[2] = out.targetPdg_;
    } else {
      ids[0] = proton ? 211 : -211;
      ids[1] = 111;
      ids[2] = proton ? 2112 : 2212;
    }
    double m2 = proton ? masses.protonMeV_ : masses.neutronMeV_;
    double m3 = ids[0] == 111 ? masses.neutralPionMeV_ : masses.chargedPionMeV_;
    double m4 = ids[1] == 111 ? masses.neutralPionMeV_ : masses.chargedPionMeV_;
    double m5 = ids[2] == 2212 ? masses.protonMeV_ : masses.neutronMeV_;
    out.targetRestEnergyMeV_ = m2;
    double ecm = ::sqrt(m2 * (m2 + 2. * energy));
    if (ecm <= m3 + m4 + m5) return out;
    double a1 = (m3 + m4) * (m3 + m4), a2 = (ecm - m5) * (ecm - m5) - a1;
    double a3 = (m3 + m5) * (m3 + m5), a4 = (ecm - m4) * (ecm - m4) - a3;
    double a5 = (m3 - m4) * (m3 + m4), a6 = (ecm - m5) * (ecm + m5), a7 = .5 / ecm;
    double beta = energy / (m2 + energy), gamma = 2. * (energy + m2) * a7, am34sq = 0.,
           am35sq = 0.;
    bool accepted = false;
    for (int trial = 1; trial <= max_trials; ++trial) {
      out.trials_ = trial;
      if (!uniform(em_rng, u) || !uniform(em_rng, v)) {
        out.status_ = InteractionStatus::random_failure;
        return out;
      }
      am34sq = a2 * u + a1;
      am35sq = a4 * v + a3;
      double inverse = .5 / ::sqrt(am34sq);
      double e3star = (a5 + am34sq) * inverse, e5star = (a6 - am34sq) * inverse;
      double root1 = ::sqrt(larger(0., (e3star - m3) * (e3star + m3))),
             root2 = ::sqrt(larger(0., (e5star - m5) * (e5star + m5)));
      double discr = am35sq - (e3star + e5star) * (e3star + e5star);
      if (discr > -(root1 - root2) * (root1 - root2) ||
          discr < -(root1 + root2) * (root1 + root2))
        continue;
      accepted = true;
      break;
    }
    if (!accepted) {
      out.status_ = InteractionStatus::trial_limit;
      return out;
    }
    double e4 = (ecm * ecm + m4 * m4 - am35sq) * a7,
           e5 = (ecm * ecm + m5 * m5 - am34sq) * a7, e3 = ecm - e4 - e5;
    if (e4 > e3) {
      double temp = e3;
      e3 = e4;
      e4 = temp;
      temp = m3;
      m3 = m4;
      m4 = temp;
      int id = ids[0];
      ids[0] = ids[1];
      ids[1] = id;
    }
    double p3sq = (e3 - m3) * (e3 + m3), p4sq = (e4 - m4) * (e4 + m4),
           p5sq = (e5 - m5) * (e5 + m5);
    double p3 = ::sqrt(larger(0., p3sq)), p4 = ::sqrt(larger(0., p4sq)),
           p5 = ::sqrt(larger(0., p5sq));
    if (p3 <= 0. || p4 <= 0. || p5 <= 0.) return out;
    double ca = (p5sq - p3sq - p4sq) / (2. * p3 * p4),
           sa = -::sqrt(larger(0., (1. - ca) * (1. + ca)));
    double cb = (p4sq - p3sq - p5sq) / (2. * p3 * p5),
           sb = ::sqrt(larger(0., (1. - cb) * (1. + cb)));
    auto pt = sample_transverse_momentum(pt_table, hadron_rng, max_trials);
    out.trials_ += pt.trials_;
    if (pt.status_ != InteractionStatus::success) {
      out.status_ = pt.status_;
      return out;
    }
    double ratio = 1.e3 * pt.GeV_ / p3, s3 = ratio < 1. ? ratio : 1.,
           c3 = ::sqrt((1. - s3) * (1. + s3)), sp, cp;
    if (!azimuth(em_rng, sp, cp)) {
      out.status_ = InteractionStatus::random_failure;
      return out;
    }
    double elab3 = gamma * (e3 + beta * p3 * c3),
           p3lab = ::sqrt(larger(0., (elab3 - m3) * (elab3 + m3)));
    if (p3lab <= 0.) return out;
    double ct = (beta * e3 + p3 * c3) * gamma / p3lab;
    ct = ct < 1. ? ct : 1.;
    double phi_s, phi_c;
    if (!azimuth(em_rng, phi_s, phi_c)) {
      out.status_ = InteractionStatus::random_failure;
      return out;
    }
    out.particle_[0] = {
        elab3,
        rotate(direction, ::sqrt(larger(0., (1. - ct) * (1. + ct))), ct, phi_s, phi_c),
        ids[0]};
    double cosines[] = {ca, cb}, sines[] = {sa, sb}, energies[] = {e4, e5},
           momenta[] = {p4, p5}, rest[] = {m4, m5};
    for (int i = 0; i < 2; ++i) {
      double ccm = c3 * cosines[i] - s3 * cp * sines[i],
             elab = gamma * (energies[i] + beta * momenta[i] * ccm);
      double scm = ::sqrt(larger(0., (1. - ccm) * (1. + ccm))), cphi = 0., sphi = 1.;
      if (scm != 0.) {
        double inverse = 1. / scm, a = c3 * cp * sines[i] + s3 * cosines[i];
        cphi = (phi_c * a - phi_s * sp * sines[i]) * inverse;
        sphi = (phi_s * a + phi_c * sp * sines[i]) * inverse;
      }
      double plab = ::sqrt(larger(0., (elab - rest[i]) * (elab + rest[i])));
      if (plab <= 0.) return out;
      ct = (beta * energies[i] + momenta[i] * ccm) * gamma / plab;
      ct = ct < 1. ? ct : 1.;
      out.particle_[i + 1] = {
          elab,
          rotate(direction, ::sqrt(larger(0., (1. - ct) * (1. + ct))), ct, sphi, cphi),
          ids[i + 1]};
    }
    for (auto const& p : out.particle_)
      if (!finite(p.energyMeV_) || !valid_direction(p.direction_)) return out;
    out.count_ = 3;
    out.status_ = InteractionStatus::success;
    return out;
  }

  // RHOGEN returns the meson BEFORE its decay. In particular, C7's polarized
  // RHO0DC(1) must not be silently replaced by an isotropic/Pythia rho decay.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION PhotonuclearOutcome sample_vector_meson(
      double energy, Direction direction, PhotonuclearMasses masses, TRandom& rng) {
    PhotonuclearOutcome out;
    out.branch_ = PhotonuclearBranch::vector_meson;
    if (!finite(energy) || energy <= 0. || !valid_direction(direction) ||
        !valid_photonuclear_masses(masses) || !finite(masses.phiMeV_) ||
        masses.phiMeV_ <= 0. || !finite(masses.omegaMeV_) || masses.omegaMeV_ <= 0. ||
        !finite(masses.rhoMeV_) || masses.rhoMeV_ <= 0.)
      return out;
    double u, v, t;
    if (!uniform(rng, u) || !uniform(rng, v) || !uniform(rng, t)) {
      out.status_ = InteractionStatus::random_failure;
      return out;
    }
    bool proton = u <= .49923;
    out.targetPdg_ = proton ? 2212 : 2112;
    double m2 = proton ? masses.protonMeV_ : masses.neutronMeV_,
           m3 = v < .06 ? masses.phiMeV_ : v < .1 ? masses.omegaMeV_ : masses.rhoMeV_,
           m4 = m2;
    int pdg = v < .06 ? 333 : v < .1 ? 223 : 113;
    out.targetRestEnergyMeV_ = m2;
    double m2i = 1. / m2, m2sq = m2 * m2, w0s = ::sqrt(m2 * (m2 + 2. * energy)),
           w0si = 1. / w0s;
    double threshold = .5 * ((m3 + m4) * (m3 + m4) - m2sq) * m2i;
    if (energy <= threshold) return out;
    double a3 = .5 * m3 * m3 * w0si, e2 = .5 * (w0s + m2sq * w0si), e4 = e2 - a3;
    double pcm2 = ::sqrt((e2 - m2) * (e2 + m2)), pcm4 = ::sqrt((e4 - m2) * (e4 + m2));
    double tmin = a3 * a3 - (pcm2 + pcm4) * (pcm2 + pcm4),
           tmax = a3 * a3 - (pcm2 - pcm4) * (pcm2 - pcm4);
    double expmin = ::exp(8.e-6 * tmin),
           argument = t * (::exp(8.e-6 * tmax) - expmin) + expmin;
    if (!finite(argument) || argument <= 0.) return out;
    t = ::log(argument) / 8.e-6;
    double signed_plng = (e2 * e4 + .5 * t - m2sq) / pcm2;
    // Preserve RHOGEN's ABS(PLNG3) fold, but evaluate its two-body lab state
    // using invariants. Subtracting two ~1e11 MeV energies to obtain a ~1 GeV
    // recoil and reconstructing pT from 1-cos^2 loses useful transverse bits.
    // These are algebraic rearrangements of the SAME sampled t/CM angle.
    double effective_t = signed_plng < 0. ? tmin + tmax - t : t;
    double kinetic4 = -effective_t * .5 * m2i;
    double longitudinal_extra = (m3 * m3 - effective_t) / (2. * energy);
    double pz4 = kinetic4 + longitudinal_extra, pz3 = energy - pz4;
    double pt_sq = 2. * kinetic4 * (m2 - longitudinal_extra) -
                   longitudinal_extra * longitudinal_extra;
    if (!finite(pt_sq) || pt_sq < 0.) return out;
    double pt = ::sqrt(pt_sq), e3lab = energy - kinetic4, e4lab = m4 + kinetic4;
    double p3 = ::sqrt(larger(0., (e3lab - m3) * (e3lab + m3))),
           p4 = ::sqrt(larger(0., kinetic4 * (e4lab + m4)));
    if (p3 <= 0. || p4 <= 0.) return out;
    double sp, cp;
    if (!azimuth(rng, sp, cp)) {
      out.status_ = InteractionStatus::random_failure;
      return out;
    }
    out.particle_[0] = {e3lab, rotate(direction, pt / p3, pz3 / p3, sp, cp), pdg};
    out.particle_[1] = {e4lab, rotate(direction, -pt / p4, pz4 / p4, sp, cp),
                        out.targetPdg_};
    for (int i = 0; i < 2; ++i)
      if (!finite(out.particle_[i].energyMeV_) ||
          !valid_direction(out.particle_[i].direction_))
        return out;
    out.count_ = 2;
    out.status_ = InteractionStatus::success;
    return out;
  }
  struct PhotonuclearSelection {
    InteractionStatus status_{InteractionStatus::invalid_input};
    PhotonuclearBranch branch_{PhotonuclearBranch::single_pion};
  };
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION PhotonuclearSelection
  select_photonuclear_branch(double energy, PhotonuclearMasses masses, TRandom& rng) {
    PhotonuclearSelection out;
    if (!finite(energy) || energy <= 0. || !valid_photonuclear_masses(masses)) return out;
    double u;
    if (!uniform(rng, u)) {
      out.status_ = InteractionStatus::random_failure;
      return out;
    }
    if (u > (energy - 400.) * .001)
      out.branch_ = PhotonuclearBranch::single_pion;
    else if (u > (energy - 2000.) * .001)
      out.branch_ = PhotonuclearBranch::double_pion;
    else {
      double target = .5 * (masses.neutronMeV_ * .001 + masses.protonMeV_ * .001);
      double ecm = ::sqrt(target * (target + 2. * energy * .001));
      if (!finite(ecm) || ecm <= 0.) return out;
      // The first exponent is a REAL literal in C7, unlike the second one.
      double fraction =
          .17560 * ::pow(ecm, double(.037303f)) + .68008 / ::pow(ecm, 1.3021);
      if (!finite(fraction)) return out;
      if (!uniform(rng, u)) {
        out.status_ = InteractionStatus::random_failure;
        return out;
      }
      out.branch_ = u < fraction ? PhotonuclearBranch::vector_meson
                                 : PhotonuclearBranch::many_hadrons;
    }
    out.status_ = InteractionStatus::success;
    return out;
  }
  template <class TEmRandom, class THadronRandom>
  KOKKOS_INLINE_FUNCTION PhotonuclearOutcome
  sample_photonuclear(double energy, Direction direction, PhotonuclearMasses masses,
                      TransverseMomentumTable const& table, TEmRandom& em,
                      THadronRandom& hadron, int max_trials = 4096) {
    PhotonuclearOutcome out;
    if (!valid_direction(direction) || max_trials < 1) return out;
    auto selected = select_photonuclear_branch(energy, masses, em);
    out.status_ = selected.status_;
    out.branch_ = selected.branch_;
    if (selected.status_ != InteractionStatus::success) return out;
    if (selected.branch_ == PhotonuclearBranch::single_pion) {
      auto r =
          sample_single_pion(energy, direction, masses, table, em, hadron, max_trials);
      out.status_ = r.secondaries_.status_;
      out.count_ = r.secondaries_.count_;
      out.trials_ = r.secondaries_.trials_;
      out.targetPdg_ = r.targetPdg_;
      out.targetRestEnergyMeV_ = r.targetRestEnergyMeV_;
      for (int i = 0; i < 2; ++i) out.particle_[i] = r.secondaries_.particle_[i];
      return out;
    }
    if (selected.branch_ == PhotonuclearBranch::double_pion)
      return sample_double_pion(energy, direction, masses, table, em, hadron, max_trials);
    if (selected.branch_ == PhotonuclearBranch::vector_meson)
      return sample_vector_meson(energy, direction, masses, em);
    out.status_ =
        InteractionStatus::host_required; // C7 SDPM policy must be supplied explicitly
    return out;
  }
} // namespace corsika::kokkos::egs4
