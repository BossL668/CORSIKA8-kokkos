/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/detail/kokkos/egs4/Egs4Interactions.inl>
#include <climits>

namespace corsika::kokkos::egs4 {
  // Same REAL coefficients and ordering as HATCH's two BREMPR READ statements.
  // A small immutable value object; contains no particle/global mutable state.
  struct RadiativeTable {
    float dl_[6][6]{}, delcm_{}, alphi_[2]{}, bpar_[2]{}, delpos_[2]{};
    double photonCutMeV_{};
  };
  inline RadiativeTable radiative_table(Tables const& t) {
    if (t.bremsPairCoefficients_.size() != 43)
      throw std::runtime_error("Incomplete BREMPR coefficients");
    RadiativeTable r;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) r.dl_[i][j] = t.bremsPairCoefficients_[6 * i + j];
    r.delcm_ = t.bremsPairCoefficients_[36];
    r.photonCutMeV_ = t.medium_.ap_;
    for (int i = 0; i < 2; ++i) {
      r.alphi_[i] = t.bremsPairCoefficients_[37 + 3 * i];
      r.bpar_[i] = t.bremsPairCoefficients_[38 + 3 * i];
      r.delpos_[i] = t.bremsPairCoefficients_[39 + 3 * i];
    }
    return r;
  }
  KOKKOS_INLINE_FUNCTION double screening(RadiativeTable const& t, int level,
                                          double delta) {
    auto c = t.dl_[level];
    if (delta < 1.) return double(c[0]) + delta * (double(c[1]) + delta * double(c[2]));
    return double(c[3]) + double(c[4]) * ::log(delta + double(c[5]));
  }
  struct LpmResult {
    InteractionStatus status_{InteractionStatus::invalid_input};
    double acceptance_{1.};
    bool sampled_{};
  };
  // Port of C7 LPMEFFECT. Density is RHOF(ALT) in g/cm^3, supplied by the
  // atmosphere adapter, NOT the reference density in EGSDAT. C7's constants
  // describe air. Return `suppressed` means leave the incident particle intact
  // and return to transport, NOT retry until an interaction is accepted.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION LpmResult lpm_effect(double e0, double e1, double e2,
                                              double density, bool pair, TRandom& rng) {
    LpmResult r;
    if (!finite(e0) || !finite(e1) || !finite(e2) || !finite(density) || e0 <= 0. ||
        e1 <= 0. || e2 <= 0. || density <= 0.)
      return r;
    double ssq = 3.5156e7 * e0 / (e1 * e2 * density);
    if (!finite(ssq)) return r;
    r.status_ = InteractionStatus::success;
    if (ssq >= 1.) return r; // exactly as C7: consumes no random number
    double xi = ssq < 1.0796917e-8 ? 2. : 1. - 5.4513722e-2 * ::log(ssq);
    ssq = ssq / xi;
    double s = ::sqrt(ssq), g, phi;
    if (s < .1) {
      g = ssq * (14.1 + 2.36 / (s + .1));
      phi = 6. * s - 16. * ssq;
    } else {
      g = ssq * (24. + .0394 / (s - .08));
      phi =
          6. * s +
          24. * ssq * (3.1415926535897932384626433832795 * .25 - ::atan(.944 + .59 / s));
    }
    g = g / (1. + g);
    double egsq = e0 * e0, ab = 4. * e1 * e2;
    if (pair) ab = -ab;
    r.acceptance_ = xi * (egsq * (g + 2. * phi) + phi * ab) / (3. * egsq + ab);
    if (!finite(r.acceptance_)) {
      r.status_ = InteractionStatus::invalid_input;
      return r;
    }
    double u;
    if (!uniform(rng, u)) {
      r.status_ = InteractionStatus::random_failure;
      return r;
    }
    r.sampled_ = true;
    if (u > r.acceptance_) r.status_ = InteractionStatus::suppressed;
    return r;
  }
  struct RadiativeInput {
    double totalMeV_{}, massMeV_{}, densityGCm3_{};
    Direction direction_;
    int pdg_{}; // 22: photon conversion; +/-11: bremsstrahlung
  };
  KOKKOS_INLINE_FUNCTION bool valid_radiative(RadiativeInput q) {
    return finite(q.totalMeV_) && finite(q.massMeV_) && finite(q.densityGCm3_) &&
           q.totalMeV_ > 0. && q.massMeV_ > 0. && q.densityGCm3_ > 0. &&
           valid_direction(q.direction_);
  }

  template <class TRandom>
  KOKKOS_INLINE_FUNCTION Secondaries sample_bremsstrahlung(RadiativeTable const& t,
                                                           RadiativeInput q, TRandom& rng,
                                                           int max_trials = 0) {
    Secondaries r;
    double E = q.totalMeV_, m = q.massMeV_, ap = t.photonCutMeV_;
    if (!valid_radiative(q) || (q.pdg_ != 11 && q.pdg_ != -11) || !finite(ap) ||
        ap <= 0. || E <= m + ap || max_trials < 0)
      return r;
    int lvx = E < 50. ? 0 : 1, base = E < 50. ? 0 : 3;
    double n = 1.442695041 * ::log(E * (1. / ap));
    // PWR2I has 60 elements in C7. Fail explicitly rather than extrapolate it.
    if (!finite(n) || n < 0. || n >= 61.) return r;
    double abrems = double(int(n));
    int inner_trials = 0;
    // C7 BREMSLPM repeats until acceptance. Near the transport cut, valid
    // vertices can need more than 4096 proposals. A positive limit is only
    // for explicit diagnostic calls; zero preserves C7's default behavior.
    while (max_trials == 0 || r.trials_ < max_trials) {
      if (r.trials_ < INT_MAX) ++r.trials_;
      double u, v, w, br;
      int level;
      if (!uniform(rng, u) || !uniform(rng, v) || !uniform(rng, w)) {
        r.status_ = InteractionStatus::random_failure;
        return r;
      }
      if ((abrems * double(t.alphi_[lvx]) + .5) * u >= .5) {
        int idistr = int(abrems * v);
        if (idistr < 0 || idistr >= 60) return r;
        // Every PWR2I entry is exactly representable; ldexp equals the HATCH
        // recurrence P=1, P=P*.5, without per-thread lookup state.
        double p = ::ldexp(1., -idistr);
        level = base;
        if (w >= .721347521) {
          do {
            if (max_trials > 0 && ++inner_trials > max_trials) {
              r.status_ = InteractionStatus::trial_limit;
              return r;
            }
            if (!uniform(rng, u) || !uniform(rng, v) || !uniform(rng, w)) {
              r.status_ = InteractionStatus::random_failure;
              return r;
            }
            double h = larger(v, w);
            br = 1. - .5 * h;
          } while (br * u > .5);
        } else {
          if (!uniform(rng, u)) {
            r.status_ = InteractionStatus::random_failure;
            return r;
          }
          br = u * .5;
        }
        br = br * p;
      } else {
        br = larger(v, w);
        level = base + 1;
      }
      double photon = E * br;
      if (photon < ap) continue;
      double lepton = E - photon;
      if (lepton < m) continue;
      double del = br / lepton;
      if (del >= double(t.delpos_[lvx])) continue;
      double rej = screening(t, level, double(t.delcm_) * del);
      if (!uniform(rng, u)) {
        r.status_ = InteractionStatus::random_failure;
        return r;
      }
      if (u > rej) continue;
      if (E > 1.e10) {
        auto lpm = lpm_effect(photon, E, lepton, q.densityGCm3_, false, rng);
        if (lpm.status_ != InteractionStatus::success) {
          r.status_ = lpm.status_;
          return r;
        }
      }
      double sp, cp, angle = m / E;
      if (!azimuth(rng, sp, cp)) {
        r.status_ = InteractionStatus::random_failure;
        return r;
      }
      Secondary g{photon, rotate(q.direction_, ::sin(angle), ::cos(angle), sp, cp), 22};
      Secondary e{lepton, q.direction_, q.pdg_}; // C7 does not recoil the parent here.
      r.particle_[0] = photon <= lepton ? e : g;
      r.particle_[1] = photon <= lepton ? g : e;
      r.status_ = InteractionStatus::success;
      r.count_ = 2;
      return r;
    }
    r.status_ = InteractionStatus::trial_limit;
    return r;
  }

  template <class TRandom>
  KOKKOS_INLINE_FUNCTION Secondaries sample_pair_production(RadiativeTable const& t,
                                                            RadiativeInput q,
                                                            TRandom& rng,
                                                            int max_trials = 4096) {
    Secondaries r;
    double E = q.totalMeV_, m = q.massMeV_, e2 = m;
    if (!valid_radiative(q) || q.pdg_ != 22 || E <= 2. * m || max_trials < 1) return r;
    if (E > 2.1) {
      int lvx = E < 50. ? 0 : 1, base = E < 50. ? 0 : 3;
      bool accepted = false;
      for (int trial = 1; trial <= max_trials; ++trial) {
        r.trials_ = trial;
        double u, v, w, z, br;
        int level;
        if (!uniform(rng, u) || !uniform(rng, v)) {
          r.status_ = InteractionStatus::random_failure;
          return r;
        }
        if (v >= double(t.bpar_[lvx])) {
          level = base;
          if (!uniform(rng, w) || !uniform(rng, z)) {
            r.status_ = InteractionStatus::random_failure;
            return r;
          }
          br = .5 * smaller(smaller(w, z), u);
        } else {
          level = base + 2;
          br = u * .5;
        }
        if (br * E < m) continue;
        double del = 1. / (E * br * (1. - br));
        if (del >= double(t.delpos_[lvx])) continue;
        double rej = screening(t, level, double(t.delcm_) * del);
        if (!uniform(rng, u)) {
          r.status_ = InteractionStatus::random_failure;
          return r;
        }
        if (u > rej) continue;
        e2 = br * E;
        accepted = true;
        break;
      }
      if (!accepted) {
        r.status_ = InteractionStatus::trial_limit;
        return r;
      }
    }
    double e1 = E - e2;
    if (E > 1.e12) {
      auto lpm = lpm_effect(E, e1, e2, q.densityGCm3_, true, rng);
      if (lpm.status_ != InteractionStatus::success) {
        r.status_ = lpm.status_;
        return r;
      }
    }
    double sp, cp, angle = m / E;
    if (!azimuth(rng, sp, cp)) {
      r.status_ = InteractionStatus::random_failure;
      return r;
    }
    double st = ::sin(angle), ct = ::cos(angle), u;
    if (!uniform(rng, u)) {
      r.status_ = InteractionStatus::random_failure;
      return r;
    }
    int pdg1 = u <= .5 ? 11 : -11;
    r.particle_[0] = {e1, rotate(q.direction_, st, ct, sp, cp), pdg1};
    r.particle_[1] = {e2, rotate(q.direction_, -st, ct, sp, cp), -pdg1};
    r.status_ = InteractionStatus::success;
    r.count_ = 2;
    return r;
  }
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION Secondaries sample_radiative(RadiativeTable const& t,
                                                      RadiativeInput q, TRandom& rng,
                                                      int max_trials) {
    if (q.pdg_ == 22) return sample_pair_production(t, q, rng, max_trials);
    return sample_bremsstrahlung(t, q, rng, max_trials);
  }
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION Secondaries sample_radiative(RadiativeTable const& t,
                                                      RadiativeInput q, TRandom& rng) {
    if (q.pdg_ == 22) return sample_pair_production(t, q, rng);
    return sample_bremsstrahlung(t, q, rng);
  }
} // namespace corsika::kokkos::egs4
