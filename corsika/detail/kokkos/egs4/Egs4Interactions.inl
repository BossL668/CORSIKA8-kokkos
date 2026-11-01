/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/egs4/Tables.hpp>

// Native EGS4 implementation: direct C++ versions of C7 COMPT, MOLLER,
// BHABHA and UPHI. No PROPOSAL calls or Fortran runtime dependency.
// Energies are TOTAL MeV. The electron initially at rest belongs to the
// medium: outgoing total energies sum to incident energy + electron mass.
// This module does not perform process selection, transport or thinning.
namespace corsika::kokkos::egs4 {
  enum class InteractionStatus {
    success,
    invalid_input,
    random_failure,
    trial_limit,
    suppressed,
    host_required
  };
  enum class Collision { compton, moller, bhabha };
  struct Direction {
    double x_{}, y_{}, z_{};
  };
  struct Secondary {
    double energyMeV_{};
    Direction direction_;
    int pdg_{};
  };
  struct Secondaries {
    InteractionStatus status_{InteractionStatus::invalid_input};
    int count_{}, trials_{};
    Secondary particle_[2]{}; // descending energy, matching the C7 stack order
  };
  struct CollisionInput {
    Collision process_{};
    double totalMeV_{}, massMeV_{}, secondaryKineticCutMeV_{};
    Direction direction_;
  };
  KOKKOS_INLINE_FUNCTION bool valid_direction(Direction a) {
    double norm = a.x_ * a.x_ + a.y_ * a.y_ + a.z_ * a.z_;
    // ELECTR's one-iteration magnetic renormalization at alpha=0.2 leaves
    // about 1.2e-9 in norm^2. Accept its output without silently normalizing
    // it differently from C7. This is an input check, not a physics change.
    return finite(norm) && ::fabs(norm - 1.) < 1.e-8;
  }
  KOKKOS_INLINE_FUNCTION double larger(double a, double b) { return a > b ? a : b; }
  // Preserve UPHI's original small-polar-angle branch and arithmetic ordering.
  // No normalization here: it would change the algorithm being ported.
  KOKKOS_INLINE_FUNCTION Direction rotate(Direction q, double st, double ct, double sp,
                                          double cp) {
    auto a = q.x_, b = q.y_, c = q.z_;
    double s2 = a * a + b * b;
    if (s2 < 1.e-20) return {st * cp, st * sp, c * ct};
    double s = ::sqrt(s2), us = st * cp, vs = st * sp;
    double sd = b * (1. / s), cd = a * (1. / s);
    return {c * cd * us - sd * vs + a * ct, c * sd * us + cd * vs + b * ct,
            (-s) * us + c * ct};
  }
  // Random is per history, with bool next(double&) returning an available
  // uniform in [0,1). Exhaustion is an explicit error, never a replacement
  // physics model or an accepted truncated rejection sample.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION bool uniform(TRandom& rng, double& u) {
    return rng.next(u) && finite(u) && u >= 0. && u < 1.;
  }
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION bool azimuth(TRandom& rng, double& sp, double& cp) {
    double u;
    if (!uniform(rng, u)) return false;
    double phi = u * 6.283185307179586476925286766559;
    sp = ::sin(phi);
    cp = ::cos(phi);
    return true;
  }

  template <class TRandom>
  KOKKOS_INLINE_FUNCTION Secondaries sample_compton(CollisionInput q, TRandom& rng,
                                                    int max_trials = 4096) {
    Secondaries r;
    if (!finite(q.totalMeV_) || q.totalMeV_ <= 0. || !finite(q.massMeV_) ||
        q.massMeV_ <= 0. || !valid_direction(q.direction_) || max_trials < 1)
      return r;
    double peig = q.totalMeV_, mass = q.massMeV_, egp = peig * (1. / mass);
    double br0i = 1. + 2. * egp, br0 = 1. / br0i;
    if (!finite(egp) || !finite(br0i)) return r;
    double alph1 = ::log(br0i), alph2 = egp * (br0i + 1.) * (br0 * br0);
    double sumalp = alph1 + alph2;
    for (int trial = 1; trial <= max_trials; ++trial) {
      r.trials_ = trial;
      double u, v, w, extra, br;
      if (!uniform(rng, u) || !uniform(rng, v) || !uniform(rng, w)) {
        r.status_ = InteractionStatus::random_failure;
        return r;
      }
      if (alph1 >= sumalp * u)
        br = ::exp(alph1 * v) * br0;
      else {
        double brp = v;
        if (egp >= (egp + 1.) * w) {
          if (!uniform(rng, extra)) {
            r.status_ = InteractionStatus::random_failure;
            return r;
          }
          brp = larger(v, extra);
        }
        br = ((br0i - 1.) * brp + 1.) * br0;
      }
      double pesg = br * peig, a1mibr = 1. - br, temp = mass * a1mibr / pesg;
      double sine2 = larger(0., temp * (2. - temp));
      if (!uniform(rng, u)) {
        r.status_ = InteractionStatus::random_failure;
        return r;
      }
      if ((1. - u) * (1. + br * br) < br * sine2) continue;
      double pese = peig - pesg + mass, sp, cp;
      if (!azimuth(rng, sp, cp)) {
        r.status_ = InteractionStatus::random_failure;
        return r;
      }
      Secondary photon{pesg, rotate(q.direction_, ::sqrt(sine2), 1. - temp, sp, cp), 22};
      double psq = pese * pese - mass * mass, ct, st;
      if (psq <= 0.) {
        ct = 0.;
        st = -1.;
      } else {
        ct = (pese + pesg) * a1mibr / ::sqrt(psq);
        st = -::sqrt(larger(0., (1. - ct) * (1. + ct)));
      }
      Secondary electron{pese, rotate(q.direction_, st, ct, sp, cp), 11};
      r.particle_[0] = pese <= pesg ? photon : electron;
      r.particle_[1] = pese <= pesg ? electron : photon;
      r.count_ = 2;
      r.status_ = InteractionStatus::success;
      return r;
    }
    r.status_ = InteractionStatus::trial_limit;
    return r;
  }

  // Shared kinematics, but the two C7 energy-sharing distributions are distinct.
  template <bool Positron, class TRandom>
  KOKKOS_INLINE_FUNCTION Secondaries sample_ionization(CollisionInput q, TRandom& rng,
                                                       int max_trials = 4096) {
    Secondaries r;
    double E = q.totalMeV_, m = q.massMeV_, te = q.secondaryKineticCutMeV_;
    if (!finite(E) || !finite(m) || m <= 0. || !finite(te) || te <= 0. ||
        E <= m + (Positron ? te : 2. * te) || !valid_direction(q.direction_) ||
        max_trials < 1)
      return r;
    double kinetic = E - m, ki = 1. / kinetic, t0 = kinetic * (1. / m), e0 = t0 + 1.;
    double e02 = e0 * e0, ep0 = te * ki;
    if (!finite(e02) || !finite(ep0)) return r;
    double g2{}, g3{}, gmax{}, beta2{}, b1{}, b2{}, b3{}, b4{};
    if constexpr (Positron) {
      double yy = 1. / (t0 + 2.), y2 = yy * yy, yp = 1. - 2. * yy, yp2 = yp * yp;
      beta2 = (e02 - 1.) / e02;
      b4 = yp2 * yp;
      b3 = b4 + yp2;
      b2 = yp * (3. + y2);
      b1 = 2. - y2;
    } else {
      g2 = t0 * t0 * (1. / e02);
      g3 = (2. * t0 + 1.) * (1. / e02);
      gmax = 1. + 1.25 * g2;
    }
    for (int trial = 1; trial <= max_trials; ++trial) {
      r.trials_ = trial;
      double u, v, br;
      if (!uniform(rng, u) || !uniform(rng, v)) {
        r.status_ = InteractionStatus::random_failure;
        return r;
      }
      int pdg1 = Positron ? -11 : 11, pdg2 = 11;
      if constexpr (Positron) {
        br = ep0 / (1. - (1. - ep0) * u);
        double rej = 1. - beta2 * br * (b1 - br * (b2 - br * (b3 - br * b4)));
        if (v > rej) continue;
        if (br >= .5) {
          pdg1 = 11;
          pdg2 = -11;
          br = 1. - br;
        }
        br = larger(0., br);
      } else {
        double extrae = E - (2. * te + m), aux = kinetic - extrae * u;
        if (aux == 0.) continue;
        br = te / aux;
        double ratio = br / (1. - br), rej = 1. + g2 * br * br + ratio * (ratio - g3);
        if (v * gmax > rej) continue;
      }
      double ek2 = br * kinetic, e1 = E - ek2, e2 = ek2 + m;
      double h1 = (E + m) * ki;
      double c1 = smaller(1., h1 * (e1 - m) / (e1 + m)),
             c2 = smaller(1., h1 * (e2 - m) / (e2 + m));
      double sp, cp;
      if (!azimuth(rng, sp, cp)) {
        r.status_ = InteractionStatus::random_failure;
        return r;
      }
      r.particle_[0] = {e1, rotate(q.direction_, ::sqrt(1. - c1), ::sqrt(c1), sp, cp),
                        pdg1};
      r.particle_[1] = {e2, rotate(q.direction_, -::sqrt(1. - c2), ::sqrt(c2), sp, cp),
                        pdg2};
      r.count_ = 2;
      r.status_ = InteractionStatus::success;
      return r;
    }
    r.status_ = InteractionStatus::trial_limit;
    return r;
  }
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION Secondaries sample_collision(CollisionInput q, TRandom& rng,
                                                      int max_trials = 4096) {
    switch (q.process_) {
      case Collision::compton:
        return sample_compton(q, rng, max_trials);
      case Collision::moller:
        return sample_ionization<false>(q, rng, max_trials);
      case Collision::bhabha:
        return sample_ionization<true>(q, rng, max_trials);
    }
    return {};
  }
} // namespace corsika::kokkos::egs4
