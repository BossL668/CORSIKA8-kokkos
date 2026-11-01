/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4Interactions.inl>
#include <corsika/detail/kokkos/egs4/Egs4ScatteringCoefficients.inl>

namespace corsika::kokkos::egs4 {
  struct ScatteringResult {
    InteractionStatus status_{InteractionStatus::invalid_input};
    double angle_{}, sine_{}, cosine_{1.};
    int trials_{};
    bool belowScatteringThreshold_{};
  };
  // Native, dependency-free adaptation of the earlier MSCAT port. The
  // material and mass are now explicit rather than frozen to C8 constants.
  // vsteff is precisely TVSTEP*RHOFAC in the C7 region-reference medium.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION ScatteringResult
  sample_scattering(Medium medium, double total_MeV, double mass_MeV, double vsteff,
                    TRandom& rng, int max_trials = 1024, bool paired_trig = false) {
    ScatteringResult r;
    if (!finite(total_MeV) || !finite(mass_MeV) || mass_MeV <= 0. ||
        total_MeV <= mass_MeV || !finite(vsteff) || vsteff < 0. ||
        !finite(medium.blcc_) || medium.blcc_ <= 0. || !finite(medium.xcc_) ||
        medium.xcc_ <= 0. || max_trials < 1)
      return r;
    auto const& t = scattering_detail::coefficients();
    double beta2 = larger(1.e-8, 1. - mass_MeV * mass_MeV / (total_MeV * total_MeV));
    double omega = medium.blcc_ * vsteff / beta2;
    if (!finite(omega)) return r;
    if (omega <= 1.) {
      r.status_ = InteractionStatus::success;
      r.belowScatteringThreshold_ = true;
      return r;
    }
    double blc = ::log(omega), B;
    if (blc <= 1.306852820)
      B = 1.530394218 * blc;
    else {
      int index = int(1.5714 + blc * .21429) - 1;
      if (index > 7) index = 7;
      auto c = t.BGB_[index];
      B = c[0] + blc * (c[1] + blc * c[2]);
    }
    double xr = medium.xcc_ * ::sqrt(larger(0., vsteff * B)) / (total_MeV * beta2);
    double bi, bm1, bm2;
    if (B > 2.) {
      bi = 1. / B;
      double bmd = 1. / (1. + 1.75 * bi);
      bm1 = (1. - 2. * bi) * bmd;
      bm2 = (1. + double(.025f) * bi) * bmd;
    } else {
      bi = .5;
      bm1 = (1. - 2. / B) * .533333333333;
      bm2 = .54;
    }
    for (int trial = 1; trial <= max_trials; ++trial) {
      r.trials_ = trial;
      double u, v, w, a, thr;
      if (!uniform(rng, u)) {
        r.status_ = InteractionStatus::random_failure;
        return r;
      }
      if (u <= bm1) {
        if (!uniform(rng, v)) {
          r.status_ = InteractionStatus::random_failure;
          return r;
        }
        thr = ::sqrt(larger(0., -::log(v)));
      } else if (u <= bm2) {
        if (!uniform(rng, v) || !uniform(rng, w) || !uniform(rng, a)) {
          r.status_ = InteractionStatus::random_failure;
          return r;
        }
        double eta = larger(v, w);
        auto c = t.G31_[int(2. + eta * 9.) - 1];
        auto d = t.G32_[int(2. + eta * 23.) - 1];
        double g31 = c[0] + eta * (c[1] + eta * c[2]),
               g32 = d[0] + eta * (d[1] + eta * d[2]);
        if (a > g31 + g32 * bi) continue;
        thr = 1. / eta;
      } else {
        if (!uniform(rng, v) || !uniform(rng, w)) {
          r.status_ = InteractionStatus::random_failure;
          return r;
        }
        thr = v;
        auto c = t.G21_[int(2. + thr * 5.) - 1];
        auto d = t.G22_[int(2. + thr * 6.) - 1];
        double g21 = c[0] + thr * (c[1] + thr * c[2]),
               g22 = d[0] + thr * (d[1] + thr * d[2]);
        if (w > g21 + g22 * bi) continue;
      }
      double theta = thr * xr;
      if (theta >= 3.1415926535897932384626433832795) continue;
      double sine, cosine = 0.;
      if (paired_trig)
        ::sincos(theta, &sine, &cosine);
      else
        sine = ::sin(theta);
      if (!uniform(rng, u)) {
        r.status_ = InteractionStatus::random_failure;
        return r;
      }
      if (u * u * theta > sine) continue;
      r.angle_ = theta;
      r.sine_ = sine;
      r.cosine_ = paired_trig ? cosine : ::cos(theta);
      r.status_ = InteractionStatus::success;
      return r;
    }
    r.status_ = InteractionStatus::trial_limit;
    return r;
  }
} // namespace corsika::kokkos::egs4
