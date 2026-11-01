/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4MesonDecay.inl>

namespace corsika::kokkos::egs4 {
  struct ThreeBodyInput {
    double totalMeV_{}, parentMassMeV_{}, massMeV_[3]{};
    int pdg_[3]{};
    Direction direction_;
  };
  struct ThreeBodyOutcome {
    InteractionStatus status_{InteractionStatus::invalid_input};
    int count_{}, trials_{};
    Secondary particle_[3]{};
  };
  // The uniform phase-space DECAY6 branch actually used by omega/phi RESDEC:
  // MODE=2, PARAMA=PARAMB=PARAMC=0, AMPMX=1. Other DECAY6 matrix elements
  // (kaon/Dalitz modes) are not silently approximated by this function.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION ThreeBodyOutcome
  sample_uniform_three_body(ThreeBodyInput q, TRandom& rng, int max_trials = 4096) {
    ThreeBodyOutcome out;
    if (!finite(q.totalMeV_) || !finite(q.parentMassMeV_) ||
        !valid_direction(q.direction_) || q.parentMassMeV_ <= 0. ||
        q.totalMeV_ < q.parentMassMeV_ || max_trials < 1)
      return out;
    double sum = 0.;
    for (int j = 0; j < 3; ++j) {
      if (!finite(q.massMeV_[j]) || q.massMeV_[j] < 0. ||
          (j < 2 && q.massMeV_[j] == 0.) || !q.pdg_[j])
        return out;
      sum += q.massMeV_[j];
    }
    if (q.parentMassMeV_ <= sum) return out;
    double m0 = q.parentMassMeV_ * .001,
           m[3]{q.massMeV_[0] * .001, q.massMeV_[1] * .001, q.massMeV_[2] * .001};
    double gamma = q.totalMeV_ / q.parentMassMeV_,
           beta = ::sqrt((gamma - 1.) * (gamma + 1.)) / gamma;
    if (!finite(gamma) || !finite(beta)) return out;
    double s0 = m0 * m0, s[3]{m[0] * m[0], m[1] * m[1], m[2] * m[2]};
    double a1 = (m[0] + m[1]) * (m[0] + m[1]), a2 = (m0 - m[2]) * (m0 - m[2]) - a1;
    double a3 = (m[0] + m[2]) * (m[0] + m[2]), a4 = (m0 - m[1]) * (m0 - m[1]) - a3;
    double a5 = s[0] - s[1], a6 = s0 - s[2], a7 = .5 / m0, e[3]{}, u[3]{};
    bool accepted = false;
    for (int trial = 1; trial <= max_trials; ++trial) {
      out.trials_ = trial;
      // Draw all THREE values, including the amplitude test's uniform even
      // though AMPMX=AMPLI=1. Dropping it changes subsequent C7 angles.
      for (int j = 0; j < 3; ++j)
        if (!uniform(rng, u[j])) {
          out.status_ = InteractionStatus::random_failure;
          return out;
        }
      double s34 = a2 * u[0] + a1, s35 = a4 * u[1] + a3, inverse = .5 / ::sqrt(s34);
      double e3star = (a5 + s34) * inverse, e5star = (a6 - s34) * inverse;
      double r1s = e3star * e3star - s[0], r2s = e5star * e5star - s[2];
      if (!finite(r1s) || !finite(r2s) || r1s < 0. || r2s < 0.) return out;
      double r1 = ::sqrt(r1s), r2 = ::sqrt(r2s),
             discr = s35 - (e3star + e5star) * (e3star + e5star);
      if (discr > -(r1 - r2) * (r1 - r2) || discr < -(r1 + r2) * (r1 + r2)) continue;
      e[1] = (s0 + s[1] - s35) * a7;
      e[2] = (s0 + s[2] - s34) * a7;
      e[0] = m0 - e[1] - e[2];
      accepted = true;
      break;
    }
    if (!accepted) {
      out.status_ = InteractionStatus::trial_limit;
      return out;
    }
    double p[3], p2[3];
    for (int j = 0; j < 3; ++j) {
      p2[j] = e[j] * e[j] - s[j];
      if (!finite(p2[j]) || p2[j] <= 0.) return out;
      p[j] = ::sqrt(p2[j]);
    }
    double ca = (p2[2] - p2[0] - p2[1]) / (2. * p[0] * p[1]);
    double sa = -::sqrt(larger(0., (1. - ca) * (1. + ca)));
    double cb = (p2[1] - p2[0] - p2[2]) / (2. * p[0] * p[2]);
    double sb = ::sqrt(larger(0., (1. - cb) * (1. + cb)));
    for (int j = 0; j < 3; ++j)
      if (!uniform(rng, u[j])) {
        out.status_ = InteractionStatus::random_failure;
        return out;
      }
    constexpr double pi = 3.1415926535897932384626433832795, twopi = 2. * pi;
    double cosine[3]{2. * u[0] - 1., 0., 0.}, phi[3]{twopi * u[1], 0., 0.};
    double st = ::sqrt(larger(0., (1. - cosine[0]) * (1. + cosine[0])));
    double cf = ::cos(phi[0]), sf = ::sin(phi[0]), psi = twopi * u[2], cp = ::cos(psi),
           sp = ::sin(psi);
    cosine[1] = cosine[0] * ca - st * cp * sa;
    cosine[2] = cosine[0] * cb - st * cp * sb;
    double triangle_cos[2]{ca, cb}, triangle_sin[2]{sa, sb};
    for (int j = 1; j < 3; ++j) {
      if (::fabs(cosine[j]) < 1.) {
        double inverse = 1. / ::sqrt((1. - cosine[j]) * (1. + cosine[j]));
        double aux = cosine[0] * cp * triangle_sin[j - 1] + st * triangle_cos[j - 1];
        double cfj = (cf * aux - sf * sp * triangle_sin[j - 1]) * inverse;
        phi[j] = ::acos(larger(-1., cfj < 1. ? cfj : 1.));
        double sfj = (sf * aux + cf * sp * triangle_sin[j - 1]) * inverse;
        if (sfj <= 0.) phi[j] = twopi - phi[j];
      }
    }
    for (int j = 0; j < 3; ++j) {
      double factor = gamma * (e[j] + beta * p[j] * cosine[j]), ct;
      if (m[j] != 0.) {
        double g = factor / m[j];
        if (!finite(g) || g <= 1.) return out;
        ct = (beta * e[j] + p[j] * cosine[j]) * gamma /
             (m[j] * ::sqrt((g - 1.) * (g + 1.)));
        factor = g * m[j]; // preserve source gamma -> total energy conversion
      } else
        ct = (beta * e[j] + p[j] * cosine[j]) * gamma / factor;
      if (!finite(ct) || ct < -1. || !finite(factor) || factor <= 0.) return out;
      ct = ct < 1. ? ct : 1.;
      out.particle_[j] = {factor * 1000., add_hadronic_angle(q.direction_, ct, phi[j]),
                          q.pdg_[j]};
      if (!valid_direction(out.particle_[j].direction_)) return {};
    }
    out.count_ = 3;
    out.status_ = InteractionStatus::success;
    return out;
  }
} // namespace corsika::kokkos::egs4
