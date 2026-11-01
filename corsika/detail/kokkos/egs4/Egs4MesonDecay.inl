/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4Interactions.inl>

namespace corsika::kokkos::egs4 {
  // C7 RHO0DC before angular cuts, deposition and TSTACK. All draws use
  // stream 1, supplied explicitly by the caller. No shared SAVE/COMMON state.
  enum class RhoOrigin { hadronic = 0, photonuclear = 1 };
  struct RhoDecayInput {
    double totalMeV_{}, rhoMassMeV_{}, pionMassMeV_{}, muonMassMeV_{};
    Direction direction_;
    RhoOrigin origin_{RhoOrigin::photonuclear};
  };
  struct RhoDecayOutcome {
    InteractionStatus status_{InteractionStatus::invalid_input};
    int count_{}, trials_{};
    Secondary particle_[2]{}; // negative first, positive second; NOT energy sorted
    double polarizationCosine_[2]{}, polarizationAzimuth_[2]{};
    double cmCosine_{};
  };
  // ADDANG3, in the EGS direction convention (U,V,W): C7's arguments are
  // (W,U,-V). Unlike UPHI it has a 1e-40 polar branch and one normalization
  // iteration. Preserve both; simply reusing rotate() changes C7 arithmetic.
  KOKKOS_INLINE_FUNCTION Direction add_hadronic_angle(Direction q, double ct,
                                                      double phi) {
    double st = ::sqrt((1. - ct) * (1. + ct)), s2 = q.x_ * q.x_ + q.y_ * q.y_;
    double x, y, w;
    if (s2 < 1.e-40) {
      x = st * ::cos(-phi);
      y = st * ::sin(-phi);
      w = ct * q.z_;
    } else {
      double px = st * ::cos(-phi), py = st * ::sin(-phi), s = ::sqrt(s2);
      double cd = q.x_ * (1. / s), sd = q.y_ * (1. / s);
      x = q.z_ * cd * px - sd * py + q.x_ * ct;
      y = -q.z_ * sd * px - cd * py - q.y_ * ct;
      w = -s * px + q.z_ * ct;
    }
    double inverse = 1.5 - .5 * (x * x + y * y + w * w);
    x = larger(-1., inverse * x);
    x = x < 1. ? x : 1.;
    y = larger(-1., inverse * y);
    y = y < 1. ? y : 1.;
    w = larger(-1., inverse * w);
    w = w < 1. ? w : 1.;
    return {x, -y, w};
  }
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION RhoDecayOutcome sample_rho_decay(RhoDecayInput q, TRandom& rng,
                                                          int max_trials = 4096) {
    RhoDecayOutcome out;
    if (!finite(q.totalMeV_) || !finite(q.rhoMassMeV_) || !finite(q.pionMassMeV_) ||
        !finite(q.muonMassMeV_) || q.rhoMassMeV_ <= 0. || q.pionMassMeV_ <= 0. ||
        q.muonMassMeV_ <= 0. || q.rhoMassMeV_ <= 2. * q.pionMassMeV_ ||
        q.rhoMassMeV_ <= 2. * q.muonMassMeV_ || q.totalMeV_ <= q.rhoMassMeV_ ||
        !valid_direction(q.direction_) || max_trials < 1 ||
        (q.origin_ != RhoOrigin::hadronic && q.origin_ != RhoOrigin::photonuclear))
      return out;
    // The source lab-angle formula is undefined at gamma=1. Return an explicit
    // error instead of inventing a rest-frame convention for a zero direction.
    double gamma = q.totalMeV_ / q.rhoMassMeV_;
    if (!finite(gamma) || gamma <= 1.) return out;
    double u, v;
    if (!uniform(rng, u)) {
      out.status_ = InteractionStatus::random_failure;
      return out;
    }
    bool muon = u >= .999955;
    double mass = muon ? q.muonMassMeV_ : q.pionMassMeV_, polar = 0., azimuth = 0.;
    constexpr double pi = 3.1415926535897932384626433832795;
    if (muon) {
      if (!uniform(rng, u) || !uniform(rng, v)) {
        out.status_ = InteractionStatus::random_failure;
        return out;
      }
      polar = 2. * u - 1.;
      azimuth = 2. * pi * v;
    }
    double beta = ::sqrt((gamma - 1.) * (gamma + 1.)) / gamma;
    double a = .5 * q.rhoMassMeV_ / mass, work1 = gamma * a;
    double work2 = beta * gamma * ::sqrt((a - 1.) * (a + 1.)), cm = 0.;
    bool accepted = false;
    for (int trial = 1; trial <= max_trials; ++trial) {
      out.trials_ = trial;
      if (!uniform(rng, u) || !uniform(rng, v)) {
        out.status_ = InteractionStatus::random_failure;
        return out;
      }
      cm = 2. * u - 1.;
      // Literal RHO0DC applies the origin-dependent rejection to BOTH pion
      // and muon branches, despite its header comment about isotropic muons.
      if (q.origin_ == RhoOrigin::photonuclear && v > 1. - .8836 * cm * cm) continue;
      accepted = true;
      break;
    }
    if (!accepted) {
      out.status_ = InteractionStatus::trial_limit;
      return out;
    }
    double g3 = work1 + work2 * cm, g4 = gamma * (q.rhoMassMeV_ / mass) - g3;
    if (!finite(g3) || !finite(g4) || g3 <= 1. || g4 <= 1.) return out;
    double ct4 = (gamma * g4 - a) / (beta * gamma * ::sqrt((g4 - 1.) * (g4 + 1.)));
    ct4 = ct4 < 1. ? ct4 : 1.;
    if (!uniform(rng, u)) {
      out.status_ = InteractionStatus::random_failure;
      return out;
    }
    double phi = 2. * pi * u;
    double ct3 = (gamma * g3 - a) / (beta * gamma * ::sqrt((g3 - 1.) * (g3 + 1.)));
    ct3 = ct3 < 1. ? ct3 : 1.;
    if (!finite(ct3) || !finite(ct4) || ct3 < -1. || ct4 < -1.) return out;
    out.particle_[0] = {g4 * mass, add_hadronic_angle(q.direction_, ct4, phi),
                        muon ? 13 : -211};
    out.particle_[1] = {g3 * mass, add_hadronic_angle(q.direction_, ct3, phi + pi),
                        muon ? -13 : 211};
    if (muon) {
      out.polarizationCosine_[0] = polar;
      out.polarizationAzimuth_[0] = azimuth;
      out.polarizationCosine_[1] = -polar;
      out.polarizationAzimuth_[1] = azimuth + pi;
    }
    for (auto const& p : out.particle_)
      if (!finite(p.energyMeV_) || !valid_direction(p.direction_)) return {};
    out.cmCosine_ = cm;
    out.count_ = 2;
    out.status_ = InteractionStatus::success;
    return out;
  }
} // namespace corsika::kokkos::egs4
