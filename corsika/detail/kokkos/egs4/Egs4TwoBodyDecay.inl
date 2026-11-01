/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4MesonDecay.inl>

namespace corsika::kokkos::egs4 {
  // C7 DECAY1/DECAY2 before angular cuts and TSTACK. Not a replacement
  // isotropic Pythia call: preserve the source arithmetic, draw order,
  // ADDANG3 convention and second-product-first output order.
  enum class TwoBodyKind { general, muon_pair };
  struct TwoBodyInput {
    double totalMeV_{}, parentMassMeV_{}, firstMassMeV_{}, secondMassMeV_{};
    int firstPdg_{}, secondPdg_{};
    Direction direction_;
    TwoBodyKind kind_{TwoBodyKind::general};
  };
  struct TwoBodyOutcome {
    InteractionStatus status_{InteractionStatus::invalid_input};
    int count_{};
    Secondary particle_[2]{}; // source M4, then M3
    double polarizationCosine_[2]{}, polarizationAzimuth_[2]{};
    double cmCosine_{};
  };
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION TwoBodyOutcome sample_two_body_decay(TwoBodyInput q,
                                                              TRandom& rng) {
    TwoBodyOutcome out;
    if (!finite(q.totalMeV_) || !finite(q.parentMassMeV_) || !finite(q.firstMassMeV_) ||
        !finite(q.secondMassMeV_) || q.firstMassMeV_ <= 0. || q.secondMassMeV_ < 0. ||
        q.parentMassMeV_ <= q.firstMassMeV_ + q.secondMassMeV_ ||
        q.totalMeV_ <= q.parentMassMeV_ || !q.firstPdg_ || !q.secondPdg_ ||
        !valid_direction(q.direction_) ||
        (q.kind_ != TwoBodyKind::general && q.kind_ != TwoBodyKind::muon_pair))
      return out;
    if (q.kind_ == TwoBodyKind::muon_pair &&
        (q.firstPdg_ != -13 || q.secondPdg_ != 13 || q.firstMassMeV_ != q.secondMassMeV_))
      return out;
    // DECAY1/2 use GeV masses in C7. Retain that internal convention, while
    // the EGS4-native public carrier consistently uses total MeV energy.
    double m0 = q.parentMassMeV_ * .001, m3 = q.firstMassMeV_ * .001,
           m4 = q.secondMassMeV_ * .001;
    double gamma = q.totalMeV_ / q.parentMassMeV_;
    if (!finite(gamma) || gamma <= 1.) return out; // literal formula divides by beta
    double beta = ::sqrt((gamma - 1.) * (gamma + 1.)) / gamma;
    double a = q.kind_ == TwoBodyKind::muon_pair
                   ? (m0 * m0) / (2. * m0)
                   : (m0 * m0 + m3 * m3 - m4 * m4) / (2. * m0);
    double a1 = a * a - m3 * m3, a2 = 1. + a1 / (m3 * m3), a2a = ::sqrt(a2),
           a3 = ::sqrt(1. - 1. / a2);
    double work1 = gamma * a2a, work2 = a3 * beta * work1, u[4]{};
    int draws = q.kind_ == TwoBodyKind::muon_pair ? 4 : 2;
    for (int i = 0; i < draws; ++i)
      if (!uniform(rng, u[i])) {
        out.status_ = InteractionStatus::random_failure;
        return out;
      }
    double cm = 2. * u[0] - 1., g3 = work1 + work2 * cm, g4, ct4, e4;
    if (m4 != 0.) {
      g4 = (m0 * gamma - m3 * g3) / m4;
      double a4 = (m0 * m0 + m4 * m4 - m3 * m3) / (2. * m0 * m4);
      if (!finite(g4) || g4 <= 1.) return out;
      ct4 = (gamma * g4 - a4) / (beta * gamma * ::sqrt((g4 - 1.) * (g4 + 1.)));
      e4 = g4 * m4;
    } else {
      e4 = m0 * gamma - m3 * g3;
      ct4 = (beta - cm) / (1. - beta * cm);
    }
    if (!finite(g3) || g3 <= 1.) return out;
    double ct3 = (gamma * g3 - a2a) / (beta * gamma * ::sqrt((g3 - 1.) * (g3 + 1.)));
    if (!finite(ct3) || !finite(ct4)) return out;
    ct3 = ct3 < 1. ? ct3 : 1.;
    ct4 = ct4 < 1. ? ct4 : 1.;
    if (ct3 < -1. || ct4 < -1. || !finite(e4) || e4 <= 0.) return out;
    constexpr double pi = 3.1415926535897932384626433832795;
    double phi = 2. * pi * u[1];
    out.particle_[0] = {e4 * 1000., add_hadronic_angle(q.direction_, ct4, phi),
                        q.secondPdg_};
    out.particle_[1] = {g3 * m3 * 1000., add_hadronic_angle(q.direction_, ct3, phi + pi),
                        q.firstPdg_};
    if (q.kind_ == TwoBodyKind::muon_pair) {
      out.polarizationCosine_[0] = 2. * u[2] - 1.;
      out.polarizationAzimuth_[0] = 2. * pi * u[3];
      out.polarizationCosine_[1] = -out.polarizationCosine_[0];
      out.polarizationAzimuth_[1] = out.polarizationAzimuth_[0] + pi;
    }
    for (auto const& p : out.particle_)
      if (!finite(p.energyMeV_) || !valid_direction(p.direction_)) return {};
    out.cmCosine_ = cm;
    out.count_ = 2;
    out.status_ = InteractionStatus::success;
    return out;
  }
} // namespace corsika::kokkos::egs4
