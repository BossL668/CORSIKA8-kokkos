/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4Interactions.inl>

namespace corsika::kokkos::egs4 {
  struct NuclearTarget {
    InteractionStatus status_{InteractionStatus::invalid_input};
    int pdg_{}, massNumber_{}, atomicNumber_{};
  };
  // Original C7 SDPM, photon-projectile block, LTA=0: literal A^0.91
  // weights, not the nucleon selection in PIGEN1/2 or RHOGEN. Consumes
  // precisely one hadronic-stream draw after PIGEN has selected branch 4.
  // This selects a target only; it does not pretend to generate HDPM hadrons.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION NuclearTarget
  sample_many_hadron_target(double const* composition, TRandom& rng) {
    NuclearTarget out;
    if (!composition) return out;
    for (int j = 0; j < 3; ++j)
      if (!finite(composition[j]) || composition[j] < 0.) return out;
    double nitrogen = composition[0] * 11.04019;
    double oxygen = nitrogen + composition[1] * 12.46663;
    double total = oxygen + composition[2] * 28.69952;
    if (!finite(total) || total <= 0.) return out;
    double u;
    if (!uniform(rng, u)) {
      out.status_ = InteractionStatus::random_failure;
      return out;
    }
    if (u * total <= nitrogen) {
      out.massNumber_ = 14;
      out.atomicNumber_ = 7;
    } else if (u * total <= oxygen) {
      out.massNumber_ = 16;
      out.atomicNumber_ = 8;
    } else {
      out.massNumber_ = 40;
      out.atomicNumber_ = 18;
    }
    out.pdg_ = 1000000000 + 10000 * out.atomicNumber_ + 10 * out.massNumber_;
    out.status_ = InteractionStatus::success;
    return out;
  }
} // namespace corsika::kokkos::egs4
