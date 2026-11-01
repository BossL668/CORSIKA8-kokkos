/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4TwoBodyDecay.inl>
#include <corsika/detail/kokkos/egs4/Egs4ThreeBodyDecay.inl>

namespace corsika::kokkos::egs4 {
  struct ResonanceMasses {
    double omegaMeV_{}, phiMeV_{}, chargedPionMeV_{}, neutralPionMeV_{}, muonMeV_{};
    double chargedKaonMeV_{}, longKaonMeV_{}, shortKaonMeV_{}, etaMeV_{};
  };
  KOKKOS_INLINE_FUNCTION bool valid_resonance_masses(ResonanceMasses const& m) {
    double fields[]{m.omegaMeV_,       m.phiMeV_,       m.chargedPionMeV_,
                    m.neutralPionMeV_, m.muonMeV_,      m.chargedKaonMeV_,
                    m.longKaonMeV_,    m.shortKaonMeV_, m.etaMeV_};
    for (double f : fields)
      if (!finite(f) || f <= 0.) return false;
    return m.omegaMeV_ > 2. * m.chargedPionMeV_ + m.neutralPionMeV_ &&
           m.omegaMeV_ > 2. * m.muonMeV_ + m.neutralPionMeV_ &&
           m.phiMeV_ > 2. * m.chargedPionMeV_ + m.neutralPionMeV_ &&
           m.phiMeV_ > 2. * m.muonMeV_ && m.phiMeV_ > 2. * m.chargedKaonMeV_ &&
           m.phiMeV_ > m.longKaonMeV_ + m.shortKaonMeV_ && m.phiMeV_ > m.etaMeV_;
  }
  enum class ResonanceBranch {
    invalid,
    omega_three_pions,
    omega_pi0_gamma,
    omega_two_pions,
    omega_muons_pi0,
    omega_muons,
    phi_charged_kaons,
    phi_neutral_kaons,
    phi_three_pions,
    phi_eta_gamma,
    phi_muons,
    phi_muons_gamma
  };
  struct ResonanceInput {
    double totalMeV_{};
    int pdg_{};
    Direction direction_;
  };
  struct ResonanceOutcome {
    InteractionStatus status_{InteractionStatus::invalid_input};
    ResonanceBranch branch_{ResonanceBranch::invalid};
    int count_{}, trials_{};
    Secondary particle_[3]{};
    double polarizationCosine_[3]{}, polarizationAzimuth_[3]{};
  };
  // RESDEC omega/phi branches, before angular cuts, thinning and TSTACK.
  // C7's published-source thresholds and <= comparisons are kept literally.
  // Unlike rho from RHOGEN, omega/phi use RESDEC, not RHO0DC's dipole law.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION ResonanceOutcome sample_resonance_decay(ResonanceInput q,
                                                                 ResonanceMasses const& m,
                                                                 TRandom& rng,
                                                                 int max_trials = 4096) {
    ResonanceOutcome out;
    if ((q.pdg_ != 223 && q.pdg_ != 333) || !valid_resonance_masses(m) ||
        !finite(q.totalMeV_) || !valid_direction(q.direction_) || max_trials < 1)
      return out;
    double parent = q.pdg_ == 223 ? m.omegaMeV_ : m.phiMeV_;
    if (q.totalMeV_ <= parent) return out; // includes two-body modes' beta denominator
    double u;
    if (!uniform(rng, u)) {
      out.status_ = InteractionStatus::random_failure;
      return out;
    }
    double masses[3]{};
    int ids[3]{}, n = 2;
    bool muons = false;
    using branch_type = ResonanceBranch;
    if (q.pdg_ == 223) {
      if (u <= .8996252) {
        out.branch_ = branch_type::omega_three_pions;
        n = 3;
        masses[0] = masses[1] = m.chargedPionMeV_;
        masses[2] = m.neutralPionMeV_;
        ids[0] = 211;
        ids[1] = -211;
        ids[2] = 111;
      } else if (u <= .9843332) {
        out.branch_ = branch_type::omega_pi0_gamma;
        masses[0] = m.neutralPionMeV_;
        ids[0] = 111;
        ids[1] = 22;
      } else if (u <= .9997739) {
        out.branch_ = branch_type::omega_two_pions;
        masses[0] = masses[1] = m.chargedPionMeV_;
        ids[0] = 211;
        ids[1] = -211;
      } else if (u <= .9999090) {
        out.branch_ = branch_type::omega_muons_pi0;
        n = 3;
        muons = true;
        masses[0] = masses[1] = m.muonMeV_;
        masses[2] = m.neutralPionMeV_;
        ids[0] = -13;
        ids[1] = 13;
        ids[2] = 111;
      } else {
        out.branch_ = branch_type::omega_muons;
        muons = true;
        masses[0] = masses[1] = m.muonMeV_;
        ids[0] = -13;
        ids[1] = 13;
      }
    } else {
      if (u <= .4901808) {
        out.branch_ = branch_type::phi_charged_kaons;
        masses[0] = masses[1] = m.chargedKaonMeV_;
        ids[0] = 321;
        ids[1] = -321;
      } else if (u <= .8330066) {
        out.branch_ = branch_type::phi_neutral_kaons;
        masses[0] = m.longKaonMeV_;
        masses[1] = m.shortKaonMeV_;
        ids[0] = 130;
        ids[1] = 310;
      } else if (u <= .9865765) {
        out.branch_ = branch_type::phi_three_pions;
        n = 3;
        masses[0] = masses[1] = m.chargedPionMeV_;
        masses[2] = m.neutralPionMeV_;
        ids[0] = 211;
        ids[1] = -211;
        ids[2] = 111;
      } else if (u <= .9996981) {
        out.branch_ = branch_type::phi_eta_gamma;
        masses[0] = m.etaMeV_;
        ids[0] = 221;
        ids[1] = 22;
      } else if (u <= .9999857) {
        out.branch_ = branch_type::phi_muons;
        muons = true;
        masses[0] = masses[1] = m.muonMeV_;
        ids[0] = -13;
        ids[1] = 13;
      } else {
        out.branch_ = branch_type::phi_muons_gamma;
        n = 3;
        muons = true;
        masses[0] = masses[1] = m.muonMeV_;
        ids[0] = -13;
        ids[1] = 13;
        ids[2] = 22;
      }
    }
    if (n == 2) {
      auto d = sample_two_body_decay(
          {q.totalMeV_, parent, masses[0], masses[1], ids[0], ids[1], q.direction_,
           muons ? TwoBodyKind::muon_pair : TwoBodyKind::general},
          rng);
      out.status_ = d.status_;
      if (d.status_ != InteractionStatus::success) return out;
      for (int j = 0; j < 2; ++j) {
        out.particle_[j] = d.particle_[j];
        out.polarizationCosine_[j] = d.polarizationCosine_[j];
        out.polarizationAzimuth_[j] = d.polarizationAzimuth_[j];
      }
    } else {
      auto d = sample_uniform_three_body({q.totalMeV_,
                                          parent,
                                          {masses[0], masses[1], masses[2]},
                                          {ids[0], ids[1], ids[2]},
                                          q.direction_},
                                         rng, max_trials);
      out.status_ = d.status_;
      out.trials_ = d.trials_;
      if (d.status_ != InteractionStatus::success) return out;
      for (int j = 0; j < 3; ++j) out.particle_[j] = d.particle_[j];
      if (muons) {
        double a, b;
        if (!uniform(rng, a) || !uniform(rng, b)) {
          out.status_ = InteractionStatus::random_failure;
          return out;
        }
        constexpr double pi = 3.1415926535897932384626433832795;
        // Three-body RESDEC order is mu+,mu-,third, unlike DECAY2's mu-,mu+.
        out.polarizationCosine_[0] = 2. * a - 1.;
        out.polarizationAzimuth_[0] = 2. * pi * b;
        out.polarizationCosine_[1] = -out.polarizationCosine_[0];
        out.polarizationAzimuth_[1] = out.polarizationAzimuth_[0] + pi;
      }
    }
    out.count_ = n;
    out.status_ = InteractionStatus::success;
    return out;
  }
} // namespace corsika::kokkos::egs4
