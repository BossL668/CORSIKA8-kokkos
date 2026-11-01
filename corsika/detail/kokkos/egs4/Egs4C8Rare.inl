/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4C8Air.inl>
#include <corsika/detail/kokkos/egs4/Egs4RareFinalStates.inl>
#include <corsika/detail/kokkos/egs4/Egs4Photonuclear.inl>
#include <corsika/detail/kokkos/egs4/Egs4MesonDecay.inl>
#include <corsika/detail/kokkos/egs4/Egs4ResonanceDecay.inl>
#include <corsika/detail/kokkos/egs4/Egs4NuclearTarget.inl>

namespace corsika::kokkos::egs4::c8_adapter {
  enum class HostVertexKind {
    muon_transport,
    photonuclear,
    hadron_transport,
    c7_vector_meson_decay,
    photonuclear_many_hadrons,
    native_em // internal queue destination, never an exported host sink
  };
  struct PhotonuclearPolicy {
    bool enabled_{};
    PhotonuclearMasses masses_;
    TransverseMomentumTable transverse_;
    bool promptRhoDecay_{};
    bool promptResonanceDecay_{};
    ResonanceMasses resonance_;
  };
  KOKKOS_INLINE_FUNCTION bool valid_photonuclear_policy(PhotonuclearPolicy const& p,
                                                        RareMaterial const& r) {
    if (!p.enabled_) return !p.promptRhoDecay_ && !p.promptResonanceDecay_;
    return valid_photonuclear_masses(p.masses_) &&
           p.masses_.protonMeV_ == r.protonMassMeV_ &&
           p.masses_.chargedPionMeV_ == r.chargedPionMassMeV_ &&
           finite(p.masses_.phiMeV_) && p.masses_.phiMeV_ > 0. &&
           finite(p.masses_.omegaMeV_) && p.masses_.omegaMeV_ > 0. &&
           finite(p.masses_.rhoMeV_) && p.masses_.rhoMeV_ > 0. &&
           p.transverse_.intervals_ > 0 && p.transverse_.intervals_ <= 50 &&
           (!p.promptRhoDecay_ || (p.masses_.rhoMeV_ > 2. * p.masses_.chargedPionMeV_ &&
                                   p.masses_.rhoMeV_ > 2. * r.muonMassMeV_)) &&
           (!p.promptResonanceDecay_ ||
            (valid_resonance_masses(p.resonance_) &&
             p.resonance_.omegaMeV_ == p.masses_.omegaMeV_ &&
             p.resonance_.phiMeV_ == p.masses_.phiMeV_ &&
             p.resonance_.chargedPionMeV_ == p.masses_.chargedPionMeV_ &&
             p.resonance_.neutralPionMeV_ == p.masses_.neutralPionMeV_ &&
             p.resonance_.muonMeV_ == r.muonMassMeV_));
  }
  struct HostVertex {
    HostVertexKind kind_{HostVertexKind::photonuclear};
    particle_state_type particle_;
    Channel source_{Channel::invalid};
    bool newHistory_{},
        virtualPhoton_{}; // virtual_photon describes vertex origin, not output PID
    bool rhoDaughter_{};
    bool resonanceDaughter_{};
    Secondary nativeSecondary_; // original C7 local frame, only for native_em
    int targetAtomicNumber_{};
    // Literal C7 polarization scalars, NOT a global C8 spin vector. Muon cuts,
    // spin mapping and hadronic generation remain explicit host responsibilities.
    double polarizationCosine_{}, polarizationAzimuth_{};
    // Nonzero only after native PIGEN branch selection. A many-hadron request
    // must not draw PIGEN again. Vector mesons require C7's decay policy.
    int photonuclearBranch_{}, targetPdg_{};
    CounterStream random_, hadronRandom_;
  };
  struct RareAirTransfer {
    bool valid_{};
    InteractionStatus status_{InteractionStatus::invalid_input};
    int count_{}, created_{};
    HostVertex vertices_[4];
    double targetRestEnergyMeV_{}; // once per vertex, not once per child
    // Prompt rho is recorded, never also transported or counted as an energy
    // sink. Its ID is allocated before the final products so genealogy is real.
    bool hasRhoDecay_{};
    particle_state_type rhoParent_;
    bool hasResonanceDecay_{};
    particle_state_type resonanceParent_;
    int resonanceChildren_{};
  };
  // Export a secondary at the collision position. Geometry is independent of
  // species; use the already validated parent for coordinate conversion, then
  // assign the secondary's energy/PID. Never ask validTrack() to transport muons.
  KOKKOS_INLINE_FUNCTION bool export_host_secondary(AirRecord const& parent, Secondary s,
                                                    AirEnvironment const& env,
                                                    particle_state_type& out) {
    if (!finite(s.energyMeV_) || s.energyMeV_ <= 0. || !valid_direction(s.direction_))
      return false;
    auto track = parent.track_;
    track.particle_.direction_ = s.direction_;
    out = parent.metadata_;
    if (!export_air_particle(track, env, out)) return false;
    out.pid_ = s.pdg_;
    out.energyGeV_ = s.energyMeV_ * .001;
    return true;
  }
  // Resolve only an ALREADY SELECTED native C7 rare vertex. Does not redraw the
  // channel or interaction distance and never invokes a Fortran/PROPOSAL model.
  // On failure leave record/outcome/RNG untouched. IDs are assigned by the queue's
  // stable prefix scan; a virtual photon MUST enter an immediate nuclear vertex,
  // not the ordinary photon EM queue (which would sample the interaction twice).
  KOKKOS_INLINE_FUNCTION RareAirTransfer resolve_rare_air_record(
      ModelView model, AirRecord& record, AirOutcome& outcome, AirEnvironment const& env,
      TransportSettings settings, RareMaterial const& material,
      PhotonuclearPolicy const& nuclear = {}) {
    RareAirTransfer transfer;
    auto candidate = record;
    auto result = outcome;
    auto& p = candidate.track_.particle_;
    auto channel = p.pendingChannel_;
    if (outcome.action_ != AirAction::needs_host_channel ||
        outcome.history_.channel_ != channel || !valid_rare_material(material) ||
        material.electronMassMeV_ != model.thresholds_.massMeV_ ||
        !valid_settings(model, settings) || !valid_photonuclear_policy(nuclear, material))
      return transfer;
    Secondary secondary[4];
    if (channel == Channel::muon_pair) {
      if (p.pdg_ != 22 || p.energyMeV_ <= model.thresholds_.muonPairMeV_) return transfer;
      auto generated =
          sample_muon_pair(p.energyMeV_, p.direction_, material, candidate.random_);
      transfer.status_ = generated.secondaries_.status_;
      if (transfer.status_ != InteractionStatus::success) return transfer;
      transfer.count_ = transfer.created_ = 2;
      for (int j = 0; j < 2; ++j) {
        secondary[j] = generated.secondaries_.particle_[j];
        auto& v = transfer.vertices_[j];
        v.kind_ = HostVertexKind::muon_transport;
        v.newHistory_ = true;
        v.targetAtomicNumber_ = generated.targetAtomicNumber_;
        v.polarizationCosine_ = generated.polarizationCosine_[j];
        v.polarizationAzimuth_ = generated.polarizationAzimuth_[j];
      }
      result.action_ = AirAction::replaced;
      result.history_.action_ = HistoryAction::replaced;
      result.history_.childCount_ = 0;
    } else if (channel == Channel::electronuclear) {
      if (p.pdg_ == 22 || p.energyMeV_ < model.thresholds_.electronuclearMeV_)
        return transfer;
      auto generated =
          sample_electronuclear(p.energyMeV_, p.pdg_, p.direction_, material,
                                settings.electronCutTotalMeV_ - material.electronMassMeV_,
                                model.thresholds_.photonuclearMeV_, candidate.random_);
      transfer.status_ = generated.status_;
      if (transfer.status_ != InteractionStatus::success &&
          transfer.status_ != InteractionStatus::suppressed)
        return transfer;
      if (transfer.status_ == InteractionStatus::success) {
        transfer.count_ = transfer.created_ = 1;
        secondary[0] = generated.virtualPhoton_;
        transfer.vertices_[0].newHistory_ = true;
        transfer.vertices_[0].virtualPhoton_ = true;
        p.energyMeV_ = generated.residual_.energyMeV_;
        p.direction_ = generated.residual_.direction_;
      }
      // Same lepton, same identity and continued random stream, but no old
      // optical depth survives the completed/rejected nuclear vertex.
      p.clock_ = {};
      p.photonClock_ = {};
      result.action_ = AirAction::active;
      result.history_.action_ = HistoryAction::active;
    } else if (channel == Channel::photonuclear) {
      if (p.pdg_ != 22 || p.energyMeV_ <= model.thresholds_.photonuclearMeV_)
        return transfer;
      transfer.status_ = InteractionStatus::host_required;
      transfer.count_ = 1;
      secondary[0] = {p.energyMeV_, p.direction_, p.pdg_}; // original photon identity
      result.action_ = AirAction::replaced;
      result.history_.action_ = HistoryAction::replaced;
      result.history_.childCount_ = 0;
    } else
      return transfer;
    // Generate the nuclear vertex now, while native per-history RNG and the
    // collision state are still available. ELNUCL's virtual photon does not
    // take an EM transport step or select a second interaction distance.
    if (nuclear.enabled_ && transfer.count_ == 1 &&
        (channel == Channel::electronuclear || channel == Channel::photonuclear)) {
      auto photon = secondary[0];
      bool virtual_photon = transfer.vertices_[0].virtualPhoton_;
      auto generated = sample_photonuclear(photon.energyMeV_, photon.direction_,
                                           nuclear.masses_, nuclear.transverse_,
                                           candidate.random_, candidate.hadronRandom_);
      if (generated.status_ != InteractionStatus::success &&
          generated.status_ != InteractionStatus::host_required) {
        transfer.valid_ = false;
        transfer.status_ = generated.status_;
        return transfer;
      }
      transfer.status_ = generated.status_;
      if (generated.status_ == InteractionStatus::success) {
        transfer.count_ = transfer.created_ = generated.count_;
        transfer.targetRestEnergyMeV_ = generated.targetRestEnergyMeV_;
        for (int j = 0; j < generated.count_; ++j) {
          secondary[j] = generated.particle_[j];
          auto& v = transfer.vertices_[j];
          v = {};
          v.newHistory_ = true;
          auto id = secondary[j].pdg_;
          v.kind_ = id == 113 || id == 223 || id == 333
                        ? HostVertexKind::c7_vector_meson_decay
                        : HostVertexKind::hadron_transport;
          v.virtualPhoton_ = virtual_photon;
          v.photonuclearBranch_ = int(generated.branch_);
          v.targetPdg_ = generated.targetPdg_;
        }
        if (nuclear.promptRhoDecay_ &&
            generated.branch_ == PhotonuclearBranch::vector_meson &&
            secondary[0].pdg_ == 113) {
          // RHO0DC(1) runs in the SAME local frame and stream-1 continuation
          // as the native RHOGEN vertex. Exporting to a generic host decay
          // first would lose that frame and may choose a different angle law.
          auto rho = secondary[0];
          auto recoil = secondary[1];
          auto recoil_vertex = transfer.vertices_[1];
          auto decay = sample_rho_decay(
              {rho.energyMeV_, nuclear.masses_.rhoMeV_, nuclear.masses_.chargedPionMeV_,
               material.muonMassMeV_, rho.direction_, RhoOrigin::photonuclear},
              candidate.hadronRandom_);
          if (decay.status_ != InteractionStatus::success) {
            transfer.valid_ = false;
            transfer.status_ = decay.status_;
            return transfer;
          }
          if (!export_host_secondary(candidate, rho, env, transfer.rhoParent_)) return {};
          transfer.hasRhoDecay_ = true;
          transfer.count_ = 3;
          transfer.created_ = 4;
          for (int j = 0; j < 2; ++j) {
            secondary[j] = decay.particle_[j];
            auto& v = transfer.vertices_[j];
            v = {};
            v.kind_ = secondary[j].pdg_ == 13 || secondary[j].pdg_ == -13
                          ? HostVertexKind::muon_transport
                          : HostVertexKind::hadron_transport;
            v.newHistory_ = true;
            v.rhoDaughter_ = true;
            v.virtualPhoton_ = virtual_photon;
            v.photonuclearBranch_ = int(generated.branch_);
            v.targetPdg_ = generated.targetPdg_;
            v.polarizationCosine_ = decay.polarizationCosine_[j];
            v.polarizationAzimuth_ = decay.polarizationAzimuth_[j];
          }
          secondary[2] = recoil;
          transfer.vertices_[2] = recoil_vertex;
        } else if (nuclear.promptResonanceDecay_ &&
                   generated.branch_ == PhotonuclearBranch::vector_meson &&
                   (secondary[0].pdg_ == 223 || secondary[0].pdg_ == 333)) {
          auto meson = secondary[0];
          auto recoil = secondary[1];
          auto recoil_vertex = transfer.vertices_[1];
          auto decay =
              sample_resonance_decay({meson.energyMeV_, meson.pdg_, meson.direction_},
                                     nuclear.resonance_, candidate.hadronRandom_);
          if (decay.status_ != InteractionStatus::success) {
            transfer.valid_ = false;
            transfer.status_ = decay.status_;
            return transfer;
          }
          if (!export_host_secondary(candidate, meson, env, transfer.resonanceParent_))
            return {};
          transfer.hasResonanceDecay_ = true;
          transfer.resonanceChildren_ = decay.count_;
          transfer.count_ = decay.count_ + 1;
          transfer.created_ = decay.count_ + 2;
          for (int j = 0; j < decay.count_; ++j) {
            secondary[j] = decay.particle_[j];
            auto& v = transfer.vertices_[j];
            v = {};
            auto id = secondary[j].pdg_;
            v.kind_ = id == 22
                          ? HostVertexKind::native_em
                          : (id == 13 || id == -13) ? HostVertexKind::muon_transport
                                                    : HostVertexKind::hadron_transport;
            v.nativeSecondary_ = secondary[j];
            v.newHistory_ = true;
            v.resonanceDaughter_ = true;
            v.virtualPhoton_ = virtual_photon;
            v.photonuclearBranch_ = int(generated.branch_);
            v.targetPdg_ = generated.targetPdg_;
            v.polarizationCosine_ = decay.polarizationCosine_[j];
            v.polarizationAzimuth_ = decay.polarizationAzimuth_[j];
          }
          secondary[decay.count_] = recoil;
          transfer.vertices_[decay.count_] = recoil_vertex;
        }
      } else {
        auto& v = transfer.vertices_[0];
        v.kind_ = HostVertexKind::photonuclear_many_hadrons;
        v.photonuclearBranch_ =
            int(generated.branch_); // host consumes THIS selected branch
        auto target =
            sample_many_hadron_target(material.composition_, candidate.hadronRandom_);
        if (target.status_ != InteractionStatus::success) {
          transfer.valid_ = false;
          transfer.status_ = target.status_;
          return transfer;
        }
        v.targetPdg_ = target.pdg_;
        v.targetAtomicNumber_ = target.atomicNumber_;
        // No target rest energy is booked here: the selected host generator
        // owns the final-state medium-energy ledger. Do not count it twice.
      }
    }
    for (int j = 0; j < transfer.count_; ++j) {
      auto& v = transfer.vertices_[j];
      v.source_ = channel;
      if (!export_host_secondary(candidate, secondary[j], env, v.particle_)) return {};
      v.random_ = candidate.random_;
      v.hadronRandom_ = candidate.hadronRandom_;
    }
    p.pendingChannel_ = Channel::invalid;
    result.history_.particle_ = p;
    if (!export_air_particle(candidate.track_, env, candidate.metadata_)) return {};
    transfer.valid_ = true;
    record = candidate;
    outcome = result;
    return transfer;
  }
  KOKKOS_INLINE_FUNCTION bool assign_host_history_ids(RareAirTransfer& transfer,
                                                      particle_state_type const& parent,
                                                      std::uint64_t first) {
    int assigned = 0;
    if (!transfer.valid_) return transfer.count_ == 0 && transfer.created_ == 0;
    if (transfer.hasRhoDecay_ && transfer.hasResonanceDecay_) return false;
    if (transfer.hasResonanceDecay_) {
      int n = transfer.resonanceChildren_;
      if (parent.generation_ >= ~std::uint32_t{0} - 1 || (n != 2 && n != 3) ||
          transfer.count_ != n + 1 || transfer.created_ != n + 2 ||
          first > ~std::uint64_t{0} - std::uint64_t(transfer.created_))
        return false;
      for (int j = 0; j < transfer.count_; ++j)
        if (transfer.vertices_[j].resonanceDaughter_ != (j < n)) return false;
      auto& meson = transfer.resonanceParent_;
      meson.historyId_ = first + assigned++;
      meson.parentHistoryId_ = parent.historyId_;
      meson.generation_ = parent.generation_ + 1;
      meson.stepId_ = 0;
    }
    if (transfer.hasRhoDecay_) {
      if (parent.generation_ >= ~std::uint32_t{0} - 1 || first == ~std::uint64_t{0} ||
          transfer.count_ != 3 || transfer.created_ != 4 ||
          !transfer.vertices_[0].rhoDaughter_ || !transfer.vertices_[1].rhoDaughter_ ||
          transfer.vertices_[2].rhoDaughter_)
        return false;
      auto& rho = transfer.rhoParent_;
      rho.historyId_ = first + assigned++;
      rho.parentHistoryId_ = parent.historyId_;
      rho.generation_ = parent.generation_ + 1;
      rho.stepId_ = 0;
    }
    for (int j = 0; j < transfer.count_; ++j)
      if (transfer.vertices_[j].newHistory_) {
        if (parent.generation_ == ~std::uint32_t{0} ||
            first > ~std::uint64_t{0} - std::uint64_t(assigned))
          return false;
        auto& p = transfer.vertices_[j].particle_;
        p.historyId_ = first + assigned++;
        auto& v = transfer.vertices_[j];
        if (v.rhoDaughter_ && !transfer.hasRhoDecay_) return false;
        if (v.resonanceDaughter_ && !transfer.hasResonanceDecay_) return false;
        auto const& mother =
            v.rhoDaughter_ ? transfer.rhoParent_
                           : v.resonanceDaughter_ ? transfer.resonanceParent_ : parent;
        p.parentHistoryId_ = mother.historyId_;
        p.generation_ = mother.generation_ + 1;
        p.stepId_ = 0;
        // New children get independent streams. A virtual photon's continued
        // hadronic calculation must not share the surviving lepton's stream.
        v.random_.key_ = {v.random_.key_.seed_,   v.random_.key_.showerId_,
                          p.historyId_,           0,
                          NativeEgs4RandomDomain, 0};
        v.hadronRandom_.key_ = {v.hadronRandom_.key_.seed_,
                                v.hadronRandom_.key_.showerId_,
                                p.historyId_,
                                0,
                                NativeEgs4HadronRandomDomain,
                                0};
      }
    return assigned == transfer.created_;
  }
} // namespace corsika::kokkos::egs4::c8_adapter
