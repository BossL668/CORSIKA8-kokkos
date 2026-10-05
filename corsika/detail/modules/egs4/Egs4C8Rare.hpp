#pragma once
#include <corsika/detail/modules/egs4/Egs4C8Air.hpp>
#include <corsika/detail/modules/egs4/Egs4RareFinalStates.hpp>
#include <corsika/detail/modules/egs4/Egs4Photonuclear.hpp>
#include <corsika/detail/modules/egs4/Egs4MesonDecay.hpp>
#include <corsika/detail/modules/egs4/Egs4ResonanceDecay.hpp>
#include <corsika/detail/modules/egs4/Egs4NuclearTarget.hpp>

namespace c7_egs4::c8_adapter {
enum class HostVertexKind {
  muon_transport, photonuclear, hadron_transport, c7_vector_meson_decay, photonuclear_many_hadrons,
  native_em // internal queue destination, never an exported host sink
};
struct PhotonuclearPolicy {
  bool enabled{};
  PhotonuclearMasses masses;
  TransverseMomentumTable transverse;
  bool prompt_rho_decay{};
  bool prompt_resonance_decay{};
  ResonanceMasses resonance;
};
KOKKOS_INLINE_FUNCTION bool validPhotonuclearPolicy(PhotonuclearPolicy const& p,RareMaterial const& r) {
  if(!p.enabled)return !p.prompt_rho_decay&&!p.prompt_resonance_decay;
  return validPhotonuclearMasses(p.masses)&&p.masses.proton_MeV==r.proton_mass_MeV&&
    p.masses.charged_pion_MeV==r.charged_pion_mass_MeV&&
    finite(p.masses.phi_MeV)&&p.masses.phi_MeV>0.&&finite(p.masses.omega_MeV)&&p.masses.omega_MeV>0.&&
    finite(p.masses.rho_MeV)&&p.masses.rho_MeV>0.&&p.transverse.intervals>0&&p.transverse.intervals<=50&&
    (!p.prompt_rho_decay||(p.masses.rho_MeV>2.*p.masses.charged_pion_MeV&&p.masses.rho_MeV>2.*r.muon_mass_MeV))&&
    (!p.prompt_resonance_decay||(validResonanceMasses(p.resonance)&&
      p.resonance.omega_MeV==p.masses.omega_MeV&&p.resonance.phi_MeV==p.masses.phi_MeV&&
      p.resonance.charged_pion_MeV==p.masses.charged_pion_MeV&&
      p.resonance.neutral_pion_MeV==p.masses.neutral_pion_MeV&&p.resonance.muon_MeV==r.muon_mass_MeV));
}
struct HostVertex {
  HostVertexKind kind{HostVertexKind::photonuclear};
  C8Particle particle;
  Channel source{Channel::invalid};
  bool new_history{},virtual_photon{}; // virtual_photon describes vertex origin, not output PID
  bool rho_daughter{};
  bool resonance_daughter{};
  Secondary native_secondary; // original C7 local frame, only for native_em
  int target_atomic_number{};
  // Literal C7 polarization scalars, NOT a global C8 spin vector. Muon cuts,
  // spin mapping and hadronic generation remain explicit host responsibilities.
  double polarization_cosine{},polarization_azimuth{};
  // Nonzero only after native PIGEN branch selection. A many-hadron request
  // must not draw PIGEN again. Vector mesons require C7's decay policy.
  int photonuclear_branch{},target_pdg{};
  CounterStream random,hadron_random;
};
struct RareAirTransfer {
  bool valid{};
  InteractionStatus status{InteractionStatus::invalid_input};
  int count{},created{};
  HostVertex vertices[4];
  double target_rest_energy_MeV{}; // once per vertex, not once per child
  // Prompt rho is recorded, never also transported or counted as an energy
  // sink. Its ID is allocated before the final products so genealogy is real.
  bool has_rho_decay{};
  C8Particle rho_parent;
  bool has_resonance_decay{};
  C8Particle resonance_parent;
  int resonance_children{};
};
// Export a secondary at the collision position. Geometry is independent of
// species; use the already validated parent for coordinate conversion, then
// assign the secondary's energy/PID. Never ask validTrack() to transport muons.
KOKKOS_INLINE_FUNCTION bool exportHostSecondary(AirRecord const& parent,Secondary s,
    AirEnvironment const& env,C8Particle& out) {
  if(!finite(s.energy_MeV)||s.energy_MeV<=0.||!validDirection(s.direction))return false;
  auto track=parent.track;track.particle.direction=s.direction;out=parent.metadata;
  if(!exportAirParticle(track,env,out))return false;
  out.pid=s.pdg;out.energy_GeV=s.energy_MeV*.001;return true;
}
// Resolve only an ALREADY SELECTED native C7 rare vertex. Does not redraw the
// channel or interaction distance and never invokes a Fortran/PROPOSAL model.
// On failure leave record/outcome/RNG untouched. IDs are assigned by the queue's
// stable prefix scan; a virtual photon MUST enter an immediate nuclear vertex,
// not the ordinary photon EM queue (which would sample the interaction twice).
KOKKOS_INLINE_FUNCTION RareAirTransfer resolveRareAirRecord(ModelView model,
    AirRecord& record,AirOutcome& outcome,AirEnvironment const& env,
    TransportSettings settings,RareMaterial const& material,PhotonuclearPolicy const& nuclear={}) {
  RareAirTransfer transfer;auto candidate=record;auto result=outcome;
  auto& p=candidate.track.particle;auto channel=p.pending_channel;
  if(outcome.action!=AirAction::needs_host_channel||outcome.history.channel!=channel||
     !validRareMaterial(material)||material.electron_mass_MeV!=model.thresholds.mass_MeV||
     !validSettings(model,settings)||!validPhotonuclearPolicy(nuclear,material))return transfer;
  Secondary secondary[4];
  if(channel==Channel::muon_pair) {
    if(p.pdg!=22||p.energy_MeV<=model.thresholds.muon_pair_MeV)return transfer;
    auto generated=sampleMuonPair(p.energy_MeV,p.direction,material,candidate.random);
    transfer.status=generated.secondaries.status;
    if(transfer.status!=InteractionStatus::success)return transfer;
    transfer.count=transfer.created=2;
    for(int j=0;j<2;++j) {
      secondary[j]=generated.secondaries.particle[j];auto& v=transfer.vertices[j];
      v.kind=HostVertexKind::muon_transport;v.new_history=true;
      v.target_atomic_number=generated.target_atomic_number;
      v.polarization_cosine=generated.polarization_cosine[j];v.polarization_azimuth=generated.polarization_azimuth[j];
    }
    result.action=AirAction::replaced;result.history.action=HistoryAction::replaced;result.history.child_count=0;
  } else if(channel==Channel::electronuclear) {
    if(p.pdg==22||p.energy_MeV<model.thresholds.electronuclear_MeV)return transfer;
    auto generated=sampleElectronuclear(p.energy_MeV,p.pdg,p.direction,material,
      settings.electron_cut_total_MeV-material.electron_mass_MeV,model.thresholds.photonuclear_MeV,candidate.random);
    transfer.status=generated.status;
    if(transfer.status!=InteractionStatus::success&&transfer.status!=InteractionStatus::suppressed)return transfer;
    if(transfer.status==InteractionStatus::success) {
      transfer.count=transfer.created=1;secondary[0]=generated.virtual_photon;
      transfer.vertices[0].new_history=true;transfer.vertices[0].virtual_photon=true;
      p.energy_MeV=generated.residual.energy_MeV;p.direction=generated.residual.direction;
    }
    // Same lepton, same identity and continued random stream, but no old
    // optical depth survives the completed/rejected nuclear vertex.
    p.clock={};p.photon_clock={};result.action=AirAction::active;result.history.action=HistoryAction::active;
  } else if(channel==Channel::photonuclear) {
    if(p.pdg!=22||p.energy_MeV<=model.thresholds.photonuclear_MeV)return transfer;
    transfer.status=InteractionStatus::host_required;transfer.count=1;
    secondary[0]={p.energy_MeV,p.direction,p.pdg}; // original photon identity
    result.action=AirAction::replaced;result.history.action=HistoryAction::replaced;result.history.child_count=0;
  } else return transfer;
  // Generate the nuclear vertex now, while native per-history RNG and the
  // collision state are still available. ELNUCL's virtual photon does not
  // take an EM transport step or select a second interaction distance.
  if(nuclear.enabled&&transfer.count==1&&
      (channel==Channel::electronuclear||channel==Channel::photonuclear)) {
    auto photon=secondary[0];bool virtual_photon=transfer.vertices[0].virtual_photon;
    auto generated=samplePhotonuclear(photon.energy_MeV,photon.direction,nuclear.masses,
      nuclear.transverse,candidate.random,candidate.hadron_random);
    if(generated.status!=InteractionStatus::success&&generated.status!=InteractionStatus::host_required) {
      transfer.valid=false;transfer.status=generated.status;return transfer;
    }
    transfer.status=generated.status;
    if(generated.status==InteractionStatus::success) {
      transfer.count=transfer.created=generated.count;transfer.target_rest_energy_MeV=generated.target_rest_energy_MeV;
      for(int j=0;j<generated.count;++j) {
        secondary[j]=generated.particle[j];auto& v=transfer.vertices[j];v={};v.new_history=true;
        auto id=secondary[j].pdg;
        v.kind=id==113||id==223||id==333?HostVertexKind::c7_vector_meson_decay:HostVertexKind::hadron_transport;
        v.virtual_photon=virtual_photon;v.photonuclear_branch=int(generated.branch);v.target_pdg=generated.target_pdg;
      }
      if(nuclear.prompt_rho_decay&&generated.branch==PhotonuclearBranch::vector_meson&&secondary[0].pdg==113) {
        // RHO0DC(1) runs in the SAME local frame and stream-1 continuation
        // as the native RHOGEN vertex. Exporting to a generic host decay
        // first would lose that frame and may choose a different angle law.
        auto rho=secondary[0];auto recoil=secondary[1];auto recoil_vertex=transfer.vertices[1];
        auto decay=sampleRhoDecay({rho.energy_MeV,nuclear.masses.rho_MeV,nuclear.masses.charged_pion_MeV,
          material.muon_mass_MeV,rho.direction,RhoOrigin::photonuclear},candidate.hadron_random);
        if(decay.status!=InteractionStatus::success){transfer.valid=false;transfer.status=decay.status;return transfer;}
        if(!exportHostSecondary(candidate,rho,env,transfer.rho_parent))return {};
        transfer.has_rho_decay=true;transfer.count=3;transfer.created=4;
        for(int j=0;j<2;++j) {
          secondary[j]=decay.particle[j];auto& v=transfer.vertices[j];v={};
          v.kind=secondary[j].pdg==13||secondary[j].pdg==-13?HostVertexKind::muon_transport:HostVertexKind::hadron_transport;
          v.new_history=true;v.rho_daughter=true;v.virtual_photon=virtual_photon;
          v.photonuclear_branch=int(generated.branch);v.target_pdg=generated.target_pdg;
          v.polarization_cosine=decay.polarization_cosine[j];v.polarization_azimuth=decay.polarization_azimuth[j];
        }
        secondary[2]=recoil;transfer.vertices[2]=recoil_vertex;
      }else if(nuclear.prompt_resonance_decay&&generated.branch==PhotonuclearBranch::vector_meson&&
          (secondary[0].pdg==223||secondary[0].pdg==333)) {
        auto meson=secondary[0];auto recoil=secondary[1];auto recoil_vertex=transfer.vertices[1];
        auto decay=sampleResonanceDecay({meson.energy_MeV,meson.pdg,meson.direction},
          nuclear.resonance,candidate.hadron_random);
        if(decay.status!=InteractionStatus::success){transfer.valid=false;transfer.status=decay.status;return transfer;}
        if(!exportHostSecondary(candidate,meson,env,transfer.resonance_parent))return {};
        transfer.has_resonance_decay=true;transfer.resonance_children=decay.count;
        transfer.count=decay.count+1;transfer.created=decay.count+2;
        for(int j=0;j<decay.count;++j) {
          secondary[j]=decay.particle[j];auto& v=transfer.vertices[j];v={};
          auto id=secondary[j].pdg;
          v.kind=id==22?HostVertexKind::native_em:(id==13||id==-13)?HostVertexKind::muon_transport:HostVertexKind::hadron_transport;
          v.native_secondary=secondary[j];v.new_history=true;v.resonance_daughter=true;v.virtual_photon=virtual_photon;
          v.photonuclear_branch=int(generated.branch);v.target_pdg=generated.target_pdg;
          v.polarization_cosine=decay.polarization_cosine[j];v.polarization_azimuth=decay.polarization_azimuth[j];
        }
        secondary[decay.count]=recoil;transfer.vertices[decay.count]=recoil_vertex;
      }
    }else {
      auto& v=transfer.vertices[0];v.kind=HostVertexKind::photonuclear_many_hadrons;
      v.photonuclear_branch=int(generated.branch); // host consumes THIS selected branch
      auto target=sampleManyHadronTarget(material.composition,candidate.hadron_random);
      if(target.status!=InteractionStatus::success){transfer.valid=false;transfer.status=target.status;return transfer;}
      v.target_pdg=target.pdg;v.target_atomic_number=target.atomic_number;
      // No target rest energy is booked here: the selected host generator
      // owns the final-state medium-energy ledger. Do not count it twice.
    }
  }
  for(int j=0;j<transfer.count;++j) {
    auto& v=transfer.vertices[j];v.source=channel;
    if(!exportHostSecondary(candidate,secondary[j],env,v.particle))return {};
    v.random=candidate.random;v.hadron_random=candidate.hadron_random;
  }
  p.pending_channel=Channel::invalid;result.history.particle=p;
  if(!exportAirParticle(candidate.track,env,candidate.metadata))return {};
  transfer.valid=true;record=candidate;outcome=result;return transfer;
}
KOKKOS_INLINE_FUNCTION bool assignHostHistoryIds(RareAirTransfer& transfer,
    C8Particle const& parent,std::uint64_t first) {
  int assigned=0;
  if(!transfer.valid)return transfer.count==0&&transfer.created==0;
  if(transfer.has_rho_decay&&transfer.has_resonance_decay)return false;
  if(transfer.has_resonance_decay) {
    int n=transfer.resonance_children;
    if(parent.generation>=~std::uint32_t{0}-1||(n!=2&&n!=3)||transfer.count!=n+1||transfer.created!=n+2||
        first>~std::uint64_t{0}-std::uint64_t(transfer.created))return false;
    for(int j=0;j<transfer.count;++j)if(transfer.vertices[j].resonance_daughter!=(j<n))return false;
    auto& meson=transfer.resonance_parent;meson.history_id=first+assigned++;
    meson.parent_history_id=parent.history_id;meson.generation=parent.generation+1;meson.step_id=0;
  }
  if(transfer.has_rho_decay) {
    if(parent.generation>=~std::uint32_t{0}-1||first==~std::uint64_t{0}||transfer.count!=3||transfer.created!=4||
        !transfer.vertices[0].rho_daughter||!transfer.vertices[1].rho_daughter||transfer.vertices[2].rho_daughter)return false;
    auto& rho=transfer.rho_parent;rho.history_id=first+assigned++;rho.parent_history_id=parent.history_id;
    rho.generation=parent.generation+1;rho.step_id=0;
  }
  for(int j=0;j<transfer.count;++j)if(transfer.vertices[j].new_history) {
    if(parent.generation==~std::uint32_t{0}||first>~std::uint64_t{0}-std::uint64_t(assigned))return false;
    auto& p=transfer.vertices[j].particle;p.history_id=first+assigned++;
    auto& v=transfer.vertices[j];
    if(v.rho_daughter&&!transfer.has_rho_decay)return false;
    if(v.resonance_daughter&&!transfer.has_resonance_decay)return false;
    auto const& mother=v.rho_daughter?transfer.rho_parent:v.resonance_daughter?transfer.resonance_parent:parent;
    p.parent_history_id=mother.history_id;p.generation=mother.generation+1;p.step_id=0;
    // New children get independent streams. A virtual photon's continued
    // hadronic calculation must not share the surviving lepton's stream.
    v.random.key={v.random.key.seed,v.random.key.shower_id,p.history_id,0,NativeEgs4RandomDomain,0};
    v.hadron_random.key={v.hadron_random.key.seed,v.hadron_random.key.shower_id,p.history_id,0,NativeEgs4HadronRandomDomain,0};
  }
  return assigned==transfer.created;
}
} // namespace c7_egs4::c8_adapter
