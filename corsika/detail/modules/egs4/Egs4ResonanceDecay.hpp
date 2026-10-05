#pragma once
#include <corsika/detail/modules/egs4/Egs4TwoBodyDecay.hpp>
#include <corsika/detail/modules/egs4/Egs4ThreeBodyDecay.hpp>

namespace c7_egs4 {
struct ResonanceMasses {
  double omega_MeV{},phi_MeV{},charged_pion_MeV{},neutral_pion_MeV{},muon_MeV{};
  double charged_kaon_MeV{},long_kaon_MeV{},short_kaon_MeV{},eta_MeV{};
};
KOKKOS_INLINE_FUNCTION bool validResonanceMasses(ResonanceMasses const& m) {
  double fields[]{m.omega_MeV,m.phi_MeV,m.charged_pion_MeV,m.neutral_pion_MeV,m.muon_MeV,
    m.charged_kaon_MeV,m.long_kaon_MeV,m.short_kaon_MeV,m.eta_MeV};
  for(double f:fields)if(!finite(f)||f<=0.)return false;
  return m.omega_MeV>2.*m.charged_pion_MeV+m.neutral_pion_MeV&&
    m.omega_MeV>2.*m.muon_MeV+m.neutral_pion_MeV&&
    m.phi_MeV>2.*m.charged_pion_MeV+m.neutral_pion_MeV&&m.phi_MeV>2.*m.muon_MeV&&
    m.phi_MeV>2.*m.charged_kaon_MeV&&m.phi_MeV>m.long_kaon_MeV+m.short_kaon_MeV&&m.phi_MeV>m.eta_MeV;
}
enum class ResonanceBranch {
  invalid, omega_three_pions, omega_pi0_gamma, omega_two_pions, omega_muons_pi0, omega_muons,
  phi_charged_kaons, phi_neutral_kaons, phi_three_pions, phi_eta_gamma, phi_muons, phi_muons_gamma
};
struct ResonanceInput {double total_MeV{};int pdg{};Direction direction;};
struct ResonanceOutcome {
  InteractionStatus status{InteractionStatus::invalid_input};
  ResonanceBranch branch{ResonanceBranch::invalid};
  int count{},trials{};
  Secondary particle[3]{};
  double polarization_cosine[3]{},polarization_azimuth[3]{};
};
// RESDEC omega/phi branches, before angular cuts, thinning and TSTACK.
// C7's published-source thresholds and <= comparisons are kept literally.
// Unlike rho from RHOGEN, omega/phi use RESDEC, not RHO0DC's dipole law.
template<class Random>
KOKKOS_INLINE_FUNCTION ResonanceOutcome sampleResonanceDecay(ResonanceInput q,
    ResonanceMasses const& m,Random& rng,int max_trials=4096) {
  ResonanceOutcome out;
  if((q.pdg!=223&&q.pdg!=333)||!validResonanceMasses(m)||!finite(q.total_MeV)||
     !validDirection(q.direction)||max_trials<1)return out;
  double parent=q.pdg==223?m.omega_MeV:m.phi_MeV;
  if(q.total_MeV<=parent)return out; // includes two-body modes' beta denominator
  double u;if(!uniform(rng,u)){out.status=InteractionStatus::random_failure;return out;}
  double masses[3]{};int ids[3]{},n=2;bool muons=false;
  using B=ResonanceBranch;
  if(q.pdg==223) {
    if(u<=.8996252) {
      out.branch=B::omega_three_pions;n=3;masses[0]=masses[1]=m.charged_pion_MeV;masses[2]=m.neutral_pion_MeV;
      ids[0]=211;ids[1]=-211;ids[2]=111;
    }else if(u<=.9843332) {
      out.branch=B::omega_pi0_gamma;masses[0]=m.neutral_pion_MeV;ids[0]=111;ids[1]=22;
    }else if(u<=.9997739) {
      out.branch=B::omega_two_pions;masses[0]=masses[1]=m.charged_pion_MeV;ids[0]=211;ids[1]=-211;
    }else if(u<=.9999090) {
      out.branch=B::omega_muons_pi0;n=3;muons=true;masses[0]=masses[1]=m.muon_MeV;masses[2]=m.neutral_pion_MeV;
      ids[0]=-13;ids[1]=13;ids[2]=111;
    }else {out.branch=B::omega_muons;muons=true;masses[0]=masses[1]=m.muon_MeV;ids[0]=-13;ids[1]=13;}
  }else {
    if(u<=.4901808) {
      out.branch=B::phi_charged_kaons;masses[0]=masses[1]=m.charged_kaon_MeV;ids[0]=321;ids[1]=-321;
    }else if(u<=.8330066) {
      out.branch=B::phi_neutral_kaons;masses[0]=m.long_kaon_MeV;masses[1]=m.short_kaon_MeV;ids[0]=130;ids[1]=310;
    }else if(u<=.9865765) {
      out.branch=B::phi_three_pions;n=3;masses[0]=masses[1]=m.charged_pion_MeV;masses[2]=m.neutral_pion_MeV;
      ids[0]=211;ids[1]=-211;ids[2]=111;
    }else if(u<=.9996981) {
      out.branch=B::phi_eta_gamma;masses[0]=m.eta_MeV;ids[0]=221;ids[1]=22;
    }else if(u<=.9999857) {
      out.branch=B::phi_muons;muons=true;masses[0]=masses[1]=m.muon_MeV;ids[0]=-13;ids[1]=13;
    }else {
      out.branch=B::phi_muons_gamma;n=3;muons=true;masses[0]=masses[1]=m.muon_MeV;ids[0]=-13;ids[1]=13;ids[2]=22;
    }
  }
  if(n==2) {
    auto d=sampleTwoBodyDecay({q.total_MeV,parent,masses[0],masses[1],ids[0],ids[1],q.direction,
      muons?TwoBodyKind::muon_pair:TwoBodyKind::general},rng);
    out.status=d.status;if(d.status!=InteractionStatus::success)return out;
    for(int j=0;j<2;++j) {
      out.particle[j]=d.particle[j];out.polarization_cosine[j]=d.polarization_cosine[j];
      out.polarization_azimuth[j]=d.polarization_azimuth[j];
    }
  }else {
    auto d=sampleUniformThreeBody({q.total_MeV,parent,{masses[0],masses[1],masses[2]},
      {ids[0],ids[1],ids[2]},q.direction},rng,max_trials);
    out.status=d.status;out.trials=d.trials;if(d.status!=InteractionStatus::success)return out;
    for(int j=0;j<3;++j)out.particle[j]=d.particle[j];
    if(muons) {
      double a,b;if(!uniform(rng,a)||!uniform(rng,b)){out.status=InteractionStatus::random_failure;return out;}
      constexpr double pi=3.1415926535897932384626433832795;
      // Three-body RESDEC order is mu+,mu-,third, unlike DECAY2's mu-,mu+.
      out.polarization_cosine[0]=2.*a-1.;out.polarization_azimuth[0]=2.*pi*b;
      out.polarization_cosine[1]=-out.polarization_cosine[0];out.polarization_azimuth[1]=out.polarization_azimuth[0]+pi;
    }
  }
  out.count=n;out.status=InteractionStatus::success;return out;
}
} // namespace c7_egs4
