#pragma once
#include "Egs4MesonDecay.hpp"

namespace c7_egs4 {
// C7 DECAY1/DECAY2 before angular cuts and TSTACK. Not a replacement
// isotropic Pythia call: preserve the source arithmetic, draw order,
// ADDANG3 convention and second-product-first output order.
enum class TwoBodyKind { general, muon_pair };
struct TwoBodyInput {
  double total_MeV{},parent_mass_MeV{},first_mass_MeV{},second_mass_MeV{};
  int first_pdg{},second_pdg{};
  Direction direction;
  TwoBodyKind kind{TwoBodyKind::general};
};
struct TwoBodyOutcome {
  InteractionStatus status{InteractionStatus::invalid_input};
  int count{};
  Secondary particle[2]{}; // source M4, then M3
  double polarization_cosine[2]{},polarization_azimuth[2]{};
  double cm_cosine{};
};
template<class Random>
KOKKOS_INLINE_FUNCTION TwoBodyOutcome sampleTwoBodyDecay(TwoBodyInput q,Random& rng) {
  TwoBodyOutcome out;
  if(!finite(q.total_MeV)||!finite(q.parent_mass_MeV)||!finite(q.first_mass_MeV)||
     !finite(q.second_mass_MeV)||q.first_mass_MeV<=0.||q.second_mass_MeV<0.||
     q.parent_mass_MeV<=q.first_mass_MeV+q.second_mass_MeV||q.total_MeV<=q.parent_mass_MeV||
     !q.first_pdg||!q.second_pdg||!validDirection(q.direction)||
     (q.kind!=TwoBodyKind::general&&q.kind!=TwoBodyKind::muon_pair))return out;
  if(q.kind==TwoBodyKind::muon_pair&&
     (q.first_pdg!=-13||q.second_pdg!=13||q.first_mass_MeV!=q.second_mass_MeV))return out;
  // DECAY1/2 use GeV masses in C7. Retain that internal convention, while
  // the EGS4-native public carrier consistently uses total MeV energy.
  double m0=q.parent_mass_MeV*.001,m3=q.first_mass_MeV*.001,m4=q.second_mass_MeV*.001;
  double gamma=q.total_MeV/q.parent_mass_MeV;
  if(!finite(gamma)||gamma<=1.)return out; // literal formula divides by beta
  double beta=::sqrt((gamma-1.)*(gamma+1.))/gamma;
  double a=q.kind==TwoBodyKind::muon_pair?(m0*m0)/(2.*m0):
    (m0*m0+m3*m3-m4*m4)/(2.*m0);
  double a1=a*a-m3*m3,a2=1.+a1/(m3*m3),a2a=::sqrt(a2),a3=::sqrt(1.-1./a2);
  double work1=gamma*a2a,work2=a3*beta*work1,u[4]{};
  int draws=q.kind==TwoBodyKind::muon_pair?4:2;
  for(int i=0;i<draws;++i)if(!uniform(rng,u[i])) {
    out.status=InteractionStatus::random_failure;return out;
  }
  double cm=2.*u[0]-1.,g3=work1+work2*cm,g4,ct4,e4;
  if(m4!=0.) {
    g4=(m0*gamma-m3*g3)/m4;
    double a4=(m0*m0+m4*m4-m3*m3)/(2.*m0*m4);
    if(!finite(g4)||g4<=1.)return out;
    ct4=(gamma*g4-a4)/(beta*gamma*::sqrt((g4-1.)*(g4+1.)));
    e4=g4*m4;
  }else {
    e4=m0*gamma-m3*g3;
    ct4=(beta-cm)/(1.-beta*cm);
  }
  if(!finite(g3)||g3<=1.)return out;
  double ct3=(gamma*g3-a2a)/(beta*gamma*::sqrt((g3-1.)*(g3+1.)));
  if(!finite(ct3)||!finite(ct4))return out;
  ct3=ct3<1.?ct3:1.;ct4=ct4<1.?ct4:1.;
  if(ct3< -1.||ct4< -1.||!finite(e4)||e4<=0.)return out;
  constexpr double pi=3.1415926535897932384626433832795;
  double phi=2.*pi*u[1];
  out.particle[0]={e4*1000.,addHadronicAngle(q.direction,ct4,phi),q.second_pdg};
  out.particle[1]={g3*m3*1000.,addHadronicAngle(q.direction,ct3,phi+pi),q.first_pdg};
  if(q.kind==TwoBodyKind::muon_pair) {
    out.polarization_cosine[0]=2.*u[2]-1.;out.polarization_azimuth[0]=2.*pi*u[3];
    out.polarization_cosine[1]=-out.polarization_cosine[0];
    out.polarization_azimuth[1]=out.polarization_azimuth[0]+pi;
  }
  for(auto const& p:out.particle)if(!finite(p.energy_MeV)||!validDirection(p.direction))return {};
  out.cm_cosine=cm;out.count=2;out.status=InteractionStatus::success;return out;
}
} // namespace c7_egs4
