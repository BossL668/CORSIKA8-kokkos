#pragma once
#include <corsika/detail/modules/egs4/Egs4Interactions.hpp>

namespace c7_egs4 {
// C7 RHO0DC before angular cuts, deposition and TSTACK. All draws use
// stream 1, supplied explicitly by the caller. No shared SAVE/COMMON state.
enum class RhoOrigin { hadronic=0, photonuclear=1 };
struct RhoDecayInput {
  double total_MeV{},rho_mass_MeV{},pion_mass_MeV{},muon_mass_MeV{};
  Direction direction;
  RhoOrigin origin{RhoOrigin::photonuclear};
};
struct RhoDecayOutcome {
  InteractionStatus status{InteractionStatus::invalid_input};
  int count{},trials{};
  Secondary particle[2]{}; // negative first, positive second; NOT energy sorted
  double polarization_cosine[2]{},polarization_azimuth[2]{};
  double cm_cosine{};
};
// ADDANG3, in the EGS direction convention (U,V,W): C7's arguments are
// (W,U,-V). Unlike UPHI it has a 1e-40 polar branch and one normalization
// iteration. Preserve both; simply reusing rotate() changes C7 arithmetic.
KOKKOS_INLINE_FUNCTION Direction addHadronicAngle(Direction q,double ct,double phi) {
  double st=::sqrt((1.-ct)*(1.+ct)),s2=q.x*q.x+q.y*q.y;
  double x,y,w;
  if(s2<1.e-40) {
    x=st*::cos(-phi);y=st*::sin(-phi);w=ct*q.z;
  }else {
    double px=st*::cos(-phi),py=st*::sin(-phi),s=::sqrt(s2);
    double cd=q.x*(1./s),sd=q.y*(1./s);
    x=q.z*cd*px-sd*py+q.x*ct;
    y=-q.z*sd*px-cd*py-q.y*ct;
    w=-s*px+q.z*ct;
  }
  double inverse=1.5-.5*(x*x+y*y+w*w);
  x=larger(-1.,inverse*x);x=x<1.?x:1.;
  y=larger(-1.,inverse*y);y=y<1.?y:1.;
  w=larger(-1.,inverse*w);w=w<1.?w:1.;
  return {x,-y,w};
}
template<class Random>
KOKKOS_INLINE_FUNCTION RhoDecayOutcome sampleRhoDecay(RhoDecayInput q,Random& rng,int max_trials=4096) {
  RhoDecayOutcome out;
  if(!finite(q.total_MeV)||!finite(q.rho_mass_MeV)||!finite(q.pion_mass_MeV)||
     !finite(q.muon_mass_MeV)||q.rho_mass_MeV<=0.||q.pion_mass_MeV<=0.||q.muon_mass_MeV<=0.||
     q.rho_mass_MeV<=2.*q.pion_mass_MeV||q.rho_mass_MeV<=2.*q.muon_mass_MeV||
     q.total_MeV<=q.rho_mass_MeV||!validDirection(q.direction)||max_trials<1||
     (q.origin!=RhoOrigin::hadronic&&q.origin!=RhoOrigin::photonuclear))return out;
  // The source lab-angle formula is undefined at gamma=1. Return an explicit
  // error instead of inventing a rest-frame convention for a zero direction.
  double gamma=q.total_MeV/q.rho_mass_MeV;
  if(!finite(gamma)||gamma<=1.)return out;
  double u,v;
  if(!uniform(rng,u)){out.status=InteractionStatus::random_failure;return out;}
  bool muon=u>=.999955;
  double mass=muon?q.muon_mass_MeV:q.pion_mass_MeV,polar=0.,azimuth=0.;
  constexpr double pi=3.1415926535897932384626433832795;
  if(muon) {
    if(!uniform(rng,u)||!uniform(rng,v)){out.status=InteractionStatus::random_failure;return out;}
    polar=2.*u-1.;azimuth=2.*pi*v;
  }
  double beta=::sqrt((gamma-1.)*(gamma+1.))/gamma;
  double a=.5*q.rho_mass_MeV/mass,work1=gamma*a;
  double work2=beta*gamma*::sqrt((a-1.)*(a+1.)),cm=0.;
  bool accepted=false;
  for(int trial=1;trial<=max_trials;++trial) {
    out.trials=trial;
    if(!uniform(rng,u)||!uniform(rng,v)){out.status=InteractionStatus::random_failure;return out;}
    cm=2.*u-1.;
    // Literal RHO0DC applies the origin-dependent rejection to BOTH pion
    // and muon branches, despite its header comment about isotropic muons.
    if(q.origin==RhoOrigin::photonuclear&&v>1.-.8836*cm*cm)continue;
    accepted=true;break;
  }
  if(!accepted){out.status=InteractionStatus::trial_limit;return out;}
  double g3=work1+work2*cm,g4=gamma*(q.rho_mass_MeV/mass)-g3;
  if(!finite(g3)||!finite(g4)||g3<=1.||g4<=1.)return out;
  double ct4=(gamma*g4-a)/(beta*gamma*::sqrt((g4-1.)*(g4+1.)));
  ct4=ct4<1.?ct4:1.;
  if(!uniform(rng,u)){out.status=InteractionStatus::random_failure;return out;}
  double phi=2.*pi*u;
  double ct3=(gamma*g3-a)/(beta*gamma*::sqrt((g3-1.)*(g3+1.)));
  ct3=ct3<1.?ct3:1.;
  if(!finite(ct3)||!finite(ct4)||ct3< -1.||ct4< -1.)return out;
  out.particle[0]={g4*mass,addHadronicAngle(q.direction,ct4,phi),muon?13:-211};
  out.particle[1]={g3*mass,addHadronicAngle(q.direction,ct3,phi+pi),muon?-13:211};
  if(muon) {
    out.polarization_cosine[0]=polar;out.polarization_azimuth[0]=azimuth;
    out.polarization_cosine[1]=-polar;out.polarization_azimuth[1]=azimuth+pi;
  }
  for(auto const& p:out.particle)if(!finite(p.energy_MeV)||!validDirection(p.direction))return {};
  out.cm_cosine=cm;out.count=2;out.status=InteractionStatus::success;return out;
}
} // namespace c7_egs4
