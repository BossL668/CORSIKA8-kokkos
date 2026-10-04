#pragma once
#include "Egs4Interactions.hpp"

namespace c7_egs4 {
// Native C7 rare-vertex arithmetic. Hadronic generation/stack transfer remains
// a separate host adapter; no PROPOSAL final state is substituted here.
struct RareMaterial {
  double electron_mass_MeV{},muon_mass_MeV{},charged_pion_mass_MeV{},proton_mass_MeV{};
  double average_atomic_weight{};
  double composition[3]{}; // N, O, Ar number abundances, as in C7 COMPOS
};
// Production CounterStream is open-(0,1). Enforce that contract here because
// MUPAIR uses 1/t and log terms; a malformed provider must not yield NaN muons.
template<class Random>
KOKKOS_INLINE_FUNCTION bool rareUniform(Random& rng,double& value) {
  return uniform(rng,value)&&value>0.;
}
KOKKOS_INLINE_FUNCTION bool validRareMaterial(RareMaterial const& m) {
  if(!finite(m.electron_mass_MeV)||m.electron_mass_MeV<=0.||
     !finite(m.muon_mass_MeV)||m.muon_mass_MeV<=m.electron_mass_MeV||
     !finite(m.charged_pion_mass_MeV)||m.charged_pion_mass_MeV<=0.||
     !finite(m.proton_mass_MeV)||m.proton_mass_MeV<=0.||
     !finite(m.average_atomic_weight)||m.average_atomic_weight<=0.)return false;
  double sum=0.;for(int i=0;i<3;++i){if(!finite(m.composition[i])||m.composition[i]<0.)return false;sum+=m.composition[i];}
  return finite(sum)&&sum>0.;
}
struct MuonPairOutcome {
  Secondaries secondaries;
  int target_atomic_number{};
  int energy_trials{},angle_trials{},azimuth_trials{};
  double polarization_cosine[2]{},polarization_azimuth[2]{};
};
// MUPAIR before MUPROP: both generated muons, including polarization, are
// returned even below the transport cut. A host policy must explicitly apply
// muon cuts/acceptance and book rejected energy; this is not muon transport.
template<class Random>
KOKKOS_INLINE_FUNCTION MuonPairOutcome sampleMuonPair(double energy,Direction direction,
    RareMaterial const& material,Random& rng,int max_trials=4096) {
  MuonPairOutcome out;auto& r=out.secondaries;
  if(!validRareMaterial(material)||!validDirection(direction)||!finite(energy)||
     energy<=4.*material.muon_mass_MeV||max_trials<1)return out;
  constexpr double one_third=.333333333333333,pi=3.1415926535897932384626433832795;
  double a[]={14.,16.,40.},z[]={7.,8.,18.};
  double sums[3],sum=0.;for(int i=0;i<3;++i)sums[i]=sum+=material.composition[i]*::pow(z[i],1.735);
  double u,v;if(!rareUniform(rng,u)){r.status=InteractionStatus::random_failure;return out;}
  int target=u*sum<=sums[0]?0:u*sum<=sums[1]?1:2;out.target_atomic_number=int(z[target]);
  double mu=material.muon_mass_MeV,m=material.electron_mass_MeV;
  double xmin=.5-::sqrt(.25-mu/energy),xmax=.5+::sqrt(.25-mu/energy);
  double aexp=::pow(a[target],.27),dn=1.54*aexp,c1num=(.335*aexp)*(.335*aexp);
  double zexp=183.*::pow(z[target],-one_third),se=::sqrt(::exp(1.));
  double winf=zexp*mu/(dn*m),delmax=2.*mu*mu/energy;
  double wmax=winf*(1.+(dn*se-2.)*delmax/mu)/(1.+zexp*se*delmax/m);
  if(!finite(wmax)||wmax<=1.)return out;
  double xsmall=0.,xlarge=0.,product=0.,pxxi=0.;bool accepted=false;
  for(int trial=1;trial<=max_trials;++trial) {
    out.energy_trials=trial;
    if(!rareUniform(rng,u)||!rareUniform(rng,v)){r.status=InteractionStatus::random_failure;return out;}
    xsmall=xmin+.5*u*(xmax-xmin);xlarge=1.-xsmall;product=xlarge*xsmall;pxxi=1./(energy*product);
    double delta=.5*mu*mu*pxxi;
    double w=winf*(1.+(dn*se-2.)*delta/mu)/(1.+zexp*se*delta/m);
    w=larger(w,1.);double probability=(1.-4.*one_third*product)*::log(w)/::log(wmax);
    if(v<=probability){accepted=true;break;}
  }
  if(!accepted){r.status=InteractionStatus::trial_limit;return out;}
  double c1=c1num*mu*pxxi,f1max=(1.-product)/(1.+c1);
  // An invalid polar angle returns to step 2, retaining the sampled energies.
  for(int retry=0;retry<max_trials;++retry) {
    double t=0.;accepted=false;
    for(int trial=0;trial<max_trials;++trial) {
      ++out.angle_trials;
      if(!rareUniform(rng,u)||!rareUniform(rng,v)){r.status=InteractionStatus::random_failure;return out;}
      double f1=(1.-2.*product+4.*product*u*(1.-u))/(1.+c1/(u*u));
      if(f1<0.||f1>f1max)f1=0.;
      if(v*f1max<=f1){t=u;accepted=true;break;}
    }
    if(!accepted){r.status=InteractionStatus::trial_limit;return out;}
    double f2max=1.-2.*product*(1.-4.*t*(1.-t)),psi=0.;accepted=false;
    for(int trial=0;trial<max_trials;++trial) {
      ++out.azimuth_trials;
      if(!rareUniform(rng,u)||!rareUniform(rng,v)){r.status=InteractionStatus::random_failure;return out;}
      psi=u*(2.*pi);double f2=1.-2.*product+4.*product*t*(1.-t)*(1.+::cos(2.*psi));
      if(f2<0.||f2>f2max)f2=0.;
      if(v*f2max<=f2){accepted=true;break;}
    }
    if(!accepted){r.status=InteractionStatus::trial_limit;return out;}
    double term2=m/(zexp*mu),term1=mu/(2.*energy*product*t),square=term1*term1+term2*term2;
    double c2=4./::sqrt(product)*square*square,rhomax=1.9/aexp*(1./t-1.);
    double beta4=::log((c2+rhomax*rhomax)/c2);
    if(!rareUniform(rng,u)){r.status=InteractionStatus::random_failure;return out;}
    double rho=::pow(c2*(::exp(u*beta4)-1.),.25),uu=::sqrt(1./t-1.);
    double aux=.5*rho*::cos(psi),theta_high=mu/(energy*xlarge)*(uu+aux),theta_low=mu/(energy*xsmall)*(uu-aux);
    if(::fabs(theta_high)>pi||::fabs(theta_low)>pi)continue;
    double phi1=::sin(psi)*.5*rho/uu;
    if(!rareUniform(rng,u)){r.status=InteractionStatus::random_failure;return out;}
    double phi=u*(2.*pi);
    r.particle[0]={energy*xlarge,rotate(direction,::sin(theta_high),::cos(theta_high),::sin(phi+phi1),::cos(phi+phi1)),13};
    r.particle[1]={energy*xsmall,rotate(direction,-::sin(theta_low),::cos(theta_low),::sin(phi-phi1),::cos(phi-phi1)),-13};
    double charge,pol,az;
    if(!rareUniform(rng,charge)||!rareUniform(rng,pol)||!rareUniform(rng,az)){r.status=InteractionStatus::random_failure;return out;}
    if(charge>.5){r.particle[0].pdg=-13;r.particle[1].pdg=13;}
    // C7 exports the lower-energy muon first, then reverses its polarization.
    out.polarization_cosine[1]=2.*pol-1.;out.polarization_azimuth[1]=2.*pi*az;
    out.polarization_cosine[0]=-out.polarization_cosine[1];out.polarization_azimuth[0]=out.polarization_azimuth[1]+pi;
    for(int i=0;i<2;++i)if(!finite(r.particle[i].energy_MeV)||!validDirection(r.particle[i].direction))return out;
    r.count=2;r.status=InteractionStatus::success;
    r.trials=out.energy_trials+out.angle_trials+out.azimuth_trials;return out;
  }
  r.status=InteractionStatus::trial_limit;return out;
}

struct ElectronuclearOutcome {
  InteractionStatus status{InteractionStatus::invalid_input};
  Secondary residual,virtual_photon;
  int trials{};
};
// ELNUCL energy-sharing and forward directions. The photon is a request for
// an immediate photonuclear vertex, NOT an ordinary gamma for EM transport.
// C7's 1000 rejected proposals leave the lepton unchanged; preserve that rule.
template<class Random>
KOKKOS_INLINE_FUNCTION ElectronuclearOutcome sampleElectronuclear(double energy,int pdg,Direction direction,
    RareMaterial const& material,double electron_kinetic_cut_MeV,double photonuclear_threshold_MeV,Random& rng) {
  ElectronuclearOutcome out;
  if(!validRareMaterial(material)||!validDirection(direction)||!finite(energy)||
     energy<material.electron_mass_MeV||(pdg!=11&&pdg!=-11)||
     !finite(electron_kinetic_cut_MeV)||electron_kinetic_cut_MeV<0.||
     !finite(photonuclear_threshold_MeV)||photonuclear_threshold_MeV<0.)return out;
  out.residual={energy,direction,pdg};
  if(energy-material.electron_mass_MeV<=electron_kinetic_cut_MeV){out.status=InteractionStatus::suppressed;return out;}
  constexpr double one_third=.333333333333333,pi=3.1415926535897932384626433832795;
  double ee=energy*.001,m=material.electron_mass_MeV*.001,mp=material.proton_mass_MeV*.001,pion=material.charged_pion_mass_MeV*.001;
  double minimum=(pion+.5*pion*pion/mp)/ee,maximum=1.-(mp+m*m/mp)*.5/ee;
  // The literal source does not guard the fractional powers of these bounds.
  // Reject invalid phase space explicitly rather than propagate a NaN photon.
  if(!finite(minimum)||!finite(maximum)||minimum<=0.||maximum<=minimum||maximum>=1.)return out;
  double le=::log10(ee),coef,factor;
  if(ee<=1.e6){coef=.073*le-1.565;factor=1.e10/::pow(10.,8.8-.1*(.2+le*le/6.))*material.average_atomic_weight/22.;}
  else {coef=.063*le-1.55326;factor=1.e10/::pow(10.,8.8-.1*le)*material.average_atomic_weight/22.;}
  double exponent=coef+1.,gmin=factor/exponent*::pow(minimum,exponent),gmax=factor/exponent*::pow(maximum,exponent);
  if(!finite(gmin)||!finite(gmax)||exponent==0.)return out;
  for(int trial=1;trial<=1000;++trial) {
    out.trials=trial;double u,v;
    if(!rareUniform(rng,u)||!rareUniform(rng,v)){out.status=InteractionStatus::random_failure;return out;}
    double argument=gmin+u*(gmax-gmin),fraction=::pow(exponent*argument/factor,1./exponent);
    double envelope=v*factor*::pow(fraction,coef),cross_section=0.;
    if(fraction<1.) {
      double tt=m*m*fraction*fraction/(1.-fraction),ss=2.*mp*fraction*ee;
      double sign=59.3*::pow(ss,.093)+120.2*::pow(ss,-.358);
      double z=sign*.00282*::pow(material.average_atomic_weight,one_third);
      double h=1.-2./fraction+2./(fraction*fraction),g=(.5+((1.+z)*::exp(-z)-1.)/(z*z))*9./z;
      double b=material.average_atomic_weight*fraction*sign*(.007297353/(8.*pi));
      double c=1.+.54/tt,d=1.+1.80/tt,e=2.*m*m/tt,f=.54/(.54+tt);
      double v1=h*::log(d)-e+g*(h*::log(c)-h*f-e);
      double v2=(2.*.25*m*m/tt)*(g*f+(1.80/tt)*::log(1.+tt/1.80));
      cross_section=larger(0.,b*(v1+v2));
    }
    if(envelope>=cross_section||fraction*ee<photonuclear_threshold_MeV*.001)continue;
    if(!finite(fraction)||!finite(cross_section)||fraction<=0.||fraction>=1.)return out;
    // UPHI(2,1) consumes an azimuth even for a zero deflection.
    double sp,cp;if(!azimuth(rng,sp,cp)){out.status=InteractionStatus::random_failure;return out;}
    auto forward=rotate(direction,0.,1.,sp,cp);
    out.residual={ee*(1.-fraction)*1.e3,forward,pdg};
    out.virtual_photon={ee*fraction*1.e3,forward,22};out.status=InteractionStatus::success;return out;
  }
  out.status=InteractionStatus::suppressed;return out;
}
} // namespace c7_egs4
