#pragma once

#include "Egs4Interactions.hpp"

namespace c7_egs4 {
// Same REAL coefficients and ordering as HATCH's two BREMPR READ statements.
// A small immutable value object; contains no particle/global mutable state.
struct RadiativeTable {
  float dl[6][6]{}, delcm{}, alphi[2]{}, bpar[2]{}, delpos[2]{};
  double photon_cut_MeV{};
};
inline RadiativeTable radiativeTable(Tables const& t) {
  if (t.brems_pair_coefficients.size()!=43) throw std::runtime_error("Incomplete BREMPR coefficients");
  RadiativeTable r;
  for(int i=0;i<6;++i) for(int j=0;j<6;++j) r.dl[i][j]=t.brems_pair_coefficients[6*i+j];
  r.delcm=t.brems_pair_coefficients[36]; r.photon_cut_MeV=t.medium.ap;
  for(int i=0;i<2;++i) {
    r.alphi[i]=t.brems_pair_coefficients[37+3*i];
    r.bpar[i]=t.brems_pair_coefficients[38+3*i];
    r.delpos[i]=t.brems_pair_coefficients[39+3*i];
  }
  return r;
}
KOKKOS_INLINE_FUNCTION double screening(RadiativeTable const& t,int level,double delta) {
  auto c=t.dl[level];
  if(delta<1.) return double(c[0])+delta*(double(c[1])+delta*double(c[2]));
  return double(c[3])+double(c[4])*::log(delta+double(c[5]));
}
struct LpmResult {
  InteractionStatus status{InteractionStatus::invalid_input};
  double acceptance{1.};
  bool sampled{};
};
// Port of C7 LPMEFFECT. Density is RHOF(ALT) in g/cm^3, supplied by the
// atmosphere adapter, NOT the reference density in EGSDAT. C7's constants
// describe air. Return `suppressed` means leave the incident particle intact
// and return to transport, NOT retry until an interaction is accepted.
template<class Random>
KOKKOS_INLINE_FUNCTION LpmResult lpmEffect(double e0,double e1,double e2,
                                         double density,bool pair,Random& rng) {
  LpmResult r;
  if(!finite(e0)||!finite(e1)||!finite(e2)||!finite(density)||
     e0<=0.||e1<=0.||e2<=0.||density<=0.) return r;
  double ssq=3.5156e7*e0/(e1*e2*density);
  if(!finite(ssq)) return r;
  r.status=InteractionStatus::success;
  if(ssq>=1.) return r; // exactly as C7: consumes no random number
  double xi=ssq<1.0796917e-8 ? 2. : 1.-5.4513722e-2*::log(ssq);
  ssq=ssq/xi;
  double s=::sqrt(ssq),g,phi;
  if(s<.1) {
    g=ssq*(14.1+2.36/(s+.1));
    phi=6.*s-16.*ssq;
  } else {
    g=ssq*(24.+.0394/(s-.08));
    phi=6.*s+24.*ssq*(3.1415926535897932384626433832795*.25-::atan(.944+.59/s));
  }
  g=g/(1.+g);
  double egsq=e0*e0,ab=4.*e1*e2;
  if(pair) ab=-ab;
  r.acceptance=xi*(egsq*(g+2.*phi)+phi*ab)/(3.*egsq+ab);
  if(!finite(r.acceptance)) {r.status=InteractionStatus::invalid_input;return r;}
  double u;
  if(!uniform(rng,u)) {r.status=InteractionStatus::random_failure;return r;}
  r.sampled=true;
  if(u>r.acceptance) r.status=InteractionStatus::suppressed;
  return r;
}
struct RadiativeInput {
  double total_MeV{},mass_MeV{},density_g_cm3{};
  Direction direction;
  int pdg{}; // 22: photon conversion; +/-11: bremsstrahlung
};
KOKKOS_INLINE_FUNCTION bool validRadiative(RadiativeInput q) {
  return finite(q.total_MeV)&&finite(q.mass_MeV)&&finite(q.density_g_cm3)&&
         q.total_MeV>0.&&q.mass_MeV>0.&&q.density_g_cm3>0.&&validDirection(q.direction);
}

template<class Random>
KOKKOS_INLINE_FUNCTION Secondaries sampleBremsstrahlung(
    RadiativeTable const& t,RadiativeInput q,Random& rng,int max_trials=4096) {
  Secondaries r;
  double E=q.total_MeV,m=q.mass_MeV,ap=t.photon_cut_MeV;
  if(!validRadiative(q)||(q.pdg!=11&&q.pdg!=-11)||!finite(ap)||ap<=0.||
     E<=m+ap||max_trials<1) return r;
  int lvx=E<50.?0:1,base=E<50.?0:3;
  double n=1.442695041*::log(E*(1./ap));
  // PWR2I has 60 elements in C7. Fail explicitly rather than extrapolate it.
  if(!finite(n)||n<0.||n>=61.) return r;
  double abrems=double(int(n));
  int inner_trials=0;
  for(int trial=1;trial<=max_trials;++trial) {
    r.trials=trial;
    double u,v,w,br; int level;
    if(!uniform(rng,u)||!uniform(rng,v)||!uniform(rng,w)) {
      r.status=InteractionStatus::random_failure;return r;
    }
    if((abrems*double(t.alphi[lvx])+.5)*u>=.5) {
      int idistr=int(abrems*v);
      if(idistr<0||idistr>=60) return r;
      // Every PWR2I entry is exactly representable; ldexp equals the HATCH
      // recurrence P=1, P=P*.5, without per-thread lookup state.
      double p=::ldexp(1.,-idistr);
      level=base;
      if(w>=.721347521) {
        do {
          if(++inner_trials>max_trials) {r.status=InteractionStatus::trial_limit;return r;}
          if(!uniform(rng,u)||!uniform(rng,v)||!uniform(rng,w)) {
            r.status=InteractionStatus::random_failure;return r;
          }
          double h=larger(v,w);
          br=1.-.5*h;
        } while(br*u>.5);
      } else {
        if(!uniform(rng,u)) {r.status=InteractionStatus::random_failure;return r;}
        br=u*.5;
      }
      br=br*p;
    } else {br=larger(v,w);level=base+1;}
    double photon=E*br;
    if(photon<ap) continue;
    double lepton=E-photon;
    if(lepton<m) continue;
    double del=br/lepton;
    if(del>=double(t.delpos[lvx])) continue;
    double rej=screening(t,level,double(t.delcm)*del);
    if(!uniform(rng,u)) {r.status=InteractionStatus::random_failure;return r;}
    if(u>rej) continue;
    if(E>1.e10) {
      auto lpm=lpmEffect(photon,E,lepton,q.density_g_cm3,false,rng);
      if(lpm.status!=InteractionStatus::success) {r.status=lpm.status;return r;}
    }
    double sp,cp,angle=m/E;
    if(!azimuth(rng,sp,cp)) {r.status=InteractionStatus::random_failure;return r;}
    Secondary g{photon,rotate(q.direction,::sin(angle),::cos(angle),sp,cp),22};
    Secondary e{lepton,q.direction,q.pdg}; // C7 does not recoil the parent here.
    r.particle[0]=photon<=lepton?e:g;
    r.particle[1]=photon<=lepton?g:e;
    r.status=InteractionStatus::success;r.count=2;return r;
  }
  r.status=InteractionStatus::trial_limit;return r;
}

template<class Random>
KOKKOS_INLINE_FUNCTION Secondaries samplePairProduction(
    RadiativeTable const& t,RadiativeInput q,Random& rng,int max_trials=4096) {
  Secondaries r;
  double E=q.total_MeV,m=q.mass_MeV,e2=m;
  if(!validRadiative(q)||q.pdg!=22||E<=2.*m||max_trials<1) return r;
  if(E>2.1) {
    int lvx=E<50.?0:1,base=E<50.?0:3;
    bool accepted=false;
    for(int trial=1;trial<=max_trials;++trial) {
      r.trials=trial;
      double u,v,w,z,br;int level;
      if(!uniform(rng,u)||!uniform(rng,v)) {r.status=InteractionStatus::random_failure;return r;}
      if(v>=double(t.bpar[lvx])) {
        level=base;
        if(!uniform(rng,w)||!uniform(rng,z)) {r.status=InteractionStatus::random_failure;return r;}
        br=.5*smaller(smaller(w,z),u);
      } else {level=base+2;br=u*.5;}
      if(br*E<m) continue;
      double del=1./(E*br*(1.-br));
      if(del>=double(t.delpos[lvx])) continue;
      double rej=screening(t,level,double(t.delcm)*del);
      if(!uniform(rng,u)) {r.status=InteractionStatus::random_failure;return r;}
      if(u>rej) continue;
      e2=br*E;accepted=true;break;
    }
    if(!accepted) {r.status=InteractionStatus::trial_limit;return r;}
  }
  double e1=E-e2;
  if(E>1.e12) {
    auto lpm=lpmEffect(E,e1,e2,q.density_g_cm3,true,rng);
    if(lpm.status!=InteractionStatus::success) {r.status=lpm.status;return r;}
  }
  double sp,cp,angle=m/E;
  if(!azimuth(rng,sp,cp)) {r.status=InteractionStatus::random_failure;return r;}
  double st=::sin(angle),ct=::cos(angle),u;
  if(!uniform(rng,u)) {r.status=InteractionStatus::random_failure;return r;}
  int pdg1=u<=.5?11:-11;
  r.particle[0]={e1,rotate(q.direction,st,ct,sp,cp),pdg1};
  r.particle[1]={e2,rotate(q.direction,-st,ct,sp,cp),-pdg1};
  r.status=InteractionStatus::success;r.count=2;return r;
}
template<class Random>
KOKKOS_INLINE_FUNCTION Secondaries sampleRadiative(
    RadiativeTable const& t,RadiativeInput q,Random& rng,int max_trials=4096) {
  if(q.pdg==22) return samplePairProduction(t,q,rng,max_trials);
  return sampleBremsstrahlung(t,q,rng,max_trials);
}
} // namespace c7_egs4
