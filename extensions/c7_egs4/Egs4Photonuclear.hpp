#pragma once
#include "Egs4Interactions.hpp"

namespace c7_egs4 {
// Native C7 photonuclear arithmetic. These are pre-cut final-state kernels,
// NOT a SOPHIA replacement or a complete nuclear transport handler.
struct PhotonuclearMasses {
  double proton_MeV{},neutron_MeV{},neutral_pion_MeV{},charged_pion_MeV{};
  double phi_MeV{},omega_MeV{},rho_MeV{};
};
KOKKOS_INLINE_FUNCTION bool validPhotonuclearMasses(PhotonuclearMasses m) {
  return finite(m.proton_MeV)&&finite(m.neutron_MeV)&&finite(m.neutral_pion_MeV)&&
    finite(m.charged_pion_MeV)&&m.proton_MeV>0.&&m.neutron_MeV>0.&&
    m.neutral_pion_MeV>0.&&m.charged_pion_MeV>0.;
}
// Immutable after preparation: C7's first-call SAVE/COMMON table becomes
// an explicit value copied into device memory. Source DATA C(12)=.1 GeV,
// C(34)=20, DX=.5. No shared mutation in the sampling kernel.
struct TransverseMomentumTable {
  double cumulative[51]{},envelope[51]{},peak_GeV{};
  int intervals{};
};
KOKKOS_INLINE_FUNCTION TransverseMomentumTable makeTransverseMomentumTable(
    double peak_GeV=.1,double cutoff=20.) {
  TransverseMomentumTable t;
  if(!finite(peak_GeV)||peak_GeV<=0.||!finite(cutoff)||cutoff<.5||cutoff>25.)return t;
  t.peak_GeV=peak_GeV;t.intervals=int(cutoff/.5);
  t.envelope[0]=.5*::exp(.5);
  for(int i=1;i<=t.intervals;++i) {
    double x=i*.5;if(x<1.)x+=.5;
    t.envelope[i]=x*::exp(1.-x);
    t.cumulative[i]=t.cumulative[i-1]+t.envelope[i-1];
  }
  double inverse=1./t.cumulative[t.intervals];
  for(int i=1;i<=t.intervals;++i)t.cumulative[i]*=inverse;
  return t;
}
struct TransverseMomentum {
  InteractionStatus status{InteractionStatus::invalid_input};
  double GeV{};int trials{};
};
template<class Random>
KOKKOS_INLINE_FUNCTION TransverseMomentum sampleTransverseMomentum(
    TransverseMomentumTable const& t,Random& rng,int max_trials=4096) {
  TransverseMomentum result;
  if(t.intervals<1||t.intervals>50||!finite(t.peak_GeV)||t.peak_GeV<=0.||max_trials<1)return result;
  for(int trial=1;trial<=max_trials;++trial) {
    result.trials=trial;double u,v;
    if(!uniform(rng,u)||!uniform(rng,v)){result.status=InteractionStatus::random_failure;return result;}
    int i=1;while(i<t.intervals&&t.cumulative[i]<u)++i;
    double width=t.cumulative[i]-t.cumulative[i-1];
    if(!finite(width)||width<=0.||t.cumulative[i]<u||!finite(t.envelope[i-1])||t.envelope[i-1]<=0.)return result;
    double x=((u-t.cumulative[i-1])/width+i-1)*.5;
    if(v*t.envelope[i-1]>x*::exp(1.-x))continue;
    result.GeV=x*t.peak_GeV;result.status=InteractionStatus::success;return result;
  }
  result.status=InteractionStatus::trial_limit;return result;
}
struct SinglePionOutcome {
  Secondaries secondaries; // pion first, recoil second; NOT energy-sorted
  int target_pdg{};
  double target_rest_energy_MeV{}; // included in outgoing total energy
};
// Separate providers preserve the C7 stream distinction: PIGEN1 and UPHI
// use stream 2; PTRANS uses stream 1. A production per-history policy may
// supply explicit substreams. There is no mutable process-global RNG here.
template<class EmRandom,class HadronRandom>
KOKKOS_INLINE_FUNCTION SinglePionOutcome sampleSinglePion(double energy,Direction direction,
    PhotonuclearMasses masses,TransverseMomentumTable const& pt_table,
    EmRandom& em_rng,HadronRandom& hadron_rng,int max_trials=4096) {
  SinglePionOutcome out;auto& r=out.secondaries;
  if(!finite(energy)||energy<=0.||!validDirection(direction)||!validPhotonuclearMasses(masses)||max_trials<1)return out;
  double u,v;
  if(!uniform(em_rng,u)||!uniform(em_rng,v)){r.status=InteractionStatus::random_failure;return out;}
  bool proton=u<=.49923,exchange=v<=.3333333;
  out.target_pdg=proton?2212:2112;
  double m2=proton?masses.proton_MeV:masses.neutron_MeV;
  double m3=exchange?masses.charged_pion_MeV:masses.neutral_pion_MeV;
  double m4=(proton!=exchange)?masses.proton_MeV:masses.neutron_MeV;
  out.target_rest_energy_MeV=m2;
  int pion=exchange?(proton?211:-211):111,recoil=(proton!=exchange)?2212:2112;
  double m2i=1./m2,w0=energy+m2,w0i=1./w0,w0s=::sqrt(m2*(m2+2.*energy)),w0si=1./w0s;
  double threshold=.5*((m3+m4)*(m3+m4)-m2*m2)*m2i;
  if(energy<=threshold)return out; // forbidden channel, never invent an alternate reaction
  double beta=energy*w0i,gamma=w0*w0si,ed=.5*((m3-m4)*(m3-m4)-m2*m2)*m2i;
  double esq=::sqrt((energy-threshold)*(energy-ed)),bratio=energy/esq;
  double g3=w0i*bratio*(energy-threshold+m3*m2i*(m3+m4));
  double e3cm=g3*m2*gamma/bratio,p3cm=m2*w0si*esq;
  double b3sq=p3cm*p3cm/(p3cm*p3cm+m3*m3),b3=::sqrt(b3sq),ct=0.;
  if(energy<=1300.) {
    bool accepted=false;
    for(int trial=1;trial<=max_trials;++trial) {
      r.trials=trial;
      if(!uniform(em_rng,u)||!uniform(em_rng,v)){r.status=InteractionStatus::random_failure;return out;}
      ct=exchange?1./b3-1./(u*2.*b3sq/(1.-b3sq)+b3/(1.+b3)):2.*u-1.;
      double probability,scale;
      if(energy<=900.) {
        probability=exchange?1.+ct*(-1.8+ct*(.65+ct*(.34-.18*ct))):1.-.6*ct*ct;
        scale=exchange?2.5:1.;
      } else {
        probability=exchange?1.+ct*(-2.18+ct*(7.20+ct*(-2.55+ct*(-15.39+ct*(6.36+ct*(13.80-ct*8.235)))))):
          1.+6.*ct*ct-5.*((ct*ct)*(ct*ct));
        scale=exchange?13.2:2.8;
      }
      if(v*scale<=probability){accepted=true;break;}
    }
    if(!accepted){r.status=InteractionStatus::trial_limit;return out;}
  } else {
    auto pt=sampleTransverseMomentum(pt_table,hadron_rng,max_trials);r.trials=pt.trials;
    if(pt.status!=InteractionStatus::success){r.status=pt.status;return out;}
    double momentum=1.e3*pt.GeV;
    ct=::sqrt(larger(0.,(p3cm-momentum)*(p3cm+momentum)))/p3cm;
  }
  double e3=gamma*(e3cm+beta*p3cm*ct);
  double p3=::sqrt(larger(0.,(e3-m3)*(e3+m3)));
  double c3=p3>0.?(m4*m4-m2*m2-m3*m3+2.*e3*w0-2.*energy*m2)/(2.*energy*p3):1.;
  double s3=::sqrt(larger(0.,(1.-c3)*(1.+c3))),sp,cp;
  if(!azimuth(em_rng,sp,cp)){r.status=InteractionStatus::random_failure;return out;}
  double e4=w0-e3,p4sq=(e4-m4)*(e4+m4);
  if(!finite(p4sq)||p4sq<=0.)return out;
  double p4=::sqrt(p4sq),c4=(m3*m3-m2*m2-m4*m4+2.*e4*w0-2.*energy*m2)/(2.*energy*p4);
  double s4=-::sqrt(larger(0.,(1.-c4)*(1.+c4)));
  r.particle[0]={e3,rotate(direction,s3,c3,sp,cp),pion};
  r.particle[1]={e4,rotate(direction,s4,c4,sp,cp),recoil};
  for(auto const& p:r.particle)if(!finite(p.energy_MeV)||!validDirection(p.direction))return out;
  r.count=2;r.status=InteractionStatus::success;return out;
}

enum class PhotonuclearBranch { single_pion=1, double_pion=2, vector_meson=3, many_hadrons=4 };
struct PhotonuclearOutcome {
  InteractionStatus status{InteractionStatus::invalid_input};
  PhotonuclearBranch branch{PhotonuclearBranch::single_pion};
  Secondary particle[3]{};
  int count{},trials{},target_pdg{};
  double target_rest_energy_MeV{};
};
// PIGEN2: original Dalitz rejection followed by the C7 transverse-momentum
// prescription and CM->lab rotations. Return both pions and the recoil,
// including subcut recoils: cut/deposit/stack bookkeeping is the caller's job.
template<class EmRandom,class HadronRandom>
KOKKOS_INLINE_FUNCTION PhotonuclearOutcome sampleDoublePion(double energy,Direction direction,
    PhotonuclearMasses masses,TransverseMomentumTable const& pt_table,
    EmRandom& em_rng,HadronRandom& hadron_rng,int max_trials=4096) {
  PhotonuclearOutcome out;out.branch=PhotonuclearBranch::double_pion;
  if(!finite(energy)||energy<=0.||!validDirection(direction)||!validPhotonuclearMasses(masses)||max_trials<1)return out;
  double u,v;
  if(!uniform(em_rng,u)||!uniform(em_rng,v)){out.status=InteractionStatus::random_failure;return out;}
  bool proton=u<=.49923;out.target_pdg=proton?2212:2112;
  int ids[3];
  if(v<=.3){ids[0]=ids[1]=111;ids[2]=out.target_pdg;}
  else if(v<=.6){ids[0]=211;ids[1]=-211;ids[2]=out.target_pdg;}
  else{ids[0]=proton?211:-211;ids[1]=111;ids[2]=proton?2112:2212;}
  double m2=proton?masses.proton_MeV:masses.neutron_MeV;
  double m3=ids[0]==111?masses.neutral_pion_MeV:masses.charged_pion_MeV;
  double m4=ids[1]==111?masses.neutral_pion_MeV:masses.charged_pion_MeV;
  double m5=ids[2]==2212?masses.proton_MeV:masses.neutron_MeV;
  out.target_rest_energy_MeV=m2;
  double ecm=::sqrt(m2*(m2+2.*energy));if(ecm<=m3+m4+m5)return out;
  double a1=(m3+m4)*(m3+m4),a2=(ecm-m5)*(ecm-m5)-a1;
  double a3=(m3+m5)*(m3+m5),a4=(ecm-m4)*(ecm-m4)-a3;
  double a5=(m3-m4)*(m3+m4),a6=(ecm-m5)*(ecm+m5),a7=.5/ecm;
  double beta=energy/(m2+energy),gamma=2.*(energy+m2)*a7,am34sq=0.,am35sq=0.;bool accepted=false;
  for(int trial=1;trial<=max_trials;++trial) {
    out.trials=trial;
    if(!uniform(em_rng,u)||!uniform(em_rng,v)){out.status=InteractionStatus::random_failure;return out;}
    am34sq=a2*u+a1;am35sq=a4*v+a3;double inverse=.5/::sqrt(am34sq);
    double e3star=(a5+am34sq)*inverse,e5star=(a6-am34sq)*inverse;
    double root1=::sqrt(larger(0.,(e3star-m3)*(e3star+m3))),root2=::sqrt(larger(0.,(e5star-m5)*(e5star+m5)));
    double discr=am35sq-(e3star+e5star)*(e3star+e5star);
    if(discr>-(root1-root2)*(root1-root2)||discr<-(root1+root2)*(root1+root2))continue;
    accepted=true;break;
  }
  if(!accepted){out.status=InteractionStatus::trial_limit;return out;}
  double e4=(ecm*ecm+m4*m4-am35sq)*a7,e5=(ecm*ecm+m5*m5-am34sq)*a7,e3=ecm-e4-e5;
  if(e4>e3){double temp=e3;e3=e4;e4=temp;temp=m3;m3=m4;m4=temp;int id=ids[0];ids[0]=ids[1];ids[1]=id;}
  double p3sq=(e3-m3)*(e3+m3),p4sq=(e4-m4)*(e4+m4),p5sq=(e5-m5)*(e5+m5);
  double p3=::sqrt(larger(0.,p3sq)),p4=::sqrt(larger(0.,p4sq)),p5=::sqrt(larger(0.,p5sq));
  if(p3<=0.||p4<=0.||p5<=0.)return out;
  double ca=(p5sq-p3sq-p4sq)/(2.*p3*p4),sa=-::sqrt(larger(0.,(1.-ca)*(1.+ca)));
  double cb=(p4sq-p3sq-p5sq)/(2.*p3*p5),sb=::sqrt(larger(0.,(1.-cb)*(1.+cb)));
  auto pt=sampleTransverseMomentum(pt_table,hadron_rng,max_trials);out.trials+=pt.trials;
  if(pt.status!=InteractionStatus::success){out.status=pt.status;return out;}
  double ratio=1.e3*pt.GeV/p3,s3=ratio<1.?ratio:1.,c3=::sqrt((1.-s3)*(1.+s3)),sp,cp;
  if(!azimuth(em_rng,sp,cp)){out.status=InteractionStatus::random_failure;return out;}
  double elab3=gamma*(e3+beta*p3*c3),p3lab=::sqrt(larger(0.,(elab3-m3)*(elab3+m3)));
  if(p3lab<=0.)return out;
  double ct=(beta*e3+p3*c3)*gamma/p3lab;ct=ct<1.?ct:1.;
  double phi_s,phi_c;
  if(!azimuth(em_rng,phi_s,phi_c)){out.status=InteractionStatus::random_failure;return out;}
  out.particle[0]={elab3,rotate(direction,::sqrt(larger(0.,(1.-ct)*(1.+ct))),ct,phi_s,phi_c),ids[0]};
  double cosines[]={ca,cb},sines[]={sa,sb},energies[]={e4,e5},momenta[]={p4,p5},rest[]={m4,m5};
  for(int i=0;i<2;++i) {
    double ccm=c3*cosines[i]-s3*cp*sines[i],elab=gamma*(energies[i]+beta*momenta[i]*ccm);
    double scm=::sqrt(larger(0.,(1.-ccm)*(1.+ccm))),cphi=0.,sphi=1.;
    if(scm!=0.) {
      double inverse=1./scm,a=c3*cp*sines[i]+s3*cosines[i];
      cphi=(phi_c*a-phi_s*sp*sines[i])*inverse;sphi=(phi_s*a+phi_c*sp*sines[i])*inverse;
    }
    double plab=::sqrt(larger(0.,(elab-rest[i])*(elab+rest[i])));if(plab<=0.)return out;
    ct=(beta*energies[i]+momenta[i]*ccm)*gamma/plab;ct=ct<1.?ct:1.;
    out.particle[i+1]={elab,rotate(direction,::sqrt(larger(0.,(1.-ct)*(1.+ct))),ct,sphi,cphi),ids[i+1]};
  }
  for(auto const& p:out.particle)if(!finite(p.energy_MeV)||!validDirection(p.direction))return out;
  out.count=3;out.status=InteractionStatus::success;return out;
}

// RHOGEN returns the meson BEFORE its decay. In particular, C7's polarized
// RHO0DC(1) must not be silently replaced by an isotropic/Pythia rho decay.
template<class Random>
KOKKOS_INLINE_FUNCTION PhotonuclearOutcome sampleVectorMeson(double energy,Direction direction,
    PhotonuclearMasses masses,Random& rng) {
  PhotonuclearOutcome out;out.branch=PhotonuclearBranch::vector_meson;
  if(!finite(energy)||energy<=0.||!validDirection(direction)||!validPhotonuclearMasses(masses)||
     !finite(masses.phi_MeV)||masses.phi_MeV<=0.||!finite(masses.omega_MeV)||masses.omega_MeV<=0.||
     !finite(masses.rho_MeV)||masses.rho_MeV<=0.)return out;
  double u,v,t;
  if(!uniform(rng,u)||!uniform(rng,v)||!uniform(rng,t)){out.status=InteractionStatus::random_failure;return out;}
  bool proton=u<=.49923;out.target_pdg=proton?2212:2112;
  double m2=proton?masses.proton_MeV:masses.neutron_MeV,m3=v<.06?masses.phi_MeV:v<.1?masses.omega_MeV:masses.rho_MeV,m4=m2;
  int pdg=v<.06?333:v<.1?223:113;out.target_rest_energy_MeV=m2;
  double m2i=1./m2,m2sq=m2*m2,w0s=::sqrt(m2*(m2+2.*energy)),w0si=1./w0s;
  double threshold=.5*((m3+m4)*(m3+m4)-m2sq)*m2i;if(energy<=threshold)return out;
  double a3=.5*m3*m3*w0si,e2=.5*(w0s+m2sq*w0si),e4=e2-a3;
  double pcm2=::sqrt((e2-m2)*(e2+m2)),pcm4=::sqrt((e4-m2)*(e4+m2));
  double tmin=a3*a3-(pcm2+pcm4)*(pcm2+pcm4),tmax=a3*a3-(pcm2-pcm4)*(pcm2-pcm4);
  double expmin=::exp(8.e-6*tmin),argument=t*(::exp(8.e-6*tmax)-expmin)+expmin;
  if(!finite(argument)||argument<=0.)return out;
  t=::log(argument)/8.e-6;
  double signed_plng=(e2*e4+.5*t-m2sq)/pcm2;
  // Preserve RHOGEN's ABS(PLNG3) fold, but evaluate its two-body lab state
  // using invariants. Subtracting two ~1e11 MeV energies to obtain a ~1 GeV
  // recoil and reconstructing pT from 1-cos^2 loses useful transverse bits.
  // These are algebraic rearrangements of the SAME sampled t/CM angle.
  double effective_t=signed_plng<0.?tmin+tmax-t:t;
  double kinetic4=-effective_t*.5*m2i;
  double longitudinal_extra=(m3*m3-effective_t)/(2.*energy);
  double pz4=kinetic4+longitudinal_extra,pz3=energy-pz4;
  double pt_sq=2.*kinetic4*(m2-longitudinal_extra)-longitudinal_extra*longitudinal_extra;
  if(!finite(pt_sq)||pt_sq<0.)return out;
  double pt=::sqrt(pt_sq),e3lab=energy-kinetic4,e4lab=m4+kinetic4;
  double p3=::sqrt(larger(0.,(e3lab-m3)*(e3lab+m3))),p4=::sqrt(larger(0.,kinetic4*(e4lab+m4)));
  if(p3<=0.||p4<=0.)return out;
  double sp,cp;if(!azimuth(rng,sp,cp)){out.status=InteractionStatus::random_failure;return out;}
  out.particle[0]={e3lab,rotate(direction,pt/p3,pz3/p3,sp,cp),pdg};
  out.particle[1]={e4lab,rotate(direction,-pt/p4,pz4/p4,sp,cp),out.target_pdg};
  for(int i=0;i<2;++i)if(!finite(out.particle[i].energy_MeV)||!validDirection(out.particle[i].direction))return out;
  out.count=2;out.status=InteractionStatus::success;return out;
}
struct PhotonuclearSelection {
  InteractionStatus status{InteractionStatus::invalid_input};
  PhotonuclearBranch branch{PhotonuclearBranch::single_pion};
};
template<class Random>
KOKKOS_INLINE_FUNCTION PhotonuclearSelection selectPhotonuclearBranch(double energy,PhotonuclearMasses masses,Random& rng) {
  PhotonuclearSelection out;
  if(!finite(energy)||energy<=0.||!validPhotonuclearMasses(masses))return out;
  double u;if(!uniform(rng,u)){out.status=InteractionStatus::random_failure;return out;}
  if(u>(energy-400.)*.001)out.branch=PhotonuclearBranch::single_pion;
  else if(u>(energy-2000.)*.001)out.branch=PhotonuclearBranch::double_pion;
  else {
    double target=.5*(masses.neutron_MeV*.001+masses.proton_MeV*.001);
    double ecm=::sqrt(target*(target+2.*energy*.001));
    if(!finite(ecm)||ecm<=0.)return out;
    // The first exponent is a REAL literal in C7, unlike the second one.
    double fraction=.17560*::pow(ecm,double(.037303f))+.68008/::pow(ecm,1.3021);
    if(!finite(fraction))return out;
    if(!uniform(rng,u)){out.status=InteractionStatus::random_failure;return out;}
    out.branch=u<fraction?PhotonuclearBranch::vector_meson:PhotonuclearBranch::many_hadrons;
  }
  out.status=InteractionStatus::success;return out;
}
template<class EmRandom,class HadronRandom>
KOKKOS_INLINE_FUNCTION PhotonuclearOutcome samplePhotonuclear(double energy,Direction direction,
    PhotonuclearMasses masses,TransverseMomentumTable const& table,EmRandom& em,HadronRandom& hadron,int max_trials=4096) {
  PhotonuclearOutcome out;
  if(!validDirection(direction)||max_trials<1)return out;
  auto selected=selectPhotonuclearBranch(energy,masses,em);out.status=selected.status;out.branch=selected.branch;
  if(selected.status!=InteractionStatus::success)return out;
  if(selected.branch==PhotonuclearBranch::single_pion) {
    auto r=sampleSinglePion(energy,direction,masses,table,em,hadron,max_trials);
    out.status=r.secondaries.status;out.count=r.secondaries.count;out.trials=r.secondaries.trials;
    out.target_pdg=r.target_pdg;out.target_rest_energy_MeV=r.target_rest_energy_MeV;
    for(int i=0;i<2;++i)out.particle[i]=r.secondaries.particle[i];return out;
  }
  if(selected.branch==PhotonuclearBranch::double_pion)return sampleDoublePion(energy,direction,masses,table,em,hadron,max_trials);
  if(selected.branch==PhotonuclearBranch::vector_meson)return sampleVectorMeson(energy,direction,masses,em);
  out.status=InteractionStatus::host_required; // C7 SDPM policy must be supplied explicitly
  return out;
}
} // namespace c7_egs4
