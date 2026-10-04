#include "Egs4Radiative.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

namespace {
using namespace c7_egs4;
void check(bool b,std::string const& why) {if(!b) throw std::runtime_error(why);}
struct Case {RadiativeInput input;std::int64_t seed;};
// This RNG is a compact test-fixture provider, NOT the planned shower RNG.
struct FixtureStream {
  std::int64_t state;int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& u) {
    if(used>=100000) return false;
    state=(state*48271)%2147483647;
    u=double(state)/2147483647.;++used;return true;
  }
};
struct Answer {Secondaries state;int used{};};
KOKKOS_INLINE_FUNCTION Answer evaluate(RadiativeTable const& t,Case q) {
  FixtureStream rng{q.seed};auto r=sampleRadiative(t,q.input,rng);return {r,rng.used};
}
std::vector<Case> cases() {
  std::vector<Case> out;std::mt19937_64 rng(2026100302);
  Direction directions[]{{0.,0.,1.},{.6,0.,-.8},{.3,.4,std::sqrt(.75)},{1.e-11,0.,-1.}};
  for(int pdg:{11,-11,22})
    for(double E:{1.1,2.1,2.10000001,49.999999,50.,1000.,1.e8,1.e10,1.e10+1.,1.e12,1.e12+1.,1.e14,1.e15})
      for(double rho:{1.225e-3,1.e-6}) for(auto dir:directions) for(int rep=0;rep<32;++rep)
        out.push_back({{E,.51099895,rho,dir,pdg},std::int64_t(rng()%2147483646)+1});
  return out;
}
void writeCases(std::string const& path) {
  auto all=cases();std::ofstream f(path);check(bool(f),"Cannot write radiative inputs");
  f<<std::setprecision(17)<<all.size()<<'\n';
  for(auto q:all) f<<q.input.pdg<<' '<<q.input.total_MeV<<' '<<q.input.mass_MeV<<' '
    <<q.input.density_g_cm3<<' '<<q.input.direction.x<<' '<<q.input.direction.y<<' '
    <<q.input.direction.z<<' '<<q.seed<<'\n';
  check(bool(f),"Incomplete radiative fixture");
}
template<class Exec> struct Batch {
  RadiativeTable table;
  Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const {output(i)=evaluate(table,input(i));}
};
int pdg(int iq) {check(iq>=1&&iq<=3,"Invalid original C7 particle code");return iq==1?22:iq==2?-11:11;}
struct Differences {double energy{},direction{};};
void compare(Answer const& a,Answer const& ref,Differences& d,std::string const& label) {
  check(a.state.status==ref.state.status,label+": acceptance/suppression differs");
  check(a.used==ref.used,label+": random consumption differs");
  if(ref.state.status==InteractionStatus::suppressed) {
    check(a.state.count==0,label+": suppressed collision emitted secondaries");return;
  }
  check(a.state.status==InteractionStatus::success&&a.state.count==2,label+": failed sampler");
  for(int j=0;j<2;++j) {
    auto x=a.state.particle[j],y=ref.state.particle[j];
    check(x.pdg==y.pdg,label+": particle order/identity differs");
    double de=std::abs(x.energy_MeV-y.energy_MeV)/std::max(1.e-20,std::abs(y.energy_MeV));
    double dq=std::max({std::abs(x.direction.x-y.direction.x),std::abs(x.direction.y-y.direction.y),
                       std::abs(x.direction.z-y.direction.z)});
    check(std::isfinite(de)&&std::isfinite(dq),label+": nonfinite output");
    d.energy=std::max(d.energy,de);d.direction=std::max(d.direction,dq);
    check(de<1.e-10&&dq<5.e-8,label+": numeric mismatch");
  }
}
struct ConstantStream {
  double value;int used{},limit{10000};
  KOKKOS_INLINE_FUNCTION bool next(double& u) {
    if(used>=limit)return false;u=value;++used;return true;
  }
};
void edgeCases(RadiativeTable const& t) {
  RadiativeInput q{.51099895+t.photon_cut_MeV,.51099895,.001,{0.,0.,1.},11};
  ConstantStream rng{.5};
  check(sampleRadiative(t,q,rng).status==InteractionStatus::invalid_input&&rng.used==0,"Brems threshold");
  q.pdg=22;q.total_MeV=2.*q.mass_MeV;
  check(sampleRadiative(t,q,rng).status==InteractionStatus::invalid_input,"Pair threshold");
  q.total_MeV=1.1;
  auto low=sampleRadiative(t,q,rng);
  check(low.status==InteractionStatus::success&&low.particle[1].energy_MeV==q.mass_MeV&&rng.used==2,
        "C7 low-energy pair approximation or draw order");
  q.pdg=13;check(sampleRadiative(t,q,rng).status==InteractionStatus::invalid_input,"Unsupported species");
  q.pdg=22;q.total_MeV=10.;ConstantStream empty{.5,0,0};
  check(sampleRadiative(t,q,empty).status==InteractionStatus::random_failure,"Empty random stream");
  ConstantStream reject{0.};
  check(sampleRadiative(t,q,reject,1).status==InteractionStatus::trial_limit,"Pair rejection fallback");
  q.pdg=11;
  check(sampleRadiative(t,q,reject,1).status==InteractionStatus::trial_limit,"Brems rejection fallback");
  ConstantStream lpm_rng{.999};
  auto negligible=lpmEffect(10.,6.,4.,.001,true,lpm_rng);
  check(negligible.status==InteractionStatus::success&&!negligible.sampled&&lpm_rng.used==0,
        "LPM outside suppression region consumed random number");
  auto strong=lpmEffect(1.e15,5.e14,5.e14,.001,true,lpm_rng);
  check(strong.status==InteractionStatus::suppressed&&strong.sampled&&lpm_rng.used==1,"LPM suppression");
  check(lpmEffect(1.,1.,1.,-1.,true,lpm_rng).status==InteractionStatus::invalid_input,"Invalid LPM density");
}
void run(std::string const& path,std::string const& data) {
  using Exec=Kokkos::DefaultExecutionSpace;
  auto table=radiativeTable(readAirTables(data));edgeCases(table);auto all=cases();
  Kokkos::View<Case*,typename Exec::memory_space> input("radiative_inputs",all.size());
  Kokkos::View<Answer*,typename Exec::memory_space> output("radiative_outputs",all.size());
  auto h=Kokkos::create_mirror_view(input);
  for(std::size_t i=0;i<all.size();++i)h(i)=all[i];
  Kokkos::deep_copy(input,h);
  Kokkos::parallel_for("C7_CPP_BREMS_PAIR_LPM",Kokkos::RangePolicy<Exec>(0,all.size()),Batch<Exec>{table,input,output});
  Exec().fence();auto results=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  std::ifstream f(path);check(bool(f),"Cannot open radiative reference");
  Differences host[3]{},device[3]{};int suppressed[3]{},rejected=0,low_pair=0,hard_photon=0;
  double balance=0.,norm_error=0.;
  for(std::size_t i=0;i<all.size();++i) {
    Answer ref;int count,iq1,iq2,skip;
    f>>count>>ref.used>>iq1>>iq2>>skip;
    for(auto& p:ref.state.particle) f>>p.energy_MeV>>p.direction.x>>p.direction.y>>p.direction.z;
    check(bool(f),"Incomplete radiative reference");
    auto q=all[i].input;int type=q.pdg==11?0:q.pdg==-11?1:2;
    ref.state.status=skip?InteractionStatus::suppressed:InteractionStatus::success;
    ref.state.count=skip?0:2;
    if(skip) {
      check(count==1&&pdg(iq1)==q.pdg&&ref.state.particle[0].energy_MeV==q.total_MeV,
            "Original suppressed particle not preserved");
    } else {
      check(count==2,"Reference child count");
      ref.state.particle[0].pdg=pdg(iq1);ref.state.particle[1].pdg=pdg(iq2);
    }
    std::string label="case "+std::to_string(i)+" E="+std::to_string(q.total_MeV);
    auto cpu=evaluate(table,all[i]);auto a=results(i);
    compare(cpu,ref,host[type],label+" CPU");compare(a,ref,device[type],label+" device");
    if(skip) {++suppressed[type];continue;}
    rejected+=a.state.trials>1;
    low_pair+=q.pdg==22&&q.total_MeV<=2.1;
    hard_photon+=q.pdg!=22&&a.state.particle[0].pdg==22;
    check(a.state.particle[0].energy_MeV>=a.state.particle[1].energy_MeV,"Unsorted radiative children");
    double sum=0.;
    for(auto p:a.state.particle) {
      sum+=p.energy_MeV;
      check(p.energy_MeV>=(p.pdg==22?table.photon_cut_MeV:q.mass_MeV),"Radiative child below threshold");
      double norm=p.direction.x*p.direction.x+p.direction.y*p.direction.y+p.direction.z*p.direction.z;
      norm_error=std::max(norm_error,std::abs(norm-1.));
    }
    balance=std::max(balance,std::abs(sum/q.total_MeV-1.));
  }
  std::string extra;check(!(f>>extra),"Extra radiative reference rows");
  check(rejected>100&&low_pair>0&&hard_photon>0&&suppressed[0]>0&&suppressed[1]>0&&suppressed[2]>0,
        "Missing radiative branch coverage");
  check(balance<1.e-12&&norm_error<1.e-9,"Radiative energy or direction norm");
  std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<"\nsynthetic_radiative_inputs="<<all.size()<<'\n';
  for(int p=0;p<3;++p)std::cout<<"species="<<p<<" host_C7_energy_relative="<<host[p].energy
    <<" host_C7_direction_absolute="<<host[p].direction<<" device_C7_energy_relative="<<device[p].energy
    <<" device_C7_direction_absolute="<<device[p].direction<<" LPM_suppressed="<<suppressed[p]<<'\n';
  std::cout<<"screening_rejections="<<rejected<<"\nlow_energy_pair_approximation="<<low_pair
    <<"\nhard_photon_sorted_first="<<hard_photon<<"\nenergy_balance_relative="<<balance
    <<"\nmax_direction_norm_error="<<norm_error
    <<"\nscope=local C7 radiative kernels; supplied density, not atmosphere transport or complete showers\n";
}
}
int main(int argc,char** argv) {
  try {
    if(argc==3&&std::string(argv[1])=="--make-cases") {writeCases(argv[2]);return 0;}
    if(argc!=3) {std::cerr<<"Usage: test_egs4_radiative REFERENCE EGSDAT or --make-cases INPUT\n";return 2;}
    Kokkos::ScopeGuard guard(argc,argv);run(argv[1],argv[2]);return 0;
  } catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
}
