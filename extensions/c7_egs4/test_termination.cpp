#include "Egs4Termination.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

namespace {
using namespace c7_egs4;
void check(bool ok,std::string const& msg){if(!ok)throw std::runtime_error(msg);}
struct Case {int process,pdg;double energy,mass,binding;Direction direction;std::int64_t seed;};
struct Stream {
  std::int64_t state;int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& u) {
    if(used>=100000)return false;
    state=(state*48271)%2147483647;u=double(state)/2147483647.;++used;return true;
  }
};
struct Answer {LocalOutcome result;int used;};
KOKKOS_INLINE_FUNCTION Answer evaluate(Case q) {
  Stream rng{q.seed};LocalOutcome r;
  if(q.process==0)r.secondaries=annihilateInFlight(q.energy,q.mass,q.direction,rng);
  else if(q.process==1)r=photoelectric(q.energy,q.mass,q.binding,q.direction);
  else r=stopLepton(q.energy,q.mass,q.pdg,rng);
  return {r,rng.used};
}
std::vector<Case> cases() {
  std::vector<Case> out;std::mt19937_64 rng(2026100304);
  Direction directions[]{{0.,0.,1.},{.6,0.,-.8},{.3,.4,std::sqrt(.75)},{1.e-11,0.,-1.}};
  auto add=[&](int process,int pdg,double energy,double binding,Direction dir){
    for(int j=0;j<64;++j)out.push_back({process,pdg,energy,.51099895,binding,dir,
                                      std::int64_t(rng()%2147483646)+1});
  };
  for(auto dir:directions) {
    for(double kinetic:{1.e-5,.01,.4,1.,10.,1.e3,1.e6})add(0,-11,.51099895+kinetic,0.,dir);
    // Arbitrary REAL binding constants test PHOTO's boundary on both sides.
    for(double b:{double(.00001f),double(.0005f),double(.001f)})
      for(double factor:{.5,1.,1.000001,10.,1.e4})add(1,22,b*factor,b,dir);
    for(int pdg:{11,-11})for(double kinetic:{0.,.0001,.1,.4,1.})add(2,pdg,.51099895+kinetic,0.,dir);
  }
  return out;
}
void writeCases(std::string const& path) {
  auto all=cases();std::ofstream f(path);check(bool(f),"Cannot write termination inputs");
  f<<std::setprecision(17)<<all.size()<<'\n';
  for(auto q:all)f<<q.process<<' '<<q.pdg<<' '<<q.energy<<' '<<q.mass<<' '<<q.binding<<' '
    <<q.direction.x<<' '<<q.direction.y<<' '<<q.direction.z<<' '<<q.seed<<'\n';
  check(bool(f),"Incomplete termination inputs");
}
template<class Exec>struct Batch {
  Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(input(i));}
};
struct Diff{double energy{},direction{},deposit{};};
void compare(Answer actual,Answer ref,Diff& diff,std::string const& label) {
  auto a=actual.result,b=ref.result;
  check(a.secondaries.status==InteractionStatus::success,label+": failed kernel");
  check(a.secondaries.count==b.secondaries.count&&actual.used==ref.used,label+": count or random mismatch");
  double dd=std::abs(a.deposit_MeV-b.deposit_MeV);
  check(std::isfinite(dd)&&dd<1.e-12,label+": deposition mismatch");diff.deposit=std::max(diff.deposit,dd);
  for(int j=0;j<b.secondaries.count;++j) {
    auto x=a.secondaries.particle[j],y=b.secondaries.particle[j];
    check(x.pdg==y.pdg,label+": particle mismatch");
    double de=std::abs(x.energy_MeV-y.energy_MeV)/y.energy_MeV;
    double dq=std::max({std::abs(x.direction.x-y.direction.x),std::abs(x.direction.y-y.direction.y),
                       std::abs(x.direction.z-y.direction.z)});
    check(std::isfinite(de)&&std::isfinite(dq)&&de<1.e-10&&dq<1.e-9,label+": numerical mismatch");
    diff.energy=std::max(diff.energy,de);diff.direction=std::max(diff.direction,dq);
  }
}
void edgeCases() {
  Stream r{1};Direction d{0.,0.,1.};
  check(annihilateInFlight(.51099895,.51099895,d,r).status==InteractionStatus::invalid_input&&r.used==0,
        "At-rest positron entered in-flight sampler");
  check(stopLepton(.4,.51099895,-11,r).secondaries.status==InteractionStatus::invalid_input,
        "Unphysical sub-rest energy accepted");
  check(photoelectric(1.,.51099895,-1.,d).secondaries.status==InteractionStatus::invalid_input,
        "Invalid binding energy accepted");
  Stream exhausted{1,100000};
  check(stopLepton(.8,.51099895,-11,exhausted).secondaries.status==InteractionStatus::random_failure,
        "Missing randomness silently accepted");
  check(annihilateInFlight(1.,.51099895,d,exhausted).status==InteractionStatus::random_failure,
        "Missing annihilation randomness silently accepted");
}
void run(std::string const& path,std::string const& tablefile) {
  using Exec=Kokkos::DefaultExecutionSpace;edgeCases();auto all=cases();
  auto medium=readAirTables(tablefile).medium;
  check(medium.binding_energy_MeV>=0.&&medium.binding_energy_MeV<medium.ap,"EBINDA invalid");
  Kokkos::View<Case*,typename Exec::memory_space> in("termination_inputs",all.size());
  Kokkos::View<Answer*,typename Exec::memory_space> out("termination_outputs",all.size());
  auto h=Kokkos::create_mirror_view(in);for(std::size_t i=0;i<all.size();++i)h(i)=all[i];
  Kokkos::deep_copy(in,h);
  Kokkos::parallel_for("C7_CPP_TERMINATION",Kokkos::RangePolicy<Exec>(0,all.size()),Batch<Exec>{in,out});
  Exec().fence();auto actual=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);
  std::ifstream f(path);check(bool(f),"Cannot open termination reference");
  Diff host[3]{},device[3]{};double balance=0.,norm=0.,inflight_transverse=0.;
  int stopped=0,absorbed=0,rejections=0;
  for(std::size_t i=0;i<all.size();++i) {
    Answer ref;int code[2];f>>ref.result.secondaries.count>>ref.used>>code[0]>>code[1]>>ref.result.deposit_MeV;
    for(int j=0;j<2;++j) {
      auto& p=ref.result.secondaries.particle[j];f>>p.energy_MeV>>p.direction.x>>p.direction.y>>p.direction.z;
      check(code[j]>=1&&code[j]<=3,"Invalid reference particle");p.pdg=code[j]==1?22:code[j]==2?-11:11;
    }
    check(bool(f),"Incomplete reference");
    auto q=all[i];auto a=actual(i);auto cpu=evaluate(q);
    compare(cpu,ref,host[q.process],"CPU case "+std::to_string(i));
    compare(a,ref,device[q.process],"device case "+std::to_string(i));
    // Energy accounting includes the medium electron rest mass where needed.
    double available=q.energy,output=a.result.deposit_MeV;
    if(q.process==0||(q.process==1&&a.result.secondaries.count==1)||(q.process==2&&q.pdg==-11))available+=q.mass;
    if(q.process==2&&q.pdg==11)available-=q.mass;
    Direction momentum{};
    for(int j=0;j<a.result.secondaries.count;++j) {
      auto p=a.result.secondaries.particle[j];output+=p.energy_MeV;
      norm=std::max(norm,std::abs(p.direction.x*p.direction.x+p.direction.y*p.direction.y+p.direction.z*p.direction.z-1.));
      if(p.pdg==22){momentum.x+=p.energy_MeV*p.direction.x;momentum.y+=p.energy_MeV*p.direction.y;momentum.z+=p.energy_MeV*p.direction.z;}
    }
    balance=std::max(balance,std::abs(output-available)/available);
    if(q.process==0) {
      double projection=momentum.x*q.direction.x+momentum.y*q.direction.y+momentum.z*q.direction.z;
      double dx=momentum.x-projection*q.direction.x,dy=momentum.y-projection*q.direction.y,dz=momentum.z-projection*q.direction.z;
      inflight_transverse=std::max(inflight_transverse,std::sqrt(dx*dx+dy*dy+dz*dz)/available);
      rejections+=a.result.secondaries.trials>1;
    } else if(q.process==2&&q.pdg==-11) {
      check(momentum.x==0.&&momentum.y==0.&&momentum.z==0.,"Stopped annihilation not back-to-back");++stopped;
    } else if(q.process==1)absorbed+=a.result.secondaries.count==0;
  }
  std::string extra;check(!(f>>extra),"Extra termination reference rows");
  check(balance<1.e-12&&norm<1.e-9,"Energy balance or unit direction failed");
  check(stopped>0&&absorbed>0&&rejections>0,"Missing branch coverage");
  // Known literal-source behavior, not silently suppressed by conservation checks.
  check(inflight_transverse>.1,"Expected original ANNIH angle behavior not observed");
  std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<"\nsynthetic_termination_inputs="<<all.size()<<'\n';
  for(int p=0;p<3;++p)std::cout<<"process="<<p<<" host_energy_relative="<<host[p].energy<<" host_direction_absolute="<<host[p].direction
    <<" device_energy_relative="<<device[p].energy<<" device_direction_absolute="<<device[p].direction<<" deposit_absolute="<<device[p].deposit<<'\n';
  std::cout<<"max_energy_balance_relative="<<balance<<"\nmax_direction_norm_error="<<norm
    <<"\nstopped_annihilations="<<stopped<<"\nabsorbed_photons="<<absorbed<<"\ninflight_rejections="<<rejections
    <<"\nKNOWN_C7_ANNIH_transverse_momentum_fraction="<<inflight_transverse
    <<"\nNOTE=literal ANNIH source equivalence is not physical momentum validation; no full shower claim\n";
}
}
int main(int argc,char** argv) {
  try {
    if(argc==3&&std::string(argv[1])=="--make-cases"){writeCases(argv[2]);return 0;}
    if(argc!=3){std::cerr<<"Usage: test_egs4_termination REFERENCE EGSDAT or --make-cases INPUT\n";return 2;}
    Kokkos::ScopeGuard guard(argc,argv);run(argv[1],argv[2]);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
