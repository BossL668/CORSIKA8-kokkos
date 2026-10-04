#include "Egs4MesonDecay.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
using namespace c7_egs4;
void check(bool value,char const* message){if(!value)throw std::runtime_error(message);}
struct Random {
  std::uint64_t state{1};int used{};double first{-1.};
  KOKKOS_INLINE_FUNCTION bool next(double& value) {
    state=(16807*state)%2147483647;value=double(state)/2147483647.;
    if(used++==0&&first>=0.)value=first;return true;
  }
};
struct Input {RhoDecayInput decay;std::uint64_t seed{};double first{};};
struct Result {RhoDecayOutcome decay;std::uint64_t state{};int used{};};
KOKKOS_INLINE_FUNCTION Result evaluate(Input q) {
  Random random{q.seed,0,q.first};auto out=sampleRhoDecay(q.decay,random);
  return {out,random.state,random.used};
}
std::vector<Input> cases() {
  std::vector<Input> result;
  for(int origin:{0,1})for(double gamma:{1.0001,1.01,1.5,2.,10.,1.e3,1.e5,1.e8})
    for(Direction d:std::vector<Direction>{{0.,0.,1.},{0.,0.,-1.},{.3,-.4,std::sqrt(.75)},
      {1.,0.,0.},{1.e-21,0.,1.},{0.,1.e-19,-1.}})
      for(double first:{.999954999999,.999955,.999999})for(int seed=0;seed<32;++seed) {
        std::uint64_t n=result.size();
        result.push_back({{gamma*775.26,775.26,139.57039,105.6583755,d,RhoOrigin(origin)},101+7919*n,first});
      }
  return result;
}
void writeCases(std::string const& path,std::vector<Input> const& input) {
  std::ofstream f(path);f<<std::setprecision(17)<<input.size()<<'\n';
  for(auto q:input) {
    auto d=q.decay;
    f<<int(d.origin)<<' '<<d.total_MeV<<' '<<d.rho_mass_MeV<<' '<<d.pion_mass_MeV<<' '<<d.muon_mass_MeV
      <<' '<<d.direction.x<<' '<<d.direction.y<<' '<<d.direction.z<<' '<<q.seed<<' '<<q.first<<'\n';
  }
  check(bool(f),"Cannot write meson inputs");
}
std::vector<Result> readReference(std::string const& path,std::vector<Input> const& input) {
  std::ifstream f(path);std::vector<Result> result(input.size());
  for(std::size_t i=0;i<input.size();++i) {
    auto& r=result[i];auto& d=r.decay;int origin,ids[2];
    f>>origin>>d.count>>r.used>>r.state>>ids[0]>>ids[1];
    for(int j=0;j<2;++j) {
      auto& p=d.particle[j];
      f>>p.energy_MeV>>p.direction.x>>p.direction.y>>p.direction.z>>d.polarization_cosine[j]>>d.polarization_azimuth[j];
      p.pdg=ids[j]==9?-211:ids[j]==8?211:ids[j]==6?13:ids[j]==5?-13:0;
    }
    d.status=InteractionStatus::success;
    check(bool(f)&&origin==int(input[i].decay.origin),"Malformed C7 meson reference");
  }
  std::string extra;check(!(f>>extra),"Extra meson reference rows");return result;
}
struct Empty {KOKKOS_INLINE_FUNCTION bool next(double&){return false;}};
struct Reject {
  int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& v){v=used==0?0.:used%2==1?0.:.99;++used;return true;}
};
struct Bad {KOKKOS_INLINE_FUNCTION bool next(double& v){v=1.;return true;}};
struct Limited {
  int left{};Random random{101,0,.999999};
  KOKKOS_INLINE_FUNCTION bool next(double& v){return left-->0&&random.next(v);}
};
struct Contracts {
  KOKKOS_INLINE_FUNCTION void operator()(int,int& errors)const {
    RhoDecayInput q{2000.,775.26,139.57039,105.6583755,{0.,0.,1.},RhoOrigin::photonuclear};
    Random r;Empty empty;Reject reject;Bad bad;
    errors+=sampleRhoDecay(q,empty).status!=InteractionStatus::random_failure;
    errors+=sampleRhoDecay(q,bad).status!=InteractionStatus::random_failure;
    errors+=sampleRhoDecay(q,reject,2).status!=InteractionStatus::trial_limit;
    // Fail at each successive draw in the rare branch, including the final
    // azimuth. A partial result must never advertise count=2 or success.
    for(int n=0;n<6;++n) {
      Limited limited{n};auto fail=sampleRhoDecay(q,limited);
      errors+=fail.status!=InteractionStatus::random_failure||fail.count!=0;
    }
    errors+=sampleRhoDecay(q,r,0).status!=InteractionStatus::invalid_input;
    q.origin=RhoOrigin(2);errors+=sampleRhoDecay(q,r).status!=InteractionStatus::invalid_input;
    q.origin=RhoOrigin::hadronic;q.total_MeV=q.rho_mass_MeV;
    errors+=sampleRhoDecay(q,r).status!=InteractionStatus::invalid_input;
    q.total_MeV=2000.;q.direction.z=2.;errors+=sampleRhoDecay(q,r).status!=InteractionStatus::invalid_input;
    q.direction.z=1.;q.muon_mass_MeV=400.;errors+=sampleRhoDecay(q,r).status!=InteractionStatus::invalid_input;
  }
};
void compare(std::vector<Input> const& inputs,std::vector<Result> const& ref,std::vector<Result> const& got,
    char const* label) {
  int controls=0,numerical=0,counts[2][2]{};
  double energy=0.,direction=0.,low_direction=0.,polar=0.,closure=0.,momentum=0.,norm=0.;
  for(std::size_t i=0;i<inputs.size();++i) {
    auto q=inputs[i].decay;auto a=ref[i],b=got[i];
    controls+=a.state!=b.state||a.used!=b.used||a.decay.count!=b.decay.count||a.decay.status!=b.decay.status;
    check(b.decay.status==InteractionStatus::success&&b.decay.count==2,"Native rho decay failed");
    bool muon=b.decay.particle[0].pdg==13;++counts[int(q.origin)][muon];
    double mass=muon?q.muon_mass_MeV:q.pion_mass_MeV,etotal=0.,ptotal[3]{};
    for(int j=0;j<2;++j) {
      auto x=a.decay.particle[j],y=b.decay.particle[j];controls+=x.pdg!=y.pdg;
      double de=std::abs(x.energy_MeV-y.energy_MeV)/std::max(1.,x.energy_MeV);
      energy=std::max(energy,de);numerical+=de>1.e-12;
      check(std::isfinite(y.energy_MeV)&&y.energy_MeV>=mass,"Invalid decay energy");
      etotal+=y.energy_MeV;double p=std::sqrt((y.energy_MeV-mass)*(y.energy_MeV+mass));
      double xd[]={x.direction.x,x.direction.y,x.direction.z},yd[]={y.direction.x,y.direction.y,y.direction.z},n=0.;
      for(int k=0;k<3;++k) {
        check(std::isfinite(yd[k]),"Nonfinite decay direction");double dd=std::abs(xd[k]-yd[k]);
        // At extreme boosts the literal C7 sqrt(1-cos^2) reconstructs a
        // tiny transverse angle from a rounded cosine. Track that separately
        // rather than weakening the ordinary-energy comparison everywhere.
        bool high_boost=q.total_MeV/q.rho_mass_MeV>=1.e5;
        if(!high_boost)low_direction=std::max(low_direction,dd);
        direction=std::max(direction,dd);numerical+=dd>(high_boost?1.e-7:2.e-10);
        n+=yd[k]*yd[k];ptotal[k]+=p*yd[k];
      }
      norm=std::max(norm,std::abs(n-1.));
      polar=std::max(polar,std::abs(a.decay.polarization_cosine[j]-b.decay.polarization_cosine[j]));
      polar=std::max(polar,std::abs(a.decay.polarization_azimuth[j]-b.decay.polarization_azimuth[j]));
    }
    closure=std::max(closure,std::abs(etotal/q.total_MeV-1.));
    double p=std::sqrt((q.total_MeV-q.rho_mass_MeV)*(q.total_MeV+q.rho_mass_MeV));
    double d[]={q.direction.x,q.direction.y,q.direction.z};
    for(int k=0;k<3;++k)momentum=std::max(momentum,std::abs(ptotal[k]-p*d[k])/q.total_MeV);
  }
  std::cout<<label<<" inputs="<<inputs.size()<<" controls="<<controls<<" numerical="<<numerical
    <<" max_energy_relative="<<energy<<" max_direction_absolute="<<direction<<" max_polarization="<<polar
    <<" max_direction_gamma_below_1e5="<<low_direction<<" energy_closure_relative="<<closure
    <<" momentum_closure_over_energy="<<momentum<<" norm_error="<<norm<<'\n';
  for(int o=0;o<2;++o)std::cout<<"origin="<<o<<" pions="<<counts[o][0]<<" muons="<<counts[o][1]<<'\n';
  check(controls==0&&numerical==0&&polar<1.e-12&&closure<1.e-12&&momentum<1.e-7&&norm<1.e-12,"RHO0DC disagreement");
  for(auto& row:counts)for(auto count:row)check(count>0,"Missing decay branch/origin coverage");
}
struct Batch {
  Kokkos::View<Input*> input;Kokkos::View<Result*> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(input(i));}
};
struct AngularMoment {
  int origin{},muon{};
  KOKKOS_INLINE_FUNCTION void operator()(int i,double& sum)const {
    Random random{std::uint64_t(101)+7919*std::uint64_t(i),0,muon?.999999:0.};
    RhoDecayInput q{7752.6,775.26,139.57039,105.6583755,{0.,0.,1.},RhoOrigin(origin)};
    auto r=sampleRhoDecay(q,random);sum+=r.status==InteractionStatus::success?r.cm_cosine*r.cm_cosine:1.e6;
  }
};
void angularMoments() {
  constexpr int n=32768;
  for(int origin:{0,1})for(int muon:{0,1}) {
    AngularMoment test{origin,muon};double host=0.,device=0.;
    for(int i=0;i<n;++i)test(i,host);
    Kokkos::parallel_reduce("rho_angular_moment",n,test,device);
    double a=origin?.8836:0.,expected=(1./3.-a/5.)/(1.-a/3.);
    double variance=(1./5.-a/7.)/(1.-a/3.)-expected*expected;
    double sampling_error=std::sqrt(variance/n);
    std::cout<<"rho_angular origin="<<origin<<" muon="<<muon<<" conditional_samples="<<n
      <<" mean_cos2="<<device/n<<" analytic="<<expected<<" host_device_difference="<<std::abs(device-host)/n<<'\n';
    check(std::abs(device-host)/n<1.e-12&&std::abs(device/n-expected)<5.*sampling_error,"Rho angular distribution differs");
  }
}
}
int main(int argc,char** argv) {
  try {
    auto input=cases();
    if(argc==3&&std::string(argv[1])=="--make-cases"){writeCases(argv[2],input);return 0;}
    if(argc!=2)return 2;Kokkos::ScopeGuard guard(argc,argv);
    std::cout<<std::setprecision(17)<<"execution_space="<<Kokkos::DefaultExecutionSpace::name()<<'\n';
    int errors=0;Contracts contracts;contracts(0,errors);check(!errors,"Host rho contracts");
    Kokkos::parallel_reduce("rho_contracts",1,contracts,errors);check(!errors,"Device rho contracts");
    std::vector<Result> host;for(auto q:input)host.push_back(evaluate(q));
    bool self=std::string(argv[1])=="--contracts";auto reference=self?host:readReference(argv[1],input);
    compare(input,reference,host,self?"host_contracts":"host_C7");
    Kokkos::View<Input*> dinput("rho_inputs",input.size());auto h=Kokkos::create_mirror_view(dinput);
    for(std::size_t i=0;i<input.size();++i)h(i)=input[i];Kokkos::deep_copy(dinput,h);
    Kokkos::View<Result*> output("rho_outputs",input.size());Kokkos::parallel_for("native_rho",input.size(),Batch{dinput,output});
    auto out=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);std::vector<Result> device;
    for(std::size_t i=0;i<input.size();++i)device.push_back(out(i));
    compare(input,reference,device,self?"device_host":"device_C7");
    angularMoments();
    std::cout<<"scope=pre-cut rho decay, both origins and rare muon branch; no omega/phi, hadronic transport, full shower or timing claim\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
