#include "Egs4TwoBodyDecay.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
using namespace c7_egs4;
void check(bool v,char const* message){if(!v)throw std::runtime_error(message);}
struct Random {
  std::uint64_t state{1};int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& v){state=(16807*state)%2147483647;v=double(state)/2147483647.;++used;return true;}
};
struct Input {TwoBodyInput decay;std::uint64_t seed{};};
struct Result {TwoBodyOutcome decay;std::uint64_t state{};int used{};};
KOKKOS_INLINE_FUNCTION Result evaluate(Input q) {
  Random r{q.seed};auto out=sampleTwoBodyDecay(q.decay,r);return {out,r.state,r.used};
}
std::vector<Input> cases() {
  std::vector<TwoBodyInput> modes{
    {0.,782.65,134.9768,0.,111,22,{},TwoBodyKind::general},
    {0.,782.65,139.57039,139.57039,211,-211,{},TwoBodyKind::general},
    {0.,1019.461,493.677,493.677,321,-321,{},TwoBodyKind::general},
    {0.,1019.461,497.611,497.611,130,310,{},TwoBodyKind::general},
    {0.,1019.461,547.862,0.,221,22,{},TwoBodyKind::general},
    {0.,782.65,105.6583755,105.6583755,-13,13,{},TwoBodyKind::muon_pair},
    {0.,1019.461,105.6583755,105.6583755,-13,13,{},TwoBodyKind::muon_pair}};
  std::vector<Input> out;
  for(auto q:modes)for(double gamma:{1.0001,1.01,1.5,2.,10.,1.e3,1.e5,1.e8})
    for(Direction d:std::vector<Direction>{{0.,0.,1.},{0.,0.,-1.},{.3,-.4,std::sqrt(.75)},
        {1.,0.,0.},{1.e-21,0.,1.},{0.,1.e-19,-1.}})for(int j=0;j<32;++j) {
      q.total_MeV=gamma*q.parent_mass_MeV;q.direction=d;
      out.push_back({q,101+7919*out.size()});
    }
  return out;
}
void writeCases(std::string const& path,std::vector<Input> const& input) {
  std::ofstream f(path);f<<std::setprecision(17)<<input.size()<<'\n';
  for(auto q:input){auto d=q.decay;f<<int(d.kind)<<' '<<d.total_MeV<<' '<<d.parent_mass_MeV<<' '
    <<d.first_mass_MeV<<' '<<d.second_mass_MeV<<' '<<d.direction.x<<' '<<d.direction.y<<' '
    <<d.direction.z<<' '<<q.seed<<'\n';}check(bool(f),"Cannot write two-body inputs");
}
std::vector<Result> readReference(std::string const& path,std::vector<Input> const& input) {
  std::ifstream f(path);std::vector<Result> out(input.size());
  for(std::size_t i=0;i<input.size();++i) {
    auto& r=out[i];int mode,id[2];f>>mode>>r.decay.count>>r.used>>r.state>>id[0]>>id[1];
    check(mode==int(input[i].decay.kind)&&id[0]==6&&id[1]==5,"Reference order or mode differs");
    for(int j=0;j<2;++j){auto& p=r.decay.particle[j];f>>p.energy_MeV>>p.direction.x>>p.direction.y>>p.direction.z
      >>r.decay.polarization_cosine[j]>>r.decay.polarization_azimuth[j];
      p.pdg=j?input[i].decay.first_pdg:input[i].decay.second_pdg;}
    check(bool(f),"Malformed two-body reference");r.decay.status=InteractionStatus::success;
  }
  std::string extra;check(!(f>>extra),"Extra two-body rows");return out;
}
struct Limited {
  int left{};Random r;
  KOKKOS_INLINE_FUNCTION bool next(double& v){return left-->0&&r.next(v);}
};
struct Invalid {KOKKOS_INLINE_FUNCTION bool next(double& v){v=1.;return true;}};
struct Contracts {
  KOKKOS_INLINE_FUNCTION void operator()(int,int& errors)const {
    TwoBodyInput q{2000.,782.65,105.6583755,105.6583755,-13,13,{0.,0.,1.},TwoBodyKind::muon_pair};
    for(int mode=0;mode<2;++mode) {
      q.kind=TwoBodyKind(mode);
      for(int draws=0;draws<(mode?4:2);++draws) {
        Limited r{draws};auto result=sampleTwoBodyDecay(q,r);
        errors+=result.status!=InteractionStatus::random_failure||result.count!=0;
      }
    }
    q.kind=TwoBodyKind::muon_pair;Random r;Invalid bad;
    errors+=sampleTwoBodyDecay(q,bad).status!=InteractionStatus::random_failure;
    auto before=r.state;q.second_mass_MeV=110.;
    errors+=sampleTwoBodyDecay(q,r).status!=InteractionStatus::invalid_input||r.state!=before;
    q.second_mass_MeV=q.first_mass_MeV;q.first_pdg=13;
    errors+=sampleTwoBodyDecay(q,r).status!=InteractionStatus::invalid_input;
    q.first_pdg=-13;q.total_MeV=q.parent_mass_MeV;
    errors+=sampleTwoBodyDecay(q,r).status!=InteractionStatus::invalid_input;
    q.total_MeV=2000.;q.direction.z=2.;
    errors+=sampleTwoBodyDecay(q,r).status!=InteractionStatus::invalid_input;
    q.direction.z=1.;q.parent_mass_MeV=200.;
    errors+=sampleTwoBodyDecay(q,r).status!=InteractionStatus::invalid_input;
    q.parent_mass_MeV=782.65;q.kind=TwoBodyKind(8);
    errors+=sampleTwoBodyDecay(q,r).status!=InteractionStatus::invalid_input;
    q.kind=TwoBodyKind::general;q.second_mass_MeV=-1.;
    errors+=sampleTwoBodyDecay(q,r).status!=InteractionStatus::invalid_input;
  }
};
void compare(std::vector<Input> const& input,std::vector<Result> const& ref,std::vector<Result> const& got,char const* label) {
  int controls=0;double de=0.,dd=0.,lowdd=0.,dp=0.,ec=0.,pc=0.,norm=0.;
  for(std::size_t i=0;i<input.size();++i) {
    auto q=input[i].decay;auto a=ref[i],b=got[i];
    check(b.decay.status==InteractionStatus::success&&b.decay.count==2,"Two-body generation failed");
    controls+=a.state!=b.state||a.used!=b.used||a.decay.status!=b.decay.status||a.decay.count!=b.decay.count;
    controls+=b.used!=(q.kind==TwoBodyKind::muon_pair?4:2);
    double total=0.,momentum[3]{};
    for(int j=0;j<2;++j) {
      auto x=a.decay.particle[j],y=b.decay.particle[j];controls+=x.pdg!=y.pdg;
      de=std::max(de,std::abs(x.energy_MeV-y.energy_MeV)/x.energy_MeV);
      double dx=std::max({std::abs(x.direction.x-y.direction.x),std::abs(x.direction.y-y.direction.y),
        std::abs(x.direction.z-y.direction.z)});
      dd=std::max(dd,dx);if(q.total_MeV/q.parent_mass_MeV<1.e5)lowdd=std::max(lowdd,dx);
      dp=std::max({dp,std::abs(a.decay.polarization_cosine[j]-b.decay.polarization_cosine[j]),
        std::abs(a.decay.polarization_azimuth[j]-b.decay.polarization_azimuth[j])});
      double mass=j?q.first_mass_MeV:q.second_mass_MeV;
      check(y.energy_MeV>=mass,"Daughter below rest mass");
      double p=std::sqrt((y.energy_MeV-mass)*(y.energy_MeV+mass));
      total+=y.energy_MeV;momentum[0]+=p*y.direction.x;momentum[1]+=p*y.direction.y;momentum[2]+=p*y.direction.z;
      norm=std::max(norm,std::abs(y.direction.x*y.direction.x+y.direction.y*y.direction.y+y.direction.z*y.direction.z-1.));
    }
    double parent_p=std::sqrt((q.total_MeV-q.parent_mass_MeV)*(q.total_MeV+q.parent_mass_MeV));
    pc=std::max({pc,std::abs(momentum[0]-parent_p*q.direction.x)/q.total_MeV,
      std::abs(momentum[1]-parent_p*q.direction.y)/q.total_MeV,std::abs(momentum[2]-parent_p*q.direction.z)/q.total_MeV});
    ec=std::max(ec,std::abs(total/q.total_MeV-1.));
  }
  std::cout<<label<<" inputs="<<input.size()<<" control_rng_order_mismatches="<<controls<<" max_energy_relative="<<de
    <<" max_direction_absolute="<<dd<<" direction_gamma_below_1e5="<<lowdd<<" polarization_difference="<<dp
    <<" energy_closure_relative="<<ec<<" momentum_closure_over_energy="<<pc<<" direction_norm_error="<<norm<<'\n';
  check(!controls&&de<1.e-11&&dd<1.e-6&&lowdd<1.e-9&&dp<1.e-13,"Two-body C7 comparison differs");
  check(ec<1.e-12&&pc<1.e-7&&norm<1.e-12,"Two-body conservation differs");
}
struct Batch {
  Kokkos::View<Input*> input;Kokkos::View<Result*> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(input(i));}
};
}
int main(int argc,char** argv) {
  try {
    auto input=cases();if(argc==3&&std::string(argv[1])=="--make-cases"){writeCases(argv[2],input);return 0;}
    if(argc!=2)return 2;Kokkos::ScopeGuard guard(argc,argv);
    std::cout<<std::setprecision(17)<<"execution_space="<<Kokkos::DefaultExecutionSpace::name()<<'\n';
    Contracts contracts;int errors=0;contracts(0,errors);check(!errors,"Host two-body contracts");
    Kokkos::parallel_reduce("two_body_contracts",1,contracts,errors);check(!errors,"Device two-body contracts");
    std::vector<Result> host;for(auto q:input)host.push_back(evaluate(q));
    bool self=std::string(argv[1])=="--contracts";auto ref=self?host:readReference(argv[1],input);
    compare(input,ref,host,self?"host_contracts":"host_C7");
    Kokkos::View<Input*> in("two_body_inputs",input.size());auto h=Kokkos::create_mirror_view(in);
    for(std::size_t i=0;i<input.size();++i)h(i)=input[i];Kokkos::deep_copy(in,h);
    Kokkos::View<Result*> out("two_body_outputs",input.size());Kokkos::parallel_for("two_body",input.size(),Batch{in,out});
    auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);std::vector<Result> device;
    for(std::size_t i=0;i<input.size();++i)device.push_back(result(i));
    compare(input,ref,device,self?"device_host":"device_C7");
    std::cout<<"scope=DECAY1/DECAY2 pre-cut two-body kernels needed by omega/phi; branch selection, three-body modes, host transport and full shower not yet covered\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
