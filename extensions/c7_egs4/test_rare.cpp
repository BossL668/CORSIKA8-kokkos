#include "Egs4RareFinalStates.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {
using namespace c7_egs4;
struct Random {
  std::uint64_t state{1};int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& x){state=(16807*state)%2147483647;++used;x=double(state)/2147483647.;return true;}
};
struct Input {int process{},pdg{};double energy{},cut{.5},threshold{152.};Direction direction;RareMaterial medium;std::uint64_t seed{};};
struct Result {int process{},count{},used{},target{},pdg[2]{},status{};std::uint64_t state{};double fields[12]{};int trials{};};
KOKKOS_INLINE_FUNCTION Result evaluate(Input const& q) {
  Result out;out.process=q.process;Random rng{q.seed};Secondary particles[2];
  if(q.process==0) {
    auto result=sampleMuonPair(q.energy,q.direction,q.medium,rng);
    auto s=result.secondaries;out.status=int(s.status);out.count=s.count;out.target=result.target_atomic_number;
    for(int i=0;i<2;++i){particles[i]=s.particle[i];out.fields[8+2*i]=result.polarization_cosine[i];out.fields[9+2*i]=result.polarization_azimuth[i];}
    out.trials=s.trials;
  } else {
    auto result=sampleElectronuclear(q.energy,q.pdg,q.direction,q.medium,q.cut,q.threshold,rng);
    out.status=int(result.status);out.count=result.status==InteractionStatus::success?2:1;
    particles[0]=result.residual;particles[1]=result.virtual_photon;out.trials=result.trials;
  }
  out.used=rng.used;out.state=rng.state;
  for(int i=0;i<2;++i) {
    out.pdg[i]=particles[i].pdg;out.fields[4*i]=particles[i].energy_MeV;
    out.fields[4*i+1]=particles[i].direction.x;out.fields[4*i+2]=particles[i].direction.y;out.fields[4*i+3]=particles[i].direction.z;
  }
  return out;
}
std::vector<Input> cases() {
  std::vector<Input> result;Direction directions[]={{0.,0.,1.},{0.,0.,-1.},{.3,-.4,std::sqrt(.75)}};
  RareMaterial medium{.51099895,105.6583755,139.57039,938.27208816,14.543,{.7847,.2105,.0048}};
  for(int process:{0,1})for(double energy:process==0?std::vector<double>{500.,1.e3,1.e4,1.e6,1.e9,1.e12,1.e15}:
      std::vector<double>{1.e3,1.e4,1.e6,1.e9,1.e9+1.,1.e12})
    for(int material=0;material<(process==0?4:1);++material)for(auto direction:directions)for(int tape=0;tape<16;++tape)
      for(int pdg:process==0?std::vector<int>{22}:std::vector<int>{11,-11}) {
        auto m=medium;
        if(material)for(int i=0;i<3;++i)m.composition[i]=i==material-1?1.:0.;
        result.push_back({process,pdg,energy,.5,152.,direction,m,17+97*result.size()});
      }
  // Source ELNUCL early return when already below the electron kinetic cut.
  for(int pdg:{11,-11})result.push_back({1,pdg,1.,.5,152.,{0.,0.,1.},medium,711});
  // C7 returns the lepton unchanged after 1000 rejected sub-threshold proposals.
  result.push_back({1,11,1000.,.5,1000.,{0.,0.,1.},medium,929});
  return result;
}
void writeCases(std::string const& path,std::vector<Input> const& inputs) {
  std::ofstream f(path);f<<std::setprecision(17)<<inputs.size()<<'\n';
  for(auto q:inputs) {
    auto m=q.medium;f<<q.process<<' '<<q.pdg<<' '<<q.energy<<' '<<m.electron_mass_MeV<<' '<<m.muon_mass_MeV<<' '
      <<m.charged_pion_mass_MeV<<' '<<m.proton_mass_MeV<<' '<<m.average_atomic_weight<<' ';
    for(double c:m.composition)f<<c<<' ';
    f<<q.cut<<' '<<q.threshold<<' '<<q.direction.x<<' '<<q.direction.y<<' '<<q.direction.z<<' '<<q.seed<<'\n';
  }
  if(!f)throw std::runtime_error("Cannot write rare-vertex fixture inputs");
}
std::vector<Result> readReference(std::string const& path,std::size_t count) {
  std::ifstream f(path);std::vector<Result> output(count);
  for(auto& r:output) {
    int target;f>>r.process>>r.count>>r.used>>r.state>>target>>r.pdg[0]>>r.pdg[1];
    for(double& x:r.fields)f>>x;
    if(!f)throw std::runtime_error("Missing/malformed original C7 rare-vertex reference");
    r.target=target==1?7:target==2?8:target==3?18:0;
    r.status=int(r.process==1&&r.count==1?InteractionStatus::suppressed:InteractionStatus::success);
    if(r.count==1){r.pdg[1]=0;for(int j=4;j<8;++j)r.fields[j]=0.;}
  }
  std::string extra;if(f>>extra)throw std::runtime_error("Extra rows in rare-vertex reference");return output;
}
struct Batch {
  Kokkos::View<Input*> input;Kokkos::View<Result*> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const {output(i)=evaluate(input(i));}
};
void check(bool value,char const* message){if(!value)throw std::runtime_error(message);}
struct Exhausted {KOKKOS_INLINE_FUNCTION bool next(double&){return false;}};
struct Zero {KOKKOS_INLINE_FUNCTION bool next(double& value){value=0.;return true;}};
struct ContractChecks {
  Input base;
  KOKKOS_INLINE_FUNCTION void operator()(int,int& errors)const {
    Exhausted empty;Zero zero;Random random{17};auto m=base.medium;
    errors+=sampleMuonPair(4.*m.muon_mass_MeV,base.direction,m,random).secondaries.status!=InteractionStatus::invalid_input;
    errors+=sampleMuonPair(1000.,base.direction,m,empty).secondaries.status!=InteractionStatus::random_failure;
    errors+=sampleMuonPair(1000.,base.direction,m,zero).secondaries.status!=InteractionStatus::random_failure;
    errors+=sampleMuonPair(1000.,base.direction,m,random,0).secondaries.status!=InteractionStatus::invalid_input;
    errors+=sampleElectronuclear(1000.,11,base.direction,m,.5,152.,empty).status!=InteractionStatus::random_failure;
    errors+=sampleElectronuclear(1000.,11,base.direction,m,.5,152.,zero).status!=InteractionStatus::random_failure;
    errors+=sampleElectronuclear(200.,11,base.direction,m,.5,152.,random).status!=InteractionStatus::invalid_input;
    errors+=sampleElectronuclear(1000.,22,base.direction,m,.5,152.,random).status!=InteractionStatus::invalid_input;
    m.composition[0]=m.composition[1]=m.composition[2]=0.;
    errors+=sampleMuonPair(1000.,base.direction,m,random).secondaries.status!=InteractionStatus::invalid_input;
  }
};
void compare(std::vector<Input> const& inputs,std::vector<Result> const& reference,std::vector<Result> const& got,char const* label) {
  double energy_error=0.,direction_error=0.,polarization_error=0.,balance=0.,norm_error=0.;int suppressions=0,rejections=0;
  int controls=0;
  for(std::size_t i=0;i<inputs.size();++i) {
    auto a=reference[i],b=got[i];controls+=a.status!=b.status||a.count!=b.count||a.used!=b.used||a.state!=b.state||a.target!=b.target;
    suppressions+=b.status==int(InteractionStatus::suppressed);rejections+=b.trials>(inputs[i].process==0?3:1);
    for(int j=0;j<2;++j)controls+=a.pdg[j]!=b.pdg[j];
    for(int j=0;j<12;++j) {
      check(std::isfinite(b.fields[j]),"Non-finite native rare-vertex result");
      auto difference=std::abs(a.fields[j]-b.fields[j]);
      if(j==0||j==4)energy_error=std::max(energy_error,difference/std::max(1.,std::abs(a.fields[j])));
      else if(j>=8)polarization_error=std::max(polarization_error,difference);
      else direction_error=std::max(direction_error,difference);
    }
    balance=std::max(balance,std::abs((b.fields[0]+b.fields[4])/inputs[i].energy-1.));
    for(int j=0;j<b.count;++j){double norm=0.;for(int k=1;k<4;++k)norm+=b.fields[4*j+k]*b.fields[4*j+k];norm_error=std::max(norm_error,std::abs(norm-1.));}
  }
  std::cout<<label<<" inputs="<<inputs.size()<<" control_rng_pid_mismatches="<<controls<<" max_energy_relative="<<energy_error
    <<" max_direction_absolute="<<direction_error<<" max_polarization_absolute="<<polarization_error
    <<" energy_closure_relative="<<balance<<" direction_norm_error="<<norm_error<<" suppressed="<<suppressions
    <<" inputs_with_rejection="<<rejections<<'\n';
  check(controls==0&&energy_error<1.e-9&&direction_error<1.e-8&&polarization_error<1.e-12&&balance<1.e-12&&norm_error<1.e-12,
    "Native rare vertices disagree with original C7 reference");
}
}
int main(int argc,char** argv) {
  try {
    auto inputs=cases();
    if(argc==3&&std::string(argv[1])=="--make-cases"){writeCases(argv[2],inputs);return 0;}
    if(argc!=2)return 2;Kokkos::ScopeGuard guard(argc,argv);std::cout<<std::setprecision(17)<<"execution_space="<<Kokkos::DefaultExecutionSpace::name()<<'\n';
    int errors=0;ContractChecks contracts{inputs[0]};contracts(0,errors);check(!errors,"Host rare-vertex contracts");
    Kokkos::parallel_reduce("native_rare_contracts",1,contracts,errors);check(!errors,"Device rare-vertex contracts");
    if(std::string(argv[1])=="--contracts"){std::cout<<"rare_vertex_contracts_passed\n";return 0;}
    auto reference=readReference(argv[1],inputs.size());std::vector<Result> host;
    for(auto q:inputs)host.push_back(evaluate(q));compare(inputs,reference,host,"host_C7");
    Kokkos::View<Input*> input("rare_inputs",inputs.size());auto h=Kokkos::create_mirror_view(input);
    for(std::size_t i=0;i<inputs.size();++i)h(i)=inputs[i];Kokkos::deep_copy(input,h);
    Kokkos::View<Result*> output("rare_outputs",inputs.size());Kokkos::parallel_for("native_rare_vertices",inputs.size(),Batch{input,output});
    auto out=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);std::vector<Result> device;
    for(std::size_t i=0;i<inputs.size();++i)device.push_back(out(i));compare(inputs,reference,device,"device_C7");
    std::cout<<"scope=C++ MUPAIR final states/polarization and ELNUCL virtual-photon sharing; PIGEN/muon cuts/stack integration remain separate\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
