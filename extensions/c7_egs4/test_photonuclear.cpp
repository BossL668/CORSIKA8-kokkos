#include "Egs4Photonuclear.hpp"
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
  KOKKOS_INLINE_FUNCTION bool next(double& value){state=(16807*state)%2147483647;++used;value=double(state)/2147483647.;return true;}
};
struct Input {int process{};double energy{};Direction direction;std::uint64_t seed1{},seed2{};};
struct Result {int status{},target{},branch{},count{},used[2]{},pdg[3]{};std::uint64_t state[2]{};double fields[12]{};};
// Same explicit masses (GeV -> MeV) supplied to C7 fixture; not a model default.
KOKKOS_INLINE_FUNCTION PhotonuclearMasses masses(){return {.93827208816*1.e3,.93956542052*1.e3,.1349768*1.e3,.13957039*1.e3,
  1.019461*1.e3,.78265*1.e3,.77526*1.e3};}
KOKKOS_INLINE_FUNCTION Result evaluate(Input q,TransverseMomentumTable const& table) {
  Random hadron{q.seed1},em{q.seed2};Result out;
  if(q.process==0){auto r=sampleTransverseMomentum(table,hadron);out.status=int(r.status);out.fields[0]=r.GeV;}
  else if(q.process==1) {
    auto r=sampleSinglePion(q.energy,q.direction,masses(),table,em,hadron);
    out.status=int(r.secondaries.status);out.target=r.target_pdg;out.branch=1;out.count=r.secondaries.count;
    for(int i=0;i<2;++i) {
      auto p=r.secondaries.particle[i];out.pdg[i]=p.pdg;out.fields[4*i]=p.energy_MeV;
      out.fields[4*i+1]=p.direction.x;out.fields[4*i+2]=p.direction.y;out.fields[4*i+3]=p.direction.z;
    }
  } else if(q.process==4) {
    auto r=selectPhotonuclearBranch(q.energy,masses(),em);out.status=int(r.status);out.branch=int(r.branch);
  } else {
    PhotonuclearOutcome r;
    if(q.process==2)r=sampleDoublePion(q.energy,q.direction,masses(),table,em,hadron);
    else if(q.process==3)r=sampleVectorMeson(q.energy,q.direction,masses(),em);
    else r=samplePhotonuclear(q.energy,q.direction,masses(),table,em,hadron);
    out.status=int(r.status);out.target=r.target_pdg;out.branch=int(r.branch);out.count=r.count;
    for(int i=0;i<3;++i) {
      auto p=r.particle[i];out.pdg[i]=p.pdg;out.fields[4*i]=p.energy_MeV;
      out.fields[4*i+1]=p.direction.x;out.fields[4*i+2]=p.direction.y;out.fields[4*i+3]=p.direction.z;
    }
  }
  out.used[0]=hadron.used;out.used[1]=em.used;out.state[0]=hadron.state;out.state[1]=em.state;return out;
}
std::vector<Input> cases() {
  std::vector<Input> result;
  for(int i=0;i<256;++i)result.push_back({0,0.,{0.,0.,1.},1009+7919*std::uint64_t(i),1});
  for(double e:{152.,155.,200.,400.,900.,900.000001,1100.,1300.,1300.000001,1399.999})
    for(Direction d:std::vector<Direction>{{0.,0.,1.},{0.,0.,-1.},{.3,-.4,std::sqrt(.75)}})
      for(int seed=0;seed<64;++seed) {
        auto n=std::uint64_t(result.size());result.push_back({1,e,d,101+2719*n,973+104729*n});
      }
  for(int process:{2,3,4,5}) {
    std::vector<double> energies=process==2?std::vector<double>{400.,400.000001,900.,1400.,2000.,2999.999}:
      process==3?std::vector<double>{2000.,3000.,1.e4,1.e6,1.e8,1.e11}:
      std::vector<double>{152.,400.,400.000001,1399.999,1400.,2000.,2000.000001,2500.,3000.,1.e6,1.e11};
    for(double e:energies)for(Direction d:std::vector<Direction>{{0.,0.,1.},{0.,0.,-1.},{.3,-.4,std::sqrt(.75)}})
      for(int seed=0;seed<64;++seed) {
        auto n=std::uint64_t(result.size());result.push_back({process,e,d,101+2719*n,973+104729*n});
      }
  }
  return result;
}
void check(bool value,char const* message){if(!value)throw std::runtime_error(message);}
void writeCases(std::string const& path,std::vector<Input> const& input) {
  std::ofstream f(path);f<<std::setprecision(17)<<input.size()<<'\n';
  for(auto q:input)f<<q.process<<' '<<q.energy<<" .93827208816 .93956542052 .1349768 .13957039 1.019461 .78265 .77526 "
    <<q.direction.x<<' '<<q.direction.y<<' '<<q.direction.z<<' '<<q.seed1<<' '<<q.seed2<<'\n';
  check(bool(f),"Cannot write photonuclear inputs");
}
std::vector<Result> readReference(std::string const& path,std::vector<Input> const& input) {
  std::ifstream f(path);std::vector<Result> result(input.size());
  for(std::size_t i=0;i<input.size();++i) {
    auto& r=result[i];int process;f>>process>>r.target>>r.branch>>r.count>>r.used[0]>>r.used[1]>>r.state[0]>>r.state[1];
    for(auto& pdg:r.pdg)f>>pdg;
    for(auto& field:r.fields)f>>field;
    check(bool(f)&&process==input[i].process,"Malformed C7 photonuclear reference");
    for(auto& pdg:r.pdg)pdg=pdg==7?111:pdg==8?211:pdg==9?-211:pdg==13?2112:pdg==14?2212:pdg==49?333:pdg==50?223:pdg==51?113:0;
    r.status=int(process==5&&r.branch==4?InteractionStatus::host_required:InteractionStatus::success);
  }
  std::string extra;check(!(f>>extra),"Extra reference rows");return result;
}
struct Empty {KOKKOS_INLINE_FUNCTION bool next(double&){return false;}};
struct Reject {int index{};KOKKOS_INLINE_FUNCTION bool next(double& v){v=(index++%2==0)?0.:.999999;return true;}};
struct Contracts {
  TransverseMomentumTable table;
  KOKKOS_INLINE_FUNCTION void operator()(int,int& errors)const {
    Empty empty;Random em{101},hadron{7919};Reject reject;
    errors+=sampleTransverseMomentum(table,empty).status!=InteractionStatus::random_failure;
    errors+=sampleTransverseMomentum(table,reject,2).status!=InteractionStatus::trial_limit;
    errors+=makeTransverseMomentumTable(.1,26.).intervals!=0;
    errors+=makeTransverseMomentumTable(-.1,20.).intervals!=0;
    errors+=sampleSinglePion(200.,{0.,0.,1.},masses(),table,empty,hadron).secondaries.status!=InteractionStatus::random_failure;
    errors+=sampleSinglePion(1400.,{0.,0.,1.},masses(),table,em,empty).secondaries.status!=InteractionStatus::random_failure;
    errors+=sampleSinglePion(10.,{0.,0.,1.},masses(),table,em,hadron).secondaries.status!=InteractionStatus::invalid_input;
    errors+=sampleSinglePion(200.,{0.,0.,2.},masses(),table,em,hadron).secondaries.status!=InteractionStatus::invalid_input;
    errors+=sampleSinglePion(200.,{0.,0.,1.},masses(),table,em,hadron,0).secondaries.status!=InteractionStatus::invalid_input;
    errors+=sampleDoublePion(200.,{0.,0.,1.},masses(),table,em,hadron).status!=InteractionStatus::invalid_input;
    errors+=sampleDoublePion(900.,{0.,0.,1.},masses(),table,empty,hadron).status!=InteractionStatus::random_failure;
    errors+=sampleDoublePion(900.,{0.,0.,1.},masses(),table,em,empty).status!=InteractionStatus::random_failure;
    errors+=sampleDoublePion(900.,{0.,0.,1.},masses(),table,em,hadron,0).status!=InteractionStatus::invalid_input;
    errors+=sampleVectorMeson(3000.,{0.,0.,1.},masses(),empty).status!=InteractionStatus::random_failure;
    errors+=sampleVectorMeson(10.,{0.,0.,1.},masses(),em).status!=InteractionStatus::invalid_input;
    auto incomplete=masses();incomplete.rho_MeV=0.;
    errors+=sampleVectorMeson(3000.,{0.,0.,1.},incomplete,em).status!=InteractionStatus::invalid_input;
    errors+=selectPhotonuclearBranch(3000.,masses(),empty).status!=InteractionStatus::random_failure;
    errors+=samplePhotonuclear(3000.,{0.,0.,2.},masses(),table,em,hadron).status!=InteractionStatus::invalid_input;
  }
};
void compare(std::vector<Input> const& inputs,std::vector<Result> const& ref,std::vector<Result> const& got,char const* label,
    bool original_reference) {
  int controls=0,numerical=0,channels[4]{},double_channels[6]{},meson_channels[6]{},counts[6]{},branches[4]{};
  double emax=0.,dmax=0.,ptmax=0.,closure=0.,momentum=0.,norm=0.,incident_error=0.,momentum_difference=0.;
  double process_energy[6]{},process_direction[6]{};
  auto m=masses();
  for(std::size_t i=0;i<inputs.size();++i) {
    auto a=ref[i],b=got[i];auto q=inputs[i];++counts[q.process];
    controls+=a.status!=b.status||a.target!=b.target||a.count!=b.count||a.branch!=b.branch;
    if(b.status!=int(InteractionStatus::success)&&b.status!=int(InteractionStatus::host_required)) {
      std::cerr<<"invalid native result row="<<i<<" process="<<q.process<<" energy="<<q.energy<<" status="<<b.status<<'\n';
      throw std::runtime_error("Native photonuclear input not processed");
    }
    for(int j=0;j<2;++j)controls+=a.used[j]!=b.used[j]||a.state[j]!=b.state[j];
    for(int j=0;j<3;++j)controls+=a.pdg[j]!=b.pdg[j];
    for(int j=0;j<12;++j) {
      check(std::isfinite(b.fields[j]),"Nonfinite photonuclear output");double delta=std::abs(a.fields[j]-b.fields[j]);
      if(q.process==0)ptmax=std::max(ptmax,delta);
      else if(j%4==0) {
        double rel=delta/std::max(1.,std::abs(a.fields[j]));emax=std::max(emax,rel);
        process_energy[q.process]=std::max(process_energy[q.process],rel);
        incident_error=std::max(incident_error,delta/(q.energy+m.neutron_MeV));
        // The original RHOGEN recoil uses E_gamma + m_target - E_meson;
        // native C++ evaluates the same sampled kinematics via invariants.
        double allowed=1.e-10*std::max(1.,std::abs(a.fields[j]))+
          64.*std::numeric_limits<double>::epsilon()*(q.energy+m.neutron_MeV);
        numerical+=delta>allowed;
      } else {dmax=std::max(dmax,delta);process_direction[q.process]=std::max(process_direction[q.process],delta);}
    }
    if(q.process==4||q.process==5)++branches[b.branch-1];
    if(!b.count)continue;
    if(q.process==1)channels[(b.target==2212?0:2)+(b.pdg[0]!=111)]++;
    if(q.process==2) {
      bool neutral=b.pdg[0]==111&&b.pdg[1]==111,opposite=b.pdg[0]*b.pdg[1]==-211*211;
      ++double_channels[(b.target==2212?0:3)+(neutral?0:opposite?1:2)];
    }
    if(q.process==3)++meson_channels[(b.target==2212?0:3)+(b.pdg[0]==333?0:b.pdg[0]==223?1:2)];
    double target=b.target==2212?m.proton_MeV:m.neutron_MeV;
    double total=0.;for(int p=0;p<b.count;++p)total+=b.fields[4*p];
    closure=std::max(closure,std::abs(total/(q.energy+target)-1.));
    double psum[3]{};
    for(int p=0;p<b.count;++p) {
      int id=b.pdg[p];double mass=id==111?m.neutral_pion_MeV:(id==211||id==-211)?m.charged_pion_MeV:
        id==2212?m.proton_MeV:id==2112?m.neutron_MeV:id==333?m.phi_MeV:id==223?m.omega_MeV:m.rho_MeV;
      double e=b.fields[4*p],mag=std::sqrt(std::max(0.,(e-mass)*(e+mass))),n=0.;
      double ae=a.fields[4*p],amag=std::sqrt(std::max(0.,(ae-mass)*(ae+mass)));
      check(e>=mass,"Submass photonuclear final state");
      for(int j=0;j<3;++j) {
        double v=b.fields[4*p+j+1],av=a.fields[4*p+j+1];psum[j]+=mag*v;n+=v*v;
        // Only comparison to the original high-energy RHOGEN arithmetic
        // needs a conditioning allowance. Native host/device comparisons
        // and other channels retain the tighter bound at every energy.
        bool conditioned_reference=original_reference&&b.branch==3&&q.energy>1.e6;
        double allowed=conditioned_reference?8.*std::sqrt(std::numeric_limits<double>::epsilon())+
          64.*std::numeric_limits<double>::epsilon()*(q.energy+target)/std::max(1.,amag):1.e-9;
        numerical+=std::abs(v-av)>allowed;
        momentum_difference=std::max(momentum_difference,std::abs(mag*v-amag*av)/q.energy);
      }
      norm=std::max(norm,std::abs(n-1.));
    }
    double d[]={q.direction.x,q.direction.y,q.direction.z};
    for(int j=0;j<3;++j)momentum=std::max(momentum,std::abs(psum[j]-q.energy*d[j])/q.energy);
  }
  std::cout<<label<<" inputs="<<inputs.size()<<" control_rng_pid_mismatches="<<controls
    <<" max_energy_relative="<<emax<<" max_direction_absolute="<<dmax<<" max_pt_GeV="<<ptmax
    <<" energy_closure_relative="<<closure<<" momentum_closure_relative="<<momentum<<" direction_norm_error="<<norm
    <<" energy_difference_over_incident="<<incident_error<<" momentum_difference_over_incident="<<momentum_difference
    <<" numerical_checks_failed="<<numerical<<" channels=";
  for(auto n:channels)std::cout<<n<<',';std::cout<<" process_counts=";
  for(auto n:counts)std::cout<<n<<',';std::cout<<" selected_branches=";
  for(auto n:branches)std::cout<<n<<',';std::cout<<'\n';
  std::cout<<label<<" double_pion_target_channels=";for(auto n:double_channels)std::cout<<n<<',';
  std::cout<<" vector_meson_target_channels=";for(auto n:meson_channels)std::cout<<n<<',';std::cout<<'\n';
  for(int p=1;p<6;++p)std::cout<<label<<" process="<<p<<" max_energy_relative="<<process_energy[p]
    <<" max_direction_absolute="<<process_direction[p]<<'\n';
  // Original RHOGEN loses transverse bits at 1e11 MeV. Native kinematics
  // must conserve momentum tightly regardless of the reference's roundoff.
  double reference_momentum_bound=original_reference?8.*std::sqrt(std::numeric_limits<double>::epsilon()):1.e-10;
  check(controls==0&&numerical==0&&incident_error<1.e-12&&momentum_difference<reference_momentum_bound&&
    ptmax<1.e-12&&closure<1.e-12&&momentum<1.e-10&&norm<1.e-10,
    "Native photonuclear disagreement");
  for(auto n:channels)check(n>0,"Missing PIGEN1 charge/target branch coverage");
  for(auto n:double_channels)check(n>0,"Missing PIGEN2 charge/target coverage");
  for(auto n:meson_channels)check(n>0,"Missing RHOGEN meson/target coverage");
  for(auto n:branches)check(n>0,"Missing PIGEN selector branch coverage");
}
struct Batch {
  Kokkos::View<Input*> input;Kokkos::View<Result*> output;TransverseMomentumTable table;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const {output(i)=evaluate(input(i),table);}
};
}
int main(int argc,char** argv) {
  try {
    auto input=cases();
    if(argc==3&&std::string(argv[1])=="--make-cases"){writeCases(argv[2],input);return 0;}
    if(argc!=2)return 2;Kokkos::ScopeGuard guard(argc,argv);
    std::cout<<std::setprecision(17)<<"execution_space="<<Kokkos::DefaultExecutionSpace::name()<<'\n';
    auto table=makeTransverseMomentumTable();int errors=0;Contracts contracts{table};contracts(0,errors);
    check(errors==0,"Host photonuclear contracts");Kokkos::parallel_reduce("pigen_contracts",1,contracts,errors);check(!errors,"Device photonuclear contracts");
    std::vector<Result> host;for(auto q:input)host.push_back(evaluate(q,table));
    bool self=std::string(argv[1])=="--contracts";auto reference=self?host:readReference(argv[1],input);
    compare(input,reference,host,self?"host_contracts":"host_C7",!self);
    Kokkos::View<Input*> dinput("pigen_inputs",input.size());auto h=Kokkos::create_mirror_view(dinput);
    for(std::size_t i=0;i<input.size();++i)h(i)=input[i];Kokkos::deep_copy(dinput,h);
    Kokkos::View<Result*> output("pigen_outputs",input.size());Kokkos::parallel_for("native_pigen1",input.size(),Batch{dinput,output,table});
    auto out=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);std::vector<Result> device;
    for(std::size_t i=0;i<input.size();++i)device.push_back(out(i));compare(input,reference,device,self?"device_host":"device_C7",!self);
    std::cout<<"scope=PIGEN1/PIGEN2/RHOGEN pre-cut pre-decay vertices, PTRANS, PIGEN selection and composition; explicit many-hadron host request; no nuclear transport, thinning, production integration or timing claim\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
