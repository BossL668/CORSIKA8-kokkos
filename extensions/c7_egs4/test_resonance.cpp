#include "Egs4ResonanceDecay.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
using namespace c7_egs4;
void check(bool v,char const* message){if(!v)throw std::runtime_error(message);}
KOKKOS_INLINE_FUNCTION ResonanceMasses masses(){return {782.65,1019.461,139.57039,134.9768,105.6583755,493.677,497.611,497.611,547.862};}
struct Random {
  std::uint64_t state{1};int used{};double first{-1.};
  KOKKOS_INLINE_FUNCTION bool next(double& v) {
    state=(16807*state)%2147483647;v=double(state)/2147483647.;if(used++==0&&first>=0.)v=first;return true;
  }
};
struct Input {ResonanceInput decay;std::uint64_t seed{};double first{};};
struct Result {ResonanceOutcome decay;std::uint64_t state{};int used{};};
KOKKOS_INLINE_FUNCTION Result evaluate(Input q) {
  Random r{q.seed,0,q.first};auto d=sampleResonanceDecay(q.decay,masses(),r);return {d,r.state,r.used};
}
std::vector<Input> cases() {
  std::vector<Input> out;
  for(int id:{223,333}) {
    std::vector<double> first{.01,.9999999};
    auto thresholds=id==223?std::vector<double>{.8996252,.9843332,.9997739,.9999090}:
      std::vector<double>{.4901808,.8330066,.9865765,.9996981,.9999857};
    for(double t:thresholds){first.push_back(std::nextafter(t,0.));first.push_back(t);first.push_back(std::nextafter(t,1.));}
    for(double gamma:{1.0001,1.01,1.5,2.,10.,1.e3,1.e5,1.e8})
      for(Direction d:std::vector<Direction>{{0.,0.,1.},{0.,0.,-1.},{.3,-.4,std::sqrt(.75)},
          {1.,0.,0.},{1.e-21,0.,1.},{0.,1.e-19,-1.}})
        for(double u:first)for(int seed=0;seed<16;++seed) {
          auto m=masses();out.push_back({{gamma*(id==223?m.omega_MeV:m.phi_MeV),id,d},101+7919*out.size(),u});
        }
  }
  return out;
}
void writeCases(std::string const& path,std::vector<Input> const& input) {
  std::ofstream f(path);f<<std::setprecision(17)<<input.size()<<'\n';auto m=masses();
  for(auto q:input){auto d=q.decay;f<<(d.pdg==223?50:49)<<' '<<d.total_MeV<<' '<<m.omega_MeV<<' '<<m.phi_MeV
    <<' '<<m.charged_pion_MeV<<' '<<m.neutral_pion_MeV<<' '<<m.muon_MeV<<' '<<m.charged_kaon_MeV
    <<' '<<m.long_kaon_MeV<<' '<<m.short_kaon_MeV<<' '<<m.eta_MeV<<' '<<d.direction.x<<' '<<d.direction.y
    <<' '<<d.direction.z<<' '<<q.seed<<' '<<q.first<<'\n';}check(bool(f),"Cannot write resonance inputs");
}
int pdg(int id) {
  switch(id){case 0:return 0;case 1:return 22;case 5:return -13;case 6:return 13;case 7:return 111;
    case 8:return 211;case 9:return -211;case 10:return 130;case 11:return 321;case 12:return -321;
    case 16:return 310;case 17:return 221;default:throw std::runtime_error("Unexpected source PID");}
}
ResonanceBranch inferBranch(int parent,ResonanceOutcome d) {
  int a=d.particle[0].pdg,b=d.particle[1].pdg,c=d.particle[2].pdg;using B=ResonanceBranch;
  if(parent==223) {
    if(d.count==3&&a==211&&b==-211&&c==111)return B::omega_three_pions;
    if(d.count==2&&a==22&&b==111)return B::omega_pi0_gamma;
    if(d.count==2&&a==-211&&b==211)return B::omega_two_pions;
    if(d.count==3&&a==-13&&b==13&&c==111)return B::omega_muons_pi0;
    if(d.count==2&&a==13&&b==-13)return B::omega_muons;
  }else if(parent==333) {
    if(d.count==2&&a==-321&&b==321)return B::phi_charged_kaons;
    if(d.count==2&&a==310&&b==130)return B::phi_neutral_kaons;
    if(d.count==3&&a==211&&b==-211&&c==111)return B::phi_three_pions;
    if(d.count==2&&a==22&&b==221)return B::phi_eta_gamma;
    if(d.count==2&&a==13&&b==-13)return B::phi_muons;
    if(d.count==3&&a==-13&&b==13&&c==22)return B::phi_muons_gamma;
  }
  throw std::runtime_error("Wrong resonance daughter ordering");
}
std::vector<Result> readReference(std::string const& path,std::vector<Input> const& input) {
  std::ifstream f(path);std::vector<Result> out(input.size());
  for(std::size_t i=0;i<input.size();++i) {
    auto& r=out[i];int parent=0,ids[3]{};f>>parent>>r.decay.count>>r.used>>r.state>>ids[0]>>ids[1]>>ids[2];
    check(parent==(input[i].decay.pdg==223?50:49),"Reference parent differs");
    for(int j=0;j<3;++j){auto& p=r.decay.particle[j];f>>p.energy_MeV>>p.direction.x>>p.direction.y>>p.direction.z
      >>r.decay.polarization_cosine[j]>>r.decay.polarization_azimuth[j];p.pdg=pdg(ids[j]);}
    check(bool(f),"Malformed resonance reference");r.decay.status=InteractionStatus::success;
    r.decay.branch=inferBranch(input[i].decay.pdg,r.decay);
  }
  std::string extra;check(!(f>>extra),"Extra resonance reference rows");return out;
}
KOKKOS_INLINE_FUNCTION double particleMass(int id) {
  auto m=masses();switch(id){case 22:return 0.;case 13:case -13:return m.muon_MeV;
    case 111:return m.neutral_pion_MeV;case 211:case -211:return m.charged_pion_MeV;
    case 321:case -321:return m.charged_kaon_MeV;case 130:return m.long_kaon_MeV;
    case 310:return m.short_kaon_MeV;case 221:return m.eta_MeV;default:return -1.;}
}
struct Limited {
  int left{};Random random;
  KOKKOS_INLINE_FUNCTION bool next(double& v){return left-->0&&random.next(v);}
};
struct Bad {KOKKOS_INLINE_FUNCTION bool next(double& v){v=1.;return true;}};
struct Rejected {
  int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& v){v=used%3==0?1.e-8:used%3==1?1.-1.e-8:.5;++used;return true;}
};
struct Contracts {
  KOKKOS_INLINE_FUNCTION void operator()(int,int& errors)const {
    ResonanceInput q{2000.,223,{0.,0.,1.}};auto m=masses();
    for(int parent:{223,333})for(double first:{.1,.95,.99985,.99999}) {
      q.pdg=parent;Random r{101,0,first};auto valid=sampleResonanceDecay(q,m,r);
      errors+=valid.status!=InteractionStatus::success;
      for(int n=0;n<r.used;++n) {
        Limited fail{n,{101,0,first}};auto d=sampleResonanceDecay(q,m,fail);
        errors+=d.status!=InteractionStatus::random_failure||d.count!=0;
      }
    }
    Random r;Bad bad;q.pdg=223;
    errors+=sampleResonanceDecay(q,m,bad).status!=InteractionStatus::random_failure;
    q.pdg=113;errors+=sampleResonanceDecay(q,m,r).status!=InteractionStatus::invalid_input;
    q.pdg=223;q.total_MeV=m.omega_MeV;errors+=sampleResonanceDecay(q,m,r).status!=InteractionStatus::invalid_input;
    q.total_MeV=2000.;m.short_kaon_MeV=-1.;errors+=sampleResonanceDecay(q,m,r).status!=InteractionStatus::invalid_input;
    m=masses();m.phi_MeV=500.;errors+=sampleResonanceDecay(q,m,r).status!=InteractionStatus::invalid_input;
    m=masses();q.direction.z=2.;errors+=sampleResonanceDecay(q,m,r).status!=InteractionStatus::invalid_input;
    ThreeBodyInput t{2000.,782.65,{139.57039,139.57039,134.9768},{211,-211,111},{0.,0.,1.}};
    Rejected reject;auto failed=sampleUniformThreeBody(t,reject,2);
    errors+=failed.status!=InteractionStatus::trial_limit||failed.count!=0||reject.used!=6;
    errors+=sampleUniformThreeBody(t,r,0).status!=InteractionStatus::invalid_input;
    t.mass_MeV[0]=0.;errors+=sampleUniformThreeBody(t,r).status!=InteractionStatus::invalid_input;
    t.mass_MeV[0]=900.;errors+=sampleUniformThreeBody(t,r).status!=InteractionStatus::invalid_input;
  }
};
void compare(std::vector<Input> const& inputs,std::vector<Result> const& ref,std::vector<Result> const& got,char const* label) {
  int control=0;std::uint64_t counts[12]{};double de=0.,dd=0.,lowdd=0.,dp=0.,ec=0.,pc=0.,norm=0.;
  int trials=0;
  for(std::size_t i=0;i<inputs.size();++i) {
    auto q=inputs[i].decay;auto a=ref[i],b=got[i];
    if(b.decay.status!=InteractionStatus::success){std::cerr<<"failed index="<<i<<" parent="<<q.pdg<<" first="<<inputs[i].first<<'\n';throw std::runtime_error("Resonance generation failed");}
    check(b.decay.count==2||b.decay.count==3,"Bad resonance count");
    control+=a.state!=b.state||a.used!=b.used||a.decay.count!=b.decay.count||a.decay.branch!=b.decay.branch;
    check(b.decay.branch==inferBranch(q.pdg,b.decay),"Branch does not describe products");
    ++counts[int(b.decay.branch)];trials=std::max(trials,b.decay.trials);
    double total=0.,pvec[3]{};
    for(int j=0;j<b.decay.count;++j) {
      auto x=a.decay.particle[j],y=b.decay.particle[j];control+=x.pdg!=y.pdg;
      de=std::max(de,std::abs(x.energy_MeV-y.energy_MeV)/x.energy_MeV);
      double delta=std::max({std::abs(x.direction.x-y.direction.x),std::abs(x.direction.y-y.direction.y),std::abs(x.direction.z-y.direction.z)});
      dd=std::max(dd,delta);if(q.total_MeV/(q.pdg==223?masses().omega_MeV:masses().phi_MeV)<1.e5)lowdd=std::max(lowdd,delta);
      dp=std::max({dp,std::abs(a.decay.polarization_cosine[j]-b.decay.polarization_cosine[j]),
        std::abs(a.decay.polarization_azimuth[j]-b.decay.polarization_azimuth[j])});
      double mass=particleMass(y.pdg);check(mass>=0.&&y.energy_MeV>=mass,"Bad daughter energy or PID");
      double p=std::sqrt((y.energy_MeV-mass)*(y.energy_MeV+mass));
      pvec[0]+=p*y.direction.x;pvec[1]+=p*y.direction.y;pvec[2]+=p*y.direction.z;total+=y.energy_MeV;
      norm=std::max(norm,std::abs(y.direction.x*y.direction.x+y.direction.y*y.direction.y+y.direction.z*y.direction.z-1.));
    }
    auto m=masses();double parent=q.pdg==223?m.omega_MeV:m.phi_MeV;
    double p=std::sqrt((q.total_MeV-parent)*(q.total_MeV+parent));
    pc=std::max({pc,std::abs(pvec[0]-p*q.direction.x)/q.total_MeV,std::abs(pvec[1]-p*q.direction.y)/q.total_MeV,
      std::abs(pvec[2]-p*q.direction.z)/q.total_MeV});ec=std::max(ec,std::abs(total/q.total_MeV-1.));
  }
  std::cout<<label<<" inputs="<<inputs.size()<<" control_rng_pid_mismatches="<<control<<" max_energy_relative="<<de
    <<" max_direction_absolute="<<dd<<" direction_gamma_below_1e5="<<lowdd<<" polarization_difference="<<dp
    <<" energy_closure_relative="<<ec<<" momentum_closure_over_energy="<<pc<<" norm_error="<<norm
    <<" max_dalitz_trials="<<trials<<" branches=";
  for(int i=1;i<=11;++i){std::cout<<counts[i]<<',';check(counts[i]>0,"Missed resonance branch");}std::cout<<'\n';
  check(!control&&de<1.e-10&&dd<1.e-6&&lowdd<1.e-8&&dp<1.e-13,"Resonance C7 comparison differs");
  check(ec<1.e-12&&pc<1.e-7&&norm<1.e-12,"Resonance conservation differs");
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
    Contracts contracts;int errors=0;contracts(0,errors);check(!errors,"Host resonance contracts");
    Kokkos::parallel_reduce("resonance_contracts",1,contracts,errors);check(!errors,"Device resonance contracts");
    std::vector<Result> host;for(auto q:input)host.push_back(evaluate(q));
    bool self=std::string(argv[1])=="--contracts";auto ref=self?host:readReference(argv[1],input);
    compare(input,ref,host,self?"host_contracts":"host_C7");
    Kokkos::View<Input*> in("resonance_inputs",input.size());auto h=Kokkos::create_mirror_view(in);
    for(std::size_t i=0;i<input.size();++i)h(i)=input[i];Kokkos::deep_copy(in,h);
    Kokkos::View<Result*> out("resonance_outputs",input.size());Kokkos::parallel_for("resonance",input.size(),Batch{in,out});
    auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);std::vector<Result> device;
    for(std::size_t i=0;i<input.size();++i)device.push_back(result(i));
    compare(input,ref,device,self?"device_host":"device_C7");
    std::cout<<"scope=pre-cut omega/phi RESDEC, 11 channels, actual <= boundaries, DECAY1/2 and uniform DECAY6; no production/host-cut integration or full-shower timing claim\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
