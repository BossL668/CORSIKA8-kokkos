#include "Egs4History.hpp"
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>

namespace {
using namespace c7_egs4;
void check(bool b,char const* message){if(!b)throw std::runtime_error(message);}
struct Stream {
  std::int64_t seed;int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& u){if(used>=100000)return false;seed=(seed*48271)%2147483647;u=double(seed)/2147483647.;++used;return true;}
};
struct Fixed {
  double value;int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& u){u=value;++used;return true;}
};
struct Case {TrackState state;double stepfc;std::int64_t seed;};
struct Answer {HistoryOutcome outcome;double deposit{},balance_error{};int steps{},used{},boundaries{},attempts{};};
KOKKOS_INLINE_FUNCTION double available(TrackState s,double m) {
  return s.energy_MeV+(s.pdg==11?-m:s.pdg==-11?m:0.);
}
KOKKOS_INLINE_FUNCTION TransportSettings settings(double stepfc) {
  TransportSettings s;s.electron_cut_total_MeV=1.;s.photon_cut_MeV=.5;s.stepfc=stepfc;return s;
}
// Composition test through the first non-suppressed reaction/cut. The periodic
// boundary is a supplied cap to test resumption, not a physical atmosphere edge.
KOKKOS_INLINE_FUNCTION Answer evaluate(ModelView model,Case q) {
  Stream rng{q.seed};auto s=q.state;Answer a;
  AirRegion region{.001225,1.25e-6,0.};MagneticField field{5.e-5*2.99792458,.6,.8,.2};
  auto config=settings(q.stepfc);
  for(int i=0;i<2048;++i) {
    auto p=planHistory(model,s,region,field,config,rng);
    auto cap=BoundaryLimit{p.proposed_cm*(i%5==0?.4:1.),i%5==0?BoundaryKind::density_layer:BoundaryKind::none};
    auto r=finishHistory(model,p,region,field,config,cap,rng);a.outcome=r;++a.attempts;
    a.steps+=r.has_track;a.boundaries+=r.action==HistoryAction::boundary;
    a.deposit+=r.continuous_deposit_MeV+r.vertex_deposit_MeV;
    if(r.action!=HistoryAction::active&&r.action!=HistoryAction::boundary)break;
    s=r.particle;
  }
  a.used=rng.used;
  if(a.outcome.action==HistoryAction::replaced) {
    double final=a.deposit;
    for(int j=0;j<a.outcome.child_count;++j)final+=available(a.outcome.children[j],model.thresholds.mass_MeV);
    a.balance_error=::fabs(final-available(q.state,model.thresholds.mass_MeV));
  }
  return a;
}
template<class Exec>struct Batch {
  ModelView model;Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(model,input(i));}
};
KOKKOS_INLINE_FUNCTION int contracts(ModelView model) {
  int errors=0;AirRegion air{.001225,1.25e-6,0.};MagneticField field{0.,0.,1.,.2};auto config=settings(.0625);
  TrackState state;state.pdg=11;state.energy_MeV=.75;state.direction={0.,0.,1.};state.position_cm={1.,2.,-1.e6};
  Fixed rng{.5};auto p=planHistory(model,state,air,field,config,rng);
  errors+=p.action!=PlanAction::energy_cut||rng.used!=0;
  auto r=finishHistory(model,p,air,field,config,{},rng);
  errors+=r.action!=HistoryAction::replaced||r.child_count!=0||r.has_track||rng.used!=0;
  errors+=r.vertex_deposit_MeV!=state.energy_MeV-model.thresholds.mass_MeV;
  state.pdg=-11;rng.used=0;p=planHistory(model,state,air,field,config,rng);
  r=finishHistory(model,p,air,field,config,{},rng);
  errors+=r.action!=HistoryAction::replaced||r.child_count!=2||rng.used!=3;
  for(int j=0;j<r.child_count;++j) {
    errors+=r.children[j].pdg!=22||r.children[j].clock.initialized||r.children[j].photon_clock.initialized;
    errors+=r.children[j].energy_MeV!=model.thresholds.mass_MeV||r.children[j].position_cm.z!=state.position_cm.z;
  }
  errors+=::fabs(r.children[0].direction.x+r.children[1].direction.x)>1.e-15;
  // PHOTON energy cut consumes no interaction uniform, even with a bad table energy.
  state.pdg=22;state.energy_MeV=.1;rng.used=0;p=planHistory(model,state,air,field,config,rng);
  r=finishHistory(model,p,air,field,config,{},rng);
  errors+=r.action!=HistoryAction::replaced||r.vertex_deposit_MeV!=.1||rng.used!=0;
  // Explicit nuclear handoff, never a silent retry as an EM process.
  state.pdg=11;state.energy_MeV=1.e5;
  auto q=electronQuery(model.tables,state.energy_MeV,model.thresholds.mass_MeV,-1);
  state.clock={0.,q.rate_per_cm,true};rng={0.};
  p=planHistory(model,state,air,field,config,rng);errors+=p.action!=PlanAction::collision||rng.used!=0;
  r=finishHistory(model,p,air,field,config,{},rng);
  errors+=r.action!=HistoryAction::needs_host_channel||r.channel!=Channel::electronuclear||r.child_count!=0||rng.used!=2;
  rng.used=0;auto blocked=planHistory(model,r.particle,air,field,config,rng);
  errors+=blocked.action!=PlanAction::error||rng.used!=0;
  // Fictitious-sigma rejection ends this collision attempt, no branch draw.
  state.clock.sampled_rate_per_cm=q.rate_per_cm*1000.;rng={.9};
  p=planHistory(model,state,air,field,config,rng);r=finishHistory(model,p,air,field,config,{},rng);
  errors+=r.action!=HistoryAction::active||r.particle.clock.initialized||rng.used!=1||r.has_track;
  // Replanning this rejected history obtains one new path uniform.
  rng={.5};p=planHistory(model,r.particle,air,field,config,rng);
  errors+=p.action!=PlanAction::transport||rng.used!=1||!p.particle.clock.initialized;
  // Escape must not be reported as local deposition or a collision.
  r=finishHistory(model,p,air,field,config,{p.proposed_cm*.25,BoundaryKind::escape},rng);
  errors+=r.action!=HistoryAction::escaped||!r.has_track||r.child_count!=0||r.vertex_deposit_MeV!=0.;
  // Invalid inputs fail before drawing randoms.
  state.energy_MeV=model.tables.medium.ue*2.;state.clock={};rng.used=0;
  p=planHistory(model,state,air,field,config,rng);errors+=p.action!=PlanAction::error||p.error!=HistoryError::table_range||rng.used!=0;
  return errors;
}
struct ContractCheck {ModelView model;KOKKOS_INLINE_FUNCTION void operator()(int,int& n)const{n+=contracts(model);}};
void run(std::string const& path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto t=readAirTables(path);ChannelThresholds thresholds{.51099895,140.,140.,422.};
  auto host=modelView(t,thresholds);DeviceModel<Exec> device(t,thresholds);int device_errors=0,host_errors=contracts(host);
  Kokkos::parallel_reduce("HISTORY_CONTRACTS",Kokkos::RangePolicy<Exec>(0,1),ContractCheck{device.view()},device_errors);
  std::cout<<"execution_space="<<Exec::name()<<"\nhost_contract_errors="<<host_errors<<"\ndevice_contract_errors="<<device_errors<<'\n';
  check(!host_errors&&!device_errors,"Native history contracts failed");
  std::vector<Case> all;std::mt19937_64 seeds(2026100307);
  for(int pdg:{11,-11,22})for(double E:{.95,1.5,10.,100.,1.e3})for(double stepfc:{1.,.0625})for(int i=0;i<32;++i) {
    TrackState s;s.pdg=pdg;s.energy_MeV=E;s.direction={.3,.4,std::sqrt(.75)};s.position_cm={10.,-20.,-1.e6};
    all.push_back({s,stepfc,std::int64_t(seeds()%2147483646)+1});
  }
  Kokkos::View<Case*,typename Exec::memory_space> input("history_input",all.size());
  Kokkos::View<Answer*,typename Exec::memory_space> output("history_output",all.size());
  auto h=Kokkos::create_mirror_view(input);for(std::size_t i=0;i<all.size();++i)h(i)=all[i];Kokkos::deep_copy(input,h);
  Kokkos::parallel_for("NATIVE_C7_HISTORY",Kokkos::RangePolicy<Exec>(0,all.size()),Batch<Exec>{device.view(),input,output});
  Exec().fence();auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  int steps=0,boundaries=0,requests=0,replaced=0,max_steps=0;double energy_error=0.,direction_error=0.,balance=0.,position_error=0.;
  for(std::size_t i=0;i<all.size();++i) {
    auto cpu=evaluate(host,all[i]),gpu=result(i);auto a=cpu.outcome,b=gpu.outcome;
    if(a.action==HistoryAction::error||b.action==HistoryAction::error) {
      std::cerr<<"case="<<i<<" host_error="<<int(a.error)<<" device_error="<<int(b.error)<<'\n';throw std::runtime_error("History failed");
    }
    check(a.action==HistoryAction::replaced||a.action==HistoryAction::needs_host_channel,"History did not reach a reaction/cut");
    check(a.action==b.action&&a.channel==b.channel&&a.child_count==b.child_count&&cpu.used==gpu.used&&cpu.steps==gpu.steps&&cpu.attempts==gpu.attempts,
          "History RNG or control-flow mismatch");
    energy_error=std::max(energy_error,std::abs(cpu.deposit-gpu.deposit)/all[i].state.energy_MeV);
    balance=std::max({balance,cpu.balance_error/all[i].state.energy_MeV,gpu.balance_error/all[i].state.energy_MeV});
    for(int j=0;j<a.child_count;++j) {
      auto x=a.children[j],y=b.children[j];check(x.pdg==y.pdg&&!y.clock.initialized&&!y.photon_clock.initialized,"Invalid child state");
      energy_error=std::max(energy_error,std::abs(x.energy_MeV-y.energy_MeV)/x.energy_MeV);
      direction_error=std::max({direction_error,std::abs(x.direction.x-y.direction.x),std::abs(x.direction.y-y.direction.y),std::abs(x.direction.z-y.direction.z)});
      position_error=std::max({position_error,std::abs(x.position_cm.x-y.position_cm.x),std::abs(x.position_cm.y-y.position_cm.y),std::abs(x.position_cm.z-y.position_cm.z)});
    }
    steps+=cpu.steps;boundaries+=cpu.boundaries;max_steps=std::max(max_steps,cpu.steps);
    requests+=a.action==HistoryAction::needs_host_channel;replaced+=a.action==HistoryAction::replaced;
  }
  std::cout<<std::setprecision(17)<<"synthetic_histories="<<all.size()<<"\ntransported_segments="<<steps
    <<"\nmax_segments_per_history="<<max_steps<<"\nboundary_returns="<<boundaries<<"\nreplaced="<<replaced<<"\nhost_channel_requests="<<requests
    <<"\nenergy_relative_host_device="<<energy_error<<"\ndirection_absolute_host_device="<<direction_error
    <<"\nposition_absolute_host_device_cm="<<position_error<<"\nmax_energy_balance_relative="<<balance
    <<"\nscope=composed local particle histories; source-checked kernels, not an independent full-C7 shower comparison\n";
  check(steps>0&&boundaries>0&&replaced>0,"Missing history coverage");
  check(energy_error<1.e-9&&direction_error<1.e-8&&balance<1.e-11&&position_error<1.e-4,"History numerical inconsistency");
}
}
int main(int argc,char** argv) {
  try {
    if(argc!=2){std::cerr<<"Usage: test_egs4_history EGSDAT\n";return 2;}
    Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
