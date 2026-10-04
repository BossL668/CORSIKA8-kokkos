#include "Egs4Observation.hpp"
#include "Egs4Geometry.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>

namespace {
using namespace c7_egs4;
void check(bool b,char const* msg){if(!b)throw std::runtime_error(msg);}
struct Stream {
  std::int64_t seed;int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& u) {
    if(used>=100000)return false;
    seed=(seed*48271)%2147483647;u=double(seed)/2147483647.;++used;return true;
  }
};
struct Case {int pdg;double energy,height,stepfc,bnorm;std::int64_t seed;};
struct Answer {DetectorHistoryOutcome result;int used{},attempted{};double balance{},time_error{},loss_error{};};
KOKKOS_INLINE_FUNCTION CurvedConfig config(){return {true,6.371315e8,1.1e5,-1.};}
KOKKOS_INLINE_FUNCTION AirRegion air(){return {.001225,1.25e-6,10.};}
KOKKOS_INLINE_FUNCTION CurvedTrack track(Case q) {
  auto c=config();CurvedTrack s;auto& p=s.particle;
  p.pdg=q.pdg;p.energy_MeV=q.energy;p.time_s=1.e-6;
  p.position_cm={1.e4,-5.e3,0.};p.direction={.48,.64,.6};
  double lateral=::sqrt(1.25e8);s.wa=::cos(lateral/c.earth_radius_cm);
  p.position_cm.z=c.earth_radius_cm-(c.earth_radius_cm+c.observation_height_cm+q.height)/s.wa;
  double scale=(c.earth_radius_cm-p.position_cm.z)*::sin(lateral/c.earth_radius_cm)/lateral;
  s.observer_cm={scale*p.position_cm.x,scale*p.position_cm.y,-c.observation_height_cm-q.height};
  s.wap=q.height/::sqrt(q.height*q.height+s.observer_cm.x*s.observer_cm.x+s.observer_cm.y*s.observer_cm.y);
  return s;
}
KOKKOS_INLINE_FUNCTION Answer evaluate(ModelView model,Case q) {
  auto s=track(q);Stream rng{q.seed};Answer a;
  auto cfg=TransportSettings{1.,.5,q.stepfc};MagneticField field{q.bnorm,.6,.8,.2};
  auto p=planHistory(model,s.particle,air(),field,cfg,rng);
  if(p.action!=PlanAction::transport)return a;
  GeometryConfig geo{true,true,6.371315e8,{1.128e7,4e6,1e6,4e5,0.,-1.},1.1e5,6.e5,2.e6,-1.4e6/1030.};
  auto g=howFar(geo,{s.particle.position_cm,s.particle.direction,5,s.wa,900.,p.proposed_cm,0.});
  if(!g.valid||g.discard!=-1)return a;
  a.attempted=1;
  a.result=finishDetectorHistory(model,p,s,air(),field,cfg,config(),geo.maximum_horizontal_cm,g.step_cm,rng);
  a.used=rng.used;
  auto r=a.result;auto h=r.advanced.history;
  if(r.action==DetectorAction::error)return a;
  double end=h.continuous_deposit_MeV+h.vertex_deposit_MeV+r.discarded_energy_MeV;
  if(r.action==DetectorAction::observed)end+=terminationEnergy(h.particle,model.thresholds.mass_MeV);
  for(int i=0;i<h.child_count;++i)end+=terminationEnergy(h.children[i],model.thresholds.mass_MeV);
  a.balance=::fabs(end-terminationEnergy(s.particle,model.thresholds.mass_MeV));
  if(h.has_track) {
    auto d=r.advanced.particle.observer_cm;auto start=r.observer_start_cm;
    double chord=::sqrt((d.x-start.x)*(d.x-start.x)+(d.y-start.y)*(d.y-start.y)+(d.z-start.z)*(d.z-start.z));
    double beta=q.pdg==22 ? 1. : ::sqrt((1.-model.thresholds.mass_MeV/q.energy)*(1.+model.thresholds.mass_MeV/q.energy));
    double expected=h.time_path_cm/cfg.speed_cm_s/beta*chord/h.geometric_step_cm;
    a.time_error=::fabs((h.particle.time_s-s.particle.time_s)-expected);
    if(q.pdg!=22)a.loss_error=::fabs(h.continuous_deposit_MeV-p.electron.local.loss_MeV_per_cm*h.loss_path_cm);
  }
  return a;
}
KOKKOS_INLINE_FUNCTION int contracts(ModelView model) {
  int errors=0;auto cfg=TransportSettings{1.,.5,.0625};MagneticField field;
  auto s=track({-11,10.,.001,.0625,0.,37});s.particle.direction={.48,.64,-.6};
  Stream rng{37};auto p=planHistory(model,s.particle,air(),field,cfg,rng);int used=rng.used;
  auto r=finishDetectorHistory(model,p,s,air(),field,cfg,config(),2.e6,.01,rng);
  errors+=r.action!=DetectorAction::angular_discard||r.advanced.history.has_track||r.advanced.history.child_count!=0;
  errors+=r.discarded_energy_MeV!=s.particle.energy_MeV+model.thresholds.mass_MeV||rng.used!=used;
  errors+=r.advanced.history.vertex_deposit_MeV!=0.||r.advanced.history.continuous_deposit_MeV!=0.;
  // A source literal non-unit direction is exposed, never normalized silently.
  s=track({11,10.,.001,.0625,0.,37});s.particle.position_cm.x=1.e7;s.particle.direction={0.,0.,1.};
  p=planHistory(model,s.particle,air(),field,cfg,rng);used=rng.used;
  r=finishDetectorHistory(model,p,s,air(),field,cfg,config(),2.e6,.01,rng);
  errors+=r.preparation.transport_ready||r.action!=DetectorAction::error||rng.used!=used;
  // Boundary conversion must not resample the optical clock or recompute the
  // original density coefficients; ordinary stopped cuts keep photon children.
  s=track({-11,1.00001,.01,.0625,0.,37});rng={37};
  p=planHistory(model,s.particle,air(),field,cfg,rng);
  r=finishDetectorHistory(model,p,s,air(),field,cfg,config(),2.e6,.1,rng);
  errors+=r.action!=DetectorAction::stopped||!r.advanced.history.has_track||r.advanced.history.child_count!=2;
  for(int j=0;j<r.advanced.history.child_count;++j) {
    errors+=r.advanced.children[j].observer_cm.z!=r.advanced.particle.observer_cm.z;
    errors+=!r.advanced.children[j].detector_local;
  }
  return errors;
}
template<class Exec>struct Batch {
  ModelView model;Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(model,input(i));}
};
struct Contracts {ModelView model;KOKKOS_INLINE_FUNCTION void operator()(int,int& n)const{n+=contracts(model);}};
void run(char const* path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto tables=readAirTables(path);ChannelThresholds thresholds{.51099895,140.,140.,422.};
  auto model=modelView(tables,thresholds);DeviceModel<Exec> device(tables,thresholds);
  int host_errors=contracts(model),device_errors=0;
  Kokkos::parallel_reduce("detector_history_contracts",Kokkos::RangePolicy<Exec>(0,1),Contracts{device.view()},device_errors);
  std::cout<<"execution_space="<<Exec::name()<<"\nhost_contract_errors="<<host_errors<<"\ndevice_contract_errors="<<device_errors<<'\n';
  check(!host_errors&&!device_errors,"Detector history contract failed");
  std::vector<Case> cases;std::mt19937_64 seeds(2026100311);
  for(int pdg:{11,-11,22})for(double e:{1.00001,1.5,10.,100.})for(double height:{.001,.1,1.})
    for(double stepfc:{1.,.0625})for(double field:{0.,5.e-5*2.99792458})for(int k=0;k<8;++k)
      cases.push_back({pdg,e,height,stepfc,field,std::int64_t(seeds()%2147483646)+1});
  Kokkos::View<Case*,typename Exec::memory_space> input("detector_cases",cases.size());
  Kokkos::View<Answer*,typename Exec::memory_space> output("detector_results",cases.size());
  auto mirror=Kokkos::create_mirror_view(input);for(std::size_t i=0;i<cases.size();++i)mirror(i)=cases[i];Kokkos::deep_copy(input,mirror);
  Kokkos::parallel_for("detector_histories",Kokkos::RangePolicy<Exec>(0,cases.size()),Batch<Exec>{device.view(),input,output});
  Exec().fence();auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  int attempted=0,observed=0,stopped=0,children=0;double position=0.,energy=0.,time=0.,balance=0.,time_contract=0.,loss_contract=0.;
  for(std::size_t i=0;i<cases.size();++i) {
    auto a=evaluate(model,cases[i]),b=result(i);check(a.attempted==b.attempted,"HOWFAR host/device differs");
    if(!a.attempted)continue;++attempted;
    auto r=a.result,t=b.result;auto h=r.advanced.history,j=t.advanced.history;
    if(r.action==DetectorAction::error||t.action==DetectorAction::error) {
      std::cerr<<"case="<<i<<" host_error="<<int(h.error)<<" device_error="<<int(j.error)<<'\n';
      throw std::runtime_error("Detector history failed");
    }
    check(r.action==t.action&&h.child_count==j.child_count&&a.used==b.used,"Detector history control/RNG differs");
    observed+=r.action==DetectorAction::observed;stopped+=r.action==DetectorAction::stopped;children+=h.child_count;
    auto x=r.advanced.particle.observer_cm,y=t.advanced.particle.observer_cm;
    position=std::max({position,std::abs(x.x-y.x),std::abs(x.y-y.y),std::abs(x.z-y.z)});
    energy=std::max(energy,std::abs(h.particle.energy_MeV-j.particle.energy_MeV)/cases[i].energy);
    time=std::max(time,std::abs(h.particle.time_s-j.particle.time_s));
    balance=std::max({balance,a.balance/cases[i].energy,b.balance/cases[i].energy});
    time_contract=std::max({time_contract,a.time_error,b.time_error});loss_contract=std::max({loss_contract,a.loss_error,b.loss_error});
  }
  std::cout<<std::setprecision(17)<<"inputs="<<cases.size()<<"\nreaching_detector="<<attempted<<"\nobserved="<<observed
    <<"\nstopped_before_observation="<<stopped<<"\nchildren="<<children<<"\nposition_absolute_host_device_cm="<<position
    <<"\nenergy_relative_host_device="<<energy<<"\ntime_absolute_host_device_s="<<time<<"\nenergy_closure_relative="<<balance
    <<"\nobserver_time_contract_absolute_s="<<time_contract<<"\noriginal_loss_coefficient_contract_MeV="<<loss_contract
    <<"\nscope=HOWFAR detector preflight and native history composition; not an independent full C7 shower comparison\n";
  check(observed&&stopped&&children,"Missing detector coverage");
  check(position<.001&&energy<1.e-8&&time<1.e-10&&balance<1.e-10&&time_contract<1.e-18&&loss_contract<1.e-12,"Detector numerical mismatch");
}
}
int main(int argc,char** argv) {
  try{if(argc!=2)return 2;Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;}
  catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
