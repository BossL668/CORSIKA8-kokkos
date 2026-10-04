#include "Egs4CurvedTransport.hpp"
#include "Egs4Geometry.hpp"
#include <algorithm>
#include <cstdint>
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
KOKKOS_INLINE_FUNCTION CurvedConfig config(){return {true,6.371315e8,1.1e5,-1.};}
KOKKOS_INLINE_FUNCTION TransportSettings settings(double stepfc) {
  TransportSettings s;s.electron_cut_total_MeV=1.;s.photon_cut_MeV=.5;s.stepfc=stepfc;return s;
}
KOKKOS_INLINE_FUNCTION CurvedTrack initial(int pdg,double energy,bool detector_local) {
  CurvedTrack r;r.detector_local=detector_local;
  auto& p=r.particle;p.pdg=pdg;p.energy_MeV=energy;p.time_s=1.e-6;
  p.position_cm={-2.e5,1.e5,-1.e6};p.direction={.3,.4,::sqrt(.75)};
  if(detector_local)r.observer_cm=p.position_cm;
  else {
    double radial=config().earth_radius_cm-p.position_cm.z;
    double lateral=::sqrt(p.position_cm.x*p.position_cm.x+p.position_cm.y*p.position_cm.y);
    r.wa=::cos(lateral/config().earth_radius_cm);
    double scale=radial*::sin(lateral/config().earth_radius_cm)/lateral;
    r.observer_cm={scale*p.position_cm.x,scale*p.position_cm.y,config().earth_radius_cm-radial*r.wa};
    double d=-r.observer_cm.z-config().observation_height_cm;
    r.wap=d/::sqrt(r.observer_cm.x*r.observer_cm.x+r.observer_cm.y*r.observer_cm.y+d*d);
  }
  return r;
}
struct Case {int pdg;double energy,stepfc;bool detector_local;std::int64_t seed;};
struct Answer {
  CurvedHistoryOutcome outcome;
  double deposit{},balance{};
  int attempts{},steps{},used{},boundaries{},layers{},inherited{};
};
KOKKOS_INLINE_FUNCTION double available(TrackState p,double m) {
  return p.energy_MeV+(p.pdg==11?-m:p.pdg==-11?m:0.);
}
// Coupled controller test. Layer boundaries use native HOWFAR. A continuous
// exponential toy atmosphere is deliberately used on both sides of each layer
// so this tests geometry/clock continuation, not fitted atmosphere coefficients.
KOKKOS_INLINE_FUNCTION Answer evaluate(ModelView model,Case q) {
  auto s=initial(q.pdg,q.energy,q.detector_local);auto start=s;
  AirRegion air{.001225,1.25e-6,0.};MagneticField field{5.e-5*2.99792458,.6,.8,.2};
  GeometryConfig geo{true,true,6.371315e8,{1.128e7,1e7,4e6,9.9e5,0.,-1.},1.1e5,1000.,1.e6,-900.};
  int region=4;Stream rng{q.seed};Answer a;
  for(int i=0;i<2048;++i) {
    auto p=planHistory(model,s.particle,air,field,settings(q.stepfc),rng);
    BoundaryLimit cap;GeometryResult g;
    if(p.action==PlanAction::transport) {
      double thickness=980.*::exp(s.particle.position_cm.z*air.inverse_scale_height_cm);
      g=howFar(geo,{s.particle.position_cm,s.particle.direction,region,s.wa,thickness,p.proposed_cm,0.});
      if(!g.valid||g.discard>0||g.step_cm<=0.)return a;
      cap={g.step_cm,g.discard==-2?BoundaryKind::escape:g.discard==-1?BoundaryKind::observation:
        g.region!=region?BoundaryKind::density_layer:BoundaryKind::none};
      // A periodic shorter observation of the queue (NOT a physical detector)
      // also checks state suspension when the same layer remains active.
      if(i%5==0&&p.proposed_cm*.4<cap.distance_cm) {
        cap={p.proposed_cm*.4,BoundaryKind::density_layer};g.region=region;
      }
    }
    auto r=finishCurvedHistory(model,p,s,air,field,settings(q.stepfc),cap,config(),rng);
    a.outcome=r;++a.attempts;a.steps+=r.history.has_track;
    a.deposit+=r.history.continuous_deposit_MeV+r.history.vertex_deposit_MeV;
    a.boundaries+=r.history.action==HistoryAction::boundary;
    if(r.history.has_track){a.layers+=g.region!=region;region=g.region;}
    for(int j=0;j<r.history.child_count;++j) {
      auto c=r.children[j];
      a.inherited+=c.observer_cm.x==r.particle.observer_cm.x&&c.observer_cm.y==r.particle.observer_cm.y&&
        c.observer_cm.z==r.particle.observer_cm.z&&c.wa==r.particle.wa&&c.wap==r.particle.wap&&
        c.detector_local==r.particle.detector_local&&!c.particle.clock.initialized&&!c.particle.photon_clock.initialized;
    }
    if(r.history.action!=HistoryAction::active&&r.history.action!=HistoryAction::boundary)break;
    if(cap.kind==BoundaryKind::observation||cap.kind==BoundaryKind::escape)break;
    s=r.particle;
  }
  a.used=rng.used;
  if(a.outcome.history.action==HistoryAction::replaced) {
    double end=a.deposit;
    for(int j=0;j<a.outcome.history.child_count;++j)end+=available(a.outcome.children[j].particle,model.thresholds.mass_MeV);
    a.balance=::fabs(end-available(start.particle,model.thresholds.mass_MeV));
  }
  return a;
}
struct ProbeFrame {
  Direction* before;
  KOKKOS_INLINE_FUNCTION FrameStepResult operator()(TrackState s,Point p,Direction d,
      double step,double path,double m,double speed)const {
    *before=d;auto r=FlatFrameStep{}(s,p,d,step,path,m,speed);
    r.direction={d.z,d.x,d.y}; // proper cyclic rotation, not the default frame
    return r;
  }
};
KOKKOS_INLINE_FUNCTION double directionError(Direction a,Direction b) {
  return larger(::fabs(a.x-b.x),larger(::fabs(a.y-b.y),::fabs(a.z-b.z)));
}
KOKKOS_INLINE_FUNCTION int contracts(ModelView model) {
  AirRegion air{.001225,1.25e-6,0.};MagneticField field{5.e-5*2.99792458,.6,.8,.2};
  auto s=initial(11,10.,false);auto cfg=settings(.0625);Stream rng{37};
  auto p=planHistory(model,s.particle,air,field,cfg,rng);int errors=p.action!=PlanAction::transport;
  auto start_rng=rng;Direction before;
  auto segment=finishElectronSegment(model,p.particle,air,field,cfg,p.electron,{p.proposed_cm*.5,BoundaryKind::none},rng,ProbeFrame{&before});
  auto scatter=sampleScattering(model.tables.medium,s.particle.energy_MeV,model.thresholds.mass_MeV,
    segment.loss_path_cm*air.reference_density_g_cm3/model.tables.medium.rho,start_rng);
  double sp{},cp{};errors+=!azimuth(start_rng,sp,cp);
  Direction reframed{before.z,before.x,before.y};
  auto expected=rotate(reframed,scatter.sine,scatter.cosine,sp,cp);
  errors+=directionError(segment.particle.direction,expected)>1.e-13||start_rng.used!=rng.used;
  errors+=directionError(segment.transport_end_direction,reframed)>1.e-15;
  // This deliberately nontrivial frame makes post-scattering rebasing detectably
  // different, proving that the hook is inside the step at the correct position.
  auto incorrectly_scattered=rotate(before,scatter.sine,scatter.cosine,sp,cp);
  errors+=directionError(expected,{incorrectly_scattered.z,incorrectly_scattered.x,incorrectly_scattered.y})<1.e-6;
  // Energy-cut children inherit observer state even when no transport occurs.
  s=initial(-11,.75,false);rng={7};p=planHistory(model,s.particle,air,field,cfg,rng);
  auto stop=finishCurvedHistory(model,p,s,air,field,cfg,{},config(),rng);
  errors+=stop.history.action!=HistoryAction::replaced||stop.history.child_count!=2||stop.frame_applied;
  for(int j=0;j<stop.history.child_count;++j)errors+=stop.children[j].observer_cm.z!=s.observer_cm.z||stop.children[j].wa!=s.wa;
  // Reject an unsupported early observation endpoint without pretending that it
  // produced a normally timed/deposited track. Its explicit frame status survives.
  s=initial(11,10.,false);s.particle.position_cm={0.,0.,-1.1e5};s.observer_cm=s.particle.position_cm;
  rng={37};p=planHistory(model,s.particle,air,field,cfg,rng);
  auto edge=finishCurvedHistory(model,p,s,air,field,cfg,{1.,BoundaryKind::none},config(),rng);
  errors+=edge.history.action!=HistoryAction::error||edge.frame.status!=CurvedStatus::observation||edge.history.has_track;
  errors+=edge.particle.particle.energy_MeV!=s.particle.energy_MeV||edge.particle.particle.time_s!=s.particle.time_s;
  errors+=edge.history.continuous_deposit_MeV!=0.||edge.history.child_count!=0;
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
  Kokkos::parallel_reduce("curved_history_contracts",Kokkos::RangePolicy<Exec>(0,1),Contracts{device.view()},device_errors);
  std::cout<<"execution_space="<<Exec::name()<<"\nhost_contract_errors="<<host_errors<<"\ndevice_contract_errors="<<device_errors<<'\n';
  check(!host_errors&&!device_errors,"Curved history contract failed");
  std::vector<Case> cases;std::mt19937_64 seeds(2026100310);
  for(int pdg:{11,-11,22})for(double energy:{.95,1.5,10.,100.,1000.})
    for(double stepfc:{1.,.0625})for(bool local:{false,true})for(int k=0;k<16;++k)
      cases.push_back({pdg,energy,stepfc,local,std::int64_t(seeds()%2147483646)+1});
  Kokkos::View<Case*,typename Exec::memory_space> input("curved_history_cases",cases.size());
  Kokkos::View<Answer*,typename Exec::memory_space> output("curved_history_answers",cases.size());
  auto h=Kokkos::create_mirror_view(input);for(std::size_t i=0;i<cases.size();++i)h(i)=cases[i];Kokkos::deep_copy(input,h);
  Kokkos::parallel_for("curved_histories",Kokkos::RangePolicy<Exec>(0,cases.size()),Batch<Exec>{device.view(),input,output});
  Exec().fence();auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  int steps=0,layers=0,boundaries=0,children=0;double energy=0.,direction=0.,position=0.,time=0.,balance=0.;
  for(std::size_t i=0;i<cases.size();++i) {
    auto a=evaluate(model,cases[i]),b=result(i);auto ha=a.outcome.history,hb=b.outcome.history;
    if(ha.action==HistoryAction::error||hb.action==HistoryAction::error) {
      std::cerr<<"case="<<i<<" host_error="<<int(ha.error)<<" device_error="<<int(hb.error)
        <<" frame_host="<<int(a.outcome.frame.status)<<" frame_device="<<int(b.outcome.frame.status)<<'\n';
      throw std::runtime_error("Curved history failed");
    }
    check(ha.action==HistoryAction::replaced||ha.action==HistoryAction::needs_host_channel,"Curved history has not completed");
    check(ha.action==hb.action&&ha.channel==hb.channel&&ha.child_count==hb.child_count&&a.used==b.used&&a.steps==b.steps&&a.layers==b.layers,
      "Curved history control/RNG mismatch");
    check(a.inherited==ha.child_count&&b.inherited==hb.child_count,"Child observer metadata was lost");
    auto x=a.outcome.particle,y=b.outcome.particle;
    energy=std::max({energy,std::abs(x.particle.energy_MeV-y.particle.energy_MeV)/cases[i].energy,std::abs(a.deposit-b.deposit)/cases[i].energy});
    direction=std::max(direction,directionError(x.particle.direction,y.particle.direction));
    position=std::max({position,std::abs(x.observer_cm.x-y.observer_cm.x),std::abs(x.observer_cm.y-y.observer_cm.y),std::abs(x.observer_cm.z-y.observer_cm.z)});
    time=std::max(time,std::abs(x.particle.time_s-y.particle.time_s));
    balance=std::max({balance,a.balance/cases[i].energy,b.balance/cases[i].energy});
    steps+=a.steps;layers+=a.layers;boundaries+=a.boundaries;children+=a.inherited;
  }
  std::cout<<std::setprecision(17)<<"histories="<<cases.size()<<"\nsteps="<<steps<<"\nlayer_changes="<<layers
    <<"\nboundary_returns="<<boundaries<<"\nchildren_with_observer_state="<<children
    <<"\nenergy_relative_host_device="<<energy<<"\ndirection_absolute_host_device="<<direction
    <<"\nobserver_position_absolute_host_device_cm="<<position<<"\ntime_absolute_host_device_s="<<time
    <<"\nenergy_balance_relative="<<balance<<"\nscope=normal curved history composition with HOWFAR caps; toy exponential atmosphere, no full C8 shower or speed claim\n";
  check(layers>0&&steps>0&&children>0,"Missing composed geometry coverage");
  check(energy<1.e-8&&direction<1.e-7&&position<.01&&time<1.e-10&&balance<1.e-10,"Curved history numerical mismatch");
}
}
int main(int argc,char** argv) {
  try{if(argc!=2)return 2;Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;}
  catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
