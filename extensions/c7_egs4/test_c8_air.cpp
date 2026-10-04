#include "AirTestFixture.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace {
using namespace c7_egs4;
using namespace c7_egs4::c8_adapter;
void check(bool b,char const* message){if(!b)throw std::runtime_error(message);}
using air_test::environment;
using air_test::input;
KOKKOS_INLINE_FUNCTION int contracts(ModelView model,AirEnvironment const& e) {
  int errors=0;auto p=input(e,11,10.,10000.,0);AirRecord r;
  errors+=!importAirRecord(p,e,17,2,r);auto back=p;
  errors+=!exportAirParticle(r.track,e,back);
  for(int j=0;j<3;++j)errors+=::fabs(back.direction[j]-p.direction[j])>1.e-12||::fabs(back.position_m[j]-p.position_m[j])>1.e-8;
  errors+=r.region!=3&&r.region!=4; // lateral displacement raises radial height slightly
  errors+=airRegion(e,-e.geometry.bound_cm[0])!=2;
  auto local_b=toLocal(e.observer_frame,{e.source.magnetic_field_T[0],e.source.magnetic_field_T[1],e.source.magnetic_field_T[2]});
  errors+=::fabs(local_b.y)>1.e-15||local_b.x<0.;
  errors+=e.regions[0].inverse_scale_height_cm!=1./772170.16;
  errors+=e.geometry.bound_cm[0]!=11280000.||e.geometry.bound_cm[1]!=4000000.;
  auto bad=p;bad.medium_id=4;errors+=importAirRecord(bad,e,17,2,r);
  // No re-use of PROPOSAL fallback: pending EGS4 channels remain explicit.
  importAirRecord(p,e,17,2,r);r.track.particle.pending_channel=Channel::electronuclear;
  auto h=advanceAirRecord(model,r,e,{1.,.5,.0625});errors+=h.action!=AirAction::error;
  return errors;
}
struct Case {C8Particle p;double stepfc;};
struct Answer {AirRecord record;AirOutcome outcome;int steps{},layers{},attempts{};double deposited{},closure{};};
KOKKOS_INLINE_FUNCTION Answer evaluate(ModelView model,AirEnvironment const& e,Case q) {
  Answer a;if(!importAirRecord(q.p,e,20261003,1,a.record))return a;
  auto original=a.record.track.particle;
  for(int i=0;i<5000;++i) {
    int region=a.record.region;
    a.outcome=advanceAirRecord(model,a.record,e,{1.,.5,q.stepfc});++a.attempts;
    auto h=a.outcome.history;
    a.steps+=h.has_track;a.layers+=region!=a.record.region;
    a.deposited+=h.continuous_deposit_MeV+h.vertex_deposit_MeV;
    if(a.outcome.action!=AirAction::active)break;
  }
  auto h=a.outcome.history;
  double accounted=a.deposited+a.outcome.discarded_energy_MeV;
  for(int j=0;j<h.child_count;++j)accounted+=terminationEnergy(h.children[j],model.thresholds.mass_MeV);
  if(a.outcome.action==AirAction::observed||a.outcome.action==AirAction::needs_host_channel||a.outcome.action==AirAction::active)
    accounted+=terminationEnergy(a.record.track.particle,model.thresholds.mass_MeV);
  a.closure=::fabs(accounted-terminationEnergy(original,model.thresholds.mass_MeV));return a;
}
template<class Exec>struct Batch {
  ModelView model;AirEnvironment env;Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(model,env,input(i));}
};
struct Contracts {ModelView model;AirEnvironment env;KOKKOS_INLINE_FUNCTION void operator()(int,int& errors)const{errors+=contracts(model,env);}};
void run(char const* path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto tables=readAirTables(path);ChannelThresholds thresholds{.51099895,140.,140.,422.};
  auto model=modelView(tables,thresholds);DeviceModel<Exec> device(tables,thresholds);
  auto env=makeAirEnvironment(environment(),6.371315e6,10.,AirConvention::c7_egs4_four_exponentials);
  int host_errors=contracts(model,env),device_errors=0;
  Kokkos::parallel_reduce("C8_air_contracts",Kokkos::RangePolicy<Exec>(0,1),Contracts{device.view(),env},device_errors);
  std::cout<<"execution_space="<<Exec::name()<<"\nhost_contract_errors="<<host_errors<<"\ndevice_contract_errors="<<device_errors<<'\n';
  check(!host_errors&&!device_errors,"C8 air contract failed");
  auto invalid=environment();invalid.atmosphere_layers[2].medium_id=4;bool rejected=false;
  try{makeAirEnvironment(invalid,6.371315e6,10.,AirConvention::c7_egs4_four_exponentials);}catch(std::invalid_argument const&){rejected=true;}
  check(rejected,"Multi-material atmosphere silently accepted");
  std::vector<Case> cases;
  for(int pdg:{11,-11,22})for(double energy:{1.00001,10.,100.})for(double h:{1100.001,3999.9,10000.1,100000.})
    for(double stepfc:{1.,.0625})for(int k=0;k<4;++k)cases.push_back({input(env,pdg,energy,h,cases.size()),stepfc});
  Kokkos::View<Case*,typename Exec::memory_space> inputs("air_inputs",cases.size());
  Kokkos::View<Answer*,typename Exec::memory_space> outputs("air_outputs",cases.size());
  auto mirror=Kokkos::create_mirror_view(inputs);for(std::size_t i=0;i<cases.size();++i)mirror(i)=cases[i];Kokkos::deep_copy(inputs,mirror);
  Kokkos::parallel_for("C8_air_histories",Kokkos::RangePolicy<Exec>(0,cases.size()),Batch<Exec>{device.view(),env,inputs,outputs});
  Exec().fence();auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},outputs);
  int steps=0,layers=0,observed=0,replaced=0,discarded=0;double energy=0.,position=0.,time=0.,closure=0.;
  for(std::size_t i=0;i<cases.size();++i) {
    auto a=evaluate(model,env,cases[i]),b=result(i);
    if(a.outcome.action==AirAction::error||b.outcome.action==AirAction::error||a.outcome.action==AirAction::active||b.outcome.action==AirAction::active) {
      std::cerr<<"case="<<i<<" action="<<int(a.outcome.action)<<" error="<<int(a.outcome.history.error)<<" region="<<a.record.region
        <<" steps="<<a.steps<<" height="<<-a.record.track.particle.position_cm.z<<" energy="<<a.record.track.particle.energy_MeV<<'\n';
      throw std::runtime_error("C8 air history not completed");
    }
    check(a.outcome.action==b.outcome.action&&a.steps==b.steps&&a.record.random.key.draw_id==b.record.random.key.draw_id,"Air control/RNG differs");
    auto x=a.record.metadata,y=b.record.metadata;
    energy=std::max(energy,std::abs(x.energy_GeV-y.energy_GeV)/cases[i].p.energy_GeV);
    for(int j=0;j<3;++j)position=std::max(position,std::abs(x.position_m[j]-y.position_m[j]));
    time=std::max(time,std::abs(x.time_s-y.time_s));
    closure=std::max({closure,a.closure/(cases[i].p.energy_GeV*1000.),b.closure/(cases[i].p.energy_GeV*1000.)});
    steps+=a.steps;layers+=a.layers;observed+=a.outcome.action==AirAction::observed;
    replaced+=a.outcome.action==AirAction::replaced;discarded+=a.outcome.action==AirAction::discarded;
  }
  std::cout<<std::setprecision(17)<<"histories="<<cases.size()<<"\nsteps="<<steps<<"\nlayer_changes="<<layers<<"\nobserved="<<observed
    <<"\nreplaced="<<replaced<<"\ndiscarded="<<discarded<<"\nenergy_relative_host_device="<<energy
    <<"\nposition_absolute_host_device_m="<<position<<"\ntime_absolute_host_device_s="<<time<<"\nenergy_closure_relative="<<closure
    <<"\nscope=C8 Cartesian carriers and five-layer snapshot into native C7-convention histories; not full hadronic shower\n";
  check(layers&&observed&&replaced,"Missing atmosphere path coverage");
  check(energy<1.e-7&&position<.001&&time<1.e-9&&closure<1.e-9,"Air history numerical mismatch");
}
}
int main(int argc,char** argv) {
  try{if(argc!=2)return 2;Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;}
  catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
