#include "Egs4C8Adapter.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace {
using namespace c7_egs4;
using namespace c7_egs4::c8_adapter;
void check(bool b,char const* text){if(!b)throw std::runtime_error(text);}
struct Cap {
  KOKKOS_INLINE_FUNCTION BoundaryLimit operator()(TrackState const&,double proposed)const{return {proposed*.4,BoundaryKind::observation};}
};
KOKKOS_INLINE_FUNCTION Frame frame() {
  // Example C8 north/west/up -> EGS north/east/down: two flips, det = +1.
  return {{10.,20.,0.},{1.,0.,0.},{0.,-1.,0.},{0.,0.,-1.}};
}
KOKKOS_INLINE_FUNCTION TransportSettings settings(){TransportSettings s;s.electron_cut_total_MeV=1.;s.photon_cut_MeV=.5;s.stepfc=.0625;return s;}
struct Answer{NativeRecord record;HistoryOutcome outcome;};
KOKKOS_INLINE_FUNCTION Answer evaluate(ModelView model,C8Particle input) {
  Answer a;
  if(!importRecord(input,frame(),731,29,a.record))return a;
  AirRegion air{.001225,1.25e-6,0.};MagneticField field{0.,0.,1.,.2};
  for(int i=0;i<3;++i) {
    a.outcome=advanceRecord(model,a.record,frame(),air,field,settings(),Cap{});
    if(a.outcome.action!=HistoryAction::active&&a.outcome.action!=HistoryAction::boundary)break;
  }
  return a;
}
KOKKOS_INLINE_FUNCTION int contracts(ModelView model,C8Particle p) {
  int errors=0;Frame f=frame();NativeRecord a;
  if(!importRecord(p,f,731,29,a))return 1;
  C8Particle roundtrip=p;errors+=!exportParticle(a.state,f,roundtrip);
  errors+=::fabs(roundtrip.energy_GeV-p.energy_GeV)>1.e-15;
  for(int j=0;j<3;++j)errors+=::fabs(roundtrip.position_m[j]-p.position_m[j])>1.e-10||::fabs(roundtrip.direction[j]-p.direction[j])>1.e-14;
  errors+=roundtrip.history_id!=p.history_id||roundtrip.weight!=p.weight||roundtrip.generation!=p.generation;
  f.z={0.,0.,1.};errors+=validFrame(f)||importRecord(p,f,731,29,a);
  f=frame();auto invalid=p;invalid.pid=13;errors+=importRecord(invalid,f,731,29,a);
  auto b=evaluate(model,p);errors+=b.outcome.action==HistoryAction::error;
  errors+=b.record.metadata.weight!=p.weight||b.record.metadata.history_id!=p.history_id;
  errors+=b.record.metadata.step_id!=p.step_id+3;
  // The native optical clock must survive across batch/geometry returns.
  errors+=p.pid==22?!b.record.state.photon_clock.initialized:!b.record.state.clock.initialized;
  auto random=b.record.random;
  for(int i=0;i<16;++i) {
    double u;auto expected=::corsika::gpu::em::uniformOpen01(random.key);
    errors+=!random.next(u)||u!=expected||u<=0.||u>=1.;
  }
  NativeRecord child;auto child_state=b.record.state;child_state.clock={.3,.4,true};child_state.photon_clock={.8,true};
  errors+=!childRecord(b.record,child_state,f,p.history_id+1,child);
  errors+=child.state.clock.initialized||child.state.photon_clock.initialized||child.state.pending_channel!=Channel::invalid;
  errors+=child.metadata.history_id!=p.history_id+1||child.metadata.parent_history_id!=p.history_id||child.metadata.generation!=p.generation+1;
  errors+=child.random.key.draw_id!=0||child.metadata.step_id!=0||child.metadata.weight!=p.weight;
  errors+=childRecord(b.record,child_state,f,p.history_id,child);
  return errors;
}
template<class Exec>struct Batch {
  ModelView model;Kokkos::View<C8Particle*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;Kokkos::View<int*,typename Exec::memory_space> errors;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(model,input(i));errors(i)=contracts(model,input(i));}
};
void run(std::string const& path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto tables=readAirTables(path);ChannelThresholds t{.51099895,140.,140.,422.};
  auto host=modelView(tables,t);DeviceModel<Exec> device(tables,t);
  std::vector<C8Particle> all;
  for(int pdg:{11,-11,22})for(std::uint64_t i=1;i<=32;++i) {
    C8Particle p;p.pid=pdg;p.energy_GeV=.1;p.position_m[0]=17.;p.position_m[1]=-20.;p.position_m[2]=10000.;
    p.direction[0]=.3;p.direction[1]=-.4;p.direction[2]=-std::sqrt(.75);p.time_s=1.e-6;
    p.weight=1.5;p.history_id=i+std::uint64_t(pdg+11)*100;p.step_id=7;p.generation=4;all.push_back(p);
  }
  Kokkos::View<C8Particle*,typename Exec::memory_space> input("C8_input",all.size());
  Kokkos::View<Answer*,typename Exec::memory_space> output("C8_output",all.size());
  Kokkos::View<int*,typename Exec::memory_space> errors("C8_adapter_errors",all.size());
  auto h=Kokkos::create_mirror_view(input);for(std::size_t i=0;i<all.size();++i)h(i)=all[i];Kokkos::deep_copy(input,h);
  Kokkos::parallel_for("C8_NATIVE_EGS4_ADAPTER",Kokkos::RangePolicy<Exec>(0,all.size()),Batch<Exec>{device.view(),input,output,errors});
  Exec().fence();auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  auto error=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},errors);
  double max_energy=0.,max_pos=0.;int host_errors=0,device_errors=0;
  for(std::size_t i=0;i<all.size();++i) {
    auto cpu=evaluate(host,all[i]),gpu=result(i);host_errors+=contracts(host,all[i]);device_errors+=error(i);
    check(cpu.outcome.action==gpu.outcome.action&&cpu.record.random.key.draw_id==gpu.record.random.key.draw_id,"Adapter control/random mismatch");
    max_energy=std::max(max_energy,std::abs(cpu.record.metadata.energy_GeV-gpu.record.metadata.energy_GeV));
    for(int j=0;j<3;++j)max_pos=std::max(max_pos,std::abs(cpu.record.metadata.position_m[j]-gpu.record.metadata.position_m[j]));
    // Reorder/batch membership does not enter the random key.
    auto again=evaluate(host,all[i]);check(cpu.record.random.key.draw_id==again.record.random.key.draw_id&&
      cpu.record.state.energy_MeV==again.record.state.energy_MeV,"Adapter not reproducible");
  }
  std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<"\nC8_carriers="<<all.size()
    <<"\nhost_contract_errors="<<host_errors<<"\ndevice_contract_errors="<<device_errors
    <<"\nmax_energy_host_device_GeV="<<max_energy<<"\nmax_position_host_device_m="<<max_pos
    <<"\nscope=C8 carrier/Philox interface plus native history steps; not installed in production router\n";
  check(!host_errors&&!device_errors&&max_energy<1.e-10&&max_pos<1.e-6,"C8 adapter mismatch");
}
}
int main(int argc,char** argv) {
  try {
    if(argc!=2){std::cerr<<"Usage: test_egs4_c8_adapter EGSDAT\n";return 2;}
    Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
