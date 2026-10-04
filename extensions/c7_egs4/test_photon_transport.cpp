#include "Egs4PhotonTransport.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
using namespace c7_egs4;
void check(bool b,char const* message){if(!b)throw std::runtime_error(message);}
struct Case{TrackState state;AirRegion region;double cap;};
struct Answer{PhotonPlan plan;PhotonSegment segment;};
KOKKOS_INLINE_FUNCTION TransportSettings settings() {
  TransportSettings s;s.photon_cut_MeV=.5;return s;
}
KOKKOS_INLINE_FUNCTION Answer evaluate(ModelView model,Case q) {
  auto s=settings();auto p=preparePhotonSegment(model,q.state,q.region,s);
  auto r=finishPhotonSegment(q.state,q.region,s,p,{p.proposed_cm*q.cap,BoundaryKind::none});
  return {p,r};
}
std::vector<Case> cases() {
  std::vector<Case> out;Direction dirs[]={{0.,0.,1.},{.3,.4,std::sqrt(.75)},{.6,.8,0.},{.3,.4,-std::sqrt(.75)}};
  for(double energy:{.500001,.511,1.022,1.1,100.,1.e5,1.e12})for(auto d:dirs)
    for(double z:{-1.1e5,-1.e6})for(double inv_h:{0.,1.25e-6})
      for(double mfp:{1.e-12,1.e-6,.1,10.,100.})for(double cap:{1.,.3,.001}) {
        TrackState s;s.pdg=22;s.energy_MeV=energy;s.position_cm={10.,-20.,z};s.direction=d;
        s.photon_clock={mfp,true};out.push_back({s,{.001225,inv_h,0.},cap});
      }
  return out;
}
void writeCases(std::string const& path) {
  auto all=cases();std::ofstream f(path);check(bool(f),"Cannot write photon transport cases");
  f<<std::setprecision(17)<<all.size()<<'\n';
  for(auto q:all) {
    auto s=q.state;f<<s.energy_MeV<<' '<<s.position_cm.x<<' '<<s.position_cm.y<<' '<<s.position_cm.z<<' '
      <<s.direction.x<<' '<<s.direction.y<<' '<<s.direction.z<<' '
      <<q.region.reference_density_g_cm3<<' '<<q.region.inverse_scale_height_cm<<' '
      <<s.photon_clock.remaining_mfp<<' '<<q.cap<<'\n';
  }
  check(bool(f),"Incomplete photon transport cases");
}
std::array<double,15> values(Answer a) {
  auto p=a.plan;auto r=a.segment;auto s=r.particle;
  return {p.coefficients.mean_free_path_cm,p.mean_free_path_cm,p.proposed_cm,r.geometric_step_cm,
    r.optical_path_cm,s.position_cm.x,s.position_cm.y,s.position_cm.z,s.time_s,s.photon_clock.remaining_mfp,
    s.energy_MeV,p.coefficients.branches[0],p.coefficients.branches[1],p.coefficients.branches[2],p.coefficients.branches[3]};
}
template<class Exec>struct Batch {
  ModelView model;
  Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(model,input(i));}
};
struct Uniform {double value;int used{};KOKKOS_INLINE_FUNCTION bool next(double& x){x=value;++used;return true;}};
KOKKOS_INLINE_FUNCTION int contracts(ModelView model) {
  int errors=0;TrackState state;state.pdg=22;state.energy_MeV=10.;state.position_cm={0.,0.,-1.e6};
  state.direction={.6,0.,.8};auto s=settings();AirRegion region{.001225,1.25e-6,0.};
  auto q=photonQuery(model.tables,state.energy_MeV);Uniform random{.5};
  errors+=startPhotonClock(q,random,state.photon_clock)!=ClockStatus::transporting||random.used!=1;
  auto p=preparePhotonSegment(model,state,region,s);
  if(p.status!=Status::success)return errors+1;
  for(auto kind:{BoundaryKind::density_layer,BoundaryKind::material_change,BoundaryKind::observation,BoundaryKind::escape}) {
    auto r=finishPhotonSegment(state,region,s,p,{p.proposed_cm*.25,kind});
    errors+=r.status!=TransportStatus::boundary||!r.particle.photon_clock.initialized||r.boundary!=kind;
    errors+=r.particle.photon_clock.remaining_mfp>=state.photon_clock.remaining_mfp;
    errors+=r.particle.energy_MeV!=state.energy_MeV||r.particle.direction.x!=state.direction.x;
  }
  state.energy_MeV=.5;errors+=preparePhotonSegment(model,state,region,s).status!=Status::invalid_input;
  state.energy_MeV=.6;state.photon_clock.initialized=false;
  errors+=preparePhotonSegment(model,state,region,s).status!=Status::invalid_input;
  Uniform zero{0.};errors+=startPhotonClock(q,zero,state.photon_clock)!=ClockStatus::random_failure||state.photon_clock.initialized;
  q.mean_free_path_cm=-1.;random.used=0;
  errors+=startPhotonClock(q,random,state.photon_clock)!=ClockStatus::invalid||random.used!=0;
  return errors;
}
struct ContractCheck{ModelView model;KOKKOS_INLINE_FUNCTION void operator()(int,int& n)const{n+=contracts(model);}};
void run(std::string const& reference,std::string const& path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto tables=readAirTables(path);
  ChannelThresholds t{.51099895,140.,140.,422.};auto host=modelView(tables,t);DeviceModel<Exec> device(tables,t);
  int device_errors=0;Kokkos::parallel_reduce("PHOTON_CONTRACTS",Kokkos::RangePolicy<Exec>(0,1),
    ContractCheck{device.view()},device_errors);
  int host_errors=contracts(host);
  std::cout<<"execution_space="<<Exec::name()<<"\nhost_contract_errors="<<host_errors<<"\ndevice_contract_errors="<<device_errors<<'\n';
  check(!host_errors&&!device_errors,"Photon transport contract violation");
  if(reference=="--contracts")return;
  auto all=cases();Kokkos::View<Case*,typename Exec::memory_space> input("photon_input",all.size());
  Kokkos::View<Answer*,typename Exec::memory_space> output("photon_output",all.size());
  auto h=Kokkos::create_mirror_view(input);for(std::size_t i=0;i<all.size();++i)h(i)=all[i];Kokkos::deep_copy(input,h);
  Kokkos::parallel_for("NATIVE_C7_PHOTON_TRANSPORT",Kokkos::RangePolicy<Exec>(0,all.size()),Batch<Exec>{device.view(),input,output});
  Exec().fence();auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  std::ifstream f(reference);check(bool(f),"Cannot open original PHOTON reference");
  std::array<double,15> host_diff{},device_diff{};int bad=0,status_mismatch=0,advanced=0,due=0,vacuum_bound=0;
  for(std::size_t i=0;i<all.size();++i) {
    std::array<double,15> ref;for(auto& v:ref)f>>v;check(bool(f),"Truncated photon reference");
    auto a=evaluate(host,all[i]),b=result(i);auto av=values(a),bv=values(b);
    check(a.plan.status==Status::success&&b.plan.status==Status::success&&
      a.segment.status!=TransportStatus::invalid&&b.segment.status!=TransportStatus::invalid,"Photon transport failed");
    status_mismatch+=a.segment.status!=b.segment.status;
    advanced+=a.segment.status==TransportStatus::advanced;due+=a.segment.status==TransportStatus::collision_due;
    vacuum_bound+=a.plan.proposed_cm==settings().vacuum_distance_cm;
    for(std::size_t j=0;j<ref.size();++j) {
      double scale=j==9?all[i].state.photon_clock.remaining_mfp:std::max(1.e-20,std::abs(ref[j]));
      if(j>=11)scale=1.; // branching coefficients can be exactly zero
      host_diff[j]=std::max(host_diff[j],std::abs(av[j]-ref[j])/scale);
      device_diff[j]=std::max(device_diff[j],std::abs(bv[j]-ref[j])/scale);
      if(!std::isfinite(av[j])||!std::isfinite(bv[j])||std::abs(av[j]-ref[j])>1.e-9*scale||std::abs(bv[j]-ref[j])>1.e-9*scale) {
        if(bad<8)std::cerr<<std::setprecision(17)<<"case="<<i<<" field="<<j<<" reference="<<ref[j]<<" host="<<av[j]<<" device="<<bv[j]<<'\n';++bad;
      }
    }
  }
  std::string extra;check(!(f>>extra),"Extra photon reference rows");
  std::cout<<std::setprecision(17)<<"photon_step_inputs="<<all.size()<<"\nadvanced="<<advanced<<"\ncollision_due="<<due
    <<"\nescape_distance_proposals="<<vacuum_bound<<"\nstatus_mismatches="<<status_mismatch
    <<"\nfields=table_mfp regional_mfp proposal step optical_path x y z time remaining_optical energy branch1 branch2 branch3 branch4\n"
    <<"normalization=relative except branching absolute and remaining optical divided by initial\nhost_difference=";
  for(auto x:host_diff)std::cout<<x<<' ';std::cout<<"\ndevice_difference=";for(auto x:device_diff)std::cout<<x<<' ';
  std::cout<<"\nscope=original C7 photon table and normal-air flat-frame local segment; not complete shower\n";
  check(advanced&&due&&vacuum_bound,"Missing photon branch coverage");
  check(!bad&&!status_mismatch,"Photon transport comparison failed");
}
}
int main(int argc,char** argv) {
  try {
    if(argc==3&&std::string(argv[1])=="--make-cases"){writeCases(argv[2]);return 0;}
    if(argc!=3){std::cerr<<"Usage: test_egs4_photon_transport REFERENCE|--contracts EGSDAT or --make-cases INPUT\n";return 2;}
    Kokkos::ScopeGuard guard(argc,argv);run(argv[1],argv[2]);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
