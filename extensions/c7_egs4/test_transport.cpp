#include "Egs4Transport.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

namespace {
using namespace c7_egs4;
void check(bool b,char const* msg){if(!b)throw std::runtime_error(msg);}
constexpr double mass=.51099895;
struct Stream {
  std::int64_t seed;int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& u) {
    if(used>=4096)return false;
    seed=(seed*48271)%2147483647;u=double(seed)/2147483647.;++used;return true;
  }
};
struct Exhausted {KOKKOS_INLINE_FUNCTION bool next(double&){return false;}};
struct Case {
  int pdg;double energy,initial_energy,demfp,cut,stepfc,cap;
  Point position;Direction direction;AirRegion region;MagneticField field;
  std::int64_t seed;
};
struct Answer {ElectronPlan plan;ElectronSegment segment;int used;};
KOKKOS_INLINE_FUNCTION TrackState track(ModelView model,Case q) {
  auto coefficients=electronQuery(model.tables,q.initial_energy,mass,q.pdg==11?-1:1);
  return {q.pdg,q.energy,0.,q.position,q.direction,{q.demfp,coefficients.rate_per_cm,true}};
}
KOKKOS_INLINE_FUNCTION TransportSettings settings(Case q) {
  TransportSettings s;s.electron_cut_total_MeV=q.cut;s.photon_cut_MeV=.5;s.stepfc=q.stepfc;return s;
}
KOKKOS_INLINE_FUNCTION Answer evaluate(ModelView model,Case q) {
  Stream rng{q.seed};auto state=track(model,q);auto s=settings(q);
  auto p=prepareElectronSegment(model,state,q.region,q.field,s);
  auto r=finishElectronSegment(model,state,q.region,q.field,s,p,{p.proposed_cm*q.cap,BoundaryKind::none},rng);
  return {p,r,rng.used};
}
std::array<double,18> values(Answer a) {
  auto p=a.plan;auto r=a.segment;auto s=r.particle;
  return {p.unclipped_cm,p.proposed_cm,r.geometric_step_cm,r.loss_path_cm,r.time_path_cm,
    r.deposit_MeV,s.energy_MeV,s.position_cm.x,s.position_cm.y,s.position_cm.z,
    s.direction.x,s.direction.y,s.direction.z,s.time_s,r.scattering_angle,r.bend_angle,
    s.clock.remaining_mfp,p.local.loss_MeV_per_cm};
}
std::vector<Case> cases() {
  std::vector<Case> out;std::mt19937_64 rng(2026100306);
  Direction dirs[]={{0.,0.,1.},{.3,.4,std::sqrt(.75)},{.6,.8,0.},{.3,.4,-std::sqrt(.75)}};
  for(int pdg:{11,-11})for(double E:{.922,1.5,2.999,3.001,100.,1.e5})
    for(int geometry=0;geometry<4;++geometry)for(double stepfc:{1.,.0625})
      for(double mfp:{1.e-6,.01,1.})for(double cap:{1.,.25,.001})for(int tape=0;tape<4;++tape) {
        Case q{};q.pdg=pdg;q.energy=E;q.initial_energy=E*(tape%2?1.1:1.);
        q.demfp=mfp;q.cut=double(.912f);q.stepfc=stepfc;q.cap=cap;
        q.position={10.,-20.,geometry?-1.e6:-1.1e5};q.direction=dirs[geometry];
        q.region={.001225,geometry==2?0.:1.25e-6,0.};
        q.field={geometry?5.e-5*2.99792458:0.,.6,.8,geometry==2?.001:.2};
        q.seed=std::int64_t(rng()%2147483646)+1;out.push_back(q);
      }
  return out;
}
void writeCases(std::string const& path) {
  auto all=cases();std::ofstream f(path);check(bool(f),"Cannot write transport inputs");
  f<<std::setprecision(17)<<all.size()<<'\n';
  for(auto q:all) {
    f<<(q.pdg==11?-1:1)<<' '<<q.energy<<' '<<mass<<' '<<q.initial_energy<<' '
      <<q.region.reference_density_g_cm3<<' '<<q.region.inverse_scale_height_cm<<' '<<q.region.sterncor<<' '
      <<q.cut<<' '<<q.stepfc<<' '<<q.demfp<<' '
      <<q.position.x<<' '<<q.position.y<<' '<<q.position.z<<' '
      <<q.direction.x<<' '<<q.direction.y<<' '<<q.direction.z<<' '
      <<q.field.bnorm_MeV_cm<<' '<<q.field.sinB<<' '<<q.field.cosB<<' '
      <<(q.field.bnorm_MeV_cm>0.?q.field.maximum_angle_rad/q.field.bnorm_MeV_cm:1.e99)<<' '
      <<q.cap<<' '<<q.seed<<'\n';
  }
  check(bool(f),"Incomplete transport inputs");
}
template<class Exec>struct Batch {
  ModelView model;
  Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(model,input(i));}
};
KOKKOS_INLINE_FUNCTION int contracts(ModelView model) {
  int errors=0;Case q{};q.pdg=11;q.energy=100.;q.initial_energy=110.;q.demfp=1.;
  q.cut=double(.912f);q.stepfc=.0625;q.cap=1.;q.position={0.,0.,-1.e6};
  q.direction={.6,0.,.8};q.region={.001225,1.25e-6,0.};q.field={0.,0.,1.,.2};q.seed=23;
  auto s=settings(q);auto state=track(model,q);auto p=prepareElectronSegment(model,state,q.region,q.field,s);
  if(p.status!=Status::success)return 1;
  auto current=electronQuery(model.tables,q.energy,mass,-1);
  errors+=p.local.rate_per_cm!=state.clock.sampled_rate_per_cm*q.region.reference_density_g_cm3/model.tables.medium.rho;
  errors+=current.rate_per_cm==state.clock.sampled_rate_per_cm; // grid deliberately differs
  for(auto kind:{BoundaryKind::density_layer,BoundaryKind::observation,BoundaryKind::material_change,BoundaryKind::escape}) {
    Stream rng{q.seed};auto r=finishElectronSegment(model,state,q.region,q.field,s,p,{p.proposed_cm*.25,kind},rng);
    errors+=r.status!=TransportStatus::boundary||r.boundary!=kind||!r.path_corrected;
    errors+=r.geometric_step_cm!=p.proposed_cm*.25;
    if(kind==BoundaryKind::material_change)errors+=r.particle.clock.initialized;
    else {
      errors+=!r.particle.clock.initialized;
      if(kind==BoundaryKind::escape)errors+=r.particle.clock.remaining_mfp!=state.clock.remaining_mfp;
      else errors+=r.particle.clock.remaining_mfp>=state.clock.remaining_mfp;
    }
  }
  Exhausted rng;
  auto failed=finishElectronSegment(model,state,q.region,q.field,s,p,{p.proposed_cm,BoundaryKind::none},rng);
  errors+=failed.status!=TransportStatus::random_failure||failed.particle.energy_MeV!=state.energy_MeV;
  auto invalid=state;invalid.direction={2.,0.,0.};
  errors+=prepareElectronSegment(model,invalid,q.region,q.field,s).status!=Status::invalid_input;
  invalid=state;invalid.clock.initialized=false;
  errors+=prepareElectronSegment(model,invalid,q.region,q.field,s).status!=Status::invalid_input;
  invalid=state;invalid.energy_MeV=s.electron_cut_total_MeV;
  errors+=prepareElectronSegment(model,invalid,q.region,q.field,s).status!=Status::invalid_input;
  // C7's one-step direction normalization at its magnetic cap is approximate.
  errors+=!validDirection({1.-5.e-10,0.,0.})||validDirection({1.-1.e-6,0.,0.});
  // A stopped history is terminated before its optical clock is decremented.
  q.energy=q.cut+.0001;q.initial_energy=q.energy;q.demfp=100.;
  state=track(model,q);p=prepareElectronSegment(model,state,q.region,q.field,s);Stream cut_rng{q.seed};
  auto cut=finishElectronSegment(model,state,q.region,q.field,s,p,{p.proposed_cm,BoundaryKind::none},cut_rng);
  errors+=cut.status!=TransportStatus::energy_cut||cut.particle.clock.remaining_mfp!=state.clock.remaining_mfp;
  // NOSCAT still draws the ELECTR azimuth (distinct from MSCAT's own uniforms).
  q.energy=1.e5;q.initial_energy=q.energy;q.demfp=1.e-8;
  auto tiny=evaluate(model,q);errors+=tiny.segment.scattering_angle!=0.||tiny.used!=1;
  return errors;
}
struct ContractCheck {
  ModelView model;
  KOKKOS_INLINE_FUNCTION void operator()(int,int& errors)const{errors+=contracts(model);}
};
void run(std::string const& reference,std::string const& path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto tables=readAirTables(path);
  ChannelThresholds thresholds{mass,140.,140.,422.};auto host=modelView(tables,thresholds);
  DeviceModel<Exec> device(tables,thresholds);auto model=device.view();
  auto floor=transportSettings(host,0.,0.,.0625);
  check(floor.electron_cut_total_MeV==tables.medium.ae&&floor.photon_cut_MeV==tables.medium.ap,"HATCH cut floor mismatch");
  int device_errors=0;Kokkos::parallel_reduce("C7_TRANSPORT_CONTRACTS",Kokkos::RangePolicy<Exec>(0,1),
    ContractCheck{model},device_errors);
  int host_errors=contracts(host);
  std::cout<<"execution_space="<<Exec::name()<<"\nhost_contract_errors="<<host_errors
           <<"\ndevice_contract_errors="<<device_errors<<'\n';
  check(host_errors==0&&device_errors==0,"Transport contract violation");
  if(reference=="--contracts")return;
  auto all=cases();
  Kokkos::View<Case*,typename Exec::memory_space> input("transport_inputs",all.size());
  Kokkos::View<Answer*,typename Exec::memory_space> output("transport_outputs",all.size());
  auto h=Kokkos::create_mirror_view(input);for(std::size_t i=0;i<all.size();++i)h(i)=all[i];
  Kokkos::deep_copy(input,h);
  Kokkos::parallel_for("C7_CPP_ELECTR_STEP",Kokkos::RangePolicy<Exec>(0,all.size()),Batch<Exec>{model,input,output});
  Exec().fence();auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  std::ifstream f(reference);check(bool(f),"Cannot read transport reference");
  std::array<double,18> host_diff{},device_diff{};
  double max_clock_absolute=0.;int shortened=0,cut=0,due=0,noscat=0,bad=0,rng_bad=0;
  for(std::size_t i=0;i<all.size();++i) {
    int used;std::array<double,18> expected;f>>used;for(auto& v:expected)f>>v;
    check(bool(f),"Truncated transport reference");
    auto cpu=evaluate(host,all[i]);auto gpu=result(i);
    check(cpu.segment.status==gpu.segment.status&&cpu.segment.path_corrected==gpu.segment.path_corrected,
          "Host/device transport branch mismatch");
    for(auto actual:{cpu,gpu}) {
      check(actual.plan.status==Status::success,"Transport plan failed");
      if(actual.segment.status!=TransportStatus::advanced&&actual.segment.status!=TransportStatus::collision_due&&
         actual.segment.status!=TransportStatus::energy_cut){std::cerr<<"invalid case="<<i<<'\n';throw std::runtime_error("Transport segment failed");}
      rng_bad+=actual.used!=used;
    }
    auto cv=values(cpu),gv=values(gpu);
    for(std::size_t j=0;j<expected.size();++j) {
      double scale=(j>=10&&j<=12)||j==14||j==15?1.:std::max(1.e-20,std::abs(expected[j]));
      // Remaining optical depth approaches zero at a collision. Normalize its
      // cancellation error to the INITIAL optical depth, not a near-zero remainder.
      if(j==16) {
        scale=all[i].demfp;
        max_clock_absolute=std::max(max_clock_absolute,std::abs(gv[j]-expected[j]));
      }
      host_diff[j]=std::max(host_diff[j],std::abs(cv[j]-expected[j])/scale);
      device_diff[j]=std::max(device_diff[j],std::abs(gv[j]-expected[j])/scale);
      if(!std::isfinite(gv[j])||!std::isfinite(cv[j])||
         std::abs(cv[j]-expected[j])>1.e-10*scale||std::abs(gv[j]-expected[j])>1.e-10*scale) {
        if(bad<8)std::cerr<<"case="<<i<<" field="<<j<<std::setprecision(17)
          <<" expected="<<expected[j]<<" host="<<cv[j]<<" device="<<gv[j]<<'\n';++bad;
      }
    }
    shortened+=cpu.segment.path_corrected;cut+=cpu.segment.status==TransportStatus::energy_cut;
    due+=cpu.segment.status==TransportStatus::collision_due;noscat+=cpu.segment.scattering_angle==0.;
  }
  std::string extra;check(!(f>>extra),"Extra transport reference data");
  std::cout<<std::setprecision(17)<<"synthetic_step_inputs="<<all.size()<<"\nshortened="<<shortened
    <<"\nenergy_cut="<<cut<<"\ncollision_due="<<due<<"\nno_scattering="<<noscat
    <<"\nrandom_count_mismatches="<<rng_bad<<"\nmax_optical_remainder_absolute="<<max_clock_absolute
    <<"\nfield_names=unclipped proposed geometry loss_path time_path deposit energy x y z u v w time theta alpha demfp dedx\n"
    <<"difference_units=relative except directions/angles absolute; demfp divided by initial optical depth\n";
  std::cout<<"host_difference=";for(auto v:host_diff)std::cout<<v<<' ';
  std::cout<<"\ndevice_difference=";for(auto v:device_diff)std::cout<<v<<' ';
  std::cout<<"\nscope=normal-air flat-frame local step with supplied geometry cap; not curved full shower\n";
  check(shortened>0&&cut>0&&due>0&&noscat>0,"Missing transport branch coverage");
  check(bad==0&&rng_bad==0,"Transport reference mismatch");
}
}
int main(int argc,char** argv) {
  try {
    if(argc==3&&std::string(argv[1])=="--make-cases"){writeCases(argv[2]);return 0;}
    if(argc!=3){std::cerr<<"Usage: test_egs4_transport REFERENCE|--contracts EGSDAT or --make-cases INPUT\n";return 2;}
    Kokkos::ScopeGuard guard(argc,argv);run(argv[1],argv[2]);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
