#include "Egs4CurvedFrame.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
using namespace c7_egs4;
void check(bool b,char const* text){if(!b)throw std::runtime_error(text);}
Point observer(Point p,double earth) {
  double lateral=std::sqrt(p.x*p.x+p.y*p.y),wa=std::cos(lateral/earth);
  double z=earth-(earth-p.z)*wa;
  if(wa==1.)return {p.x,p.y,z};
  double phi=std::atan2(p.y,p.x),rr=std::sqrt((1.-wa)*(1.+wa))*(earth-z)/wa;
  return {rr*std::cos(phi),rr*std::sin(phi),z};
}
std::vector<CurvedInput> cases() {
  std::vector<CurvedInput> result;
  for(bool photon:{false,true})for(int mode:{0,1,2})for(Point p:{Point{0.,0.,-1.e6},Point{1.e6,-2.e6,-1.e7},Point{-1.e7,3.e7,-1.2e5}})
  for(Direction d:{Direction{0.,0.,1.},Direction{0.,0.,-1.},Direction{.6,0.,.8},Direction{0.,.8,-.6},Direction{.6,.8,0.}})
  for(double step:{.0001,.01,100.,1.e5,1.e7})for(double wcut:{-1.,0.}) {
    CurvedInput q;q.photon=photon;q.detector_local=mode==1;q.flat_output=mode!=2;
    q.previous_local=p;q.earth_radius_cm=6.371315e8;q.observation_height_cm=1.1e5;
    q.previous_observer=q.detector_local?p:observer(p,q.earth_radius_cm);
    q.advanced_local={p.x+d.x*step,p.y+d.y*step,p.z+d.z*step};q.advanced_direction=d;
    q.minimum_cosine=wcut;q.geometric_step_cm=step;q.time_path_cm=step*(photon?1.:1.01);
    q.total_energy_MeV=photon?10.:1.;result.push_back(q);
  }
  return result;
}
std::array<double,22> input(CurvedInput q) {
  return {double(q.photon),double(q.detector_local),double(q.flat_output),
    q.previous_local.x,q.previous_local.y,q.previous_local.z,
    q.previous_observer.x,q.previous_observer.y,q.previous_observer.z,
    q.advanced_local.x,q.advanced_local.y,q.advanced_local.z,
    q.advanced_direction.x,q.advanced_direction.y,q.advanced_direction.z,
    q.earth_radius_cm,q.observation_height_cm,q.minimum_cosine,q.geometric_step_cm,q.time_path_cm,q.total_energy_MeV,q.mass_MeV};
}
std::array<double,13> output(CurvedResult r) {
  return {r.local.x,r.local.y,r.local.z,r.direction.x,r.direction.y,r.direction.z,
    r.observer.x,r.observer.y,r.observer.z,r.wa,r.wap,r.time_increment_s,double(r.status)};
}
void writeCases(char const* path) {
  auto all=cases();std::ofstream file(path);check(bool(file),"Cannot write curved fixture inputs");
  file<<std::setprecision(17)<<all.size()<<'\n';for(auto q:all){for(auto x:input(q))file<<x<<' ';file<<'\n';}
}
template<class Exec>struct Batch {
  Kokkos::View<CurvedInput*,typename Exec::memory_space> input;
  Kokkos::View<CurvedResult*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=rebaseCurvedStep(input(i));}
};
void run(char const* path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto all=cases();std::ifstream file(path);check(bool(file),"Cannot read curved reference");
  std::vector<std::array<double,13>> oracle(all.size());for(auto& row:oracle)for(auto& x:row)check(bool(file>>x),"Truncated curved fixture");
  Kokkos::View<CurvedInput*,typename Exec::memory_space> inputs("curved_inputs",all.size());
  Kokkos::View<CurvedResult*,typename Exec::memory_space> outputs("curved_outputs",all.size());
  auto h=Kokkos::create_mirror_view(inputs);for(std::size_t i=0;i<all.size();++i)h(i)=all[i];Kokkos::deep_copy(inputs,h);
  Kokkos::parallel_for("C7_CURVED_FRAME",Kokkos::RangePolicy<Exec>(0,all.size()),Batch<Exec>{inputs,outputs});Exec().fence();
  auto device=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},outputs);
  double cpu_max[12]{},gpu_max[12]{};int statuses[4]{},mismatches=0;
  for(std::size_t i=0;i<all.size();++i) {
    auto cpu=output(rebaseCurvedStep(all[i])),gpu=output(device(i));
    ++statuses[int(cpu[12])];mismatches+=cpu[12]!=oracle[i][12]||gpu[12]!=oracle[i][12];
    for(int j=0;j<12;++j) {
      cpu_max[j]=std::max(cpu_max[j],std::abs(cpu[j]-oracle[i][j]));
      gpu_max[j]=std::max(gpu_max[j],std::abs(gpu[j]-oracle[i][j]));
    }
  }
  std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<"\ncurved_inputs="<<all.size()
    <<"\nadvanced="<<statuses[1]<<"\nangle_cut="<<statuses[2]<<"\nobservation="<<statuses[3]
    <<"\nstatus_mismatches="<<mismatches<<"\nfields=x y z u v w observer_x observer_y observer_z WA WAP time\n"
    <<"units=absolute cm, direction cosine, and seconds\nhost_max_absolute=";
  for(auto x:cpu_max)std::cout<<x<<' ';std::cout<<"\ndevice_max_absolute=";
  for(auto x:gpu_max)std::cout<<x<<' ';std::cout<<"\nscope=original C7 CURVED normal-step rebase/time correction, not HOWFAR or a shower benchmark\n";
  check(!statuses[0]&&!mismatches,"Curved frame disposition mismatch");
  for(int j=0;j<12;++j) {
    double limit=j==11?1.e-11:((j>=3&&j<=5)||j==9||j==10?1.e-7:1.e-3);
    check(cpu_max[j]<limit&&gpu_max[j]<limit,"Curved arithmetic mismatch");
  }
  auto bad=all[0];bad.earth_radius_cm=0.;check(rebaseCurvedStep(bad).status==CurvedStatus::invalid,"Zero Earth radius accepted");
}
}
int main(int argc,char** argv) {
  try {
    if(argc==3&&std::string(argv[1])=="--make-cases"){writeCases(argv[2]);return 0;}
    if(argc!=2){std::cerr<<"Usage: test_egs4_curved [--make-cases] file\n";return 2;}
    Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
