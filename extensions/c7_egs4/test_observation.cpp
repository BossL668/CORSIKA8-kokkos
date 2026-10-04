#include "Egs4Observation.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
using namespace c7_egs4;
void check(bool b,char const* m){if(!b)throw std::runtime_error(m);}
struct Case {CurvedTrack track;CurvedConfig config;double horizontal_cm,step_cm;};
std::vector<Case> cases() {
  std::vector<Case> result;
  Point points[]={{0.,0.,-110001.},{1.e5,-2.e5,-1.2e5},{-1.e7,2.e6,-2.e5},
    {1.e7,-2.e6,-2.e5},{-1.e5,0.,-110000.},{0.,0.,-109999.}};
  Direction directions[]={{0.,0.,1.},{.3,.4,std::sqrt(.75)},{0.,.8,.6},
    {-.8,.6,0.},{.3,.4,-std::sqrt(.75)},{0.,0.,-1.}};
  for(int pid:{11,22})for(bool flat:{true,false})for(double minimum:{-1.,0.})
    for(auto point:points)for(auto direction:directions) {
      CurvedTrack t;t.particle.pdg=pid;t.particle.energy_MeV=100.;t.particle.position_cm=point;
      t.particle.direction=direction;t.observer_cm=point;t.wa=1.;t.wap=.8;
      result.push_back({t,{flat,6.371315e8,1.1e5,minimum},2.e6,2345.});
    }
  return result;
}
std::array<double,12> values(ObservationPreparation const& r) {
  auto t=r.particle;auto p=t.particle;
  return {p.position_cm.x,p.position_cm.y,p.position_cm.z,p.direction.x,p.direction.y,p.direction.z,
    t.observer_cm.z,t.wa,t.wap,r.step_cm,double(t.detector_local),
    r.status==ObservationPreparationStatus::transport?1.:r.status==ObservationPreparationStatus::angular_discard?2.:0.};
}
void writeCases(char const* path) {
  std::ofstream f(path);check(bool(f),"Cannot write observation cases");auto all=cases();
  f<<std::setprecision(17)<<all.size()<<'\n';
  for(auto q:all) {
    auto t=q.track;auto p=t.particle;
    f<<(p.pdg==22?1:0)<<' '<<q.config.flat_output<<' '<<p.position_cm.x<<' '<<p.position_cm.y<<' '<<p.position_cm.z<<' '
      <<p.direction.x<<' '<<p.direction.y<<' '<<p.direction.z<<' '<<t.wa<<' '<<t.wap<<' '<<t.observer_cm.z<<' '
      <<q.config.earth_radius_cm<<' '<<q.horizontal_cm<<' '<<q.config.observation_height_cm<<' '
      <<q.config.minimum_cosine<<' '<<q.step_cm<<'\n';
  }
  check(bool(f),"Cannot finish observation inputs");
}
template<class Exec>struct Batch {
  Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<ObservationPreparation*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const {
    auto q=input(i);output(i)=prepareObservationStep(q.track,q.config,q.horizontal_cm,q.step_cm);
  }
};
void run(char const* path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto all=cases();
  Kokkos::View<Case*,typename Exec::memory_space> input("observation_cases",all.size());
  Kokkos::View<ObservationPreparation*,typename Exec::memory_space> output("observation_result",all.size());
  auto h=Kokkos::create_mirror_view(input);for(std::size_t i=0;i<all.size();++i)h(i)=all[i];Kokkos::deep_copy(input,h);
  Kokkos::parallel_for("native_observation_preparation",Kokkos::RangePolicy<Exec>(0,all.size()),Batch<Exec>{input,output});
  Exec().fence();auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  std::ifstream f(path);check(bool(f),"Cannot read observation reference");
  std::array<double,12> host_error{},device_error{};int transported=0,discarded=0,control=0,nonunit=0;
  for(std::size_t i=0;i<all.size();++i) {
    std::array<double,12> reference;for(auto& x:reference)f>>x;check(bool(f),"Truncated observation reference");
    auto q=all[i];auto host=prepareObservationStep(q.track,q.config,q.horizontal_cm,q.step_cm);
    auto a=values(host),b=values(result(i));
    for(int j=0;j<12;++j) {
      host_error[j]=std::max(host_error[j],std::abs(a[j]-reference[j]));
      device_error[j]=std::max(device_error[j],std::abs(b[j]-reference[j]));
      double bound=(j<3||j==6||j==9)?1.e-3:1.e-10;
      if(std::abs(a[j]-reference[j])>bound||std::abs(b[j]-reference[j])>bound) {
        std::cerr<<"case="<<i<<" field="<<j<<std::setprecision(17)<<" reference="<<reference[j]<<" host="<<a[j]<<" device="<<b[j]<<'\n';
        throw std::runtime_error("Observation preparation differs from C7");
      }
    }
    control+=a[10]!=reference[10]||a[11]!=reference[11]||b[10]!=reference[10]||b[11]!=reference[11];
    transported+=host.status==ObservationPreparationStatus::transport;
    discarded+=host.status==ObservationPreparationStatus::angular_discard;
    nonunit+=host.status==ObservationPreparationStatus::transport&&!host.transport_ready;
    check(host.transport_ready==result(i).transport_ready,"Host/device transport readiness differs");
  }
  std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<"\ninputs="<<all.size()<<"\ntransported="<<transported
    <<"\nangular_discard="<<discarded<<"\nliteral_nonunit_direction="<<nonunit<<"\ncontrol_mismatches="<<control<<"\nfields=x y z u v w observer_z wa wap step detector_local status\n"
    <<"host_max_absolute=";for(auto x:host_error)std::cout<<x<<' ';
  std::cout<<"\ndevice_max_absolute=";for(auto x:device_error)std::cout<<x<<' ';
  std::cout<<"\nscope=original C7 curved detector-approach preprocessing, not complete shower\n";
  check(transported&&discarded&&!control,"Missing observation branch coverage");
}
}
int main(int argc,char** argv) {
  try{if(argc==3&&std::string(argv[1])=="--make-cases"){writeCases(argv[2]);return 0;}
    if(argc!=2)return 2;Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
