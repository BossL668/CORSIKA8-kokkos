#include "Egs4Geometry.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
using namespace c7_egs4;
void check(bool b,char const* text){if(!b)throw std::runtime_error(text);}
struct Case{GeometryConfig config;GeometryInput input;};
std::vector<Case> cases() {
  std::vector<Case> all;
  for(bool curved:{false,true})for(bool flat:{false,true})for(int region:{1,2,3,4,5,6})
  for(double w:{-1.,-.6,-.0031,-.002,0.,.0002,.001,.01,.6,1.})
  for(double step:{.0001,1000.,1.e6,1.e8})for(double x:{0.,2.e7}) {
    Case c;c.config={curved,flat,6.371315e8,{1.128e7,1.e7,4.e6,4.e5,0.,-1.},1.1e5,1000.,1.e6,-900.};
    double height=region==1?1.128e7:(region==6?-1.:.5*(c.config.bound_cm[region-2]+c.config.bound_cm[region-1]));
    c.input.local={x,0.,-height};c.input.direction={-std::sqrt((1.-w)*(1.+w)),0.,w};
    c.input.wa=std::cos(x/c.config.earth_radius_cm);c.input.region=region;c.input.requested_cm=step;
    c.input.thickness_g_cm2=500.;c.input.nearest_cm=123.;all.push_back(c);
  }
  // Explicit near-boundary and stale-index recovery fixtures.
  for(int region:{2,3,4,5})for(double delta:{-.01,0.,.01})for(double w:{-.002,-.7,.7}) {
    Case c=all[0];c.config.curved=true;c.input.region=region;
    c.input.local={0.,0.,-(c.config.bound_cm[region-2]+delta)};
    c.input.direction={std::sqrt((1.-w)*(1.+w)),0.,w};c.input.wa=1.;c.input.requested_cm=1.e6;
    all.push_back(c);
  }
  return all;
}
std::array<double,25> input(Case q) {
  auto c=q.config;auto p=q.input;
  return {double(c.curved),double(c.flat_output),c.earth_radius_cm,
    c.bound_cm[0],c.bound_cm[1],c.bound_cm[2],c.bound_cm[3],c.bound_cm[4],c.bound_cm[5],
    c.observation_height_cm,c.minimum_horizontal_cm,c.maximum_horizontal_cm,c.horizontal_slope_cm_per_g_cm2,
    p.local.x,p.local.y,p.local.z,p.direction.x,p.direction.y,p.direction.z,double(p.region),p.wa,p.thickness_g_cm2,
    p.requested_cm,p.nearest_cm,0.};
}
std::array<double,6> output(GeometryResult r) {
  return {r.step_cm,r.nearest_cm,double(r.region),double(r.next_observation),double(r.discard),double(r.energy_discard)};
}
void writeCases(char const* path) {
  auto all=cases();std::ofstream file(path);check(bool(file),"Cannot write geometry cases");
  file<<std::setprecision(17)<<all.size()<<'\n';for(auto q:all){for(double x:input(q))file<<x<<' ';file<<'\n';}
}
template<class Exec>struct Kernel {
  Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<GeometryResult*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{auto q=input(i);output(i)=howFar(q.config,q.input);}
};
void run(char const* path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto all=cases();std::ifstream file(path);check(bool(file),"Cannot read HOWFAR fixture");
  std::vector<std::array<double,6>> oracle(all.size());for(auto& r:oracle)for(double& x:r)check(bool(file>>x),"Truncated HOWFAR fixture");
  Kokkos::View<Case*,typename Exec::memory_space> inputs("geometry_inputs",all.size());
  Kokkos::View<GeometryResult*,typename Exec::memory_space> outputs("geometry_outputs",all.size());
  auto h=Kokkos::create_mirror_view(inputs);for(std::size_t i=0;i<all.size();++i)h(i)=all[i];Kokkos::deep_copy(inputs,h);
  Kokkos::parallel_for("C7_HOWFAR",Kokkos::RangePolicy<Exec>(0,all.size()),Kernel<Exec>{inputs,outputs});Exec().fence();
  auto device=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},outputs);
  double host[2]{},gpu[2]{};int mismatches=0,invalid=0,boundaries=0,observations=0,discard=0;
  for(std::size_t i=0;i<all.size();++i) {
    auto r=howFar(all[i].config,all[i].input),d=device(i);auto a=output(r),b=output(d);
    invalid+=!r.valid||!d.valid;
    bool mismatch=false;for(int j=2;j<6;++j)mismatch|=a[j]!=oracle[i][j]||b[j]!=oracle[i][j];
    mismatches+=mismatch;
    if(mismatch&&mismatches<=3)std::cerr<<"geometry_mismatch_index="<<i<<" expected="<<oracle[i][2]<<','<<oracle[i][3]<<','<<oracle[i][4]<<" actual="<<a[2]<<','<<a[3]<<','<<a[4]<<'\n';
    for(int j=0;j<2;++j){host[j]=std::max(host[j],std::abs(a[j]-oracle[i][j]));gpu[j]=std::max(gpu[j],std::abs(b[j]-oracle[i][j]));}
    boundaries+=r.region!=all[i].input.region;observations+=r.next_observation!=1;discard+=r.discard!=0;
  }
  std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<"\ngeometry_inputs="<<all.size()
    <<"\nregion_changes="<<boundaries<<"\nobservations="<<observations<<"\ndiscard_dispositions="<<discard
    <<"\ninvalid="<<invalid<<"\ncontrol_mismatches="<<mismatches<<"\nhost_step_difference_cm="<<host[0]
    <<"\nhost_nearest_difference_cm="<<host[1]<<"\ndevice_step_difference_cm="<<gpu[0]
    <<"\ndevice_nearest_difference_cm="<<gpu[1]<<"\nscope=original HOWFAR, single observation level, CURVED/UPWARD and flat/UPWARD; supplied atmospheric thickness\n";
  check(!invalid&&!mismatches&&host[0]<1.e-5&&host[1]<1.e-5&&gpu[0]<1.e-5&&gpu[1]<1.e-5,"HOWFAR mismatch");
  auto bad=all[0];bad.config.bound_cm[1]=bad.config.bound_cm[0];check(!howFar(bad.config,bad.input).valid,"Unordered layers accepted");
}
}
int main(int argc,char** argv) {
  try {
    if(argc==3&&std::string(argv[1])=="--make-cases"){writeCases(argv[2]);return 0;}
    if(argc!=2){std::cerr<<"Usage: test_egs4_geometry [--make-cases] file\n";return 2;}
    Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
