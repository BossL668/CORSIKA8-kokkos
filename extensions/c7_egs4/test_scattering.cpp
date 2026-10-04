#include "Egs4Scattering.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

namespace {
using namespace c7_egs4;
void check(bool b,char const* msg){if(!b)throw std::runtime_error(msg);}
struct Case {double energy,grammage;std::int64_t seed;};
struct Stream {
  std::int64_t seed;int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& u) {
    if(used>=4096)return false;
    seed=(seed*48271)%2147483647;u=double(seed)/2147483647.;++used;return true;
  }
};
struct Answer{ScatteringResult state;int used;};
KOKKOS_INLINE_FUNCTION Answer evaluate(Medium m,Case q) {
  Stream rng{q.seed};auto r=sampleScattering(m,q.energy,.51099895,q.grammage/m.rho,rng);
  return {r,rng.used};
}
std::vector<Case> cases() {
  std::vector<Case> out;std::mt19937_64 rng(2026100303);
  for(double kinetic:{.5,1.,10.,1000.,1.e5})
    for(double gram:{0.,1.e-9,1.e-7,1.e-6,1.e-5,1.e-4,1.e-3,.01,.034372012168238514,.1})
      for(int j=0;j<128;++j)out.push_back({kinetic+.51099895,gram,std::int64_t(rng()%2147483646)+1});
  return out;
}
void writeCases(std::string const& path) {
  auto all=cases();std::ofstream f(path);check(bool(f),"Cannot write scatter inputs");
  f<<std::setprecision(17)<<all.size()<<'\n';
  for(auto q:all)f<<q.energy<<' '<<q.grammage<<' '<<q.seed<<'\n';
  check(bool(f),"Incomplete scatter inputs");
}
template<class Exec>struct Batch {
  Medium medium;
  Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(medium,input(i));}
};
void run(std::string const& reference,std::string const& egspath) {
  using Exec=Kokkos::DefaultExecutionSpace;auto m=readAirTables(egspath).medium;
  check(m.rho==scattering_detail::Rho&&m.blcc==scattering_detail::Blcc&&m.xcc==scattering_detail::Xcc,
        "Material differs from reused B reference");
  auto all=cases();
  Kokkos::View<Case*,typename Exec::memory_space> input("scattering_inputs",all.size());
  Kokkos::View<Answer*,typename Exec::memory_space> output("scattering_outputs",all.size());
  auto h=Kokkos::create_mirror_view(input);for(std::size_t i=0;i<all.size();++i)h(i)=all[i];
  Kokkos::deep_copy(input,h);
  Kokkos::parallel_for("C7_CPP_MSCAT",Kokkos::RangePolicy<Exec>(0,all.size()),Batch<Exec>{m,input,output});
  Exec().fence();auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  std::ifstream f(reference);check(bool(f),"Cannot open scattering reference");
  double host_diff=0.,device_diff=0.;int zero=0,retries=0;
  for(std::size_t i=0;i<all.size();++i) {
    int used,status;double theta;f>>used>>status>>theta;
    check(bool(f)&&(status==0||status==1),"C7 reference sampling failed");
    auto a=result(i),host=evaluate(m,all[i]);
    for(auto actual:{a,host}) {
      check(actual.state.status==InteractionStatus::success,"C++ MSCAT failed");
      check(actual.used==used,"MSCAT random-consumption mismatch");
      check(actual.state.below_scattering_threshold==(status==1),"MSCAT threshold mismatch");
      check(std::isfinite(actual.state.angle)&&std::abs(actual.state.angle-theta)<1.e-12,"MSCAT angle mismatch");
    }
    host_diff=std::max(host_diff,std::abs(host.state.angle-theta));
    device_diff=std::max(device_diff,std::abs(a.state.angle-theta));
    zero+=status==1;retries+=a.state.trials>1;
  }
  std::string extra;check(!(f>>extra),"Extra scattering reference data");
  check(zero>0&&zero<int(all.size())&&retries>0,"Missing scattering branch coverage");
  Stream rng{1};check(sampleScattering(m,.4,.51099895,1.,rng).status==InteractionStatus::invalid_input,
                     "Invalid scattering energy accepted");
  check(rng.used==0,"Invalid input consumed randoms");
  std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<"\nsynthetic_MSCAT_inputs="<<all.size()
    <<"\nmax_host_angle_difference="<<host_diff<<"\nmax_device_angle_difference="<<device_diff
    <<"\nno_scattering_inputs="<<zero<<"\ninputs_with_rejection="<<retries
    <<"\nreference=earlier isolated B C7 MSCAT fixture; not a full trajectory comparison\n";
}
}
int main(int argc,char** argv) {
  try {
    if(argc==3&&std::string(argv[1])=="--make-cases"){writeCases(argv[2]);return 0;}
    if(argc!=3){std::cerr<<"Usage: test_egs4_scattering REFERENCE EGSDAT or --make-cases INPUT\n";return 2;}
    Kokkos::ScopeGuard guard(argc,argv);run(argv[1],argv[2]);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
