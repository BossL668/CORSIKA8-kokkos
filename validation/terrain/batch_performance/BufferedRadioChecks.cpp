// Same sources with different batch boundaries, CPU/device interleaving and tails.
#include <Kokkos_Core.hpp>
#include <corsika/modules/radio/interface/KokkosAccumulator.hpp>
#include <corsika/modules/radio/interface/Output.hpp>
#include <corsika/modules/transport/kokkos/ExecutionSpace.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace ri=corsika::radio::interface;
using Space=corsika::interfaces::kokkos::ExecutionSpace;
using Pair=ri::KokkosAccumulator<Space,true>;
void require(bool p,char const* why){if(!p)throw std::runtime_error(why);}
double compare(std::vector<double> const& a,std::vector<double> const& b){
  require(a.size()==b.size(),"array extent");long double delta=0.,norm=0.;
  for(std::size_t i=0;i<a.size();++i){
    require(std::isfinite(a[i])&&std::isfinite(b[i]),"finite arrays");
    delta+=(a[i]-b[i])*(a[i]-b[i]);norm+=b[i]*b[i];
  }
  require(norm>0.,"nonzero reference field/moments");
  double relative=std::sqrt(delta/norm);require(relative<1.e-11,"buffering changed moments/field");return relative;
}
void same(ri::Result const& a,ri::Result const& b){
  compare(a.moments,b.moments);
  if(!a.regularized_moments.empty()){
    // These can be exactly zero away from the Cherenkov fallback.
    long double norm=0;for(auto v:b.regularized_moments)norm+=v*v;
    if(norm>0)compare(a.regularized_moments,b.regularized_moments);
    else require(a.regularized_moments==b.regularized_moments,"zero regularized grid");
  }
  compare(ri::render(a).field,ri::render(b).field);
  auto const& x=a.statistics;auto const& y=b.statistics;
  require(x.device_tracks==y.device_tracks&&x.cpu_tracks==y.cpu_tracks&&
      x.track_observer_pairs==y.track_observer_pairs&&x.paths==y.paths&&x.leaves==y.leaves&&
      x.endpoint_contributions==y.endpoint_contributions&&x.regularized_pairs==y.regularized_pairs,
      "buffering lost/repeated sources or paths");
}
int main(){try{
  Kokkos::InitializationSettings settings;settings.set_num_threads(256);
  Kokkos::ScopeGuard runtime(settings);Space execution;
  for(bool transmitted:{false,true}){
    ri::Config c;c.enabled=true;c.samples=1024;c.start_time_s=0.;c.maximum_subdivision_depth=20;
    c.propagation.geometry=transmitted?ri::Geometry::Plane:ri::Geometry::Uniform;
    c.propagation.media[0].refractive_index=1.5;c.propagation.media[1].refractive_index=2.;
    c.observers={{"normal",{10.,0.,20.},0},{"oblique",{30.,3.,20.},0}};
    Pair::Tracks input("buffering_sources",8211);auto host=Kokkos::create_mirror_view(input);
    for(std::size_t i=0;i<input.extent(0);++i){
      ri::Vec3 a{.001*double(i%7),0.,transmitted?-2.:2.},b{a.x,.001,a.z+.02};
      double dt=ri::detail::norm(ri::detail::sub(b,a))/(.9*ri::detail::light_speed),t=(i%5)*1.e-11;
      host(i)={a,b,t,t+dt,i%3? -1.:1.,1.+.01*(i%3),i+1,0,unsigned(transmitted?1:0),1};
    }
    Kokkos::deep_copy(execution,input,host);
    std::vector<ri::Track> cpu{host(0),host(1),host(2),host(3),host(4)};
    auto feed=[&](Pair& radio){
      radio.accumulate(Kokkos::subview(input,std::make_pair(std::size_t{0},std::size_t{8190})),8190,execution);
      radio.accumulateCpu(cpu,execution);
      radio.accumulate(Kokkos::subview(input,std::make_pair(std::size_t{8190},input.extent(0))),21,execution);
    };
    Pair original(c,{},0,8192,execution);feed(original);auto reference=original.download(execution);
    for(std::size_t capacity:{std::size_t{7},std::size_t{8192},std::size_t{16384}}){
      auto buffered=c;buffered.host_track_buffer_capacity=capacity;
      auto needed=Pair::projectedBytes(buffered,8192);
      auto rejected=buffered;rejected.maximum_device_bytes=needed-1;
      bool caught=false;try{Pair bad(rejected,{},0,8192,execution);}catch(std::length_error const&){caught=true;}
      require(caught,"buffer memory not budgeted");
      Pair radio(buffered,{},0,8192,execution);feed(radio);auto result=radio.download(execution);
      same(result.coreas,reference.coreas);same(result.zhs,reference.zhs);
      caught=false;try{radio.download(execution);}catch(std::logic_error const&){caught=true;}
      require(caught,"double finalization");
      caught=false;try{radio.accumulateCpu(cpu,execution);}catch(std::logic_error const&){caught=true;}
      require(caught,"source admitted after seal");
      std::cout<<"PASS "<<(transmitted?"transmitted":"uniform")<<" buffer="<<capacity
        <<" device_sources="<<result.coreas.statistics.device_tracks<<" cpu_sources="<<result.coreas.statistics.cpu_tracks
        <<" launches_per_algorithm="<<result.coreas.statistics.wavefronts<<'\n';
    }
  }
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
