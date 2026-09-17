// Verify real allocation lifetime while the accumulator object remains alive.
#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Profiling.hpp>
#include <corsika/modules/radio/interface/KokkosAccumulator.hpp>
#include <corsika/modules/transport/kokkos/ExecutionSpace.hpp>
#include <iostream>
#include <string>
namespace ri=corsika::radio::interface;
using Space=corsika::interfaces::kokkos::ExecutionSpace;
unsigned released=0;
void freed(Kokkos::Tools::SpaceHandle,const char* label,const void*,const uint64_t) {
  std::string name=label;
  if(name=="interface_radio_moments"||name=="interface_radio_coreas_regularized_moments"||
     name=="interface_radio_zhs_moments")++released;
}
int main(){try{
  Kokkos::InitializationSettings settings;settings.set_num_threads(2);
  Kokkos::ScopeGuard guard(settings);Space execution;
  Kokkos::Tools::Experimental::set_deallocate_data_callback(freed);
  ri::Config c;c.enabled=true;c.samples=65536;c.start_time_s=0.;c.sample_rate_Hz=1.e9;
  c.observers={{"air",{20,0,20},0},{"air2",{30,0,20},0}};
  c.propagation.geometry=ri::Geometry::Uniform;
  ri::KokkosAccumulator<Space,true> accumulator(c,{},0,64,execution);
  auto result=accumulator.download(execution);
  if(released!=3)throw std::runtime_error("Moment allocations retained after download");
  auto size=(c.moment_order+1)*c.observers.size()*6*c.samples;
  if(result.coreas.moments.size()!=size||result.coreas.regularized_moments.size()!=size||
     result.zhs.moments.size()!=size)throw std::runtime_error("Incomplete result arrays");
  std::cout<<"{\"passed\":true,\"released_grids_before_accumulator_destruction\":"<<released<<"}\n";
  Kokkos::Tools::Experimental::set_deallocate_data_callback(nullptr);
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
