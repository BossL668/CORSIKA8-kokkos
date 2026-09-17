#include <Kokkos_Core.hpp>
#include <corsika/modules/radio/interface/KokkosAccumulator.hpp>
#include <corsika/modules/radio/interface/Output.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#if defined(RADIO_CONSUMER_CUDA)
using Space=Kokkos::Cuda;
#else
using Space=Kokkos::OpenMP;
#endif
namespace ri=corsika::radio::interface;
int main(int argc,char** argv){try{
  if(argc!=2)throw std::invalid_argument("usage: radio_consumer NEW_OUTPUT_DIRECTORY");
  Kokkos::InitializationSettings settings;settings.set_num_threads(256);settings.set_device_id(0);
  Kokkos::ScopeGuard runtime(settings);Space execution;
  ri::Config config;config.enabled=true;config.samples=128;config.start_time_s=0.;
  config.propagation.media[0].refractive_index=1.5;
  config.observers={{"standalone",{10.,0.,0.},0}};
  ri::KokkosAccumulator<Space> radio(config,{},0,4,execution);
  ri::Track track{{0.,0.,0.},{0.,0.,.1},0.,.1/(.99*299792458.),-1.,1.,1,0,0,1};
  radio.accumulateCpu({track},execution);auto result=radio.download(execution);
  auto rendered=ri::render(result);double peak=0.;
  for(double e:rendered.field){if(!std::isfinite(e))throw std::runtime_error("nonfinite standalone field");peak=std::max(peak,std::abs(e));}
  if(!(peak>0.)||result.statistics.cpu_tracks!=1||result.statistics.downloads!=1)
    throw std::runtime_error("invalid standalone radio result");
  ri::writeResult(result,argv[1]);
  std::cout<<"PASS exported InterfaceRadio consumer "<<Space::name()<<" peak="<<peak<<'\n';return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
