// Shared-queue regression: compare both resident grids with separate algorithms.
#include <Kokkos_Core.hpp>
#include <corsika/modules/radio/interface/KokkosAccumulator.hpp>
#include <corsika/modules/radio/interface/Output.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#if defined(RADIO_CONSUMER_CUDA)
using Space=Kokkos::Cuda;
#else
using Space=Kokkos::OpenMP;
#endif
namespace ri=corsika::radio::interface;
namespace d=ri::detail;
using Pair=ri::KokkosAccumulator<Space,true>;
void require(bool value,char const* message){if(!value)throw std::runtime_error(message);}
double difference(std::vector<double> const& a,std::vector<double> const& b){
  require(a.size()==b.size(),"moment shape");long double delta=0.,norm=0.;
  for(std::size_t i=0;i<a.size();++i){delta+=(a[i]-b[i])*(a[i]-b[i]);norm+=b[i]*b[i];}
  return std::sqrt(delta/std::max(norm,1.e-100L));
}
template<class Radio> void feed(Radio& radio,Pair::Tracks const& device,
    std::vector<ri::Track> const& cpu,Space const& execution){
  radio.accumulate(Kokkos::subview(device,std::make_pair(std::size_t{0},std::size_t{8190})),8190,execution);
  radio.accumulateCpu(cpu,execution);
  radio.accumulate(Kokkos::subview(device,std::make_pair(std::size_t{8190},device.extent(0))),device.extent(0)-8190,execution);
}
int main(int argc,char** argv){try{
  if(argc<3||argc>4)throw std::invalid_argument("usage: coreas_pair_fixture OUTPUT uniform|plane [profile-only]");
  std::string kind=argv[2];bool profile=argc==4;
  require(!profile||std::string(argv[3])=="profile-only","profile argument");
  require(kind=="uniform"||kind=="plane","fixture geometry");
  std::filesystem::path root=argv[1];require(!std::filesystem::exists(root),"output exists");
  Kokkos::InitializationSettings settings;settings.set_num_threads(256);settings.set_device_id(0);
  Kokkos::ScopeGuard runtime(settings);Space execution;
  ri::Config c;c.enabled=true;c.samples=256;c.start_time_s=0.;c.moment_order=12;
  c.maximum_subdivision_depth=20;c.propagation.media[0].refractive_index=1.33;
  c.propagation.media[1].refractive_index=1.7;c.observers={{"receiver",{10.,3.,20.},0}};
  if(kind=="plane")c.propagation.geometry=ri::Geometry::Plane;
  std::vector<ri::Track> cpu;
  auto track=[&](std::size_t i){
    unsigned region=kind=="plane"&&i%2?1:0;double z=region?-2.:2.;
    ri::Vec3 a{.001*double(i%5),0.,z},b{a.x,.001,z+.02};
    double time=(i%7)*1.e-11,dt=d::norm(d::sub(b,a))/(.7*d::light_speed);
    return ri::Track{a,b,time,time+dt,-1.,1.+.01*(i%3),i+1,0,region,1};
  };
  Pair::Tracks device("interface_radio_transport_tracks",8195);
  auto host=Kokkos::create_mirror_view(device);
  for(std::size_t i=0;i<device.extent(0);++i)host(i)=track(i);
  for(std::size_t i=0;i<5;++i)cpu.push_back(track(10000+i));
  Kokkos::deep_copy(execution,device,host);execution.fence();
  // Combined admission must fail even when each individual algorithm fits.
  if(!profile){
    auto limited=c;limited.algorithm=ri::Algorithm::CoREAS;
    limited.maximum_device_bytes=ri::projectedRadioBytes(limited,3,Pair::deviceBuffering);
    bool caught=false;try{Pair invalid(limited,{},0,3,execution);}catch(std::length_error const&){caught=true;}
    require(caught,"combined memory budget not enforced");
    auto zero=c;zero.moment_order=0;caught=false;
    try{Pair::projectedBytes(zero,3);}catch(std::invalid_argument const&){caught=true;}
    require(caught,"CoREAS order zero accepted in paired mode");
  }
  Pair paired(c,{},0,3,execution);feed(paired,device,cpu,execution);auto result=paired.download(execution);
  require(result.coreas.config.algorithm==ri::Algorithm::CoREAS&&result.zhs.config.algorithm==ri::Algorithm::ZHS,"both algorithms");
  for(auto const* r:{&result.coreas,&result.zhs}){
    require(r->statistics.device_tracks==8195&&r->statistics.cpu_tracks==5,"shared source count");
    require(r->statistics.downloads==1,"finalization count");
  }
  bool caught=false;try{paired.download(execution);}catch(std::logic_error const&){caught=true;}
  require(caught,"second finalization accepted");caught=false;
  try{paired.accumulateCpu(cpu,execution);}catch(std::logic_error const&){caught=true;}
  require(caught,"sources accepted after finalization");
  double maximum=0.;
  if(!profile)for(auto algorithm:{ri::Algorithm::CoREAS,ri::Algorithm::ZHS}){
    auto config=c;config.algorithm=algorithm;ri::KokkosAccumulator<Space> separate(config,{},0,3,execution);
    feed(separate,device,cpu,execution);auto reference=separate.download(execution);
    auto const& actual=algorithm==ri::Algorithm::CoREAS?result.coreas:result.zhs;
    maximum=std::max(maximum,difference(actual.moments,reference.moments));
    maximum=std::max(maximum,difference(actual.regularized_moments,reference.regularized_moments));
    require(actual.statistics.paths==reference.statistics.paths,"path count differs");
    require(actual.statistics.wavefronts==reference.statistics.wavefronts,"batch count differs");
  }
  require(maximum<1.e-10,"paired results differ from independent algorithms");
  ri::writeResult(result.coreas,(root/"CoREAS").string());ri::writeResult(result.zhs,(root/"ZHS").string());
  std::ofstream out(root/"pair.json");out<<"{\"passed\":true,\"execution\":\""<<Space::name()<<"\",\"geometry\":\""<<kind
    <<"\",\"device_tracks\":8195,\"cpu_tracks\":5,\"batches_per_algorithm\":"<<result.coreas.statistics.wavefronts
    <<",\"device_bytes\":"<<result.device_bytes<<",\"reference_relative_l2\":"<<maximum<<"}\n";
  std::cout<<"PASS paired radio "<<Space::name()<<' '<<kind<<" relative="<<maximum<<'\n';return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
