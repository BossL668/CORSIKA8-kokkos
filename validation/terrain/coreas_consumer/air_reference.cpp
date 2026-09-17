// Validation only: invoke the unmodified air CoREAS kernel in uniform air.
// No air implementation is linked into the production interface radio target.
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/radio/detail/RadioProjectionStep.hpp>
#include <corsika/modules/radio/interface/KokkosAccumulator.hpp>
#include <corsika/modules/radio/interface/Output.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#if defined(RADIO_CONSUMER_CUDA)
using Space=Kokkos::Cuda;
#else
using Space=Kokkos::OpenMP;
#endif
namespace ri=corsika::radio::interface;
namespace d=ri::detail;
namespace air=corsika::accelerator::radio::detail;
struct Atomics {
  template<class T> KOKKOS_INLINE_FUNCTION static T add(T* pointer,T value){
    return Kokkos::atomic_fetch_add(pointer,value);
  }
};
int main(int argc,char** argv){try{
  if(argc!=4)throw std::invalid_argument("usage: coreas_air_reference OUTPUT DISTANCE_M RATE_GHZ");
  std::filesystem::path output=argv[1];double distance=std::stod(argv[2]),rate=std::stod(argv[3])*1.e9;
  if(std::filesystem::exists(output)||!(distance>=100.&&distance<=10000.)||!(rate>=8.e9&&rate<=32.e9))
    throw std::invalid_argument("invalid air reference parameters/output");
  Kokkos::InitializationSettings settings;settings.set_num_threads(256);settings.set_device_id(0);
  Kokkos::ScopeGuard runtime(settings);Space execution;
  ri::Config c;c.enabled=true;c.samples=8192;c.sample_rate_Hz=rate;c.moment_order=16;
  c.maximum_subdivision_depth=20;c.subdivision_frequency_Hz=2.e9;c.fraunhofer_limit=.00625;
  c.propagation.media[0].refractive_index=1.0003;
  c.start_time_s=distance*1.0003/d::light_speed-64.e-9;
  c.observers={{"uniform_air",{.8*distance,0.,.6*distance},0}};
  std::vector<ri::Track> tracks;
  auto add=[&](ri::Vec3 a,ri::Vec3 b,double t,double weight){
    double end=t+d::norm(d::sub(b,a))/(.8*d::light_speed);
    tracks.push_back({a,b,t,end,-1.,weight,tracks.size(),0,0,1});
  };
  add({0.,0.,0.},{.02,.01,1.},0.,1.);
  add({.1,.02,.3},{.2,.03,1.1},.17e-9,.7);
  using Radio=ri::KokkosAccumulator<Space>;Radio::Tracks input("air_comparison_input",tracks.size());
  auto host=Kokkos::create_mirror_view(input);
  for(std::size_t i=0;i<tracks.size();++i)host(i)=tracks[i];
  Kokkos::deep_copy(execution,input,host);execution.fence();
  std::filesystem::create_directories(output);
  for(auto algorithm:{ri::Algorithm::CoREAS,ri::Algorithm::ZHS}){
    c.algorithm=algorithm;Radio radio(c,{},0,3,execution);
    radio.accumulate(input,tracks.size(),execution);
    ri::writeResult(radio.download(execution),(output/(algorithm==ri::Algorithm::CoREAS?"interface_CoREAS":"interface_ZHS")).string());
  }
  using Memory=typename Space::memory_space;
  Kokkos::View<air::RadioTrackKinematics*,Memory> airTracks("air_reference_tracks",tracks.size());
  auto airHost=Kokkos::create_mirror_view(airTracks);
  for(std::size_t i=0;i<tracks.size();++i){
    auto t=tracks[i];air::RadioTrackKinematics a;
    a.start={t.start_m.x,t.start_m.y,t.start_m.z};a.end={t.end_m.x,t.end_m.y,t.end_m.z};
    a.start_time_s=t.start_time_s;a.end_time_s=t.end_time_s;a.duration_s=t.end_time_s-t.start_time_s;
    a.displacement=air::operator-(a.end,a.start);a.track_length_m=d::norm(d::sub(t.end_m,t.start_m));
    auto beta=d::scale(d::sub(t.end_m,t.start_m),1./(d::light_speed*a.duration_s));a.beta={beta.x,beta.y,beta.z};
    a.beta_module=d::norm(beta);a.constant=t.charge_e*air::ElementaryChargeC*t.weight*air::EmConstant;a.valid=1;airHost(i)=a;
  }
  Kokkos::deep_copy(execution,airTracks,airHost);
  Kokkos::View<double*,Memory> x("air_reference_x",c.samples),y("air_reference_y",c.samples),z("air_reference_z",c.samples);
  Kokkos::View<air::DeviceRadioCounters,Memory> counters("air_reference_counters");
  air::DevicePropagation propagation;propagation.homogeneous_refractive_index=1.0003;
  air::DeviceObserver observer;observer.position_m[0]=.8*distance;observer.position_m[2]=.6*distance;
  observer.start_time_s=c.start_time_s;observer.duration_s=(c.samples-1)/rate;
  observer.sample_rate_Hz=rate;observer.number_of_bins=c.samples;
  air::DeviceWaveforms waves;waves.floating_x=x.data();waves.floating_y=y.data();waves.floating_z=z.data();
  Kokkos::parallel_for("unmodified_air_coreas_reference",Kokkos::RangePolicy<Space>(execution,0,tracks.size()),KOKKOS_LAMBDA(std::size_t i){
    air::accumulateCoREAS<Atomics>(airTracks(i),propagation,observer,waves,counters.data());
  });
  auto hx=Kokkos::create_mirror_view(x),hy=Kokkos::create_mirror_view(y),hz=Kokkos::create_mirror_view(z);
  auto hc=Kokkos::create_mirror_view(counters);
  Kokkos::deep_copy(execution,hx,x);Kokkos::deep_copy(execution,hy,y);Kokkos::deep_copy(execution,hz,z);Kokkos::deep_copy(execution,hc,counters);execution.fence();
  if(hc().coreas_contributions!=2*tracks.size()||hc().fixed_point_overflows)throw std::runtime_error("air reference source count");
  std::ofstream field(output/"air_field.csv");field<<std::setprecision(17)<<"time_s,Ex_V_m,Ey_V_m,Ez_V_m\n";
  for(std::size_t i=0;i<c.samples;++i)field<<c.start_time_s+i/rate<<','<<hx(i)<<','<<hy(i)<<','<<hz(i)<<'\n';
  std::ofstream data(output/"tracks.csv");data<<std::setprecision(17)<<"sx,sy,sz,ex,ey,ez,t0,t1,charge,weight,region\n";
  for(auto t:tracks)data<<t.start_m.x<<','<<t.start_m.y<<','<<t.start_m.z<<','<<t.end_m.x<<','<<t.end_m.y<<','<<t.end_m.z<<','<<t.start_time_s<<','<<t.end_time_s<<','<<t.charge_e<<','<<t.weight<<','<<t.region<<'\n';
  std::ofstream metadata(output/"air_reference.json");metadata<<std::setprecision(17)<<"{\"execution\":\""<<Space::name()<<"\",\"distance_m\":"<<distance<<",\"sample_rate_Hz\":"<<rate<<",\"air_endpoint_contributions\":"<<hc().coreas_contributions<<",\"algorithm_header\":\"corsika/accelerator/radio/detail/RadioProjectionStep.hpp\"}\n";
  std::cout<<"PASS unmodified air CoREAS reference "<<Space::name()<<" R="<<distance<<" fs="<<rate<<'\n';return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
