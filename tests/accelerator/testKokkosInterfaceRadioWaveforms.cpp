#include <Kokkos_Core.hpp>
#include <corsika/modules/radio/interface/KokkosAccumulator.hpp>
#include <corsika/modules/radio/interface/Output.hpp>
#include <corsika/modules/transport/kokkos/ExecutionSpace.hpp>
#include <corsika/geometry/terrain/KokkosTerrainSession.hpp>
#include <memory>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdlib>
namespace ri=corsika::radio::interface;
namespace d=ri::detail;
using Space=corsika::interfaces::kokkos::ExecutionSpace;
void require(bool p,char const* s){if(!p)throw std::runtime_error(s);}
corsika::terrain::FlatTerrainData cube(){
  corsika::terrain::FlatTerrainData r;
  r.vertices={{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
  int faces[][3]={{0,2,1},{0,3,2},{4,5,6},{4,6,7},{0,1,5},{0,5,4},{1,2,6},{1,6,5},{2,3,7},{2,7,6},{3,0,4},{3,4,7}};
  for(auto const& f:faces){auto a=r.vertices[f[0]],e1=d::sub(r.vertices[f[1]],a),e2=d::sub(r.vertices[f[2]],a),n=d::unit(d::cross(e1,e2));
    r.triangles.push_back({a,e1,e2,n});r.indexed_triangles.push_back({{unsigned(f[0]),unsigned(f[1]),unsigned(f[2])},n});r.indices.push_back(r.indices.size());}
  r.nodes.push_back({{-1,-1,-1},{1,1,1},1,0,12});return r;
}
int main(int argc,char** argv){try{
  if(argc!=5)throw std::invalid_argument("usage: testKokkosInterfaceRadioWaveforms PREFIX uniform|plane|mesh_shadow|mesh_transmitted ORDER LIMIT");
  std::string prefix=argv[1],kind=argv[2];
  require(kind=="uniform"||kind=="radial"||kind=="plane"||kind=="mesh_shadow"||kind=="mesh_transmitted","unknown fixture");
  bool homogeneous=kind=="uniform"||kind=="radial";
  int threads=256;if(auto p=std::getenv("C8_INTERFACE_TEST_THREADS"))threads=std::stoi(p);
  Kokkos::InitializationSettings settings;settings.set_num_threads(threads);settings.set_device_id(0);
  Kokkos::ScopeGuard guard(settings);Space execution;
  ri::Config c;c.enabled=true;c.samples=4096;c.start_time_s=0.;c.moment_order=std::stoul(argv[3]);
  c.fraunhofer_limit=std::stod(argv[4]);c.subdivision_frequency_Hz=2.e9;
  c.mesh_maximum_segment_m=4*c.fraunhofer_limit;c.maximum_subdivision_depth=20;
  c.propagation.geometry=homogeneous?ri::Geometry::Uniform:ri::Geometry::Plane;
  c.propagation.media[0].refractive_index=homogeneous?1.5:1.;
  c.propagation.media[1].refractive_index=2.;c.propagation.media[1].attenuation_length_m=100.;
  if(homogeneous){
    c.observers={{"normal",{20.,0.,0.},0},{"near_cherenkov",{20.,0.,20./std::sqrt(1.5*1.5*.99*.99-1.)},0},
      {"negative_denominator",{.01,0.,20.},0}};
  }else c.observers={{"normal",{0.,0.,20.},0},{"inclined",{20.,0.,20.},0},{"oblique",{80.,10.,20.},0}};
  std::vector<ri::Track> tracks;
  for(int i=0;i<8;++i){
    double z=homogeneous?-.2:-2.2;
    double length=.2+i*.05,dt=length/(.99*d::light_speed);
    ri::Vec3 a{.003*i,.005*i,z},b{a.x+.02,a.y,z+length};
    dt=d::norm(d::sub(b,a))/(.99*d::light_speed);
    tracks.push_back({a,b,i*2.e-11,i*2.e-11+dt,i==2?1.:-1.,1.+.2*i,unsigned(i+1),0,unsigned(homogeneous?0:1),1});
  }
  if(kind=="radial"){
    auto& medium=c.propagation.media[0];medium.center_m={0.,0.,-100.};medium.reference_radius_m=100.;
    medium.radial_index={{-1.,1.49},{100.,2.5}};c.propagation.optical_integration_samples=1024;
  }
  auto mesh=cube();std::unique_ptr<corsika::terrain::KokkosTerrainSession<Space>> geometry;
  corsika::terrain::flat::View meshView{};std::size_t faces=0;
  if(kind=="mesh_shadow"||kind=="mesh_transmitted"){
    c.propagation.geometry=ri::Geometry::Mesh;
    geometry=std::make_unique<corsika::terrain::KokkosTerrainSession<Space>>(mesh,1);
    meshView=geometry->deviceView();faces=mesh.triangles.size();
    if(kind=="mesh_shadow"){
      c.observers={{"partially_blocked",{0.,0.,-3.},0}};
      tracks={{{-6.,0.,2.},{6.,0.,2.},0.,12./(.99*d::light_speed),-1.,1.,1,0,0,1}};
    }else{
      c.observers={{"normal",{0.,0.,5.},0},{"inclined",{.8,0.,5.},0},{"oblique",{.8,.8,5.},0}};
      for(auto& t:tracks){t.start_m.z+=2.;t.end_m.z+=2.;}
    }
  }
  using Tracks=ri::KokkosAccumulator<Space>::Tracks;
  Tracks input("interface_radio_fixture_tracks",tracks.size());auto host=Kokkos::create_mirror_view(input);
  for(std::size_t i=0;i<tracks.size();++i)host(i)=tracks[i];Kokkos::deep_copy(execution,input,host);
  ri::KokkosAccumulator<Space> accumulator(c,meshView,faces,64,execution);
  accumulator.accumulate(input,tracks.size(),execution);
  auto result=accumulator.download(execution);
  // Exercise the public renderer with a nonzero absolute time origin and a
  // name requiring JSON escaping. The independent Python oracle reads this.
  auto renderedResult=result;renderedResult.config.start_time_s=-1.e-6;
  renderedResult.config.observers[0].name="normal\n\"quoted\"";
  ri::writeResult(renderedResult,prefix+"_rendered");
  require(result.statistics.device_tracks==tracks.size()&&result.statistics.cpu_tracks==0,"device source accounting");
  require(result.statistics.paths>0&&result.statistics.downloads==1,"empty/double radio result");
  bool rejected=false;try{accumulator.download(execution);}catch(std::logic_error const&){rejected=true;}
  require(rejected,"duplicate radio download accepted");
  ri::KokkosAccumulator<Space> cpu(c,meshView,faces,3,execution);
  cpu.accumulateCpu(tracks,execution);auto cpuResult=cpu.download(execution);
  require(cpuResult.statistics.cpu_tracks==tracks.size()&&cpuResult.statistics.device_tracks==0,"CPU source accounting");
  double difference=0.,peak=0.;
  for(std::size_t i=0;i<result.moments.size();++i){difference=std::max(difference,std::abs(cpuResult.moments[i]-result.moments[i]));peak=std::max(peak,std::abs(result.moments[i]));}
  require(peak>0.&&difference/peak<2.e-13,"CPU-upload/device-input waveform mismatch");
  if(kind=="uniform"){
    auto invalid=c;invalid.maximum_device_bytes=1;
    bool caught=false;try{ri::projectedRadioBytes(invalid,64);}catch(std::length_error const&){caught=true;}
    require(caught,"memory budget was not enforced");
    invalid=c;invalid.samples=65;caught=false;
    try{ri::projectedRadioBytes(invalid,64);}catch(std::invalid_argument const&){caught=true;}
    require(caught,"odd sample count was accepted");
    for(unsigned mode=0;mode<3;++mode){
      auto bounded=c;bounded.observers.resize(1);bounded.samples=128;bounded.moment_order=0;
      auto track=tracks.front();unsigned error=0;
      if(mode==0){bounded.start_time_s=1.;error=2;}
      if(mode==1){bounded.maximum_subdivision_depth=0;bounded.fraunhofer_limit=1.e-10;error=6;}
      if(mode==2){track.end_time_s=track.start_time_s+1.e-15;error=3;}
      ri::KokkosAccumulator<Space> probe(bounded,{},0,1,execution);probe.accumulateCpu({track},execution);
      caught=false;try{probe.download(execution);}catch(std::runtime_error const& e){caught=std::string(e.what()).find("device error "+std::to_string(error))!=std::string::npos;}
      require(caught,"invalid/incomplete radio input silently accepted");
    }
  }
  std::ofstream binary(prefix+".bin",std::ios::binary);binary.write(reinterpret_cast<char const*>(result.moments.data()),result.moments.size()*sizeof(double));
  require(bool(binary),"moment write failed");
  std::ofstream tsv(prefix+"_tracks.csv");tsv<<std::setprecision(17)<<"x0,y0,z0,x1,y1,z1,t0,t1,charge,weight,region\n";
  for(auto t:tracks)tsv<<t.start_m.x<<','<<t.start_m.y<<','<<t.start_m.z<<','<<t.end_m.x<<','<<t.end_m.y<<','<<t.end_m.z<<','<<t.start_time_s<<','<<t.end_time_s<<','<<t.charge_e<<','<<t.weight<<','<<t.region<<'\n';
  std::ofstream meta(prefix+".json");meta<<std::setprecision(17)<<"{\"kind\":\""<<kind<<"\",\"execution\":\""<<Space::name()<<"\",\"concurrency\":"<<execution.concurrency()
    <<",\"samples\":"<<c.samples<<",\"sample_rate\":"<<c.sample_rate_Hz<<",\"order\":"<<c.moment_order<<",\"limit\":"<<c.fraunhofer_limit
    <<",\"paths\":"<<result.statistics.paths<<",\"leaves\":"<<result.statistics.leaves<<",\"cpu_device_input_relative_difference\":"<<difference/peak
    <<",\"n0\":"<<c.propagation.media[0].refractive_index<<",\"n1\":"<<c.propagation.media[1].refractive_index<<",\"attenuation1\":100,\"observers\":[";
  for(std::size_t i=0;i<c.observers.size();++i){auto o=c.observers[i];if(i)meta<<',';meta<<'['<<o.position_m.x<<','<<o.position_m.y<<','<<o.position_m.z<<']';}meta<<"]}\n";
  std::cout<<"PASS interface waveform "<<kind<<" execution="<<Space::name()<<" paths="<<result.statistics.paths<<" leaves="<<result.statistics.leaves<<" input_difference="<<difference/peak<<'\n';return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
