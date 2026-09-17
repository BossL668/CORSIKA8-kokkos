// Standalone exported-target consumer and fixed-source CoREAS fixtures.
#include <Kokkos_Core.hpp>
#include <corsika/modules/radio/interface/KokkosAccumulator.hpp>
#include <corsika/modules/radio/interface/Output.hpp>
#include <corsika/geometry/terrain/KokkosTerrainSession.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <cstdlib>
#if defined(RADIO_CONSUMER_CUDA)
using Space=Kokkos::Cuda;
#else
using Space=Kokkos::OpenMP;
#endif
namespace ri=corsika::radio::interface;
namespace d=ri::detail;
using Radio=ri::KokkosAccumulator<Space>;
void require(bool p,char const* s){if(!p)throw std::runtime_error(s);}
corsika::terrain::FlatTerrainData cube(){
  corsika::terrain::FlatTerrainData r;
  r.vertices={{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
  int faces[][3]={{0,2,1},{0,3,2},{4,5,6},{4,6,7},{0,1,5},{0,5,4},{1,2,6},{1,6,5},{2,3,7},{2,7,6},{3,0,4},{3,4,7}};
  for(auto const& f:faces){auto a=r.vertices[f[0]],e1=d::sub(r.vertices[f[1]],a),e2=d::sub(r.vertices[f[2]],a),n=d::unit(d::cross(e1,e2));
    r.triangles.push_back({a,e1,e2,n});r.indexed_triangles.push_back({{unsigned(f[0]),unsigned(f[1]),unsigned(f[2])},n});r.indices.push_back(r.indices.size());}
  r.nodes.push_back({{-1,-1,-1},{1,1,1},1,0,12});return r;
}
ri::Result accumulate(ri::Config const& c,std::vector<ri::Track> const& tracks,
    corsika::terrain::flat::View mesh,std::size_t faces,Space const& execution,bool cpu){
  Radio radio(c,mesh,faces,3,execution);
  if(cpu)radio.accumulateCpu(tracks,execution);
  else{
    Radio::Tracks input("coreas_fixture_tracks",tracks.size());auto h=Kokkos::create_mirror_view(input);
    for(std::size_t i=0;i<tracks.size();++i)h(i)=tracks[i];
    Kokkos::deep_copy(execution,input,h);execution.fence();radio.accumulate(input,tracks.size(),execution);
  }
  auto result=radio.download(execution);
  require(result.statistics.downloads==1,"finalization count");
  require((cpu?result.statistics.cpu_tracks:result.statistics.device_tracks)==tracks.size(),"source accounting");
  bool rejected=false;try{radio.download(execution);}catch(std::logic_error const&){rejected=true;}
  require(rejected,"second download accepted");return result;
}
int main(int argc,char** argv){try{
  if(argc!=4&&argc!=5)throw std::invalid_argument("usage: coreas_fixture OUTPUT KIND CoREAS|ZHS [MOMENT_ORDER]");
  std::string output=argv[1],kind=argv[2],algorithm=argv[3];
  require(algorithm=="CoREAS"||algorithm=="ZHS","algorithm");
  Kokkos::InitializationSettings settings;settings.set_num_threads(256);settings.set_device_id(0);
  Kokkos::ScopeGuard runtime(settings);Space execution;
  ri::Config c;c.enabled=true;c.algorithm=algorithm=="CoREAS"?ri::Algorithm::CoREAS:ri::Algorithm::ZHS;
  c.samples=2048;c.start_time_s=-1.e-7;c.moment_order=16;c.maximum_subdivision_depth=20;
  if(argc==5)c.moment_order=std::stoul(argv[4]);
  c.subdivision_frequency_Hz=2.e9;c.fraunhofer_limit=.00625;c.mesh_maximum_segment_m=.025;
  // Keep the high-accuracy benchmark as the default; optionally exercise the
  // actual high-energy run resolution against the same independent oracle.
  if(std::getenv("C8_RADIO_PRODUCTION_RESOLUTION")){
    c.sample_rate_Hz=.256e9;c.moment_order=12;c.subdivision_frequency_Hz=.256e9;
    c.fraunhofer_limit=.025;c.mesh_maximum_segment_m=.1;
  }
  c.propagation.media[0].refractive_index=1.33;c.propagation.media[1].refractive_index=1.7;
  c.observers={{"receiver",{10.,3.,20.},0}};
  std::vector<ri::Track> tracks;
  auto add=[&](ri::Vec3 a,ri::Vec3 b,double time,double beta,unsigned region,double charge=-1.,double weight=1.){
    double end=time+d::norm(d::sub(b,a))/(beta*d::light_speed);
    tracks.push_back({a,b,time,end,charge,weight,tracks.size()+1,0,region,1});return end;
  };
  auto mesh=cube();std::unique_ptr<corsika::terrain::KokkosTerrainSession<Space>> geometry;
  corsika::terrain::flat::View view{};std::size_t faces=0;
  if(kind=="uniform"||kind=="radial"){
    add({0.,0.,-.2},{.01,.02,.3},0.,.7,0);add({.02,.01,-.1},{.015,.02,.2},1.e-9,.6,0,1.,2.);
    if(kind=="radial"){
      auto& m=c.propagation.media[0];m.center_m={0.,0.,-100.};m.reference_radius_m=100.;
      m.radial_index={{-1.,1.32},{100.,2.33}};c.propagation.optical_integration_samples=1024;
    }
  }else if(kind=="negative"||kind=="cherenkov"||kind=="vacuum_forward"){
    c.propagation.media[0].refractive_index=kind=="vacuum_forward"?1.:1.5;
    c.observers[0].position_m={.01,0.,20.};double beta=kind=="vacuum_forward"?1.-1.e-8:.99;
    if(kind=="cherenkov")c.observers[0].position_m={20.,0.,20./std::sqrt(1.5*1.5*beta*beta-1.)};
    add({0.,0.,-.02},{0.,0.,.02},0.,beta,0);
  }else if(kind=="plane"||kind=="plane_reverse"){
    c.propagation.geometry=ri::Geometry::Plane;
    if(kind=="plane")add({0.,0.,-2.},{.02,0.,-1.7},0.,.7,1);
    else {c.observers[0].position_m={10.,3.,-20.};c.observers[0].region=1;add({0.,0.,2.},{.02,0.,1.7},0.,.7,0);}
  }else if(kind=="boundary"||kind=="matched_boundary"||kind=="matched_unsplit"||kind=="mesh_boundary"||kind=="boundary_reverse"||kind=="matched_boundary_reverse"||kind=="matched_unsplit_reverse"||kind=="boundary_grazing"){
    c.propagation.geometry=ri::Geometry::Plane;
    if(kind=="matched_boundary"||kind=="matched_unsplit"||kind=="matched_boundary_reverse"||kind=="matched_unsplit_reverse")c.propagation.media[1]=c.propagation.media[0];
    double z=0.;
    if(kind=="mesh_boundary"){
      c.propagation.geometry=ri::Geometry::Mesh;geometry=std::make_unique<corsika::terrain::KokkosTerrainSession<Space>>(mesh,1);
      view=geometry->deviceView();faces=mesh.triangles.size();z=1.;c.observers[0].position_m={.5,.25,10.};
    }
    if(kind=="boundary_reverse"||kind=="matched_boundary_reverse"||kind=="matched_unsplit_reverse"||kind=="boundary_grazing"){
      c.observers[0].position_m={10.,3.,-20.};c.observers[0].region=1;
      if(kind=="boundary_grazing"){c.observers[0].position_m={20.*1.33/std::sqrt(1.7*1.7-1.33*1.33),0.,-20.};c.fraunhofer_limit=1.e-5;}
      if(kind=="matched_unsplit_reverse"){
        c.propagation.geometry=ri::Geometry::Uniform;add({0.,0.,.2},{0.,0.,-.2},0.,.5,1);
      }else{double time=add({0.,0.,.2},{0.,0.,0.},0.,.5,0);add({0.,0.,0.},{0.,0.,-.2},time,.5,1);}
    }else if(kind=="matched_unsplit"){
      c.propagation.geometry=ri::Geometry::Uniform;add({0.,0.,-.2},{0.,0.,.2},0.,.5,0);
    }else{double time=add({0.,0.,z-.2},{0.,0.,z},0.,.5,1);add({0.,0.,z},{0.,0.,z+.2},time,.5,0);}
  }else if(kind=="shadow"){
    c.propagation.geometry=ri::Geometry::Mesh;geometry=std::make_unique<corsika::terrain::KokkosTerrainSession<Space>>(mesh,1);
    view=geometry->deviceView();faces=mesh.triangles.size();c.observers[0].position_m={0.,0.,-3.};
    add({-6.,0.,2.},{6.,0.,2.},0.,.4,0);
  }else if(kind=="curved"||kind=="curved_reverse"){
    double time=0.,sign=kind=="curved"?1.:-1.;
    for(unsigned i=0;i<32;++i){double a=-.2+.4*i/32,b=-.2+.4*(i+1)/32;
      time=add({sign*.5*std::sin(a),0.,.5*std::cos(a)},{sign*.5*std::sin(b),0.,.5*std::cos(b)},time,.8,0);}
  }else throw std::invalid_argument("unknown CoREAS fixture");
  auto result=accumulate(c,tracks,view,faces,execution,false),cpu=accumulate(c,tracks,view,faces,execution,true);
  auto field=ri::render(result),cpuField=ri::render(cpu);long double error=0.,norm=0.;
  for(std::size_t i=0;i<field.field_spectrum.size();++i){error+=std::norm(field.field_spectrum[i]-cpuField.field_spectrum[i]);norm+=std::norm(field.field_spectrum[i]);}
  require(norm>0.&&std::sqrt(error/norm)<1.e-10,"CPU/device source mismatch");
  if(c.algorithm==ri::Algorithm::CoREAS){
    if(kind=="cherenkov"||kind=="vacuum_forward")require(result.statistics.regularized_pairs>0,"Cherenkov limit not exercised");
    else require(result.statistics.endpoint_contributions>0,"endpoint algorithm not exercised");
    if(kind=="boundary"||kind=="matched_boundary"||kind=="mesh_boundary"||kind=="boundary_reverse"||kind=="matched_boundary_reverse"||kind=="boundary_grazing")require(result.statistics.boundary_endpoints==1,"one-sided interface endpoint ownership");
  }
  ri::writeResult(result,output);
  std::ofstream data(output+"/tracks.csv");data<<std::setprecision(17)<<"sx,sy,sz,ex,ey,ez,t0,t1,charge,weight,region\n";
  for(auto t:tracks)data<<t.start_m.x<<','<<t.start_m.y<<','<<t.start_m.z<<','<<t.end_m.x<<','<<t.end_m.y<<','<<t.end_m.z<<','<<t.start_time_s<<','<<t.end_time_s<<','<<t.charge_e<<','<<t.weight<<','<<t.region<<'\n';
  std::ofstream stats(output+"/statistics.json");auto s=result.statistics;
  stats<<"{\"device_tracks\":"<<s.device_tracks<<",\"endpoint_contributions\":"<<s.endpoint_contributions<<",\"regularized_pairs\":"<<s.regularized_pairs<<",\"boundary_endpoints\":"<<s.boundary_endpoints<<",\"device_bytes\":"<<s.device_bytes<<"}\n";
  std::cout<<"PASS interface CoREAS fixture "<<Space::name()<<' '<<kind<<' '<<algorithm<<" endpoints="<<s.endpoint_contributions<<" regularized="<<s.regularized_pairs<<" boundaries="<<s.boundary_endpoints<<'\n';
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
