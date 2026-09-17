// Real DEM property test: conservative candidate traversal versus every face.
// Both directions, shallow/deep sources, and the actual radial optical table.
#include <Kokkos_Core.hpp>
#include <corsika/geometry/terrain/KokkosTerrainSession.hpp>
#include <corsika/modules/radio/interface/KokkosPropagation.hpp>
#include <corsika/modules/transport/kokkos/ExecutionSpace.hpp>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <fstream>
#include <iostream>
namespace ri=corsika::radio::interface;
namespace d=ri::detail;
using Space=corsika::interfaces::kokkos::ExecutionSpace;
using Memory=Space::memory_space;
corsika::terrain::FlatTerrainData loadRadioBvhProbeMesh(char const* path);
struct Counts {unsigned kept{},valid{},blocked{},missed{},duplicates{},invalid{},invalid_kept{},first_bad{UINT32_MAX};ri::Path first_bad_path;};
struct Probe {
  d::PropagationView propagation;
  Kokkos::View<ri::RayQuery*,Memory> queries;
  Kokkos::View<unsigned char**,Kokkos::LayoutRight,Memory> mask;
  Kokkos::View<Counts*,Memory> result;
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t i)const {
    auto q=queries(i);Counts r;d::CandidateFaces candidates{propagation,q,false};std::uint32_t id;
    while(candidates.next(id)){
      if(id>=propagation.faces){++r.invalid;continue;}
      if(mask(i,id))++r.duplicates;
      mask(i,id)=1;++r.kept;
    }
    for(unsigned face=0;face<propagation.faces;++face){
      auto p=d::transmittedPath(propagation,q,face);
      if(p.status==ri::PathStatus::Valid){++r.valid;if(!mask(i,face))++r.missed;}
      if(p.status==ri::PathStatus::Blocked){++r.blocked;if(!mask(i,face))++r.missed;}
      if(p.status==ri::PathStatus::Invalid||p.status==ri::PathStatus::Unconverged){++r.invalid;r.invalid_kept+=mask(i,face)!=0;if(r.first_bad==UINT32_MAX){r.first_bad=face;r.first_bad_path=p;}}
    }
    result(i)=r;
  }
};
int main(int argc,char** argv){try{
  using namespace corsika;
  if(argc!=4)throw std::invalid_argument("usage: testKokkosInterfaceRadioBvh MESH CONFIG_JSON OUTPUT_JSON");
  auto flat=loadRadioBvhProbeMesh(argv[1]);
  auto input=YAML::LoadFile(argv[2]);ri::PropagationConfig c;c.geometry=ri::Geometry::Mesh;
  for(unsigned i=0;i<2;++i){auto n=input["media"][i];auto& m=c.media[i];
    m.refractive_index=n["index"].as<double>();
    if(n["attenuation_length_m"]&&!n["attenuation_length_m"].IsNull())m.attenuation_length_m=n["attenuation_length_m"].as<double>();
    m.center_m={n["center_m"][0].as<double>(),n["center_m"][1].as<double>(),n["center_m"][2].as<double>()};
    m.reference_radius_m=n["reference_radius_m"].as<double>();
    for(auto row:n["radial_index"])m.radial_index.push_back({row[0].as<double>(),row[1].as<double>()});
    for(auto row:n["radial_integration_breaks_m"])m.radial_integration_breaks_m.push_back(row.as<double>());
  }
  Kokkos::InitializationSettings settings;settings.set_num_threads(256);settings.set_device_id(0);
  Kokkos::ScopeGuard runtime(settings);Space execution;
  terrain::KokkosTerrainSession<Space> geometry(flat,1);
  ri::KokkosPropagation<Space> optics(c,geometry.deviceView(),flat.triangles.size(),execution);
  constexpr unsigned count=256;
  Kokkos::View<ri::RayQuery*,Memory> queries("bvh_probe_queries",count);
  auto host=Kokkos::create_mirror_view(queries);
  for(unsigned i=0;i<count;++i){
    unsigned face=(std::uint64_t(i)*11939)%flat.triangles.size();
    while(flat.triangles[face].normal.z<=.2)face=(face+1)%flat.triangles.size();
    auto t=flat.triangles[face];auto centroid=d::add(t.origin,d::scale(d::add(t.edge1,t.edge2),1./3.));
    double depth=std::pow(10.,-6.+(i%29)*(9./28.));
    auto source=d::sub(centroid,d::scale(t.normal,depth));
    ri::Vec3 observer=i%3==0?ri::Vec3{1677.6818310890271,-2030.0016098328992,-114.16758852890496}:
      (i%3==1?d::add(centroid,d::scale(t.normal,1000.)):ri::Vec3{-1160.5292565842756,537.1068996418721,-100.32429011659633});
    if(i<32)source={2516.5354,3016.549+(int(i)-16)*50.,500.5295};
    host(i)=i%2?ri::RayQuery{observer,source,0,1}:ri::RayQuery{source,observer,1,0};
  }
  Kokkos::deep_copy(execution,queries,host);
  Kokkos::View<unsigned char**,Kokkos::LayoutRight,Memory> mask("bvh_probe_candidate_mask",count,flat.triangles.size());
  Kokkos::View<Counts*,Memory> result("bvh_probe_counts",count);
  auto start=std::chrono::steady_clock::now();
  Kokkos::parallel_for("bvh_probe_against_exhaustive_faces",Kokkos::RangePolicy<Space>(execution,0,count),Probe{optics.view(),queries,mask,result});
  auto counts=Kokkos::create_mirror_view(result);Kokkos::deep_copy(execution,counts,result);execution.fence();
  std::uint64_t kept=0,valid=0,blocked=0,missed=0,duplicates=0,invalid=0;
  for(unsigned i=0;i<count;++i){auto r=counts(i);kept+=r.kept;valid+=r.valid;blocked+=r.blocked;missed+=r.missed;duplicates+=r.duplicates;invalid+=r.invalid;}
  auto hostOptics=ri::hostPropagation(c,flat.view(),flat.triangles.size());
  unsigned printed=0;
  for(unsigned i=0;i<count&&printed<8;++i){auto r=counts(i);if(!r.invalid)continue;++printed;
    auto q=host(i);auto p=r.first_bad_path;auto t=flat.triangles[r.first_bad];
    auto normal=q.source_region==1?t.normal:d::scale(t.normal,-1.);
    std::cerr<<"INVALID query="<<i<<" face="<<r.first_bad<<" status="<<unsigned(p.status)<<" iterations="<<p.iterations
      <<" invalid_kept="<<r.invalid_kept<<" source="<<q.source.x<<","<<q.source.y<<","<<q.source.z
      <<" observer="<<q.observer.x<<","<<q.observer.y<<","<<q.observer.z<<" regions="<<q.source_region<<","<<q.observer_region
      <<" d1="<<d::dot(d::sub(t.origin,q.source),normal)<<" d2="<<d::dot(d::sub(q.observer,t.origin),normal)
      <<" point="<<p.interface_point.x<<","<<p.interface_point.y<<","<<p.interface_point.z
      <<" point_inside="<<d::containsTriangle(hostOptics,r.first_bad,p.interface_point)<<" residual="<<p.snell_residual<<"\n";
  }
  bool passed=valid>0&&blocked>0&&missed==0&&duplicates==0&&invalid==0;
  std::ofstream out(argv[3]);out<<"{\"passed\":"<<(passed?"true":"false")<<",\"execution\":\""<<Space::name()<<"\",\"queries\":"<<count
    <<",\"faces\":"<<flat.triangles.size()<<",\"exhaustive_candidates\":"<<std::uint64_t(count)*flat.triangles.size()
    <<",\"retained_candidates\":"<<kept<<",\"valid_paths\":"<<valid<<",\"blocked_stationary_paths\":"<<blocked
    <<",\"missed_paths\":"<<missed<<",\"duplicates\":"<<duplicates<<",\"invalid\":"<<invalid
    <<",\"wall_s\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
  std::cout<<"BVH "<<Space::name()<<" passed="<<passed<<" kept="<<kept<<" valid="<<valid<<" blocked="<<blocked<<" missed="<<missed<<" invalid="<<invalid<<"\n";
  return passed?0:1;
}catch(std::exception const& e){std::cerr<<e.what()<<"\n";return 1;}}
