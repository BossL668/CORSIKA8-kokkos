// Read-only real-event ray diagnostics; never changes the production program.
#include <corsika/modules/radio/interface/Propagation.hpp>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>
namespace ri=corsika::radio::interface;
namespace d=ri::detail;
using Json=nlohmann::json;
corsika::terrain::FlatTerrainData loadRadioBvhProbeMesh(char const* path);
ri::Vec3 vec(YAML::Node a){return {a[0].as<double>(),a[1].as<double>(),a[2].as<double>()};}
Json xyz(ri::Vec3 a){return {a.x,a.y,a.z};}

// Independent Moller-Trumbore intersections in long double, scanning every
// physical face. No production BVH, plane ownership or hit routine is used.
struct V {long double x,y,z;};
V cv(ri::Vec3 a){return {a.x,a.y,a.z};}
V sub(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
long double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
Json firstHit(corsika::terrain::FlatTerrainData const& mesh,ri::Vec3 aa,ri::Vec3 bb){
  auto a=cv(aa),b=cv(bb),delta=sub(b,a);auto length=std::sqrt(dot(delta,delta));
  if(length<=2e-7L)return nullptr;
  V direction{delta.x/length,delta.y/length,delta.z/length};
  long double closest=length;unsigned chosen=UINT32_MAX;
  for(unsigned id=0;id<mesh.indexed_triangles.size();++id){
    auto t=mesh.indexed_triangles[id];if(t.normal.z<=0)continue;
    auto o=cv(mesh.vertices[t.vertex[0]]),e1=sub(cv(mesh.vertices[t.vertex[1]]),o),e2=sub(cv(mesh.vertices[t.vertex[2]]),o);
    auto h=cross(direction,e2);auto det=dot(e1,h);if(std::abs(det)<1e-16L)continue;
    auto rel=sub(a,o);auto u=dot(rel,h)/det;auto q=cross(rel,e1);auto v=dot(direction,q)/det;
    if(u<0||v<0||u+v>1)continue;
    auto distance=dot(e2,q)/det;
    if(distance>1e-7L&&distance<length-1e-7L&&distance<closest){closest=distance;chosen=id;}
  }
  if(chosen==UINT32_MAX)return nullptr;
  auto point=d::add(aa,d::scale(d::sub(bb,aa),double(closest/length)));
  return Json{{"face",chosen},{"distance_m",double(closest)},{"point_m",xyz(point)},
    {"remaining_to_endpoint_m",double(length-closest)}};
}

int main(int argc,char** argv){try{
  if(argc!=5)throw std::runtime_error("usage: TransmissionAudit MESH CONFIG QUERY_JSON OUTPUT");
  auto start=std::chrono::steady_clock::now();auto mesh=loadRadioBvhProbeMesh(argv[1]);
  auto c=YAML::LoadFile(argv[2]),queries=YAML::LoadFile(argv[3]);
  auto coverage=corsika::terrain::coverage::fromHeightfield(mesh);
  d::PropagationView v;v.geometry=ri::Geometry::Mesh;v.mesh=mesh.view();v.faces=mesh.triangles.size();
  if(c["dem_coverage_boundary"].as<bool>())v.coverage=coverage.view();
  v.visibility_tolerance=c["visibility_tolerance_m"].as<double>();
  v.edge_tolerance=c["shared_edge_tolerance_m"].as<double>();
  v.integration_samples=c["optical_integration_samples"].as<unsigned>();
  std::vector<ri::IndexSample> tables[2];std::vector<double> breaks[2];
  for(int i=0;i<2;++i){auto a=c["media"][i];auto& m=v.media[i];m.index=a["index"].as<double>();
    m.attenuation_length_m=a["attenuation_length_m"].IsNull()?HUGE_VAL:a["attenuation_length_m"].as<double>();
    m.center=vec(a["center_m"]);m.radius=a["reference_radius_m"].as<double>();
    m.minimum_index=m.maximum_index=m.index;
    for(auto row:a["radial_index"]){tables[i].push_back({row[0].as<double>(),row[1].as<double>()});
      m.minimum_index=std::min(m.minimum_index,tables[i].back().index);m.maximum_index=std::max(m.maximum_index,tables[i].back().index);}
    for(auto x:a["radial_integration_breaks_m"])breaks[i].push_back(x.as<double>());
    m.samples=tables[i].data();m.count=tables[i].size();m.breaks=breaks[i].data();m.break_count=breaks[i].size();
  }
  std::vector<d::TransmissionNormalBounds> bounds(mesh.nodes.size());
  for(std::size_t i=mesh.nodes.size();i-->0;){
    auto& b=bounds[i];b={{HUGE_VAL,HUGE_VAL,HUGE_VAL},{-HUGE_VAL,-HUGE_VAL,-HUGE_VAL}};
    auto add=[&](ri::Vec3 n){b.low={std::min(b.low.x,n.x),std::min(b.low.y,n.y),std::min(b.low.z,n.z)};b.high={std::max(b.high.x,n.x),std::max(b.high.y,n.y),std::max(b.high.z,n.z)};};
    auto const& node=mesh.nodes[i];
    for(unsigned k=0;k<node.count;++k)add(mesh.indexed_triangles[mesh.indices[node.first+k]].normal);
    for(auto child=i+1;child<node.skip;child=mesh.nodes[child].skip){add(bounds[child].low);add(bounds[child].high);}
  }
  v.transmission_normals=bounds.data();Json results=Json::array();bool passed=true;
  for(auto input:queries){
    ri::RayQuery q{vec(input["source_m"]),vec(input["observer_m"]),1,0};
    std::vector<unsigned char> mask(v.faces);d::CandidateFaces iterator{v,q,false};unsigned face;
    while(iterator.next(face))mask[face]=1;
    std::vector<unsigned> fastValid,allValid,rawValid,fastBlocked,allBlocked,rawBlocked;
    unsigned fastError=0,allError=0,rawError=0,sideEligible=0,projectionKept=0;
    Json rays=Json::array();unsigned visibilityMismatch=0;
    for(unsigned id=0;id<v.faces;++id){
      auto const& t=mesh.indexed_triangles[id];auto point=mesh.vertices[t.vertex[0]];
      if(t.normal.z>0){double ds=d::dot(d::sub(point,q.source),t.normal),dt=d::dot(d::sub(q.observer,point),t.normal);
        if(ds>1e-10&&dt>1e-10){++sideEligible;
          if(d::projectedSegmentMayHit(v,id,d::add(q.source,d::scale(t.normal,ds)),d::sub(q.observer,d::scale(t.normal,dt))))++projectionKept;}}
      auto filtered=d::transmittedPath(v,q,id,false,true);
      auto raw=d::transmittedPath(v,q,id,false,false);
      auto accept=[&](ri::Path const& p,std::vector<unsigned>& valid,std::vector<unsigned>& blocked,unsigned& error){
        if(p.status==ri::PathStatus::Valid)valid.push_back(id);
        else if(p.status==ri::PathStatus::Blocked)blocked.push_back(id);
        else if(p.status==ri::PathStatus::Invalid||p.status==ri::PathStatus::Unconverged)++error;
      };
      if(mask[id])accept(filtered,fastValid,fastBlocked,fastError);
      accept(filtered,allValid,allBlocked,allError);accept(raw,rawValid,rawBlocked,rawError);
      if(raw.status==ri::PathStatus::Valid||raw.status==ri::PathStatus::Blocked){
        bool sourceClear=d::segmentClear(v,q.source,raw.interface_point),destClear=d::segmentClear(v,raw.interface_point,q.observer);
        auto sourceHit=firstHit(mesh,q.source,raw.interface_point),destHit=firstHit(mesh,raw.interface_point,q.observer);
        if(sourceClear!=sourceHit.is_null()||destClear!=destHit.is_null())++visibilityMismatch;
        rays.push_back(Json{{"face",id},{"status",unsigned(raw.status)},{"interface_m",xyz(raw.interface_point)},
          {"source_clear_native",sourceClear},{"destination_clear_native",destClear},
          {"source_first_hit_independent",sourceHit},{"destination_first_hit_independent",destHit},
          {"snell_residual",raw.snell_residual},{"attenuation",raw.attenuation}});
      }
    }
    bool agree=fastValid==allValid&&allValid==rawValid&&fastBlocked==allBlocked&&allBlocked==rawBlocked;
    passed=passed&&agree&&fastError==0&&allError==0&&rawError==0&&visibilityMismatch==0;
    results.push_back(Json{{"label",input["label"].as<std::string>()},{"source_id",input["source_id"].as<std::string>()},
      {"observer",input["observer"].as<std::string>()},{"source_m",xyz(q.source)},{"observer_m",xyz(q.observer)},
      {"candidate_faces",std::count(mask.begin(),mask.end(),1)},{"same_side_rejection_passed",sideEligible},
      {"projected_segment_filter_passed",projectionKept},
      {"filtered_valid",fastValid.size()},{"exhaustive_valid",allValid.size()},{"unfiltered_valid",rawValid.size()},
      {"filtered_blocked",fastBlocked.size()},{"exhaustive_blocked",allBlocked.size()},{"unfiltered_blocked",rawBlocked.size()},
      {"invalid_or_unconverged",Json::array({fastError,allError,rawError})},{"candidate_filters_agree",agree},
      {"independent_visibility_mismatches",visibilityMismatch},{"stationary_rays",rays}});
    std::cout<<input["label"].as<std::string>()<<" valid="<<rawValid.size()<<" blocked="<<rawBlocked.size()<<" agree="<<agree<<" visibility_mismatch="<<visibilityMismatch<<std::endl;
    Json report{{"passed",passed},{"mesh_faces",v.faces},{"rows",results},
      {"elapsed_s",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},
      {"scope","Saved source points. Native BVH vs exhaustive vs disabled projected-face/ownership filters. Independent long-double full-mesh segment intersections. Not a replay of all shower segments."}};
    std::ofstream(argv[4])<<report.dump(2)<<'\n';
  }
  return passed?0:1;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 2;}}
