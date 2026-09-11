// Independent geometry oracle: actual Kokkos execution, NOT a shower backend.
#include <corsika/geometry/terrain/FlatTerrainExport.hpp>
#include <corsika/geometry/terrain/TerrainBoundary.hpp>
#include <CLI/CLI.hpp>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace corsika;
using namespace corsika::units::si;
namespace flat=corsika::terrain::flat;
int main(int argc,char** argv) {
  CLI::App app{"Terrain Kokkos geometry oracle (no transport/radio claim)"};
  std::string path,output;int samples=100000,threads=4,device=0,repeats=1,batch=4096;
  bool logical=false;int sharedVertex=-1;
  app.add_option("--mesh",path)->required()->check(CLI::ExistingFile);
  app.add_option("--output",output)->required();
  app.add_option("--samples",samples)->check(CLI::Range(1,1000000));
  app.add_option("--threads",threads)->check(CLI::Range(1,256));
  app.add_option("--device",device)->check(CLI::NonNegativeNumber);
  app.add_flag("--logical-boundary",logical,"Compare CPU logical-side terrain tracking candidates");
  app.add_option("--batch-size",batch)->check(CLI::Range(1,1000000));
  app.add_option("--repeats",repeats)->check(CLI::Range(1,100));
  app.add_option("--shared-vertex-id",sharedVertex,"Add six original-vertex rays; require identical scalar/device distance and face")
      ->check(CLI::NonNegativeNumber);
  CLI11_PARSE(app,argc,argv);
  try {
    if(std::filesystem::exists(output))throw std::invalid_argument("refuse overwrite output");
    logging::set_level(logging::level::warn);
    auto cs=get_root_CoordinateSystem();auto mesh=terrain::loadValidatedTerrain(path,cs);
    auto data=terrain::exportFlatTerrain(mesh);
    std::mt19937_64 generator(20260908);std::uniform_real_distribution<double> unit(0.,1.);
    auto low=mesh.getBounds().getMin(),high=mesh.getBounds().getMax();
    double lo[3]={low.getX(cs)/1_m,low.getY(cs)/1_m,low.getZ(cs)/1_m};
    double hi[3]={high.getX(cs)/1_m,high.getY(cs)/1_m,high.getZ(cs)/1_m};
    std::vector<flat::Ray> rays;std::vector<flat::Hit> oracle,indexedOracle;
    for(int i=0;i<samples;++i) {
      double p[3];for(int j=0;j<3;++j)p[j]=lo[j]+(hi[j]-lo[j])*(1.4*unit(generator)-.2);
      DirectionVector dir(cs,{2.*unit(generator)-1.,2.*unit(generator)-1.,2.*unit(generator)-1.});
      if(i%10==0)dir=DirectionVector(cs,{0.,0.,i%20==0?1.:-1.});
      dir=dir.normalized();Point origin(cs,p[0]*1_m,p[1]*1_m,p[2]*1_m);
      int orientation=logical?(i%2==0?1:-1):(i%3==0?0:i%3==1?1:-1);
      auto hit=mesh.intersectRay(origin,dir);
      if(orientation) {
        hit=terrain::MeshRayHit{};
        auto const queryOrigin=logical?origin-1.e-8*1_m*dir:origin;
        for(auto const& candidate:mesh.intersectRayAll(queryOrigin,dir))
          if(orientation*candidate.normal.dot(dir)>1.e-12){hit=candidate;break;}
      }
      if(logical&&hit.hit)hit.distance=std::max(hit.distance-1.e-8*1_m,1.e-9*1_m);
      rays.push_back({{p[0],p[1],p[2]},{dir.getX(cs),dir.getY(cs),dir.getZ(cs)},orientation});
      auto const& ray=rays.back();
      indexedOracle.push_back(logical?flat::nextBoundary(data.view(),{ray.origin,ray.direction,orientation==1}).hit
                                     :flat::intersect(data.view(),ray,orientation));
      oracle.push_back(hit.hit?flat::Hit{hit.distance/1_m,static_cast<std::uint32_t>(hit.triangleIndex)}:flat::Hit{});
    }
    // Focused regression for the original DEM origin shared-vertex failure.
    // Keep the independent legacy mesh oracle above for non-feature rays;
    // report the known face-ownership change separately, never hide it in a tolerance.
    std::size_t legacyFeatureDifferences=0;
    if(sharedVertex>=0) {
      if(std::size_t(sharedVertex)>=data.vertices.size())throw std::invalid_argument("shared vertex outside mesh");
      auto v=data.vertices[sharedVertex];
      for(double length:{.01,1.,100.})for(double sign:{1.,-1.}) {
        flat::Ray ray{{v.x,v.y,v.z-sign*length},{0.,0.,sign},sign>0?1:-1};
        auto expected=logical?flat::nextBoundary(data.view(),{ray.origin,ray.direction,sign>0}).hit
                             :flat::intersect(data.view(),ray,ray.orientation);
        if(!expected.found())throw std::runtime_error("shared vertex fixture has no crossing");
        if(!logical&&v.z==0.&&expected.distance!=length)
          throw std::runtime_error("shared vertex independent distance oracle failed");
        auto legacy=data.view();legacy.vertices=nullptr;legacy.indexed_triangles=nullptr;
        auto old=logical?flat::nextBoundary(legacy,{ray.origin,ray.direction,sign>0}).hit
                        :flat::intersect(legacy,ray,ray.orientation);
        if(old.distance!=expected.distance||old.triangle!=expected.triangle)++legacyFeatureDifferences;
        rays.push_back(ray);oracle.push_back(expected);indexedOracle.push_back(expected);
      }
    }
    auto deviceResult=terrain::queryTerrainDevice(data,rays,threads,device,logical,batch,repeats);
    auto const& results=deviceResult.hits;
    std::size_t presence=0,distance=0,face=0,hitCount=0,featureMismatch=0,indexedExactMismatch=0;double maxError=0.;
    for(std::size_t i=0;i<rays.size();++i) {
      if(results[i].triangle!=indexedOracle[i].triangle||results[i].distance!=indexedOracle[i].distance)++indexedExactMismatch;
      if(i>=std::size_t(samples)&&(results[i].triangle!=oracle[i].triangle||results[i].distance!=oracle[i].distance))++featureMismatch;
      if(results[i].found()!=oracle[i].found()){++presence;continue;}
      if(!oracle[i].found())continue;
      ++hitCount;double err=std::abs(results[i].distance-oracle[i].distance);maxError=std::max(maxError,err);
      if(err>1.e-8+1.e-12*std::abs(oracle[i].distance))++distance;
      if(results[i].triangle!=oracle[i].triangle)++face;
    }
    YAML::Node report;
    report["complete"]=true;report["scope"]="geometry oracle only; not integrated Kokkos terrain shower/radio";
    report["execution_space"]=deviceResult.execution_space;report["concurrency"]=deviceResult.concurrency;
    report["samples"]=samples;report["hits"]=hitCount;report["mesh_faces"]=data.triangles.size();
    report["intersection_algorithm"]="indexed-shared-vertices-v2";
    report["indexed_host_device_exact_mismatches"]=indexedExactMismatch;
    report["shared_vertex_id"]=sharedVertex;
    report["shared_vertex_queries"]=rays.size()-samples;
    report["shared_vertex_exact_mismatches"]=featureMismatch;
    report["shared_vertex_legacy_differences"]=legacyFeatureDifferences;
    YAML::Node shared;
    for(std::size_t i=samples;i<rays.size();++i) {
      YAML::Node h;h["distance_m"]=results[i].distance;h["triangle"]=results[i].triangle;shared.push_back(h);
    }
    report["shared_vertex_results"]=shared;
    report["query_modes"]=logical?"logical rock exit / air entry; CPU query-origin tolerance":"closest, outward exit, inward entry (round robin)";
    report["bvh_nodes"]=data.nodes.size();report["geometry_bytes"]=data.bytes();
    report["hit_presence_mismatches"]=presence;report["distance_mismatches"]=distance;
    report["triangle_mismatches"]=face;report["maximum_distance_error_m"]=maxError;
    report["query_fenced_wall_seconds"]=deviceResult.query_fenced_wall_seconds;
    report["timing_scope"]="all repeats including staging and transfers; NOT kernel-only";
    report["repeats"]=repeats;report["geometry_uploads"]=deviceResult.geometry_uploads;
    report["workspace_capacity"]=deviceResult.workspace_capacity;
    report["workspace_bytes"]=deviceResult.workspace_bytes;report["query_batches"]=deviceResult.batches;
    report["repeat_mismatches"]=deviceResult.repeat_mismatches;
    bool pass=presence==0&&distance==0&&face==0&&featureMismatch==0&&indexedExactMismatch==0&&deviceResult.repeat_mismatches==0;report["passed"]=pass;
    std::ofstream file(output);if(!file)throw std::runtime_error("cannot create report");
    file<<report<<'\n';if(!file)throw std::runtime_error("report write failure");
    std::cout<<report<<'\n';return pass?0:2;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
