// Geometry-only acceptance. This does not generate shower or radio data.
#include <corsika/geometry/terrain/FlatTerrainExport.hpp>
#include <corsika/geometry/terrain/TerrainCurvedBoundary.hpp>
#include <CLI/CLI.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
namespace f=corsika::terrain::flat;
namespace t=corsika::terrain;
using corsika::terrain::indexed::component;
namespace {
// Independent long-double plane solve/area predicate; BVH boxes are shared.
// Extended precision here is an oracle, never an accelerator physics path.
f::Hit oracle(t::FlatTerrainData const& data,f::QuadraticPath const& p) {
  f::Hit result;
  for(std::uint32_t i=0;i<data.nodes.size();) {
    auto const& n=data.nodes[i];
    if(!f::curveBox(n,p,p.maximum_length_m)){i=n.skip;continue;}
    for(std::uint32_t j=0;j<n.count;++j) {
      auto id=data.indices[n.first+j];auto const& tri=data.indexed_triangles[id];
      auto anchor=data.vertices[tri.vertex[0]];
      long double a=0,b=0,c=0;
      for(int k=0;k<3;++k) {
        long double nk=component(tri.normal,k);
        a+=nk*component(p.quadratic,k);b+=nk*component(p.start.direction,k);
        c+=nk*((long double)component(p.start.origin,k)-component(anchor,k));
      }
      long double r[2];int count=0;
      if(a==0){if(b!=0)r[count++]=-c/b;}
      else {
        auto d=b*b-4*a*c;if(d<0)continue;
        auto q=-.5L*(b+std::copysign(std::sqrt(d),b));
        if(q==0)r[count++]=-b/(2*a);else {r[count++]=q/a;r[count++]=c/q;}
      }
      int kz=0;for(int k=1;k<3;++k)if(std::abs(component(tri.normal,k))>std::abs(component(tri.normal,kz)))kz=k;
      int x=(kz+1)%3,y=(x+1)%3;
      for(int k=0;k<count;++k) {
        auto l=r[k];if(l<-1.e-8L||l>p.maximum_length_m)continue;
        auto slope=b+2*a*l;
        if(p.start.logically_inside?slope<=1.e-12L:slope>=-1.e-12L)continue;
        long double vx[3],vy[3];
        for(int h=0;h<3;++h) {
          auto v=data.vertices[tri.vertex[h]];
          vx[h]=(long double)component(v,x)-component(p.start.origin,x)
            -l*component(p.start.direction,x)-l*l*component(p.quadratic,x);
          vy[h]=(long double)component(v,y)-component(p.start.origin,y)
            -l*component(p.start.direction,y)-l*l*component(p.quadratic,y);
        }
        bool pos=false,neg=false;
        for(int h=0;h<3;++h){int h2=(h+1)%3;auto e=vx[h]*vy[h2]-vy[h]*vx[h2];pos|=e>0;neg|=e<0;}
        if(!(pos||neg)||(pos&&neg))continue;
        double flight=std::max((double)l,1.e-9);
        if(flight<=p.maximum_length_m&&(flight<result.distance||(flight==result.distance&&id<result.triangle)))result={flight,id};
      }
    }
    ++i;
  }
  return result;
}
}
int main(int argc,char** argv) {
  CLI::App app{"Real DEM curved-boundary acceptance (not radio)"};
  std::string mesh,output;int samples=100000,threads=2,device=0,vertex=-1;
  app.add_option("--mesh",mesh)->required()->check(CLI::ExistingFile);
  app.add_option("--output",output)->required();
  app.add_option("--samples",samples)->check(CLI::Range(1,1000000));
  app.add_option("--threads",threads)->check(CLI::Range(1,256));
  app.add_option("--device",device)->check(CLI::NonNegativeNumber);
  app.add_option("--shared-vertex-id",vertex)->check(CLI::NonNegativeNumber);
  CLI11_PARSE(app,argc,argv);
  try {
    if(std::filesystem::exists(output))throw std::runtime_error("refuse overwrite output");
    corsika::logging::set_level(corsika::logging::level::warn);
    auto data=t::exportFlatTerrain(t::loadValidatedTerrain(mesh,corsika::get_root_CoordinateSystem()));
    std::mt19937_64 rng(20260909);std::uniform_real_distribution<double> u(0.,1.);
    std::vector<f::QuadraticPath> paths;std::vector<f::Hit> host,reference;
    for(int i=0;i<samples;++i) {
      auto const& tri=data.indexed_triangles[rng()%data.triangles.size()];
      f::Vec3 center{};
      for(auto id:tri.vertex){auto v=data.vertices[id];center.x+=v.x/3;center.y+=v.y/3;center.z+=v.z/3;}
      auto n=tri.normal;double sign=i%2?1.:-1.;double l=.001+.099*u(rng);
      f::Vec3 d{sign*n.x,sign*n.y,sign*n.z};
      f::Vec3 q=f::cross(d,{.013,-.007,.021});
      f::Vec3 o{center.x-l*d.x-l*l*q.x,center.y-l*d.y-l*l*q.y,center.z-l*d.z-l*l*q.z};
      paths.push_back({{o,d,sign>0},q,2*l});
      host.push_back(f::nextCurvedBoundary(data.view(),paths.back()).hit);
      reference.push_back(oracle(data,paths.back()));
    }
    // Binary lengths/curvature make the original origin vertex an exactly
    // representable target. The expected distance is NOT copied from a query.
    std::uint32_t owner=f::NoTriangle;
    if(vertex>=0) {
      if(std::size_t(vertex)>=data.vertices.size())throw std::runtime_error("invalid vertex");
      auto v=data.vertices[vertex];
      if(v.x!=0.||v.y!=0.||v.z!=0.)throw std::runtime_error("exact feature fixture requires origin vertex");
      for(std::size_t id=0;id<data.indexed_triangles.size();++id) {
        auto const& tri=data.indexed_triangles[id];
        for(auto vid:tri.vertex)if(vid==std::uint32_t(vertex)&&tri.normal.z>1.e-12)owner=std::min(owner,(std::uint32_t)id);
      }
      for(double l:{1./128.,1./64.,1./32.})for(double s:{-1.,1.}) {
        paths.push_back({{{-l*l/64.,0.,-s*l},{0.,0.,s},s>0},{1./64.,0.,0.},2*l});
        host.push_back(f::nextCurvedBoundary(data.view(),paths.back()).hit);
        reference.push_back({l,owner});
      }
    }
    auto actual=t::queryTerrainCurvesDevice(data,paths,threads,device,4096,2);
    std::size_t exact=0,presence=0,face=0,distance=0,feature=0;double maxError=0.;
    YAML::Node features;
    for(std::size_t i=0;i<paths.size();++i) {
      auto a=actual.hits[i],h=host[i],r=reference[i];
      exact+=a.triangle!=h.triangle||a.distance!=h.distance;
      presence+=a.found()!=r.found();face+=a.triangle!=r.triangle;
      if(a.found()&&r.found()) {
        double e=std::abs(a.distance-r.distance);maxError=std::max(maxError,e);
        distance+=e>1.e-10+1.e-12*std::abs(r.distance);
      }
      if(i>=std::size_t(samples)) {
        feature+=a.distance!=r.distance||a.triangle!=r.triangle;
        YAML::Node v;v["distance_m"]=a.distance;v["expected_m"]=r.distance;
        v["face"]=a.triangle;v["expected_face"]=r.triangle;features.push_back(v);
      }
    }
    YAML::Node report;report["complete"]=true;report["scope"]="quadratic geometry only; long-double oracle shares BVH broad phase";
    report["execution_space"]=actual.execution_space;report["random_queries"]=samples;
    report["host_device_exact_mismatches"]=exact;report["presence_mismatches"]=presence;
    report["face_mismatches"]=face;report["distance_mismatches"]=distance;
    report["maximum_oracle_distance_error_m"]=maxError;report["shared_vertex_exact_mismatches"]=feature;
    report["shared_vertex_queries"]=features;report["repeat_mismatches"]=actual.repeat_mismatches;
    report["fenced_query_wall_seconds"]=actual.query_fenced_wall_seconds;
    report["geometry_bytes"]=data.bytes();report["workspace_bytes"]=actual.workspace_bytes;
    bool pass=!(exact||presence||face||distance||feature||actual.repeat_mismatches);report["passed"]=pass;
    std::ofstream file(output);if(!file)throw std::runtime_error("cannot write report");
    file<<report<<'\n';if(!file)throw std::runtime_error("failed writing report");
    std::cout<<report<<'\n';return pass?0:2;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
