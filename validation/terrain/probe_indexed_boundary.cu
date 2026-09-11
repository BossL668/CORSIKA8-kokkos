// Independent geometry acceptance. No shower, PROPOSAL, radio, or RNG changes.
// nvcc -O3 -std=c++17 -arch=sm_89 -I. <this file> -o <probe>
// Also build with --fmad=false, or g++ -x c++ (CPU-only diagnostic).
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define KOKKOS_INLINE_FUNCTION __host__ __device__ inline
#endif
#include <corsika/geometry/terrain/IndexedTerrainIntersection.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace f = corsika::terrain::flat;
namespace g = corsika::terrain::indexed;
void require(bool valid, char const* message) {
  if (!valid) throw std::runtime_error(message);
}
struct Mesh {
  std::vector<f::Vec3> vertices;
  std::vector<g::Triangle> triangles;
  std::vector<std::uint32_t> indices;
  std::vector<f::Node> nodes;
  g::View view() const {
    return {nodes.data(), triangles.data(), indices.data(), vertices.data(),
            static_cast<std::uint32_t>(nodes.size())};
  }
  void leaf() {
    indices.resize(triangles.size());
    std::iota(indices.begin(), indices.end(), 0);
    nodes = {{{-1.e6,-1.e6,-1.e6}, {1.e6,1.e6,1.e6}, 1, 0,
              static_cast<std::uint32_t>(triangles.size())}};
  }
};
Mesh square() {
  Mesh m;
  m.vertices = {{-1,-1,0}, {1,-1,0}, {1,1,0}, {-1,1,0}};
  m.triangles = {{{0,1,2},{0,0,1}}, {{0,2,3},{0,0,1}}};
  m.leaf();
  return m;
}
Mesh readDem(char const* path) {
  std::ifstream in(path, std::ios::binary);
  std::string line, header;
  std::size_t nv=0, nf=0;
  while (std::getline(in,line) && line!="end_header") {
    header += line + '\n';
    std::istringstream row(line); std::string a,b; row>>a>>b;
    if(a=="element"&&b=="vertex") row>>nv;
    if(a=="element"&&b=="face") row>>nf;
  }
  // This is a fixture reader, not a general PLY parser. Fail on other layouts.
  std::string const expected = "ply\nformat binary_little_endian 1.0\n"
      "comment metre-scale CORSIKA local ENU terrain solid\n"
      "element vertex 64694\nproperty double x\nproperty double y\nproperty double z\n"
      "element face 129384\nproperty list uchar uint vertex_indices\n";
  require(header==expected && nv==64694 && nf==129384, "unsupported DEM fixture layout");
  Mesh m; m.vertices.resize(nv);
  in.read(reinterpret_cast<char*>(m.vertices.data()), nv*sizeof(f::Vec3));
  for(std::uint32_t i=0; i<nf; ++i) {
    unsigned char count; std::uint32_t face[3];
    in.read(reinterpret_cast<char*>(&count),1);
    in.read(reinterpret_cast<char*>(face),sizeof(face));
    require(bool(in)&&count==3&&face[0]<nv&&face[1]<nv&&face[2]<nv,"invalid DEM face");
    auto a=f::sub(m.vertices[face[1]],m.vertices[face[0]]);
    auto b=f::sub(m.vertices[face[2]],m.vertices[face[0]]);
    auto n=f::cross(a,b); double norm=std::sqrt(f::dot(n,n));
    require(norm>0.&&std::isfinite(norm),"invalid DEM normal");
    m.triangles.push_back({{face[0],face[1],face[2]}, {n.x/norm,n.y/norm,n.z/norm}});
  }
  require(in.peek()==std::char_traits<char>::eof(),"trailing DEM data");
  m.leaf(); return m;
}
bool same(g::Hit a, g::Hit b) {
  return std::memcmp(&a.distance,&b.distance,sizeof(double))==0 &&
         a.triangle==b.triangle && a.feature==b.feature &&
         a.first==b.first && a.second==b.second;
}
std::uint64_t hash=14695981039346656037ull;
void hashValue(std::uint64_t value) {
  for(int i=0;i<8;++i) { hash^=(value>>(8*i))&255; hash*=1099511628211ull; }
}
void hashHit(g::Hit h) {
  std::uint64_t bits; std::memcpy(&bits,&h.distance,sizeof(bits));
  hashValue(bits); hashValue(h.triangle); hashValue(static_cast<unsigned>(h.feature));
  hashValue(h.first); hashValue(h.second);
}

#if defined(__CUDACC__)
void check(cudaError_t e) {
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> struct DeviceArray {
  T* ptr{}; std::size_t count{};
  explicit DeviceArray(std::size_t n):count(n) { check(cudaMalloc(&ptr,n*sizeof(T))); }
  ~DeviceArray() { if(ptr) cudaFree(ptr); }
  DeviceArray(DeviceArray const&)=delete;
  void upload(T const* p) { check(cudaMemcpy(ptr,p,count*sizeof(T),cudaMemcpyHostToDevice)); }
};
__global__ void query(g::View view, f::Ray const* rays, g::Hit* hits, std::size_t n) {
  auto i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) hits[i]=g::intersect(view,rays[i],rays[i].orientation);
}
struct DeviceMesh {
  DeviceArray<f::Node> nodes;
  DeviceArray<g::Triangle> triangles;
  DeviceArray<std::uint32_t> indices;
  DeviceArray<f::Vec3> vertices;
  explicit DeviceMesh(Mesh const& m):nodes(m.nodes.size()),triangles(m.triangles.size()),
      indices(m.indices.size()),vertices(m.vertices.size()) {
    nodes.upload(m.nodes.data()); triangles.upload(m.triangles.data());
    indices.upload(m.indices.data()); vertices.upload(m.vertices.data());
  }
  g::View view() const {
    return {nodes.ptr,triangles.ptr,indices.ptr,vertices.ptr,static_cast<std::uint32_t>(nodes.count)};
  }
};
#endif

std::size_t comparisons=0, oracleChecks=0, edgeSideMismatches=0;
std::uint64_t planeMaxUlp=0;
double planeMaxAbsolute=0., demMaxAbsolute=0.;
void planeOracle(g::Hit h, double expected) {
  require(h.found()&&expected>0.,"plane oracle missing hit");
  std::uint64_t a,b;
  std::memcpy(&a,&h.distance,sizeof(a)); std::memcpy(&b,&expected,sizeof(b));
  auto ulp=a>b?a-b:b-a;
  planeMaxUlp=std::max(planeMaxUlp,ulp);
  planeMaxAbsolute=std::max(planeMaxAbsolute,std::abs(h.distance-expected));
  require(ulp<=16,"analytic plane distance exceeds 16 ULP");
}
std::vector<g::Hit> compare(Mesh const& m, std::vector<f::Ray> const& rays) {
  std::vector<g::Hit> host; host.reserve(rays.size());
  for(auto const& ray:rays) host.push_back(g::intersect(m.view(),ray,ray.orientation));
#if defined(__CUDACC__)
  DeviceMesh device(m);
  DeviceArray<f::Ray> input(rays.size()); DeviceArray<g::Hit> output(rays.size());
  input.upload(rays.data());
  query<<<(rays.size()+127)/128,128>>>(device.view(),input.ptr,output.ptr,rays.size());
  check(cudaGetLastError());
  std::vector<g::Hit> gpu(rays.size());
  check(cudaMemcpy(gpu.data(),output.ptr,gpu.size()*sizeof(g::Hit),cudaMemcpyDeviceToHost));
  for(std::size_t i=0;i<host.size();++i) {
    if(!same(host[i],gpu[i])) {
      std::cerr<<std::setprecision(17)<<"first host/device mismatch query="<<i
          <<" host_d="<<host[i].distance<<" device_d="<<gpu[i].distance
          <<" host_face="<<host[i].triangle<<" device_face="<<gpu[i].triangle<<'\n';
      throw std::runtime_error("host/device geometry mismatch");
    }
  }
  comparisons+=host.size();
#endif
  for(auto hit:host) hashHit(hit);
  return host;
}
void checkSynthetic() {
  auto m=square();
  std::vector<f::Ray> rays={
      {{0,0,-.01},{0,0,1},1}, // shared edge
      {{-1,-1,-.01},{0,0,1},1}, // shared vertex
      {{.5,-.5,1},{0,0,-1},-1}, // interior, entrance
      {{1.01,0,-1},{0,0,1},0}, // outside
      {{0,0,1},{1,0,0},0}, // parallel
      {{0,0,-1},{0,0,0},0}, // degenerate
      {{0,0,1},{0,0,1},0}, // hit behind ray
      {{0,0,-1},{0,0,1},-1}}; // wrong orientation
  auto ref=compare(m,rays);
  require(ref[0].feature==g::Feature::Edge&&ref[0].triangle==0&&ref[0].distance==.01,"edge oracle");
  require(ref[1].feature==g::Feature::Vertex&&ref[1].first==0&&ref[1].distance==.01,"vertex oracle");
  require(ref[2].triangle==0&&ref[2].distance==1.,"interior oracle");
  for(int i=3;i<8;++i) require(!ref[i].found(),"invalid query oracle");
  oracleChecks+=8;
  // Same original face IDs, different BVH order/layout and winding.
  m.indices={1,0};
  m.nodes={{{-1,-1,0},{1,1,0},3,0,0},{{-1,-1,0},{1,1,0},2,0,1},
           {{-1,-1,0},{1,1,0},3,1,1}};
  for(auto& t:m.triangles) std::swap(t.vertex[0],t.vertex[2]);
  auto reordered=compare(m,rays);
  for(std::size_t i=0;i<ref.size();++i) require(same(ref[i],reordered[i]),"BVH/winding invariant");
  oracleChecks+=ref.size();
  // No finite reciprocal approximation for near-parallel slab traversal.
  f::Node box{{1,-1,9.e11},{2,1,2.e12},1,0,0};
  require(g::slab(box,{{0,0,0},{1.e-12,0,1},0}),"near parallel slab hit");
  require(!g::slab(box,{{0,0,0},{0,0,1},0}),"parallel slab miss");
  oracleChecks+=2;
}
void checkObliqueAndNearEdge() {
  auto m=square();
  std::vector<f::Ray> rays;
  // All three dominant-axis permutations, both ray senses, no exact shared
  // feature assumed for oblique rays. Plane intersection is the long-double oracle.
  std::array<f::Vec3,3> directions={{{.6,0,.8},{0,.8,.6},{.8,0,.6}}};
  for(int i=0;i<20000;++i) {
    auto d=directions[i%3];
    if(i%2) d={-d.x,-d.y,-d.z};
    double t=.01+(i%91)*.03125;
    double x=(i%301-150)/512., y=(i%211-105)/512.;
    rays.push_back({{x-t*d.x,y-t*d.y,-t*d.z},d,d.z>0?1:-1});
  }
  auto hits=compare(m,rays);
  for(std::size_t i=0;i<rays.size();++i) {
    auto const& r=rays[i]; auto h=hits[i];
    long double t=-static_cast<long double>(r.origin.z)/r.direction.z;
    require(h.found()&&std::abs(static_cast<long double>(h.distance)-t)<1.e-13L,
            "oblique long-double plane oracle");
    ++oracleChecks;
  }
  rays.clear();
  for(double x:{-.75,-.125,0.,.125,.75})
    for(double target:{-f::NoHitDistance,f::NoHitDistance})
      rays.push_back({{x,std::nextafter(x,target),-1},{0,0,1},1});
  hits=compare(m,rays);
  for(std::size_t i=0;i<rays.size();++i) {
    auto const& r=rays[i]; auto h=hits[i];
    require(h.found(),"near-edge crack");
    // A rounded vertex-origin subtraction can erase an input-ULP side offset.
    // Keep this diagnostic visible. Backend agreement alone does not certify
    // an exact predicate, and this is a blocker for production integration.
    if(h.triangle!=(r.origin.x>r.origin.y?0u:1u)) ++edgeSideMismatches;
  }
}
void checkMillion() {
  auto m=square();
  std::uint64_t state=67101;
  auto random=[&] { state=state*6364136223846793005ull+1442695040888963407ull;
                    return double(state>>32)*0x1p-32; };
  for(std::size_t begin=0;begin<1000000;begin+=8192) {
    std::vector<f::Ray> rays;
    auto n=std::min<std::size_t>(8192,1000000-begin);
    for(std::size_t i=0;i<n;++i) {
      double x=(2*random()-1)*.999,y=(2*random()-1)*.999,z=.5+100*random();
      if(i%7==0) y=x; // exact diagonal
      if(i%31==0) x=y=-1.; // exact vertex
      bool up=(i%2==0);
      rays.push_back({{x,y,up?-z:z},{0,0,up?1.:-1.},up?1:-1});
    }
    auto hits=compare(m,rays);
    for(std::size_t i=0;i<hits.size();++i) {
      auto const& r=rays[i]; auto h=hits[i];
      planeOracle(h,std::abs(r.origin.z));
      require(h.triangle==(r.origin.x>=r.origin.y?0u:1u),"million face oracle");
      ++oracleChecks;
    }
  }
}
void checkDem(char const* path) {
  auto m=readDem(path);
  require(m.vertices[31842].x==0.&&m.vertices[31842].y==0.&&m.vertices[31842].z==0.,"DEM shared vertex");
  std::vector<f::Ray> rays;
  for(double offset:{.01,1.,100.})
    for(double sign:{1.,-1.}) rays.push_back({{0,0,-sign*offset},{0,0,sign},sign>0?1:-1});
  auto ref=compare(m,rays);
  std::uint32_t owner=f::NoTriangle;
  for(std::uint32_t id=0;id<m.triangles.size();++id) {
    auto const& t=m.triangles[id];
    if(t.normal.z>1.e-12&&(t.vertex[0]==31842||t.vertex[1]==31842||t.vertex[2]==31842))
      owner=std::min(owner,id);
  }
  for(std::size_t i=0;i<ref.size();++i) {
    require(ref[i].triangle==owner&&ref[i].feature==g::Feature::Vertex&&ref[i].first==31842,
            "DEM original shared-vertex owner");
    require(ref[i].distance==std::abs(rays[i].origin.z),"DEM shared-vertex exact distance");
    ++oracleChecks;
  }
  std::reverse(m.indices.begin(),m.indices.end());
  auto reverse=compare(m,rays);
  for(std::size_t i=0;i<ref.size();++i) require(same(ref[i],reverse[i]),"DEM traversal invariant");
  oracleChecks+=ref.size();
  // Interior oblique DEM facets: compare to a separate long-double plane
  // calculation from original vertices, not the candidate's barycentric code.
  rays.clear(); std::vector<std::uint32_t> selected;
  for(std::uint32_t id=17;id<m.triangles.size()&&selected.size()<32;id+=1817) {
    auto const& t=m.triangles[id]; if(t.normal.z<.1) continue;
    auto a=m.vertices[t.vertex[0]],b=m.vertices[t.vertex[1]],c=m.vertices[t.vertex[2]];
    f::Vec3 target{(a.x+b.x+c.x)/3.,(a.y+b.y+c.y)/3.,(a.z+b.z+c.z)/3.};
    rays.push_back({{target.x,target.y,target.z-.01},{0,0,1},1});
    selected.push_back(id);
  }
  auto interior=compare(m,rays);
  for(std::size_t i=0;i<rays.size();++i) {
    auto const& t=m.triangles[selected[i]];
    auto av=m.vertices[t.vertex[0]],bv=m.vertices[t.vertex[1]],cv=m.vertices[t.vertex[2]];
    long double a[3]={av.x,av.y,av.z}, b[3]={bv.x,bv.y,bv.z}, c[3]={cv.x,cv.y,cv.z};
    for(int k=0;k<3;++k) { b[k]-=a[k]; c[k]-=a[k]; }
    long double n[3]={b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0]};
    auto o=rays[i].origin;
    long double expected=(n[0]*(a[0]-o.x)+n[1]*(a[1]-o.y)+n[2]*(a[2]-o.z))/n[2];
    double error=static_cast<double>(std::abs(interior[i].distance-expected));
    require(interior[i].triangle==selected[i]&&error<1.e-10,"DEM interior plane oracle");
    demMaxAbsolute=std::max(demMaxAbsolute,error); ++oracleChecks;
  }
  // Repartition the SAME triangles into bounded leaves. This tests traversal
  // and slab rejection without importing the original CORSIKA BVH implementation.
  m.nodes={m.nodes.front()}; m.nodes[0].count=0;
  for(std::uint32_t first=0;first<m.indices.size();first+=2048) {
    f::Vec3 low{f::NoHitDistance,f::NoHitDistance,f::NoHitDistance};
    f::Vec3 high{-f::NoHitDistance,-f::NoHitDistance,-f::NoHitDistance};
    auto count=std::min<std::uint32_t>(2048,m.indices.size()-first);
    for(std::uint32_t j=first;j<first+count;++j)
      for(auto v:m.triangles[m.indices[j]].vertex) {
        auto p=m.vertices[v];
        low={std::min(low.x,p.x),std::min(low.y,p.y),std::min(low.z,p.z)};
        high={std::max(high.x,p.x),std::max(high.y,p.y),std::max(high.z,p.z)};
      }
    m.nodes.push_back({low,high,static_cast<std::uint32_t>(m.nodes.size()+1),first,count});
  }
  m.nodes[0].skip=m.nodes.size();
  auto partitioned=compare(m,rays);
  for(std::size_t i=0;i<rays.size();++i) require(same(interior[i],partitioned[i]),"DEM leaf partition invariant");
  oracleChecks+=rays.size();
  std::cout<<"  \"dem_vertex_owner\": "<<owner<<",\n";
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"usage: probe_indexed_boundary validated-terrain.ply");
    checkSynthetic(); checkObliqueAndNearEdge(); checkMillion();
    std::cout<<"{\n"; checkDem(argv[1]);
    std::cout<<"  \"tested_contract_passed\": true,\n  \"production_ready\": false,\n"
        <<"  \"host_device_comparisons\": "<<comparisons
        <<",\n  \"oracle_checks\": "<<oracleChecks
        <<",\n  \"one_ulp_edge_side_mismatches\": "<<edgeSideMismatches
        <<",\n  \"plane_distance_max_ulp\": "<<planeMaxUlp
        <<",\n  \"plane_distance_max_absolute_m\": "<<std::setprecision(17)<<planeMaxAbsolute
        <<",\n  \"dem_interior_max_absolute_m\": "<<demMaxAbsolute
        <<",\n  \"result_fnv1a64\": \""<<std::hex<<hash<<"\",\n"
        <<"  \"scope\": \"standalone indexed straight rays; not production transport\"\n}\n";
    return 0;
  } catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
