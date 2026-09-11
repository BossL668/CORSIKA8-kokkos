// Standalone numerical diagnostic, not linked into any application.
// Build twice with nvcc: default and --fmad=false. Same portable query and data.
#include <cuda_runtime.h>
#define KOKKOS_INLINE_FUNCTION __host__ __device__ inline
#include <corsika/geometry/terrain/TerrainBoundary.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
namespace f = corsika::terrain::flat;
void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
__global__ void query(f::View view, f::BoundaryQuery input, f::BoundaryCandidate* out) {
  *out=f::nextBoundary(view,input);
}
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  std::ifstream in(argv[1],std::ios::binary);
  std::string line;std::size_t nv=0,nf=0;bool binary=false;
  while(std::getline(in,line)&&line!="end_header") {
    std::istringstream row(line);std::string a,b;row>>a>>b;
    if(a=="format")binary=b=="binary_little_endian";
    if(a=="element"&&b=="vertex")row>>nv;
    if(a=="element"&&b=="face")row>>nf;
  }
  if(!in||!binary||nv!=64694||nf!=129384)throw std::runtime_error("diagnostic requires validated 21CMA PLY layout");
  std::vector<f::Vec3> vertices(nv);in.read(reinterpret_cast<char*>(vertices.data()),nv*sizeof(f::Vec3));
  std::vector<f::Triangle> triangles;std::vector<std::uint32_t> indices;
  for(std::uint32_t i=0;i<nf;++i) {
    unsigned char count;std::uint32_t face[3];
    in.read(reinterpret_cast<char*>(&count),1);in.read(reinterpret_cast<char*>(face),sizeof(face));
    if(!in||count!=3||face[0]>=nv||face[1]>=nv||face[2]>=nv)throw std::runtime_error("bad face");
    auto origin=vertices[face[0]],a=f::sub(vertices[face[1]],origin),b=f::sub(vertices[face[2]],origin);
    auto n=f::cross(a,b);double norm=std::sqrt(f::dot(n,n));n={n.x/norm,n.y/norm,n.z/norm};
    triangles.push_back({origin,a,b,n});indices.push_back(i);
  }
  // A single enclosing leaf removes BVH traversal as a possible confounder.
  f::Node node{{-1.e6,-1.e6,-1.e6},{1.e6,1.e6,1.e6},1,0,static_cast<std::uint32_t>(nf)};
  f::Node* dn;f::Triangle* dt;std::uint32_t* di;f::BoundaryCandidate* dr;
  check(cudaMalloc(&dn,sizeof(node)));check(cudaMalloc(&dt,nf*sizeof(f::Triangle)));
  check(cudaMalloc(&di,nf*sizeof(std::uint32_t)));check(cudaMalloc(&dr,sizeof(f::BoundaryCandidate)));
  check(cudaMemcpy(dn,&node,sizeof(node),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dt,triangles.data(),nf*sizeof(f::Triangle),cudaMemcpyHostToDevice));
  check(cudaMemcpy(di,indices.data(),nf*sizeof(std::uint32_t),cudaMemcpyHostToDevice));
  for(bool inside:{true,false}) {
    f::BoundaryQuery input{{0,0,inside?-.01:.01},{0,0,inside?1.:-1.},inside};
    auto host=f::nextBoundary({&node,triangles.data(),indices.data(),1},input);
    query<<<1,1>>>({dn,dt,di,1},input,dr);check(cudaGetLastError());
    f::BoundaryCandidate device;check(cudaMemcpy(&device,dr,sizeof(device),cudaMemcpyDeviceToHost));
    std::cout<<std::setprecision(17)<<"inside="<<inside<<" host_distance="<<host.hit.distance
      <<" device_distance="<<device.hit.distance<<" delta="<<device.hit.distance-host.hit.distance
      <<" host_face="<<host.hit.triangle<<" device_face="<<device.hit.triangle
      <<" host_z="<<input.origin.z+input.direction.z*host.hit.distance
      <<" device_z="<<input.origin.z+input.direction.z*device.hit.distance<<'\n';
  }
  check(cudaFree(dr));check(cudaFree(di));check(cudaFree(dt));check(cudaFree(dn));
}
