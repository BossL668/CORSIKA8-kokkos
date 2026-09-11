/* Opt-in terrain admission; geometry algorithms are reused from mountain. */
#pragma once
#include <corsika/geometry/terrain/ClosedMesh.hpp>
#include <map>
#include <cmath>
#include <stdexcept>

namespace corsika::terrain {
  // A warning is insufficient for material transport. Validate BEFORE building
  // triangles/BVH (including indices, which would otherwise be unchecked).
  inline void validateTerrainData(MeshData const& data) {
    if (data.vertices.size()<4 || data.faces.size()<4)
      throw std::invalid_argument("terrain needs a closed three-dimensional mesh");
    std::map<std::pair<std::size_t,std::size_t>,std::pair<unsigned,int>> edges;
    for (auto const& v:data.vertices)
      for (auto x:v) if (!std::isfinite(x))
        throw std::invalid_argument("nonfinite terrain vertex");
    long double volume=0.;
    auto const origin=data.vertices.front();
    for (auto const& face:data.faces) {
      for (auto index:face) if(index>=data.vertices.size())
        throw std::invalid_argument("terrain face index out of range");
      std::array<long double,3> a{},b{},c{},ab{},ac{},cross{};
      for(int k=0;k<3;++k) {
        a[k]=static_cast<long double>(data.vertices[face[0]][k])-origin[k];
        b[k]=static_cast<long double>(data.vertices[face[1]][k])-origin[k];
        c[k]=static_cast<long double>(data.vertices[face[2]][k])-origin[k];
        ab[k]=b[k]-a[k]; ac[k]=c[k]-a[k];
      }
      for(int k=0;k<3;++k)cross[k]=ab[(k+1)%3]*ac[(k+2)%3]-ab[(k+2)%3]*ac[(k+1)%3];
      if (!(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]>0))
        throw std::invalid_argument("degenerate terrain triangle");
      for(int k=0;k<3;++k)volume+=a[k]*(b[(k+1)%3]*c[(k+2)%3]-b[(k+2)%3]*c[(k+1)%3]);
      for(int k=0;k<3;++k) {
        auto i=face[k],j=face[(k+1)%3];auto& e=edges[std::minmax(i,j)];
        ++e.first;e.second+=i<j?1:-1;
      }
    }
    for(auto const& item:edges) if(item.second.first!=2 || item.second.second!=0)
      throw std::invalid_argument("terrain is open, nonmanifold, or inconsistently oriented");
    if (!(volume>0)) throw std::invalid_argument("terrain winding must be outward with positive volume");
    // Self-intersection is not certified by edge incidence. The prepared DEM
    // height-field contract, or an independently validated mesh, is required.
  }

  inline ClosedMesh loadValidatedTerrain(std::string const& path,
                                        CoordinateSystemPtr cs,
                                        LengthType padding=1.e-6*1_m) {
    if (!(padding>0_m) || !std::isfinite(padding/1_m))
      throw std::invalid_argument("invalid terrain padding");
    auto data=MeshLoader::load(path);
    validateTerrainData(data);
    return ClosedMesh(MeshLoader::toPoints(data,cs,1_m),data.faces,padding);
  }
}
