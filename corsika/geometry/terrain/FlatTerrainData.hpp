#pragma once
#include <corsika/geometry/terrain/FlatTerrain.hpp>
#include <vector>
#include <string>
namespace corsika::terrain {
struct FlatTerrainData {
  std::vector<flat::Node> nodes;
  std::vector<flat::Triangle> triangles;
  std::vector<std::uint32_t> indices;
  std::vector<flat::Vec3> vertices;
  std::vector<flat::IndexedTriangle> indexed_triangles;
  flat::View view()const{return {nodes.data(),triangles.data(),indices.data(),static_cast<std::uint32_t>(nodes.size()),indexed_triangles.data(),vertices.data()};}
  std::size_t bytes()const{return nodes.size()*sizeof(flat::Node)+triangles.size()*sizeof(flat::Triangle)+indices.size()*sizeof(std::uint32_t)+vertices.size()*sizeof(flat::Vec3)+indexed_triangles.size()*sizeof(flat::IndexedTriangle);}
};
struct TerrainQueryResult {
  std::vector<flat::Hit> hits;
  std::string execution_space;
  int concurrency{};
  double query_fenced_wall_seconds{};
  std::size_t geometry_uploads{};
  std::size_t workspace_capacity{};
  std::size_t workspace_bytes{};
  std::size_t batches{};
  std::size_t repeat_mismatches{};
};
// Standalone diagnostic wrapper owns one runtime and one resident session.
// Production callers instead borrow their runtime via KokkosTerrainSession.
TerrainQueryResult queryTerrainDevice(FlatTerrainData const&,std::vector<flat::Ray> const&,
    int threads,int device,bool logical_boundaries=false,std::size_t capacity=4096,
    int repeats=1);
namespace flat { struct QuadraticPath; }
TerrainQueryResult queryTerrainCurvesDevice(FlatTerrainData const&,
    std::vector<flat::QuadraticPath> const&,int threads,int device,
    std::size_t capacity=4096,int repeats=1);
}
