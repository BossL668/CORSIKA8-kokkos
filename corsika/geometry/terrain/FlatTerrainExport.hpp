#pragma once
#include <corsika/geometry/terrain/FlatTerrainData.hpp>
#include <corsika/geometry/terrain/ValidatedTerrain.hpp>

namespace corsika::terrain {
inline FlatTerrainData exportFlatTerrain(TriangularMesh const& mesh) {
  FlatTerrainData out;
  auto const max=std::numeric_limits<std::uint32_t>::max();
  if(mesh.getTriangleCount()>max/2||mesh.getVertexCount()>max)throw std::invalid_argument("terrain exceeds 32-bit BVH address budget");
  auto cs=mesh.getCoordinateSystem();
  auto point=[&](Point const& p){return flat::Vec3{p.getX(cs)/1_m,p.getY(cs)/1_m,p.getZ(cs)/1_m};};
  out.vertices.reserve(mesh.getVertexCount());
  for(std::size_t i=0;i<mesh.getVertexCount();++i)out.vertices.push_back(point(mesh.getVertex(i)));
  for(auto const& tri:mesh.getTriangles()) {
    auto e1=tri.getEdge1(),e2=tri.getEdge2();auto n=tri.getNormal();
    auto const ids=tri.getVertexIndices();
    out.indexed_triangles.push_back({{static_cast<std::uint32_t>(ids[0]),
        static_cast<std::uint32_t>(ids[1]),static_cast<std::uint32_t>(ids[2])},
        {n.getX(cs),n.getY(cs),n.getZ(cs)}});
    out.triangles.push_back({point(mesh.getVertex(tri.getVertexIndices()[0])),
        {e1.getX(cs)/1_m,e1.getY(cs)/1_m,e1.getZ(cs)/1_m},
        {e2.getX(cs)/1_m,e2.getY(cs)/1_m,e2.getZ(cs)/1_m},
        {n.getX(cs),n.getY(cs),n.getZ(cs)}});
  }
  std::function<void(BVHNode const*)> visit=[&](BVHNode const* node) {
    if(!node)throw std::invalid_argument("terrain BVH has a missing node");
    auto index=out.nodes.size();
    out.nodes.push_back({point(node->bounds.getMin()),point(node->bounds.getMax()),0,
                         static_cast<std::uint32_t>(out.indices.size()),static_cast<std::uint32_t>(node->triangleIndices.size())});
    for(auto triangle:node->triangleIndices)out.indices.push_back(static_cast<std::uint32_t>(triangle));
    if(!node->isLeaf()){visit(node->left.get());visit(node->right.get());}
    out.nodes[index].skip=static_cast<std::uint32_t>(out.nodes.size());
  };
  visit(mesh.getBVH().getRoot());
  if(out.indices.size()!=out.triangles.size())throw std::logic_error("BVH does not cover each terrain face exactly once");
  for(std::size_t i=0;i<out.nodes.size();++i)
    if(out.nodes[i].skip<=i||out.nodes[i].skip>out.nodes.size())throw std::logic_error("invalid BVH forward link");
  return out;
}
} // namespace corsika::terrain
