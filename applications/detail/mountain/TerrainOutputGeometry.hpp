#pragma once
#include <corsika/geometry/terrain/FlatTerrainExport.hpp>
#include <Eigen/Dense>
#include <optional>

namespace corsika::applications::terrain {
// Read-only CPU output audit. Export the existing BVH/edges once and repeat the
// CPU mesh's three-ray vote with the same Eigen arithmetic and tolerances.
// This avoids constructing Point/Vector/shared-coordinate objects at every BVH
// node and triangle. It is deliberately separate from transport/radio queries.
class OutputMeshAudit {
 public:
  explicit OutputMeshAudit(corsika::terrain::TriangularMesh const* mesh):mesh_(mesh) {
    if(!mesh_||!mesh_->getBVH().getRoot())return;
    // Other derived volumes may override contains() with different semantics.
    if(typeid(*mesh_)!=typeid(corsika::terrain::TriangularMesh)&&
       typeid(*mesh_)!=typeid(corsika::terrain::ClosedMesh))return;
    cs_=mesh_->getCoordinateSystem();data_=corsika::terrain::exportFlatTerrain(*mesh_);
    data_.vertices.clear();data_.vertices.shrink_to_fit();
    data_.indexed_triangles.clear();data_.indexed_triangles.shrink_to_fit();
    padding_=mesh_->getBoundaryPadding()/1_m;
  }
  std::size_t bytes()const{return data_.bytes();}
  bool enabled()const{return !data_.nodes.empty();}
  template<class Node> Node const* containingNode(Node const& root,Point const& p)const {
    if(!enabled())return root.getContainingNode(p);
    std::optional<bool> meshContains;
    return find(root,p,meshContains);
  }
  bool contains(Point const& p)const {
    if(!enabled())return mesh_&&mesh_->contains(p);
    auto c=p.getCoordinates(cs_);
    Eigen::Vector3d origin(c.getX()/1_m,c.getY()/1_m,c.getZ()/1_m);
    auto const& bounds=data_.nodes.front();
    if(origin.x()<bounds.low.x||origin.x()>bounds.high.x||
       origin.y()<bounds.low.y||origin.y()>bounds.high.y||
       origin.z()<bounds.low.z||origin.z()>bounds.high.z)return false;
    int votes=0;bool near=false;std::vector<double> hits;
    for(auto const& direction:directions()) {
      hits.clear();intersections(origin,direction,hits);
      std::sort(hits.begin(),hits.end());
      std::size_t count=0;double last=0.;bool first=true;
      for(double distance:hits) {
        // Exactly BVH::findAllIntersections' sorted tolerance deduplication.
        if(!first&&distance-last<=padding_)continue;
        first=false;last=distance;
        if(distance>padding_)++count;
        else if(distance>0.){near=true;++count;}
      }
      votes+=count%2==1;
    }
    // IVolume contains() includes kSurface, as in TriangularMesh::inside().
    return near||votes>=2;
  }
 private:
  using Vec=corsika::terrain::flat::Vec3;
  using FlatNode=corsika::terrain::flat::Node;
  static Eigen::Vector3d vector(Vec p){return {p.x,p.y,p.z};}
  static std::array<Eigen::Vector3d,3> const& directions(){
    static const std::array<Eigen::Vector3d,3> value{{{1.,.31784,.19273},{.19273,1.,.31784},{.31784,.19273,1.}}};
    return value;
  }
  static bool intersects(FlatNode const& node,Eigen::Vector3d const& p,Eigen::Vector3d const& inverse) {
    double x1=(node.low.x-p.x())*inverse.x(),x2=(node.high.x-p.x())*inverse.x();
    double y1=(node.low.y-p.y())*inverse.y(),y2=(node.high.y-p.y())*inverse.y();
    double z1=(node.low.z-p.z())*inverse.z(),z2=(node.high.z-p.z())*inverse.z();
    if(x1>x2)std::swap(x1,x2);if(y1>y2)std::swap(y1,y2);if(z1>z2)std::swap(z1,z2);
    auto minimum=std::max({x1,y1,z1}),maximum=std::min({x2,y2,z2});
    return maximum>=minimum&&maximum>=0.;
  }
  void intersections(Eigen::Vector3d const& origin,Eigen::Vector3d const& direction,std::vector<double>& hits)const {
    // All three fixed CPU probe directions exceed BVH::kDirEpsilon.
    Eigen::Vector3d inverse(1./direction.x(),1./direction.y(),1./direction.z());
    for(std::size_t i=0;i<data_.nodes.size();) {
      auto const& node=data_.nodes[i];
      if(!intersects(node,origin,inverse)){i=node.skip;continue;}
      for(std::size_t j=node.first;j<node.first+node.count;++j) {
        auto const& tri=data_.triangles[data_.indices[j]];
        Eigen::Vector3d e1=vector(tri.edge1),e2=vector(tri.edge2),P=direction.cross(e2);
        double det=e1.dot(P);
        if(std::abs(det)<corsika::terrain::Triangle::kParallelEpsilon)continue;
        double inverseDet=1./det;
        Eigen::Vector3d T=origin-vector(tri.origin);
        double u=T.dot(P)*inverseDet;if(u<0.||u>1.)continue;
        Eigen::Vector3d Q=T.cross(e1);
        double v=direction.dot(Q)*inverseDet;if(v<0.||u+v>1.)continue;
        double t=e2.dot(Q)*inverseDet;if(t<0.)continue;
        hits.push_back(t);
      }
      ++i;
    }
  }
  template<class Node> bool containsNode(Node const& node,Point const& p,std::optional<bool>& cached)const {
    if(&node.getVolume()!=mesh_)return node.contains(p);
    if(!cached)cached=contains(p);
    return *cached;
  }
  template<class Node> Node const* find(Node const& node,Point const& p,std::optional<bool>& cached)const {
    if(!containsNode(node,p,cached))return nullptr;
    for(auto const& child:node.getChildNodes())if(containsNode(*child,p,cached))return find(*child,p,cached);
    for(auto const* excluded:node.getExcludedNodes())if(containsNode(*excluded,p,cached))return find(*excluded,p,cached);
    return &node;
  }
  corsika::terrain::TriangularMesh const* mesh_{};
  CoordinateSystemPtr cs_;
  corsika::terrain::FlatTerrainData data_;
  double padding_{};
};
}
