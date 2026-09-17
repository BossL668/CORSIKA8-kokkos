#pragma once
#include <corsika/geometry/terrain/DemCoverage.hpp>
#include <corsika/geometry/terrain/FlatTerrainData.hpp>
#include <map>
#include <set>
#include <functional>
#include <numeric>
#include <algorithm>
#include <stdexcept>
namespace corsika::terrain::coverage {
struct Data {
  std::vector<Edge> edges;
  std::vector<flat::Node> nodes;
  std::vector<std::uint32_t> indices;
  View view()const{return {edges.data(),nodes.data(),indices.data(),static_cast<std::uint32_t>(nodes.size())};}
  std::size_t bytes()const{return edges.size()*sizeof(Edge)+nodes.size()*sizeof(flat::Node)+indices.size()*sizeof(std::uint32_t);}
};
// Only opt-in heightfield DEMs: positive-z faces are the physical top, vertical
// sides and the negative-z bottom are artificial mesh closure. Reject ambiguous
// topology rather than guessing an ENU rectangle from a bounding box.
inline Data fromHeightfield(FlatTerrainData const& mesh) {
  using Key=std::pair<std::uint32_t,std::uint32_t>;
  std::map<Key,std::pair<unsigned,Key>> incidence;
  for(auto const& t:mesh.indexed_triangles)if(t.normal.z>0.) {
    for(int j=0;j<3;++j) {
      auto a=t.vertex[j],b=t.vertex[(j+1)%3];auto key=std::minmax(a,b);
      auto& entry=incidence[key];++entry.first;entry.second={a,b};
      if(entry.first>2)throw std::invalid_argument("DEM top is nonmanifold");
    }
  }
  std::map<std::uint32_t,std::uint32_t> next;
  std::set<std::uint32_t> incoming;
  for(auto const& item:incidence)if(item.second.first==1) {
    auto [a,b]=item.second.second;
    if(!next.emplace(a,b).second||!incoming.insert(b).second)throw std::invalid_argument("DEM perimeter branches");
  }
  if(next.size()<3)throw std::invalid_argument("DEM has no heightfield perimeter");
  Data d;auto first=next.begin()->first,id=first;
  do {
    auto it=next.find(id);if(it==next.end())throw std::invalid_argument("DEM perimeter is open");
    auto a=mesh.vertices.at(id),b=mesh.vertices.at(it->second);a.z=b.z=0.;
    if(a.x==b.x&&a.y==b.y)throw std::invalid_argument("DEM perimeter has zero length");
    d.edges.push_back({a,b});id=it->second;
    if(d.edges.size()>next.size())throw std::invalid_argument("DEM perimeter does not close");
  }while(id!=first);
  if(d.edges.size()!=next.size())throw std::invalid_argument("DEM coverage has holes or disconnected components");
  // Reject self-intersections and clockwise perimeters. Cost occurs once.
  long double area=0.;
  auto orient=[](Vec3 a,Vec3 b,Vec3 c){return (static_cast<long double>(b.x)-a.x)*(c.y-a.y)-(static_cast<long double>(b.y)-a.y)*(c.x-a.x);};
  for(std::size_t i=0;i<d.edges.size();++i) {
    auto e=d.edges[i];area+=static_cast<long double>(e.a.x)*e.b.y-static_cast<long double>(e.b.x)*e.a.y;
    for(std::size_t j=i+2;j<d.edges.size();++j) {
      if(i==0&&j+1==d.edges.size())continue;
      auto f=d.edges[j];auto a=orient(e.a,e.b,f.a),b=orient(e.a,e.b,f.b),c=orient(f.a,f.b,e.a),q=orient(f.a,f.b,e.b);
      if(a*b<0.&&c*q<0.)throw std::invalid_argument("DEM perimeter self-intersects");
    }
  }
  if(!(area>0.))throw std::invalid_argument("DEM top winding is invalid");
  d.indices.resize(d.edges.size());std::iota(d.indices.begin(),d.indices.end(),0);
  std::function<void(std::size_t,std::size_t)> build=[&](std::size_t begin,std::size_t end) {
    auto index=d.nodes.size();flat::Node n{{HUGE_VAL,HUGE_VAL,0.},{-HUGE_VAL,-HUGE_VAL,0.},0,0,0};
    for(auto j=begin;j<end;++j)for(auto p:{d.edges[d.indices[j]].a,d.edges[d.indices[j]].b}) {
      n.low.x=std::min(n.low.x,p.x);n.low.y=std::min(n.low.y,p.y);
      n.high.x=std::max(n.high.x,p.x);n.high.y=std::max(n.high.y,p.y);
    }
    d.nodes.push_back(n);
    if(end-begin<=4){d.nodes[index].first=begin;d.nodes[index].count=end-begin;}
    else {
      bool x=n.high.x-n.low.x>n.high.y-n.low.y;auto mid=(begin+end)/2;
      std::nth_element(d.indices.begin()+begin,d.indices.begin()+mid,d.indices.begin()+end,[&](auto a,auto b){
        auto e=d.edges[a],f=d.edges[b];return x?e.a.x+e.b.x<f.a.x+f.b.x:e.a.y+e.b.y<f.a.y+f.b.y;});
      build(begin,mid);build(mid,end);
    }
    d.nodes[index].skip=d.nodes.size();
  };
  build(0,d.indices.size());return d;
}
} // namespace corsika::terrain::coverage
