#pragma once
#include <Kokkos_Core.hpp>
#include <corsika/geometry/terrain/KokkosDemCoverage.hpp>
#include <corsika/modules/radio/interface/Propagation.hpp>
#include <stdexcept>

namespace corsika::radio::interface {
inline void validatePropagation(PropagationConfig const& c) {
  using namespace detail;
  if(c.geometry!=Geometry::Uniform&&c.geometry!=Geometry::Plane&&c.geometry!=Geometry::Mesh)
    throw std::invalid_argument("unsupported interface radio geometry");
  if(!finite(c.plane_point_m)||!finite(c.plane_outward_normal)||
     std::abs(norm(c.plane_outward_normal)-1.)>1.e-12||
     c.optical_integration_samples<4||c.optical_integration_samples>4096||
     !std::isfinite(c.visibility_tolerance_m)||c.visibility_tolerance_m<=0.||c.visibility_tolerance_m>1.e-4||
     !std::isfinite(c.shared_edge_tolerance_m)||c.shared_edge_tolerance_m<=0.||c.shared_edge_tolerance_m>1.e-4)
    throw std::invalid_argument("invalid interface optical geometry/integration settings");
  for(auto const& m:c.media){
    if(!std::isfinite(m.refractive_index)||m.refractive_index<1.||
       !(m.attenuation_length_m>0.)||!finite(m.center_m)||!std::isfinite(m.reference_radius_m)||
       m.radial_index.size()==1||m.radial_index.size()>1000000||m.radial_integration_breaks_m.size()>64||
       (m.radial_index.empty()&&!m.radial_integration_breaks_m.empty()))
      throw std::invalid_argument("invalid interface optical material");
    double previous=-HUGE_VAL;
    for(auto s:m.radial_index){
      if(!std::isfinite(s.height_m)||s.height_m<=previous||!std::isfinite(s.index)||s.index<1.)
        throw std::invalid_argument("invalid radial refractive-index table");
      previous=s.height_m;
    }
    previous=-HUGE_VAL;
    for(double h:m.radial_integration_breaks_m){
      if(!std::isfinite(h)||h<=previous||h<m.radial_index.front().height_m||h>m.radial_index.back().height_m)
        throw std::invalid_argument("invalid radial optical integration break");
      previous=h;
    }
  }
}
inline detail::PropagationView hostPropagation(PropagationConfig const& c,
    terrain::flat::View mesh={},std::size_t faces=0) {
  validatePropagation(c);
  detail::PropagationView v;v.coverage=c.coverage.view();
  v.geometry=c.geometry;v.plane_point=c.plane_point_m;v.outward=c.plane_outward_normal;
  v.integration_samples=c.optical_integration_samples;v.visibility_tolerance=c.visibility_tolerance_m;
  v.edge_tolerance=c.shared_edge_tolerance_m;v.mesh=mesh;v.faces=faces;
  for(int i=0;i<2;++i){auto const& m=c.media[i];
    v.media[i]={m.refractive_index,m.attenuation_length_m,m.center_m,m.reference_radius_m,
      m.radial_index.data(),static_cast<std::uint32_t>(m.radial_index.size()),
      m.radial_integration_breaks_m.data(),static_cast<std::uint32_t>(m.radial_integration_breaks_m.size())};
    v.media[i].minimum_index=v.media[i].maximum_index=m.refractive_index;
    for(auto s:m.radial_index){v.media[i].minimum_index=std::min(v.media[i].minimum_index,s.index);v.media[i].maximum_index=std::max(v.media[i].maximum_index,s.index);}
  }
  return v;
}
// Borrows an already resident immutable mesh; owns only its optical tables.
template<class Space> class KokkosPropagation {
 public:
  using Memory=typename Space::memory_space;
  using NormalBounds=Kokkos::View<detail::TransmissionNormalBounds*,Memory>;
  struct BuildNormalBounds {
    terrain::flat::View mesh;NormalBounds bounds;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i)const {
      detail::TransmissionNormalBounds value{{HUGE_VAL,HUGE_VAL,HUGE_VAL},{-HUGE_VAL,-HUGE_VAL,-HUGE_VAL}};
      for(auto j=i;j<mesh.nodes[i].skip;++j){
        auto const& node=mesh.nodes[j];
        for(std::uint32_t k=0;k<node.count;++k){
          auto n=mesh.indexed_triangles[mesh.indices[node.first+k]].normal;
          value.low={detail::min(value.low.x,n.x),detail::min(value.low.y,n.y),detail::min(value.low.z,n.z)};
          value.high={detail::max(value.high.x,n.x),detail::max(value.high.y,n.y),detail::max(value.high.z,n.z)};
        }
      }
      bounds(i)=value;
    }
  };
  KokkosPropagation(PropagationConfig const& c,terrain::flat::View mesh,
                    std::size_t faces,Space const& execution) :coverage_(c.coverage),view_(hostPropagation(c,mesh,faces)) {
    view_.coverage=coverage_.view();bytes_+=c.coverage.bytes();
    if(c.geometry==Geometry::Mesh&&(!mesh.nodes||!mesh.indexed_triangles||!mesh.vertices||!mesh.node_count||!faces))
      throw std::invalid_argument("interface radio requires a validated resident indexed mesh");
    if(c.geometry==Geometry::Mesh&&c.transmission_bvh){
      normals_=NormalBounds("interface_radio_transmission_normals",mesh.node_count);
      Kokkos::parallel_for("interface_radio_build_transmission_bounds",Kokkos::RangePolicy<Space>(execution,0,mesh.node_count),BuildNormalBounds{mesh,normals_});
      execution.fence("interface radio transmission bounds initialization");
      view_.transmission_normals=normals_.data();bytes_+=normals_.span()*sizeof(detail::TransmissionNormalBounds);
    }
    for(int i=0;i<2;++i){
      tables_[i]=Table("interface_radio_index_table",c.media[i].radial_index.size());
      auto host=Kokkos::create_mirror_view(tables_[i]);
      for(std::size_t j=0;j<host.extent(0);++j)host(j)=c.media[i].radial_index[j];
      Kokkos::deep_copy(execution,tables_[i],host);
      // Host staging lifetime ends here; do not leave asynchronous uploads pending.
      execution.fence("interface radio optical table initialization");
      view_.media[i].samples=tables_[i].data();bytes_+=tables_[i].span()*sizeof(IndexSample);
      breaks_[i]=Breaks("interface_radio_optical_breaks",c.media[i].radial_integration_breaks_m.size());
      auto breakHost=Kokkos::create_mirror_view(breaks_[i]);
      for(std::size_t j=0;j<breakHost.extent(0);++j)breakHost(j)=c.media[i].radial_integration_breaks_m[j];
      Kokkos::deep_copy(execution,breaks_[i],breakHost);execution.fence("interface radio optical break initialization");
      view_.media[i].breaks=breaks_[i].data();bytes_+=breaks_[i].span()*sizeof(double);
    }
  }
  detail::PropagationView view()const{return view_;}
  std::size_t bytes()const{return bytes_;}
 private:
  using Table=Kokkos::View<IndexSample*,Memory>;
  using Breaks=Kokkos::View<double*,Memory>;
  terrain::coverage::KokkosData<Space> coverage_;
  Table tables_[2];Breaks breaks_[2];NormalBounds normals_;detail::PropagationView view_;std::size_t bytes_{};
};
} // namespace corsika::radio::interface
