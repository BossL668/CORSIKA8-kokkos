#pragma once
#include <Kokkos_Core.hpp>
#include <corsika/geometry/terrain/DemCoverageData.hpp>
namespace corsika::terrain::coverage {
template<class Space> class KokkosData {
  Kokkos::View<Edge*,typename Space::memory_space> edges_;
  Kokkos::View<flat::Node*,typename Space::memory_space> nodes_;
  Kokkos::View<std::uint32_t*,typename Space::memory_space> indices_;
 public:
  explicit KokkosData(Data const& d) {
    auto upload=[](auto& view,auto const& values,char const* label) {
      using V=std::decay_t<decltype(view)>;view=V(std::string(label),values.size());
      auto host=Kokkos::create_mirror_view(view);
      for(std::size_t i=0;i<values.size();++i)host(i)=values[i];Kokkos::deep_copy(view,host);
    };
    upload(edges_,d.edges,"dem_coverage_edges");upload(nodes_,d.nodes,"dem_coverage_nodes");upload(indices_,d.indices,"dem_coverage_indices");
  }
  View view()const{return {edges_.data(),nodes_.data(),indices_.data(),static_cast<std::uint32_t>(nodes_.extent(0))};}
};
}
