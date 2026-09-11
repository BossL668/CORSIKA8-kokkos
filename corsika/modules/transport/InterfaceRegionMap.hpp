#pragma once
#include <corsika/geometry/interfaces/MaterialInterface.hpp>

namespace corsika::interfaces {
// Logical node ownership, not volume C++ type or an endpoint nudge, determines
// the region at a surface. Different-material descendants need another interface.
template<class Node> class InterfaceRegionMap {
 public:
  InterfaceRegionMap(Node const& inside,MaterialInterface binding)
      :inside_(&inside),binding_(binding) {}
  int operator()(Node const* node) const {
    if(!node)throw std::invalid_argument("missing logical interface node");
    for(auto* current=node;current;current=current->getParent())
      if(current==inside_)return binding_.inside_region;
    return binding_.outside_region;
  }
 private:
  Node const* inside_;
  MaterialInterface binding_;
};
} // namespace corsika::interfaces
