/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once

#include <corsika/accelerator/em/common/EnvironmentTreeSnapshot.hpp>
#include <corsika/framework/geometry/Box.hpp>
#include <corsika/framework/geometry/SeparationPlane.hpp>
#include <corsika/media/Universe.hpp>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace corsika::gpu::em::environment_tree {
  template <typename NativeNode> struct HostTree {
    std::vector<Node> nodes;
    std::vector<NodeId> children, exclusions;
    std::vector<NativeNode const*> native_nodes;
    std::unordered_map<NativeNode const*,NodeId> ids;
    TreeView view() const {
      return {nodes.data(),children.data(),exclusions.data(),static_cast<unsigned>(nodes.size())};
    }
    NodeId id(NativeNode const* p) const { return p?ids.at(p):Outside; }
  };

  /**
   * Export the actual C8 tree, not an atmosphere-ID reconstruction. The callback
   * assigns model payloads from the actual node properties (including shared
   * models). Unsupported shapes or external exclusion pointers fail explicitly.
   * No node, child, or exclusion is silently omitted; no fixed 5-layer limit.
   */
  template <typename NativeNode, typename ExportModel>
  HostTree<NativeNode> exportEnvironmentTree(NativeNode const& root,
      CoordinateSystemPtr const& frame, ExportModel&& export_model) {
    HostTree<NativeNode> tree;
    std::function<void(NativeNode const&)> visit=[&](NativeNode const& node) {
      if (tree.native_nodes.size()>=static_cast<std::size_t>(std::numeric_limits<NodeId>::max()))
        throw std::invalid_argument("environment tree exceeds NodeId range");
      tree.ids.emplace(&node,static_cast<NodeId>(tree.native_nodes.size()));
      tree.native_nodes.push_back(&node);
      for (auto const& child:node.getChildNodes()) visit(*child);
    };
    visit(root);
    tree.nodes.resize(tree.native_nodes.size());
    auto point=[&](Point const& p,double* out) {
      auto c=p.getCoordinates(frame);
      for (int axis=0;axis<3;++axis) out[axis]=c[axis]/1_m;
    };
    auto vector=[&](DirectionVector const& v,double* out) {
      auto c=v.getComponents(frame);
      for (int axis=0;axis<3;++axis) out[axis]=c[axis].magnitude();
    };
    for (unsigned i=0;i<tree.nodes.size();++i) {
      auto const& native=*tree.native_nodes[i]; auto& node=tree.nodes[i];
      node.parent=tree.id(native.getParent());
      node.child_begin=static_cast<unsigned>(tree.children.size());
      node.child_count=static_cast<unsigned>(native.getChildNodes().size());
      for (auto const& child:native.getChildNodes()) tree.children.push_back(tree.id(child.get()));
      node.excluded_begin=static_cast<unsigned>(tree.exclusions.size());
      node.excluded_count=static_cast<unsigned>(native.getExcludedNodes().size());
      for (auto excluded:native.getExcludedNodes()) tree.exclusions.push_back(tree.id(excluded));
      auto const& volume=native.getVolume();
      if (dynamic_cast<Universe const*>(&volume)) node.shape=Shape::Universe;
      else if (auto s=dynamic_cast<Sphere const*>(&volume)) {
        node.shape=Shape::Sphere; point(s->getCenter(),node.center); node.radius=s->getRadius()/1_m;
      } else if (auto b=dynamic_cast<Box const*>(&volume)) {
        node.shape=Shape::Box;
        // Box::contains uses its current CS, which may have been rotated after
        // construction. Do not reconstruct the center from stale cached fields.
        point(Point(b->getCoordinateSystem(),0_m,0_m,0_m),node.center);
        vector(DirectionVector(b->getCoordinateSystem(),{1.,0.,0.}),node.axes[0]);
        vector(DirectionVector(b->getCoordinateSystem(),{0.,1.,0.}),node.axes[1]);
        vector(DirectionVector(b->getCoordinateSystem(),{0.,0.,1.}),node.axes[2]);
        node.half_extent[0]=b->getX()/1_m; node.half_extent[1]=b->getY()/1_m; node.half_extent[2]=b->getZ()/1_m;
      } else if (auto p=dynamic_cast<SeparationPlane const*>(&volume)) {
        node.shape=Shape::SeparationPlane; point(p->getPlane().getCenter(),node.center);
        vector(p->getPlane().getNormal(),node.axes[0]);
      } else throw std::invalid_argument("unsupported native volume in full environment snapshot");
      node.model_id=export_model(native,i);
    }
    return tree;
  }
} // namespace corsika::gpu::em::environment_tree
