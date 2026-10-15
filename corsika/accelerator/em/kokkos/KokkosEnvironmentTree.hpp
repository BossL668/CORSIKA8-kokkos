/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <corsika/accelerator/em/common/EnvironmentTreeSnapshot.hpp>
#include <corsika/accelerator/em/common/EnvironmentModelSnapshot.hpp>

namespace corsika::accelerator::em::kokkos_detail {
  template <typename Execution> class EnvironmentTreeOwner {
    using Node=::corsika::gpu::em::environment_tree::Node;
    using Id=::corsika::gpu::em::environment_tree::NodeId;
    Kokkos::View<Node*,Execution> nodes_;
    Kokkos::View<Id*,Execution> children_,exclusions_;
  public:
    template <typename HostTree>
    explicit EnvironmentTreeOwner(HostTree const& tree)
        :nodes_("environment nodes",tree.nodes.size()),
         children_("environment children",tree.children.size()),
         exclusions_("environment exclusions",tree.exclusions.size()) {
      auto nodes=Kokkos::create_mirror_view(nodes_);
      auto children=Kokkos::create_mirror_view(children_);
      auto exclusions=Kokkos::create_mirror_view(exclusions_);
      for (unsigned i=0;i<tree.nodes.size();++i) nodes(i)=tree.nodes[i];
      for (unsigned i=0;i<tree.children.size();++i) children(i)=tree.children[i];
      for (unsigned i=0;i<tree.exclusions.size();++i) exclusions(i)=tree.exclusions[i];
      Kokkos::deep_copy(nodes_,nodes); Kokkos::deep_copy(children_,children);
      Kokkos::deep_copy(exclusions_,exclusions);
    }
    ::corsika::gpu::em::environment_tree::TreeView view() const {
      return {nodes_.data(),children_.data(),exclusions_.data(),static_cast<unsigned>(nodes_.size())};
    }
  };

  template <typename Execution> class FullEnvironmentOwner {
    using Node=::corsika::gpu::em::environment_tree::Node;
    using Id=::corsika::gpu::em::environment_tree::NodeId;
    using Model=::corsika::gpu::em::environment_tree::Model;
    using Component=::corsika::gpu::em::environment_tree::Component;
    using FullView=::corsika::gpu::em::environment_tree::FullView;
    Kokkos::View<Node*,Execution> nodes_;
    Kokkos::View<Id*,Execution> children_,exclusions_;
    Kokkos::View<Model*,Execution> models_;
    Kokkos::View<Component*,Execution> composition_;
    template <typename View,typename T>
    static void upload(View& out,char const* name,T const* data,unsigned count) {
      if (count && !data) throw std::invalid_argument("null full-environment snapshot array");
      out=View(std::string(name),count);auto host=Kokkos::create_mirror_view(out);
      for (unsigned i=0;i<count;++i) host(i)=data[i];
      Kokkos::deep_copy(out,host);
    }
  public:
    FullEnvironmentOwner()=default;
    explicit FullEnvironmentOwner(FullView const& host) {
      upload(nodes_,"full environment nodes",host.topology.nodes,host.topology.node_count);
      upload(children_,"full environment children",host.topology.children,host.child_count);
      upload(exclusions_,"full environment exclusions",host.topology.exclusions,host.exclusion_count);
      upload(models_,"full environment models",host.models,host.model_count);
      upload(composition_,"full environment composition",host.composition,host.composition_count);
    }
    FullView view() const {
      return {{nodes_.data(),children_.data(),exclusions_.data(),static_cast<unsigned>(nodes_.size())},
              models_.data(),composition_.data(),static_cast<unsigned>(models_.size()),
              static_cast<unsigned>(composition_.size()),static_cast<unsigned>(children_.size()),
              static_cast<unsigned>(exclusions_.size()),nullptr};
    }
    std::size_t deviceBytes() const {
      return nodes_.span()*sizeof(Node)+(children_.span()+exclusions_.span())*sizeof(Id)+
             models_.span()*sizeof(Model)+composition_.span()*sizeof(Component);
    }
  };
} // namespace corsika::accelerator::em::kokkos_detail
