/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once
#include <corsika/accelerator/em/common/EnvironmentModelSnapshot.hpp>
#include <vector>
#include <cstring>
#include <cstdint>

namespace corsika::gpu::em::environment_tree {
  // Own the full export independently of the native model's C++ template type.
  // This remains host-side; FullView is the trivially-copyable kernel payload.
  struct SnapshotStorage {
    std::vector<Node> nodes;
    std::vector<NodeId> children,exclusions;
    std::vector<Model> models;
    std::vector<Component> composition;
    std::vector<std::uintptr_t> native_nodes;
    template <typename Export>
    explicit SnapshotStorage(Export const& e)
        :nodes(e.topology.nodes),children(e.topology.children),exclusions(e.topology.exclusions),
         models(e.models),composition(e.composition) {
      for (auto node:e.topology.native_nodes)
        native_nodes.push_back(reinterpret_cast<std::uintptr_t>(node));
    }
    FullView view() const {
      return {{nodes.data(),children.data(),exclusions.data(),static_cast<unsigned>(nodes.size())},
              models.data(),composition.data(),static_cast<unsigned>(models.size()),
              static_cast<unsigned>(composition.size()),static_cast<unsigned>(children.size()),
              static_cast<unsigned>(exclusions.size()),native_nodes.data()};
    }
  };

  // Initialization/session validation only. Compare named fields, not C++
  // padding bytes or host addresses (the native tree may be recreated).
  inline std::vector<double> snapshotKey(FullView const& tree) {
    std::vector<double> key;
    auto add=[&](auto v){key.push_back(static_cast<double>(v));};
    add(tree.topology.node_count);add(tree.child_count);add(tree.exclusion_count);
    add(tree.model_count);add(tree.composition_count);
    for(unsigned i=0;i<tree.topology.node_count;++i) {
      auto const& n=tree.topology.nodes[i];
      add(n.parent);add(n.child_begin);add(n.child_count);add(n.excluded_begin);add(n.excluded_count);
      add(n.shape);add(n.radius);add(n.model_id);
      for(int a=0;a<3;++a) {add(n.center[a]);add(n.half_extent[a]);for(int b=0;b<3;++b)add(n.axes[a][b]);}
    }
    for(unsigned i=0;i<tree.child_count;++i)add(tree.topology.children[i]);
    for(unsigned i=0;i<tree.exclusion_count;++i)add(tree.topology.exclusions[i]);
    for(unsigned i=0;i<tree.model_count;++i) {
      auto const& m=tree.models[i];
      add(m.density_kind);add(m.rho0_g_cm3);add(m.lambda_m);add(m.inverse_lambda_m);add(m.reference_height_m);
      add(m.refractivity_kind);add(m.refractivity_value);add(m.inverse_reference_density_cm3_g);
      add(m.medium_id);add(m.composition_begin);add(m.composition_count);
      for(int a=0;a<3;++a){add(m.anchor_m[a]);add(m.axis[a]);add(m.magnetic_field_T[a]);}
    }
    for(unsigned i=0;i<tree.composition_count;++i){add(tree.composition[i].code);add(tree.composition[i].fraction);}
    return key;
  }

  // Stable across processes and host pointer addresses. This is a compatibility
  // tag, not a cryptographic digest; archive integrity is checked separately.
  inline std::uint64_t snapshotFingerprint(FullView const& tree) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (double value : snapshotKey(tree)) {
      if (value == 0.) value = 0.; // canonicalize negative zero
      std::uint64_t bits{};
      static_assert(sizeof(bits) == sizeof(value));
      std::memcpy(&bits, &value, sizeof(bits));
      for (unsigned i = 0; i < 8; ++i) {
        hash ^= (bits >> (8*i)) & 255;
        hash *= UINT64_C(1099511628211);
      }
    }
    return hash;
  }
} // namespace corsika::gpu::em::environment_tree
