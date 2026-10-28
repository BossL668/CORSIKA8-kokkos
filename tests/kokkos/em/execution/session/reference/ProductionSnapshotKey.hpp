/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
// Original host annotations expand to empty in this host-only reference.
#define C8_ACCELERATOR_INLINE_FUNCTION
namespace corsika::gpu::em::environment_tree {
  using NodeId = std::int32_t;
  constexpr NodeId Outside = -1;
  enum class Shape : std::uint32_t { Universe, Sphere, Box, SeparationPlane };
  struct Node {
    NodeId parent{Outside};
    std::uint32_t child_begin{}, child_count{}, excluded_begin{}, excluded_count{};
    Shape shape{Shape::Universe};
    // World-frame center, local-frame axes for boxes, outward normal for planes.
    double center[3]{};
    double axes[3][3]{};
    double half_extent[3]{};
    double radius{};
    // Model payload index, not a reconstructed layer number. The host exporter
    // retains every node, including nodes with no model properties.
    std::int32_t model_id{-1};
  };
  struct TreeView {
    Node const* nodes{};
    NodeId const* children{};
    NodeId const* exclusions{};
    std::uint32_t node_count{};
    C8_ACCELERATOR_INLINE_FUNCTION NodeId parent(NodeId i) const {
      return nodes[i].parent;
    }
    C8_ACCELERATOR_INLINE_FUNCTION unsigned childCount(NodeId i) const {
      return nodes[i].child_count;
    }
    C8_ACCELERATOR_INLINE_FUNCTION NodeId child(NodeId i, unsigned j) const {
      return children[nodes[i].child_begin + j];
    }
    C8_ACCELERATOR_INLINE_FUNCTION unsigned excludedCount(NodeId i) const {
      return nodes[i].excluded_count;
    }
    C8_ACCELERATOR_INLINE_FUNCTION NodeId excluded(NodeId i, unsigned j) const {
      return exclusions[nodes[i].excluded_begin + j];
    }
  };
  enum class Density : unsigned {
    Homogeneous,
    SlidingPlanarExponential,
    FlatExponential
  };
  enum class Refractivity : unsigned { Unavailable, Uniform, GladstoneDale };
  struct Model {
    Density density_kind{};
    double rho0_g_cm3{}, lambda_m{}, inverse_lambda_m{}, reference_height_m{};
    double anchor_m[3]{}, axis[3]{};
    double magnetic_field_T[3]{};
    Refractivity refractivity_kind{Refractivity::Unavailable};
    double refractivity_value{}, inverse_reference_density_cm3_g{};
    int medium_id{-1};
    unsigned composition_begin{}, composition_count{};
  };
  struct Component {
    int code;
    double fraction;
  };
  struct FullView {
    TreeView topology{};
    Model const* models{};
    Component const* composition{};
    unsigned model_count{}, composition_count{}, child_count{}, exclusion_count{};
    // Host-only identity map. Device owners clear this pointer after upload.
    // Logical-node identity must not be reconstructed from a rounded endpoint.
    std::uintptr_t const* native_nodes{};
    C8_ACCELERATOR_INLINE_FUNCTION Model const* model(NodeId id) const {
      if (!topology.nodes || !models || id < 0 ||
          static_cast<unsigned>(id) >= topology.node_count)
        return nullptr;
      auto index = topology.nodes[id].model_id;
      return index >= 0 && static_cast<unsigned>(index) < model_count ? models + index
                                                                      : nullptr;
    }
  };
  inline std::vector<double> snapshotKey(FullView const& tree) {
    std::vector<double> key;
    auto add = [&](auto v) { key.push_back(static_cast<double>(v)); };
    add(tree.topology.node_count);
    add(tree.child_count);
    add(tree.exclusion_count);
    add(tree.model_count);
    add(tree.composition_count);
    for (unsigned i = 0; i < tree.topology.node_count; ++i) {
      auto const& n = tree.topology.nodes[i];
      add(n.parent);
      add(n.child_begin);
      add(n.child_count);
      add(n.excluded_begin);
      add(n.excluded_count);
      add(n.shape);
      add(n.radius);
      add(n.model_id);
      for (int a = 0; a < 3; ++a) {
        add(n.center[a]);
        add(n.half_extent[a]);
        for (int b = 0; b < 3; ++b) add(n.axes[a][b]);
      }
    }
    for (unsigned i = 0; i < tree.child_count; ++i) add(tree.topology.children[i]);
    for (unsigned i = 0; i < tree.exclusion_count; ++i) add(tree.topology.exclusions[i]);
    for (unsigned i = 0; i < tree.model_count; ++i) {
      auto const& m = tree.models[i];
      add(m.density_kind);
      add(m.rho0_g_cm3);
      add(m.lambda_m);
      add(m.inverse_lambda_m);
      add(m.reference_height_m);
      add(m.refractivity_kind);
      add(m.refractivity_value);
      add(m.inverse_reference_density_cm3_g);
      add(m.medium_id);
      add(m.composition_begin);
      add(m.composition_count);
      for (int a = 0; a < 3; ++a) {
        add(m.anchor_m[a]);
        add(m.axis[a]);
        add(m.magnetic_field_T[a]);
      }
    }
    for (unsigned i = 0; i < tree.composition_count; ++i) {
      add(tree.composition[i].code);
      add(tree.composition[i].fraction);
    }
    return key;
  }
  inline std::uint64_t snapshotFingerprint(FullView const& tree) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (double value : snapshotKey(tree)) {
      if (value == 0.) value = 0.; // canonicalize negative zero
      std::uint64_t bits{};
      static_assert(sizeof(bits) == sizeof(value));
      std::memcpy(&bits, &value, sizeof(bits));
      for (unsigned i = 0; i < 8; ++i) {
        hash ^= (bits >> (8 * i)) & 255;
        hash *= UINT64_C(1099511628211);
      }
    }
    return hash;
  }
} // namespace corsika::gpu::em::environment_tree
#undef C8_ACCELERATOR_INLINE_FUNCTION
