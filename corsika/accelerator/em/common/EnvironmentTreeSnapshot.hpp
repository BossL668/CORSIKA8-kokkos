/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once

#include <corsika/geometry/NextEnvironmentBoundary.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#if defined(__CUDACC__)
#include <math_constants.h>
#endif

namespace corsika::gpu::em::environment_tree {
  using NodeId = std::int32_t;
  constexpr NodeId Outside = -1;
  constexpr NodeId InvalidTopology = -2;
  C8_ACCELERATOR_INLINE_FUNCTION inline double infinity() {
#if defined(__CUDA_ARCH__)
    return CUDART_INF;
#else
    return std::numeric_limits<double>::infinity();
#endif
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline double notANumber() {
#if defined(__CUDA_ARCH__)
    return CUDART_NAN;
#else
    return std::numeric_limits<double>::quiet_NaN();
#endif
  }
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
    C8_ACCELERATOR_INLINE_FUNCTION NodeId parent(NodeId i) const { return nodes[i].parent; }
    C8_ACCELERATOR_INLINE_FUNCTION unsigned childCount(NodeId i) const { return nodes[i].child_count; }
    C8_ACCELERATOR_INLINE_FUNCTION NodeId child(NodeId i, unsigned j) const {
      return children[nodes[i].child_begin + j];
    }
    C8_ACCELERATOR_INLINE_FUNCTION unsigned excludedCount(NodeId i) const { return nodes[i].excluded_count; }
    C8_ACCELERATOR_INLINE_FUNCTION NodeId excluded(NodeId i, unsigned j) const {
      return exclusions[nodes[i].excluded_begin + j];
    }
  };

  // Native C8's CPU geometry is evaluated without FMA contraction in the
  // reference build. Preserve rounding locally: Earth-radius discriminants
  // subtract large nearly equal values. Do not change global physics flags.
  C8_ACCELERATOR_INLINE_FUNCTION inline double product(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a,b);
#else
    return a*b;
#endif
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline double sum(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a,b);
#else
    return a+b;
#endif
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline double difference(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dsub_rn(a,b);
#else
    return a-b;
#endif
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline double dot(double const* a, double const* b) {
    return sum(sum(product(a[0],b[0]),product(a[1],b[1])),product(a[2],b[2]));
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool contains(Node const& node, double const* p) {
    if (node.shape == Shape::Universe) return true;
    double delta[3]{p[0]-node.center[0], p[1]-node.center[1], p[2]-node.center[2]};
    if (node.shape == Shape::Sphere) return dot(delta, delta) < node.radius*node.radius;
    if (node.shape == Shape::SeparationPlane) return dot(delta, node.axes[0]) <= 0.;
    for (int axis=0; axis<3; ++axis)
      if (!(::fabs(dot(delta, node.axes[axis])) < node.half_extent[axis])) return false;
    return true;
  }

  /** Iterative equivalent of VolumeTreeNode::getContainingNode. */
  C8_ACCELERATOR_INLINE_FUNCTION inline NodeId locate(TreeView const& tree,
      double const* p, NodeId start=0) {
    auto current=start;
    for (unsigned hop=0; hop<tree.node_count; ++hop) {
      if (current<0 || static_cast<unsigned>(current)>=tree.node_count) return InvalidTopology;
      if (!contains(tree.nodes[current],p)) return Outside;
      NodeId next=Outside;
      for (unsigned i=0; i<tree.childCount(current); ++i) {
        auto child=tree.child(current,i);
        if (contains(tree.nodes[child],p)) { next=child; break; }
      }
      if (next==Outside) for (unsigned i=0; i<tree.excludedCount(current); ++i) {
        auto excluded=tree.excluded(current,i);
        if (contains(tree.nodes[excluded],p)) { next=excluded; break; }
      }
      if (next==Outside) return current;
      current=next;
    }
    return InvalidTopology; // Cyclic exclusion, never silently select a material.
  }

  struct Intersections {
    bool valid{};
    double entry{}, exit{};
    C8_ACCELERATOR_INLINE_FUNCTION bool hasIntersections() const { return valid; }
    C8_ACCELERATOR_INLINE_FUNCTION double getEntry() const { return entry; }
    C8_ACCELERATOR_INLINE_FUNCTION double getExit() const { return exit; }
  };

  // Same shape algebra as TrackingStraight; the parameter is time if velocity
  // is supplied, or distance if the direction is supplied.
  C8_ACCELERATOR_INLINE_FUNCTION inline Intersections straightIntersect(
      Node const& node, double const* position, double const* velocity) {
    double const inf=infinity();
    if (node.shape==Shape::Universe) return {true,-inf,inf};
    double delta[3]{position[0]-node.center[0],position[1]-node.center[1],position[2]-node.center[2]};
    if (node.shape==Shape::Sphere) {
      auto const v2=dot(velocity,velocity), vd=dot(velocity,delta);
      auto const discriminant=difference(product(vd,vd),
          product(v2,difference(dot(delta,delta),product(node.radius,node.radius))));
      if (!(discriminant>0.)) return {};
      double const root=::sqrt(discriminant);
      return {true,(-vd-root)/v2,(-vd+root)/v2};
    }
    if (node.shape==Shape::SeparationPlane) {
      double const vn=dot(velocity,node.axes[0]);
      if (vn==0.) return {};
      // C8 Intersections(t) represents (entry=t, exit=infinity), independently
      // of the direction of travel; preserve this native convention exactly.
      return {true,-dot(delta,node.axes[0])/vn,inf};
    }
    double entry=-inf,exit=inf;
    for (int axis=0; axis<3; ++axis) {
      auto const p=dot(delta,node.axes[axis]), v=dot(velocity,node.axes[axis]);
      double const t1=(node.half_extent[axis]-p)/v;
      double const t2=(-node.half_extent[axis]-p)/v;
      // Preserve C8's operand order, including the NaN case at a slab edge.
      auto const lo=t1>t2?t2:t1, hi=t1>t2?t1:t2;
      if (axis==0) { entry=lo; exit=hi; }
      else { entry=entry<lo?lo:entry; exit=hi<exit?hi:exit; }
    }
    return exit>entry?Intersections{true,entry,exit}:Intersections{};
  }

  struct StraightAccess : TreeView {
    double position[3]{}, velocity[3]{};
    C8_ACCELERATOR_INLINE_FUNCTION Intersections intersect(NodeId id) const {
      return straightIntersect(nodes[id],position,velocity);
    }
  };
  static_assert(std::is_trivially_copyable_v<Node> && std::is_trivially_copyable_v<TreeView>);
} // namespace corsika::gpu::em::environment_tree
