/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once
#include <corsika/accelerator/em/common/EnvironmentTreeSnapshot.hpp>

namespace corsika::gpu::em::environment_tree {
  enum class Density : unsigned { Homogeneous, SlidingPlanarExponential, FlatExponential };
  enum class Refractivity : unsigned { Unavailable, Uniform, GladstoneDale };
  struct Model {
    Density density_kind{};
    double rho0_g_cm3{},lambda_m{},inverse_lambda_m{},reference_height_m{};
    double anchor_m[3]{},axis[3]{};
    double magnetic_field_T[3]{};
    Refractivity refractivity_kind{Refractivity::Unavailable};
    double refractivity_value{},inverse_reference_density_cm3_g{};
    int medium_id{-1};
    unsigned composition_begin{},composition_count{};
  };
  struct Component {int code;double fraction;};
  struct FullView {
    TreeView topology{};
    Model const* models{};
    Component const* composition{};
    unsigned model_count{},composition_count{},child_count{},exclusion_count{};
    // Host-only identity map. Device owners clear this pointer after upload.
    // Logical-node identity must not be reconstructed from a rounded endpoint.
    std::uintptr_t const* native_nodes{};
    C8_ACCELERATOR_INLINE_FUNCTION Model const* model(NodeId id) const {
      if (!topology.nodes || !models || id<0 || static_cast<unsigned>(id)>=topology.node_count) return nullptr;
      auto index=topology.nodes[id].model_id;
      return index>=0 && static_cast<unsigned>(index)<model_count?models+index:nullptr;
    }
  };

  C8_ACCELERATOR_INLINE_FUNCTION inline double density(Model const& m,double const* p) {
    if (m.density_kind==Density::Homogeneous) return m.rho0_g_cm3;
    double delta[3]{p[0]-m.anchor_m[0],p[1]-m.anchor_m[1],p[2]-m.anchor_m[2]};
    auto h=m.density_kind==Density::SlidingPlanarExponential?::sqrt(dot(delta,delta)):dot(delta,m.axis);
    return m.rho0_g_cm3*::exp(m.inverse_lambda_m*(h-m.reference_height_m));
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline double refractiveIndex(Model const& m,double const* p) {
    if (m.refractivity_kind==Refractivity::Uniform) return m.refractivity_value;
    if (m.refractivity_kind==Refractivity::GladstoneDale)
      return m.refractivity_value*(density(m,p)*m.inverse_reference_density_cm3_g)+1.;
    return notANumber();
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline double projectedAxis(Model const& m,
      double const* p,double const* direction) {
    if (m.density_kind==Density::FlatExponential) return dot(direction,m.axis);
    double delta[3]{p[0]-m.anchor_m[0],p[1]-m.anchor_m[1],p[2]-m.anchor_m[2]};
    return dot(delta,direction)/::sqrt(dot(delta,delta));
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline double grammage(Model const& m,
      double const* p,double const* direction,double length_m) {
    if (length_m==0.) return 0.;
    auto const rho=density(m,p);
    if (m.density_kind==Density::Homogeneous) return length_m*rho*100.;
    auto const projection=projectedAxis(m,p,direction);
    if (projection==0.) return length_m*rho*100.;
    return rho*(m.lambda_m/projection)*::expm1(projection*length_m*m.inverse_lambda_m)*100.;
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline double arclength(Model const& m,
      double const* p,double const* direction,double x_g_cm2) {
    auto const rho=density(m,p);
    if (m.density_kind==Density::Homogeneous) return x_g_cm2/(rho*100.);
    auto const projection=projectedAxis(m,p,direction);
    if (projection==0.) return x_g_cm2/(rho*100.);
    auto const logArg=x_g_cm2*.01*m.inverse_lambda_m*projection/rho;
    return logArg>-1.?m.lambda_m/projection*::log1p(logArg):infinity();
  }
} // namespace corsika::gpu::em::environment_tree
