/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once
#include <corsika/accelerator/em/common/EnvironmentTreeBuilder.hpp>
#include <corsika/accelerator/em/common/EnvironmentModelSnapshot.hpp>
#include <corsika/media/EnvironmentModelExport.hpp>

namespace corsika::gpu::em::environment_tree {
  template <typename NativeNode> struct FullHostTree {
    HostTree<NativeNode> topology;
    std::vector<Model> models;
    std::vector<Component> composition;
  };

  // Interface must supply the four C8 air-environment properties: density,
  // composition, magnetic field, medium ID and refractive index. No parameter
  // is inferred from an atmosphere name or fitted to getMassDensity samples.
  template <typename NativeNode>
  FullHostTree<NativeNode> exportFullEnvironmentTree(NativeNode const& root,
      CoordinateSystemPtr const& frame) {
    FullHostTree<NativeNode> result;
    std::unordered_map<void const*,int> model_ids;
    result.topology=exportEnvironmentTree(root,frame,[&](NativeNode const& node,unsigned) {
      if (!node.hasModelProperties()) return -1;
      auto const& native=node.getModelProperties();
      auto key=static_cast<void const*>(&native);
      if (model_ids.count(key)) return model_ids.at(key);
      auto d=dynamic_cast<IDensityExport const*>(&native);
      auto b=dynamic_cast<IUniformFieldExport const*>(&native);
      auto n=dynamic_cast<IRefractivityExport const*>(&native);
      if (!d || !b || !n)
        throw std::invalid_argument("C8 model has no exact full-snapshot exporter; no approximate replacement allowed");
      auto density=d->exportDensity();auto field=b->exportUniformField();auto refr=n->exportRefractivity();
      Model m;
      m.density_kind=static_cast<Density>(density.kind);
      m.rho0_g_cm3=density.rho0/(1_g/(1_cm*1_cm*1_cm));
      m.lambda_m=density.lambda/1_m;
      m.inverse_lambda_m=m.lambda_m!=0.?1./m.lambda_m:0.;
      m.reference_height_m=density.reference_height/1_m;
      auto anchor=density.anchor.getCoordinates(frame);
      auto axis=density.axis.getComponents(frame);
      auto magnetic=field.getComponents(frame);
      for (int i=0;i<3;++i) {
        m.anchor_m[i]=anchor[i]/1_m;m.axis[i]=axis[i].magnitude();m.magnetic_field_T[i]=magnetic[i]/1_T;
      }
      m.refractivity_kind=refr.kind==ExportRefractivity::Kind::Uniform?Refractivity::Uniform:Refractivity::GladstoneDale;
      m.refractivity_value=refr.value;
      m.inverse_reference_density_cm3_g=refr.inverse_density*(1_g/(1_cm*1_cm*1_cm));
      m.medium_id=static_cast<int>(native.getMedium());
      auto const& comp=native.getNuclearComposition();
      m.composition_begin=result.composition.size();m.composition_count=comp.getSize();
      for (unsigned i=0;i<comp.getSize();++i)
        result.composition.push_back({static_cast<int>(comp.getComponents()[i]),comp.getFractions()[i]});
      auto id=static_cast<int>(result.models.size());result.models.push_back(m);model_ids.emplace(key,id);
      return id;
    });
    return result;
  }
} // namespace corsika::gpu::em::environment_tree
