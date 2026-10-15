#pragma once
#include <cmath>
#include <corsika/accelerator/em/common/SphericalAtmosphere.hpp>

namespace corsika::gpu::em {
  // Optional, non-absorbing straight-track material interface. Default arguments
  // retain the air-shower path exactly. The caller owns logical material routing.
  // Curved terrain intersections are deliberately not supported by this contract.
  struct ExternalTransportBoundary {
    double distance_m{HUGE_VAL};
    bool enabled{};
    bool disable_observation{};
    bool allow_unbounded_inward_grammage{};
  };
  C8_ACCELERATOR_INLINE_FUNCTION inline AtmosphereGrammageQuery geometryCompetitionGrammage(
      EnvironmentSnapshot const& env,int layer,double const position[3],
      double const direction[3],double distance,ExternalTransportBoundary const& boundary) {
    auto result=atmosphereGrammage(env,layer,position,direction,distance);
    if(env.geometry==EnvironmentGeometry::NodeTree &&
        result.status==AtmosphereStatus::GrammageOutOfRange &&
        result.value==HUGE_VAL && atmosphere_detail::finite(distance) && distance>0.) {
      auto const* model=env.node_tree.model(layer);
      if(model &&
          (model->density_kind==environment_tree::Density::SlidingPlanarExponential ||
           model->density_kind==environment_tree::Density::FlatExponential)) {
        auto const rho=environment_tree::density(*model,position);
        auto const projection=environment_tree::projectedAxis(*model,position,direction);
        auto const growth=projection*model->inverse_lambda_m;
        // C8 competes finite arclengths, not the integrated depth of the
        // entire (possibly Earth-sized) geometry candidate. For a positive,
        // increasing exponential, +infinity is a valid comparison sentinel.
        // Do NOT relax atmosphereGrammage itself: the selected physical step
        // must still have finite length and finite traversed column depth.
        if(atmosphere_detail::finite(rho) && rho>0. &&
            atmosphere_detail::finite(growth) && growth>0. &&
            atmosphere_detail::finite(model->lambda_m) &&
            model->lambda_m*model->inverse_lambda_m>0.)
          result.status=AtmosphereStatus::Success;
      }
    }
    if(boundary.allow_unbounded_inward_grammage &&
        result.status==AtmosphereStatus::GrammageOutOfRange &&
        env.geometry==EnvironmentGeometry::SphericalLayers && layer>=0 &&
        layer<static_cast<int>(env.number_of_layers)) {
      double radial[3]{};
      auto radius=atmosphere_detail::radiusVector(env,position,radial);
      auto const& m=env.atmosphere_layers[layer];
      // CPU's lowest atmosphere is a ball. With no absorbing ground, its
      // far-side geometry competitor can have an overflowing positive column
      // depth. Infinity is a comparison sentinel ONLY; the actually selected
      // finite interaction/continuous step still uses the ordinary integral.
      if(radius>0. && m.density_model==DensityModel::Exponential &&
          m.density_parameter_c<0. && distance>0. &&
          atmosphere_detail::dot(radial,direction)<0.)
        result={AtmosphereStatus::Success,0,HUGE_VAL};
    }
    return result;
  }
}
