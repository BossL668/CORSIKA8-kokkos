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
