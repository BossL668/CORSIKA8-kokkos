#pragma once
#include <corsika/detail/modules/egs4/Egs4C8Air.hpp>

namespace c7_egs4::air_test {
using namespace c8_adapter;
::corsika::gpu::em::EnvironmentSnapshot environment() {
  using namespace ::corsika::gpu::em;EnvironmentSnapshot e;
  // The schema and Linsley values used by C8 makeCorsika7AtmosphereSnapshot.
  constexpr double R=6.371315e6;
  double heights[]={0.,4000.,10000.,40000.,100000.,112800.};
  double offsets[]={1222.6562,1144.9069,1305.5948,540.1778,1.};
  double scales[]={994186.38,878153.55,636143.04,772170.16,1.e9};
  e.number_of_layers=5;e.earth_center_m[0]=123.;e.earth_center_m[1]=-456.;e.earth_center_m[2]=789.;
  e.observation_plane_normal[2]=1.;e.observation_radius_m=R+1100.;
  for(int j=0;j<3;++j)e.observation_plane_point_m[j]=e.earth_center_m[j];
  e.observation_plane_point_m[2]+=R+1100.;
  e.magnetic_field_T[0]=2.e-5;e.magnetic_field_T[1]=1.e-5;e.magnetic_field_T[2]=-4.e-5;
  for(int i=0;i<5;++i) {
    auto& l=e.atmosphere_layers[i];l.inner_radius_m=R+heights[i];l.outer_radius_m=R+heights[i+1];
    l.density_parameter_a=offsets[i]/scales[i];l.density_parameter_b=R;l.density_parameter_c=-scales[i]*.01;
    l.density_model=i<4?DensityModel::Exponential:DensityModel::Homogeneous;l.medium_id=3;
  }
  return e;
}
KOKKOS_INLINE_FUNCTION C8Particle input(AirEnvironment const& e,int pdg,double energy,double h,int serial) {
  C8Particle p;p.pid=pdg;p.energy_GeV=energy*.001;p.medium_id=3;p.weight=1.25;p.history_id=serial+1;p.step_id=4;
  p.position_m[0]=e.observer_frame.origin_m.x+10.;p.position_m[1]=e.observer_frame.origin_m.y+20.;
  p.position_m[2]=e.observer_frame.origin_m.z+h;p.direction[0]=.3;p.direction[1]=.4;p.direction[2]=-::sqrt(.75);
  p.time_s=1.e-6;return p;
}
} // namespace c7_egs4::air_test
