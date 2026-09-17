#pragma once
#include <corsika/geometry/terrain/DemCoverageData.hpp>
#include <corsika/framework/geometry/LeapFrogTrajectory.hpp>
#include <type_traits>
namespace corsika::terrain::coverage {
// Same SI expression and linear policy as terrain MagneticTracking. Never
// reconstruct curvature from subtracting kilometre-sized endpoint positions.
template<class P,class T>
flat::QuadraticPath trajectory(P const& p,T const& track,CoordinateSystemPtr const& cs) {
  using namespace units::si;
  auto x=p.getPosition();auto d=p.getVelocity().normalized();
  flat::QuadraticPath path{{{x.getX(cs)/1_m,x.getY(cs)/1_m,x.getZ(cs)/1_m},
      {d.getX(cs),d.getY(cs),d.getZ(cs)},true},{0.,0.,0.},track.getLength()/1_m};
  if constexpr(std::is_same_v<std::decay_t<T>,LeapFrogTrajectory>) {
    auto b=p.getNode()->getModelProperties().getMagneticField(x);auto charge=p.getCharge();
    if(charge!=0*constants::e&&b.getNorm()!=0_T) {
      auto perpendicular=p.getMomentum().cross(b).getNorm()/b.getNorm();
      auto radius=convert_HEP_to_SI<MassType::dimension_type>(perpendicular)*constants::c/(abs(charge)*b.getNorm());
      if(perpendicular>=1_eV&&radius<=1.e9_m) {
        auto momentum=constants::c*convert_HEP_to_SI<MassType::dimension_type>(p.getMomentum().getNorm());
        auto q=.5*d.cross(b)*charge/momentum;
        path.quadratic={q.getX(cs)*1_m,q.getY(cs)*1_m,q.getZ(cs)*1_m};
      }
    }
  }
  // Mesh and coverage solvers round independently at a coincident sidewall.
  // Search 10 nm beyond the material endpoint, then arbitrate against actual
  // travelled distance; this cannot skip an earlier physical interaction.
  if(std::isfinite(path.maximum_length_m))path.maximum_length_m+=1.e-8;
  return path;
}
}
