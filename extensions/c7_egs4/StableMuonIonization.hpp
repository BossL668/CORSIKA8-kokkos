#pragma once
#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <cmath>

namespace c7_egs4::application {
// The same two-body energy/momentum conservation as PROPOSAL NaivIonization.
// Compute transverse momentum, rather than recovering a tiny muon angle from
// 1-cos(theta). No change to rates, sampled loss, random draws or final energies.
struct MuonIonizationAngles { double muon_sine{},muon_cosine{},electron_sine{},electron_cosine{}; };
C8_ACCELERATOR_INLINE_FUNCTION inline MuonIonizationAngles stableMuonIonizationAngles(
    double energy,double loss,double muon_mass,double electron_mass) {
  double p2=(energy-muon_mass)*(energy+muon_mass),p=::sqrt(p2);
  double ef=energy-loss,pf=::sqrt((ef-muon_mass)*(ef+muon_mass));
  double pe=::sqrt(loss*(loss+2.*electron_mass));
  double tmax=2.*electron_mass*p2/(muon_mass*muon_mass+2.*energy*electron_mass+electron_mass*electron_mass);
  double transverse2=2.*electron_mass*loss*(1.-loss/tmax);
  double transverse=::sqrt(transverse2>0.?transverse2:0.);
  double parallel=loss*(energy+electron_mass)/p;
  return {transverse/pf,(p-parallel)/pf,transverse/pe,parallel/pe};
}
C8_ACCELERATOR_INLINE_FUNCTION inline void deflectWithSine(
    double const input[3],double sine,double cosine,double azimuth,double output[3]) {
  double transverse=::sqrt(input[0]*input[0]+input[1]*input[1]);
  double cp=transverse>0.?input[0]/transverse:1.;
  double sp=transverse>0.?input[1]/transverse:0.;
  double x=sine*::cos(azimuth),y=sine*::sin(azimuth);
  // Keep the original Cartesian3D local-z reconstruction/rotation convention,
  // but do not discard sine when cosine rounds to 1 for a high-energy muon.
  double z2=1.-x*x-y*y,z=::sqrt(z2>0.?z2:0.);if(cosine<0.)z=-z;
  output[0]=z*input[0]+x*input[2]*cp-y*sp;
  output[1]=z*input[1]+x*input[2]*sp+y*cp;
  output[2]=z*input[2]-x*transverse;
}
} // namespace c7_egs4::application
