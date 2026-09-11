// Host-side admission for a CPU -> terrain-device handoff. No RNG draws,
// energy/weight changes, or modification of the caller's CPU particle stack.
#pragma once
#include <corsika/accelerator/em/common/Types.hpp>
#include <cmath>
#include <stdexcept>
namespace corsika::terrain {
inline bool canonicalizeCpuDirection(gpu::em::EmParticleState& state) {
  double norm2=0.;
  for(double d:state.direction) {
    if(!std::isfinite(d))throw std::invalid_argument("nonfinite CPU terrain direction");
    norm2+=d*d;
  }
  // CPU rotations/boosts can leave a few 1e-12 in the norm. This was admitted
  // by the former 1e-10 host gate, then rejected by the 1e-12 device geometry
  // gate as UnsupportedGeometry. Canonicalize only this roundoff interval;
  // arbitrary non-unit vectors remain hard failures, not silently repaired.
  double error=std::abs(norm2-1.);
  if(error>1.e-10)throw std::invalid_argument("CPU terrain direction exceeds roundoff admission bound");
  if(error<=1.e-12)return false; // preserve already valid bits
  double norm=std::sqrt(norm2);
  for(double& d:state.direction)d/=norm;
  return true;
}
} // namespace corsika::terrain
