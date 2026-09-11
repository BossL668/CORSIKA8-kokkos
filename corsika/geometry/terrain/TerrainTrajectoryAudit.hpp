// Diagnostic position only: never used to move a particle or integrate losses.
#pragma once
#include <corsika/geometry/terrain/FlatTerrainPrimitives.hpp>
#if defined(KOKKOS_INLINE_FUNCTION)
#define C8_TERRAIN_AUDIT_INLINE KOKKOS_INLINE_FUNCTION
#else
#define C8_TERRAIN_AUDIT_INLINE inline
#endif
namespace corsika::terrain::flat {
// A leapfrog path is quadratic in time: r(u)=r0+u*v0*dt+u^2*a.
// Therefore r(1/2)=r0+(r1-r0+v0*dt)/4. Post-step direction cannot be
// used here: the continuous process may already have applied Moliere scattering.
// CPU provides its original velocity*dt; device provides direction*flight L.
// Do not replace the scalar PROPOSAL chord-based energy-loss integration.
C8_TERRAIN_AUDIT_INLINE Vec3 quadraticMidpoint(
    Vec3 start,Vec3 end,Vec3 initial_displacement) {
  return {start.x+.25*((end.x-start.x)+initial_displacement.x),
          start.y+.25*((end.y-start.y)+initial_displacement.y),
          start.z+.25*((end.z-start.z)+initial_displacement.z)};
}
} // namespace corsika::terrain::flat
#undef C8_TERRAIN_AUDIT_INLINE
