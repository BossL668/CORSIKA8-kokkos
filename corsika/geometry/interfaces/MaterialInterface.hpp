/* Two-sided, oriented mesh interface. No material names, RNG or deposition.
 * Normals point from the enclosed region into the enclosing region. The caller
 * supplies a validated closed mesh; self-intersections are not certified here.
 * Region IDs and calculator-bank indices are deliberately separate. */
#pragma once
#include <corsika/geometry/terrain/TerrainCurvedBoundary.hpp>
#include <cstddef>
#include <stdexcept>

#if defined(KOKKOS_INLINE_FUNCTION)
#define C8_INTERFACE_INLINE KOKKOS_INLINE_FUNCTION
#else
#define C8_INTERFACE_INLINE inline
#endif

namespace corsika::interfaces {
struct MaterialInterface {
  int outside_region{0}, inside_region{1};
  int outside_material{0}, inside_material{1};

  C8_INTERFACE_INLINE bool containsRegion(int region) const {
    return region==outside_region || region==inside_region;
  }
  C8_INTERFACE_INLINE int material(int region) const {
    return region==outside_region ? outside_material :
           region==inside_region ? inside_material : -1;
  }
  C8_INTERFACE_INLINE int opposite(int region) const {
    return region==outside_region ? inside_region :
           region==inside_region ? outside_region : -1;
  }
  void validate(std::size_t banks) const {
    if(outside_region<0 || inside_region<0 || outside_region==inside_region ||
       outside_material<0 || inside_material<0 ||
       static_cast<std::size_t>(outside_material)>=banks ||
       static_cast<std::size_t>(inside_material)>=banks)
      throw std::invalid_argument("invalid two-sided region/material binding");
  }
};

struct InterfaceCrossing {
  terrain::flat::Hit hit;
  int from_region{-1}, to_region{-1};
  bool invalid_region{};
};

// Query the SAME straight/quadratic solver as scalar terrain tracking. This
// function cannot change particle state; the transport competition must first
// establish that this surface, rather than an interaction/cut, was reached.
C8_INTERFACE_INLINE InterfaceCrossing nextCrossing(
    terrain::flat::View const& mesh, MaterialInterface const& binding,
    int region, terrain::flat::QuadraticPath path) {
  if(!binding.containsRegion(region)) return {{},region,-1,true};
  path.start.logically_inside=region==binding.inside_region;
  auto candidate=terrain::flat::nextCurvedBoundary(mesh,path);
  return {candidate.hit,region,candidate.hit.found()?binding.opposite(region):region,false};
}
} // namespace corsika::interfaces
#undef C8_INTERFACE_INLINE
