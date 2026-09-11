/* Logical-side terrain boundary candidates, in metres.
 * Mirrors modules/terrain/TerrainBoundaryTracking.hpp. This query does not
 * move a particle, sample a process, deposit energy, or terminate a shower. */
#pragma once

#include <corsika/geometry/terrain/FlatTerrain.hpp>

#if defined(KOKKOS_INLINE_FUNCTION)
#define C8_TERRAIN_BOUNDARY_INLINE KOKKOS_INLINE_FUNCTION
#else
#define C8_TERRAIN_BOUNDARY_INLINE inline
#endif

namespace corsika::terrain::flat {

enum class Crossing : std::uint32_t { None, EnterRock, ExitRock };

struct BoundaryQuery {
  Vec3 origin;
  Vec3 direction; // Unit vector; logical side comes from the transport node.
  bool logically_inside{};
};

struct BoundaryCandidate {
  Hit hit;
  Crossing crossing{Crossing::None};
};

// CPU tracking deliberately searches backwards to include a surface rounded
// behind the endpoint. Only the query origin is shifted, not the particle.
// Its 1 nm minimum flight is retained here (not an arbitrary surface padding).
C8_TERRAIN_BOUNDARY_INLINE BoundaryCandidate nextBoundary(
    View const& view, BoundaryQuery const& query) {
  constexpr double tolerance_m = 1.e-8;
  constexpr double minimum_flight_m = 1.e-9;
  Ray ray{{query.origin.x - tolerance_m * query.direction.x,
           query.origin.y - tolerance_m * query.direction.y,
           query.origin.z - tolerance_m * query.direction.z},
          query.direction};
  auto hit = intersect(view, ray, query.logically_inside ? 1 : -1);
  if (!hit.found()) return {};
  double const distance = hit.distance - tolerance_m;
  hit.distance = distance > minimum_flight_m ? distance : minimum_flight_m;
  return {hit, query.logically_inside ? Crossing::ExitRock : Crossing::EnterRock};
}

} // namespace corsika::terrain::flat
#undef C8_TERRAIN_BOUNDARY_INLINE
