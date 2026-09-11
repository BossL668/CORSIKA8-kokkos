/* Shared straight-ray geometry for scalar and Kokkos terrain transport.
 * Original indexed vertices are essential: reconstructing a shared vertex as
 * origin + edge can give different bits in its two incident triangles.
 * No physical displacement, distance epsilon, normal averaging, or RNG here.
 */
#pragma once

#include <corsika/geometry/terrain/FlatTerrainPrimitives.hpp>

#if defined(KOKKOS_INLINE_FUNCTION)
#define C8_INDEXED_TERRAIN_INLINE KOKKOS_INLINE_FUNCTION
#else
#define C8_INDEXED_TERRAIN_INLINE inline
#endif

namespace corsika::terrain::indexed {
using flat::Vec3;
using flat::Ray;
using flat::Node;

using Triangle = flat::IndexedTriangle;
struct View {
  Node const* nodes{};
  Triangle const* triangles{};
  std::uint32_t const* indices{};
  Vec3 const* vertices{};
  std::uint32_t node_count{};
};
enum class Feature : std::uint32_t { Vertex, Edge, Interior, None };
struct Hit {
  double distance = flat::NoHitDistance;
  std::uint32_t triangle = flat::NoTriangle;
  Feature feature = Feature::None;
  std::uint32_t first = flat::NoTriangle, second = flat::NoTriangle;
  C8_INDEXED_TERRAIN_INLINE bool found() const {
    return triangle != flat::NoTriangle;
  }
};
static_assert(std::is_trivially_copyable<View>::value, "device geometry is POD");

// Explicit operation order, not a global --fmad=false policy. The symmetric
// product residuals make determinant(a,b) = -determinant(b,a) under round-to-
// nearest (ignoring the sign of zero). All backends use the same fma calls.
C8_INDEXED_TERRAIN_INLINE double determinant(double ax, double ay,
                                             double bx, double by) {
  double p = std::fma(ax, by, 0.0), q = std::fma(ay, bx, 0.0);
  double pe = std::fma(ax, by, -p), qe = std::fma(ay, bx, -q);
  return (p - q) + (pe - qe);
}
C8_INDEXED_TERRAIN_INLINE double dot(Vec3 a, Vec3 b) {
  return std::fma(a.x, b.x, std::fma(a.y, b.y, std::fma(a.z, b.z, 0.0)));
}
C8_INDEXED_TERRAIN_INLINE double component(Vec3 v, int axis) {
  return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

// Compensated local coordinates for nearly cancelled edge functions. Keeping
// the subtraction residual matters: (-1 - origin.x) can discard an input ULP
// before an otherwise watertight edge test even starts. This is not an exact
// arbitrary-precision predicate; its limits still need adversarial acceptance.
struct Twofold { double high, low; };
C8_INDEXED_TERRAIN_INLINE Twofold sum(double a, double b) {
  double s = a + b, part = s - a;
  return {s, (a - (s - part)) + (b - part)};
}
C8_INDEXED_TERRAIN_INLINE Twofold add(Twofold a, Twofold b) {
  auto s = sum(a.high, b.high);
  return sum(s.high, s.low + (a.low + b.low));
}
C8_INDEXED_TERRAIN_INLINE Twofold subtract(Twofold a, Twofold b) {
  return add(a, {-b.high, -b.low});
}
C8_INDEXED_TERRAIN_INLINE Twofold multiply(Twofold a, Twofold b) {
  double p = std::fma(a.high, b.high, 0.0);
  double error = std::fma(a.high, b.high, -p);
  double cross = std::fma(a.high, b.low, std::fma(a.low, b.high, 0.0));
  return sum(p, error + (cross + std::fma(a.low, b.low, 0.0)));
}
C8_INDEXED_TERRAIN_INLINE double compensatedEdge(
    Twofold ax, Twofold ay, Twofold bx, Twofold by) {
  auto result = subtract(multiply(ax, by), multiply(ay, bx));
  return result.high != 0. ? result.high : result.low;
}

// Inclusive slab intervals. A genuinely parallel ray is handled explicitly;
// replacing 1/0 by an arbitrary large reciprocal can drop valid distant hits.
// One outward ULP protects rounded endpoints, not a metric boundary padding.
C8_INDEXED_TERRAIN_INLINE bool slab(Node const& n, Ray const& ray) {
  double low = -flat::NoHitDistance, high = flat::NoHitDistance;
  for (int k = 0; k < 3; ++k) {
    double d = component(ray.direction, k), p = component(ray.origin, k);
    double a = component(n.low, k), b = component(n.high, k);
    if (d == 0.) {
      if (p < a || p > b) return false;
      continue;
    }
    a = (a - p) / d;
    b = (b - p) / d;
    if (a > b) { double tmp = a; a = b; b = tmp; }
    a = std::nextafter(a, -flat::NoHitDistance);
    b = std::nextafter(b, flat::NoHitDistance);
    low = low > a ? low : a;
    high = high < b ? high : b;
  }
  return high >= low && high >= 0.;
}

// Ray-space projected edge functions, evaluated identically at a shared vertex.
// Exact projected zeros identify shared features; no near-edge tolerance snaps
// an interior hit onto an edge. Non-finite/degenerate queries fail closed.
C8_INDEXED_TERRAIN_INLINE Hit triangleHit(View const& view,
                                         std::uint32_t id, Ray const& ray) {
  Hit hit;
  int kz = 0;
  for (int k = 1; k < 3; ++k)
    if (std::abs(component(ray.direction, k)) >
        std::abs(component(ray.direction, kz))) kz = k;
  double dz = component(ray.direction, kz);
  if (dz == 0. || !std::isfinite(dz)) return hit;
  int kx = (kz + 1) % 3, ky = (kx + 1) % 3;
  double sx = component(ray.direction, kx) / dz;
  double sy = component(ray.direction, ky) / dz;
  auto const& tri = view.triangles[id];
  // Sorting by global ID fixes the arithmetic order even for reverse winding
  // or different triangle-local vertex numbering. Orientation uses the normal.
  std::uint32_t ids[3] = {tri.vertex[0], tri.vertex[1], tri.vertex[2]};
  for (int i = 0; i < 2; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (ids[i] > ids[j]) { auto tmp = ids[i]; ids[i] = ids[j]; ids[j] = tmp; }
  double x[3], y[3], z[3];
  for (int i = 0; i < 3; ++i) {
    Vec3 v = flat::sub(view.vertices[ids[i]], ray.origin);
    z[i] = component(v, kz);
    x[i] = std::fma(-sx, z[i], component(v, kx));
    y[i] = std::fma(-sy, z[i], component(v, ky));
  }
  double e[3] = {determinant(x[1], y[1], x[2], y[2]),
                 determinant(x[2], y[2], x[0], y[0]),
                 determinant(x[0], y[0], x[1], y[1])};
  bool cancelled = false;
  for (int i = 0; i < 3; ++i) {
    int a = (i + 1) % 3, b = (i + 2) % 3;
    double scale = std::abs(std::fma(x[a], y[b], 0.0)) +
                   std::abs(std::fma(y[a], x[b], 0.0));
    // Numerical work trigger only, NEVER an acceptance/padding tolerance.
    cancelled = cancelled || std::abs(e[i]) <= 0x1p-47 * scale; // 32 * binary64 epsilon
  }
  if (cancelled) {
    Twofold cx[3], cy[3];
    for (int i = 0; i < 3; ++i) {
      auto v = view.vertices[ids[i]];
      auto zz = sum(component(v, kz), -component(ray.origin, kz));
      cx[i] = subtract(sum(component(v, kx), -component(ray.origin, kx)), multiply({sx,0.},zz));
      cy[i] = subtract(sum(component(v, ky), -component(ray.origin, ky)), multiply({sy,0.},zz));
    }
    for (int i = 0; i < 3; ++i) {
      int a = (i + 1) % 3, b = (i + 2) % 3;
      e[i] = compensatedEdge(cx[a], cy[a], cx[b], cy[b]);
    }
  }
  bool negative = false, positive = false;
  int nonzero[3], count = 0;
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(e[i])) return hit;
    negative = negative || e[i] < 0.;
    positive = positive || e[i] > 0.;
    if (e[i] != 0.) nonzero[count++] = i;
  }
  if ((negative && positive) || count == 0) return hit;
  double det = (e[0] + e[1]) + e[2];
  if (!std::isfinite(det) || det == 0.) return hit;
  if (count == 1) {
    int a = nonzero[0];
    hit.distance = z[a] / dz;
    hit.feature = Feature::Vertex;
    hit.first = ids[a];
  } else if (count == 2) {
    int a = nonzero[0], b = nonzero[1]; // Already in canonical ID order.
    double dx = x[b] - x[a], dy = y[b] - y[a];
    double denominator = std::abs(dx) >= std::abs(dy) ? dx : dy;
    if (denominator == 0.) return Hit{};
    double alpha = -(std::abs(dx) >= std::abs(dy) ? x[a] : y[a]) / denominator;
    hit.distance = std::fma(alpha, z[b] - z[a], z[a]) / dz;
    hit.feature = Feature::Edge;
    hit.first = ids[a]; hit.second = ids[b];
  } else {
    hit.distance = (std::fma(e[0], z[0],
        std::fma(e[1], z[1], std::fma(e[2], z[2], 0.0))) / det) / dz;
    hit.feature = Feature::Interior;
  }
  if (!std::isfinite(hit.distance) || hit.distance < 0.) return Hit{};
  hit.triangle = id;
  return hit;
}

// At a non-smooth shared feature the face normal is not physically unique.
// Return the lowest ORIGINAL face ID among eligible incident crossings. This
// is a deterministic reporting convention, NOT a Fresnel/edge-diffraction model.
C8_INDEXED_TERRAIN_INLINE bool prefer(Hit const& next, Hit const& previous) {
  if (!next.found()) return false;
  if (!previous.found()) return true;
  bool shared = next.feature != Feature::Interior && next.feature == previous.feature &&
                next.first == previous.first && next.second == previous.second;
  if (shared || next.distance == previous.distance) return next.triangle < previous.triangle;
  return next.distance < previous.distance;
}
C8_INDEXED_TERRAIN_INLINE Hit intersect(View const& view, Ray const& ray,
                                       int orientation = 0) {
  Hit result;
  for (std::uint32_t i = 0; i < view.node_count;) {
    auto const& node = view.nodes[i];
    if (!indexed::slab(node, ray)) { i = node.skip; continue; }
    for (std::uint32_t j = 0; j < node.count; ++j) {
      auto id = view.indices[node.first + j];
      if (orientation && orientation * indexed::dot(view.triangles[id].normal, ray.direction) <= 1.e-12)
        continue; // Preserve the existing logical-side / grazing threshold.
      auto hit = triangleHit(view, id, ray);
      if (prefer(hit, result)) result = hit;
    }
    ++i;
  }
  return result;
}
} // namespace corsika::terrain::indexed

#undef C8_INDEXED_TERRAIN_INLINE
