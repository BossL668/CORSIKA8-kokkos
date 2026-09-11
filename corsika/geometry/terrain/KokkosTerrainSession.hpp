/* Read-only terrain geometry belonging to an already initialized Kokkos runtime.
 * No CLI, atmosphere, physics table, RNG, or second runtime is hidden here.
 * Destroy this owner before finalizing Kokkos. Not a concurrent host container. */
#pragma once

#include <Kokkos_Core.hpp>
#include <corsika/geometry/terrain/FlatTerrainData.hpp>
#include <corsika/geometry/terrain/TerrainBoundary.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace corsika::terrain {

// Fail before device upload; the query kernel must never traverse a corrupt
// forward link or index. Surface topology is still checked by ValidatedTerrain.
inline void validateFlatTerrainForUpload(FlatTerrainData const& data) {
  if (data.nodes.empty() || data.triangles.empty() || data.vertices.empty() ||
      data.vertices.size() > std::numeric_limits<std::uint32_t>::max() ||
      data.indexed_triangles.size() != data.triangles.size() ||
      data.nodes.size() > std::numeric_limits<std::uint32_t>::max() ||
      data.indices.size() != data.triangles.size() ||
      data.nodes.front().skip != data.nodes.size())
    throw std::invalid_argument("invalid terrain BVH sizes/root");
  auto finite = [](flat::Vec3 p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
  };
  for (auto const& v : data.vertices)
    if (!finite(v)) throw std::invalid_argument("invalid terrain original vertex");
  for (std::size_t i=0;i<data.indexed_triangles.size();++i) {
    auto const& t=data.indexed_triangles[i];
    for (auto id : t.vertex)
      if (id>=data.vertices.size()) throw std::invalid_argument("invalid terrain vertex index");
    if(t.vertex[0]==t.vertex[1]||t.vertex[0]==t.vertex[2]||t.vertex[1]==t.vertex[2])
      throw std::invalid_argument("duplicate terrain vertex index");
    auto const& old=data.triangles[i];
    auto same=[](flat::Vec3 a,flat::Vec3 b){return a.x==b.x&&a.y==b.y&&a.z==b.z;};
    auto origin=data.vertices[t.vertex[0]];
    if(!same(old.origin,origin)||!same(old.edge1,flat::sub(data.vertices[t.vertex[1]],origin))||
       !same(old.edge2,flat::sub(data.vertices[t.vertex[2]],origin))||!same(t.normal,old.normal))
      throw std::invalid_argument("terrain indexed/curved face representations disagree");
  }
  std::vector<bool> seen(data.triangles.size(), false);
  std::vector<bool> covered(data.indices.size(), false);
  std::vector<std::uint32_t> parents;
  for (std::size_t i = 0; i < data.nodes.size(); ++i) {
    auto const& n = data.nodes[i];
    while (!parents.empty() && i == parents.back()) parents.pop_back();
    if (n.skip <= i || n.skip > data.nodes.size() ||
        (!parents.empty() && n.skip > parents.back()) ||
        n.first > data.indices.size() || n.count > data.indices.size() - n.first ||
        !finite(n.low) || !finite(n.high) || n.low.x > n.high.x ||
        n.low.y > n.high.y || n.low.z > n.high.z)
      throw std::invalid_argument("invalid terrain BVH node");
    if (n.skip > i + 1) parents.push_back(n.skip);
    for (std::size_t j = n.first; j < std::size_t(n.first) + n.count; ++j) {
      auto const id = data.indices[j];
      if (covered[j] || id >= seen.size() || seen[id])
        throw std::invalid_argument("invalid/duplicate terrain face index");
      covered[j] = true;
      seen[id] = true;
    }
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    throw std::invalid_argument("terrain BVH misses a face");
  for (auto const& t : data.triangles) {
    if (!finite(t.origin) || !finite(t.edge1) || !finite(t.edge2) ||
        !finite(t.normal) || std::abs(flat::dot(t.normal, t.normal) - 1.) > 1.e-10 ||
        !(flat::dot(flat::cross(t.edge1, t.edge2), t.normal) > 0.))
      throw std::invalid_argument("invalid terrain face data");
  }
}

namespace kokkos_detail {
struct Query {
  flat::Ray ray;
  bool logical_boundary{};
};
template <class ExecutionSpace> struct QueryKernel {
  using Memory = typename ExecutionSpace::memory_space;
  flat::View geometry;
  Kokkos::View<Query*, Memory> queries;
  Kokkos::View<flat::BoundaryCandidate*, Memory> results;
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
    auto const& q = queries(i);
    if (q.logical_boundary) {
      flat::BoundaryQuery boundary;
      boundary.origin = q.ray.origin;
      boundary.direction = q.ray.direction;
      boundary.logically_inside = q.ray.orientation == 1;
      results(i) = flat::nextBoundary(geometry, boundary);
    } else {
      flat::BoundaryCandidate candidate;
      candidate.hit = flat::intersect(geometry, q.ray, q.ray.orientation);
      candidate.crossing = flat::Crossing::None;
      results(i) = candidate;
    }
  }
};
} // namespace kokkos_detail

template <class ExecutionSpace> class KokkosTerrainSession {
 public:
  using Memory = typename ExecutionSpace::memory_space;
  using QueryView = Kokkos::View<kokkos_detail::Query*, Memory>;
  using ResultView = Kokkos::View<flat::BoundaryCandidate*, Memory>;

  explicit KokkosTerrainSession(FlatTerrainData const& data,
                               std::size_t capacity = 4096) {
    if (!Kokkos::is_initialized() || Kokkos::is_finalized())
      throw std::logic_error("terrain session requires an active external Kokkos runtime");
    if (capacity == 0 || capacity > 1000000)
      throw std::invalid_argument("terrain query capacity must be in [1, 1000000]");
    validateFlatTerrainForUpload(data);
    // Constructing an execution-space instance also requires an active runtime.
    execution_ = std::make_unique<ExecutionSpace>();
    nodes_ = upload(data.nodes, "terrain_nodes");
    triangles_ = upload(data.triangles, "terrain_triangles");
    indices_ = upload(data.indices, "terrain_indices");
    vertices_ = upload(data.vertices, "terrain_original_vertices");
    indexed_triangles_ = upload(data.indexed_triangles, "terrain_indexed_triangles");
    queries_ = QueryView("terrain_query_workspace", capacity);
    results_ = ResultView("terrain_result_workspace", capacity);
    host_queries_ = Kokkos::create_mirror_view(queries_);
    host_results_ = Kokkos::create_mirror_view(results_);
    geometry_bytes_ = data.bytes();
    execution_->fence("terrain immutable upload");
  }

  KokkosTerrainSession(KokkosTerrainSession const&) = delete;
  KokkosTerrainSession& operator=(KokkosTerrainSession const&) = delete;

  // Pointers are in ExecutionSpace::memory_space, NOT necessarily host-readable.
  // Kernels borrowing this view must finish before this session is destroyed.
  flat::View deviceView() const {
    return {nodes_.data(), triangles_.data(), indices_.data(),
            static_cast<std::uint32_t>(nodes_.extent(0)), indexed_triangles_.data(), vertices_.data()};
  }
  std::size_t geometryBytes() const { return geometry_bytes_; }
  std::size_t workspaceCapacity() const { return queries_.extent(0); }
  std::size_t workspaceBytes() const {
    return workspaceCapacity() *
           (sizeof(kokkos_detail::Query) + sizeof(flat::BoundaryCandidate));
  }
  std::size_t batches() const { return batches_; }

  std::vector<flat::Hit> query(std::vector<flat::Ray> const& input) {
    return run<flat::Hit>(input.size(), [&](std::size_t i) {
      validateRay(input[i]);
      return kokkos_detail::Query{input[i], false};
    }, [](flat::BoundaryCandidate const& result) { return result.hit; });
  }

  std::vector<flat::BoundaryCandidate> queryBoundaries(
      std::vector<flat::BoundaryQuery> const& input) {
    return run<flat::BoundaryCandidate>(input.size(), [&](std::size_t i) {
      flat::Ray ray{input[i].origin, input[i].direction,
                    input[i].logically_inside ? 1 : -1};
      validateRay(ray);
      return kokkos_detail::Query{ray, true};
    }, [](flat::BoundaryCandidate const& result) { return result; });
  }

 private:
  static void validateRay(flat::Ray const& ray) {
    auto finite = [](flat::Vec3 p) {
      return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    };
    if (!finite(ray.origin) || !finite(ray.direction) || ray.orientation < -1 ||
        ray.orientation > 1 ||
        std::abs(flat::dot(ray.direction, ray.direction) - 1.) > 1.e-10)
      throw std::invalid_argument("terrain query requires finite position and unit direction");
  }

  template <class T> auto upload(std::vector<T> const& source, char const* name) {
    Kokkos::View<T*, Memory> view(std::string(name), source.size());
    auto host = Kokkos::create_mirror_view(view);
    for (std::size_t i = 0; i < source.size(); ++i) host(i) = source[i];
    // Blocking copy: the temporary host mirror must outlive the transfer.
    Kokkos::deep_copy(view, host);
    return view;
  }

  template <class Result, class GetQuery, class GetResult>
  std::vector<Result> run(std::size_t size, GetQuery get, GetResult convert) {
    std::vector<Result> output(size);
    using Policy = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>;
    for (std::size_t offset = 0; offset < size;) {
      std::size_t const count = std::min(size - offset, workspaceCapacity());
      for (std::size_t i = 0; i < count; ++i) host_queries_(i) = get(offset + i);
      auto const range = std::make_pair(std::size_t(0), count);
      Kokkos::deep_copy(*execution_, Kokkos::subview(queries_, range),
                       Kokkos::subview(host_queries_, range));
      Kokkos::parallel_for("terrain_resident_query", Policy(*execution_, 0, count),
          kokkos_detail::QueryKernel<ExecutionSpace>{deviceView(), queries_, results_});
      Kokkos::deep_copy(*execution_, Kokkos::subview(host_results_, range),
                       Kokkos::subview(results_, range));
      execution_->fence("terrain query results ready");
      for (std::size_t i = 0; i < count; ++i) output[offset + i] = convert(host_results_(i));
      ++batches_;
      offset += count;
    }
    return output;
  }

  // Views are destroyed before the execution-space instance; caller owns runtime.
  std::unique_ptr<ExecutionSpace> execution_;
  Kokkos::View<flat::Node*, Memory> nodes_;
  Kokkos::View<flat::Triangle*, Memory> triangles_;
  Kokkos::View<std::uint32_t*, Memory> indices_;
  Kokkos::View<flat::Vec3*, Memory> vertices_;
  Kokkos::View<flat::IndexedTriangle*, Memory> indexed_triangles_;
  QueryView queries_;
  ResultView results_;
  typename QueryView::HostMirror host_queries_;
  typename ResultView::HostMirror host_results_;
  std::size_t geometry_bytes_{};
  std::size_t batches_{};
};

} // namespace corsika::terrain
