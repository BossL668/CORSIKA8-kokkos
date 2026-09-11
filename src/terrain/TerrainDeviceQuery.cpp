// Compile only POD geometry and Kokkos with the device compiler. The CPU
// CORSIKA unit/coordinate templates deliberately stay in host translation units.
#include <Kokkos_Core.hpp>
#include <corsika/geometry/terrain/FlatTerrainData.hpp>
#include <corsika/geometry/terrain/KokkosTerrainSession.hpp>
#include <corsika/geometry/terrain/TerrainCurvedBoundary.hpp>
#include <chrono>
namespace corsika::terrain {
using Space=Kokkos::DefaultExecutionSpace;
namespace {
struct CurveQueryKernel {
  flat::View mesh;
  Kokkos::View<flat::QuadraticPath*,Space::memory_space> paths;
  Kokkos::View<flat::Hit*,Space::memory_space> hits;
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t i)const {
    hits(i)=flat::nextCurvedBoundary(mesh,paths(i)).hit;
  }
};
}
TerrainQueryResult queryTerrainCurvesDevice(FlatTerrainData const& data,
    std::vector<flat::QuadraticPath> const& paths,int threads,int device,
    std::size_t capacity,int repeats) {
  if(repeats<1)throw std::invalid_argument("terrain repeats must be positive");
  Kokkos::InitializationSettings settings;
  settings.set_num_threads(threads);settings.set_device_id(device);
  Kokkos::ScopeGuard runtime(settings);
  KokkosTerrainSession<Space> session(data,capacity);
  Kokkos::View<flat::QuadraticPath*,Space::memory_space> input("curve_workspace",capacity);
  Kokkos::View<flat::Hit*,Space::memory_space> output("curve_hits",capacity);
  auto hostInput=Kokkos::create_mirror_view(input);
  auto hostOutput=Kokkos::create_mirror_view(output);
  TerrainQueryResult result;result.hits.resize(paths.size());
  result.execution_space=Space::name();result.concurrency=Space().concurrency();
  auto started=std::chrono::steady_clock::now();
  for(int repeat=0;repeat<repeats;++repeat)for(std::size_t offset=0;offset<paths.size();offset+=capacity) {
    auto n=std::min(capacity,paths.size()-offset);
    for(std::size_t i=0;i<n;++i)hostInput(i)=paths[offset+i];
    Kokkos::deep_copy(input,hostInput);
    Kokkos::parallel_for("terrain_curved_query",Kokkos::RangePolicy<Space>(0,n),
                        CurveQueryKernel{session.deviceView(),input,output});
    Kokkos::deep_copy(hostOutput,output);
    for(std::size_t i=0;i<n;++i) {
      auto& expected=result.hits[offset+i];auto actual=hostOutput(i);
      if(repeat==0)expected=actual;
      else if(actual.triangle!=expected.triangle||actual.distance!=expected.distance)
        ++result.repeat_mismatches;
    }
    ++result.batches;
  }
  result.query_fenced_wall_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  result.geometry_uploads=1;result.workspace_capacity=capacity;
  result.workspace_bytes=session.workspaceBytes()+capacity*(sizeof(flat::QuadraticPath)+sizeof(flat::Hit));
  return result;
}
TerrainQueryResult queryTerrainDevice(FlatTerrainData const& data,std::vector<flat::Ray> const& rays,
    int threads,int device,bool logical_boundaries,std::size_t capacity,int repeats) {
  if(repeats<1)throw std::invalid_argument("terrain repeats must be positive");
  Kokkos::InitializationSettings settings;
  settings.set_num_threads(threads);settings.set_device_id(device);
  Kokkos::ScopeGuard runtime(settings);
  KokkosTerrainSession<Space> session(data,capacity);
  std::vector<flat::BoundaryQuery> boundaries;
  if(logical_boundaries)for(auto const& ray:rays) {
    if(ray.orientation!=1&&ray.orientation!=-1)
      throw std::invalid_argument("logical query requires entry/exit orientation");
    boundaries.push_back({ray.origin,ray.direction,ray.orientation==1});
  }
  TerrainQueryResult result;result.execution_space=Space::name();result.concurrency=Space().concurrency();
  auto started=std::chrono::steady_clock::now();
  for(int repeat=0;repeat<repeats;++repeat) {
    std::vector<flat::Hit> actual;
    if(logical_boundaries) {
      auto values=session.queryBoundaries(boundaries);actual.reserve(values.size());
      for(std::size_t i=0;i<values.size();++i) {
        auto const expected=values[i].hit.found()?
            (boundaries[i].logically_inside?flat::Crossing::ExitRock:flat::Crossing::EnterRock):flat::Crossing::None;
        if(values[i].crossing!=expected)throw std::logic_error("terrain crossing classification mismatch");
        actual.push_back(values[i].hit);
      }
    } else actual=session.query(rays);
    if(repeat==0)result.hits=std::move(actual);
    else for(std::size_t i=0;i<actual.size();++i)
      if(actual[i].triangle!=result.hits[i].triangle||actual[i].distance!=result.hits[i].distance)
        ++result.repeat_mismatches;
  }
  // End-to-end queries include transfers/staging; do not label this kernel time.
  result.query_fenced_wall_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  result.geometry_uploads=1;result.workspace_capacity=session.workspaceCapacity();
  result.workspace_bytes=session.workspaceBytes();result.batches=session.batches();
  return result;
}
}
