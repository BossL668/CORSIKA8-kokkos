// Device translation unit: no CORSIKA units, CPU geometry templates, or physics.
#include <corsika/geometry/terrain/KokkosTerrainSession.hpp>
#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
#include <corsika/geometry/terrain/TerrainCurvedBoundary.hpp>
#include <array>
#include <iostream>

namespace terrain = corsika::terrain;
namespace flat = terrain::flat;
using Session = terrain::KokkosTerrainSession<Kokkos::DefaultExecutionSpace>;

namespace {
void require(bool value, char const* what) {
  if (!value) throw std::runtime_error(what);
}
template <class F> void rejects(F f) {
  bool rejected = false;
  try { f(); } catch (std::exception const&) { rejected = true; }
  require(rejected, "invalid input was accepted");
}
terrain::FlatTerrainData cube() {
  std::array<flat::Vec3, 8> points{{{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
                                  {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}}};
  std::array<std::array<int,3>,12> faces{{{0,2,1},{0,3,2},{4,5,6},{4,6,7},
      {0,1,5},{0,5,4},{1,2,6},{1,6,5},{2,3,7},{2,7,6},{3,0,4},{3,4,7}}};
  terrain::FlatTerrainData result;
  result.vertices.assign(points.begin(),points.end());
  for (auto const& f : faces) {
    auto const a = points[f[0]];
    auto const e1 = flat::sub(points[f[1]], a), e2 = flat::sub(points[f[2]], a);
    auto n = flat::cross(e1, e2);
    double norm = std::sqrt(flat::dot(n,n));
    n = {n.x/norm,n.y/norm,n.z/norm};
    result.triangles.push_back({a,e1,e2,n});
    result.indexed_triangles.push_back({{static_cast<std::uint32_t>(f[0]),
        static_cast<std::uint32_t>(f[1]),static_cast<std::uint32_t>(f[2])},n});
    result.indices.push_back(result.indices.size());
  }
  result.nodes.push_back({{-1,-1,-1},{1,1,1},1,0,12});
  return result;
}
void exercise() {
  auto data = cube();
  auto bad = data;
  bad.nodes[0].skip = 0;
  rejects([&] { Session broken(bad); });
  bad = data; bad.indices[0] = 12;
  rejects([&] { Session broken(bad); });
  bad = data; bad.indices[0] = bad.indices[1];
  rejects([&] { Session broken(bad); });
  bad = data; bad.nodes[0].count = 11;
  rejects([&] { Session broken(bad); });
  bad = data; bad.triangles[0].normal.x = std::numeric_limits<double>::quiet_NaN();
  rejects([&] { Session broken(bad); });
  bad=data;bad.vertices.clear();rejects([&]{Session broken(bad);});
  bad=data;bad.indexed_triangles[0].vertex[0]=8;rejects([&]{Session broken(bad);});
  bad=data;bad.indexed_triangles[0].vertex[0]=bad.indexed_triangles[0].vertex[1];
  rejects([&]{Session broken(bad);});
  bad=data;bad.vertices[0].z=std::nextafter(bad.vertices[0].z,0.);
  rejects([&]{Session broken(bad);});
  bad=data;bad.indexed_triangles[0].normal.z=-bad.indexed_triangles[0].normal.z;
  rejects([&]{Session broken(bad);});
  rejects([&] { Session broken(data, 0); });

  Session session(data, 31);
  auto const pointer = session.deviceView().nodes;
  auto const bytes = session.geometryBytes() + session.workspaceBytes();
  require(session.query({}).empty() && session.batches() == 0, "empty batch launched");
  // Actual session/upload/kernel path, not just a standalone device function.
  // Inclusive exact edge ownership and the two neighbouring binary64 inputs.
  std::vector<flat::Ray> edges;
  for(double d : {0.,std::nextafter(0.,1.),std::nextafter(0.,-1.)})
    edges.push_back({{d,0.,2.},{0.,0.,-1.},-1});
  auto edgeHits=session.query(edges);
  for(std::size_t i=0;i<edges.size();++i) {
    auto expected=flat::intersect(data.view(),edges[i],-1);
    require(edgeHits[i].distance==1.&&edgeHits[i].triangle==expected.triangle,
            "indexed shared edge host/device mismatch");
    require(edgeHits[i].triangle==(i==2?3u:2u),"one-ULP edge side or face ownership changed");
  }
  require(session.deviceView().vertices!=nullptr&&session.deviceView().indexed_triangles!=nullptr,
          "resident session lost original topology");
  rejects([&] { session.query({{{0,0,0},{2,0,0}}}); });
  rejects([&] { session.query({{{0,0,0},{1,0,0},2}}); });

  // Reusing one owner across differently sized batches/shower-like phases must
  // not re-upload geometry or grow workspace to the total number of queries.
  for (int repeat = 0; repeat < 20; ++repeat) {
    std::vector<flat::Ray> rays;
    for (int i=0; i<4097; ++i) {
      double y = -.9 + 1.8 * (i % 101) / 100.;
      double z = -.87 + 1.74 * (i % 97) / 96.;
      rays.push_back({{-2,y,z},{1,0,0},-1});
    }
    auto hits = session.query(rays);
    for (std::size_t i=0;i<hits.size();++i) {
      auto const& h=hits[i];auto expected=flat::intersect(data.view(),rays[i],-1);
      require(h.found() && std::abs(h.distance-1.) <= 2.*std::numeric_limits<double>::epsilon(),
              "analytic cube entry exceeds two ULP");
      require(h.distance==expected.distance&&h.triangle==expected.triangle,
              "indexed cube scalar/device bits differ");
    }
    require(session.deviceView().nodes == pointer, "resident geometry changed");
    require(session.workspaceCapacity() == 31 &&
            session.geometryBytes() + session.workspaceBytes() == bytes,
            "workspace grew between calls");
  }
  auto hits = session.queryBoundaries({
      {{-2,.1,.2},{1,0,0},false}, // enter, distance 1
      {{-1,.1,.2},{1,0,0},true},  // from entry face to far exit
      {{1,.1,.2},{1,0,0},false},  // air, pointing away: no immediate reentry
      {{1,.1,.2},{-1,0,0},false}, // inward at surface: CPU minimum flight
      {{1-2.e-9,.1,.2},{-1,0,0},false}, // roundoff behind surface
      {{0,.1,.2},{1,0,0},true},
      {{0,2,0},{1,0,0},false}});
  require(hits[0].crossing == flat::Crossing::EnterRock &&
          std::abs(hits[0].hit.distance-1.) < 1.e-14, "entry mismatch");
  require(hits[1].crossing == flat::Crossing::ExitRock &&
          std::abs(hits[1].hit.distance-2.) < 1.e-14, "exit mismatch");
  require(!hits[2].hit.found() && hits[2].crossing == flat::Crossing::None,
          "spurious immediate reentry");
  for (int i : {3,4})
    require(hits[i].crossing == flat::Crossing::EnterRock &&
            hits[i].hit.distance == 1.e-9, "CPU minimum boundary flight differs");
  require(hits[5].crossing == flat::Crossing::ExitRock &&
          std::abs(hits[5].hit.distance-1.) < 1.e-14, "inside exit mismatch");
  require(!hits[6].hit.found(), "outside ray spuriously hits terrain");
  // More than one session may borrow the existing runtime sequentially.
  { Session second(data, 1); require(second.query({{{-2,0,0},{1,0,0}}})[0].found(),
                                     "second session failed"); }
  require(Kokkos::is_initialized() && !Kokkos::is_finalized(),
          "session finalized its caller's runtime");
  std::cout << "execution=" << Kokkos::DefaultExecutionSpace::name()
            << " concurrency=" << Kokkos::DefaultExecutionSpace().concurrency()
            << " queries=81940+7 capacity=" << session.workspaceCapacity()
            << " geometry_bytes=" << session.geometryBytes()
            << " workspace_bytes=" << session.workspaceBytes()
            << " batches=" << session.batches() << '\n';
}
struct BoundaryTransportKernel {
  using Em= corsika::gpu::em::EmParticleState;
  Kokkos::View<corsika::accelerator::em::detail::PhotonTransportOutcome*,
      Kokkos::DefaultExecutionSpace::memory_space> results;
  KOKKOS_INLINE_FUNCTION void operator()(int index)const {
    namespace em=corsika::gpu::em;
    em::EnvironmentSnapshot env;
    env.geometry=em::EnvironmentGeometry::HomogeneousConvexPolyhedron;
    env.number_of_layers=1;env.number_of_convex_planes=6;env.convex_boundary_tolerance_m=1.e-8;
    env.atmosphere_layers[0].density_model=em::DensityModel::Homogeneous;
    env.atmosphere_layers[0].density_parameter_a=2.;
    env.convex_planes[0]={1,0,0,2};env.convex_planes[1]={-1,0,0,2};
    env.convex_planes[2]={0,1,0,2};env.convex_planes[3]={0,-1,0,2};
    env.convex_planes[4]={0,0,1,2};env.convex_planes[5]={0,0,-1,2};
    em::EmInteractionRecord interaction;interaction.particle.pid=22;
    interaction.particle.energy_GeV=1.;interaction.particle.direction[0]=1.;
    interaction.particle.history_id=19;interaction.particle.step_id=3;interaction.particle.weight=7.;
    interaction.status=em::EmInteractionStatus::NoDiscreteInteraction;
    em::ExternalTransportBoundary boundary;boundary.disable_observation=true;
    boundary.enabled=index!=0;boundary.distance_m=.25;
    if(index==2){interaction.status=em::EmInteractionStatus::Selected;interaction.interaction_grammage_g_per_cm2=10.;}
    if(index==3){interaction.status=em::EmInteractionStatus::ParticleCut;}
    if(index==4){interaction.particle.time_s=.010001;}
    results(index)=corsika::accelerator::em::detail::transportPhoton(env,interaction,boundary);
  }
};
void transportExercise() {
  namespace em=corsika::gpu::em;
  using View=decltype(BoundaryTransportKernel::results);
  View results("material_boundary_photon_results",5);
  Kokkos::parallel_for("material_boundary_photon_test",Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,5),
      BoundaryTransportKernel{results});
  auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},results);
  for(int i=0;i<5;++i)require(!host(i).fallback_flag,"unexpected boundary transport fallback");
  auto const& original=host(0).record;
  require(original.limit==em::PhotonTransportLimit::EscapedEnvironment&&original.distance_m==2.,"disabled boundary changed old path");
  auto const& crossing=host(1).record;
  require(crossing.limit==em::PhotonTransportLimit::MaterialBoundary&&crossing.distance_m==.25,"material did not limit step");
  require(crossing.end.history_id==19&&crossing.end.step_id==4&&crossing.end.weight==7.&&
      crossing.end.energy_GeV==1.&&crossing.cut_deposited_energy_GeV==0.,"boundary changed particle/energy ownership");
  require(std::abs(crossing.traversed_grammage_g_per_cm2-50.)<1.e-12,"material grammage wrong");
  require(std::abs(crossing.end.time_s-.25/299792458.)<1.e-20,"boundary flight time wrong");
  require(host(2).record.limit==em::PhotonTransportLimit::Interaction&&
      std::abs(host(2).record.distance_m-.05)<1.e-14,"interaction/boundary competition wrong");
  require(host(3).record.limit==em::PhotonTransportLimit::ParticleCut&&host(3).record.distance_m==0.,"cut did not win");
  require(host(4).record.limit==em::PhotonTransportLimit::ParticleCut&&
      !host(4).record.observation_surface_reached_before_cut,"material surface became an observation");
  std::cout<<"material transport: nonabsorbing crossing, timing, grammage, cut, competition PASS\n";
}
struct CurvedBoundaryKernel {
  flat::View mesh;
  Kokkos::View<double*,Kokkos::DefaultExecutionSpace::memory_space> distances;
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t i)const {
    double a=.001+double(i%997)*.00001;
    double b=(double(i%991)/991.-.5)*.02;
    double c=(double(i%983)/983.-.5)*.02;
    flat::QuadraticPath path{{{-2.,0.,0.},{1.,0.,0.},bool(i%2)}, {a,b,c},4.};
    distances(i)=flat::nextCurvedBoundary(mesh,path).hit.distance;
  }
};
void curvedExercise() {
  Session session(cube(),1);
  decltype(CurvedBoundaryKernel::distances) output("curved_analytic_distances",1000000);
  Kokkos::parallel_for("curved_terrain_million",Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,output.size()),
      CurvedBoundaryKernel{session.deviceView(),output});
  auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  double maximum=0.;
  for(std::size_t i=0;i<output.size();++i) {
    double a=.001+double(i%997)*.00001,offset=i%2?3.:1.;
    double expected=2.*offset/(1.+std::sqrt(1.+4.*a*offset));
    double error=std::abs(host(i)-expected);
    require(std::isfinite(host(i))&&error<2.e-14,"curved DEM analytic root mismatch");
    maximum=std::max(maximum,error);
  }
  std::cout<<"curved DEM independent analytic oracle: queries=1000000 max_error_m="<<maximum<<" PASS\n";
}
} // namespace

int main() {
  try {
    auto data = cube();
    rejects([&] { Session missing_runtime(data); });
    Kokkos::ScopeGuard runtime;
    exercise();
    transportExercise();
    curvedExercise();
    require(Kokkos::is_initialized() && !Kokkos::is_finalized(),
            "terrain session owns runtime");
    std::cout << "terrain resident session: PASS\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
