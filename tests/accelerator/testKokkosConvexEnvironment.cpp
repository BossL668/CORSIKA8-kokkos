/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */

#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/common/ConvexEnvironmentSnapshot.hpp>
#include <corsika/accelerator/em/kokkos/KokkosPhotonTransport.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
  using namespace corsika::gpu::em;
  struct PlaneFixture {
    std::vector<corsika::geometry_detail::ConvexPlane> planes;
    auto const& exportPlanes() const { return planes; }
  };
  void require(bool ok, char const* message) {
    if (!ok) throw std::runtime_error(message);
  }
  void close(double actual, double expected, char const* message) {
    require(std::isfinite(actual) &&
                std::abs(actual - expected) <=
                    2.e-12 * std::max(1., std::abs(expected)), message);
  }
  EmInteractionRecord photon(double x, double y, double z,
                             double dx, double dy, double dz) {
    EmInteractionRecord interaction{};
    interaction.status = EmInteractionStatus::NoDiscreteInteraction;
    auto& p = interaction.particle;
    p.pid = 22;
    p.energy_GeV = 1.;
    p.weight = 1.;
    p.position_m[0] = x;
    p.position_m[1] = y;
    p.position_m[2] = z;
    p.direction[0] = dx;
    p.direction[1] = dy;
    p.direction[2] = dz;
    return interaction;
  }
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard scope(argc, argv);
  using Execution = Kokkos::DefaultExecutionSpace;
  using namespace corsika::gpu::em;
  PlaneFixture const box{{{1., 0., 0., 2.}, {-1., 0., 0., 2.},
                         {0., 1., 0., 3.}, {0., -1., 0., 3.},
                         {0., 0., 1., 4.}, {0., 0., -1., 4.}}};
  auto const environment = makeHomogeneousConvexSnapshot(2.65, box, 7);
  require(atmosphere_detail::validEnvironment(environment), "valid convex snapshot");
  require(environment.atmosphere_layers[0].medium_id == 7, "material identity");
  auto const badB = [&]() {
    try { (void)makeHomogeneousConvexSnapshot(2.65, box, 7, {0., 0., 1.e-9}); }
    catch (std::invalid_argument const&) { return true; }
    return false;
  }();
  require(badB, "nonzero magnetic field must fail before transport");
  auto invalid = environment;
  invalid.number_of_convex_planes = 3;
  require(!atmosphere_detail::validEnvironment(invalid), "open solid rejected");
  invalid = environment;
  invalid.convex_planes[0].offset_m = std::numeric_limits<double>::quiet_NaN();
  require(!atmosphere_detail::validEnvironment(invalid), "NaN plane rejected");
  double const origin[3]{0., 0., 0.};
  double const alongX[3]{1., 0., 0.};
  auto const atOrigin = queryAtmosphereLayer(environment, origin, alongX);
  require(atOrigin.status == AtmosphereStatus::Success, "local origin is valid rock");
  auto const grammage = atmosphereGrammage(environment, 0, origin, alongX, 1.25);
  close(grammage.value, 331.25, "dense grammage at local origin");
  close(atmosphereDistanceFromGrammage(environment, 0, origin, alongX,
                                      grammage.value).value,
        1.25, "inverse dense grammage");

  std::vector<EmInteractionRecord> inputs;
  std::vector<double> expectedDistance;
  constexpr std::size_t samples = 10000;
  for (std::size_t i = 0; i < samples; ++i) {
    // Independent analytical axis-aligned box oracle, not a second call to
    // the shared half-space clipping implementation.
    double const x = 1.9 * std::sin(0.13 * static_cast<double>(i));
    double const y = 2.9 * std::cos(0.17 * static_cast<double>(i));
    double const z = 3.9 * std::sin(0.19 * static_cast<double>(i));
    double dx = std::sin(0.23 * static_cast<double>(i)) + .1;
    double dy = std::cos(0.31 * static_cast<double>(i)) + .2;
    double dz = std::sin(0.41 * static_cast<double>(i)) + .3;
    double const norm = std::sqrt(dx*dx + dy*dy + dz*dz);
    dx /= norm; dy /= norm; dz /= norm;
    auto const axisExit = [](double pos, double dir, double extent) {
      return dir > 0. ? (extent-pos)/dir :
             dir < 0. ? (-extent-pos)/dir : std::numeric_limits<double>::infinity();
    };
    auto const boundary = std::min({axisExit(x, dx, 2.), axisExit(y, dy, 3.),
                                    axisExit(z, dz, 4.)});
    auto interaction = photon(x, y, z, dx, dy, dz);
    interaction.input_index = static_cast<std::uint32_t>(i);
    interaction.particle.history_id = i+1;
    if (i % 2 == 0) {
      interaction.status = EmInteractionStatus::Selected;
      interaction.interaction_grammage_g_per_cm2 = boundary * .25 * 265.;
      interaction.process_id = 1;
    }
    inputs.push_back(interaction);
    expectedDistance.push_back(i % 2 == 0 ? boundary*.25 : boundary);
  }
  auto const batch = corsika::accelerator::em::kokkos_detail::transportPhotons<Execution>(
      environment, inputs);
  require(batch.fallback_events.empty(), "bounded photon kernel must not fallback");
  require(batch.records.size() == samples, "all photon records returned");
  for (std::size_t i = 0; i < samples; ++i) {
    auto const& r = batch.records[i];
    auto const& p = inputs[i].particle;
    close(r.distance_m, expectedDistance[i], "device/analytic boundary disagreement");
    require(r.limit == (i % 2 == 0 ? PhotonTransportLimit::Interaction :
                       PhotonTransportLimit::EscapedEnvironment), "wrong terminal limit");
    close(r.traversed_grammage_g_per_cm2, expectedDistance[i]*265., "device grammage");
    close(r.end.time_s, expectedDistance[i]/SpeedOfLightMPerS, "device flight time");
    for (int axis = 0; axis < 3; ++axis)
      close(r.end.position_m[axis], p.position_m[axis] +
                                      expectedDistance[i]*p.direction[axis],
            "device endpoint");
    close(r.end.energy_GeV, 1., "transport must not alter photon energy");
  }

  // Exact-face inward/outward, edge, corner, tangent and an interior point
  // closer than the old spherical 0.1 mm guard. Never jump across the solid.
  std::vector<EmInteractionRecord> boundaryCases{
      photon(2.,0.,0.,1.,0.,0.), photon(2.,0.,0.,-1.,0.,0.),
      photon(2.,3.,0.,0.,0.,1.), photon(2.,3.,4.,0.,0.,-1.),
      photon(2.-1.e-7,0.,0.,1.,0.,0.), photon(0.,0.,0.,0.,1.,0.)};
  double const boundaryExpected[]{0.,4.,4.,8.,1.e-7,3.};
  auto const edges = corsika::accelerator::em::kokkos_detail::transportPhotons<Execution>(
      environment, boundaryCases);
  require(edges.fallback_events.empty() && edges.records.size()==6, "edge ray failure");
  for (std::size_t i=0; i<6; ++i) {
    close(edges.records[i].distance_m,boundaryExpected[i],"edge ray distance");
    require(edges.records[i].limit==PhotonTransportLimit::EscapedEnvironment,
            "edge ray escaped flag");
  }

  double const invSqrt3 = 1./std::sqrt(3.);
  PlaneFixture const tetra{{{-1.,0.,0.,0.},{0.,-1.,0.,0.},{0.,0.,-1.,0.},
                            {invSqrt3,invSqrt3,invSqrt3,invSqrt3}}};
  auto const tetraEnvironment=makeHomogeneousConvexSnapshot(2.65,tetra);
  auto const tetraBatch=corsika::accelerator::em::kokkos_detail::transportPhotons<Execution>(
      tetraEnvironment,{photon(.1,.1,.1,1.,0.,0.)});
  require(tetraBatch.fallback_events.empty(),"tetrahedron transport");
  close(tetraBatch.records[0].distance_m,.7,"tetrahedron sloped face intersection");

  // Existing spherical branch remains the default and still honors the
  // independent flat observation plane rather than the lower sphere.
  EnvironmentSnapshot spherical{};
  spherical.number_of_layers=1;
  spherical.earth_center_m[2]=-1.e6;
  spherical.atmosphere_layers[0]={1.e6-100.,1.e6+1000.,.001,0.,0.,0,DensityModel::Homogeneous};
  spherical.observation_radius_m=1.e6;
  spherical.observation_plane_normal[2]=1.;
  require(spherical.geometry==EnvironmentGeometry::SphericalLayers,"default geometry");
  auto const atmosphere=corsika::accelerator::em::kokkos_detail::transportPhotons<Execution>(
      spherical,{photon(0.,0.,500.,0.,0.,-1.)});
  require(atmosphere.fallback_events.empty(),"historical sphere transport");
  close(atmosphere.records[0].distance_m,500.,"historical observation distance");
  require(atmosphere.records[0].limit==PhotonTransportLimit::ObservationSurface,
          "historical observation limit");
  std::cout << "PASS: " << samples << " analytic box rays, exact boundary cases, "
            << "tetrahedron, magnetic gate and unchanged spherical observation; backend="
            << Execution::name() << '\n';
}
