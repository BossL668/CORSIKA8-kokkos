/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <corsika/framework/geometry/Line.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/EnvironmentSnapshotBuilder.hpp>
#include <corsika/gpu/em/SphericalAtmosphere.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/IMediumModel.hpp>

namespace {

  using namespace corsika;
  using namespace corsika::gpu::em;

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void requireClose(double actual, double expected,
                    double relative_tolerance,
                    std::string const& message) {
    ++checks;
    auto const scale =
        std::max({1.e-300, std::abs(actual), std::abs(expected)});
    if (std::abs(actual - expected) >
        relative_tolerance * scale) {
      std::ostringstream detail;
      detail << message << ": actual=" << std::setprecision(17)
             << actual << ", expected=" << expected
             << ", relative="
             << std::abs(actual - expected) / scale;
      throw std::runtime_error(detail.str());
    }
  }

  std::array<double, 3> normalized(
      std::array<double, 3> direction) {
    auto const norm = std::sqrt(
        direction[0] * direction[0] +
        direction[1] * direction[1] +
        direction[2] * direction[2]);
    for (auto& value : direction) {
      value /= norm;
    }
    return direction;
  }

  EmParticleState makePhoton(
      double earth_radius_m, double altitude_m,
      std::array<double, 3> const& direction,
      std::uint64_t history_id) {
    EmParticleState particle{};
    particle.pid = static_cast<std::int32_t>(EmPid::Photon);
    particle.medium_id = 17;
    particle.energy_GeV = 1.e7;
    particle.position_m[2] = earth_radius_m + altitude_m;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      particle.direction[axis] = direction[axis];
    }
    particle.weight = 1.;
    particle.history_id = history_id;
    particle.step_id = 3;
    return particle;
  }

  EmInteractionRecord noInteraction(
      EmParticleState const& particle, std::size_t input_index) {
    EmInteractionRecord interaction{};
    interaction.particle = particle;
    interaction.input_index = input_index;
    interaction.status =
        EmInteractionStatus::NoDiscreteInteraction;
    interaction.interaction_grammage_g_per_cm2 =
        std::numeric_limits<double>::infinity();
    return interaction;
  }

} // namespace

int main() {
  try {
    int device_count = 0;
    auto const device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
      std::cerr << "SKIP: no usable CUDA device\n";
      return 77;
    }

    auto const earth_radius_m =
        constants::EarthRadius::Mean / 1_m;
    auto const observation_radius_m = earth_radius_m + 100.;
    auto const snapshot = makeCorsika7AtmosphereSnapshot(
        AtmosphereId::USStdBK, {0., 0., 0.}, 17,
        observation_radius_m);
    require(atmosphere_detail::validEnvironment(snapshot),
            "CORSIKA-7 GPU atmosphere snapshot must be valid");
    require(snapshot.number_of_layers == 5,
            "CORSIKA-7 snapshot must have five layers");
    require(snapshot.atmosphere_layers[4].density_model ==
                DensityModel::Homogeneous,
            "fifth CORSIKA-7 layer must match CPU HomogeneousMedium");
    requireClose(
        snapshot.maximum_magnetic_deflection_rad, 0.2, 1.e-15,
        "default GPU magnetic deflection differs from scalar tracking");
    auto short_track_snapshot = makeCorsika7AtmosphereSnapshot(
        AtmosphereId::USStdBK, {0., 0., 0.}, 17,
        observation_radius_m, {0., 0., 0.}, 0.001);
    require(
        atmosphere_detail::validEnvironment(short_track_snapshot),
        "short-track GPU atmosphere snapshot must be valid");
    requireClose(
        short_track_snapshot.maximum_magnetic_deflection_rad,
        0.001, 1.e-15,
        "GPU atmosphere snapshot lost the configured deflection");
    short_track_snapshot.maximum_magnetic_deflection_rad = 0.;
    require(
        !atmosphere_detail::validEnvironment(short_track_snapshot),
        "GPU atmosphere accepted a zero magnetic deflection limit");

    auto const& coordinate_system =
        get_root_CoordinateSystem();
    Point const center(
        coordinate_system, {0_m, 0_m, 0_m});
    Environment<IMediumModel> cpu_environment;
    create_5layer_atmosphere<IMediumModel>(
        cpu_environment, AtmosphereId::USStdBK, center);
    auto const* universe =
        cpu_environment.getUniverse().get();
    auto const density_unit = 1_g / cube(1_cm);
    auto const grammage_unit = 1_g / square(1_cm);

    std::array<double, 9> const altitudes_m{
        100.1, 3999., 4001., 9999., 10001.,
        39999., 40001., 99999., 110000.};
    for (auto const altitude_m : altitudes_m) {
      Point const point(
          coordinate_system,
          {0_m, 0_m,
           (earth_radius_m + altitude_m) * 1_m});
      double const position_m[3]{
          0., 0., earth_radius_m + altitude_m};
      auto const gpu_layer =
          queryAtmosphereLayer(snapshot, position_m);
      require(gpu_layer.status == AtmosphereStatus::Success,
              "GPU layer lookup must succeed");
      auto const* node = universe->getContainingNode(point);
      require(node != universe,
              "CPU environment must contain density point");
      auto const cpu_density =
          node->getModelProperties().getMassDensity(point) /
          density_unit;
      requireClose(gpu_layer.density_g_per_cm3, cpu_density,
                   2.e-13,
                   "GPU radial density must equal CORSIKA CPU density");
    }

    // Regression for a production failure found at 50 MeV cut. A charged
    // particle was 45 micrometres outside a layer boundary and moving inward.
    // The leapfrog sphere solver correctly ignored that sub-0.1-mm near root,
    // but the old layer lookup selected the outer layer and then found a
    // remote second crossing. Layer ownership and intersection guards must be
    // identical.
    {
      auto const boundary_radius_m =
          snapshot.atmosphere_layers[1].inner_radius_m;
      double const near_outer_position[3]{
          0., 0., boundary_radius_m + 0.5 * AtmosphereBoundaryGuardM};
      double const inward[3]{0., 0., -1.};
      double const outward[3]{0., 0., 1.};
      auto const inward_layer = queryAtmosphereLayer(
          snapshot, near_outer_position, inward);
      auto const outward_layer = queryAtmosphereLayer(
          snapshot, near_outer_position, outward);
      require(
          inward_layer.status == AtmosphereStatus::Success &&
              inward_layer.layer_index == 0,
          "sub-guard inward state must belong to the inner layer");
      require(
          outward_layer.status == AtmosphereStatus::Success &&
              outward_layer.layer_index == 1,
          "sub-guard outward state must belong to the outer layer");
      auto const inward_boundary = distanceToAtmosphereBoundary(
          snapshot, near_outer_position, inward);
      require(
          inward_boundary.status == AtmosphereStatus::Success &&
              inward_boundary.distance_m >
                  AtmosphereBoundaryGuardM,
          "straight boundary lookup must ignore the same near root");
      requireClose(
          inward_boundary.radius_m,
          snapshot.observation_radius_m, 1.e-15,
          "inward near-boundary state must advance to the next physical boundary");
    }

    // A terminal observation surface is different from a layer transition:
    // the sub-0.1-mm near root must be retained.  Otherwise the quadratic
    // solver selects the second crossing on the far side of Earth and the
    // one-layer grammage integral overflows.  These coordinates reproduce a
    // production 1e17 eV ensemble failure down to the failing direction.
    {
      auto const production_snapshot =
          makeCorsika7AtmosphereSnapshot(
              AtmosphereId::USStdBK, {0., 0., 0.}, 17,
              earth_radius_m);
      double const near_observation_position[3]{
          76.4524280117653, -39.517778763940704,
          6370999.999478279};
      double const inward[3]{
          -0.46213010322832354, 0.1607580485828842,
          -0.8721196119260258};
      auto const radius = std::sqrt(
          near_observation_position[0] *
                  near_observation_position[0] +
              near_observation_position[1] *
                  near_observation_position[1] +
              near_observation_position[2] *
                  near_observation_position[2]);
      require(
          radius > production_snapshot.observation_radius_m &&
              radius -
                      production_snapshot.observation_radius_m <
                  AtmosphereBoundaryGuardM,
          "observation regression state must lie inside the terminal guard");
      auto const boundary = distanceToAtmosphereBoundary(
          production_snapshot, near_observation_position,
          inward);
      require(
          boundary.status == AtmosphereStatus::Success,
          "near-observation boundary lookup must succeed");
      requireClose(
          boundary.radius_m,
          production_snapshot.observation_radius_m,
          1.e-15,
          "near-observation state must select the terminal sphere");
      require(
          boundary.distance_m > 0. &&
              boundary.distance_m <
                  AtmosphereBoundaryGuardM,
          "near-observation state must retain its sub-guard root");
      requireClose(
          boundary.distance_m, 6.828736513853073e-5,
          2.e-5,
          "near-observation stable sphere root differs");
      auto const layer = queryAtmosphereLayer(
          production_snapshot, near_observation_position,
          inward);
      auto const grammage = atmosphereGrammage(
          production_snapshot, layer.layer_index,
          near_observation_position, inward,
          boundary.distance_m);
      require(
          grammage.status == AtmosphereStatus::Success &&
              grammage.value >= 0.,
          "near-observation grammage must remain finite");
    }

    struct SegmentCase {
      double altitude_m;
      std::array<double, 3> direction;
      double distance_m;
    };
    std::array<SegmentCase, 6> const segments{{
        {2000., normalized({0.6, 0., 0.8}), 80.},
        {6000., normalized({0.3, 0.4, -0.866}), 70.},
        {20000., normalized({0.8, 0., 0.6}), 120.},
        {60000., normalized({0.2, 0.9, -0.38}), 200.},
        {105000., normalized({0.5, 0.5, 0.707}), 300.},
        {110000., normalized({1., 0., 0.}), 500.},
    }};
    for (auto const& segment : segments) {
      Point const point(
          coordinate_system,
          {0_m, 0_m,
           (earth_radius_m + segment.altitude_m) * 1_m});
      double const position_m[3]{
          0., 0., earth_radius_m + segment.altitude_m};
      double const direction[3]{
          segment.direction[0], segment.direction[1],
          segment.direction[2]};
      auto const layer =
          queryAtmosphereLayer(snapshot, position_m, direction);
      require(layer.status == AtmosphereStatus::Success,
              "segment layer lookup must succeed");
      auto const gpu_grammage = atmosphereGrammage(
          snapshot, layer.layer_index, position_m, direction,
          segment.distance_m);
      require(gpu_grammage.status == AtmosphereStatus::Success,
              "GPU grammage integration must succeed");

      VelocityVector const velocity(
          coordinate_system,
          {segment.direction[0] * constants::c,
           segment.direction[1] * constants::c,
           segment.direction[2] * constants::c});
      StraightTrajectory const trajectory(
          Line(point, velocity),
          segment.distance_m * 1_m / constants::c);
      auto const* node = universe->getContainingNode(point);
      auto const cpu_grammage =
          node->getModelProperties()
              .getIntegratedGrammage(trajectory) /
          grammage_unit;
      requireClose(gpu_grammage.value, cpu_grammage, 3.e-13,
                   "GPU grammage must equal SlidingPlanarExponential");

      auto const target_grammage =
          0.37 * gpu_grammage.value;
      auto const gpu_distance =
          atmosphereDistanceFromGrammage(
              snapshot, layer.layer_index, position_m, direction,
              target_grammage);
      require(gpu_distance.status == AtmosphereStatus::Success,
              "GPU inverse grammage must succeed");
      auto const cpu_distance =
          node->getModelProperties()
              .getArclengthFromGrammage(
                  trajectory,
                  target_grammage * grammage_unit) /
          1_m;
      requireClose(gpu_distance.value, cpu_distance, 5.e-13,
                   "GPU inverse grammage must equal CORSIKA CPU");
    }

    auto const down = normalized({0., 0., -1.});
    auto const up = normalized({0., 0., 1.});
    std::vector<EmInteractionRecord> interactions;

    auto interaction_particle =
        makePhoton(earth_radius_m, 50000., down, 1);
    auto const interaction_boundary =
        distanceToAtmosphereBoundary(
            snapshot, interaction_particle.position_m,
            interaction_particle.direction);
    auto const interaction_layer = queryAtmosphereLayer(
        snapshot, interaction_particle.position_m,
        interaction_particle.direction);
    auto const interaction_boundary_x = atmosphereGrammage(
        snapshot, interaction_layer.layer_index,
        interaction_particle.position_m,
        interaction_particle.direction,
        interaction_boundary.distance_m);
    auto interaction =
        noInteraction(interaction_particle, 0);
    interaction.status = EmInteractionStatus::Selected;
    interaction.process_id = 1;
    interaction.interaction_grammage_g_per_cm2 =
        0.5 * interaction_boundary_x.value;
    interactions.push_back(interaction);

    auto boundary = interaction;
    boundary.particle.history_id = 2;
    boundary.input_index = 1;
    boundary.interaction_grammage_g_per_cm2 =
        2. * interaction_boundary_x.value;
    interactions.push_back(boundary);
    interactions.push_back(noInteraction(
        makePhoton(earth_radius_m, 50000., up, 3), 2));
    interactions.push_back(noInteraction(
        makePhoton(earth_radius_m, 500., down, 4), 3));
    interactions.push_back(noInteraction(
        makePhoton(earth_radius_m, 110000., up, 5), 4));
    auto unsupported = noInteraction(
        makePhoton(earth_radius_m, 50000., down, 6), 5);
    unsupported.particle.pid =
        static_cast<std::int32_t>(EmPid::Electron);
    interactions.push_back(unsupported);

    CudaEmBackend backend;
    GpuEmConfig config{};
    config.device = 0;
    config.min_batch_size = 16;
    backend.initialize(snapshot, ProposalTableSet{}, config);
    auto const transported =
        backend.transportPhotonsForValidation(interactions);
    require(transported.input_interactions == interactions.size(),
            "transport input count must be retained");
    require(transported.records.size() == 5,
            "five photons must produce transport records");
    require(transported.fallback_events.size() == 1,
            "non-photon input must explicitly fall back");
    require(transported.fallback_events[0].input_index == 5,
            "fallback order and source index must be stable");
    require(
        transported.fallback_events[0].reason ==
            ProposalFallbackReason::UnsupportedParticle,
        "non-photon transport fallback reason must be explicit");
    require(transported.records[0].limit ==
                PhotonTransportLimit::Interaction,
            "first photon must stop at sampled interaction");
    require(transported.records[1].limit ==
                PhotonTransportLimit::LayerBoundary,
            "second photon must stop at lower layer boundary");
    require(transported.records[2].limit ==
                PhotonTransportLimit::LayerBoundary,
            "third photon must stop at upper layer boundary");
    require(transported.records[3].limit ==
                PhotonTransportLimit::ObservationSurface,
            "fourth photon must stop at configured observation surface");
    require(transported.records[4].limit ==
                PhotonTransportLimit::EscapedEnvironment,
            "fifth photon must escape at atmosphere top");
    for (std::size_t index = 0;
         index < transported.records.size(); ++index) {
      require(transported.records[index].input_index == index,
              "successful records must preserve stable input order");
      require(transported.records[index].distance_m >= 0.,
              "transport distance must be non-negative");
      require(
          transported.records[index]
                  .traversed_grammage_g_per_cm2 >= 0.,
          "transport grammage must be non-negative");
    }

    auto near_observation = noInteraction(
        makePhoton(
            earth_radius_m,
            100. + 0.5 * AtmosphereBoundaryGuardM,
            down, 7),
        0);
    CudaEmBackend near_observation_backend;
    near_observation_backend.initialize(
        snapshot, ProposalTableSet{}, config);
    auto const near_observation_transport =
        near_observation_backend
            .transportPhotonsForValidation(
                {near_observation});
    require(
        near_observation_transport.fallback_events.empty() &&
            near_observation_transport.records.size() == 1,
        "near-observation photon must not fall back");
    require(
        near_observation_transport.records[0].limit ==
                PhotonTransportLimit::ObservationSurface &&
            near_observation_transport.records[0].distance_m >
                0. &&
            near_observation_transport.records[0].distance_m <
                AtmosphereBoundaryGuardM,
        "near-observation photon must terminate at the sub-guard root");
    auto const vertex = queryAtmosphereLayer(
        snapshot, transported.records[0].end.position_m,
        transported.records[0].end.direction);
    requireClose(
        transported.records[0]
            .interaction.mass_density_g_per_cm3,
        vertex.density_g_per_cm3, 2.e-13,
        "interaction transport must inject vertex density for LPM");
    require(transported.records[0].end.step_id ==
                interactions[0].particle.step_id,
            "interaction stop must retain the selection step id");
    for (std::size_t index = 1; index < 5; ++index) {
      require(
          transported.records[index].end.step_id ==
              interactions[index].particle.step_id + 1,
          "boundary stop must advance Philox step id");
    }

    auto const& statistics = backend.statistics();
    require(statistics.photon_transport_batches == 1,
            "transport batch statistic must increment");
    require(statistics.photon_transport_interactions == 1,
            "interaction statistic must increment");
    require(statistics.photon_transport_boundaries == 2,
            "boundary statistic must increment");
    require(statistics.photon_transport_observations == 1,
            "observation statistic must increment");
    require(statistics.photon_transport_escapes == 1,
            "escape statistic must increment");

    std::cout << "GPU spherical-atmosphere validation passed: "
              << checks << " checks, "
              << transported.records.size()
              << " transport records, "
              << transported.fallback_events.size()
              << " fallback\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "GPU spherical-atmosphere validation failed after "
              << checks << " checks: " << error.what() << '\n';
    return 1;
  }
}
