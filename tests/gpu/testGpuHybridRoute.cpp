/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <GpuEmFlatRateTableFixture.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#include <testCascade.hpp>

#include <SetupTestTrajectory.hpp>

#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/geometry/Line.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/Sphere.hpp>
#include <corsika/framework/process/ContinuousProcess.hpp>
#include <corsika/framework/process/InteractionProcess.hpp>
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/EnvironmentSnapshotBuilder.hpp>
#include <corsika/gpu/em/PhysicalCudaEmRouter.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/ToyCudaEmRouter.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/setup/SetupTrajectory.hpp>

namespace {

  using namespace corsika;

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  auto makeEnvironment() {
    TestEnvironmentType environment;
    auto& universe = *environment.getUniverse();
    auto world = TestEnvironmentType::createNode<Sphere>(
        Point{environment.getCoordinateSystem(), 0_m, 0_m, 0_m},
        1_km * std::numeric_limits<double>::infinity());
    NuclearComposition const composition({Code::Proton}, {1.});
    world->setModelProperties<TestEnvironmentInterface>(
        19.2_g / cube(1_cm), composition);
    universe.addChild(std::move(world));
    return environment;
  }

  class TestTracking {
  public:
    template <typename TParticle>
    auto getTrack(TParticle const& particle) {
      ++calls_;
      auto const velocity =
          particle.getMomentum() / particle.getEnergy() * constants::c;
      Line const line{particle.getPosition(), velocity};
      auto const duration =
          std::numeric_limits<TimeType::value_type>::infinity() * second;
      return std::make_tuple(
          setup::testing::make_track<setup::Trajectory>(line, duration),
          particle.getNode());
    }

    static std::string getName() { return "GPU hybrid route test tracking"; }
    static std::string getVersion() { return "1"; }
    std::uint64_t calls() const { return calls_; }

  private:
    std::uint64_t calls_{};
  };

  class ZeroInteraction : public InteractionProcess<ZeroInteraction> {
  public:
    CrossSectionType getCrossSection(Code const, Code const,
                                     FourMomentum const&,
                                     FourMomentum const&) const {
      return 0_mb;
    }

    template <typename TView>
    void doInteraction(TView&, Code, Code, FourMomentum const&,
                       FourMomentum const&) {
      throw std::runtime_error("zero cross-section interaction was selected");
    }
  };

  class AbsorbScalarParticle : public ContinuousProcess<AbsorbScalarParticle> {
  public:
    template <typename TTrajectory>
    ProcessReturn doContinuous(Step<TTrajectory>&, bool) {
      ++calls_;
      return ProcessReturn::ParticleAbsorbed;
    }

    template <typename TParticle, typename TTrack>
    LengthType getMaxStepLength(TParticle&, TTrack&) {
      return std::numeric_limits<double>::infinity() * meter;
    }

    std::uint64_t calls() const { return calls_; }

  private:
    std::uint64_t calls_{};
  };

  class CountingOutput {
  public:
    void startOfShower() { ++starts_; }
    void endOfShower() { ++ends_; }
    std::uint64_t starts() const { return starts_; }
    std::uint64_t ends() const { return ends_; }

  private:
    std::uint64_t starts_{};
    std::uint64_t ends_{};
  };

  std::filesystem::path temporaryTablePath() {
    auto const stamp =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();
    return std::filesystem::temp_directory_path() /
           ("c8_physical_hybrid_route_" +
            std::to_string(stamp) + ".c8emrt");
  }

  void addLeptonRateTables(
      gpu::em::tables::RateTableSet& source) {
    using namespace gpu::em;
    using namespace gpu::em::tables;
    auto const makeParticle =
        [&](std::int32_t pdg_id, std::string name) {
          ParticleRateTable particle;
          particle.pdg_id = pdg_id;
          particle.particle_name = std::move(name);
          particle.interaction_hash =
              pdg_id == 11 ? 9102 : 9103;
          particle.energies_MeV = {1., 10., 100.};
          auto brems = source.particles.front().columns.front();
          brems.process_id = BremsProcessId;
          brems.component_hash = 101;
          brems.process_name = "Brems";
          brems.parameterization =
              "BremsKelnerKokoulinPetrukhin";
          brems.target_name = "nitrogen";
          brems.rates_cm2_per_g = {2., 2., 2.};
          auto ionization = brems;
          ionization.process_id = IonizationProcessId;
          ionization.component_hash = 102;
          ionization.process_name = "Ioniz";
          ionization.parameterization =
              "IonizBetheBlochRossi";
          ionization.target_name = "oxygen";
          ionization.rates_cm2_per_g = {1., 1., 1.};
          particle.columns = {
              std::move(brems), std::move(ionization)};
          if (pdg_id == -11) {
            auto annihilation =
                source.particles.front().columns.at(3);
            annihilation.process_id =
                AnnihilationProcessId;
            annihilation.component_hash = 101;
            annihilation.process_name = "Annihilation";
            annihilation.parameterization =
                "AnnihilationHeitler";
            annihilation.target_name = "nitrogen";
            annihilation.rates_cm2_per_g =
                {2., 2., 2.};
            particle.columns.push_back(
                std::move(annihilation));
          }
          return particle;
        };
    source.particles.push_back(
        makeParticle(11, "electron"));
    source.particles.push_back(
        makeParticle(-11, "positron"));
  }

  std::size_t processCount(
      gpu::em::tables::RateTableSet const& table) {
    std::size_t count = 0;
    for (auto const& particle : table.particles) {
      count += particle.columns.size();
    }
    return count;
  }

} // namespace

int main() {
  int device_count = 0;
  auto const cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no CUDA device is accessible: "
              << cudaGetErrorString(cuda_status) << '\n';
    return 77;
  }

  try {
    logging::set_level(logging::level::warn);
    auto& random = RNGManager<>::getInstance();
    random.registerRandomStream("cascade");
    random.setSeed(0x48594252);
    random.getRandomStream("cascade").reset();

    auto environment = makeEnvironment();
    auto const coordinate_system = environment.getCoordinateSystem();

    ZeroInteraction zero;
    AbsorbScalarParticle absorb;
    auto sequence = make_sequence(zero, absorb);

    TestCascadeIdentityStack stack;
    stack.addParticle(std::make_tuple(
        Code::Proton, 1_GeV,
        DirectionVector{coordinate_system, {0., 0., -1.}},
        Point{coordinate_system, 0_m, 0_m, 10_km}, 0_s));
    stack.addParticle(std::make_tuple(
        Code::Photon, 100_GeV,
        DirectionVector{coordinate_system, {0., 0., -1.}},
        Point{coordinate_system, 0_m, 0_m, 10_km}, 0_s));

    gpu::em::EnvironmentSnapshot environment_snapshot{};
    gpu::em::ProposalTableSet tables{};
    gpu::em::GpuEmConfig config{};
    config.min_batch_size = 16;
    config.memory_fraction = 0.01;
    config.random_seed = 0x47505548;
    config.shower_id = 1;

    gpu::em::CudaEmBackend backend;
    backend.initialize(environment_snapshot, tables, config);
    gpu::em::ToyCudaEmRouter<TestCascadeIdentityStack> router{
        backend, coordinate_system};

    TestTracking tracking;
    CountingOutput output;
    using Router =
        gpu::em::ToyCudaEmRouter<TestCascadeIdentityStack>;
    HybridCascade<TestTracking, decltype(sequence), CountingOutput,
                  TestCascadeIdentityStack, Router>
        cascade{environment, tracking, sequence, output, stack, router};
    cascade.run();

    auto const& router_statistics = router.statistics();
    auto const& gpu_statistics = backend.statistics();
    auto const& scheduler_statistics = cascade.schedulerStatistics();

    require(stack.getEntries() == 0, "hybrid cascade left particles on the CPU stack");
    require(backend.empty(), "hybrid cascade left particles on the GPU");
    require(output.starts() == 1 && output.ends() == 1,
            "hybrid cascade output lifecycle is incomplete");
    require(tracking.calls() == 1,
            "the non-EM proton did not use exactly one scalar step");
    require(absorb.calls() == 1,
            "the scalar continuous process did not absorb exactly one proton");
    require(router_statistics.wavefronts > 0,
            "the EM particle never entered a GPU wavefront");
    require(router_statistics.maximum_input_batch > 0,
            "the GPU router observed an empty maximum batch");
    require(router_statistics.particles_staged ==
                gpu_statistics.particles_advanced,
            "CPU staging and GPU advanced-particle counts differ");
    require(router_statistics.particles_returned_to_cpu ==
                gpu_statistics.particles_produced,
            "GPU production and CPU re-import counts differ");
    require(scheduler_statistics.acquired_particle_steps ==
                router_statistics.particles_staged + tracking.calls(),
            "hybrid scheduler lost or duplicated a particle");
    require(scheduler_statistics.acquired_particle_steps ==
                scheduler_statistics.completed_particle_steps,
            "hybrid scheduler has an incomplete particle step");

    std::cout << "GPU hybrid route passed " << checks << " checks: "
              << router_statistics.particles_staged << " EM particle steps in "
              << router_statistics.wavefronts << " wavefronts, max batch "
              << router_statistics.maximum_input_batch << '\n';

    auto const table_path = temporaryTablePath();
    auto source =
        gpu::em::tables::testing::makeFlatRateTableFixture();
    for (auto& column : source.particles.front().columns) {
      if (column.process_id ==
          gpu::em::PhotoelectricProcessId) {
        std::fill(
            column.rates_cm2_per_g.begin(),
            column.rates_cm2_per_g.end(), 1.e6);
      } else {
        std::fill(
            column.rates_cm2_per_g.begin(),
            column.rates_cm2_per_g.end(), 0.);
      }
    }
    addLeptonRateTables(source);
    auto const digest =
        gpu::em::tables::writeRateTable(table_path, source);
    gpu::em::ProposalTableSet descriptor{};
    descriptor.process_count =
        static_cast<std::uint32_t>(processCount(source));
    descriptor.content_hash = digest;
    auto const earth_radius_m =
        constants::EarthRadius::Mean / 1_m;
    auto physical_environment =
        gpu::em::makeCorsika7AtmosphereSnapshot(
            AtmosphereId::USStdBK, {0., 0., 0.}, 17,
            earth_radius_m + 100., {1.e-5, 0., 0.});
    gpu::em::GpuEmConfig physical_config{};
    physical_config.device = 0;
    // This test exercises direct GPU routing. Small-batch scalar expansion is
    // covered separately by the production application smoke test.
    physical_config.min_batch_size = 1;
    physical_config.memory_fraction = 0.10;
    physical_config.table_tolerance = 1.e-3;
    physical_config.random_seed = 0x50485953;
    physical_config.shower_id = 2;
    physical_config.table_cache = table_path;
    physical_config.resident_cross_species = true;
    gpu::em::CudaEmBackend physical_backend;
    physical_backend.initialize(
        physical_environment, descriptor, physical_config);

    auto physical_cpu_environment = makeEnvironment();
    auto const physical_coordinate_system =
        physical_cpu_environment.getCoordinateSystem();
    {
      auto photoelectric_config =
          physical_config;
      photoelectric_config.resident_cross_species =
          false;
      gpu::em::CudaEmBackend photoelectric_backend;
      photoelectric_backend.initialize(
          physical_environment, descriptor,
          photoelectric_config);
      TestCascadeIdentityStack photoelectric_stack;
      auto photon = photoelectric_stack.addParticle(
          std::make_tuple(
              Code::Photon, 10_MeV,
              DirectionVector{
                  physical_coordinate_system,
                  {0., 0., -1.}},
              Point{
                  physical_coordinate_system, 0_m, 0_m,
                  (earth_radius_m + 2000.) * meter},
              0_s));
      auto const history = photon.getHistoryId();
      auto const step = photon.beginTransportStep();
      using PhotoelectricRouter =
          gpu::em::PhysicalCudaEmRouter<
              TestCascadeIdentityStack>;
      require(
          PhotoelectricRouter::isPermittedScalarFallbackReason(
              gpu::em::ProposalFallbackReason::UnsupportedGeometry) &&
              !PhotoelectricRouter::isPermittedScalarFallbackReason(
                  gpu::em::ProposalFallbackReason::InvalidMassDensity) &&
              !PhotoelectricRouter::isPermittedScalarFallbackReason(
                  gpu::em::ProposalFallbackReason::
                      AtmosphereGrammageFailed),
          "production strict fallback policy accepts a numerical transport failure");
      PhotoelectricRouter photoelectric_router{
          photoelectric_backend, physical_coordinate_system,
          physical_environment};
      require(
          photoelectric_router.canRoute(photon, step),
          "photoelectric deposit fixture was not GPU routable");
      photoelectric_router.stage(
          photon, history, photon.getParentHistoryId(),
          photon.getGeneration(), step);
      photon.erase();
      photoelectric_router.advanceOneWavefrontAndReturn(
          photoelectric_stack);
      auto const deposit_step = std::find_if(
          photoelectric_router.stepRecords().begin(),
          photoelectric_router.stepRecords().end(),
          [](gpu::em::EmStepRecord const& record) {
            return record.process_id ==
                   gpu::em::PhotoelectricProcessId;
          });
      require(
          photoelectric_router.statistics()
                      .deposited_energy_GeV > 0. &&
              deposit_step !=
                  photoelectric_router.stepRecords().end() &&
              deposit_step->deposited_energy_GeV > 0.,
          "photoelectric K-shell binding deposit was not recorded");
    }
    {
      auto expansion_config = physical_config;
      expansion_config.min_batch_size = 16;
      expansion_config.shower_id = 3;
      expansion_config.resident_cross_species = false;
      gpu::em::CudaEmBackend expansion_backend;
      expansion_backend.initialize(
          physical_environment, descriptor,
          expansion_config);
      TestCascadeIdentityStack expansion_stack;
      expansion_stack.addParticle(std::make_tuple(
          Code::Electron,
          100_MeV - get_mass(Code::Electron),
          DirectionVector{
              physical_coordinate_system, {0., 0., -1.}},
          Point{
              physical_coordinate_system, 0_m, 0_m,
              (earth_radius_m + 50000.) * meter},
          0_s));
      ZeroInteraction expansion_zero;
      AbsorbScalarParticle expansion_absorb;
      auto expansion_sequence =
          make_sequence(expansion_zero, expansion_absorb);
      TestTracking expansion_tracking;
      CountingOutput expansion_output;
      using ExpansionRouter =
          gpu::em::PhysicalCudaEmRouter<
              TestCascadeIdentityStack>;
      ExpansionRouter expansion_router{
          expansion_backend, physical_coordinate_system,
          physical_environment};
      HybridCascade<
          TestTracking, decltype(expansion_sequence),
          CountingOutput, TestCascadeIdentityStack,
          ExpansionRouter>
          expansion_cascade{
              physical_cpu_environment, expansion_tracking,
              expansion_sequence, expansion_output,
              expansion_stack, expansion_router};
      expansion_cascade.run();
      auto const& expansion_statistics =
          expansion_router.statistics();
      require(
          expansion_statistics.small_batch_expansions == 1 &&
              expansion_statistics
                      .particles_returned_for_cpu_wavefront_expansion ==
                  1 &&
              expansion_statistics
                      .cpu_wavefront_expansion_steps_executed ==
                  1 &&
              expansion_statistics.wavefronts == 0 &&
              expansion_tracking.calls() == 1 &&
              expansion_absorb.calls() == 1,
          "small physical wavefront was not expanded by exactly one scalar step");

      TestCascadeIdentityStack stalled_stack;
      auto stalled = stalled_stack.addParticle(
          std::make_tuple(
              Code::Photon, 10_MeV,
              DirectionVector{
                  physical_coordinate_system,
                  {0., 0., -1.}},
              Point{
                  physical_coordinate_system, 0_m, 0_m,
                  (earth_radius_m + 50000.) * meter},
              0_s));
      auto const stalled_history =
          stalled.getHistoryId();
      auto const staged_step =
          stalled.beginTransportStep();
      require(
          expansion_router.canRoute(
              stalled, staged_step),
          "stalled-boundary fixture was not initially routable");
      expansion_router.stage(
          stalled, stalled_history,
          stalled.getParentHistoryId(),
          stalled.getGeneration(), staged_step);
      stalled.erase();
      expansion_router.advanceOneWavefrontAndReturn(
          stalled_stack);
      auto returned = stalled_stack.getNextParticle();
      auto const scalar_step =
          returned.beginTransportStep();
      require(
          !expansion_router.canRoute(
              returned, scalar_step),
          "small-batch scalar bypass was not consumed");
      // Model a numerically zero boundary step: the transport identity
      // advances, but all physical state remains unchanged.
      auto const repeated_step =
          returned.beginTransportStep();
      require(
          expansion_router.canRoute(
              returned, repeated_step),
          "zero-progress boundary state was not GPU routable");
      expansion_router.stage(
          returned, returned.getHistoryId(),
          returned.getParentHistoryId(),
          returned.getGeneration(), repeated_step);
      returned.erase();
      expansion_router.advanceOneWavefrontAndReturn(
          stalled_stack);
      require(
          expansion_router.statistics()
                  .stalled_boundary_gpu_flushes == 1 &&
              expansion_router.statistics().wavefronts == 1,
          "zero-progress scalar expansion was not forced onto the GPU");

      TestCascadeIdentityStack budget_stack;
      budget_stack.addParticle(std::make_tuple(
          Code::Photon, 10_MeV,
          DirectionVector{
              physical_coordinate_system,
              {0., 0., -1.}},
          Point{
              physical_coordinate_system, 0_m, 0_m,
              (earth_radius_m + 50000.) * meter},
          0_s));
      auto const expansions_before =
          expansion_router.statistics()
              .small_batch_expansions;
      auto const budget_flushes_before =
          expansion_router.statistics()
              .scalar_expansion_budget_gpu_flushes;
      auto const wavefronts_before =
          expansion_router.statistics().wavefronts;
      for (int round = 0; round < 9; ++round) {
        auto routed = budget_stack.getNextParticle();
        auto const route_step =
            routed.beginTransportStep();
        require(
            expansion_router.canRoute(
                routed, route_step),
            "scalar-expansion budget fixture was not GPU routable");
        expansion_router.stage(
            routed, routed.getHistoryId(),
            routed.getParentHistoryId(),
            routed.getGeneration(), route_step);
        routed.erase();
        expansion_router.advanceOneWavefrontAndReturn(
            budget_stack);
        if (round < 8) {
          auto scalar =
              budget_stack.getNextParticle();
          auto const scalar_step =
              scalar.beginTransportStep();
          require(
              !expansion_router.canRoute(
                  scalar, scalar_step),
              "scalar-expansion bypass was not consumed");
          scalar.setPosition(Point{
              physical_coordinate_system,
              0_m, 0_m,
              (earth_radius_m + 50000. -
               static_cast<double>(round + 1) *
                   1.e-3) *
                  meter});
        }
      }
      require(
          expansion_router.statistics()
                      .small_batch_expansions -
                  expansions_before ==
              8 &&
              expansion_router.statistics()
                      .scalar_expansion_budget_gpu_flushes -
                  budget_flushes_before ==
              1 &&
              expansion_router.statistics()
                      .wavefronts -
                  wavefronts_before ==
              1,
          "scalar-expansion round budget did not force an underfilled GPU wavefront");
    }
    {
      // A state exactly on the outer atmosphere boundary and pointing
      // outwards is deliberately owned by queryAtmosphereLayer() within its
      // floating-point guard, but has no forward sphere intersection.  The
      // device reports UnsupportedGeometry; HybridCascade must import the
      // unchanged transport identity and bypass GPU routing for exactly one
      // scalar step.
      auto boundary_config = physical_config;
      boundary_config.shower_id = 4;
      gpu::em::CudaEmBackend boundary_backend;
      boundary_backend.initialize(
          physical_environment, descriptor,
          boundary_config);
      auto const outer_radius_m =
          physical_environment
              .atmosphere_layers[
                  physical_environment.number_of_layers - 1]
              .outer_radius_m;
      TestCascadeIdentityStack boundary_stack;
      boundary_stack.addParticle(std::make_tuple(
          Code::Photon, 10_MeV,
          DirectionVector{
              physical_coordinate_system, {0., 0., 1.}},
          Point{
              physical_coordinate_system, 0_m, 0_m,
              outer_radius_m * meter},
          0_s));
      ZeroInteraction boundary_zero;
      AbsorbScalarParticle boundary_absorb;
      auto boundary_sequence =
          make_sequence(boundary_zero, boundary_absorb);
      TestTracking boundary_tracking;
      CountingOutput boundary_output;
      using BoundaryRouter =
          gpu::em::PhysicalCudaEmRouter<
              TestCascadeIdentityStack>;
      BoundaryRouter boundary_router{
          boundary_backend, physical_coordinate_system,
          physical_environment};
      boundary_router.setFailOnUnexpectedFallback(true);
      HybridCascade<
          TestTracking, decltype(boundary_sequence),
          CountingOutput, TestCascadeIdentityStack,
          BoundaryRouter>
          boundary_cascade{
              physical_cpu_environment, boundary_tracking,
              boundary_sequence, boundary_output,
              boundary_stack, boundary_router};
      boundary_cascade.run();
      auto const& boundary_statistics =
          boundary_router.statistics();
      auto const& boundary_scheduler =
          boundary_cascade.schedulerStatistics();
      require(
          boundary_statistics.particles_staged == 1 &&
              boundary_statistics.wavefronts == 1 &&
              boundary_statistics
                      .particles_returned_for_cpu_fallback ==
                  1 &&
              boundary_statistics
                      .cpu_fallback_steps_executed ==
                  1 &&
              boundary_statistics
                      .cpu_fallbacks_by_reason.at(
                          static_cast<std::int32_t>(
                              gpu::em::ProposalFallbackReason::
                                  UnsupportedGeometry)) ==
                  1 &&
              boundary_router.fallbackEvents().size() == 1 &&
              boundary_router.fallbackEvents().front().reason ==
                  gpu::em::ProposalFallbackReason::
                      UnsupportedGeometry,
          "unsupported geometry did not produce one explicit scalar fallback");
      require(
          boundary_tracking.calls() == 1 &&
              boundary_absorb.calls() == 1 &&
              boundary_stack.getEntries() == 0 &&
              !boundary_router.pending(),
          "unsupported geometry did not execute exactly one scalar step");
      require(
          boundary_scheduler.acquired_particle_steps == 2 &&
              boundary_scheduler.completed_particle_steps == 2,
          "unsupported-geometry fallback lost or duplicated a transport step");
    }
    {
      // Keep a valid schema while forcing its exponential density evaluation
      // to overflow.  This is a numerical transport failure, not a declared
      // geometry capability boundary, and strict production routing must
      // abort instead of importing it for a silent scalar retry.
      auto invalid_density_environment =
          physical_environment;
      auto& first_layer =
          invalid_density_environment.atmosphere_layers[0];
      first_layer.density_parameter_b =
          first_layer.outer_radius_m + 1.e9;
      first_layer.density_parameter_c = -1.;
      require(
          gpu::em::atmosphere_detail::validEnvironment(
              invalid_density_environment),
          "invalid-density fixture does not retain a valid snapshot schema");
      auto strict_config = physical_config;
      strict_config.shower_id = 5;
      gpu::em::CudaEmBackend strict_backend;
      strict_backend.initialize(
          invalid_density_environment, descriptor,
          strict_config);
      TestCascadeIdentityStack strict_stack;
      strict_stack.addParticle(std::make_tuple(
          Code::Photon, 10_MeV,
          DirectionVector{
              physical_coordinate_system, {0., 0., -1.}},
          Point{
              physical_coordinate_system, 0_m, 0_m,
              (earth_radius_m + 2000.) * meter},
          0_s));
      ZeroInteraction strict_zero;
      AbsorbScalarParticle strict_absorb;
      auto strict_sequence =
          make_sequence(strict_zero, strict_absorb);
      TestTracking strict_tracking;
      CountingOutput strict_output;
      using StrictRouter =
          gpu::em::PhysicalCudaEmRouter<
              TestCascadeIdentityStack>;
      StrictRouter strict_router{
          strict_backend, physical_coordinate_system,
          invalid_density_environment};
      strict_router.setFailOnUnexpectedFallback(true);
      HybridCascade<
          TestTracking, decltype(strict_sequence),
          CountingOutput, TestCascadeIdentityStack,
          StrictRouter>
          strict_cascade{
              physical_cpu_environment, strict_tracking,
              strict_sequence, strict_output,
              strict_stack, strict_router};
      bool rejected_numerical_fallback = false;
      try {
        strict_cascade.run();
      } catch (std::runtime_error const& error) {
        auto const message = std::string{error.what()};
        rejected_numerical_fallback =
            message.find(
                "unexpected CUDA EM fallback cannot be retried silently") !=
                std::string::npos &&
            message.find("atmosphere_grammage_failed") !=
                std::string::npos;
      }
      require(
          rejected_numerical_fallback,
          "strict routing did not abort on a numerical atmosphere failure");
      require(
          strict_router.statistics()
                      .cpu_fallbacks_by_reason.at(
                          static_cast<std::int32_t>(
                              gpu::em::ProposalFallbackReason::
                                  AtmosphereGrammageFailed)) ==
                  1 &&
              strict_router.statistics()
                      .particles_returned_for_cpu_fallback ==
                  0 &&
              strict_router.statistics()
                      .cpu_fallback_steps_executed ==
                  0 &&
              strict_tracking.calls() == 0 &&
              strict_absorb.calls() == 0,
          "numerical atmosphere failure was silently retried on the CPU");
    }
    TestCascadeIdentityStack physical_stack;
    physical_stack.addParticle(std::make_tuple(
        Code::Proton, 1_GeV,
        DirectionVector{
            physical_coordinate_system, {0., 0., -1.}},
        Point{
            physical_coordinate_system, 0_m, 0_m,
            (earth_radius_m + 50000.) * meter},
        0_s));
    physical_stack.addParticle(std::make_tuple(
        Code::Electron,
        100_MeV - get_mass(Code::Electron),
        DirectionVector{
            physical_coordinate_system, {0., 0., -1.}},
        Point{
            physical_coordinate_system, 0_m, 0_m,
            (earth_radius_m + 50000.) * meter},
        0_s));

    ZeroInteraction physical_zero;
    AbsorbScalarParticle physical_absorb;
    auto physical_sequence =
        make_sequence(physical_zero, physical_absorb);
    TestTracking physical_tracking;
    CountingOutput physical_output;
    using PhysicalRouter =
        gpu::em::PhysicalCudaEmRouter<
            TestCascadeIdentityStack>;
    PhysicalRouter physical_router{
        physical_backend, physical_coordinate_system,
        physical_environment};
    HybridCascade<
        TestTracking, decltype(physical_sequence),
        CountingOutput, TestCascadeIdentityStack,
        PhysicalRouter>
        physical_cascade{
            physical_cpu_environment, physical_tracking,
            physical_sequence, physical_output,
            physical_stack, physical_router};
    physical_cascade.run();

    auto const& physical_statistics =
        physical_router.statistics();
    auto const& physical_scheduler =
        physical_cascade.schedulerStatistics();
    require(
        physical_stack.getEntries() == 0 &&
            !physical_router.pending(),
        "physical HybridCascade left owned particles");
    require(
        physical_output.starts() == 1 &&
            physical_output.ends() == 1,
        "physical HybridCascade output lifecycle is incomplete");
    require(
        physical_statistics.particles_staged == 1 &&
            physical_statistics.leptons_advanced > 0 &&
            physical_statistics.wavefronts > 0 &&
            physical_statistics.reserved_history_ids > 0 &&
            physical_backend.statistics()
                    .cross_species_particles_kept_on_device >
                0 &&
            physical_backend.pendingPhotonCount() == 0 &&
            physical_backend.pendingLeptonCount() == 0,
        "physical HybridCascade did not drain its device-only cross-species front");
    require(
        !physical_router.stepRecords().empty() &&
            !physical_router.radioTracks().empty() &&
            physical_router.radioTracks().size() <=
                physical_router.stepRecords().size(),
        "physical HybridCascade lost its e+/e- output tracks");
    require(
        physical_statistics.cpu_fallback_steps_executed ==
            physical_statistics
                .particles_returned_for_cpu_fallback,
        "physical HybridCascade fallback bypass count differs");
    require(
        physical_tracking.calls() ==
            1 +
                physical_statistics
                    .cpu_fallback_steps_executed,
        "physical HybridCascade scalar ownership count differs");
    require(
        physical_scheduler.acquired_particle_steps ==
                physical_scheduler.completed_particle_steps &&
            physical_scheduler.acquired_particle_steps ==
                physical_statistics.particles_staged +
                    physical_tracking.calls(),
        "physical HybridCascade scheduler lost a particle");
    std::filesystem::remove(table_path);
    std::cout
        << "Physical GPU hybrid route passed: "
        << physical_statistics.leptons_advanced
        << " lepton advances in "
        << physical_statistics.wavefronts
        << " wavefronts, "
        << physical_statistics
               .particles_returned_for_cpu_fallback
        << " CPU fallbacks\n";
  } catch (std::exception const& error) {
    std::cerr << "GPU hybrid route failed after " << checks
              << " checks: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
