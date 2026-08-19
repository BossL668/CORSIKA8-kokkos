/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

/* clang-format off */
// InteractionCounter used boost/histogram, which
// fails if boost/type_traits have been included before. Thus, we have
// to include it first...
#include <corsika/framework/process/InteractionCounter.hpp>
/* clang-format on */
#include <corsika/framework/core/Cascade.hpp>
#include <corsika/framework/core/HadronicWorkQueue.hpp>
#ifdef CORSIKA8_WITH_CUDA_EM
#include <cuda_runtime_api.h>

#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/gpu/em/CorsikaOutputSink.hpp>
#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/EnvironmentSnapshotBuilder.hpp>
#include <corsika/gpu/em/GpuEmRunOutput.hpp>
#include <corsika/gpu/em/PhysicalCudaEmRouter.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/ProcessSequenceCompatibility.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/detail/DeviceWavefrontBucketing.hpp>
#include <corsika/gpu/em/ProposalCpuFallbackHandler.hpp>
#include <corsika/gpu/em/tables/MediumConfig.hpp>
#include <corsika/gpu/em/tables/ProposalMedium.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>
#include <corsika/gpu/em/tables/Sha256.hpp>
#include <corsika/gpu/radio/RadioSnapshotBuilder.hpp>
#endif
#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/framework/geometry/Plane.hpp>
#include <corsika/framework/geometry/Sphere.hpp>
#include <corsika/framework/process/DynamicInteractionProcess.hpp>
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/framework/process/SwitchProcessSequence.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/framework/random/PowerLawDistribution.hpp>
#include <corsika/framework/utility/CorsikaFenv.hpp>
#include <corsika/framework/utility/SaveBoostHistogram.hpp>

#include <corsika/modules/writers/EnergyLossWriter.hpp>
#include <corsika/modules/writers/InteractionWriter.hpp>
#include <corsika/modules/writers/LongitudinalWriter.hpp>
#include <corsika/modules/writers/ProductionWriter.hpp>
#include <corsika/modules/writers/PrimaryWriter.hpp>
#include <corsika/modules/writers/SubWriter.hpp>
#include <corsika/output/OutputManager.hpp>
#include <corsika/output/EnergyLedgerRunOutput.hpp>
#include <corsika/output/SimulationTiming.hpp>

#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/media/ShowerAxis.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>

#include <corsika/modules/BetheBlochPDG.hpp>
#include <corsika/modules/Epos.hpp>
#include <corsika/modules/ObservationPlane.hpp>
#include <corsika/modules/PROPOSAL.hpp>
#include <corsika/modules/proposal/ThresholdPhotoproductionModel.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <corsika/modules/Pythia8.hpp>
#include <corsika/modules/QGSJetII.hpp>
#include <corsika/modules/QGSJetIII.hpp>
#include <corsika/modules/Sibyll.hpp>
#include <corsika/modules/Sophia.hpp>
#include <corsika/modules/StackInspector.hpp>
#include <corsika/modules/thinning/EMThinning.hpp>
#include <corsika/modules/LongitudinalProfile.hpp>
#include <corsika/modules/ProductionProfile.hpp>

// for ICRC2023
#ifdef WITH_FLUKA
#include <corsika/modules/FLUKA.hpp>
#else
#include <corsika/modules/UrQMD.hpp>
#endif
#include <corsika/modules/TAUOLA.hpp>

#include <corsika/modules/radio/CoREAS.hpp>
#include <corsika/modules/radio/RadioProcess.hpp>
#include <corsika/modules/radio/ZHS.hpp>
#include <corsika/modules/radio/observers/Observer.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/propagators/RadioPropagator.hpp>

#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupTrajectory.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>
#include <corsika/validation/CudaDecisionReplayTape.hpp>
#include <corsika/validation/CudaReplayTrace.hpp>

#include <boost/filesystem.hpp>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <unistd.h>

using namespace corsika;
using namespace std;

using EnvironmentInterface =
    IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
using EnvType = Environment<EnvironmentInterface>;
using StackType = setup::HybridStack<EnvType>;
using TrackingType = setup::Tracking;
using Particle = StackType::particle_type;

//
// Dedicated homogeneous-ice cascade application.  This source deliberately
// does not share a runtime environment switch with c8_air_shower: its CORSIKA
// and CUDA medium contracts are always WaterIce.
// w.r.t. what was implemented in CORSIKA 7. Users may want to change some of the
// specifics (observation altitude, magnetic field, energy cuts, etc.), but this
// example is the most physics-complete one and should be used for full simulations
// of particle cascades in homogeneous ice
//

long registerRandomStreams(long seed) {
  RNGManager<>::getInstance().registerRandomStream("cascade");
  RNGManager<>::getInstance().registerRandomStream("qgsjet");
  RNGManager<>::getInstance().registerRandomStream("qgsjetIII");
  RNGManager<>::getInstance().registerRandomStream("sibyll");
  RNGManager<>::getInstance().registerRandomStream("sophia");
  RNGManager<>::getInstance().registerRandomStream("epos");
  RNGManager<>::getInstance().registerRandomStream("pythia");
  RNGManager<>::getInstance().registerRandomStream("urqmd");
  RNGManager<>::getInstance().registerRandomStream("fluka");
  RNGManager<>::getInstance().registerRandomStream("proposal");
  RNGManager<>::getInstance().registerRandomStream("thinning");
  RNGManager<>::getInstance().registerRandomStream("primary_particle");
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
    CORSIKA_LOG_INFO("random seed (auto) {}", seed);
  } else {
    CORSIKA_LOG_INFO("random seed {}", seed);
  }
  RNGManager<>::getInstance().setSeed(seed);
  return seed;
}

template <typename T>
using MyHomogeneousEnv = UniformRefractiveIndex<
    MediumPropertyModel<UniformMagneticField<HomogeneousMedium<T>>>>;

/** Straight radio propagation specialized for the uniform ice contract.
 *
 * Unlike DummyTestPropagator this implementation does not require the track
 * particle to expose getNode(), so it accepts both scalar CORSIKA particles
 * and the compact CUDA radio-track adapter.  The source/destination material
 * indices are still queried independently and a non-uniform path is rejected.
 */
template <typename TEnvironment>
class HomogeneousIceRadioPropagator final
    : public RadioPropagator<HomogeneousIceRadioPropagator<TEnvironment>,
                             TEnvironment> {
  using Base =
      RadioPropagator<HomogeneousIceRadioPropagator<TEnvironment>, TEnvironment>;
  using SignalPathCollection = typename Base::SignalPathCollection;

public:
  explicit HomogeneousIceRadioPropagator(TEnvironment const& environment)
      : Base(environment) {}

  template <typename ParticleType>
  SignalPathCollection propagate([[maybe_unused]] ParticleType const& particle,
                                 Point const& source,
                                 Point const& destination) {
    auto const* universe = Base::env_.getUniverse().get();
    auto const* source_node = universe->getContainingNode(source);
    auto const* destination_node = universe->getContainingNode(destination);
    double const source_index =
        source_node->getModelProperties().getRefractiveIndex(source);
    double const destination_index =
        destination_node->getModelProperties().getRefractiveIndex(destination);
    double const index_scale =
        std::max({1., std::abs(source_index), std::abs(destination_index)});
    if (std::abs(source_index - destination_index) > 1.e-12 * index_scale) {
      throw std::runtime_error(
          "c8_ice_cascade radio path left the homogeneous ice medium");
    }

    auto const emit = (destination - source).normalized();
    auto const receive = -emit;
    auto const distance = (destination - source).getNorm();
    double const average_index = 0.5 * (source_index + destination_index);
    TimeType const propagation_time = average_index * distance / constants::c;
    std::deque<Point> points{source, destination};
    return {SignalPath(propagation_time, average_index, source_index,
                       destination_index, emit, receive, distance, points)};
  }
};

template <typename TEnvironment>
HomogeneousIceRadioPropagator<TEnvironment>
make_homogeneous_ice_radio_propagator(TEnvironment const& environment) {
  return HomogeneousIceRadioPropagator<TEnvironment>(environment);
}

ParticleCutStatistics subtractStatistics(
    ParticleCutStatistics const& after,
    ParticleCutStatistics const& before) {
  ParticleCutStatistics result{};
  result.particles = after.particles - before.particles;
  result.invisible_particles =
      after.invisible_particles - before.invisible_particles;
  result.weighted_kinetic_energy_GeV =
      after.weighted_kinetic_energy_GeV -
      before.weighted_kinetic_energy_GeV;
  result.weighted_rest_mass_energy_GeV =
      after.weighted_rest_mass_energy_GeV -
      before.weighted_rest_mass_energy_GeV;
  result.weighted_invisible_kinetic_energy_GeV =
      after.weighted_invisible_kinetic_energy_GeV -
      before.weighted_invisible_kinetic_energy_GeV;
  for (auto const& [pdg, current] : after.by_pdg) {
    auto const found = before.by_pdg.find(pdg);
    auto const previous =
        found == before.by_pdg.end()
            ? ParticleCutSpeciesStatistics{}
            : found->second;
    auto& delta = result.by_pdg[pdg];
    delta.particles = current.particles - previous.particles;
    delta.weighted_kinetic_energy_GeV =
        current.weighted_kinetic_energy_GeV -
        previous.weighted_kinetic_energy_GeV;
    delta.weighted_rest_mass_energy_GeV =
        current.weighted_rest_mass_energy_GeV -
        previous.weighted_rest_mass_energy_GeV;
  }
  return result;
}

InteractionEnergyLedgerStatistics subtractStatistics(
    InteractionEnergyLedgerStatistics const& after,
    InteractionEnergyLedgerStatistics const& before) {
  InteractionEnergyLedgerStatistics result{};
  result.audited_interactions =
      after.audited_interactions - before.audited_interactions;
  result.deferred_interactions =
      after.deferred_interactions - before.deferred_interactions;
  result.weighted_projectile_total_energy_GeV =
      after.weighted_projectile_total_energy_GeV -
      before.weighted_projectile_total_energy_GeV;
  result.weighted_target_total_energy_GeV =
      after.weighted_target_total_energy_GeV -
      before.weighted_target_total_energy_GeV;
  result.weighted_effective_target_total_energy_GeV =
      after.weighted_effective_target_total_energy_GeV -
      before.weighted_effective_target_total_energy_GeV;
  result.weighted_target_isotope_correction_GeV =
      after.weighted_target_isotope_correction_GeV -
      before.weighted_target_isotope_correction_GeV;
  result.target_isotope_adjusted_interactions =
      after.target_isotope_adjusted_interactions -
      before.target_isotope_adjusted_interactions;
  result.weighted_secondary_total_energy_GeV =
      after.weighted_secondary_total_energy_GeV -
      before.weighted_secondary_total_energy_GeV;
  result.weighted_energy_residual_GeV =
      after.weighted_energy_residual_GeV -
      before.weighted_energy_residual_GeV;
  result.weighted_absolute_energy_residual_GeV =
      after.weighted_absolute_energy_residual_GeV -
      before.weighted_absolute_energy_residual_GeV;
  // The cumulative maximum cannot be subtracted.  It is retained as a
  // conservative per-run upper bound; validation jobs use one shower.
  result.maximum_absolute_residual_GeV =
      after.maximum_absolute_residual_GeV;
  result.weighted_isotope_corrected_energy_residual_GeV =
      after.weighted_isotope_corrected_energy_residual_GeV -
      before.weighted_isotope_corrected_energy_residual_GeV;
  result.weighted_absolute_isotope_corrected_energy_residual_GeV =
      after.weighted_absolute_isotope_corrected_energy_residual_GeV -
      before.weighted_absolute_isotope_corrected_energy_residual_GeV;
  result.maximum_absolute_isotope_corrected_residual_GeV =
      after.maximum_absolute_isotope_corrected_residual_GeV;
  result.negative_isotope_corrected_residual_interactions =
      after.negative_isotope_corrected_residual_interactions -
      before.negative_isotope_corrected_residual_interactions;
  result.maximum_negative_isotope_corrected_residual_GeV =
      after.maximum_negative_isotope_corrected_residual_GeV;
  result.spacelike_isotope_corrected_residual_interactions =
      after.spacelike_isotope_corrected_residual_interactions -
      before.spacelike_isotope_corrected_residual_interactions;
  result.maximum_isotope_corrected_spacelike_excess_GeV =
      after.maximum_isotope_corrected_spacelike_excess_GeV;
  result.minimum_isotope_corrected_residual_mass_squared_GeV2 =
      after.minimum_isotope_corrected_residual_mass_squared_GeV2;
  result.maximum_target_total_energy_GeV =
      after.maximum_target_total_energy_GeV;
  result.weighted_momentum_residual_norm_GeV =
      after.weighted_momentum_residual_norm_GeV -
      before.weighted_momentum_residual_norm_GeV;
  result.maximum_momentum_residual_norm_GeV =
      after.maximum_momentum_residual_norm_GeV;
  result.negative_energy_residual_interactions =
      after.negative_energy_residual_interactions -
      before.negative_energy_residual_interactions;
  result.spacelike_residual_interactions =
      after.spacelike_residual_interactions -
      before.spacelike_residual_interactions;
  result.maximum_negative_energy_residual_GeV =
      after.maximum_negative_energy_residual_GeV;
  result.maximum_negative_projectile_pdg =
      after.maximum_negative_projectile_pdg;
  result.maximum_negative_target_pdg =
      after.maximum_negative_target_pdg;
  result.maximum_negative_projectile_total_energy_GeV =
      after.maximum_negative_projectile_total_energy_GeV;
  result.maximum_negative_target_total_energy_GeV =
      after.maximum_negative_target_total_energy_GeV;
  result.maximum_negative_secondary_total_energy_GeV =
      after.maximum_negative_secondary_total_energy_GeV;
  result.maximum_negative_secondary_count =
      after.maximum_negative_secondary_count;
  result.maximum_negative_baryon_number_residual =
      after.maximum_negative_baryon_number_residual;
  result.maximum_negative_charge_number_residual =
      after.maximum_negative_charge_number_residual;
  result.maximum_negative_secondary_pdgs =
      after.maximum_negative_secondary_pdgs;
  result.maximum_spacelike_excess_GeV =
      after.maximum_spacelike_excess_GeV;
  result.minimum_residual_mass_squared_GeV2 =
      after.minimum_residual_mass_squared_GeV2;
  result.maximum_timelike_residual_mass_GeV =
      after.maximum_timelike_residual_mass_GeV;
  result.maximum_positive_residual_beta =
      after.maximum_positive_residual_beta;
  return result;
}

proposal::HadronicPhotonEnergyLedgerStatistics subtractStatistics(
    proposal::HadronicPhotonEnergyLedgerStatistics const& after,
    proposal::HadronicPhotonEnergyLedgerStatistics const& before) {
  proposal::HadronicPhotonEnergyLedgerStatistics result{};
  result.interactions = after.interactions - before.interactions;
  result.low_energy_interactions =
      after.low_energy_interactions - before.low_energy_interactions;
  result.high_energy_interactions =
      after.high_energy_interactions - before.high_energy_interactions;
  result.weighted_projectile_total_energy_GeV =
      after.weighted_projectile_total_energy_GeV -
      before.weighted_projectile_total_energy_GeV;
  result.weighted_target_total_energy_GeV =
      after.weighted_target_total_energy_GeV -
      before.weighted_target_total_energy_GeV;
  result.weighted_secondary_total_energy_GeV =
      after.weighted_secondary_total_energy_GeV -
      before.weighted_secondary_total_energy_GeV;
  result.weighted_energy_residual_GeV =
      after.weighted_energy_residual_GeV -
      before.weighted_energy_residual_GeV;
  result.weighted_absolute_energy_residual_GeV =
      after.weighted_absolute_energy_residual_GeV -
      before.weighted_absolute_energy_residual_GeV;
  result.maximum_absolute_residual_GeV =
      after.maximum_absolute_residual_GeV;
  result.maximum_target_total_energy_GeV =
      after.maximum_target_total_energy_GeV;
  result.weighted_momentum_residual_norm_GeV =
      after.weighted_momentum_residual_norm_GeV -
      before.weighted_momentum_residual_norm_GeV;
  result.maximum_momentum_residual_norm_GeV =
      after.maximum_momentum_residual_norm_GeV;
  result.negative_energy_residual_interactions =
      after.negative_energy_residual_interactions -
      before.negative_energy_residual_interactions;
  result.spacelike_residual_interactions =
      after.spacelike_residual_interactions -
      before.spacelike_residual_interactions;
  result.maximum_negative_energy_residual_GeV =
      after.maximum_negative_energy_residual_GeV;
  result.maximum_negative_projectile_pdg =
      after.maximum_negative_projectile_pdg;
  result.maximum_negative_target_pdg =
      after.maximum_negative_target_pdg;
  result.maximum_negative_projectile_total_energy_GeV =
      after.maximum_negative_projectile_total_energy_GeV;
  result.maximum_negative_target_total_energy_GeV =
      after.maximum_negative_target_total_energy_GeV;
  result.maximum_negative_secondary_total_energy_GeV =
      after.maximum_negative_secondary_total_energy_GeV;
  result.maximum_negative_secondary_count =
      after.maximum_negative_secondary_count;
  result.maximum_negative_baryon_number_residual =
      after.maximum_negative_baryon_number_residual;
  result.maximum_negative_charge_number_residual =
      after.maximum_negative_charge_number_residual;
  result.maximum_negative_secondary_pdgs =
      after.maximum_negative_secondary_pdgs;
  result.maximum_spacelike_excess_GeV =
      after.maximum_spacelike_excess_GeV;
  result.minimum_residual_mass_squared_GeV2 =
      after.minimum_residual_mass_squared_GeV2;
  result.maximum_timelike_residual_mass_GeV =
      after.maximum_timelike_residual_mass_GeV;
  result.maximum_positive_residual_beta =
      after.maximum_positive_residual_beta;
  return result;
}

ObservationPlaneStatistics subtractStatistics(
    ObservationPlaneStatistics const& after,
    ObservationPlaneStatistics const& before) {
  return {
      after.particles - before.particles,
      after.weighted_total_energy_GeV -
          before.weighted_total_energy_GeV};
}

std::vector<std::tuple<double, double, double>> read_antenna_positions(
    std::string const& filename) {
  std::vector<std::tuple<double, double, double>> antennas;
  std::ifstream input(filename);
  if (!input.is_open()) {
    CORSIKA_LOG_WARN("Cannot open antenna coordinate file: {}", filename);
    return antennas;
  }

  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') { continue; }
    std::stringstream fields(line);
    double north_m = 0.;
    double west_m = 0.;
    double up_m = 0.;
    if (fields >> north_m >> west_m >> up_m) {
      antennas.emplace_back(north_m, west_m, up_m);
    }
  }
  CORSIKA_LOG_INFO("Loaded {} antenna positions from {}", antennas.size(), filename);
  return antennas;
}

int main(int argc, char** argv) {

  // the main command line description
  CLI::App app{
      "Simulate particle cascades in homogeneous ice with CORSIKA 8 CUDA."};

  CORSIKA_LOG_INFO(
      "Please cite the following references when using CORSIKA 8:\n"
      " - \"Towards a Next Generation of CORSIKA: A Framework for the Simulation of "
      "Particle Cascades in Astroparticle Physics\", Comput. Softw. Big Sci. 3 (2019) "
      "2, https://doi.org/10.1007/s41781-018-0013-0\n"
      " - \"Simulating radio emission from particle cascades with CORSIKA 8\", "
      "Astropart. Phys. 166 (2025) 103072, "
      "https://doi.org/10.1016/j.astropartphys.2024.103072\n"
      " - \"CORSIKA 8 (icrc2025)\", Zenodo, "
      "https://doi.org/10.5281/zenodo.18195386");

  //////// Primary options ////////

  // some options that we want to fill in
  int A, Z, nevent = 0;
  std::vector<double> cli_energy_range;

  // the following section adds the options to the parser

  // we start by definining a sub-group for the primary ID
  auto opt_Z = app.add_option("-Z", Z, "Atomic number for primary")
                   ->check(CLI::Range(0, 26))
                   ->group("Primary");
  auto opt_A = app.add_option("-A", A, "Atomic mass number for primary")
                   ->needs(opt_Z)
                   ->check(CLI::Range(1, 58))
                   ->group("Primary");
  app.add_option("-p,--pdg",
                 "PDG code for primary (p=2212, gamma=22, e-=11, nu_e=12, mu-=13, "
                 "nu_mu=14, tau=15, nu_tau=16).")
      ->excludes(opt_A)
      ->excludes(opt_Z)
      ->group("Primary");
  app.add_option("-E,--energy", "Primary energy in GeV")->default_val(0);
  app.add_option("--energy_range", cli_energy_range,
                 "Low and high values that define the range of the primary energy in GeV")
      ->expected(2)
      ->check(CLI::PositiveNumber)
      ->group("Primary");
  app.add_option("--eslope", "Spectral index for sampling energies, dN/dE = E^eSlope")
      ->default_val(-1.0)
      ->group("Primary");
  app.add_option("-z,--zenith", "Primary zenith angle (deg)")
      ->default_val(0.)
      ->check(CLI::Range(0., 90.))
      ->group("Primary");
  app.add_option("-a,--azimuth", "Primary azimuth angle (deg)")
      ->default_val(0.)
      ->check(CLI::Range(0., 360.))
      ->group("Primary");

  //////// Config options ////////

  app.add_option("--emcut",
                 "Min. kin. energy of photons, electrons and "
                 "positrons in tracking (GeV)")
      ->default_val(0.5e-3)
      ->check(CLI::Range(0.000001, 1.e13))
      ->group("Config");
  app.add_option("--hadcut", "Min. kin. energy of hadrons in tracking (GeV)")
      ->default_val(0.3)
      ->check(CLI::Range(0.02, 1.e13))
      ->group("Config");
  app.add_option("--mucut", "Min. kin. energy of muons in tracking (GeV)")
      ->default_val(0.3)
      ->check(CLI::Range(0.000001, 1.e13))
      ->group("Config");
  app.add_option("--taucut", "Min. kin. energy of tau leptons in tracking (GeV)")
      ->default_val(0.3)
      ->check(CLI::Range(0.000001, 1.e13))
      ->group("Config");
  app.add_option("--max-deflection-angle",
                 "maximal deflection angle in tracking in radians")
      ->default_val(0.2)
      ->check(CLI::Range(1.e-8, 1.))
      ->group("Config");
  double ice_density_g_per_cm3{0.919};
  double ice_refractive_index{1.78};
  double profile_bin_g_per_cm2{10.};
  app.add_option(
         "--density-g-cm3", ice_density_g_per_cm3,
         "Homogeneous water-ice mass density used by CPU and CUDA transport")
      ->default_val(0.919)
      ->check(CLI::PositiveNumber)
      ->group("Config");
  app.add_option(
         "--refractive-index", ice_refractive_index,
         "Uniform radio refractive index of the ice")
      ->default_val(1.78)
      ->check(CLI::Range(1., 10.))
      ->group("Config");
  app.add_option(
         "--profile-bin-g-cm2", profile_bin_g_per_cm2,
         "Longitudinal-profile bin width in g/cm^2")
      ->default_val(10.)
      ->check(CLI::Range(0.01, 1.e4))
      ->group("Config");
  bool track_neutrinos = false;
  app.add_flag("--track-neutrinos", track_neutrinos, "switch on tracking of neutrinos")
      ->group("Config");

  std::string em_backend{"cuda"};
  int gpu_device{0};
  std::size_t gpu_min_batch{4096};
  double gpu_memory_fraction{0.70};
  std::filesystem::path gpu_table_cache;
  double gpu_table_tolerance{1.e-3};
  bool gpu_deterministic{true};
  bool gpu_detailed_stage_timing{false};
  bool gpu_full_step_records{false};
  bool gpu_resident_cross_species{true};
  std::string radio_backend{"cuda"};
  double gpu_radio_field_limit{1.};
  bool gpu_radio_track_diagnostics{false};
  std::uint32_t gpu_zhs_subtrack_refinement{1};
  double radio_sampling_rate_GHz{1.};
  double radio_window_duration_ns{400.};
  double radio_pretrigger_ns{10.};
  std::string cuda_replay_trace;
  std::filesystem::path cuda_replay_tape_out;
  int hadronic_plan_workers{4};
  double hadronic_plan_target_ms{5.};
  std::size_t hadronic_plan_max_batch{256};
  std::string hadronic_backend{"scalar"};
  int hadronic_workers{4};
  std::size_t hadronic_min_batch{64};
  double hadronic_target_batch_ms{5.};
  std::size_t hadronic_max_batch{256};
  double hadronic_initial_cost_ms{0.1};
  bool cpu_detailed_step_timing{false};
  std::filesystem::path
      hadronic_worker_executable;
  app.add_option("--em-backend", em_backend,
                 "Electromagnetic transport backend: proposal or cuda")
      ->default_val("cuda")
      ->check(CLI::IsMember({"proposal", "cuda"}))
      ->group("GPU EM");
  app.add_option("--gpu-device", gpu_device, "CUDA device index")
      ->check(CLI::NonNegativeNumber)
      ->group("GPU EM");
  app.add_option("--gpu-min-batch", gpu_min_batch,
                 "Minimum CUDA execution batch; smaller fronts take one "
                 "scalar expansion step")
      ->check(CLI::PositiveNumber)
      ->group("GPU EM");
  app.add_option("--gpu-memory-fraction", gpu_memory_fraction,
                 "Fraction of currently free device memory available to CUDA EM")
      ->check(CLI::Range(0.01, 1.0))
      ->group("GPU EM");
  app.add_option("--gpu-table-cache", gpu_table_cache,
                 "Versioned .c8emrt PROPOSAL table required by --em-backend cuda")
      ->group("GPU EM");
  app.add_option("--gpu-table-tolerance", gpu_table_tolerance,
                 "Maximum accepted relative table error")
      ->check(CLI::Range(1.e-8, 1.0))
      ->group("GPU EM");
  app.add_option("--gpu-deterministic", gpu_deterministic,
                 "Use history-keyed deterministic CUDA random numbers")
      ->group("GPU EM");
  app.add_flag(
         "--gpu-detailed-stage-timing",
         gpu_detailed_stage_timing,
         "Record per-stage CUDA event timings for the fused lepton "
         "pipeline (profiling only)")
      ->group("GPU EM");
  app.add_flag("--gpu-full-step-records", gpu_full_step_records,
               "Disable compact GPU profile projection and return full "
               "transport records (validation/debug only)")
      ->group("GPU EM");
  app.add_option(
         "--gpu-resident-cross-species",
         gpu_resident_cross_species,
         "Keep photon-to-lepton and lepton-to-photon secondaries in "
         "persistent device queues")
      ->group("GPU EM");
  app.add_option("--radio-backend", radio_backend,
                 "Radio projection backend for CUDA EM tracks: cpu or cuda")
      ->default_val("cuda")
      ->check(CLI::IsMember({"cpu", "cuda"}))
      ->group("Radio");
  app.add_option(
         "--gpu-radio-field-limit", gpu_radio_field_limit,
         "Checked fixed-point waveform range in V/m for deterministic "
         "CUDA CoREAS/ZHS accumulation")
      ->check(CLI::PositiveNumber)
      ->group("Radio");
  app.add_flag(
         "--gpu-radio-track-diagnostics",
         gpu_radio_track_diagnostics,
         "Collect scalar-compatible electron/positron track diagnostics "
         "on the GPU (validation only; adds reduction overhead)")
      ->group("Radio");
  app.add_option(
         "--gpu-zhs-subtrack-refinement",
         gpu_zhs_subtrack_refinement,
         "Validation-only integer refinement of each stock CUDA ZHS "
         "Fraunhofer subtrack; does not alter shower transport")
      ->default_val(1)
      ->check(CLI::Range(1, 64))
      ->group("Radio");
  app.add_option(
         "--radio-sampling-rate-ghz", radio_sampling_rate_GHz,
         "Time-domain radio sampling rate in GHz. Use at least 10 GHz "
         "(0.1 ns bins) for a 50--350 MHz CoREAS/ZHS comparison")
      ->check(CLI::Range(0.1, 1000.))
      ->group("Radio");
  app.add_option(
         "--radio-window-duration-ns", radio_window_duration_ns,
         "Time-domain radio observer window duration in ns")
      ->check(CLI::Range(1., 1.e6))
      ->group("Radio");
  app.add_option(
         "--radio-pretrigger-ns", radio_pretrigger_ns,
         "Start antenna-file radio windows this many ns before the "
         "geometrical direct-arrival time")
      ->check(CLI::Range(0., 1.e6))
      ->group("Radio");
  app.add_option(
         "--cuda-replay-trace", cuda_replay_trace,
         "Write a process-level CSV trace for scalar/CUDA replay comparison")
      ->group("GPU EM");
  app.add_option(
         "--cuda-replay-tape-out", cuda_replay_tape_out,
         "Record every scalar transport segment and the exact radio "
         "observer snapshot for offline CUDA decision replay")
      ->group("GPU EM");
  app.add_option(
         "--hadronic-plan-workers", hadronic_plan_workers,
         "Worker count used for the retrospective hadronic final-state "
         "batch oracle (does not enable parallel execution)")
      ->check(CLI::Range(1, 256))
      ->group("Hadronic scheduling");
  app.add_option(
         "--hadronic-plan-target-ms", hadronic_plan_target_ms,
         "Target measured final-state cost per retrospective hadronic batch")
      ->check(CLI::Range(1.e-6, 1.e6))
      ->group("Hadronic scheduling");
  app.add_option(
         "--hadronic-plan-max-batch", hadronic_plan_max_batch,
         "Maximum interactions per retrospective homogeneous hadronic batch")
      ->check(CLI::PositiveNumber)
      ->group("Hadronic scheduling");
  app.add_option(
         "--hadronic-backend", hadronic_backend,
         "Low-energy hadronic final-state backend: scalar or fluka-process")
      ->check(CLI::IsMember(
          {"scalar", "fluka-process"}))
      ->group("Hadronic scheduling");
  app.add_option(
         "--hadronic-workers", hadronic_workers,
         "Persistent FLUKA worker process count")
      ->check(CLI::Range(1, 256))
      ->group("Hadronic scheduling");
  app.add_option(
         "--hadronic-min-batch", hadronic_min_batch,
         "Minimum parked FLUKA vertices before the predicted-cost flush test")
      ->check(CLI::PositiveNumber)
      ->group("Hadronic scheduling");
  app.add_option(
         "--hadronic-target-batch-ms",
         hadronic_target_batch_ms,
         "Online estimated FLUKA time per homogeneous worker batch")
      ->check(CLI::Range(1.e-6, 1.e6))
      ->group("Hadronic scheduling");
  app.add_option(
         "--hadronic-max-batch", hadronic_max_batch,
         "Maximum FLUKA requests in one homogeneous process-worker batch")
      ->check(CLI::PositiveNumber)
      ->group("Hadronic scheduling");
  app.add_option(
         "--hadronic-initial-cost-ms",
         hadronic_initial_cost_ms,
         "Initial per-interaction cost before a class has worker timing samples")
      ->check(CLI::Range(1.e-6, 1.e6))
      ->group("Hadronic scheduling");
  app.add_option(
         "--hadronic-worker-executable",
         hadronic_worker_executable,
         "Path to fluka_batch_worker; empty selects the executable beside c8_ice_cascade")
      ->group("Hadronic scheduling");
  app.add_flag(
         "--cpu-detailed-step-timing",
         cpu_detailed_step_timing,
         "Record scalar cross-section/tracking/continuous/discrete phase times")
      ->group("Hadronic scheduling");

  //////// Misc options ////////

  app.add_option("--neutrino-interaction-type",
                 "charged (CC) or neutral current (NC) or both")
      ->default_val("both")
      ->check(CLI::IsMember({"neutral", "NC", "charged", "CC", "both"}))
      ->group("Misc.");
  app.add_option("--observation-level",
                 "Height above earth radius of the observation level (in m)")
      ->default_val(2680.444195)
      ->check(CLI::Range(-1.e3, 1.e5))
      ->group("Config");
  app.add_option("--injection-height",
                 "Height above earth radius of the injection point (in m)")
      ->default_val(112.75e3)
      ->check(CLI::Range(-1.e3, 1.e6))
      ->group("Config");
  app.add_option("--shower-core-x",
                 "X coordinate of the shower core position (in m)")
      ->default_val(0.)
      ->group("Config");
  app.add_option("--shower-core-y",
                 "Y coordinate of the shower core position (in m)")
      ->default_val(0.)
      ->group("Config");
  app.add_option("-N,--nevent", nevent, "The number of events/showers to run.")
      ->default_val(1)
      ->check(CLI::PositiveNumber)
      ->group("Library/Output");
  app.add_option("-f,--filename", "Filename for output library.")
      ->required()
      ->default_val("corsika_library")
      ->check(CLI::NonexistentPath)
      ->group("Library/Output");
  bool compressOutput = false;
  app.add_flag("--compress", compressOutput, "Compress the output directory to a tarball")
      ->group("Library/Output");
  app.add_option("-s,--seed", "The random number seed.")
      ->default_val(0)
      ->check(CLI::NonNegativeNumber)
      ->group("Misc.");
  bool force_interaction = false;
  app.add_flag("--force-interaction", force_interaction,
               "Force the location of the first interaction.")
      ->group("Misc.");
  bool force_decay = false;
  app.add_flag("--force-decay", force_decay, "Force the primary to immediately decay")
      ->group("Misc.");
  bool disable_interaction_hists = false;
  app.add_flag("--disable-interaction-histograms", disable_interaction_hists,
               "Store interaction histograms")
      ->group("Misc.");
  app.add_option("-v,--verbosity", "Verbosity level: warn, info, debug, trace.")
      ->default_val("info")
      ->check(CLI::IsMember({"warn", "info", "debug", "trace"}))
      ->group("Misc.");
  app.add_option("-M,--hadronModel", "High-energy hadronic interaction model")
      ->default_val("SIBYLL-2.3d")
      ->check(CLI::IsMember(
          {"SIBYLL-2.3d", "QGSJet-II.04", "QGSJet-III", "EPOS-LHC", "Pythia8"}))
      ->group("Misc.");
  app.add_option("-T,--hadronModelTransitionEnergy",
                 "Transition between high-/low-energy hadronic interaction "
                 "model in GeV")
      ->default_val(std::pow(10, 1.9)) // 79.4 GeV
      ->check(CLI::NonNegativeNumber)
      ->group("Misc.");

  //////// Thinning options ////////

  app.add_option("--emthin",
                 "fraction of primary energy at which thinning of EM particles starts")
      ->default_val(1.e-6)
      ->check(CLI::Range(0., 1.))
      ->group("Thinning");
  app.add_option("--max-weight",
                 "maximum weight for thinning of EM particles (0 to select Kobal's "
                 "optimum times 0.5)")
      ->default_val(0)
      ->check(CLI::NonNegativeNumber)
      ->group("Thinning");
  bool multithin = false;
  app.add_flag("--multithin", multithin, "keep thinned particles (with weight=0)")
      ->group("Thinning");
  app.add_option("--ring", "concentric ring of star shape pattern of observers")
      ->default_val(0)
      ->check(CLI::Range(0, 20))
      ->group("Radio");
  app.add_option("--antenna-file",
                 "Path to antenna positions in NWU coordinates (m)")
      ->default_val("antennas.txt")
      ->group("Radio");
  // parse the command line options into the variables
  CLI11_PARSE(app, argc, argv);

  if (!cuda_replay_trace.empty()) {
    gpu_full_step_records = true;
    validation::CudaReplayTrace::instance().open(
        cuda_replay_trace,
        em_backend == "cuda" ? "cuda" : "refactor_proposal");
  }
  if (!cuda_replay_tape_out.empty() && em_backend != "proposal") {
    CORSIKA_LOG_CRITICAL(
        "--cuda-replay-tape-out records the scalar decision oracle and "
        "therefore requires --em-backend proposal");
    return EXIT_FAILURE;
  }

  if (em_backend == "cuda") {
#ifndef CORSIKA8_WITH_CUDA_EM
    CORSIKA_LOG_CRITICAL(
        "--em-backend cuda was requested, but this c8_ice_cascade binary was "
        "built without CORSIKA_ENABLE_CUDA");
    return EXIT_FAILURE;
#else
    if (gpu_table_cache.empty()) {
      CORSIKA_LOG_CRITICAL("--em-backend cuda requires --gpu-table-cache");
      return EXIT_FAILURE;
    }
    if (!std::filesystem::is_regular_file(gpu_table_cache)) {
      CORSIKA_LOG_CRITICAL("CUDA EM table cache is not a regular file: {}",
                           gpu_table_cache.string());
      return EXIT_FAILURE;
    }
#endif
  }
  if (radio_backend == "cuda") {
    if (em_backend != "cuda") {
      CORSIKA_LOG_CRITICAL(
          "--radio-backend cuda requires --em-backend cuda");
      return EXIT_FAILURE;
    }
#ifndef CORSIKA8_WITH_CUDA_EM
    CORSIKA_LOG_CRITICAL(
        "--radio-backend cuda was requested, but CUDA support is unavailable");
    return EXIT_FAILURE;
#endif
  }
  if (hadronic_backend == "fluka-process") {
    if (em_backend != "cuda") {
      CORSIKA_LOG_CRITICAL(
          "--hadronic-backend fluka-process currently requires "
          "--em-backend cuda and HybridCascade");
      return EXIT_FAILURE;
    }
#ifndef WITH_FLUKA
    CORSIKA_LOG_CRITICAL(
        "--hadronic-backend fluka-process requires a WITH_FLUKA build");
    return EXIT_FAILURE;
#endif
#ifndef CORSIKA8_WITH_CUDA_EM
    CORSIKA_LOG_CRITICAL(
        "--hadronic-backend fluka-process requires the CUDA HybridCascade build");
    return EXIT_FAILURE;
#endif
    if (hadronic_worker_executable.empty()) {
      hadronic_worker_executable =
          std::filesystem::absolute(argv[0])
              .parent_path() /
          "fluka_batch_worker";
    }
    if (!std::filesystem::is_regular_file(
            hadronic_worker_executable)) {
      CORSIKA_LOG_CRITICAL(
          "FLUKA worker executable is not a regular file: {}",
          hadronic_worker_executable.string());
      return EXIT_FAILURE;
    }
  }

  if (app.count("--verbosity")) {
    auto const loglevel = app["--verbosity"]->as<std::string>();
    if (loglevel == "warn") {
      logging::set_level(logging::level::warn);
    } else if (loglevel == "info") {
      logging::set_level(logging::level::info);
    } else if (loglevel == "debug") {
      logging::set_level(logging::level::debug);
    } else if (loglevel == "trace") {
#ifndef _C8_DEBUG_
      CORSIKA_LOG_ERROR("trace log level requires a Debug build.");
      return 1;
#endif
      logging::set_level(logging::level::trace);
    }
  }

  // check that we got either PDG or A/Z
  // this can be done with option_groups but the ordering
  // gets all messed up
  if (app.count("--pdg") == 0) {
    if ((app.count("-A") == 0) || (app.count("-Z") == 0)) {
      CORSIKA_LOG_ERROR("If --pdg is not provided, then both -A and -Z are required.");
      return 1;
    }
  }

  // initialize random number sequence(s)
  auto seed = registerRandomStreams(app["--seed"]->as<long>());

#ifdef CORSIKA8_WITH_CUDA_EM
  // fork/exec the process-isolated FLUKA workers before the parent touches the
  // CUDA runtime. Forking after CUDA context creation is unsupported.
  std::unique_ptr<HadronicProcessPool>
      hadronic_process_pool;
  if (hadronic_backend == "fluka-process") {
    try {
      hadronic_process_pool =
          std::make_unique<HadronicProcessPool>(
              hadronic_worker_executable,
              static_cast<std::size_t>(
                  hadronic_workers));
    } catch (std::exception const& error) {
      CORSIKA_LOG_CRITICAL(
          "Could not start process-isolated FLUKA workers: {}",
          error.what());
      return EXIT_FAILURE;
    }
  }
#endif

  /* === START: SETUP ENVIRONMENT AND ROOT COORDINATE SYSTEM === */
  EnvType env;
  CoordinateSystemPtr const& rootCS = env.getCoordinateSystem();
  Point const center{rootCS, 0_m, 0_m, 0_m};
  Point const surface_{rootCS, 0_m, 0_m, constants::EarthRadius::Mean};

  // The validation medium is homogeneous water ice with no geomagnetic field.
  // Keeping this fixed avoids an air-specific IGRF/data dependency and makes
  // the CPU and CUDA material snapshots identical by construction.
  MagneticFieldVector const transport_field(rootCS, 0_T, 0_T, 0_T);
  auto const earth_radius_m =
      constants::EarthRadius::Mean / 1_m;
  auto const configured_observation_altitude_m =
      app["--observation-level"]->as<double>();
  auto const configured_injection_altitude_m =
      app["--injection-height"]->as<double>();
  auto const ice_inner_radius_m =
      earth_radius_m - 5000.;
  auto const ice_outer_radius_m =
      earth_radius_m +
      std::max(configured_observation_altitude_m,
               configured_injection_altitude_m) +
      5000.;

  if (!(configured_injection_altitude_m >
        configured_observation_altitude_m)) {
    CORSIKA_LOG_CRITICAL(
        "c8_ice_cascade requires --injection-height above "
        "--observation-level");
    return EXIT_FAILURE;
  }
  NuclearComposition const ice_composition(
      {Code::Hydrogen, Code::Oxygen},
      {2. / 3., 1. / 3.});
  auto ice = EnvType::createNode<Sphere>(center, ice_outer_radius_m * 1_m);
  ice->setModelProperties<MyHomogeneousEnv<EnvironmentInterface>>(
      ice_refractive_index, Medium::WaterIce, transport_field,
      ice_density_g_per_cm3 * 1_g / cube(1_cm), ice_composition);
  env.getUniverse()->addChild(std::move(ice));
  CORSIKA_LOG_INFO(
      "Using dedicated homogeneous ice: rho={} g/cm3, n={}, B=0, "
      "inner GPU radius={} m, outer radius={} m",
      ice_density_g_per_cm3, ice_refractive_index, ice_inner_radius_m,
      ice_outer_radius_m);

  /* === END: SETUP ENVIRONMENT AND ROOT COORDINATE SYSTEM === */

  /* === START: CONSTRUCT PRIMARY PARTICLE === */

  // parse the primary ID as a PDG or A/Z code
  Code beamCode;

  // check if we want to use a PDG code instead
  if (app.count("--pdg") > 0) {
    beamCode = convert_from_PDG(PDGCode(app["--pdg"]->as<int>()));
  } else {
    // check manually for proton and neutrons
    if ((A == 1) && (Z == 1))
      beamCode = Code::Proton;
    else if ((A == 1) && (Z == 0))
      beamCode = Code::Neutron;
    else
      beamCode = get_nucleus_code(A, Z);
  }

  HEPEnergyType eMin = 0_GeV;
  HEPEnergyType eMax = 0_GeV;
  // check the particle energy parameters
  if (app["--energy"]->as<double>() > 0.0) {
    eMin = app["--energy"]->as<double>() * 1_GeV;
    eMax = app["--energy"]->as<double>() * 1_GeV;
  } else if (cli_energy_range.size()) {
    if (cli_energy_range[0] > cli_energy_range[1]) {
      CORSIKA_LOG_WARN(
          "Energy range lower bound is greater than upper bound. swapping...");
      eMin = cli_energy_range[1] * 1_GeV;
      eMax = cli_energy_range[0] * 1_GeV;
    } else {
      eMin = cli_energy_range[0] * 1_GeV;
      eMax = cli_energy_range[1] * 1_GeV;
    }
  } else {
    CORSIKA_LOG_CRITICAL(
        "Must set either the (--energy) flag or the (--energy_range) flag to "
        "positive value(s)");
    return 0;
  }

  // direction of the shower in (theta, phi) space
  auto const thetaRad = app["--zenith"]->as<double>() / 180. * M_PI;
  auto const phiRad = app["--azimuth"]->as<double>() / 180. * M_PI;

  auto const [nx, ny, nz] = std::make_tuple(sin(thetaRad) * cos(phiRad),
                                            sin(thetaRad) * sin(phiRad), -cos(thetaRad));
  auto propDir = DirectionVector(rootCS, {nx, ny, nz});
  /* === END: CONSTRUCT PRIMARY PARTICLE === */

  /* === START: CONSTRUCT GEOMETRY === */
  auto const observationHeight =
      app["--observation-level"]->as<double>() * 1_m + constants::EarthRadius::Mean;
  auto const injectionHeight =
      app["--injection-height"]->as<double>() * 1_m + constants::EarthRadius::Mean;
  auto const t = -observationHeight * cos(thetaRad) +
                 sqrt(-static_pow<2>(sin(thetaRad) * observationHeight) +
                      static_pow<2>(injectionHeight));
  auto const showerCoreX =
      app["--shower-core-x"]->as<double>() * 1_m;
  auto const showerCoreY =
      app["--shower-core-y"]->as<double>() * 1_m;
  Point const showerCore{
      rootCS, showerCoreX, showerCoreY, observationHeight};
  Point const injectionPos =
      showerCore + DirectionVector{rootCS,
                                   {-sin(thetaRad) * cos(phiRad),
                                    -sin(thetaRad) * sin(phiRad), cos(thetaRad)}} *
                       t;

  // we make the axis much longer than the inj-core distance since the
  // profile will go beyond the core, depending on zenith angle
  ShowerAxis const showerAxis{injectionPos, (showerCore - injectionPos) * 1.2, env};
  auto const dX = profile_bin_g_per_cm2 * 1_g / square(1_cm);
  /* === END: CONSTRUCT GEOMETRY === */

  std::stringstream args;
  for (int i = 0; i < argc; ++i) { args << argv[i] << " "; }

#ifdef CORSIKA8_WITH_CUDA_EM
  std::optional<gpu::em::tables::RateTableSet>
      loaded_gpu_table;
  std::unique_ptr<gpu::em::GpuEmRunOutput>
      gpu_run_output;
  std::unique_ptr<gpu::em::CudaEmBackend>
      reusable_gpu_em_backend;
  if (em_backend == "cuda") {
    using namespace corsika::gpu::em::tables;
    loaded_gpu_table.emplace(
        readRateTable(gpu_table_cache));
    if (loaded_gpu_table->metadata.generator_version !=
        TableGeneratorContractVersion) {
      throw std::runtime_error(
          "CUDA EM table generator contract mismatch: table=" +
          loaded_gpu_table->metadata.generator_version +
          ", required=" + TableGeneratorContractVersion +
          "; regenerate the table with gpu_em_table_prepare");
    }
    constexpr auto expected_medium_name = "water_ice";
    if (loaded_gpu_table->metadata.medium_name != expected_medium_name) {
      throw std::runtime_error(
          "c8_ice_cascade requires a water_ice CUDA EM table: table=" +
          loaded_gpu_table->metadata.medium_name +
          ", expected=" + expected_medium_name);
    }
    MediumConfig expected_ice_config;
    auto const& ice_medium = mediumData(Medium::WaterIce);
    expected_ice_config.name = ice_medium.getName();
    expected_ice_config.mean_excitation_energy_eV = ice_medium.getIeff();
    expected_ice_config.density_correction_C = -ice_medium.getCbar();
    expected_ice_config.density_correction_a = ice_medium.getAA();
    expected_ice_config.density_correction_m = ice_medium.getSK();
    expected_ice_config.density_correction_x0 = ice_medium.getX0();
    expected_ice_config.density_correction_x1 = ice_medium.getX1();
    expected_ice_config.density_correction_delta0 = ice_medium.getDlt0();
    expected_ice_config.reference_mass_density_g_per_cm3 =
        ice_medium.getCorrectedDensity();
    expected_ice_config.components = {
        {std::string(get_name(Code::Hydrogen)),
         static_cast<std::int32_t>(get_PDG(Code::Hydrogen)),
         static_cast<double>(get_nucleus_Z(Code::Hydrogen)),
         static_cast<double>(get_nucleus_A(Code::Hydrogen)), 2. / 3.},
        {std::string(get_name(Code::Oxygen)),
         static_cast<std::int32_t>(get_PDG(Code::Oxygen)),
         static_cast<double>(get_nucleus_Z(Code::Oxygen)),
         static_cast<double>(get_nucleus_A(Code::Oxygen)), 1. / 3.}};
    expected_ice_config = normalizeMediumConfig(std::move(expected_ice_config));
    auto const expected_proposal_medium =
        makeProposalMedium(expected_ice_config);
    auto const expected_components = makeRateTableMediumComponents(
        expected_ice_config, expected_proposal_medium);
    auto const same_component_contract = [&]() {
      auto const& actual = loaded_gpu_table->metadata.components;
      if (actual.size() != expected_components.size()) {
        return false;
      }
      for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index].corsika_pid != expected_components[index].corsika_pid ||
            actual[index].proposal_hash !=
                expected_components[index].proposal_hash ||
            actual[index].name != expected_components[index].name ||
            actual[index].number_fraction !=
                expected_components[index].number_fraction) {
          return false;
        }
      }
      return true;
    };
    if (loaded_gpu_table->metadata.proposal_medium_hash !=
            static_cast<std::uint64_t>(expected_proposal_medium.GetHash()) ||
        !same_component_contract()) {
      throw std::runtime_error(
          "CUDA ice table PROPOSAL medium/component identity differs from "
          "the scalar CORSIKA WaterIce contract; regenerate the table from "
          "configs/media/water_ice_0p919.yaml");
    }
    auto const density_tolerance =
        1.e-12 * std::max(1., ice_medium.getCorrectedDensity());
    auto const brems_density = loaded_gpu_table->metadata
                                    .brems_lpm
                                    .baseline_mass_density_g_per_cm3;
    auto const pair_density = loaded_gpu_table->metadata
                                  .photon_pair_lpm
                                  .baseline_mass_density_g_per_cm3;
    if (std::abs(brems_density - ice_medium.getCorrectedDensity()) >
            density_tolerance ||
        std::abs(pair_density - ice_medium.getCorrectedDensity()) >
            density_tolerance) {
      throw std::runtime_error(
          "CUDA ice table LPM reference density does not match the scalar "
          "CORSIKA WaterIce baseline");
    }
    for (auto const& particle : loaded_gpu_table->particles) {
      for (auto const& column : particle.columns) {
        if (column.inverse_cdf.reference_mode ==
            SelectedLossCpuFallbackReferenceMode) {
          throw std::runtime_error(
              "CUDA EM table contains a runtime selected-loss CPU fallback; "
              "regenerate it with table contract " +
              std::string(TableGeneratorContractVersion));
        }
      }
    }

    cudaDeviceProp properties{};
    auto const property_status =
        cudaGetDeviceProperties(
            &properties, gpu_device);
    if (property_status != cudaSuccess) {
      throw std::runtime_error(
          std::string(
              "cannot query requested CUDA device: ") +
          cudaGetErrorString(property_status));
    }
    int driver_version = 0;
    int runtime_version = 0;
    auto const driver_status =
        cudaDriverGetVersion(&driver_version);
    auto const runtime_status =
        cudaRuntimeGetVersion(&runtime_version);
    if (driver_status != cudaSuccess ||
        runtime_status != cudaSuccess) {
      throw std::runtime_error(
          "cannot query CUDA driver/runtime version");
    }
    auto version_string = [](int version) {
      return std::to_string(version / 1000) + "." +
             std::to_string((version % 1000) / 10);
    };

    auto const& table = *loaded_gpu_table;
    YAML::Node configuration;
    configuration["backend"] =
        "CORSIKA8GpuEm";
    configuration["backend_version"] =
        "cuda-em-v3-configurable-magnetic-step";
    configuration["device"]["index"] =
        gpu_device;
    configuration["device"]["name"] =
        properties.name;
    configuration["device"]["compute_capability"] =
        std::to_string(properties.major) + "." +
        std::to_string(properties.minor);
    configuration["device"]["total_memory_bytes"] =
        static_cast<std::uint64_t>(
            properties.totalGlobalMem);
    configuration["cuda"]["driver_version"] =
        version_string(driver_version);
    configuration["cuda"]["runtime_version"] =
        version_string(runtime_version);
    configuration["deterministic"] =
        gpu_deterministic;
    configuration["detailed_stage_timing"] =
        gpu_detailed_stage_timing;
    configuration["radio_backend"] =
        radio_backend;
    configuration["radio_fixed_point_field_limit_V_per_m"] =
        gpu_radio_field_limit;
    configuration["radio_track_diagnostics"] =
        gpu_radio_track_diagnostics;
    configuration["zhs_subtrack_refinement"] =
        gpu_zhs_subtrack_refinement;
    configuration["minimum_batch_size"] =
        static_cast<std::uint64_t>(
            gpu_min_batch);
    configuration["memory_fraction"] =
        gpu_memory_fraction;
    configuration["accepted_table_tolerance"] =
        gpu_table_tolerance;
    configuration["environment"]["kind"] =
        "homogeneous-ice";
    configuration["environment"]["mass_density_g_per_cm3"] =
        ice_density_g_per_cm3;
    configuration["environment"]["refractive_index"] =
        ice_refractive_index;
    configuration["environment"]["magnetic_field_model"] =
        "disabled";
    configuration["environment"]["magnetic_field_T"]["x"] =
        transport_field.getX(rootCS) / 1_T;
    configuration["environment"]["magnetic_field_T"]["y"] =
        transport_field.getY(rootCS) / 1_T;
    configuration["environment"]["magnetic_field_T"]["z"] =
        transport_field.getZ(rootCS) / 1_T;
    configuration["environment"]["maximum_magnetic_deflection_rad"] =
        app["--max-deflection-angle"]->as<double>();
    configuration["environment"]["observation_geometry"] =
        "plane";
    configuration["environment"]["observation_plane_point_m"]["x"] =
        showerCoreX / 1_m;
    configuration["environment"]["observation_plane_point_m"]["y"] =
        showerCoreY / 1_m;
    configuration["environment"]["observation_plane_point_m"]["z"] =
        observationHeight / 1_m;
    configuration["environment"]["observation_plane_normal"]["x"] = 0.;
    configuration["environment"]["observation_plane_normal"]["y"] = 0.;
    configuration["environment"]["observation_plane_normal"]["z"] = 1.;
    configuration["environment"]["antenna_file"] =
        app["--antenna-file"]->as<std::string>();
    configuration["table"]["path"] =
        gpu_table_cache.string();
    configuration["table"]["format_version"] =
        RateTableFormatVersion;
    configuration["table"]["sha256"] =
        toHex(table.content_hash);
    configuration["table"]["proposal_version"] =
        table.metadata.proposal_version;
    configuration["table"]["generator_version"] =
        table.metadata.generator_version;
    configuration["table"]["medium"] =
        table.metadata.medium_name;
    configuration["table"]["stochastic_cut_MeV"] =
        table.metadata.energy_cut_MeV;
    configuration["table"]["energy_min_MeV"] =
        table.metadata.energy_min_MeV;
    configuration["table"]["energy_max_MeV"] =
        table.metadata.energy_max_MeV;
    configuration["table"]["measured_rate_error"] =
        table.metadata.measured_max_relative_error;
    configuration["table"]["measured_inverse_cdf_error"] =
        table.metadata.measured_max_loss_relative_error;
    gpu_run_output =
        std::make_unique<
            gpu::em::GpuEmRunOutput>(
            std::move(configuration));
  }
#endif

  // create the output manager that we then register outputs with
  OutputManager output(app["--filename"]->as<std::string>(), seed, args.str(),
                       compressOutput);

  std::unique_ptr<EnergyLedgerRunOutput> scalar_energy_ledger_output;
  if (em_backend == "proposal") {
    YAML::Node configuration;
    configuration["backend"] = "proposal_cpu";
    configuration["definition"] =
        "whole-event total-energy ledger including medium target rest energy";
    configuration["acceptance_tolerance"] = 1.e-4;
    scalar_energy_ledger_output =
        std::make_unique<EnergyLedgerRunOutput>(std::move(configuration));
    output.add("energy_ledger", *scalar_energy_ledger_output);
  }

#ifdef CORSIKA8_WITH_CUDA_EM
  if (gpu_run_output) {
    output.add("gpu_em", *gpu_run_output);
  }
#endif

  // register energy losses as output
  EnergyLossWriter dEdX{showerAxis, dX};
  output.add("energyloss", dEdX);

  DynamicInteractionProcess<StackType> heModel;

  auto const all_elements = corsika::get_all_elements_in_universe(env);
  // have SIBYLL always for PROPOSAL photo-hadronic interactions
  auto sibyll = std::make_shared<corsika::sibyll::Interaction>(
      all_elements, corsika::setup::C7trackedParticles);

  if (auto const modelStr = app["--hadronModel"]->as<std::string>();
      modelStr == "SIBYLL-2.3d") {
    heModel = DynamicInteractionProcess<StackType>{sibyll};
  } else if (modelStr == "QGSJet-II.04") {
    heModel = DynamicInteractionProcess<StackType>{
        std::make_shared<corsika::qgsjetII::Interaction>()};
  } else if (modelStr == "QGSJet-III") {
    heModel = DynamicInteractionProcess<StackType>{
        std::make_shared<corsika::qgsjetIII::Interaction>()};
  } else if (modelStr == "EPOS-LHC") {
    heModel = DynamicInteractionProcess<StackType>{
        std::make_shared<corsika::epos::Interaction>(corsika::setup::C7trackedParticles)};
  } else if (modelStr == "Pythia8") {
    heModel = DynamicInteractionProcess<StackType>{
        std::make_shared<corsika::pythia8::Interaction>(
            corsika::setup::C7trackedParticles)};
  } else {
    CORSIKA_LOG_CRITICAL("invalid choice \"{}\"; also check argument parser", modelStr);
    return EXIT_FAILURE;
  }

  InteractionCounter heCounted{heModel};

  corsika::pythia8::Decay decayPythia;
  // tau decay via TAUOLA (hard coded to left handed)
  corsika::tauola::Decay decayTauola(corsika::tauola::Helicity::LeftHanded);

  struct IsTauSwitch {
    bool operator()(const Particle& p) const {
      return (p.getPID() == Code::TauMinus || p.getPID() == Code::TauPlus);
    }
  };

  auto decaySequence = make_select(IsTauSwitch(), decayTauola, decayPythia);

  // neutrino interactions with pythia (options are: NC, CC)
  bool NC = false;
  bool CC = false;
  if (auto const nuIntStr = app["--neutrino-interaction-type"]->as<std::string>();
      nuIntStr == "neutral" || nuIntStr == "NC") {
    NC = true;
    CC = false;
  } else if (nuIntStr == "charged" || nuIntStr == "CC") {
    NC = false;
    CC = true;
  } else if (nuIntStr == "both") {
    NC = true;
    CC = true;
  }
  corsika::pythia8::NeutrinoInteraction neutrinoPrimaryPythia(
      corsika::setup::C7trackedParticles, NC, CC);

  // hadronic photon interactions in resonance region
  corsika::sophia::InteractionModel sophia;
  // PROPOSAL's real-photon cross section opens slightly below SOPHIA's hard
  // lower limit. Preserve those selected vertices with an explicitly counted,
  // exact two-body gamma+N->pi0+N threshold final state.
  corsika::proposal::ThresholdPhotoproductionModel
      photoHadronicThresholdFallback;
  corsika::proposal::HadronicInteractionModelFallback
      photoHadronicLowEnergy{
          sophia, photoHadronicThresholdFallback};
  // Preserve SIBYLL for supported rho0-nucleus final states, but never discard
  // a PROPOSAL photoproduction vertex merely because SIBYLL cannot represent
  // atmospheric argon. QGSJet-II covers that target and every use is counted.
  corsika::proposal::LazyHadronicInteractionModel<
      corsika::qgsjetII::InteractionModel>
      photoHadronicQgsjetFallback;
  corsika::proposal::HadronicInteractionModelFallback
      photoHadronicHighEnergy{
          sibyll->getHadronInteractionModel(),
          photoHadronicQgsjetFallback};

  HEPEnergyType const emcut = 1_GeV * app["--emcut"]->as<double>();
  HEPEnergyType const hadcut = 1_GeV * app["--hadcut"]->as<double>();
  HEPEnergyType const mucut = 1_GeV * app["--mucut"]->as<double>();
  HEPEnergyType const taucut = 1_GeV * app["--taucut"]->as<double>();
  ParticleCut<SubWriter<decltype(dEdX)>> cut(emcut, emcut, hadcut, mucut, taucut,
                                             !track_neutrinos, dEdX);

  // tell proposal that we are interested in all energy losses above the particle cut
  auto const prod_threshold = std::min({emcut, hadcut, mucut, taucut});
  set_energy_production_threshold(Code::Electron, prod_threshold);
  set_energy_production_threshold(Code::Positron, prod_threshold);
  set_energy_production_threshold(Code::Photon, prod_threshold);
  set_energy_production_threshold(Code::MuMinus, prod_threshold);
  set_energy_production_threshold(Code::MuPlus, prod_threshold);
  set_energy_production_threshold(Code::TauMinus, prod_threshold);
  set_energy_production_threshold(Code::TauPlus, prod_threshold);

  // energy threshold for high energy hadronic model. Affects LE/HE switch for
  // hadron interactions and the hadronic photon model in proposal
  HEPEnergyType const heHadronModelThreshold =
      1_GeV * app["--hadronModelTransitionEnergy"]->as<double>();

  corsika::proposal::Interaction emCascade(
      env, photoHadronicLowEnergy,
      photoHadronicHighEnergy,
      heHadronModelThreshold);

  // use BetheBlochPDG for hadronic continuous losses, and proposal otherwise
  corsika::proposal::ContinuousProcess<SubWriter<decltype(dEdX)>> emContinuousProposal(
      env, dEdX);
  BetheBlochPDG<SubWriter<decltype(dEdX)>> emContinuousBethe{dEdX};
  struct EMHadronSwitch {
    EMHadronSwitch() = default;
    bool operator()(const Particle& p) const { return is_hadron(p.getPID()); }
  };
  auto emContinuous =
      make_select(EMHadronSwitch(), emContinuousBethe, emContinuousProposal);

  LongitudinalWriter profile{showerAxis, dX};
  output.add("profile", profile);
  LongitudinalProfile<SubWriter<decltype(profile)>> longprof{profile};

  ProductionWriter prod_profile{showerAxis, dX};
  output.add("production_profile", prod_profile);
  ProductionProfile<SubWriter<decltype(prod_profile)>> prodprof{prod_profile};

// for ICRC2023
#ifdef WITH_FLUKA
  corsika::fluka::Interaction leIntModel{all_elements};
#ifdef CORSIKA8_WITH_CUDA_EM
  if (hadronic_process_pool) {
    // FLUKA's fpenab_ installs a SIGALRM handler that performs fopen/fwrite
    // every 60 seconds.  Those operations are not async-signal-safe: when the
    // CUDA parent has helper threads, the handler can interrupt a libc stdio
    // critical section and wait forever on the lock held by that same thread.
    // The process-isolated FLUKA workers were exec'ed above and retain their
    // own native timer.  The parent only supplies rate/classification state,
    // so cancel its redundant timer before CUDA creates more helper threads.
    if (::signal(SIGALRM, SIG_IGN) == SIG_ERR) {
      throw std::runtime_error(
          "could not disable the parent FLUKA SIGALRM timer");
    }
    ::alarm(0);
    CORSIKA_LOG_INFO(
        "Disabled the parent FLUKA SIGALRM timer; process-isolated FLUKA "
        "workers retain their native timers");
  }
#endif
#else
  corsika::urqmd::UrQMD leIntModel{};
#endif
  InteractionCounter leIntCounted{leIntModel};

  // assemble all processes into an ordered process list
  struct EnergySwitch {
    HEPEnergyType cutE_;
    EnergySwitch(HEPEnergyType cutE)
        : cutE_(cutE) {}
    bool operator()(const Particle& p) const { return (p.getEnergyNN() < cutE_); }
  };
  auto hadronSequence =
      make_select(EnergySwitch(heHadronModelThreshold), leIntCounted, heCounted);

  // observation plane
  Plane const obsPlane(showerCore, DirectionVector(rootCS, {0., 0., 1.}));
  ObservationPlane<TrackingType, ParticleWriterParquet> observationLevel{
      obsPlane, DirectionVector(rootCS, {1., 0., 0.}),
      true,   // plane should "absorb" particles
      false}; // do not print z-coordinate
  // register ground particle output
  output.add("particles", observationLevel);

  PrimaryWriter<TrackingType, ParticleWriterParquet> primaryWriter(observationLevel);
  output.add("primary", primaryWriter);

  int ring_number{app["--ring"]->as<int>()};
  auto const radius_{ring_number * 25_m};
  const int rr_ = static_cast<int>(radius_ / 1_m);
  auto const antenna_positions =
      read_antenna_positions(app["--antenna-file"]->as<std::string>());

  // Radio observers and relevant information
  // the observer time variables
  const TimeType duration_{radio_window_duration_ns * 1_ns};
  const InverseTimeType sampleRate_{radio_sampling_rate_GHz * 1_GHz};

  // the observer collection for CoREAS and ZHS
  ObserverCollection<TimeDomainObserver> detectorCoREAS;
  ObserverCollection<TimeDomainObserver> detectorZHS;

  auto const showerCoreX_{showerCore.getCoordinates().getX()};
  auto const showerCoreY_{showerCore.getCoordinates().getY()};
  auto const centerX_{center.getCoordinates().getX()};
  auto const centerY_{center.getCoordinates().getY()};
  auto const injectionPosX_{injectionPos.getCoordinates().getX()};
  auto const injectionPosY_{injectionPos.getCoordinates().getY()};
  auto const injectionPosZ_{injectionPos.getCoordinates().getZ()};
  auto const triggerpoint_{Point(rootCS, injectionPosX_, injectionPosY_, injectionPosZ_)};
  double const radio_trigger_refractive_index = ice_refractive_index;

  if (!antenna_positions.empty()) {
    for (std::size_t index = 0; index < antenna_positions.size(); ++index) {
      auto const [north_m, west_m, up_m] = antenna_positions[index];
      auto const point{Point(rootCS, centerX_ + north_m * 1_m,
                             centerY_ + west_m * 1_m,
                             constants::EarthRadius::Mean + up_m * 1_m)};
      auto const trigger_time =
          radio_trigger_refractive_index *
              (triggerpoint_ - point).getNorm() / constants::c -
          radio_pretrigger_ns * 1_ns;

      TimeDomainObserver coreas_observer(
          "CoREAS_Antenna_" + std::to_string(index), point, rootCS,
          trigger_time, duration_, sampleRate_, trigger_time);
      detectorCoREAS.addObserver(coreas_observer);

      TimeDomainObserver zhs_observer(
          "ZHS_Antenna_" + std::to_string(index), point, rootCS,
          trigger_time, duration_, sampleRate_, trigger_time);
      detectorZHS.addObserver(zhs_observer);
    }
  } else if (ring_number != 0) {
    // setup CoREAS observers - use the for loop for star shape pattern
    for (auto phi_1 = 0; phi_1 <= 315; phi_1 += 45) {
      auto phiRad_1 = phi_1 / 180. * M_PI;
      auto const point_1{Point(rootCS, showerCoreX_ + radius_ * cos(phiRad_1),
                               showerCoreY_ + radius_ * sin(phiRad_1),
                               constants::EarthRadius::Mean)};
      std::cout << "Observer point CoREAS: " << point_1 << std::endl;
      auto triggertime_1{
          radio_trigger_refractive_index *
          (triggerpoint_ - point_1).getNorm() / constants::c};
      std::string name_1 = "CoREAS_R=" + std::to_string(rr_) +
                           "_m--Phi=" + std::to_string(phi_1) + "degrees";
      TimeDomainObserver observer_1(name_1, point_1, rootCS, triggertime_1, duration_,
                                    sampleRate_, triggertime_1);
      detectorCoREAS.addObserver(observer_1);
    }

    // setup ZHS observers - use the for loop for star shape pattern
    for (auto phi_ = 0; phi_ <= 315; phi_ += 45) {
      auto phiRad_ = phi_ / 180. * M_PI;
      auto const point_{Point(rootCS, showerCoreX_ + radius_ * cos(phiRad_),
                              showerCoreY_ + radius_ * sin(phiRad_),
                              constants::EarthRadius::Mean)};
      std::cout << "Observer point ZHS: " << point_ << std::endl;
      auto triggertime_{
          radio_trigger_refractive_index *
          (triggerpoint_ - point_).getNorm() / constants::c};
      std::string name_ =
          "ZHS_R=" + std::to_string(rr_) + "_m--Phi=" + std::to_string(phi_) + "degrees";
      TimeDomainObserver observer_2(name_, point_, rootCS, triggertime_, duration_,
                                    sampleRate_, triggertime_);
      detectorZHS.addObserver(observer_2);
    }
  }
  LengthType const step = 1_m;
  // In a uniform refractive index the exact ray is straight and its delay is
  // n*R/c.  This ice-local propagator also accepts CUDA's radio-track adapter.
  auto TP = make_homogeneous_ice_radio_propagator(env);

  // initiate CoREAS
  RadioProcess<decltype(detectorCoREAS), CoREAS<decltype(detectorCoREAS), decltype(TP)>,
               decltype(TP)>
      coreas(detectorCoREAS, TP);

  // register CoREAS with the output manager
  output.add("CoREAS", coreas);

  // initiate ZHS
  RadioProcess<decltype(detectorZHS), ZHS<decltype(detectorZHS), decltype(TP)>,
               decltype(TP)>
      zhs(detectorZHS, TP);

  // register ZHS with the output manager
  output.add("ZHS", zhs);

  // make and register the first interaction writer
  InteractionWriter<setup::Tracking, ParticleWriterParquet> inter_writer(
      showerAxis, observationLevel);
  output.add("interactions", inter_writer);

  YAML::Node timing_configuration;
  timing_configuration["em_backend"] = em_backend;
  timing_configuration["radio_backend"] = radio_backend;
  timing_configuration["clock"] = "steady_clock";
  timing_configuration["scope"] =
      "OutputManager startOfShower through endOfShower";
  SimulationTiming simulation_timing{
      std::move(timing_configuration)};
  // OutputManager iterates its ordered name map. "simulation_timing" sorts
  // after the current physics writer keys, so this includes their shower-end
  // flushes on both scalar and hybrid paths.
  output.add("simulation_timing", simulation_timing);

  /* === END: SETUP PROCESS LIST === */

  if (!cuda_replay_tape_out.empty()) {
    auto const replay_radio_snapshot =
        validation::makeReplayRadioSnapshot(
            env, injectionPos, surface_, step,
            detectorCoREAS, detectorZHS);
    validation::CudaDecisionReplayTape::instance().open(
        cuda_replay_tape_out, replay_radio_snapshot,
        static_cast<std::uint64_t>(seed),
        static_cast<std::uint64_t>(nevent));
  }

  // trigger the output manager to open the library for writing
  output.startOfLibrary();

  // loop over each shower
  for (int i_shower = 1; i_shower < nevent + 1; i_shower++) {

    CORSIKA_LOG_INFO("Shower {} / {} ", i_shower, nevent);
    validation::CudaReplayTrace::instance().beginShower(i_shower - 1);
    validation::CudaDecisionReplayTape::instance().beginShower(
        static_cast<std::uint64_t>(i_shower - 1));
    auto const photo_hadronic_le_before =
        photoHadronicLowEnergy.statistics();
    auto const photo_hadronic_before =
        photoHadronicHighEnergy.statistics();
    auto const low_energy_hadronic_interactions_before =
        leIntCounted.getCount();
    auto const high_energy_hadronic_interactions_before =
        heCounted.getCount();
    auto const low_energy_hadronic_timings_before =
        leIntCounted.getTimingSamples().size();
    auto const high_energy_hadronic_timings_before =
        heCounted.getTimingSamples().size();
    auto const particle_cut_statistics_before =
        cut.statistics();
    auto const observation_statistics_before =
        observationLevel.statistics();
    auto const low_energy_ledger_before =
        leIntCounted.getEnergyLedgerStatistics();
    auto const high_energy_ledger_before =
        heCounted.getEnergyLedgerStatistics();
    auto const photo_hadronic_ledger_before =
        emCascade.energyLedgerStatistics();
    auto const scalar_atomic_electron_interactions_before =
        emCascade.atomicElectronTargetInteractions();
    auto const scalar_atomic_electron_rest_input_before_GeV =
        emCascade.weightedAtomicElectronRestMassInputGeV();
#ifdef CORSIKA8_WITH_CUDA_EM
    auto const hadronic_pool_statistics_before =
        hadronic_process_pool
            ? hadronic_process_pool->statistics()
            : HadronicProcessPoolStatistics{};
#endif

    // randomize the primary energy
    double const eSlope = app["--eslope"]->as<double>();
    PowerLawDistribution<HEPEnergyType> powerLawRng(eSlope, eMin, eMax);
    HEPEnergyType const primaryTotalEnergy =
        (eMax == eMin) ? eMin
                       : powerLawRng(RNGManager<>::getInstance().getRandomStream(
                             "primary_particle"));

    auto const eKin = primaryTotalEnergy - get_mass(beamCode);

    // set up thinning based on primary parameters
    double const emthinfrac = app["--emthin"]->as<double>();
    double const configuredMaxWeight =
        app["--max-weight"]->as<double>();
    bool const automaticMaxWeight =
        configuredMaxWeight <= 0.;
    double const maxWeight = std::invoke([&]() {
      if (configuredMaxWeight > 0.)
        return configuredMaxWeight;
      else
        return 0.5 * emthinfrac * primaryTotalEnergy / 1_GeV;
    });
    bool const thinningCanActivateFromUnitWeight =
        emthinfrac > 0. && maxWeight > 1.;
    if (emthinfrac > 0. &&
        !thinningCanActivateFromUnitWeight) {
      CORSIKA_LOG_WARN(
          "EM thinning is configured with threshold={} GeV, but "
          "maximum weight={} is not above the unit initial particle "
          "weight. EMThinning's parentWeight >= maxWeight guard "
          "prevents thinning from starting. Specify --max-weight > 1 "
          "or use a larger emthin*primary-energy product if thinning "
          "is intended.",
          emthinfrac * primaryTotalEnergy / 1_GeV,
          maxWeight);
    }
    EMThinning thinning{emthinfrac * primaryTotalEnergy, maxWeight, !multithin};

    // set up the stack inspector
    StackInspector<StackType> stackInspect(10000, false, primaryTotalEnergy);

    // assemble the final process sequence
    auto sequence =
        make_sequence(stackInspect, neutrinoPrimaryPythia, hadronSequence, decaySequence,
                      emCascade, prodprof, emContinuous, coreas, zhs, longprof,
                      observationLevel, inter_writer, thinning, cut);

    // The transport-identity stack is physically equivalent to the default
    // stack on the scalar path and supplies stable history keys on CUDA.
    TrackingType tracking(app["--max-deflection-angle"]->as<double>());
    StackType stack;

    // setup particle stack, and add primary particle
    stack.clear();

    // print our primary parameters all in one place
    CORSIKA_LOG_INFO("Primary name:         {}", beamCode);
    if (app["--pdg"]->count() > 0) {
      CORSIKA_LOG_INFO("Primary PDG ID:       {}", app["--pdg"]->as<int>());
    } else {
      CORSIKA_LOG_INFO("Primary Z/A:          {}/{}", Z, A);
    }
    CORSIKA_LOG_INFO("Primary Total Energy: {}", primaryTotalEnergy);
    CORSIKA_LOG_INFO("Primary Momentum:     {}",
                     calculate_momentum(primaryTotalEnergy, get_mass(beamCode)));
    CORSIKA_LOG_INFO("Primary Direction:    {}", propDir.getNorm());
    CORSIKA_LOG_INFO("Point of Injection:   {}", injectionPos.getCoordinates());
    CORSIKA_LOG_INFO("Shower Axis Length:   {}",
                     (showerCore - injectionPos).getNorm() * 1.2);

    // add the desired particle to the stack
    auto const primaryProperties =
        std::make_tuple(beamCode, eKin, propDir.normalized(), injectionPos, 0_ns);
    stack.addParticle(primaryProperties);

    primaryWriter.recordPrimary(primaryProperties);

    auto configure_forced_primary = [&](auto& cascade) {
      if (force_interaction) {
        CORSIKA_LOG_INFO("Fixing first interaction at injection point.");
        cascade.forceInteraction();
      }
      if (force_decay) {
        CORSIKA_LOG_INFO("Forcing the primary to decay");
        cascade.forceDecay();
      }
    };

    auto const output_shower_id =
        static_cast<unsigned int>(
            output.getEventId());
    if (em_backend == "proposal") {
      Cascade EAS(env, tracking, sequence, output, stack);
      configure_forced_primary(EAS);
      EAS.run();
      auto const particle_cut_statistics =
          subtractStatistics(cut.statistics(), particle_cut_statistics_before);
      auto const observation_statistics = subtractStatistics(
          observationLevel.statistics(), observation_statistics_before);
      auto const low_energy_ledger = subtractStatistics(
          leIntCounted.getEnergyLedgerStatistics(), low_energy_ledger_before);
      auto const high_energy_ledger = subtractStatistics(
          heCounted.getEnergyLedgerStatistics(), high_energy_ledger_before);
      auto const photo_hadronic_ledger = subtractStatistics(
          emCascade.energyLedgerStatistics(), photo_hadronic_ledger_before);
      auto const scalar_atomic_electron_interactions =
          emCascade.atomicElectronTargetInteractions() -
          scalar_atomic_electron_interactions_before;
      auto const scalar_atomic_electron_rest_input_GeV =
          emCascade.weightedAtomicElectronRestMassInputGeV() -
          scalar_atomic_electron_rest_input_before_GeV;
      auto const hadronic_target_input_GeV =
          low_energy_ledger.weighted_effective_target_total_energy_GeV +
          high_energy_ledger.weighted_target_total_energy_GeV +
          photo_hadronic_ledger.weighted_target_total_energy_GeV;
      auto const generator_residual_GeV =
          low_energy_ledger.weighted_isotope_corrected_energy_residual_GeV +
          high_energy_ledger.weighted_energy_residual_GeV +
          photo_hadronic_ledger.weighted_energy_residual_GeV;
      auto const deposited_and_cut_kinetic_GeV =
          dEdX.getEnergyLost() / 1_GeV;
      auto const scalar_cut_rest_GeV =
          particle_cut_statistics.weighted_rest_mass_energy_GeV;
      auto const source_GeV = primaryTotalEnergy / 1_GeV +
                              scalar_atomic_electron_rest_input_GeV +
                              hadronic_target_input_GeV;
      auto const terminal_GeV = deposited_and_cut_kinetic_GeV +
                                scalar_cut_rest_GeV +
                                observation_statistics.weighted_total_energy_GeV +
                                generator_residual_GeV;
      auto const residual_GeV = source_GeV - terminal_GeV;
      auto const relative_error = std::abs(residual_GeV) /
          std::max(source_GeV, std::numeric_limits<double>::min());
      auto const low_energy_interaction_count =
          leIntCounted.getCount() - low_energy_hadronic_interactions_before;
      auto const high_energy_interaction_count =
          heCounted.getCount() - high_energy_hadronic_interactions_before;
      auto const photo_hadronic_le_current =
          photoHadronicLowEnergy.statistics();
      auto const photo_hadronic_he_current =
          photoHadronicHighEnergy.statistics();
      auto const photo_hadronic_interaction_count =
          photo_hadronic_le_current.preferred_interactions -
              photo_hadronic_le_before.preferred_interactions +
          photo_hadronic_le_current.fallback_interactions -
              photo_hadronic_le_before.fallback_interactions +
          photo_hadronic_he_current.preferred_interactions -
              photo_hadronic_before.preferred_interactions +
          photo_hadronic_he_current.fallback_interactions -
              photo_hadronic_before.fallback_interactions;
      auto const every_hadronic_vertex_was_audited =
          low_energy_ledger.audited_interactions +
                  low_energy_ledger.deferred_interactions ==
              low_energy_interaction_count &&
          high_energy_ledger.audited_interactions +
                  high_energy_ledger.deferred_interactions ==
              high_energy_interaction_count &&
          photo_hadronic_ledger.interactions ==
              photo_hadronic_interaction_count;
      auto const complete_coverage =
          is_em(beamCode) && emthinfrac == 0. &&
          low_energy_ledger.deferred_interactions == 0 &&
          high_energy_ledger.deferred_interactions == 0 &&
          every_hadronic_vertex_was_audited;
      YAML::Node metadata;
      auto ledger = metadata["energy_ledger"];
      ledger["definition"] =
          "initial_total + scalar_medium_electron_rest + hadronic_target_total = "
          "deposited_and_cut_kinetic + scalar_cut_rest + observed_total + "
          "untracked_interaction_local_energy";
      ledger["complete_coverage"] = complete_coverage;
      ledger["initial_total_GeV"] = primaryTotalEnergy / 1_GeV;
      ledger["scalar_medium_electron_rest_mass_input_GeV"] =
          scalar_atomic_electron_rest_input_GeV;
      ledger["medium_electron_rest_mass_input_GeV"] =
          scalar_atomic_electron_rest_input_GeV;
      ledger["scalar_atomic_electron_target_interactions"] =
          scalar_atomic_electron_interactions;
      ledger["hadronic_target_total_energy_input_GeV"] =
          hadronic_target_input_GeV;
      ledger["deposited_and_cut_kinetic_GeV"] =
          deposited_and_cut_kinetic_GeV;
      ledger["scalar_cut_rest_mass_energy_GeV"] =
          scalar_cut_rest_GeV;
      ledger["observed_total_energy_GeV"] =
          observation_statistics.weighted_total_energy_GeV;
      ledger["interaction_generator_residual_GeV"] =
          generator_residual_GeV;
      ledger["source_GeV"] = source_GeV;
      ledger["terminal_GeV"] = terminal_GeV;
      ledger["residual_GeV"] = residual_GeV;
      ledger["relative_closure_error"] = relative_error;
      ledger["acceptance_tolerance"] = 1.e-4;
      ledger["accepted"] = complete_coverage && relative_error <= 1.e-4;
      if (scalar_energy_ledger_output) {
        scalar_energy_ledger_output->recordComplete(
            output_shower_id, std::move(metadata));
      }
      CORSIKA_LOG_INFO(
          "scalar whole-event energy ledger: complete={}, source={} GeV, "
          "terminal={} GeV, residual={} GeV, relative error={}",
          complete_coverage, source_GeV, terminal_GeV, residual_GeV,
          relative_error);
      auto const photo_hadronic_after =
          photoHadronicHighEnergy.statistics();
      auto const photo_hadronic_le_after =
          photoHadronicLowEnergy.statistics();
      CORSIKA_LOG_INFO(
          "PROPOSAL photo-hadronic generators: SOPHIA={}, threshold fallback={}, "
          "HE preferred={}, HE fallback={}, QGSJet-II initialized={}",
          photo_hadronic_le_after.preferred_interactions -
              photo_hadronic_le_before.preferred_interactions,
          photo_hadronic_le_after.fallback_interactions -
              photo_hadronic_le_before.fallback_interactions,
          photo_hadronic_after.preferred_interactions -
              photo_hadronic_before.preferred_interactions,
          photo_hadronic_after.fallback_interactions -
              photo_hadronic_before.fallback_interactions,
          photoHadronicQgsjetFallback.initialized());
    } else {
#ifdef CORSIKA8_WITH_CUDA_EM
      using namespace corsika::gpu::em;
      using namespace corsika::gpu::em::tables;

      try {
      if (!loaded_gpu_table) {
        throw std::logic_error(
            "CUDA EM table was not loaded during application initialization");
      }
      auto const& table = *loaded_gpu_table;
      auto const requested_cut_MeV = emcut / 1_MeV;
      auto const proposal_stochastic_cut =
          proposal::optimized_proposal_energy_cut(prod_threshold);
      auto const proposal_stochastic_cut_MeV =
          proposal_stochastic_cut / 1_MeV;
      auto const cut_scale =
          std::max({1., std::abs(proposal_stochastic_cut_MeV),
                    std::abs(table.metadata.energy_cut_MeV)});
      if (std::abs(
              table.metadata.energy_cut_MeV -
              proposal_stochastic_cut_MeV) >
          32. * std::numeric_limits<double>::epsilon() * cut_scale) {
        throw std::runtime_error(
            "CUDA EM table stochastic cut does not match the scalar "
            "PROPOSAL cut resolved from the configured production threshold");
      }
      if (table.metadata.energy_min_MeV >
          std::min(
              requested_cut_MeV,
              proposal_stochastic_cut_MeV)) {
        throw std::runtime_error(
            "CUDA EM table energy domain does not cover the configured "
            "transport and stochastic cuts");
      }
      for (auto const pdg : {11, -11}) {
        auto const& continuous = findContinuousEnergyTable(table, pdg);
        auto const code = convert_from_PDG(static_cast<PDGCode>(pdg));
        auto const table_transport_cut_MeV =
            continuous.minimum_total_energy_MeV - get_mass(code) / 1_MeV;
        auto const expected_transport_cut_MeV =
            ContinuousCutSafetyFactor * requested_cut_MeV;
        if (std::abs(table_transport_cut_MeV - expected_transport_cut_MeV) >
            2.e-6 * cut_scale) {
          throw std::runtime_error(
              "CUDA EM table transport cut does not match --emcut");
        }
      }
      auto const tableHasPid = [&](std::int32_t pdg) {
        return std::any_of(
            table.particles.begin(), table.particles.end(),
            [pdg](auto const& particle) {
              return particle.pdg_id == pdg;
            });
      };
      auto const gpu_muon_transport_available =
          tableHasPid(13) && tableHasPid(-13);
      if (tableHasPid(13) != tableHasPid(-13)) {
        throw std::runtime_error(
            "CUDA EM table contains only one muon charge state");
      }
      if (gpu_muon_transport_available) {
        auto const requested_muon_cut_MeV =
            mucut / 1_MeV;
        auto const expected_muon_transport_cut_MeV =
            ContinuousCutSafetyFactor *
            requested_muon_cut_MeV;
        for (auto const pdg : {13, -13}) {
          auto const& continuous =
              findContinuousEnergyTable(table, pdg);
          auto const code =
              convert_from_PDG(static_cast<PDGCode>(pdg));
          auto const table_transport_cut_MeV =
              continuous.minimum_total_energy_MeV -
              get_mass(code) / 1_MeV;
          auto const muon_cut_scale =
              std::max(
                  {1., std::abs(table_transport_cut_MeV),
                   std::abs(
                       expected_muon_transport_cut_MeV)});
          if (std::abs(
                  table_transport_cut_MeV -
                  expected_muon_transport_cut_MeV) >
              2.e-6 * muon_cut_scale) {
            throw std::runtime_error(
                "CUDA muon table transport cut does not match --mucut");
          }
        }
      }
      if (primaryTotalEnergy / 1_MeV > table.metadata.energy_max_MeV) {
        throw std::runtime_error(
            "primary energy exceeds the CUDA EM table energy domain");
      }

      ProposalTableSet descriptor{};
      descriptor.process_count = rateTableProcessCount(table);
      descriptor.content_hash = table.content_hash;
      auto environment_snapshot = [&]() {
        std::array<double, 3> const device_field_T{
            transport_field.getX(rootCS) / 1_T,
            transport_field.getY(rootCS) / 1_T,
            transport_field.getZ(rootCS) / 1_T};
        return makeHomogeneousSphericalSnapshot(
            ice_density_g_per_cm3, ice_inner_radius_m, ice_outer_radius_m,
            observationHeight / 1_m, {0., 0., 0.},
            static_cast<std::int32_t>(Medium::WaterIce), device_field_T,
            app["--max-deflection-angle"]->as<double>());
      }();
      setObservationPlane(
          environment_snapshot,
          {showerCoreX / 1_m, showerCoreY / 1_m,
           observationHeight / 1_m},
          {0., 0., 1.});
      GpuEmConfig gpu_config{};
      gpu_config.device = gpu_device;
      gpu_config.min_batch_size = gpu_min_batch;
      gpu_config.memory_fraction = gpu_memory_fraction;
      gpu_config.table_tolerance = gpu_table_tolerance;
      gpu_config.deterministic = gpu_deterministic;
      gpu_config.detailed_stage_timing =
          gpu_detailed_stage_timing;
      gpu_config.random_seed = static_cast<std::uint64_t>(seed);
      gpu_config.shower_id = static_cast<std::uint64_t>(i_shower);
      gpu_config.table_cache = gpu_table_cache;
      gpu_config.thinning.enabled = emthinfrac > 0. ? 1 : 0;
      gpu_config.thinning.threshold_GeV =
          emthinfrac * primaryTotalEnergy / 1_GeV;
      gpu_config.thinning.maximum_weight = maxWeight;
      gpu_config.thinning.erase_zero_weight = multithin ? 0 : 1;
      gpu_config.resident_cross_species =
          gpu_resident_cross_species;
      auto const gpu_radio_enabled =
          radio_backend == "cuda" && detectorCoREAS.size() != 0;
      if (gpu_radio_enabled &&
          !reusable_gpu_em_backend) {
        gpu_config.radio =
            gpu::radio::makeGpuRadioConfig(
                env, injectionPos, surface_, step,
                detectorCoREAS, detectorZHS);
        gpu_config.radio.deterministic =
            gpu_config.deterministic;
        gpu_config.radio.zhs_subtrack_refinement =
            gpu_zhs_subtrack_refinement;
        gpu_config.radio.fixed_point_field_limit_V_per_m =
            gpu_radio_field_limit;
        gpu_config.radio.track_diagnostics =
            gpu_radio_track_diagnostics;
      }
      if ((detectorCoREAS.size() == 0 || gpu_radio_enabled) &&
          !gpu_full_step_records) {
        auto& projection =
            gpu_config.profile_projection;
        auto const primary_energy_GeV =
            primaryTotalEnergy / 1_GeV;
        auto const em_cut_GeV = emcut / 1_GeV;
        projection.fixed_point_weight_limit =
            2. * primary_energy_GeV / em_cut_GeV;
        projection.fixed_point_energy_limit_GeV =
            2. * primary_energy_GeV;
        if (!reusable_gpu_em_backend) {
          projection.enabled = true;
          auto const axis_start =
              showerAxis.getStart().getCoordinates(rootCS);
          auto const axis_direction =
              showerAxis.getDirection().getComponents(rootCS);
          for (int axis = 0; axis < 3; ++axis) {
            projection.axis_start_position_m[axis] =
                axis_start[axis] / 1_m;
            projection.axis_direction[axis] =
                axis_direction[axis].magnitude();
          }
          projection.axis_step_length_m =
              showerAxis.getSteplength() / 1_m;
          auto const& support =
              showerAxis.getGrammageSupport();
          projection.axis_grammage_g_per_cm2.reserve(
              support.size());
          for (auto const grammage : support) {
            projection.axis_grammage_g_per_cm2.push_back(
                grammage /
                (1_g / square(1_cm)));
          }
          if (dEdX.GetNBins() != profile.getNBins()) {
            throw std::runtime_error(
                "CUDA resident profile requires identical energy-loss and "
                "longitudinal bin counts");
          }
          projection.accumulate_on_device = true;
          projection.output_bin_count = dEdX.GetNBins();
          projection.output_bin_width_g_per_cm2 =
              dX / (1_g / square(1_cm));
          projection.energy_loss_threshold_g_per_cm2 =
              1.e-4;
        }
      }

      using Sequence = decltype(sequence);
      using GpuEmStepRegistry = GpuEmStepProcessRegistry<
          GpuEmStepProcessRegistration<
              decltype(stackInspect),
              GpuEmStepProcessPolicy::DiagnosticOnly>,
          GpuEmStepProcessRegistration<
              decltype(neutrinoPrimaryPythia),
              GpuEmStepProcessPolicy::InapplicableToRoutedEm>,
          GpuEmStepProcessRegistration<
              decltype(hadronSequence),
              GpuEmStepProcessPolicy::InapplicableToRoutedEm>,
          GpuEmStepProcessRegistration<
              decltype(decaySequence),
              GpuEmStepProcessPolicy::DeferredToCpu>,
          GpuEmStepProcessRegistration<
              decltype(emCascade),
              GpuEmStepProcessPolicy::ReplacedOnDevice>,
          GpuEmStepProcessRegistration<
              decltype(emContinuous),
              GpuEmStepProcessPolicy::ReplacedOnDevice>,
          GpuEmStepProcessRegistration<
              decltype(coreas),
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              decltype(zhs),
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              decltype(longprof),
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              decltype(observationLevel),
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              decltype(prodprof),
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              decltype(inter_writer),
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              decltype(thinning),
              GpuEmStepProcessPolicy::ReplacedOnDevice>,
          GpuEmStepProcessRegistration<
              decltype(cut),
              GpuEmStepProcessPolicy::ReplacedOnDevice>>;
      GpuEmStepRegistry::template validateOrThrow<Sequence>();
      CORSIKA_LOG_INFO(
          "CUDA EM process registry accepted {} process contracts "
          "({} device-replaced, {} record-replayed, {} deferred-to-CPU, "
          "{} inapplicable-to-routed-EM, {} diagnostic-only)",
          GpuEmStepRegistry::registrationCount(),
          GpuEmStepRegistry::replacedOnDeviceCount(),
          GpuEmStepRegistry::replayedFromDeviceRecordCount(),
          GpuEmStepRegistry::deferredToCpuCount(),
          GpuEmStepRegistry::inapplicableToRoutedEmCount(),
          GpuEmStepRegistry::diagnosticOnlyCount());

      auto const backend_reused_for_shower =
          reusable_gpu_em_backend != nullptr;
      if (!reusable_gpu_em_backend) {
        reusable_gpu_em_backend =
            std::make_unique<CudaEmBackend>();
        reusable_gpu_em_backend->initialize(
            environment_snapshot, descriptor, gpu_config);
      } else {
        reusable_gpu_em_backend->beginShower(
            makeGpuEmShowerConfig(gpu_config));
      }
      auto& backend = *reusable_gpu_em_backend;

      using FallbackHandler =
          ProposalCpuFallbackHandler<StackType, decltype(emCascade), Sequence, EnvType>;
      FallbackHandler fallback_handler{
          emCascade, sequence, env, rootCS, static_cast<std::uint64_t>(seed),
          static_cast<std::uint64_t>(i_shower),
          proposal_stochastic_cut};

      CorsikaOutputSink output_sink{
          rootCS, dEdX, profile, prod_profile, observationLevel,
          inter_writer, coreas, zhs,
          detectorCoREAS.size() != 0, gpu_radio_enabled};
      using OutputSink = decltype(output_sink);
      using Router = PhysicalCudaEmRouter<StackType, FallbackHandler, OutputSink>;
      Router router{backend, rootCS, environment_snapshot, fallback_handler, output_sink};
      router.setRetainRecords(
          validation::CudaReplayTrace::instance().enabled());
      router.setFailOnUnexpectedFallback(true);
      HybridCascade<TrackingType, Sequence, OutputManager, StackType, Router> EAS(
          env, tracking, sequence, output, stack, router);
      HadronicWorkClassifierConfig hadronic_work_classifier;
      hadronic_work_classifier.transition_energy_GeV =
          heHadronModelThreshold / 1_GeV;
      EAS.configureHadronicWorkClassification(
          hadronic_work_classifier);
      EAS.enableScalarDetailedPhaseTiming(
          cpu_detailed_step_timing);
      if (hadronic_process_pool) {
        HybridHadronicWorkerConfig
            hadronic_worker_config;
        hadronic_worker_config.seed =
            static_cast<std::uint64_t>(seed);
        hadronic_worker_config.shower_id =
            static_cast<std::uint64_t>(
                output_shower_id);
        hadronic_worker_config
            .minimum_pending_interactions =
            hadronic_min_batch;
        hadronic_worker_config
            .target_batch_cost_ms =
            hadronic_target_batch_ms;
        hadronic_worker_config
            .maximum_batch_items =
            hadronic_max_batch;
        hadronic_worker_config
            .initial_interaction_cost_ms =
            hadronic_initial_cost_ms;
        EAS.configureHadronicProcessPool(
            *hadronic_process_pool,
            hadronic_worker_config);
      }
      configure_forced_primary(EAS);
      EAS.run();

      for (auto const& record : router.stepRecords()) {
        validation::CudaReplayTrace::instance().recordGpuStep(
            record.history_id, record.step_id, record.pid,
            record.process_id, record.start_energy_GeV,
            record.end_energy_GeV, record.deposited_energy_GeV,
            record.weight, record.start_position_m[0],
            record.start_position_m[1], record.start_position_m[2],
            record.start_time_s);
      }

      auto const& backend_stats = backend.statistics();
      auto const& router_stats = router.statistics();
      auto const& sink_stats = output_sink.statistics();
      auto const& proposal_fallback_stats =
          fallback_handler.statistics();
      auto const& hybrid_timing =
          EAS.timingStatistics();
      auto const photo_hadronic_after =
          photoHadronicHighEnergy.statistics();
      auto const photo_hadronic_le_after =
          photoHadronicLowEnergy.statistics();
      auto const photo_hadronic_sophia =
          photo_hadronic_le_after.preferred_interactions -
          photo_hadronic_le_before.preferred_interactions;
      auto const photo_hadronic_threshold_fallback =
          photo_hadronic_le_after.fallback_interactions -
          photo_hadronic_le_before.fallback_interactions;
      auto const photo_hadronic_preferred =
          photo_hadronic_after.preferred_interactions -
          photo_hadronic_before.preferred_interactions;
      auto const photo_hadronic_fallback =
          photo_hadronic_after.fallback_interactions -
          photo_hadronic_before.fallback_interactions;
      CORSIKA_LOG_INFO(
          "CUDA EM summary: staged={}, photon steps={}, lepton steps={}, "
          "wavefronts={}, GPU final states={}, CPU fallbacks={}, "
          "CPU expansion steps={}, forced CPU decays={}/{}, "
          "specified CPU final states={}, "
          "observed={}, escaped={}, "
          "weighted deposit={} GeV, radio tracks={}, "
          "resident wavefronts(photon/lepton)={}/{}, "
          "backend wall(photon/lepton)={}/{} ms, "
          "host postprocess(photon/lepton)={}/{} ms, "
          "annihilation={}, thinning(hillas/statistical/discarded)="
          "{}/{}/{}, peak device={} MiB, kernel={} ms, transfer={} ms",
          router_stats.particles_staged, router_stats.photons_advanced,
          router_stats.leptons_advanced, router_stats.wavefronts,
          backend_stats.gpu_final_states,
          router_stats.particles_returned_for_cpu_fallback,
          router_stats.cpu_wavefront_expansion_steps_executed,
          router_stats.particles_returned_for_cpu_decay,
          router_stats.forced_cpu_decays_executed,
          router_stats.specified_cpu_final_states, router_stats.particles_observed,
          router_stats.particles_escaped, sink_stats.weighted_deposited_energy_GeV,
          sink_stats.radio_tracks,
          router_stats.resident_photon_wavefronts,
          router_stats.resident_lepton_wavefronts,
          router_stats.photon_backend_wall_time_ms,
          router_stats.lepton_backend_wall_time_ms,
          router_stats.photon_host_postprocess_time_ms,
          router_stats.lepton_host_postprocess_time_ms,
          backend_stats.annihilation_final_states,
          backend_stats.thinning_hillas_vertices,
          backend_stats.thinning_statistical_vertices,
          backend_stats.thinning_particles_discarded,
          static_cast<double>(backend_stats.peak_device_bytes) /
              (1024. * 1024.),
          backend_stats.kernel_time_ms, backend_stats.transfer_time_ms);

      auto const& low_energy_timing_samples =
          leIntCounted.getTimingSamples();
      auto const& high_energy_timing_samples =
          heCounted.getTimingSamples();
      ClassifiedHadronicWorkQueue<std::uint64_t>
          hadronic_worker_queue;
      std::map<HadronicWorkKey, HadronicWorkClassStatistics>
          hadronic_final_state_classes;
      std::uint64_t hadronic_worker_sequence = 0;
      double low_energy_final_state_time_ms = 0.;
      double high_energy_final_state_time_ms = 0.;
      std::uint64_t deferred_timing_samples = 0;
      auto stage_timing_samples =
          [&](auto const& samples, std::size_t const first,
              double& model_time_ms) {
            for (std::size_t index = first;
                 index < samples.size(); ++index) {
              auto const& sample = samples[index];
              if (sample.deferred) {
                ++deferred_timing_samples;
                continue;
              }
              model_time_ms += sample.final_state_time_ms;
              auto const key = classifyHadronicWork(
                  sample.projectile, sample.kinetic_energy,
                  hadronic_work_classifier);
              if (!key) {
                throw std::runtime_error(
                    "InteractionCounter recorded a non-hadronic "
                    "projectile in the hadronic sequence");
              }
              hadronic_final_state_classes[*key].record(
                  sample.final_state_time_ms);
              hadronic_worker_queue.enqueue(
                  *key, hadronic_worker_sequence,
                  std::max(sample.final_state_time_ms, 1.e-9),
                  hadronic_worker_sequence);
              ++hadronic_worker_sequence;
            }
          };
      stage_timing_samples(
          low_energy_timing_samples,
          low_energy_hadronic_timings_before,
          low_energy_final_state_time_ms);
      stage_timing_samples(
          high_energy_timing_samples,
          high_energy_hadronic_timings_before,
          high_energy_final_state_time_ms);
      if (hadronic_process_pool) {
        for (auto const& [key, statistics] :
             hybrid_timing
                 .hadronic_worker_final_state_classes) {
          auto& combined =
              hadronic_final_state_classes[key];
          combined.steps += statistics.steps;
          combined.total_time_ms +=
              statistics.total_time_ms;
          combined.squared_time_ms2 +=
              statistics.squared_time_ms2;
          combined.minimum_time_ms =
              std::min(
                  combined.minimum_time_ms,
                  statistics.minimum_time_ms);
          combined.maximum_time_ms =
              std::max(
                  combined.maximum_time_ms,
                  statistics.maximum_time_ms);
          low_energy_final_state_time_ms +=
              statistics.total_time_ms;
        }
      }
      auto const hadronic_worker_assignments =
          planHadronicWorkerAssignments(
              hadronic_worker_queue,
              static_cast<std::size_t>(hadronic_plan_workers),
              hadronic_plan_target_ms,
              hadronic_plan_max_batch);
      double hadronic_worker_predicted_makespan_ms = 0.;
      double hadronic_worker_minimum_load_ms =
          std::numeric_limits<double>::infinity();
      std::size_t hadronic_worker_batch_count = 0;
      for (auto const& assignment :
           hadronic_worker_assignments) {
        hadronic_worker_predicted_makespan_ms =
            std::max(
                hadronic_worker_predicted_makespan_ms,
                assignment.estimated_cost);
        hadronic_worker_minimum_load_ms =
            std::min(
                hadronic_worker_minimum_load_ms,
                assignment.estimated_cost);
        hadronic_worker_batch_count +=
            assignment.batches.size();
      }
      if (hadronic_worker_assignments.empty()) {
        hadronic_worker_minimum_load_ms = 0.;
      }
      auto const hadronic_final_state_serial_time_ms =
          low_energy_final_state_time_ms +
          high_energy_final_state_time_ms;

      YAML::Node shower_metadata;
      shower_metadata["hadronic_models"]["high_energy"]["name"] =
          app["--hadronModel"]->as<std::string>();
      shower_metadata["hadronic_models"]["high_energy"]["interactions"] =
          static_cast<std::uint64_t>(
              heCounted.getCount() -
              high_energy_hadronic_interactions_before);
      shower_metadata["hadronic_models"]["high_energy"]
                     ["final_state_time_ms"] =
          high_energy_final_state_time_ms;
#ifdef WITH_FLUKA
      shower_metadata["hadronic_models"]["low_energy"]["name"] =
          "FLUKA";
      shower_metadata["hadronic_models"]["low_energy"]["version"] =
          ::fluka::get_version();
#else
      shower_metadata["hadronic_models"]["low_energy"]["name"] =
          "UrQMD";
#endif
      shower_metadata["hadronic_models"]["low_energy"]["interactions"] =
          static_cast<std::uint64_t>(
              leIntCounted.getCount() -
              low_energy_hadronic_interactions_before);
      shower_metadata["hadronic_models"]["low_energy"]
                     ["final_state_time_ms"] =
          low_energy_final_state_time_ms;
      shower_metadata["hadronic_models"]["transition_energy_GeV"] =
          heHadronModelThreshold / 1_GeV;
      shower_metadata["thinning"]["em_fraction"] =
          emthinfrac;
      shower_metadata["thinning"]["maximum_weight"] =
          maxWeight;
      shower_metadata["thinning"]
                     ["automatic_maximum_weight"] =
          automaticMaxWeight;
      shower_metadata["thinning"]
                     ["can_activate_from_unit_weight"] =
          thinningCanActivateFromUnitWeight;
      shower_metadata["process_registry"]["registrations"] =
          GpuEmStepRegistry::registrationCount();
      shower_metadata["process_registry"]["device_replaced"] =
          GpuEmStepRegistry::replacedOnDeviceCount();
      shower_metadata["process_registry"]["record_replayed"] =
          GpuEmStepRegistry::
              replayedFromDeviceRecordCount();
      shower_metadata["process_registry"]["deferred_to_cpu"] =
          GpuEmStepRegistry::deferredToCpuCount();
      shower_metadata["process_registry"]
                     ["inapplicable_to_routed_em"] =
          GpuEmStepRegistry::inapplicableToRoutedEmCount();
      shower_metadata["process_registry"]["diagnostic_only"] =
          GpuEmStepRegistry::diagnosticOnlyCount();
      shower_metadata["process_registry"]
                     ["unregistered_continuous_processes"] =
          GpuEmStepRegistry::template
              unregisteredContinuousProcessCount<Sequence>();
      shower_metadata["process_registry"]
                     ["unregistered_secondaries_processes"] =
          GpuEmStepRegistry::template
              unregisteredSecondariesProcessCount<Sequence>();
      shower_metadata["process_registry"]
                     ["unregistered_interaction_processes"] =
          GpuEmStepRegistry::template
              unregisteredInteractionProcessCount<Sequence>();
      shower_metadata["process_registry"]
                     ["unregistered_decay_processes"] =
          GpuEmStepRegistry::template
              unregisteredDecayProcessCount<Sequence>();
      shower_metadata["process_registry"]
                     ["unregistered_boundary_processes"] =
          GpuEmStepRegistry::template
              unregisteredBoundaryProcessCount<Sequence>();
      shower_metadata["process_registry"]
                     ["unregistered_stack_processes"] =
          GpuEmStepRegistry::template
              unregisteredStackProcessCount<Sequence>();
      shower_metadata["process_registry"]["accepted"] = true;
      shower_metadata["forced_primary"]
                     ["interaction_executed"] =
          hybrid_timing.forced_primary_interactions;
      shower_metadata["forced_primary"]["decay_executed"] =
          hybrid_timing.forced_primary_decays;
      shower_metadata["gpu_particles"] =
          router_stats.photons_advanced +
          router_stats.leptons_advanced;
      shower_metadata["backend_lifecycle"]["reused"] =
          backend_reused_for_shower;
      shower_metadata["backend_lifecycle"]["shower_ordinal"] =
          backend_stats.shower_ordinal;
      shower_metadata["backend_lifecycle"]
                     ["one_time_initialization_ms"] =
          backend_stats.one_time_initialization_ms;
      shower_metadata["backend_lifecycle"]
                     ["static_host_to_device_bytes"] =
          backend_stats.static_host_to_device_bytes;
      shower_metadata["cpu_particle_steps"] =
          EAS.schedulerStatistics()
              .acquired_particle_steps -
          router_stats.particles_staged;
      shower_metadata["particles_staged"] =
          router_stats.particles_staged;
      shower_metadata["gpu_muon_transport_enabled"] =
          gpu_muon_transport_available;
      shower_metadata["wavefronts"] =
          router_stats.wavefronts;
      shower_metadata["resident_photon_wavefronts"] =
          router_stats.resident_photon_wavefronts;
      shower_metadata["resident_lepton_wavefronts"] =
          router_stats.resident_lepton_wavefronts;
      shower_metadata["below_minimum_batch_checkpoints"] =
          router_stats.below_minimum_batch_checkpoints;
      shower_metadata["workspace_limit_checkpoints"] =
          router_stats.workspace_limit_checkpoints;
      shower_metadata["input_batch_splits"] =
          router_stats.input_batch_splits;
      shower_metadata["maximum_input_batch"] =
          static_cast<std::uint64_t>(
              router_stats.maximum_input_batch);
      shower_metadata["gpu_final_states"] =
          backend_stats.gpu_final_states;
      shower_metadata["first_interaction_candidates"] =
          backend_stats.first_interaction_candidates;
      shower_metadata["first_interactions_written"] =
          output_sink.statistics().first_interactions;
      shower_metadata["physical_secondaries"] =
          backend_stats.physical_secondaries_generated;
      shower_metadata["cpu_generic_fallbacks"] =
          router_stats
              .particles_returned_for_cpu_fallback;
      shower_metadata["cpu_memory_spill_particles"] =
          router_stats
              .particles_returned_for_cpu_memory_spill;
      shower_metadata["cpu_decay_particles"] =
          router_stats.particles_returned_for_cpu_decay;
      shower_metadata["forced_cpu_decays_executed"] =
          router_stats.forced_cpu_decays_executed;
      shower_metadata["cpu_specified_final_states"] =
          router_stats.specified_cpu_final_states;
      shower_metadata["deferred_cpu_fallbacks_queued"] =
          router_stats.deferred_cpu_fallbacks_queued;
      shower_metadata["deferred_cpu_fallbacks_flushed"] =
          router_stats.deferred_cpu_fallbacks_flushed;
      shower_metadata["deferred_cpu_fallback_flushes"] =
          router_stats.deferred_cpu_fallback_flushes;
      shower_metadata["deferred_fallback_scalar_expansion_rounds"] =
          router_stats.deferred_fallback_scalar_expansion_rounds;
      shower_metadata["deferred_front_gpu_flushes"] =
          router_stats.deferred_front_gpu_flushes;
      shower_metadata["deferred_product_gpu_flushes"] =
          router_stats.deferred_product_gpu_flushes;
      shower_metadata["maximum_deferred_cpu_fallback_batch"] =
          static_cast<std::uint64_t>(
              router_stats.maximum_deferred_cpu_fallback_batch);
      shower_metadata["cpu_completed_selected_losses"] =
          proposal_fallback_stats
              .completed_selected_losses;
      shower_metadata["photo_hadronic_generator"]
                     ["sophia_interactions"] =
          photo_hadronic_sophia;
      shower_metadata["photo_hadronic_generator"]
                     ["threshold_fallback_interactions"] =
          photo_hadronic_threshold_fallback;
      shower_metadata["photo_hadronic_generator"]
                     ["preferred_interactions"] =
          photo_hadronic_preferred;
      shower_metadata["photo_hadronic_generator"]
                     ["fallback_interactions"] =
          photo_hadronic_fallback;
      shower_metadata["photo_hadronic_generator"]
                     ["discarded_final_states"] = 0;
      shower_metadata["photo_hadronic_generator"]
                     ["fallback_model_initialized"] =
          photoHadronicQgsjetFallback.initialized();
      shower_metadata["hybrid_timing_ms"]["total_run"] =
          hybrid_timing.total_run_time_ms;
      shower_metadata["hybrid_timing_ms"]["output_start"] =
          hybrid_timing.output_start_time_ms;
      shower_metadata["hybrid_timing_ms"]["set_nodes"] =
          hybrid_timing.set_nodes_time_ms;
      shower_metadata["hybrid_timing_ms"]["route_stage"] =
          hybrid_timing.route_stage_time_ms;
      shower_metadata["hybrid_timing_ms"]["scalar_stepper"] =
          hybrid_timing.scalar_stepper_time_ms;
      shower_metadata["hybrid_timing_ms"]["do_stack"] =
          hybrid_timing.do_stack_time_ms;
      shower_metadata["hybrid_timing_ms"]["router_advance"] =
          hybrid_timing.router_advance_time_ms;
      shower_metadata["hybrid_timing_ms"]
                     ["init_cascade_equations"] =
          hybrid_timing.init_cascade_equations_time_ms;
      shower_metadata["hybrid_timing_ms"]
                     ["cascade_equations"] =
          hybrid_timing.cascade_equations_time_ms;
      shower_metadata["hybrid_timing_ms"]["output_end"] =
          hybrid_timing.output_end_time_ms;
      shower_metadata["hadronic_process_pool"]["backend"] =
          hadronic_backend;
      shower_metadata["hadronic_process_pool"]["enabled"] =
          hybrid_timing.hadronic_process_pool_enabled;
      shower_metadata["hadronic_process_pool"]["workers"] =
          hadronic_process_pool
              ? hadronic_process_pool->workerCount()
              : 0;
      shower_metadata["hadronic_process_pool"]
                     ["minimum_pending_interactions"] =
          hadronic_min_batch;
      shower_metadata["hadronic_process_pool"]
                     ["target_batch_cost_ms"] =
          hadronic_target_batch_ms;
      shower_metadata["hadronic_process_pool"]
                     ["maximum_batch_items"] =
          hadronic_max_batch;
      shower_metadata["hadronic_process_pool"]
                     ["initial_interaction_cost_ms"] =
          hadronic_initial_cost_ms;
      shower_metadata["hadronic_process_pool"]
                     ["target_total_cost_ms"] =
          hadronic_target_batch_ms *
          static_cast<double>(
              hadronic_process_pool
                  ? hadronic_process_pool->workerCount()
                  : 0);
      shower_metadata["hadronic_process_pool"]
                     ["prepared_interactions"] =
          hybrid_timing.hadronic_interactions_prepared;
      shower_metadata["hadronic_process_pool"]
                     ["committed_interactions"] =
          hybrid_timing.hadronic_interactions_committed;
      shower_metadata["hadronic_process_pool"]["flushes"] =
          hybrid_timing.hadronic_worker_flushes;
      shower_metadata["hadronic_process_pool"]
                     ["cost_triggered_flushes"] =
          hybrid_timing
              .hadronic_worker_cost_triggered_flushes;
      shower_metadata["hadronic_process_pool"]
                     ["capacity_triggered_flushes"] =
          hybrid_timing
              .hadronic_worker_capacity_triggered_flushes;
      shower_metadata["hadronic_process_pool"]
                     ["drain_triggered_flushes"] =
          hybrid_timing
              .hadronic_worker_drain_triggered_flushes;
      shower_metadata["hadronic_process_pool"]
                     ["classified_batches"] =
          hybrid_timing
              .hadronic_worker_classified_batches;
      shower_metadata["hadronic_process_pool"]["batches"] =
          hybrid_timing.hadronic_worker_batches;
      shower_metadata["hadronic_process_pool"]["secondaries"] =
          hybrid_timing.hadronic_worker_secondaries;
      shower_metadata["hadronic_process_pool"]
                     ["prepare_time_ms"] =
          hybrid_timing.hadronic_prepare_time_ms;
      shower_metadata["hadronic_process_pool"]
                     ["execute_time_ms"] =
          hybrid_timing.hadronic_worker_execute_time_ms;
      shower_metadata["hadronic_process_pool"]
                     ["commit_time_ms"] =
          hybrid_timing.hadronic_commit_time_ms;
      shower_metadata["hadronic_process_pool"]
                     ["deferred_counter_samples"] =
          deferred_timing_samples;
      YAML::Node hadronic_flush_fingerprints;
      for (auto const& fingerprint :
           hybrid_timing.hadronic_flush_fingerprints) {
        YAML::Node node;
        node["requests"] = fingerprint.requests;
        node["request_hash"] =
            std::to_string(fingerprint.request_hash);
        node["response_hash"] =
            std::to_string(fingerprint.response_hash);
        hadronic_flush_fingerprints.push_back(
            std::move(node));
      }
      shower_metadata["hadronic_process_pool"]
                     ["flush_fingerprints"] =
          std::move(hadronic_flush_fingerprints);
      YAML::Node hadronic_flush_loads;
      for (auto const& load :
           hybrid_timing.hadronic_flush_load_records) {
        YAML::Node node;
        node["trigger"] = std::string(
            hadronicQueueFlushTriggerName(
                load.trigger));
        node["requests"] = load.requests;
        YAML::Node predicted;
        YAML::Node actual;
        for (auto const value :
             load.predicted_cost_by_worker) {
          predicted.push_back(value);
        }
        for (auto const value :
             load
                 .actual_final_state_time_by_worker_ms) {
          actual.push_back(value);
        }
        node["predicted_cost_by_worker"] =
            std::move(predicted);
        node["actual_final_state_time_by_worker_ms"] =
            std::move(actual);
        hadronic_flush_loads.push_back(
            std::move(node));
      }
      shower_metadata["hadronic_process_pool"]
                     ["flush_loads"] =
          std::move(hadronic_flush_loads);
      shower_metadata["hadronic_process_pool"]
                     ["maximum_parked_projectiles"] =
          EAS.schedulerStatistics()
              .maximum_suspended_particles;
      if (hadronic_process_pool) {
        auto const pool_after =
            hadronic_process_pool->statistics();
        shower_metadata["hadronic_process_pool"]
                       ["ipc_batches"] =
            pool_after.batches -
            hadronic_pool_statistics_before.batches;
        shower_metadata["hadronic_process_pool"]
                       ["ipc_requests"] =
            pool_after.requests -
            hadronic_pool_statistics_before.requests;
        shower_metadata["hadronic_process_pool"]
                       ["ipc_secondaries"] =
            pool_after.secondaries -
            hadronic_pool_statistics_before.secondaries;
        shower_metadata["hadronic_process_pool"]
                       ["bytes_sent"] =
            pool_after.bytes_sent -
            hadronic_pool_statistics_before.bytes_sent;
        shower_metadata["hadronic_process_pool"]
                       ["bytes_received"] =
            pool_after.bytes_received -
            hadronic_pool_statistics_before.bytes_received;
        shower_metadata["hadronic_process_pool"]
                       ["dispatch_time_ms"] =
            pool_after.dispatch_time_ms -
            hadronic_pool_statistics_before
                .dispatch_time_ms;
        shower_metadata["hadronic_process_pool"]
                       ["poll_wait_time_ms"] =
            pool_after.poll_wait_time_ms -
            hadronic_pool_statistics_before
                .poll_wait_time_ms;
        shower_metadata["hadronic_process_pool"]
                       ["receive_time_ms"] =
            pool_after.receive_time_ms -
            hadronic_pool_statistics_before
                .receive_time_ms;
        shower_metadata["hadronic_process_pool"]
                       ["wire_protocol_version"] =
            HadronicBatchProtocolVersion;
        shower_metadata["hadronic_process_pool"]
                       ["bulk_response_batches"] =
            pool_after.bulk_response_batches -
            hadronic_pool_statistics_before
                .bulk_response_batches;
        shower_metadata["hadronic_process_pool"]
                       ["bulk_response_payload_reads"] =
            pool_after.bulk_response_payload_reads -
            hadronic_pool_statistics_before
                .bulk_response_payload_reads;
      }
      YAML::Node actual_hadronic_worker_loads;
      double actual_worker_minimum_ms =
          std::numeric_limits<double>::infinity();
      double actual_worker_maximum_ms = 0.;
      for (std::size_t worker = 0;
           hadronic_process_pool &&
           worker <
               hadronic_process_pool->workerCount();
           ++worker) {
        auto const load_found =
            hybrid_timing
                .hadronic_worker_final_state_time_by_worker_ms
                .find(worker);
        auto const load_ms =
            load_found ==
                    hybrid_timing
                        .hadronic_worker_final_state_time_by_worker_ms
                        .end()
                ? 0.
                : load_found->second;
        actual_worker_minimum_ms =
            std::min(actual_worker_minimum_ms, load_ms);
        actual_worker_maximum_ms =
            std::max(actual_worker_maximum_ms, load_ms);
        YAML::Node worker_node;
        worker_node["worker_id"] = worker;
        worker_node["final_state_time_ms"] =
            load_ms;
        worker_node["batches"] =
            hybrid_timing
                .hadronic_worker_batches_by_worker
                .count(worker)
                ? hybrid_timing
                      .hadronic_worker_batches_by_worker
                      .at(worker)
                : 0;
        worker_node["requests"] =
            hybrid_timing
                .hadronic_worker_requests_by_worker
                .count(worker)
                ? hybrid_timing
                      .hadronic_worker_requests_by_worker
                      .at(worker)
                : 0;
        actual_hadronic_worker_loads.push_back(
            std::move(worker_node));
      }
      if (!hadronic_process_pool) {
        actual_worker_minimum_ms = 0.;
      }
      shower_metadata["hadronic_process_pool"]
                     ["actual_worker_loads"] =
          std::move(actual_hadronic_worker_loads);
      shower_metadata["hadronic_process_pool"]
                     ["actual_generator_load_imbalance_ms"] =
          actual_worker_maximum_ms -
          actual_worker_minimum_ms;
      YAML::Node scalar_steps_by_pdg;
      for (auto const& [pdg, count] :
           hybrid_timing.scalar_steps_by_pdg) {
        scalar_steps_by_pdg[std::to_string(pdg)] =
            count;
      }
      shower_metadata["scalar_steps_by_pdg"] =
          std::move(scalar_steps_by_pdg);
      YAML::Node scalar_time_by_pdg;
      for (auto const& [pdg, time_ms] :
           hybrid_timing
               .scalar_stepper_time_by_pdg_ms) {
        scalar_time_by_pdg[std::to_string(pdg)] =
            time_ms;
      }
      shower_metadata[
          "scalar_stepper_time_by_pdg_ms"] =
          std::move(scalar_time_by_pdg);
      YAML::Node scalar_phase_timing;
      for (auto const& [pdg, phases] :
           hybrid_timing
               .scalar_phase_timing_by_pdg) {
        YAML::Node node;
        node["steps"] = phases.steps;
        node["geometry_boundary_crossings"] =
            phases.geometry_boundary_crossings;
        node["geometry_step_limits"] =
            phases.geometry_step_limits;
        node["interactions"] = phases.interactions;
        node["decays"] = phases.decays;
        node["secondary_processing_calls"] =
            phases.secondary_processing_calls;
        node["cross_section_ms"] =
            phases.cross_section_time_ms;
        node["distance_sampling_ms"] =
            phases.distance_sampling_time_ms;
        node["tracking_ms"] =
            phases.tracking_time_ms;
        node["continuous_limit_ms"] =
            phases.continuous_limit_time_ms;
        node["continuous_process_ms"] =
            phases.continuous_process_time_ms;
        node["geometry_boundary_ms"] =
            phases.geometry_boundary_time_ms;
        node["geometry_step_limit_ms"] =
            phases.geometry_step_limit_time_ms;
        node["interaction_ms"] =
            phases.interaction_time_ms;
        node["decay_ms"] = phases.decay_time_ms;
        node["secondary_processing_ms"] =
            phases.secondary_processing_time_ms;
        scalar_phase_timing[
            std::to_string(pdg)] =
            std::move(node);
      }
      shower_metadata[
          "scalar_phase_timing_by_pdg_ms"] =
          std::move(scalar_phase_timing);
      shower_metadata["hadronic_work_classifier"]
                     ["transition_energy_GeV"] =
          hybrid_timing.hadronic_work_classifier
              .transition_energy_GeV;
      shower_metadata["hadronic_work_classifier"]
                     ["energy_bins_per_octave"] =
          hybrid_timing.hadronic_work_classifier
              .energy_bins_per_octave;
      YAML::Node hadronic_work_classes;
      for (auto const& [key, statistics] :
           hybrid_timing.hadronic_work_classes) {
        YAML::Node work_class;
        work_class["model"] =
            std::string(hadronicModelClassName(key.model));
        work_class["species"] =
            std::string(hadronicSpeciesClassName(key.species));
        work_class["energy_bin"] = key.energy_bin;
        work_class["energy_lower_GeV"] =
            hadronicEnergyBinLowerGeV(
                key, hybrid_timing.hadronic_work_classifier);
        work_class["energy_upper_GeV"] =
            hadronicEnergyBinUpperGeV(
                key, hybrid_timing.hadronic_work_classifier);
        work_class["mass_number_bin"] =
            static_cast<unsigned int>(key.mass_number_bin);
        work_class["steps"] = statistics.steps;
        work_class["total_time_ms"] =
            statistics.total_time_ms;
        work_class["mean_time_ms"] =
            statistics.meanTimeMs();
        work_class["standard_deviation_ms"] =
            statistics.standardDeviationMs();
        work_class["minimum_time_ms"] =
            statistics.steps == 0
                ? 0.
                : statistics.minimum_time_ms;
        work_class["maximum_time_ms"] =
            statistics.maximum_time_ms;
        hadronic_work_classes.push_back(
            std::move(work_class));
      }
      shower_metadata["hadronic_work_classes"] =
          std::move(hadronic_work_classes);
      YAML::Node hadronic_final_state_class_nodes;
      for (auto const& [key, statistics] :
           hadronic_final_state_classes) {
        YAML::Node work_class;
        work_class["model"] =
            std::string(hadronicModelClassName(key.model));
        work_class["species"] =
            std::string(hadronicSpeciesClassName(key.species));
        work_class["energy_bin"] = key.energy_bin;
        work_class["energy_lower_GeV"] =
            hadronicEnergyBinLowerGeV(
                key, hadronic_work_classifier);
        work_class["energy_upper_GeV"] =
            hadronicEnergyBinUpperGeV(
                key, hadronic_work_classifier);
        work_class["mass_number_bin"] =
            static_cast<unsigned int>(key.mass_number_bin);
        work_class["interactions"] = statistics.steps;
        work_class["total_time_ms"] =
            statistics.total_time_ms;
        work_class["mean_time_ms"] =
            statistics.meanTimeMs();
        work_class["standard_deviation_ms"] =
            statistics.standardDeviationMs();
        work_class["minimum_time_ms"] =
            statistics.minimum_time_ms;
        work_class["maximum_time_ms"] =
            statistics.maximum_time_ms;
        hadronic_final_state_class_nodes.push_back(
            std::move(work_class));
      }
      shower_metadata["hadronic_final_state_classes"] =
          std::move(hadronic_final_state_class_nodes);
      shower_metadata["hadronic_worker_oracle"]["mode"] =
          hadronic_process_pool
              ? "disabled_actual_process_pool_active"
              : "retrospective_measured_final_state_only";
      shower_metadata["hadronic_worker_oracle"]["workers"] =
          hadronic_plan_workers;
      shower_metadata["hadronic_worker_oracle"]
                     ["target_batch_cost_ms"] =
          hadronic_plan_target_ms;
      shower_metadata["hadronic_worker_oracle"]
                     ["maximum_batch_items"] =
          hadronic_plan_max_batch;
      shower_metadata["hadronic_worker_oracle"]["interactions"] =
          hadronic_worker_sequence;
      shower_metadata["hadronic_worker_oracle"]["batches"] =
          hadronic_worker_batch_count;
      shower_metadata["hadronic_worker_oracle"]
                     ["serial_final_state_time_ms"] =
          hadronic_final_state_serial_time_ms;
      shower_metadata["hadronic_worker_oracle"]
                     ["predicted_makespan_ms"] =
          hadronic_worker_predicted_makespan_ms;
      shower_metadata["hadronic_worker_oracle"]
                     ["predicted_kernel_speedup"] =
          hadronic_worker_predicted_makespan_ms > 0.
              ? hadronic_final_state_serial_time_ms /
                    hadronic_worker_predicted_makespan_ms
              : 0.;
      shower_metadata["hadronic_worker_oracle"]
                     ["load_imbalance_ms"] =
          hadronic_worker_predicted_makespan_ms -
          hadronic_worker_minimum_load_ms;
      YAML::Node hadronic_worker_loads;
      for (auto const& assignment :
           hadronic_worker_assignments) {
        YAML::Node worker;
        worker["worker_id"] = assignment.worker_id;
        worker["estimated_load_ms"] =
            assignment.estimated_cost;
        worker["batches"] = assignment.batches.size();
        hadronic_worker_loads.push_back(std::move(worker));
      }
      shower_metadata["hadronic_worker_oracle"]["worker_loads"] =
          std::move(hadronic_worker_loads);
      shower_metadata["cpu_fallback_steps_executed"] =
          router_stats.cpu_fallback_steps_executed;
      shower_metadata["cpu_wavefront_expansion_steps_executed"] =
          router_stats
              .cpu_wavefront_expansion_steps_executed;
      shower_metadata[
          "particles_returned_for_cpu_wavefront_expansion"] =
          router_stats
              .particles_returned_for_cpu_wavefront_expansion;
      shower_metadata["small_batch_expansions"] =
          router_stats.small_batch_expansions;
      shower_metadata["stalled_boundary_gpu_flushes"] =
          router_stats.stalled_boundary_gpu_flushes;
      shower_metadata[
          "scalar_expansion_budget_gpu_flushes"] =
          router_stats
              .scalar_expansion_budget_gpu_flushes;
      shower_metadata["cpu_specified_fallback_time_ms"] =
          router_stats
              .specified_cpu_fallback_time_ms;
      shower_metadata["photon_backend_wall_time_ms"] =
          router_stats.photon_backend_wall_time_ms;
      shower_metadata["lepton_backend_wall_time_ms"] =
          router_stats.lepton_backend_wall_time_ms;
      shower_metadata["photon_host_postprocess_time_ms"] =
          router_stats.photon_host_postprocess_time_ms;
      shower_metadata["lepton_host_postprocess_time_ms"] =
          router_stats.lepton_host_postprocess_time_ms;
      YAML::Node fallbacks_by_process;
      YAML::Node fallbacks_by_process_name;
      for (auto const& [process_id, count] :
           router_stats.cpu_fallbacks_by_process) {
        fallbacks_by_process[
            std::to_string(process_id)] = count;
        fallbacks_by_process_name[
            gpuEmProcessName(process_id)] = count;
      }
      shower_metadata["cpu_fallbacks_by_process"] =
          std::move(fallbacks_by_process);
      shower_metadata["cpu_fallbacks_by_process_name"] =
          std::move(fallbacks_by_process_name);
      YAML::Node fallbacks_by_reason;
      YAML::Node fallbacks_by_reason_name;
      for (auto const& [reason, count] :
           router_stats.cpu_fallbacks_by_reason) {
        fallbacks_by_reason[
            std::to_string(reason)] = count;
        auto const typed_reason =
            static_cast<ProposalFallbackReason>(
                reason);
        fallbacks_by_reason_name[
            proposalFallbackReasonName(typed_reason)] =
            count;
      }
      shower_metadata["cpu_fallbacks_by_reason"] =
          std::move(fallbacks_by_reason);
      shower_metadata["cpu_fallbacks_by_reason_name"] =
          std::move(fallbacks_by_reason_name);
      shower_metadata["observed"] =
          router_stats.particles_observed;
      shower_metadata["escaped"] =
          router_stats.particles_escaped;
      shower_metadata["cut"] =
          router_stats.particles_cut;
      shower_metadata["weighted_deposit_GeV"] =
          sink_stats.weighted_deposited_energy_GeV;
      shower_metadata["weighted_muon_parent_productions"] =
          sink_stats.weighted_muon_parent_productions;
      shower_metadata["radio_tracks"] =
          sink_stats.radio_tracks;
      shower_metadata["radio"]["backend"] =
          gpu_radio_enabled ? "cuda" : "cpu";
      shower_metadata["radio"]["track_observer_pairs"] =
          backend_stats.radio.track_observer_pairs;
      shower_metadata["radio"]["fused_track_observer_pairs"] =
          backend_stats.radio.fused_track_observer_pairs;
      shower_metadata["radio"]["coreas_contributions"] =
          backend_stats.radio.coreas_contributions;
      shower_metadata["radio"]["zhs_contributions"] =
          backend_stats.radio.zhs_contributions;
      shower_metadata["radio"]["zhs_subtracks"] =
          backend_stats.radio.zhs_subtracks;
      shower_metadata["radio"]["zhs_subtrack_refinement"] =
          gpu_zhs_subtrack_refinement;
      shower_metadata["radio"]["deterministic"] =
          gpu_deterministic;
      shower_metadata["radio"]
                     ["fixed_point_field_limit_V_per_m"] =
          gpu_radio_field_limit;
      shower_metadata["radio"]["fixed_point_overflows"] =
          backend_stats.radio.fixed_point_overflows;
      shower_metadata["radio"]["track_diagnostics_enabled"] =
          backend_stats.radio.track_diagnostics_enabled;
      shower_metadata["radio"]["segment_count"] =
          backend_stats.radio.lepton_tracks;
      shower_metadata["radio"]["weighted_segment_count"] =
          backend_stats.radio.weighted_segment_count;
      shower_metadata["radio"]["track_length_m"] =
          backend_stats.radio.track_length_m;
      shower_metadata["radio"]["weighted_track_length_m"] =
          backend_stats.radio.weighted_track_length_m;
      shower_metadata["radio"]
                     ["electron_weighted_track_length_m"] =
          backend_stats.radio
              .electron_weighted_track_length_m;
      shower_metadata["radio"]
                     ["positron_weighted_track_length_m"] =
          backend_stats.radio
              .positron_weighted_track_length_m;
      shower_metadata["radio"]
                     ["signed_charge_weighted_track_length_m"] =
          backend_stats.radio
              .signed_charge_weighted_track_length_m;
      shower_metadata["radio"]
                     ["energy_weighted_track_length_GeV_m"] =
          backend_stats.radio
              .energy_weighted_track_length_GeV_m;
      shower_metadata["radio"]["maximum_segment_length_m"] =
          backend_stats.radio.maximum_segment_length_m;
      shower_metadata["radio"]["weighted_direction_change_rad"] =
          backend_stats.radio.weighted_direction_change_rad;
      shower_metadata["radio"]
                     ["weighted_direction_change_squared_rad2"] =
          backend_stats.radio
              .weighted_direction_change_squared_rad2;
      shower_metadata["radio"]
                     ["weighted_beta_deficit_track_length_m"] =
          backend_stats.radio
              .weighted_beta_deficit_track_length_m;
      shower_metadata["radio"]["weighted_time_residual_s"] =
          backend_stats.radio.weighted_time_residual_s;
      shower_metadata["radio"]["maximum_direction_change_rad"] =
          backend_stats.radio.maximum_direction_change_rad;
      shower_metadata["radio"]
                     ["signed_charge_weighted_direction_change"]
                     ["x"] =
          backend_stats.radio
              .signed_charge_weighted_direction_change[0];
      shower_metadata["radio"]
                     ["signed_charge_weighted_direction_change"]
                     ["y"] =
          backend_stats.radio
              .signed_charge_weighted_direction_change[1];
      shower_metadata["radio"]
                     ["signed_charge_weighted_direction_change"]
                     ["z"] =
          backend_stats.radio
              .signed_charge_weighted_direction_change[2];
      auto radio_energy_bins =
          shower_metadata["radio"]
                         ["weighted_track_length_by_kinetic_energy"];
      radio_energy_bins["units"]["upper_edge"] = "GeV";
      radio_energy_bins["units"]["weighted_track_length"] =
          "m";
      constexpr std::array<double, 14>
          radio_energy_upper_edges_GeV{
              1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2,
              5.e-2, 1.e-1, 2.e-1, 5.e-1, 1.,
              2., 5., 10., 100.};
      for (std::size_t index = 0; index < 15; ++index) {
        radio_energy_bins["upper_edge_GeV"].push_back(
            index < radio_energy_upper_edges_GeV.size()
                ? YAML::Node(
                      radio_energy_upper_edges_GeV[index])
                : YAML::Node("inf"));
        radio_energy_bins["weighted_track_length_m"]
            .push_back(
                backend_stats.radio
                    .weighted_track_length_by_kinetic_energy_m
                        [index]);
      }
      shower_metadata["radio"]["device_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.radio.device_bytes);
      shower_metadata["radio"]["host_to_device_bytes"] =
          backend_stats.radio.host_to_device_bytes;
      shower_metadata["radio"]["device_to_host_bytes"] =
          backend_stats.radio.device_to_host_bytes;
      shower_metadata["radio"]["device_time_ms"] =
          backend_stats.radio.device_time_ms;
      shower_metadata["radio"]["kernel_launch_time_ms"] =
          backend_stats.radio.kernel_time_ms;
      shower_metadata["radio"]["transfer_time_ms"] =
          backend_stats.radio.transfer_time_ms;
      shower_metadata["radio"]["input_slot_waits"] =
          backend_stats.radio.input_slot_waits;
      shower_metadata["radio"]["input_slot_host_wait_time_ms"] =
          backend_stats.radio.input_slot_host_wait_time_ms;
      shower_metadata["radio"]["track_precompute_enabled"] =
          backend_stats.radio.track_precompute_enabled;
      shower_metadata["radio"]["track_tile_size"] =
          backend_stats.radio.track_tile_size;
      shower_metadata["radio"]["observer_tile_size"] =
          backend_stats.radio.observer_tile_size;
      shower_metadata["radio"]["track_precompute_batches"] =
          backend_stats.radio.track_precompute_batches;
      shower_metadata["radio"]["track_precomputed_records"] =
          backend_stats.radio.track_precomputed_records;
      shower_metadata["radio"]["direct_projection_batches"] =
          backend_stats.radio.direct_projection_batches;
      shower_metadata["radio"]["direct_projection_records"] =
          backend_stats.radio.direct_projection_records;
      shower_metadata["radio"]["projection_tiles"] =
          backend_stats.radio.projection_tiles;
      shower_metadata["radio"]["track_workspace_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.radio.track_workspace_bytes);
      shower_metadata["radio"]["maximum_track_batch"] =
          static_cast<std::uint64_t>(
              backend_stats.radio.maximum_track_batch);
      shower_metadata["radio"]
                     ["track_precompute_device_time_ms"] =
          backend_stats.radio.track_precompute_device_time_ms;
      shower_metadata["radio"]["projection_device_time_ms"] =
          backend_stats.radio.projection_device_time_ms;
      shower_metadata["profile"]["backend"] =
          backend_stats.profile.enabled ? "cuda" : "host";
      shower_metadata["profile"]["deterministic"] =
          backend_stats.profile.deterministic;
      shower_metadata["profile"]["bins"] =
          static_cast<std::uint64_t>(
              backend_stats.profile.bins);
      shower_metadata["profile"]["steps"] =
          backend_stats.profile.steps;
      shower_metadata["profile"]["deposited_steps"] =
          backend_stats.profile.deposited_steps;
      shower_metadata["profile"]["fixed_point_overflows"] =
          backend_stats.profile.fixed_point_overflows;
      shower_metadata["profile"]["invalid_records"] =
          backend_stats.profile.invalid_records;
      shower_metadata["profile"]["device_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.profile.device_bytes);
      shower_metadata["profile"]["device_to_host_bytes"] =
          backend_stats.profile.device_to_host_bytes;
      shower_metadata["profile"]["kernel_time_ms"] =
          backend_stats.profile.kernel_time_ms;
      shower_metadata["profile"]["transfer_time_ms"] =
          backend_stats.profile.transfer_time_ms;
      shower_metadata["thinning"]["hillas_vertices"] =
          backend_stats.thinning_hillas_vertices;
      shower_metadata["thinning"]["statistical_vertices"] =
          backend_stats.thinning_statistical_vertices;
      shower_metadata["thinning"]["particles_discarded"] =
          backend_stats.thinning_particles_discarded;
      shower_metadata["processes"]["annihilation"] =
          backend_stats.annihilation_final_states;
      shower_metadata["processes"]["photon_pair"] =
          backend_stats.photon_pair_final_states;
      shower_metadata["processes"]["bremsstrahlung"] =
          backend_stats.brems_final_states;
      shower_metadata["processes"]["compton"] =
          backend_stats.compton_final_states;
      shower_metadata["processes"]["photoelectric"] =
          backend_stats.photoelectric_final_states;
      shower_metadata["processes"]["ionization"] =
          backend_stats.ionization_final_states;
      shower_metadata["processes"]["electron_pair"] =
          backend_stats.electron_pair_final_states;
      shower_metadata["epair_sampler"]["rejection_trials"] =
          backend_stats.electron_pair_rejection_trials;
      shower_metadata["epair_sampler"]["zero_weight_samples"] =
          backend_stats.electron_pair_zero_weight_samples;
      shower_metadata["epair_sampler"]["cpu_fallbacks"] =
          backend_stats.electron_pair_rejection_fallbacks;
      shower_metadata["epair_sampler"]["envelope_violations"] =
          backend_stats.electron_pair_envelope_violations;
      shower_metadata["moliere"]["trials"] =
          backend_stats.moliere_trials;
      shower_metadata["moliere"]["deflections"] =
          backend_stats.moliere_deflections;
      shower_metadata["moliere"]["zero_deflections"] =
          backend_stats.moliere_zero_deflections;
      shower_metadata["moliere"]["newton_iterations"] =
          backend_stats.moliere_newton_iterations;
      shower_metadata["moliere"]["maximum_newton_iterations"] =
          backend_stats.moliere_max_newton_iterations;
      shower_metadata["moliere"]["mean_newton_iterations"] =
          backend_stats.moliere_deflections == 0
              ? 0.
              : static_cast<double>(
                    backend_stats.moliere_newton_iterations) /
                    static_cast<double>(
                        backend_stats.moliere_deflections);
      shower_metadata["queue_overflows"] =
          backend_stats.queue_overflows;
      shower_metadata["peak_device_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.peak_device_bytes);
      shower_metadata["table_device_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.table_device_bytes);
      shower_metadata["workspace_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.physical_workspace_bytes);
      shower_metadata["maximum_resident_photon_batch"] =
          static_cast<std::uint64_t>(
              backend_stats.maximum_resident_photon_batch);
      shower_metadata["maximum_resident_lepton_batch"] =
          static_cast<std::uint64_t>(
              backend_stats.maximum_resident_lepton_batch);
      shower_metadata["cross_species"]["queue_device_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.cross_species_queue_device_bytes);
      shower_metadata["cross_species"]["enabled"] =
          gpu_config.resident_cross_species;
      shower_metadata["cross_species"]["queue_capacity_per_pid"] =
          static_cast<std::uint64_t>(
              backend_stats.cross_species_queue_capacity_per_pid);
      shower_metadata["cross_species"]["peak_pending_photons"] =
          static_cast<std::uint64_t>(
              backend_stats.peak_pending_photons);
      shower_metadata["cross_species"]["peak_pending_leptons"] =
          static_cast<std::uint64_t>(
              backend_stats.peak_pending_leptons);
      shower_metadata["cross_species"]["particles_kept_on_device"] =
          backend_stats.cross_species_particles_kept_on_device;
      shower_metadata["cross_species"]["device_to_device_bytes"] =
          backend_stats.cross_species_device_to_device_bytes;
      shower_metadata["cross_species"]["host_spills"] =
          backend_stats.cross_species_host_spills;
      shower_metadata["cross_species"]["spill_rebalances"] =
          backend_stats.cross_species_spill_rebalances;
      shower_metadata["cross_species"]["particles_spilled_to_cpu"] =
          backend_stats.cross_species_particles_spilled_to_cpu;
      shower_metadata["cross_species"]
                     ["low_energy_ordering_checks"] =
          backend_stats.cross_species_low_energy_ordering_checks;
      shower_metadata["cross_species"]
                     ["cpu_spill_steps_executed"] =
          router_stats.cpu_memory_spill_steps_executed;
      shower_metadata["wavefront_bucketing"]["key"] =
          "PID x medium_id x logarithmic_energy_bin";
      shower_metadata["wavefront_bucketing"]
                     ["energy_bins_per_octave"] = 16;
      shower_metadata["wavefront_bucketing"]["stable"] = true;
      shower_metadata["wavefront_bucketing"]
                     ["minimum_radix_sort_size"] =
          static_cast<std::uint64_t>(
              corsika::gpu::em::detail::
                  MinimumWavefrontRadixSortSize);
      shower_metadata["wavefront_bucketing"]["batches"] =
          backend_stats.wavefront_bucketing_batches;
      shower_metadata["wavefront_bucketing"]["particles"] =
          backend_stats.wavefront_bucketing_particles;
      shower_metadata["wavefront_bucketing"]["small_batches"] =
          backend_stats.wavefront_bucketing_small_batches;
      shower_metadata["wavefront_bucketing"]["small_particles"] =
          backend_stats.wavefront_bucketing_small_particles;
      shower_metadata["lepton_transport"]["interaction_candidates"] =
          backend_stats.lepton_transport_interaction_candidates;
      shower_metadata["lepton_transport"]["continuous_steps"] =
          backend_stats.lepton_transport_continuous_steps;
      shower_metadata["lepton_transport"]["particle_cuts"] =
          backend_stats.lepton_transport_cuts;
      shower_metadata["lepton_transport"]["layer_boundaries"] =
          backend_stats.lepton_transport_boundaries;
      shower_metadata["lepton_transport"]["observations"] =
          backend_stats.lepton_transport_observations;
      shower_metadata["lepton_transport"]["escapes"] =
          backend_stats.lepton_transport_escapes;
      shower_metadata["lepton_transport"]["magnetic_steps"] =
          backend_stats.lepton_transport_magnetic_steps;
      shower_metadata["lepton_transport"]["decay_candidates"] =
          backend_stats.lepton_transport_decay_candidates;
      shower_metadata["cross_species"]["final_pending_photons"] =
          static_cast<std::uint64_t>(
              backend.pendingPhotonCount());
      shower_metadata["cross_species"]["final_pending_leptons"] =
          static_cast<std::uint64_t>(
              backend.pendingLeptonCount());
      shower_metadata["host_to_device_bytes"] =
          backend_stats.physical_host_to_device_bytes;
      shower_metadata["device_to_host_bytes"] =
          backend_stats.physical_device_to_host_bytes;
      shower_metadata["pipeline_control"]
                     ["photon_selection_transport_summary_fusions"] =
          backend_stats
              .photon_selection_transport_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["lepton_selection_transport_summary_fusions"] =
          backend_stats
              .lepton_selection_transport_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["photon_transport_final_state_summary_fusions"] =
          backend_stats
              .photon_transport_final_state_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["lepton_transport_vertex_summary_fusions"] =
          backend_stats
              .lepton_transport_vertex_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["lepton_vertex_final_state_summary_fusions"] =
          backend_stats
              .lepton_vertex_final_state_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["photon_final_state_endpoint_summary_fusions"] =
          backend_stats
              .photon_final_state_endpoint_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["lepton_final_state_endpoint_summary_fusions"] =
          backend_stats
              .lepton_final_state_endpoint_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["host_synchronizations_eliminated"] =
          backend_stats
              .pipeline_host_synchronizations_eliminated;
      shower_metadata["pipeline_control"]
                     ["device_to_host_bytes_eliminated"] =
          backend_stats
              .pipeline_device_to_host_bytes_eliminated;
      shower_metadata["lepton_pipeline_timing"]["enabled"] =
          backend_stats.lepton_pipeline_timing.enabled;
      shower_metadata["lepton_pipeline_timing"]["wavefronts"] =
          backend_stats.lepton_pipeline_timing.wavefronts;
      shower_metadata["lepton_pipeline_timing"]["selection_ms"] =
          backend_stats.lepton_pipeline_timing.selection_ms;
      shower_metadata["lepton_pipeline_timing"]["transport_ms"] =
          backend_stats.lepton_pipeline_timing.transport_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["transport_physics_ms"] =
          backend_stats.lepton_pipeline_timing
              .transport_physics_ms;
      shower_metadata["lepton_pipeline_timing"]["moliere_ms"] =
          backend_stats.lepton_pipeline_timing.moliere_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["transport_control_ms"] =
          backend_stats.lepton_pipeline_timing
              .transport_control_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["transport_compaction_ms"] =
          backend_stats.lepton_pipeline_timing
              .transport_compaction_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["interaction_extraction_ms"] =
          backend_stats.lepton_pipeline_timing
              .interaction_extraction_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["vertex_selection_ms"] =
          backend_stats.lepton_pipeline_timing
              .vertex_selection_ms;
      shower_metadata["lepton_pipeline_timing"]["final_state_ms"] =
          backend_stats.lepton_pipeline_timing.final_state_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["final_state_classification_ms"] =
          backend_stats.lepton_pipeline_timing
              .final_state_classification_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["final_state_scan_ms"] =
          backend_stats.lepton_pipeline_timing
              .final_state_scan_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["final_state_summary_ms"] =
          backend_stats.lepton_pipeline_timing
              .final_state_summary_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["final_state_write_ms"] =
          backend_stats.lepton_pipeline_timing
              .final_state_write_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["endpoint_compaction_ms"] =
          backend_stats.lepton_pipeline_timing
              .endpoint_compaction_ms;
      shower_metadata["lepton_pipeline_timing"]["post_endpoint_ms"] =
          backend_stats.lepton_pipeline_timing.post_endpoint_ms;
      shower_metadata["kernel_time_ms"] =
          backend_stats.kernel_time_ms;
      shower_metadata["transfer_time_ms"] =
          backend_stats.transfer_time_ms;
      shower_metadata["timing_schema_version"] = 2;
      shower_metadata["timing_semantics"]["kernel_time_ms"] =
          "sum of CUDA-event durations across streams; overlapping streams "
          "must not be added to shower wall time";
      shower_metadata["timing_semantics"]["transfer_time_ms"] =
          "legacy host API wall time around selected synchronous CUDA copies";
      shower_metadata["transfer_timing"]
                     ["device_event_timing_enabled"] =
          backend_stats.transfer_timing.device_event_timing_enabled;
      shower_metadata["transfer_timing"]["operations"] =
          backend_stats.transfer_timing.operations;
      shower_metadata["transfer_timing"]
                     ["host_to_device_operations"] =
          backend_stats.transfer_timing.host_to_device_operations;
      shower_metadata["transfer_timing"]
                     ["device_to_host_operations"] =
          backend_stats.transfer_timing.device_to_host_operations;
      shower_metadata["transfer_timing"]
                     ["device_to_device_operations"] =
          backend_stats.transfer_timing.device_to_device_operations;
      shower_metadata["transfer_timing"]["host_api_time_ms"] =
          backend_stats.transfer_timing.host_api_time_ms;
      shower_metadata["transfer_timing"]["device_copy_time_ms"] =
          backend_stats.transfer_timing.device_copy_time_ms;
      shower_metadata["transfer_timing"]["host_wait_upper_bound_ms"] =
          backend_stats.transfer_timing.host_wait_upper_bound_ms;
      shower_metadata["synchronization_timing"]
                     ["physical_pipeline_waits"] =
          backend_stats.synchronization_timing.physical_pipeline_waits;
      shower_metadata["synchronization_timing"]
                     ["physical_pipeline_wait_time_ms"] =
          backend_stats.synchronization_timing
              .physical_pipeline_wait_time_ms;
      shower_metadata["synchronization_timing"]
                     ["profile_input_waits"] =
          backend_stats.synchronization_timing.profile_input_waits;
      shower_metadata["synchronization_timing"]
                     ["profile_input_wait_time_ms"] =
          backend_stats.synchronization_timing.profile_input_wait_time_ms;
      auto const particle_cut_statistics =
          subtractStatistics(
              cut.statistics(),
              particle_cut_statistics_before);
      auto const observation_statistics =
          subtractStatistics(
              observationLevel.statistics(),
              observation_statistics_before);
      auto const low_energy_ledger =
          subtractStatistics(
              leIntCounted.getEnergyLedgerStatistics(),
              low_energy_ledger_before);
      auto const high_energy_ledger =
          subtractStatistics(
              heCounted.getEnergyLedgerStatistics(),
              high_energy_ledger_before);
      auto const photo_hadronic_ledger =
          subtractStatistics(
              emCascade.energyLedgerStatistics(),
              photo_hadronic_ledger_before);
      auto scalar_em_steps = std::uint64_t{0};
      for (auto const& [pdg, count] :
           hybrid_timing.scalar_steps_by_pdg) {
        if (pdg == 22 || pdg == 11 || pdg == -11) {
          scalar_em_steps += count;
        }
      }
      auto const ledger_initial_GeV =
          primaryTotalEnergy / 1_GeV;
      auto const ledger_scalar_atomic_electron_interactions =
          emCascade.atomicElectronTargetInteractions() -
          scalar_atomic_electron_interactions_before;
      auto const ledger_scalar_atomic_electron_rest_input_GeV =
          emCascade.weightedAtomicElectronRestMassInputGeV() -
          scalar_atomic_electron_rest_input_before_GeV;
      auto const ledger_hadronic_target_input_GeV =
          low_energy_ledger
                  .weighted_effective_target_total_energy_GeV +
          high_energy_ledger
                  .weighted_target_total_energy_GeV +
          photo_hadronic_ledger
                  .weighted_target_total_energy_GeV;
      auto const ledger_generator_residual_GeV =
          low_energy_ledger
                  .weighted_isotope_corrected_energy_residual_GeV +
          high_energy_ledger
                  .weighted_energy_residual_GeV +
          photo_hadronic_ledger
                  .weighted_energy_residual_GeV;
      auto const ledger_deposited_and_cut_kinetic_GeV =
          dEdX.getEnergyLost() / 1_GeV;
      auto const ledger_invisible_cut_kinetic_GeV =
          particle_cut_statistics
              .weighted_invisible_kinetic_energy_GeV;
      auto const ledger_visible_deposit_GeV =
          ledger_deposited_and_cut_kinetic_GeV -
          ledger_invisible_cut_kinetic_GeV;
      auto const ledger_scalar_cut_rest_GeV =
          particle_cut_statistics
              .weighted_rest_mass_energy_GeV;
      auto const ledger_source_GeV =
          ledger_initial_GeV +
          router_stats.medium_rest_mass_input_GeV +
          ledger_scalar_atomic_electron_rest_input_GeV +
          ledger_hadronic_target_input_GeV;
      auto const ledger_transport_terminal_GeV =
          ledger_deposited_and_cut_kinetic_GeV +
          router_stats.cut_rest_mass_energy_GeV +
          ledger_scalar_cut_rest_GeV +
          observation_statistics.weighted_total_energy_GeV +
          router_stats.escaped_total_energy_GeV;
      auto const ledger_terminal_GeV =
          ledger_transport_terminal_GeV +
          ledger_generator_residual_GeV;
      auto const ledger_residual_GeV =
          ledger_source_GeV - ledger_terminal_GeV;
      auto const ledger_relative_error =
          std::abs(ledger_residual_GeV) /
          std::max(
              ledger_source_GeV,
              std::numeric_limits<double>::min());
      auto const thinning_was_inactive =
          backend_stats.thinning_hillas_vertices == 0 &&
          backend_stats.thinning_statistical_vertices == 0 &&
          backend_stats.thinning_particles_discarded == 0;
      auto const low_energy_interaction_count =
          leIntCounted.getCount() -
          low_energy_hadronic_interactions_before;
      auto const high_energy_interaction_count =
          heCounted.getCount() -
          high_energy_hadronic_interactions_before;
      auto const photo_hadronic_interaction_count =
          photo_hadronic_sophia +
          photo_hadronic_threshold_fallback +
          photo_hadronic_preferred +
          photo_hadronic_fallback;
      auto const every_hadronic_vertex_was_audited =
          low_energy_ledger.audited_interactions +
                  low_energy_ledger.deferred_interactions ==
              low_energy_interaction_count &&
          high_energy_ledger.audited_interactions +
                  high_energy_ledger.deferred_interactions ==
              high_energy_interaction_count &&
          photo_hadronic_ledger.interactions ==
              photo_hadronic_interaction_count;
      auto const ledger_complete_coverage =
          is_em(beamCode) && thinning_was_inactive &&
          router_stats
                  .particles_returned_for_cpu_memory_spill ==
              0 &&
          low_energy_ledger.deferred_interactions == 0 &&
          high_energy_ledger.deferred_interactions == 0 &&
          every_hadronic_vertex_was_audited &&
          observation_statistics.particles ==
              router_stats.particles_observed;
      auto const ledger_generator_residual_fraction =
          std::abs(ledger_generator_residual_GeV) /
          std::max(
              ledger_source_GeV,
              std::numeric_limits<double>::min());
      auto ledger =
          shower_metadata["energy_ledger"];
      ledger["definition"] =
          "initial_total + medium_electron_rest + hadronic_target_total = "
          "visible_deposit + invisible_cut_kinetic + gpu_cut_rest + "
          "scalar_cut_rest + observed_total + escaped_total + "
          "untracked_interaction_local_energy";
      ledger["complete_coverage"] =
          ledger_complete_coverage;
      ledger["scalar_em_steps"] = scalar_em_steps;
      ledger["thinning_was_inactive"] =
          thinning_was_inactive;
      ledger["every_hadronic_vertex_was_audited"] =
          every_hadronic_vertex_was_audited;
      ledger["initial_total_GeV"] =
          ledger_initial_GeV;
      ledger["medium_electron_rest_mass_input_GeV"] =
          router_stats.medium_rest_mass_input_GeV +
          ledger_scalar_atomic_electron_rest_input_GeV;
      ledger["medium_rest_mass_input_GeV"] =
          router_stats.medium_rest_mass_input_GeV +
          ledger_scalar_atomic_electron_rest_input_GeV;
      ledger["gpu_medium_electron_rest_mass_input_GeV"] =
          router_stats.medium_rest_mass_input_GeV;
      ledger["scalar_medium_electron_rest_mass_input_GeV"] =
          ledger_scalar_atomic_electron_rest_input_GeV;
      ledger["scalar_atomic_electron_target_interactions"] =
          ledger_scalar_atomic_electron_interactions;
      ledger["hadronic_target_total_energy_input_GeV"] =
          ledger_hadronic_target_input_GeV;
      ledger["deposited_and_cut_kinetic_GeV"] =
          ledger_deposited_and_cut_kinetic_GeV;
      ledger["deposited_GeV"] =
          ledger_deposited_and_cut_kinetic_GeV;
      ledger["visible_deposited_GeV"] =
          ledger_visible_deposit_GeV;
      ledger["invisible_cut_kinetic_energy_GeV"] =
          ledger_invisible_cut_kinetic_GeV;
      ledger["gpu_cut_rest_mass_energy_GeV"] =
          router_stats.cut_rest_mass_energy_GeV;
      ledger["scalar_cut_rest_mass_energy_GeV"] =
          ledger_scalar_cut_rest_GeV;
      ledger["cut_rest_mass_energy_GeV"] =
          router_stats.cut_rest_mass_energy_GeV +
          ledger_scalar_cut_rest_GeV;
      ledger["observed_total_energy_GeV"] =
          observation_statistics.weighted_total_energy_GeV;
      ledger["escaped_total_energy_GeV"] =
          router_stats.escaped_total_energy_GeV;
      ledger["interaction_generator_residual_GeV"] =
          ledger_generator_residual_GeV;
      ledger["untracked_interaction_local_energy_GeV"] =
          ledger_generator_residual_GeV;
      ledger["interaction_generator_residual_fraction"] =
          ledger_generator_residual_fraction;
      ledger["untracked_interaction_local_energy_fraction"] =
          ledger_generator_residual_fraction;
      ledger["transport_terminal_before_generator_residual_GeV"] =
          ledger_transport_terminal_GeV;
      ledger["source_GeV"] = ledger_source_GeV;
      ledger["terminal_GeV"] =
          ledger_terminal_GeV;
      ledger["residual_GeV"] =
          ledger_residual_GeV;
      ledger["relative_closure_error"] =
          ledger_relative_error;
      ledger["acceptance_tolerance"] = 1.e-4;
      ledger["accepted"] =
          ledger_complete_coverage &&
          ledger_relative_error <= 1.e-4;
      auto particle_cut_ledger = ledger["scalar_particle_cut"];
      particle_cut_ledger["particles"] =
          particle_cut_statistics.particles;
      particle_cut_ledger["invisible_particles"] =
          particle_cut_statistics.invisible_particles;
      particle_cut_ledger["weighted_kinetic_energy_GeV"] =
          particle_cut_statistics
              .weighted_kinetic_energy_GeV;
      particle_cut_ledger["weighted_rest_mass_energy_GeV"] =
          particle_cut_statistics
              .weighted_rest_mass_energy_GeV;
      YAML::Node particle_cut_species;
      for (auto const& [pdg, statistics] :
           particle_cut_statistics.by_pdg) {
        auto species = particle_cut_species[
            std::to_string(pdg)];
        species["particles"] = statistics.particles;
        species["weighted_kinetic_energy_GeV"] =
            statistics.weighted_kinetic_energy_GeV;
        species["weighted_rest_mass_energy_GeV"] =
            statistics.weighted_rest_mass_energy_GeV;
      }
      particle_cut_ledger["by_pdg"] =
          std::move(particle_cut_species);
      auto write_interaction_ledger = [](
          YAML::Node node,
          auto const& statistics) {
        node["audited_interactions"] =
            statistics.audited_interactions;
        node["deferred_interactions"] =
            statistics.deferred_interactions;
        node["weighted_projectile_total_energy_GeV"] =
            statistics
                .weighted_projectile_total_energy_GeV;
        node["weighted_target_total_energy_GeV"] =
            statistics.weighted_target_total_energy_GeV;
        node["weighted_effective_target_total_energy_GeV"] =
            statistics
                .weighted_effective_target_total_energy_GeV;
        node["weighted_target_isotope_correction_GeV"] =
            statistics
                .weighted_target_isotope_correction_GeV;
        node["target_isotope_adjusted_interactions"] =
            statistics.target_isotope_adjusted_interactions;
        node["weighted_secondary_total_energy_GeV"] =
            statistics
                .weighted_secondary_total_energy_GeV;
        node["weighted_energy_residual_GeV"] =
            statistics.weighted_energy_residual_GeV;
        node["weighted_absolute_energy_residual_GeV"] =
            statistics
                .weighted_absolute_energy_residual_GeV;
        node["maximum_absolute_residual_GeV"] =
            statistics.maximum_absolute_residual_GeV;
        node["weighted_isotope_corrected_energy_residual_GeV"] =
            statistics
                .weighted_isotope_corrected_energy_residual_GeV;
        node["weighted_absolute_isotope_corrected_energy_residual_GeV"] =
            statistics
                .weighted_absolute_isotope_corrected_energy_residual_GeV;
        node["maximum_absolute_isotope_corrected_residual_GeV"] =
            statistics
                .maximum_absolute_isotope_corrected_residual_GeV;
        node["negative_isotope_corrected_residual_interactions"] =
            statistics
                .negative_isotope_corrected_residual_interactions;
        node["maximum_negative_isotope_corrected_residual_GeV"] =
            statistics
                .maximum_negative_isotope_corrected_residual_GeV;
        node["spacelike_isotope_corrected_residual_interactions"] =
            statistics
                .spacelike_isotope_corrected_residual_interactions;
        node["maximum_isotope_corrected_spacelike_excess_GeV"] =
            statistics
                .maximum_isotope_corrected_spacelike_excess_GeV;
        node["minimum_isotope_corrected_residual_mass_squared_GeV2"] =
            statistics
                .minimum_isotope_corrected_residual_mass_squared_GeV2;
        node["maximum_target_total_energy_GeV"] =
            statistics.maximum_target_total_energy_GeV;
        node["weighted_momentum_residual_norm_GeV"] =
            statistics
                .weighted_momentum_residual_norm_GeV;
        node["maximum_momentum_residual_norm_GeV"] =
            statistics
                .maximum_momentum_residual_norm_GeV;
        node["negative_energy_residual_interactions"] =
            statistics
                .negative_energy_residual_interactions;
        node["spacelike_residual_interactions"] =
            statistics.spacelike_residual_interactions;
        node["maximum_negative_energy_residual_GeV"] =
            statistics
                .maximum_negative_energy_residual_GeV;
        node["maximum_negative_projectile_pdg"] =
            statistics.maximum_negative_projectile_pdg;
        node["maximum_negative_target_pdg"] =
            statistics.maximum_negative_target_pdg;
        node["maximum_negative_projectile_total_energy_GeV"] =
            statistics
                .maximum_negative_projectile_total_energy_GeV;
        node["maximum_negative_target_total_energy_GeV"] =
            statistics
                .maximum_negative_target_total_energy_GeV;
        node["maximum_negative_secondary_total_energy_GeV"] =
            statistics
                .maximum_negative_secondary_total_energy_GeV;
        node["maximum_negative_secondary_count"] =
            statistics.maximum_negative_secondary_count;
        node["maximum_negative_baryon_number_residual"] =
            statistics
                .maximum_negative_baryon_number_residual;
        node["maximum_negative_charge_number_residual"] =
            statistics
                .maximum_negative_charge_number_residual;
        YAML::Node negative_secondary_pdgs;
        for (auto const pdg :
             statistics.maximum_negative_secondary_pdgs) {
          negative_secondary_pdgs.push_back(pdg);
        }
        node["maximum_negative_secondary_pdgs"] =
            std::move(negative_secondary_pdgs);
        node["maximum_spacelike_excess_GeV"] =
            statistics.maximum_spacelike_excess_GeV;
        node["minimum_residual_mass_squared_GeV2"] =
            statistics
                .minimum_residual_mass_squared_GeV2;
        node["maximum_timelike_residual_mass_GeV"] =
            statistics
                .maximum_timelike_residual_mass_GeV;
        node["maximum_positive_residual_beta"] =
            statistics.maximum_positive_residual_beta;
      };
      write_interaction_ledger(
          ledger["hadronic_interactions"]["low_energy"],
          low_energy_ledger);
      write_interaction_ledger(
          ledger["hadronic_interactions"]["high_energy"],
          high_energy_ledger);
      auto photo_ledger =
          ledger["hadronic_interactions"]["photo_hadronic"];
      photo_ledger["audited_interactions"] =
          photo_hadronic_ledger.interactions;
      photo_ledger["low_energy_interactions"] =
          photo_hadronic_ledger.low_energy_interactions;
      photo_ledger["high_energy_interactions"] =
          photo_hadronic_ledger.high_energy_interactions;
      photo_ledger["weighted_projectile_total_energy_GeV"] =
          photo_hadronic_ledger
              .weighted_projectile_total_energy_GeV;
      photo_ledger["weighted_target_total_energy_GeV"] =
          photo_hadronic_ledger
              .weighted_target_total_energy_GeV;
      photo_ledger["weighted_secondary_total_energy_GeV"] =
          photo_hadronic_ledger
              .weighted_secondary_total_energy_GeV;
      photo_ledger["weighted_energy_residual_GeV"] =
          photo_hadronic_ledger
              .weighted_energy_residual_GeV;
      photo_ledger["maximum_absolute_residual_GeV"] =
          photo_hadronic_ledger
              .maximum_absolute_residual_GeV;
      photo_ledger["weighted_absolute_energy_residual_GeV"] =
          photo_hadronic_ledger
              .weighted_absolute_energy_residual_GeV;
      photo_ledger["maximum_target_total_energy_GeV"] =
          photo_hadronic_ledger
              .maximum_target_total_energy_GeV;
      photo_ledger["weighted_momentum_residual_norm_GeV"] =
          photo_hadronic_ledger
              .weighted_momentum_residual_norm_GeV;
      photo_ledger["maximum_momentum_residual_norm_GeV"] =
          photo_hadronic_ledger
              .maximum_momentum_residual_norm_GeV;
      photo_ledger["negative_energy_residual_interactions"] =
          photo_hadronic_ledger
              .negative_energy_residual_interactions;
      photo_ledger["spacelike_residual_interactions"] =
          photo_hadronic_ledger
              .spacelike_residual_interactions;
      photo_ledger["maximum_negative_energy_residual_GeV"] =
          photo_hadronic_ledger
              .maximum_negative_energy_residual_GeV;
      photo_ledger["maximum_negative_projectile_pdg"] =
          photo_hadronic_ledger
              .maximum_negative_projectile_pdg;
      photo_ledger["maximum_negative_target_pdg"] =
          photo_hadronic_ledger.maximum_negative_target_pdg;
      photo_ledger["maximum_negative_projectile_total_energy_GeV"] =
          photo_hadronic_ledger
              .maximum_negative_projectile_total_energy_GeV;
      photo_ledger["maximum_negative_target_total_energy_GeV"] =
          photo_hadronic_ledger
              .maximum_negative_target_total_energy_GeV;
      photo_ledger["maximum_negative_secondary_total_energy_GeV"] =
          photo_hadronic_ledger
              .maximum_negative_secondary_total_energy_GeV;
      photo_ledger["maximum_negative_secondary_count"] =
          photo_hadronic_ledger
              .maximum_negative_secondary_count;
      photo_ledger["maximum_negative_baryon_number_residual"] =
          photo_hadronic_ledger
              .maximum_negative_baryon_number_residual;
      photo_ledger["maximum_negative_charge_number_residual"] =
          photo_hadronic_ledger
              .maximum_negative_charge_number_residual;
      YAML::Node photo_negative_secondary_pdgs;
      for (auto const pdg :
           photo_hadronic_ledger.maximum_negative_secondary_pdgs) {
        photo_negative_secondary_pdgs.push_back(pdg);
      }
      photo_ledger["maximum_negative_secondary_pdgs"] =
          std::move(photo_negative_secondary_pdgs);
      photo_ledger["maximum_spacelike_excess_GeV"] =
          photo_hadronic_ledger
              .maximum_spacelike_excess_GeV;
      photo_ledger["minimum_residual_mass_squared_GeV2"] =
          photo_hadronic_ledger
              .minimum_residual_mass_squared_GeV2;
      photo_ledger["maximum_timelike_residual_mass_GeV"] =
          photo_hadronic_ledger
              .maximum_timelike_residual_mass_GeV;
      photo_ledger["maximum_positive_residual_beta"] =
          photo_hadronic_ledger
              .maximum_positive_residual_beta;
      CORSIKA_LOG_INFO(
          "whole-event energy ledger: complete={}, source={} GeV, "
          "terminal={} GeV, generator residual={} GeV, "
          "closure residual={} GeV, relative error={}",
          ledger_complete_coverage, ledger_source_GeV,
          ledger_terminal_GeV,
          ledger_generator_residual_GeV,
          ledger_residual_GeV,
          ledger_relative_error);
      if (ledger_complete_coverage &&
          ledger_relative_error > 1.e-4) {
        CORSIKA_LOG_ERROR(
            "strict energy-ledger components: initial={} GeV, "
            "medium rest input={} GeV, hadronic target input={} GeV, "
            "deposit+cut kinetic={} GeV, GPU cut rest={} GeV, "
            "scalar cut rest={} GeV, observed={} GeV, escaped={} GeV, "
            "generator residual={} GeV",
            ledger_initial_GeV,
            router_stats.medium_rest_mass_input_GeV +
                ledger_scalar_atomic_electron_rest_input_GeV,
            ledger_hadronic_target_input_GeV,
            ledger_deposited_and_cut_kinetic_GeV,
            router_stats.cut_rest_mass_energy_GeV,
            ledger_scalar_cut_rest_GeV,
            observation_statistics.weighted_total_energy_GeV,
            router_stats.escaped_total_energy_GeV,
            ledger_generator_residual_GeV);
        if (gpu_run_output) {
          gpu_run_output->recordIncomplete(
              output_shower_id,
              "CUDA EM strict energy closure exceeded 1e-4",
              std::move(shower_metadata));
        }
        throw std::runtime_error(
            "CUDA EM strict energy closure exceeded 1e-4");
      }
      if (gpu_run_output) {
        gpu_run_output->recordComplete(
            output_shower_id,
            std::move(shower_metadata));
      }
      } catch (std::exception const& error) {
        if (gpu_run_output) {
          gpu_run_output->recordIncomplete(
              output_shower_id, error.what());
        }
        if (output.showerInProgress()) {
          output.endOfShower();
        }
        output.endOfLibrary();
        CORSIKA_LOG_CRITICAL(
            "CUDA EM shower {} aborted; output is marked incomplete: {}",
            i_shower, error.what());
        return EXIT_FAILURE;
      }
#endif
    }

    HEPEnergyType const Efinal =
        dEdX.getEnergyLost() + observationLevel.getEnergyGround();

    CORSIKA_LOG_INFO(
        "total energy budget (GeV): {} (dEdX={} ground={}), "
        "relative difference (%): {}",
        Efinal / 1_GeV, dEdX.getEnergyLost() / 1_GeV,
        observationLevel.getEnergyGround() / 1_GeV,
        (Efinal / primaryTotalEnergy - 1) * 100);

    if (!disable_interaction_hists) {
      CORSIKA_LOG_INFO("Saving interaction histograms");
      auto const hists = heCounted.getHistogram() + leIntCounted.getHistogram();

      // directory for output of interaction histograms
      string const outdir(app["--filename"]->as<std::string>() + "/interaction_hist");
      boost::filesystem::create_directories(outdir);

      string const labHist_file = outdir + "/inthist_lab_" + to_string(i_shower) + ".npz";
      string const cMSHist_file = outdir + "/inthist_cms_" + to_string(i_shower) + ".npz";
      save_hist(hists.labHist(), labHist_file, true);
      save_hist(hists.CMSHist(), cMSHist_file, true);
    }
  }

  // and finalize the output on disk
  output.endOfLibrary();
  validation::CudaReplayTrace::instance().close();
  validation::CudaDecisionReplayTape::instance().close();

  return EXIT_SUCCESS;
}
