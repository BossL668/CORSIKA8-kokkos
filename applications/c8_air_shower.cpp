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
#include <corsika/framework/core/HadronicProcessPool.hpp>
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
#include <corsika/output/SimulationTiming.hpp>

#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/GeomagneticModel.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/LayeredSphericalAtmosphereBuilder.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/media/ShowerAxis.hpp>
#include <corsika/media/UniformMagneticField.hpp>

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
#include <corsika/modules/radio/propagators/TabulatedFlatAtmospherePropagator.hpp>

#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupTrajectory.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>
#include <corsika/validation/CudaDecisionReplayTape.hpp>
#include <corsika/validation/CudaReplayTrace.hpp>

#include <boost/filesystem.hpp>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

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

#include "detail/air_shower_kokkos/GpuCliOptions.hpp"
#ifdef C8_EXPERIMENTAL_STATIC_MULTIGPU
#include "detail/air_shower_multigpu/Frontier.hpp"
#endif
#if defined(CORSIKA8_WITH_KOKKOS_EM)
#include "detail/air_shower_kokkos/KokkosRunSession.hpp"
#include "detail/air_shower_kokkos/KokkosAirShowerRunner.hpp"
#endif

using namespace corsika;
using namespace std;

using EnvironmentInterface =
    IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
using EnvType = Environment<EnvironmentInterface>;
using StackType = setup::HybridStack<EnvType>;
using TrackingType = setup::Tracking;
using Particle = StackType::particle_type;

//
// This is the main example script which runs EAS with fairly standard settings
// w.r.t. what was implemented in CORSIKA 7. Users may want to change some of the
// specifics (observation altitude, magnetic field, energy cuts, etc.), but this
// example is the most physics-complete one and should be used for full simulations
// of particle cascades in air
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
using MyExtraEnv =
    GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<T>>>;

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

namespace {

  using corsika::applications::air_shower::GpuCliOptions;

  void addGpuCliOptions(CLI::App& app, GpuCliOptions& options) {
    app.add_option("--em-backend", options.em_backend,
                   "Electromagnetic transport backend: proposal or kokkos")
        ->check(CLI::IsMember({"proposal", "kokkos"}))
        ->group("GPU EM");
    app.add_option("--gpu-min-batch", options.gpu_min_batch,
                   "Minimum Kokkos execution batch; smaller fronts take one "
                   "scalar expansion step")
        ->check(CLI::PositiveNumber)->group("GPU EM");
    app.add_option(
           "--gpu-resident-batch-limit",
           options.gpu_resident_batch_limit,
           "Optional common resident-wavefront checkpoint for deterministic "
           "replay; zero selects the memory-budgeted automatic limit")
        ->check(CLI::NonNegativeNumber)->group("GPU EM");
    app.add_option(
           "--gpu-memory-fraction", options.gpu_memory_fraction,
           "Fraction of currently free device memory available to the Kokkos backend")
        ->check(CLI::Range(0.01, 1.0))->group("GPU EM");
    app.add_option("--gpu-physics-source", options.gpu_physics_source,
                   "Physics source: proposal-native")
        ->check(CLI::IsMember({"proposal-native"}))
        ->group("GPU EM");
    app.add_option(
           "--gpu-aux-cache-dir", options.gpu_aux_cache_dir,
           "XDG-compatible .c8emaux cache directory for proposal-native")
        ->group("GPU EM");
    app.add_option("--gpu-deterministic", options.gpu_deterministic,
                   "Use history-keyed deterministic random numbers")
        ->group("GPU EM");
    app.add_flag(
           "--gpu-detailed-stage-timing", options.gpu_detailed_stage_timing,
           "Record per-stage CUDA event timings for the fused lepton "
           "pipeline (profiling only)")
        ->group("GPU EM");
    app.add_flag("--gpu-full-step-records", options.gpu_full_step_records,
                 "Disable compact GPU profile projection and return full "
                 "transport records (validation/debug only)")
        ->group("GPU EM");
    app.add_option(
           "--gpu-resident-cross-species", options.gpu_resident_cross_species,
           "Keep photon-to-lepton and lepton-to-photon secondaries in "
           "persistent device queues")
        ->group("GPU EM");
    app.add_option("--kokkos-execution", options.kokkos_execution,
                   "Execution space in this binary: cuda, openmp, cuda-openmp (GPU primary) or openmp-cuda (CPU primary) for the "
                   "experimental dual build; empty selects the build default")
        ->check(CLI::IsMember({"cuda", "openmp", "hip", "sycl", "cuda-openmp", "openmp-cuda"}))
        ->group("GPU EM");
    app.add_option("--kokkos-cooperative-policy", options.kokkos_cooperative_policy,
                   "Cooperative queue scheduling: legacy or experimental adaptive")
        ->check(CLI::IsMember({"legacy", "adaptive"}))->group("Kokkos");
    app.add_option("--kokkos-num-threads", options.kokkos_num_threads,
                   "OpenMP thread count for an OpenMP-only Kokkos build; zero uses the runtime default")
        ->check(CLI::NonNegativeNumber)->group("Kokkos");
    app.add_option("--kokkos-device", options.kokkos_device,
                   "Device index for a GPU-only Kokkos build")
        ->check(CLI::NonNegativeNumber)->group("Kokkos");
    app.add_option("--kokkos-tuning-cache", options.kokkos_tuning_cache,
                   "Device-specific Kokkos tuning cache")
        ->group("Kokkos");
    app.add_flag("--kokkos-require-tuning", options.kokkos_require_tuning,
                 "Fail unless an exactly matching Kokkos tuning cache exists")
        ->group("Kokkos");
    app.add_option("--radio-backend", options.radio_backend,
                   "Radio projection backend: cpu or kokkos")
        ->check(CLI::IsMember({"cpu", "kokkos"}))->group("Radio");
    app.add_option(
           "--gpu-radio-field-limit", options.gpu_radio_field_limit,
           "Checked fixed-point waveform range in V/m for deterministic "
           "CUDA CoREAS/ZHS accumulation")
        ->check(CLI::PositiveNumber)->group("Radio");
    app.add_flag(
           "--gpu-radio-track-diagnostics",
           options.gpu_radio_track_diagnostics,
           "Collect scalar-compatible electron/positron track diagnostics "
           "on the GPU (validation only; adds reduction overhead)")
        ->group("Radio");
    app.add_option(
           "--radio-sampling-rate-ghz", options.radio_sampling_rate_GHz,
           "Time-domain radio sampling rate in GHz. Use at least 10 GHz "
           "(0.1 ns bins) for a 50--350 MHz CoREAS/ZHS comparison")
        ->check(CLI::Range(0.1, 1000.))->group("Radio");
    app.add_option("--radio-window-duration-ns",
                   options.radio_window_duration_ns,
                   "Time-domain radio observer window duration in ns")
        ->check(CLI::Range(1., 1.e6))->group("Radio");
    app.add_option(
           "--radio-pretrigger-ns", options.radio_pretrigger_ns,
           "Start antenna-file radio windows this many ns before the "
           "geometrical direct-arrival time")
        ->check(CLI::Range(0., 1.e6))->group("Radio");
    app.add_option(
           "--cuda-replay-trace", options.cuda_replay_trace,
           "Write a process-level CSV trace for scalar/CUDA replay comparison")
        ->group("GPU EM");
    app.add_option(
           "--cuda-replay-tape-out", options.cuda_replay_tape_out,
           "Record every scalar transport segment and the exact radio "
           "observer snapshot for offline CUDA decision replay")
        ->group("GPU EM");
    app.add_option(
           "--hadronic-plan-workers", options.hadronic_plan_workers,
           "Worker count used for the retrospective hadronic final-state "
           "batch oracle (does not enable parallel execution)")
        ->check(CLI::Range(1, 256))->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-plan-target-ms", options.hadronic_plan_target_ms,
           "Target measured final-state cost per retrospective hadronic batch")
        ->check(CLI::Range(1.e-6, 1.e6))->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-plan-max-batch", options.hadronic_plan_max_batch,
           "Maximum interactions per retrospective homogeneous hadronic batch")
        ->check(CLI::PositiveNumber)->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-backend", options.hadronic_backend,
           "Low-energy hadronic final-state backend: scalar or fluka-process")
        ->check(CLI::IsMember({"scalar"}))
        ->group("Hadronic scheduling");
    app.add_option("--hadronic-workers", options.hadronic_workers,
                   "Persistent FLUKA worker process count")
        ->check(CLI::Range(1, 256))->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-min-batch", options.hadronic_min_batch,
           "Minimum parked FLUKA vertices before the predicted-cost flush test")
        ->check(CLI::PositiveNumber)->group("Hadronic scheduling");
    app.add_option("--hadronic-target-batch-ms",
                   options.hadronic_target_batch_ms,
                   "Online estimated FLUKA time per homogeneous worker batch")
        ->check(CLI::Range(1.e-6, 1.e6))->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-max-batch", options.hadronic_max_batch,
           "Maximum FLUKA requests in one homogeneous process-worker batch")
        ->check(CLI::PositiveNumber)->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-initial-cost-ms", options.hadronic_initial_cost_ms,
           "Initial per-interaction cost before a class has worker timing samples")
        ->check(CLI::Range(1.e-6, 1.e6))->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-worker-executable",
           options.hadronic_worker_executable,
           "Path to fluka_batch_worker; empty selects the executable beside c8_air_shower")
        ->group("Hadronic scheduling");
    app.add_flag(
           "--cpu-detailed-step-timing", options.cpu_detailed_step_timing,
           "Record scalar cross-section/tracking/continuous/discrete phase times")
        ->group("Hadronic scheduling");
  }

  bool validateGpuCliOptions(GpuCliOptions& options,
                             std::filesystem::path const& executable) {
    if (!options.kokkos_execution.empty() && options.em_backend != "kokkos") {
      CORSIKA_LOG_CRITICAL("--kokkos-execution requires --em-backend kokkos");
      return false;
    }
    if (!options.cuda_replay_trace.empty()) {
      options.gpu_full_step_records = true;
      validation::CudaReplayTrace::instance().open(
          options.cuda_replay_trace,
          options.em_backend == "proposal" ? "refactor_proposal"
                                           : options.em_backend);
    }
    if (!options.cuda_replay_tape_out.empty() &&
        options.em_backend != "proposal") {
      CORSIKA_LOG_CRITICAL(
          "--cuda-replay-tape-out records the scalar decision oracle and "
          "therefore requires --em-backend proposal");
      return false;
    }
    
    if (options.kokkos_cooperative_policy != "legacy" &&
        (options.em_backend != "kokkos" || options.kokkos_execution != "cuda-openmp")) {
      CORSIKA_LOG_CRITICAL("Adaptive scheduling requires Kokkos cuda-openmp execution");
      return false;
    }
    if (options.em_backend == "kokkos") {
#ifndef CORSIKA8_WITH_KOKKOS_EM
      CORSIKA_LOG_CRITICAL(
          "--em-backend kokkos was requested, but this c8_air_shower binary "
          "was built without CORSIKA_ENABLE_KOKKOS");
      return false;
#else
      if (options.gpu_physics_source != "proposal-native") {
        CORSIKA_LOG_CRITICAL(
            "--em-backend kokkos requires --gpu-physics-source proposal-native");
        return false;
      }
      if (options.radio_backend != "kokkos") {
        CORSIKA_LOG_CRITICAL(
            "--em-backend kokkos requires --radio-backend kokkos; portable "
            "EM and radio execution spaces cannot be mixed");
        return false;
      }
      if (options.hadronic_backend != "scalar") {
        CORSIKA_LOG_CRITICAL(
            "Kokkos EM is mutually exclusive with the shower-internal "
            "hadronic worker pool; use --hadronic-backend scalar");
        return false;
      }
      std::string selected_execution;
      try {
        selected_execution = accelerator::em::resolveKokkosExecutionBackend(
            options.kokkos_execution);
      } catch (std::exception const& error) {
        CORSIKA_LOG_CRITICAL("{}", error.what());
        return false;
      }
      if ((selected_execution == "cuda-openmp" || selected_execution == "openmp-cuda") && options.hadronic_workers != 1) {
        CORSIKA_LOG_CRITICAL(
            "Cooperative EM requires scalar hadrons: --hadronic-workers must be 1");
        return false;
      }
      if (selected_execution != "openmp" && selected_execution != "cuda-openmp" && selected_execution != "openmp-cuda" &&
          options.kokkos_num_threads > 1) {
        CORSIKA_LOG_CRITICAL(
            "A GPU Kokkos build uses Serial host scheduling and rejects "
            "--kokkos-num-threads > 1");
        return false;
      }
#endif
    }
    
    if (options.radio_backend == "kokkos") {
      if (options.em_backend != "kokkos") {
        CORSIKA_LOG_CRITICAL(
            "--radio-backend kokkos requires --em-backend kokkos");
        return false;
      }
#ifndef CORSIKA8_WITH_KOKKOS_EM
      CORSIKA_LOG_CRITICAL(
          "--radio-backend kokkos was requested, but Kokkos support is unavailable");
      return false;
#endif
    }
    
    return true;
  }

} // namespace

int main(int argc, char** argv) {

  // the main command line description
  CLI::App app{"Simulate standard (downgoing) showers with CORSIKA 8."};

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

  std::string profile_crossings{"original-c8"};
  app.add_option("--profile-crossings", profile_crossings,
                 "Longitudinal counts only: both, forward (half-open), or "
                 "original-c8 (forward, closed endpoints); transport/radio unchanged")
      ->check(CLI::IsMember({"both", "forward", "original-c8"}))
      ->default_val("original-c8")
      ->group("Config");

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
  std::string geomagnetic_model{"IGRF14"};
  double geomagnetic_year{2027.};
  app.add_option("--geomagnetic-model", geomagnetic_model,
                 "IGRF coefficient model used for the 21CMA magnetic field")
      ->check(CLI::IsMember({"IGRF13", "IGRF14"}))
      ->group("Config");
  app.add_option("--geomagnetic-year", geomagnetic_year,
                 "Decimal year used to evaluate the selected IGRF model")
      ->check(CLI::Range(1900., 2030.))
      ->group("Config");
  bool track_neutrinos = false;
  app.add_flag("--track-neutrinos", track_neutrinos, "switch on tracking of neutrinos")
      ->group("Config");

  corsika::applications::air_shower::GpuCliOptions gpu_cli;
  addGpuCliOptions(app, gpu_cli);

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
  // These hooks exist only in the separate experimental worker executable.
#ifdef C8_EXPERIMENTAL_STATIC_MULTIGPU
  std::string frontier_out, frontier_in;
  double frontier_max_energy = 0.;
  unsigned frontier_worker_id = 0;
  bool frontier_legacy_import = false;
  app.add_flag("--frontier-legacy-import", frontier_legacy_import,
               "Diagnostic control: import the entire frontier into the scalar stack");
  app.add_option("--frontier-out", frontier_out, "Capture independent EM subshower roots");
  app.add_option("--frontier-in", frontier_in, "Import roots of the SAME global primary");
  app.add_option("--frontier-max-energy", frontier_max_energy, "Root scheduling cap in GeV (not a cut)");
  app.add_option("--frontier-worker-id", frontier_worker_id, "Exclusive child history namespace [1,255]");
#endif
  // parse the command line options into the variables
  CLI11_PARSE(app, argc, argv);
#ifdef C8_EXPERIMENTAL_STATIC_MULTIGPU
  if ((!frontier_in.empty() && !frontier_legacy_import && gpu_cli.em_backend != "kokkos") ||
      (frontier_legacy_import && frontier_in.empty())) {
    CORSIKA_LOG_CRITICAL("Buffered frontier input requires the Kokkos worker; legacy control requires --frontier-in");
    return EXIT_FAILURE;
  }
  if ((!frontier_out.empty() || !frontier_in.empty()) &&
      (nevent != 1 || (!frontier_out.empty() && !frontier_in.empty()) ||
       !cli_energy_range.empty() ||
       (!frontier_out.empty() && (gpu_cli.em_backend != "proposal" || frontier_max_energy <= 0.)) ||
       (!frontier_in.empty() && (force_interaction || force_decay || frontier_worker_id == 0)))) {
    CORSIKA_LOG_CRITICAL("Static handoff requires N=1, fixed energy, exclusive input/output; capture uses PROPOSAL and a positive cap; workers must not force the primary again");
    return EXIT_FAILURE;
  }
#endif
  if ((gpu_cli.kokkos_execution == "cuda-openmp" || gpu_cli.kokkos_execution == "openmp-cuda") &&
      app.count("--hadronic-workers") == 0)
    gpu_cli.hadronic_workers = 1;

  if (!validateGpuCliOptions(gpu_cli, argv[0])) {
    return EXIT_FAILURE;
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

#if defined(CORSIKA8_WITH_KOKKOS_EM)
  // fork/exec the process-isolated FLUKA workers before the parent touches the
  // CUDA runtime. Forking after CUDA context creation is unsupported.
  std::unique_ptr<HadronicProcessPool>
      hadronic_process_pool;
#endif

  /* === START: SETUP ENVIRONMENT AND ROOT COORDINATE SYSTEM === */
  EnvType env;
  CoordinateSystemPtr const& rootCS = env.getCoordinateSystem();
  Point const center{rootCS, 0_m, 0_m, 0_m};
  Point const surface_{rootCS, 0_m, 0_m, constants::EarthRadius::Mean};

  // Keep the reference CPU and CUDA backends on the same 21CMA field.
  double constexpr cma21_latitude_deg = 42.5527;
  double constexpr cma21_longitude_deg = 86.4153816422;
  double constexpr cma21_altitude_m = 2680.444195;
  auto const geomagnetic_data =
      std::string{"GeoMag/"} + geomagnetic_model + ".COF";
  auto geomagnetic_path = corsika_data(geomagnetic_data);
#ifdef CORSIKA8_BUNDLED_IGRF14_FILE
  if (geomagnetic_model == "IGRF14" &&
      !boost::filesystem::exists(geomagnetic_path)) {
    geomagnetic_path = CORSIKA8_BUNDLED_IGRF14_FILE;
    CORSIKA_LOG_INFO(
        "Using the bundled IGRF14 coefficient file: {}",
        geomagnetic_path.string());
  }
#endif
  if (!boost::filesystem::exists(geomagnetic_path)) {
    CORSIKA_LOG_CRITICAL(
        "Geomagnetic coefficient file does not exist: {}",
        geomagnetic_path.string());
    return EXIT_FAILURE;
  }
  GeomagneticModel igrf(center, geomagnetic_path);
  MagneticFieldVector const cma21_field = igrf.getField(
      geomagnetic_year, cma21_altitude_m * 1_m, cma21_latitude_deg,
      cma21_longitude_deg);

  // build an atmosphere with Keilhauer's parametrization of the
  // US standard atmosphere into `env`
  create_5layer_atmosphere<EnvironmentInterface, MyExtraEnv>(
      env, AtmosphereId::USStdBK, center, 1.000327, surface_, Medium::AirDry1Atm,
      cma21_field);

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
  auto const dX = 10_g / square(1_cm); // Binning of the writers along the shower axis
  /* === END: CONSTRUCT GEOMETRY === */

  std::stringstream args;
  for (int i = 0; i < argc; ++i) { args << argv[i] << " "; }

#if defined(CORSIKA8_WITH_KOKKOS_EM)
  corsika::applications::air_shower::AcceleratedRunEnvironmentConfig
      accelerated_run_environment{
          geomagnetic_model,
          geomagnetic_year,
          cma21_latitude_deg,
          cma21_longitude_deg,
          cma21_altitude_m,
          {cma21_field.getX(rootCS) / 1_T,
           cma21_field.getY(rootCS) / 1_T,
           cma21_field.getZ(rootCS) / 1_T},
          app["--max-deflection-angle"]->as<double>(),
          {showerCoreX / 1_m, showerCoreY / 1_m,
           observationHeight / 1_m},
          app["--antenna-file"]->as<std::string>()};
  using AcceleratedRunSession =
      corsika::applications::air_shower::KokkosRunSession;
  AcceleratedRunSession accelerated_session{
      gpu_cli, accelerated_run_environment};
#endif

  // create the output manager that we then register outputs with
  OutputManager output(app["--filename"]->as<std::string>(), seed, args.str(),
                       compressOutput);

#if defined(CORSIKA8_WITH_KOKKOS_EM)
  if (accelerated_session.runOutput()) {
    output.add("gpu_em", *accelerated_session.runOutput());
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

  auto const profile_mode = profile_crossings == "original-c8"
      ? ProfileCrossingMode::OriginalC8
      : profile_crossings == "forward" ? ProfileCrossingMode::Forward
                                       : ProfileCrossingMode::Both;
  LongitudinalWriter profile{showerAxis, dX, profile_mode};
  output.add("profile", profile);
  LongitudinalProfile<SubWriter<decltype(profile)>> longprof{profile};

  ProductionWriter prod_profile{showerAxis, dX};
  output.add("production_profile", prod_profile);
  ProductionProfile<SubWriter<decltype(prod_profile)>> prodprof{prod_profile};

// for ICRC2023
#ifdef WITH_FLUKA
  corsika::fluka::Interaction leIntModel{all_elements};
  // libfluka guards the diagnostic SIGALRM at initialization for every backend,
  // including the single-worker case. No physics/RNG or worker policy changes.
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
  const TimeType duration_{gpu_cli.radio_window_duration_ns * 1_ns};
  const InverseTimeType sampleRate_{gpu_cli.radio_sampling_rate_GHz * 1_GHz};

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

  if (!antenna_positions.empty()) {
    for (std::size_t index = 0; index < antenna_positions.size(); ++index) {
      auto const [north_m, west_m, up_m] = antenna_positions[index];
      auto const point{Point(rootCS, centerX_ + north_m * 1_m,
                             centerY_ + west_m * 1_m,
                             constants::EarthRadius::Mean + up_m * 1_m)};
      auto const trigger_time =
          (triggerpoint_ - point).getNorm() / constants::c -
          gpu_cli.radio_pretrigger_ns * 1_ns;

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
      auto triggertime_1{(triggerpoint_ - point_1).getNorm() / constants::c};
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
      auto triggertime_{(triggerpoint_ - point_).getNorm() / constants::c};
      std::string name_ =
          "ZHS_R=" + std::to_string(rr_) + "_m--Phi=" + std::to_string(phi_) + "degrees";
      TimeDomainObserver observer_2(name_, point_, rootCS, triggertime_, duration_,
                                    sampleRate_, triggertime_);
      detectorZHS.addObserver(observer_2);
    }
  }
  LengthType const step = 1_m;
  auto TP =
      make_tabulated_flat_atmosphere_radio_propagator(env, injectionPos, surface_, step);

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
  timing_configuration["em_backend"] = gpu_cli.em_backend;
  timing_configuration["gpu_physics_source"] = gpu_cli.gpu_physics_source;
  timing_configuration["radio_backend"] = gpu_cli.radio_backend;
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

  if (!gpu_cli.cuda_replay_tape_out.empty()) {
    auto const replay_radio_snapshot =
        validation::makeReplayRadioSnapshot(
            env, injectionPos, surface_, step,
            detectorCoREAS, detectorZHS);
    validation::CudaDecisionReplayTape::instance().open(
        gpu_cli.cuda_replay_tape_out, replay_radio_snapshot,
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
#if defined(CORSIKA8_WITH_KOKKOS_EM)
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
#ifdef C8_EXPERIMENTAL_STATIC_MULTIGPU
    std::unique_ptr<applications::multigpu::FrontierInput> frontier_source;
    if (!frontier_in.empty()) {
      if (frontier_legacy_import) {
        applications::multigpu::importFrontier(stack, rootCS, frontier_in, frontier_worker_id);
      } else {
        frontier_source = std::make_unique<applications::multigpu::FrontierInput>(frontier_in);
        frontier_source->reserveNamespace(stack, frontier_worker_id);
      }
    } else
#endif
      stack.addParticle(primaryProperties);

#ifdef C8_EXPERIMENTAL_STATIC_MULTIGPU
    applications::multigpu::ScopedFrontierInput frontier_context(frontier_source.get());
#endif

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
#ifdef C8_EXPERIMENTAL_STATIC_MULTIGPU
    if (!frontier_out.empty()) {
      applications::multigpu::CaptureRouter router(frontier_out, rootCS, frontier_max_energy);
      HybridCascade<TrackingType, decltype(sequence), decltype(output), StackType,
                    applications::multigpu::CaptureRouter> cascade(
          env, tracking, sequence, output, stack, router);
      configure_forced_primary(cascade);
      cascade.run();
    } else
#endif
    if (gpu_cli.em_backend == "proposal") {
      Cascade EAS(env, tracking, sequence, output, stack);
      configure_forced_primary(EAS);
      EAS.run();
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
#if defined(CORSIKA8_WITH_KOKKOS_EM)
      try {
        corsika::applications::air_shower::KokkosEventConfig cuda_event{
            static_cast<std::uint64_t>(seed),
            static_cast<std::uint64_t>(i_shower),
            output_shower_id,
            eMax,
            primaryTotalEnergy,
            emcut,
            mucut,
            prod_threshold,
            heHadronModelThreshold,
            emthinfrac,
            maxWeight,
            automaticMaxWeight,
            thinningCanActivateFromUnitWeight,
            multithin,
            beamCode,
            app["--hadronModel"]->as<std::string>(),
            app["--max-deflection-angle"]->as<double>()};
        corsika::applications::air_shower::runKokkosAirShower(
            accelerated_session, gpu_cli, cuda_event, env, rootCS, injectionPos,
            surface_, step, detectorCoREAS, detectorZHS, showerAxis, dX,
            observationHeight / 1_m, showerCoreX / 1_m,
            showerCoreY / 1_m,
            std::array<double, 3>{
                cma21_field.getX(rootCS) / 1_T,
                cma21_field.getY(rootCS) / 1_T,
                cma21_field.getZ(rootCS) / 1_T},
            dEdX, profile, prod_profile, observationLevel, inter_writer,
            coreas, zhs, emCascade, emContinuousProposal, stackInspect,
            neutrinoPrimaryPythia, hadronSequence, decaySequence,
            emContinuous, longprof, prodprof, thinning, cut, sequence,
            tracking, stack, output, hadronic_process_pool.get(),
            configure_forced_primary, photoHadronicHighEnergy,
            photoHadronicLowEnergy, photo_hadronic_before,
            photo_hadronic_le_before, heCounted, leIntCounted,
            high_energy_hadronic_interactions_before,
            low_energy_hadronic_interactions_before,
            high_energy_hadronic_timings_before,
            low_energy_hadronic_timings_before,
            hadronic_pool_statistics_before,
            photoHadronicQgsjetFallback);
      } catch (std::exception const& error) {
        if (accelerated_session.runOutput()) {
          accelerated_session.runOutput()->recordIncomplete(
              output_shower_id, error.what());
        }
        if (output.showerInProgress()) {
          output.endOfShower();
        }
        output.endOfLibrary();
        CORSIKA_LOG_CRITICAL(
            "Accelerated EM shower {} aborted; output is marked incomplete: {}",
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

    // Reports above have consumed this shower's samples. Do not retain one
    // entry per hadronic interaction over the entire multi-shower library.
    leIntCounted.releaseTimingSamples();
    heCounted.releaseTimingSamples();
  }

  // and finalize the output on disk
  output.endOfLibrary();
  validation::CudaReplayTrace::instance().close();
  validation::CudaDecisionReplayTape::instance().close();

  return EXIT_SUCCESS;
}
