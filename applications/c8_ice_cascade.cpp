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

#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/media/ShowerAxis.hpp>
#include <corsika/media/UniformMagneticField.hpp>

#include <corsika/modules/BetheBlochPDG.hpp>
#include <corsika/modules/Epos.hpp>
#include <corsika/modules/ObservationPlane.hpp>
#include <corsika/modules/PROPOSAL.hpp>
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
#include <corsika/modules/radio/propagators/HomogeneousRadioPropagator.hpp>

#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupTrajectory.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>

#include <boost/filesystem.hpp>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <corsika/kokkos/application/cascade/CascadeExtension.hpp>
#include <corsika/kokkos/application/cascade/SouthPoleSite.hpp>

using namespace corsika;
using namespace std;
namespace application = corsika::kokkos::application;

using EnvironmentInterface =
    IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
using EnvType = Environment<EnvironmentInterface>;
using StackType = setup::HybridStack<EnvType>;
using TrackingType = setup::Tracking;
using Particle = StackType::particle_type;

// Independent homogeneous-ice application using the shared cascade backend.

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
using IceMedium = UniformRefractiveIndex<
    MediumPropertyModel<UniformMagneticField<HomogeneousMedium<T>>>>;

std::vector<std::tuple<double, double, double>> read_antenna_positions(
    std::string const& filename) {
  std::vector<std::tuple<double, double, double>> antennas;
  if (filename.empty()) return antennas;
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

// Keep the coordinator out of the large shower/Fortran stack frame. In
// particular, explicit-loader deployments have little heap/stack headroom.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
int runIceCascade(int argc, char** argv,
                corsika::kokkos::application::DeviceSelection const& selected_devices);

int main(int argc, char** argv) {
  return kokkos::application::dispatch_cascade(argc, argv, runIceCascade);
}

int runIceCascade(int argc, char** argv,
                  corsika::kokkos::application::DeviceSelection const& selected_devices) {
  // the main command line description
  CLI::App app{"Simulate cascades in homogeneous South Pole ice with CORSIKA 8."};

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
  application::SouthPoleSite site;
  double iceDensity = 0.919;
  double iceIndex = 1.78;
  double profileBin = 10.;
  app.add_option("--surface-altitude", site.surfaceAltitudeM_,
                 "South Pole reference surface altitude above mean Earth radius (m)")
      ->default_val(2835.)
      ->group("Config");
  app.add_option("--density-g-cm3", iceDensity, "Homogeneous water-ice density (g/cm3)")
      ->default_val(0.919)
      ->check(CLI::PositiveNumber)
      ->group("Config");
  app.add_option("--refractive-index", iceIndex, "Uniform ice refractive index")
      ->default_val(1.78)
      ->check(CLI::Range(1., 10.))
      ->group("Config");
  app.add_option("--profile-bin-g-cm2", profileBin, "Profile grammage bin width (g/cm2)")
      ->default_val(10.)
      ->check(CLI::PositiveNumber)
      ->group("Config");
  bool track_neutrinos = false;
  app.add_flag("--track-neutrinos", track_neutrinos, "switch on tracking of neutrinos")
      ->group("Config");

  application::CascadeExtension backend{app};

  //////// Misc options ////////

  app.add_option("--neutrino-interaction-type",
                 "charged (CC) or neutral current (NC) or both")
      ->default_val("both")
      ->check(CLI::IsMember({"neutral", "NC", "charged", "CC", "both"}))
      ->group("Misc.");
  app.add_option("--observation-level",
                 "Height above earth radius of the observation level (in m)")
      ->default_val(2585.)
      ->check(CLI::Range(-1.e3, 1.e5))
      ->group("Config");
  app.add_option("--injection-height",
                 "Height above earth radius of the injection point (in m)")
      ->default_val(2635.)
      ->check(CLI::Range(-1.e3, 1.e6))
      ->group("Config");
  app.add_option("--shower-core-x", "X coordinate of the shower core position (in m)")
      ->default_val(0.)
      ->group("Config");
  app.add_option("--shower-core-y", "Y coordinate of the shower core position (in m)")
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
  app.add_option("--antenna-file", "Path to antenna positions in NWU coordinates (m)")
      ->default_val("")
      ->group("Radio");
  // parse the command line options into the variables
  CLI11_PARSE(app, argc, argv);
  if (!backend.prepare(selected_devices, argv[0])) return EXIT_FAILURE;

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

  /* === START: SETUP ENVIRONMENT AND ROOT COORDINATE SYSTEM === */
  EnvType env;
  auto const rootCS = site.coordinates(env.getCoordinateSystem());
  Point const center{rootCS, 0_m, 0_m, 0_m};
  Point const surface_ = site.surface(rootCS);
  MagneticFieldVector const iceField(rootCS, 0_T, 0_T, 0_T);
  double const injectionAltitude = app["--injection-height"]->as<double>();
  double const observationAltitude = app["--observation-level"]->as<double>();
  if (!(observationAltitude < injectionAltitude &&
        injectionAltitude < site.surfaceAltitudeM_)) {
    CORSIKA_LOG_ERROR("Require observation-level < injection-height < surface-altitude");
    return EXIT_FAILURE;
  }
  NuclearComposition const iceComposition({Code::Hydrogen, Code::Oxygen},
                                          {2. / 3., 1. / 3.});
  // Preserve the archived bulk-ice benchmark's remote spherical enclosure.
  // The site surface is a depth reference, not an ice/air propagation interface.
  auto ice = EnvType::createNode<Sphere>(
      center, constants::EarthRadius::Mean +
                  (std::max(injectionAltitude, observationAltitude) + 5000.) * 1_m);
  ice->setModelProperties<IceMedium<EnvironmentInterface>>(
      iceIndex, Medium::WaterIce, iceField, iceDensity * 1_g / cube(1_cm),
      iceComposition);
  env.getUniverse()->addChild(std::move(ice));
  CORSIKA_LOG_INFO(
      "South Pole ice: latitude={}, longitude convention={}, surface={} m, "
      "rho={} g/cm3, n={}, B=0; local axes NWU",
      site.latitudeDeg, site.longitudeDeg, site.surfaceAltitudeM_, iceDensity, iceIndex);

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
  auto const showerCoreX = app["--shower-core-x"]->as<double>() * 1_m;
  auto const showerCoreY = app["--shower-core-y"]->as<double>() * 1_m;
  Point const showerCore{rootCS, showerCoreX, showerCoreY, observationHeight};
  Point const injectionPos =
      showerCore + DirectionVector{rootCS,
                                   {-sin(thetaRad) * cos(phiRad),
                                    -sin(thetaRad) * sin(phiRad), cos(thetaRad)}} *
                       t;

  // we make the axis much longer than the inj-core distance since the
  // profile will go beyond the core, depending on zenith angle
  ShowerAxis const showerAxis{injectionPos, (showerCore - injectionPos) * 1.2, env};
  auto const dX =
      profileBin * 1_g / square(1_cm); // Binning of the writers along the shower axis
  /* === END: CONSTRUCT GEOMETRY === */

  std::stringstream args;
  for (int i = 0; i < argc; ++i) { args << argv[i] << " "; }

  application::RunEnvironmentConfig run_environment{
      "zero",
      0.,
      site.latitudeDeg,
      site.longitudeDeg,
      site.surfaceAltitudeM_,
      {0., 0., 0.},
      app["--max-deflection-angle"]->as<double>(),
      {showerCoreX / 1_m, showerCoreY / 1_m, observationHeight / 1_m},
      app["--antenna-file"]->as<std::string>(),
      application::CascadeScene::HomogeneousIce};
  backend.configure(run_environment);

  // create the output manager that we then register outputs with
  OutputManager output(app["--filename"]->as<std::string>(), seed, args.str(),
                       compressOutput);

  backend.registerOutput(output);

  // register energy losses as output
  auto dEdX = backend.energyLossWriter(showerAxis, dX);
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

  auto emCascade = backend.interaction(env, sophia, sibyll->getHadronInteractionModel(),
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

  auto profile = backend.longitudinalWriter(showerAxis, dX);
  output.add("profile", profile);
  LongitudinalProfile<SubWriter<decltype(profile)>> longprof{profile};

  ProductionWriter prod_profile{showerAxis, dX};
  output.add("production_profile", prod_profile);
  ProductionProfile<SubWriter<decltype(prod_profile)>> prodprof{prod_profile};

// for ICRC2023
#ifdef WITH_FLUKA
  corsika::fluka::Interaction leIntModel{all_elements};
#else
  corsika::urqmd::UrQMD leIntModel{};
#endif
  auto leIntCounted = backend.interactionCounter(leIntModel);

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
  const TimeType duration_{backend.radioWindowDurationNs() * 1_ns};
  const InverseTimeType sampleRate_{backend.radioSamplingRateGHz() * 1_GHz};

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
      auto const point{Point(rootCS, centerX_ + north_m * 1_m, centerY_ + west_m * 1_m,
                             constants::EarthRadius::Mean + up_m * 1_m)};
      auto const* observerNode = env.getUniverse()->getContainingNode(point);
      if (!observerNode->hasModelProperties()) {
        throw std::invalid_argument("antenna lies outside the homogeneous ice volume");
      }
      auto const trigger_time =
          iceIndex * (triggerpoint_ - point).getNorm() / constants::c -
          backend.radioPretriggerNs() * 1_ns;

      TimeDomainObserver coreas_observer("CoREAS_Antenna_" + std::to_string(index), point,
                                         rootCS, trigger_time, duration_, sampleRate_,
                                         trigger_time);
      detectorCoREAS.addObserver(coreas_observer);

      TimeDomainObserver zhs_observer(
          "ZHS_Antenna_" + std::to_string(index), point, rootCS, trigger_time, duration_,
          sampleRate_, trigger_time, VectorPotentialWindowMode::RoundedBins);
      detectorZHS.addObserver(zhs_observer);
    }
  } else if (ring_number != 0) {
    // setup CoREAS observers - use the for loop for star shape pattern
    for (auto phi_1 = 0; phi_1 <= 315; phi_1 += 45) {
      auto phiRad_1 = phi_1 / 180. * M_PI;
      auto const point_1{Point(rootCS, showerCoreX_ + radius_ * cos(phiRad_1),
                               showerCoreY_ + radius_ * sin(phiRad_1),
                               observationHeight)};
      std::cout << "Observer point CoREAS: " << point_1 << std::endl;
      auto triggertime_1{iceIndex * (triggerpoint_ - point_1).getNorm() / constants::c};
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
                              showerCoreY_ + radius_ * sin(phiRad_), observationHeight)};
      std::cout << "Observer point ZHS: " << point_ << std::endl;
      auto triggertime_{iceIndex * (triggerpoint_ - point_).getNorm() / constants::c};
      std::string name_ =
          "ZHS_R=" + std::to_string(rr_) + "_m--Phi=" + std::to_string(phi_) + "degrees";
      TimeDomainObserver observer_2(name_, point_, rootCS, triggertime_, duration_,
                                    sampleRate_, triggertime_,
                                    VectorPotentialWindowMode::RoundedBins);
      detectorZHS.addObserver(observer_2);
    }
  }
  LengthType const step = 1_m;
  HomogeneousRadioPropagator<EnvType> TP(env);

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

  // Attach the optional backend to the native objects; all ownership remains here.
  auto application = backend.bind(
      env, rootCS, injectionPos, surface_, step, showerAxis, dX, seed, eMax, beamCode,
      prod_threshold, heHadronModelThreshold, multithin, dEdX, profile, prod_profile,
      observationLevel, inter_writer, detectorCoREAS, detectorZHS, coreas, zhs, emCascade,
      emContinuousProposal, neutrinoPrimaryPythia, hadronSequence, decaySequence,
      emContinuous, longprof, prodprof, cut, heCounted, leIntCounted, output);

  // trigger the output manager to open the library for writing
  output.startOfLibrary();

  // loop over each shower
  for (int i_shower = 1; i_shower < nevent + 1; i_shower++) {

    auto event = application.shower(i_shower);

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
    double const configuredMaxWeight = app["--max-weight"]->as<double>();
    double const maxWeight = std::invoke([&]() {
      if (configuredMaxWeight > 0.)
        return configuredMaxWeight;
      else
        return 0.5 * emthinfrac * primaryTotalEnergy / 1_GeV;
    });
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
    event.addPrimary(stack, primaryProperties);

    primaryWriter.recordPrimary(primaryProperties);

    auto EAS = event.cascade(tracking, sequence, output, stack, stackInspect, thinning,
                             primaryTotalEnergy, maxWeight);
    if (force_interaction) {
      CORSIKA_LOG_INFO("Fixing first interaction at injection point.");
      EAS.forceInteraction();
    }
    if (force_decay) {
      CORSIKA_LOG_INFO("Forcing the primary to decay");
      EAS.forceDecay();
    }
    EAS.run();

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

  return EXIT_SUCCESS;
}
