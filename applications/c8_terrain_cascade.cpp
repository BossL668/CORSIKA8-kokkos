/* Independent geographic terrain transport application.
 * CPU reference or bounded Kokkos multi-material EM; IGRF14 in air, material B inside.
 * Optional interface ZHS/CoREAS radio: single transmitted GO branches, straight legs.
 */
#include <corsika/framework/core/Cascade.hpp>
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/framework/process/SwitchProcessSequence.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/modules/BetheBlochPDG.hpp>
#include <corsika/modules/FLUKA.hpp>
#include <corsika/modules/PROPOSAL.hpp>
#include <corsika/modules/proposal/ThresholdPhotoproductionModel.hpp>
#include <corsika/modules/QGSJetII.hpp>
#include <corsika/modules/Pythia8.hpp>
#include <corsika/modules/Sophia.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <corsika/modules/thinning/EMThinning.hpp>
#include <corsika/modules/neutrino/MountainNeutrinoInteraction.hpp>
#include <corsika/modules/neutrino/NeutrinoPhysicsCapabilities.hpp>
#include <corsika/modules/neutrino/TransportedLeptons.hpp>
#include <corsika/modules/terrain/TerrainMagneticTracking.hpp>
#include <corsika/modules/transport/InterfaceTracking.hpp>
#include <corsika/modules/transport/MaterialProposalEnvironment.hpp>
#include <corsika/modules/transport/MaterialHadronLoss.hpp>
#include "detail/mountain/MaterialFlukaInteraction.hpp"
#include <corsika/modules/terrain/TerrainMagneticField.hpp>
#include <corsika/modules/writers/SubWriter.hpp>
#include <corsika/output/OutputManager.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>
#include "detail/mountain/TerrainScene.hpp"
#include "detail/mountain/TerrainShowerOutput.hpp"
#include "detail/mountain/TerrainTauDecay.hpp"
#include "detail/mountain/TerrainPrimaries.hpp"
#ifdef CORSIKA8_WITH_KOKKOS_EM
#include "detail/mountain/TerrainAcceleratedRun.hpp"
#endif
#include <CLI/CLI.hpp>
#include <chrono>
#include <iostream>
#include <csignal>
#include <unistd.h>

using namespace corsika;
using namespace corsika::units::si;
namespace terrainapp = corsika::applications::terrain;
using TerrainStack = setup::HybridStack<corsika::terrain::Environment>;

int main(int argc, char** argv) {
  // 1. Application parameters only. Coordinates and all module boundaries use ENU.
  CLI::App app{"Geographic DEM + native USStdBK transport and optional Kokkos interface radio"};
  std::string sceneFile, output, primary = "photon", backend = "proposal", auxCache;
  std::string emScheduler = "resident";
  std::size_t residentCapacity = 65536;
  std::size_t residentWavefront = 0, residentRecords = 0, radioBatch = 8192;
  long seed = 67101;
  int threads = 1, device = 0;
  std::size_t batch = 64, memoryMiB = 128, trackRows = 200000;
  std::uint64_t transportStepLimit = 2000000;
  double energy = 1., emcut = .0005, hadcut = .3, mucut = .3, emthin = 1.e-6;
  double maxWeight = 0., maxStep = std::numeric_limits<double>::infinity();
  double windowNs = std::numeric_limits<double>::infinity();
  std::string magnetic = "igrf14";
  std::string igrfFile = corsika_data("GeoMag/IGRF14.COF").string();
  if (!std::filesystem::is_regular_file(igrfFile)) {
    igrfFile = CORSIKA8_TERRAIN_IGRF14_FILE;
  }
  double year = 2027.;
  bool force = false, forceNc = false, requireNeutrinoCoverage = false;
  bool energyLedger = false;
  bool radioEnabled = false;
  bool compressTerrainCsv = false;
  bool profileDeviceOutput = false;
  unsigned deviceOutputThreads = 0;
  bool referenceDeviceGeometry = false;
  neutrino::NeutrinoPhysicsRequirements physicsRequirements;
  std::string neutrinoChannels = "cc+nc";
  std::string tauDecayModel = "tauola", tauolaHelicity = "left";
  double tauPolarization = 0.;
  std::vector<double> position, direction{0., 0., 1.};

  app.add_option("--scene", sceneFile)->required()->check(CLI::ExistingFile);
  app.add_option("--output", output)->required();
  app.add_flag("--radio",radioEnabled,"Compute both CoREAS and ZHS interface radio using scene radio settings");
  app.add_flag("--compress-terrain-csv",compressTerrainCsv,
               "Losslessly compress complete terrain diagnostic CSV streams with gzip");
  app.add_flag("--profile-device-output",profileDeviceOutput,
               "Measure host geometry checks and complete CSV output without changing transport");
  app.add_option("--device-output-threads",deviceOutputThreads,
                 "CPU geometry audit threads: 0 bounded automatic, 1 serial reference")
      ->check(CLI::Range(0,256));
  app.add_flag("--reference-device-geometry",referenceDeviceGeometry,
               "Use the original CPU geometry objects for independent audit comparisons");
  app.add_option("--primary", primary)
      ->check(CLI::IsMember(terrainapp::primaries()));
  app.add_option("--energy-GeV", energy)->check(CLI::PositiveNumber);
  app.add_option("--position-m", position,
                 "ENU injection coordinates (required; never silently moved)")
      ->required()->expected(3);
  app.add_option("--direction", direction)->expected(3);
  app.add_option("--emcut-GeV", emcut)->check(CLI::PositiveNumber);
  app.add_option("--hadcut-GeV", hadcut)->check(CLI::PositiveNumber);
  app.add_option("--mucut-GeV", mucut)->check(CLI::PositiveNumber);
  app.add_option("--emthin", emthin)->check(CLI::Range(0., 1.));
  auto* weightOption = app.add_option("--max-weight", maxWeight,
      "EM maximum weight; omitted: retain 0.5 * emthin * primary GeV")
      ->check(CLI::PositiveNumber);
  auto* stepOption = app.add_option("--max-step-m", maxStep,
      "Optional diagnostic step cap; neutrinos exempt as in mountain")
      ->check(CLI::PositiveNumber);
  auto* windowOption = app.add_option("--transport-window-ns", windowNs,
      "Finite physical-time window; survivors recorded as escaped, not deposited")
      ->check(CLI::Range(1., 1.e6));
  app.add_option("--seed", seed)->check(CLI::NonNegativeNumber);
  app.add_option("--em-backend", backend)->check(CLI::IsMember({"proposal", "kokkos"}));
  app.add_option("--em-scheduler", emScheduler,
                 "Kokkos particle queue: resident, or batched reference for comparison")
      ->check(CLI::IsMember({"resident", "batched"}));
  app.add_option("--resident-capacity", residentCapacity,
                 "Maximum particles retained in the Kokkos execution space")
      ->check(CLI::Range(1LL, 16777216LL));
  app.add_option("--threads", threads)->check(CLI::Range(1, 256));
  app.add_option("--device", device)->check(CLI::NonNegativeNumber);
  app.add_option("--batch", batch)->check(CLI::Range(1, 65536));
  app.add_option("--resident-wavefront-capacity", residentWavefront,
                 "Resident front capacity; zero uses --batch, independently of CPU staging")
      ->check(CLI::Range(0, 65536));
  app.add_option("--resident-record-capacity", residentRecords,
                 "Records retained before host output; zero uses max(4096, 16 * front capacity)")
      ->check(CLI::Range(0LL, 1048576LL));
  app.add_option("--radio-batch", radioBatch,
                 "OpenMP source records per radio launch; zero retains immediate wavefront dispatch")
      ->check(CLI::Range(0LL, 1048576LL));
  app.add_option("--device-memory-MiB", memoryMiB,
                 "Hard allocation budget for transport, radio, and device queues")
      ->check(CLI::Range(32, 262144));
  app.add_option("--aux-cache", auxCache);
  app.add_flag("--energy-ledger", energyLedger,
               "Read-only weighted-energy ledger (extra diagnostic materialization; no extra random draws)");
  app.add_option("--track-row-limit", trackRows,
                 "Streaming CSV limit; no in-memory track retention")
      ->check(CLI::Range(1LL, 10000000000LL));
  app.add_option("--transport-step-limit", transportStepLimit,
                 "Maximum transport records before explicit incomplete failure")
      ->check(CLI::Range(1LL, 10000000000LL));
  app.add_flag("--force-vertex-cc", force,
               "Conditional neutrino CC at the specified in-rock point; not an event rate");
  app.add_flag("--force-vertex-nc", forceNc,
               "Conditional neutrino NC at the specified in-rock point; not an event rate");
  app.add_option("--neutrino-channels",neutrinoChannels,
                 "CTW channels; regenerated neutrinos reenter the same transport (>=10 TeV)")
      ->check(CLI::IsMember({"cc","nc","cc+nc"}));
  app.add_flag("--require-neutrino-model-coverage",requireNeutrinoCoverage,
      "Stop before transporting any secondary neutrino outside the physical CTW energy domain");
  app.add_flag("--require-complete-weak-transport", physicsRequirements.completeWeakTransport,
      "Require a complete weak model; reject this limited CTW/DIS implementation before running");
  app.add_flag("--require-event-cc-polarization", physicsRequirements.eventCCPolarization,
      "Require CC-derived spin; prescribed decay polarization is not sufficient");
  app.add_flag("--require-tau-depolarization", physicsRequirements.materialDepolarization,
      "Require material spin-transfer physics, not only PROPOSAL energy loss");
  auto* tauPolarizationOption = app.add_option("--tau-minus-polarization",tauPolarization,
      "Explicit DECAY polarization control [-1,1]; tau+ has opposite sign; not a CC spin model")
      ->check(CLI::Range(-1.,1.));
  app.add_option("--tau-decay-model", tauDecayModel,
      "Transported tau decay: original CORSIKA TAUOLA (default), or legacy Pythia")
      ->check(CLI::IsMember({"tauola", "pythia"}));
  auto* tauolaHelicityOption = app.add_option("--tauola-helicity", tauolaHelicity,
      "Original TAUOLA fixed-helicity convention; not an event-level CC spin calculation")
      ->check(CLI::IsMember({"left", "unpolarized", "right"}));
  app.add_option("--magnetic-field", magnetic, "Air field; embedded field comes from the material card (default zero)")
      ->check(CLI::IsMember({"igrf14", "none"}));
  app.add_option("--magnetic-year", year)->check(CLI::Range(2025., 2030.));
  app.add_option("--igrf-file", igrfFile)->check(CLI::ExistingFile);
  CLI11_PARSE(app, argc, argv);
  auto save = [&](YAML::Node const& summary) {
    std::ofstream file(std::filesystem::path(output) / "terrain_run.yaml");
    if (!file) throw std::runtime_error("cannot write terrain summary");
    file << summary << '\n';
    if (!file) throw std::runtime_error("terrain summary write failed");
  };
  YAML::Node summary;
  bool ownsOutput = false;
  try {
#ifndef CORSIKA8_WITH_KOKKOS_EM
    if (backend == "kokkos") {
      throw std::invalid_argument("terrain application was built without Kokkos");
    }
#endif
    logging::set_level(logging::level::warn);
    neutrino::validateNeutrinoPhysicsRequirements(physicsRequirements);
    if (force && forceNc) throw std::invalid_argument("select at most one forced weak current");
    if ((force && neutrinoChannels == "nc") || (forceNc && neutrinoChannels == "cc"))
      throw std::invalid_argument("forced weak current disabled by --neutrino-channels");
    if (tauPolarizationOption->count() && !std::isfinite(tauPolarization))
      throw std::invalid_argument("nonfinite tau polarization");
    if (tauPolarizationOption->count() && tauolaHelicityOption->count())
      throw std::invalid_argument("select either --tau-minus-polarization or --tauola-helicity");
    if (tauolaHelicityOption->count() && tauDecayModel != "tauola")
      throw std::invalid_argument("--tauola-helicity requires --tau-decay-model tauola");
    bool const forcedCC = force;
    force = force || forceNc;
    if ((weightOption->count() && !std::isfinite(maxWeight)) ||
        (stepOption->count() && !std::isfinite(maxStep)) ||
        (windowOption->count() && !std::isfinite(windowNs)))
      throw std::invalid_argument("nonfinite diagnostic transport control");
    if (!weightOption->count()) maxWeight = .5*emthin*energy;
    if (std::filesystem::exists(output)) {
      throw std::invalid_argument("output exists; choose a new directory");
    }
    // 2. Geographic scene admission: validated mesh, actual observer locations,
    //    ASL/ellipsoidal-height convention and material properties.
    terrainapp::Scene scene(sceneFile);
    radioEnabled=radioEnabled||scene.yaml["radio"]["enabled"].as<bool>(false);
    if(radioEnabled&&backend!="kokkos")throw std::invalid_argument("interface radio currently requires --em-backend kokkos (resident or batched)");
    terrain::Environment env;
    auto cs = env.getCoordinateSystem();
    // Same IGRF evaluator as the air application; explicit NWU -> ENU rotation.
    if (magnetic == "igrf14") {
      scene.atmosphere.air_magnetic_field_enu_T = terrain::igrf14AirFieldEnu(
          cs, igrfFile, year, scene.yaml["site"]["origin_ellipsoidal_height_m"].as<double>(),
          scene.yaml["site"]["latitude_deg"].as<double>(),
          scene.yaml["site"]["longitude_deg"].as<double>());
    }
    // Native five-layer atmosphere owns the DEM child and its selected material.
    auto mesh = scene.build(env);
    // Stock PROPOSAL keys use composition: reject ambiguous material properties
    // before constructing any calculator (e.g. coexisting water and ice banks).
    interfaces::validateCalculatorMaterialKeys(env);
    interfaces::MaterialProposalEnvironment materialEnvironment(env);
    std::clog << "[terrain] material: " << scene.material.description << " [" << scene.material.id << "]\n";
    if(scene.legacy_material_overrides)
      std::clog << "[terrain] legacy density/index/attenuation overrides selected material; see resolved_material in terrain_run.yaml\n";
    for (double v : position) {
      if (!std::isfinite(v)) throw std::invalid_argument("nonfinite injection");
    }
    for (double v : direction) {
      if (!std::isfinite(v)) throw std::invalid_argument("nonfinite direction");
    }
    if (!std::isfinite(energy) || !std::isfinite(emcut) || !std::isfinite(hadcut) ||
        !std::isfinite(mucut) || !std::isfinite(emthin)) {
      throw std::invalid_argument("nonfinite energy/cut/thinning");
    }
    Point injection(cs, position[0]*1_m, position[1]*1_m, position[2]*1_m);
    DirectionVector dir(cs, {direction[0], direction[1], direction[2]});
    if (!(dir.getNorm() > 0.)) throw std::invalid_argument("zero direction");
    dir = dir.normalized();
    auto pid = terrainapp::primaries().at(primary);
    if (energy*1_GeV <= get_mass(pid)) throw std::invalid_argument("energy below primary mass");
    if (is_neutrino(pid)) neutrino::MountainNeutrinoInteraction::validatePrimary(pid, energy);
    if (force && (!is_neutrino(pid) ||
                  mesh.mesh->inside(injection) != terrain::EInside::kInside)) {
      throw std::invalid_argument("forced CC/NC requires a neutrino vertex strictly inside rock");
    }
    auto* node = env.getUniverse()->getContainingNode(injection);
    if (!node || !node->hasModelProperties()) {
      throw std::invalid_argument("injection outside physical atmosphere/terrain");
    }
    // 3. Original random-stream registration order and production thresholds.
    //    Device history keys are managed by the router, not by this application.
    auto& rng = RNGManager<>::getInstance();
    for (auto const* name : {"cascade", "qgsjet", "sophia", "pythia", "fluka", "proposal", "thinning"}) {
      rng.registerRandomStream(name);
    }
    // Append, never reorder existing stream indices. The legacy Pythia option
    // does not initialize TAUOLA or consume any of its random numbers.
    if (tauDecayModel == "tauola") rng.registerRandomStream("tauola");
    rng.setSeed(seed);
    auto productionCut = std::min({emcut, hadcut, mucut})*1_GeV;
    for (auto code : {Code::Photon, Code::Electron, Code::Positron, Code::MuMinus,
                     Code::MuPlus, Code::TauMinus, Code::TauPlus}) {
      set_energy_production_threshold(code, productionCut);
    }
    terrainapp::ShowerOutput diagnostics(env, mesh.mesh, trackRows, maxStep, windowNs*1.e-9,
                                        requireNeutrinoCoverage, transportStepLimit,compressTerrainCsv);
    diagnostics.ledger.enabled=energyLedger;
    diagnostics.coverage=scene.coverage.view();
    if(!corsika::terrain::coverage::contains(diagnostics.coverage,
        {injection.getX(cs)/1_m,injection.getY(cs)/1_m,injection.getZ(cs)/1_m}))
      throw std::invalid_argument("injection outside DEM coverage");
    diagnostics.radio_enabled=radioEnabled;
    diagnostics.profile_device_output=profileDeviceOutput;
    diagnostics.device_output_threads=deviceOutputThreads;
    diagnostics.reference_device_geometry=referenceDeviceGeometry;
    // 4. Visible physics assembly, following the air application pattern.
    //    FLUKA/QGSJet-II cover hadrons; PROPOSAL covers EM and lepton losses.
    //    Derive targets from actual materials (including H in water/ice), not
    //    from the geometry's name. The default SiO2/air target set is unchanged.
    auto targets=interfaces::collectNuclearTargets(env);
    std::clog << "[terrain] initializing FLUKA environment targets\n";
    terrainapp::MaterialFlukaInteraction low(targets);
    if (std::signal(SIGALRM, SIG_IGN) == SIG_ERR) {
      throw std::runtime_error("cannot disable FLUKA diagnostic timer");
    }
    ::alarm(0);
    std::clog << "[terrain] initializing QGSJet-II and photonuclear models\n";
    qgsjetII::Interaction high;
    corsika::sophia::InteractionModel photo;
    proposal::ThresholdPhotoproductionModel threshold;
    proposal::HadronicInteractionModelFallback photoWithThreshold(photo, threshold);
    std::clog << "[terrain] initializing native PROPOSAL interaction calculators\n";
    proposal::Interaction em(materialEnvironment, photoWithThreshold, high, 80_GeV);
    std::clog << "[terrain] initializing native PROPOSAL continuous calculators\n";
    proposal::ContinuousProcess<SubWriter<terrainapp::ShowerOutput>> continuous(materialEnvironment, diagnostics);
    interfaces::MaterialHadronLoss<SubWriter<terrainapp::ShowerOutput>> hadronLoss(diagnostics);
    auto losses = make_select(
        [](TerrainStack::particle_type const& p) { return is_hadron(p.getPID()); },
        hadronLoss, continuous);
    auto hadrons = make_select(
        [](TerrainStack::particle_type const& p) { return p.getEnergyNN() < 80_GeV; },
        low, high);
    neutrino::TransportedLeptonDecayConfig decayConfig;
    decayConfig.model = tauDecayModel == "tauola" ? neutrino::TauDecayModel::Tauola
                                                 : neutrino::TauDecayModel::Pythia;
    decayConfig.helicity = tauolaHelicity == "left" ? tauola::Helicity::LeftHanded :
        (tauolaHelicity == "right" ? tauola::Helicity::RightHanded : tauola::Helicity::Unpolarized);
    if (tauPolarizationOption->count()) {
      if (tauDecayModel == "tauola") decayConfig.prescribedTauolaPolarization = tauPolarization;
      else decayConfig.prescribedPythiaPolarization = tauPolarization;
    }
    // Same tau/non-tau switch as original c8_air_shower; all neutrino daughters
    // return to the normal stack, where CC+NC remain enabled.
    terrainapp::TauDecay decay(cs, decayConfig);
    // Tau retention and its audit are local to this independent executable.
    // Competing neutrino CC+NC distances are sampled naturally. A forced first
    // vertex is conditional on the requested current, never an event-rate weight.
    neutrino::MountainNeutrinoInteraction neutrinos(
        neutrino::withTransportedTaus(setup::C7trackedParticles),
        forceNc ? neutrino::SamplingMode::ForcedVertexNC :
          (force ? neutrino::SamplingMode::ForcedVertexCC : neutrino::SamplingMode::NaturalCC),
        neutrinoChannels == "cc" ? neutrino::InteractionChannels::CCOnly :
          (neutrinoChannels == "nc" ? neutrino::InteractionChannels::NCOnly :
                                   neutrino::InteractionChannels::CCAndNC));
    ParticleCut<SubWriter<terrainapp::ShowerOutput>> cut(
        emcut*1_GeV, emcut*1_GeV, hadcut*1_GeV, mucut*1_GeV, mucut*1_GeV, false, diagnostics);
    terrainapp::AuditedEmThinning thinning(emthin*energy*1_GeV, maxWeight, true,diagnostics.ledger);
    // Preserve this process order across scalar and Kokkos execution.
    auto sequence = make_sequence(neutrinos, hadrons, decay, em, losses, diagnostics, thinning, cut);
    // 5. Tracking owns curved DEM intersections. Interfaces switch material,
    //    rather than acting as the absorbing observation plane of an air shower.
    interfaces::MeshInterfaceTracking tracking(*mesh.mesh);
    TerrainStack stack;
    stack.addParticle(std::make_tuple(pid, energy*1_GeV-get_mass(pid), dir, injection, 0_s));
    OutputManager manager(output, seed, sceneFile, false);
    manager.add("terrain", diagnostics);
    manager.startOfLibrary();
    std::clog << "[terrain] calculators ready; starting shower transport\n";
    ownsOutput = true;
    summary["complete"] = false;
    summary["backend"] = "cpu-proposal";
    summary["radio"] = radioEnabled?"kokkos-interface-coreas-zhs":"disabled";
    summary["magnetic_field_model"] = magnetic;
    summary["magnetic_year"] = year;
    summary["air_magnetic_field_enu_T"] = scene.atmosphere.air_magnetic_field_enu_T;
    summary["magnetic_height_reference"] = "WGS84 ellipsoidal; atmosphere density uses ASL";
    summary["magnetic_coefficients_file"] = igrfFile;
    summary["rock_magnetic_field_enu_T"] = scene.material.magnetic_field_T;
    summary["scene"] = scene.yaml;
    summary["material_interface"]["outside_region"] = 0;
    summary["material_interface"]["inside_region"] = 1;
    summary["material_interface"]["inside_medium"] = scene.material.id;
    summary["resolved_material"] = terrainapp::materialConfig(scene.material);
    summary["fluka_target_initialization"] = low.expandedSetup() ?
        "one bootstrap compound; individual elemental targets" : "stock elemental regions";
    summary["fluka_isotope_policy"] = "elemental natural-isotope FLUKA targets; Z-only native setup API";
    summary["legacy_material_overrides"] = scene.legacy_material_overrides;
    summary["material_ionisation_model"] = scene.material.ionisation_model==interfaces::IonisationModel::BraggSternheimer ?
        "Bragg log-I + approximate condensed Sternheimer" :
        (scene.material.ionisation_model==interfaces::IonisationModel::NistDensityScaled ? "NIST compound, density scaled" : "explicit coefficients");
    summary["hadron_continuous_loss"] = hadronLoss.getConfig();
    for(auto const& v:em.nativeCalculatorViews()) {
      if(v.projectile!=Code::Photon) continue;
      auto const& m=*v.medium; YAML::Node p;
      p["name"]=m.GetName(); p["density_g_cm3"]=m.GetMassDensity();
      p["I_eV"]=m.GetI(); p["Z_over_A"]=m.GetZA();
      p["radiation_length_g_cm2"]=m.GetRadiationLength();
      p["components"]=m.GetNumComponents(); p["composition_hash"]=v.medium_hash;
      p["photon_pair_LPM"]=v.photon_pair_lpm!=nullptr;
      summary["proposal_materials"].push_back(p);
    }
    summary["material_interface"]["legacy_rock_label"] = "enclosed region, not necessarily SiO2";
    summary["mesh_sha256"] = scene.mesh_hash;
    summary["terrain_straight_intersection"] = "indexed-shared-vertices-v2";
    summary["terrain_curved_intersection"] = "quadratic-plane-v1 (indexed feature ownership pending)";
    summary["seed"] = seed;
    summary["primary"] = primary;
    summary["energy_GeV"] = energy;
    summary["position_m"] = position;
    summary["direction"] = direction;
    summary["conditional_forced_CC"] = forcedCC;
    summary["conditional_forced_NC"] = forceNc;
    summary["neutrino_channels"] = neutrinoChannels;
    summary["neutrino_limits"] = "CTW CC/NC >=10 TeV; no nuclear shadowing, sub-10-TeV weak transport or validated CC spin/energy-loss depolarization";
    summary["physics_capabilities"]["weak_transport_below_10TeV"] = false;
    summary["physics_capabilities"]["event_CC_spin_density"] = false;
    summary["physics_capabilities"]["tau_material_depolarization"] = false;
    summary["physics_capabilities"]["tauola_prescribed_longitudinal_density_matrix"] = true;
    summary["transported_CC_taus"] = true;
    summary["cpu_only_species"] = "hadrons, muons, taus, neutrinos; Kokkos transports gamma/e-/e+ only";
    summary["emcut_GeV"] = emcut;
    summary["hadcut_GeV"] = hadcut;
    summary["mucut_GeV"] = mucut;
    summary["emthin"] = emthin;
    summary["max_weight"] = maxWeight;
    summary["max_step_m"] = std::isfinite(maxStep) ? YAML::Node(maxStep) : YAML::Node("unlimited");
    summary["transport_step_limit"] = transportStepLimit;
    summary["transport_window_ns"] = std::isfinite(windowNs) ? YAML::Node(windowNs) : YAML::Node("stock time cut");
    summary["hadronic_models"] = "FLUKA + QGSJet-II.04 (80 GeV/nucleon)";
    save(summary);
    auto started = std::chrono::steady_clock::now();
    // 6. Same models and output lifecycle, two execution schedulers.
    auto runBackend = [&](auto& selectedTracking) {
      using Tracking = std::decay_t<decltype(selectedTracking)>;
      if (backend == "proposal") {
        Cascade<Tracking, decltype(sequence), OutputManager, TerrainStack>
            cascade(env, selectedTracking, sequence, manager, stack);
        if (force) cascade.forceInteraction();
        cascade.run();
      }
#ifdef CORSIKA8_WITH_KOKKOS_EM
      else {
        // Device lifecycle and fallback stay behind the application adapter.
        terrainapp::runAcceleratedTerrain(scene, env, selectedTracking, sequence, manager,
            stack, neutrinos, hadrons, decay, em, losses, continuous, diagnostics, thinning,
            cut, productionCut,
            {threads, device, batch, memoryMiB, energy, emcut, mucut, emthin, seed, force, auxCache,
             maxWeight, maxStep, windowNs*1.e-9, emScheduler=="resident", residentCapacity,radioEnabled,std::filesystem::path(output)/"radio",
             residentWavefront,residentRecords,radioBatch},
            summary, save);
      }
#endif
    };
    if (magnetic == "none") {
      // Retain StraightTrajectory; use the shared indexed DEM boundary solver.
      terrain::StraightTrackingView straight(tracking.flatTerrain());
      runBackend(straight);
    } else {
      runBackend(tracking);
    }
    // 7. Flush all streamed records before marking the shower complete.
    manager.endOfLibrary();
    summary["diagnostics"] = diagnostics.getSummary();
    summary["energy_ledger"] = diagnostics.energySummary(energy,cut.statistics());
    summary["neutrino"] = neutrinos.summary();
    summary["tau"] = decay.summary();
    summary["shower_seconds"] = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-started).count();
    summary["complete"] = true;
    save(summary);
    std::cout << summary["diagnostics"] << '\n';
    return 0;
  } catch (std::exception const& e) {
    if (ownsOutput) {
      summary["complete"] = false;
      summary["error"] = e.what();
      try { save(summary); } catch (...) {}
    }
    std::cerr << "terrain reference: " << e.what() << '\n';
    return 1;
  }
}
