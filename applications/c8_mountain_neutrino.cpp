/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 * Independent finite convex rock application. No air-shower defaults are changed.
 */
#include <corsika/framework/core/Cascade.hpp>
#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/framework/process/SwitchProcessSequence.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/framework/utility/CorsikaData.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <corsika/modules/BetheBlochPDG.hpp>
#include <corsika/modules/FLUKA.hpp>
#include <corsika/modules/PROPOSAL.hpp>
#include <corsika/modules/proposal/ThresholdPhotoproductionModel.hpp>
#include <corsika/modules/Pythia8.hpp>
#include <corsika/modules/QGSJetII.hpp>
#include <corsika/modules/Sophia.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <corsika/modules/thinning/EMThinning.hpp>
#include <corsika/modules/neutrino/MountainNeutrinoInteraction.hpp>
#include <corsika/modules/tracking/TrackingConvexPolyhedron.hpp>
#include <corsika/modules/writers/SubWriter.hpp>
#include <corsika/modules/radio/CoREAS.hpp>
#include <corsika/modules/radio/ZHS.hpp>
#include <corsika/modules/radio/RadioProcess.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/output/OutputManager.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>
#include "detail/mountain/MountainDiagnostics.hpp"
#include "detail/mountain/MountainRadio.hpp"
#if defined(CORSIKA8_WITH_KOKKOS_EM)
#include <corsika/accelerator/em/common/ConvexEnvironmentSnapshot.hpp>
#include <corsika/accelerator/em/common/ProcessSequenceCompatibility.hpp>
#include <corsika/accelerator/em/detail/KokkosEmRunSession.hpp>
#include <corsika/accelerator/em/PhysicalAcceleratedEmRouter.hpp>
#include <corsika/accelerator/em/common/ProposalCpuFallbackHandler.hpp>
#endif
#include <CLI/CLI.hpp>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <csignal>
#include <unistd.h>

using namespace corsika;
using namespace corsika::units::si;
namespace mountain = corsika::applications::mountain;
using MediumInterface =
    IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
using MountainEnvironment = Environment<MediumInterface>;
using MountainStack = setup::HybridStack<MountainEnvironment>;

namespace {
std::array<double,3> triple(YAML::Node const& n, char const* name) {
  if (!n || !n.IsSequence() || n.size()!=3)
    throw std::invalid_argument(std::string(name)+" must have three numbers");
  std::array<double,3> r{n[0].as<double>(),n[1].as<double>(),n[2].as<double>()};
  for(auto v:r) if(!std::isfinite(v)) throw std::invalid_argument("nonfinite coordinate");
  return r;
}
template<class T> T value(YAML::Node const& n,char const* key,T fallback) {
  return n && n[key] ? n[key].as<T>() : fallback;
}
void positive(double v,char const* name) {
  if(!std::isfinite(v)||v<=0.)throw std::invalid_argument(std::string(name)+" must be positive and finite");
}
ConvexPolyhedron geometry(YAML::Node const& n,CoordinateSystemPtr const& cs,
                          std::filesystem::path const& directory) {
  auto const shape=n["shape"].as<std::string>();
  if(shape=="box")return ConvexPolyhedron::box(cs,triple(n["center_m"],"center_m"),
                                               triple(n["half_lengths_m"],"half_lengths_m"));
  if(shape=="tetrahedron") {
    auto vs=n["vertices_m"];
    if(!vs.IsSequence()||vs.size()!=4)throw std::invalid_argument("tetrahedron needs four vertices");
    auto p=[&](std::size_t i){auto v=triple(vs[i],"vertex");return Point(cs,v[0]*1_m,v[1]*1_m,v[2]*1_m);};
    return ConvexPolyhedron::tetrahedron({p(0),p(1),p(2),p(3)});
  }
  if(shape=="obj")return ConvexPolyhedron::fromOBJ(
      directory/n["file"].as<std::string>(),cs,value(n,"scale_m",1.)*1_m);
  throw std::invalid_argument("shape must be box, tetrahedron, or convex triangular obj");
}
void save(std::filesystem::path const& p,YAML::Node const& n) {
  std::ofstream f(p);if(!f)throw std::runtime_error("cannot open "+p.string());
  f<<n<<'\n';if(!f)throw std::runtime_error("cannot write "+p.string());
}
}

int main(int argc,char** argv) {
  CLI::App app{"Finite convex SiO2 mountain: CC neutrinos and Kokkos EM transport"};
  std::string configFile, outputPath="mountain_output", backend="proposal", execution;
  std::string auxiliaryCache;
  long seed=67101;
  int threads=0,device=0;
  std::size_t batch=4096,capacity=0;
  double memoryFraction=0.7;
  bool geometryOnly=false;
  app.add_option("--config",configFile,"YAML geometry, primary and physics configuration")->required()->check(CLI::ExistingFile);
  app.add_option("-f,--output",outputPath,"New output directory; existing results are never overwritten");
  app.add_option("--em-backend",backend,"proposal or kokkos (EM and internal radio together)")->check(CLI::IsMember({"proposal","kokkos"}));
  app.add_option("--kokkos-execution",execution,"openmp, cuda, hip or sycl; empty uses build default");
  app.add_option("--kokkos-num-threads",threads)->check(CLI::NonNegativeNumber);
  app.add_option("--kokkos-device",device)->check(CLI::NonNegativeNumber);
  app.add_option("--gpu-min-batch",batch)->check(CLI::PositiveNumber);
  app.add_option("--gpu-resident-batch-limit",capacity,"0 uses beta5 memory-aware capacity");
  app.add_option("--gpu-memory-fraction",memoryFraction)->check(CLI::Range(0.01,0.95));
  app.add_option("--gpu-aux-cache-dir",auxiliaryCache,"Optional writable proposal-native auxiliary cache");
  app.add_option("-s,--seed",seed)->check(CLI::NonNegativeNumber);
  app.add_flag("--geometry-only",geometryOnly,"Validate configuration/chord, without physics or runtime initialization");
  CLI11_PARSE(app,argc,argv);
  try {
    logging::set_level(logging::level::warn);
    auto const started=std::chrono::steady_clock::now();
    auto yaml=YAML::LoadFile(configFile);
    MountainEnvironment env;
    auto const cs=env.getCoordinateSystem();
    auto volume=geometry(yaml["geometry"],cs,std::filesystem::absolute(configFile).parent_path());
    auto primary=yaml["primary"];
    auto pid=convert_from_PDG(static_cast<PDGCode>(value(primary,"pdg",12)));
    double const energy=value(primary,"energy_GeV",1.e4);
    positive(energy,"primary energy");
    if(is_neutrino(pid))neutrino::MountainNeutrinoInteraction::validatePrimary(pid,energy);
    else if(pid!=Code::Photon&&pid!=Code::Electron&&pid!=Code::Positron)
      throw std::invalid_argument("primary must be a neutrino, or photon/electron/positron for transport validation");
    if(is_neutrino(pid)&&std::abs(static_cast<int>(get_PDG(pid)))!=12)
      throw std::invalid_argument("first mountain application supports nu_e/anti-nu_e CC only; other flavors require separate decay/regeneration validation");
    if(energy*1_GeV<=get_mass(pid))throw std::invalid_argument("total energy is below particle mass");
    auto xyz=triple(primary["position_m"],"primary position_m");
    Point injection(cs,xyz[0]*1_m,xyz[1]*1_m,xyz[2]*1_m);
    auto uv=triple(primary["direction"],"direction");
    DirectionVector direction(cs,{uv[0],uv[1],uv[2]});
    positive(direction.getNorm(),"direction norm"); direction=direction.normalized();
    auto mode=value(primary,"sampling",std::string("natural_cc"));
    bool forced=mode=="forced_vertex_cc";
    if(mode!="natural_cc"&&!forced)throw std::invalid_argument("sampling must be natural_cc or forced_vertex_cc");
    if(forced&&!is_neutrino(pid))throw std::invalid_argument("forced_vertex_cc requires a neutrino primary");
    // Force only at an explicitly specified interior point. Natural incoming rays
    // are clipped analytically; no atmosphere or artificial bounding sphere.
    auto interval=volume.intersectRay(injection,direction);
    if(!interval.intersects)throw std::invalid_argument("primary ray misses mountain");
    TimeType injectionTime=0_s;
    if(!volume.contains(injection)) {
      if(forced)throw std::invalid_argument("forced vertex must be inside mountain");
      auto distance=std::max(0.,interval.entry_m)+1.e-7;
      injection=injection+direction*(distance*1_m);
      injectionTime=distance*1_m/constants::c;
      interval=volume.intersectRay(injection,direction);
    }
    if(!interval.intersects||interval.exit_m<=1e-7)throw std::invalid_argument("primary has no positive rock chord");
    double const chord=interval.exit_m;
    auto material=yaml["material"];
    if(value(material,"name",std::string("silica"))!="silica")
      throw std::invalid_argument("first mountain application supports homogeneous silica (Si28:O16=1:2) only");
    double const density=value(material,"density_g_cm3",2.2);
    double const n=value(material,"refractive_index",2.0);
    positive(density,"density");positive(n,"refractive_index");
    if(n<1.)throw std::invalid_argument("refractive_index must be at least one");
    auto physics=yaml["physics"];
    auto const emcut=value(physics,"emcut_GeV",0.0005)*1_GeV;
    auto const hadcut=value(physics,"hadcut_GeV",0.3)*1_GeV;
    auto const mucut=value(physics,"mucut_GeV",0.3)*1_GeV;
    auto const taucut=value(physics,"taucut_GeV",0.3)*1_GeV;
    for(auto cut:{emcut,hadcut,mucut,taucut})positive(cut/1_GeV,"transport cut");
    double const emthin=value(physics,"emthin",1.e-6);
    double maxWeight=value(physics,"max_weight",0.);
    if(!std::isfinite(emthin)||emthin<0.||emthin>1.||!std::isfinite(maxWeight)||maxWeight<0.)
      throw std::invalid_argument("invalid thinning configuration");
    if(maxWeight==0.)maxWeight=0.5*emthin*energy;
    auto radio=yaml["radio"];
    bool const radioEnabled=value(radio,"enabled",false);
    double const tStart=value(radio,"start_ns",-10.);
    double const duration=value(radio,"duration_ns",4000.);
    double const sampleRate=value(radio,"sample_rate_GHz",1.);
    positive(duration,"radio duration");positive(sampleRate,"sample rate");
    if(!std::isfinite(tStart)||duration*sampleRate>10000000.)
      throw std::invalid_argument("invalid or excessive radio time window");
    ObserverCollection<TimeDomainObserver> coreasDetector,zhsDetector;
    if(radioEnabled) {
      auto observers=radio["observers_m"];
      if(!observers.IsSequence()||!observers.size())throw std::invalid_argument("internal radio requires observers_m");
      for(std::size_t i=0;i<observers.size();++i) {
        auto v=triple(observers[i],"observer");Point p(cs,v[0]*1_m,v[1]*1_m,v[2]*1_m);
        if(!volume.contains(p))throw std::invalid_argument("all observers must be inside mountain: exterior radio is unsupported");
        auto name="observer_"+std::to_string(i);
        coreasDetector.addObserver(TimeDomainObserver(name,p,cs,tStart*1_ns,duration*1_ns,sampleRate*1_GHz,0_ns));
        zhsDetector.addObserver(TimeDomainObserver(name,p,cs,tStart*1_ns,duration*1_ns,sampleRate*1_GHz,0_ns));
      }
    }
    YAML::Node summary;
    summary["complete"]=false;summary["seed"]=seed;summary["primary"]=primary;
    summary["geometry"]=yaml["geometry"];summary["chord_m"]=chord;
    summary["density_g_cm3"]=density;summary["refractive_index"]=n;
    summary["em_backend"]=backend;summary["sampling"]=mode;
    summary["physics"]["emcut_GeV"]=emcut/1_GeV;
    summary["physics"]["hadcut_GeV"]=hadcut/1_GeV;
    summary["physics"]["mucut_GeV"]=mucut/1_GeV;
    summary["physics"]["taucut_GeV"]=taucut/1_GeV;
    summary["physics"]["emthin"]=emthin;
    summary["physics"]["effective_max_weight"]=maxWeight;
    summary["radio"]=radio;
    summary["hadronic_models"]="FLUKA + QGSJet-II.04 (80 GeV/nucleon transition)";
    summary["limits"]="CC-only free-nucleon impulse approximation; no NC/nuclear shadowing; absorbing exterior; no refraction/reflection/attenuation";
    summary["forced_event_is_conditional"]=forced;
    if(geometryOnly) { std::cout<<summary<<'\n';return 0; }
#if defined(CORSIKA8_WITH_KOKKOS_EM)
    if(backend=="kokkos") {
      execution=accelerator::em::resolveKokkosExecutionBackend(execution);
      if(execution!="openmp"&&threads>1)throw std::invalid_argument("GPU execution requires single-thread host scheduling");
    }
#else
    if(backend=="kokkos")throw std::invalid_argument("this binary was built without Kokkos");
#endif
    if(std::filesystem::exists(outputPath))throw std::invalid_argument("output already exists; choose a new output path");
    auto& rng=RNGManager<>::getInstance();
    for(auto const* stream:{"cascade","qgsjet","sophia","pythia","fluka","proposal","thinning"})rng.registerRandomStream(stream);
    rng.setSeed(seed);
    using Rock=UniformRefractiveIndex<MediumPropertyModel<UniformMagneticField<HomogeneousMedium<MediumInterface>>>>;
    NuclearComposition composition({get_nucleus_code(28,14),Code::Oxygen},{1./3.,2./3.});
    auto rock=env.createNode<ConvexPolyhedron>(volume);
    rock->setModelProperties(std::make_unique<Rock>(n,Medium::SiliconDioxideFusedQuartz,
        MagneticFieldVector(cs,{0_T,0_T,0_T}),density*1_g/(1_cm*1_cm*1_cm),composition));
    env.getUniverse()->addChild(std::move(rock));
    auto const prodCut=std::min({emcut,hadcut,mucut,taucut});
    for(auto code:{Code::Photon,Code::Electron,Code::Positron,Code::MuMinus,Code::MuPlus,Code::TauMinus,Code::TauPlus})
      set_energy_production_threshold(code,prodCut);
    double extent=1.;for(auto const& v:volume.verticesMeters()) {
      Point p(cs,v[0]*1_m,v[1]*1_m,v[2]*1_m);extent=std::max(extent,std::abs((p-injection).dot(direction)/1_m)+1.);
    }
    mountain::Diagnostics diagnostics(volume,injection,direction,extent);
    corsika::fluka::Interaction lowEnergy(std::set<Code>{get_nucleus_code(28,14),Code::Oxygen});
    // FLUKA's periodic diagnostic handler uses non-async-signal-safe stdio.
    // The physics calls remain synchronous, but Kokkos may own helper threads.
    if(std::signal(SIGALRM,SIG_IGN)==SIG_ERR)
      throw std::runtime_error("cannot disable FLUKA diagnostic SIGALRM timer");
    ::alarm(0);
    qgsjetII::Interaction highEnergy;
    corsika::sophia::InteractionModel photoLow;
    proposal::ThresholdPhotoproductionModel thresholdPhoto;
    proposal::HadronicInteractionModelFallback photoLowWithThreshold(photoLow,thresholdPhoto);
    proposal::Interaction em(env,photoLowWithThreshold,highEnergy,80_GeV);
    proposal::ContinuousProcess<SubWriter<mountain::Diagnostics>> continuous(env,diagnostics);
    BetheBlochPDG<SubWriter<mountain::Diagnostics>> hadronLoss(diagnostics);
    auto continuousSequence=make_select([](MountainStack::particle_type const& p){return is_hadron(p.getPID());},hadronLoss,continuous);
    auto hadronicSequence=make_select([](MountainStack::particle_type const& p){return p.getEnergyNN()<80_GeV;},lowEnergy,highEnergy);
    pythia8::Decay decay;
    neutrino::MountainNeutrinoInteraction neutrinoInteraction(setup::C7trackedParticles,
        forced?neutrino::SamplingMode::ForcedVertexCC:neutrino::SamplingMode::NaturalCC);
    ParticleCut<SubWriter<mountain::Diagnostics>> cut(emcut,emcut,hadcut,mucut,taucut,false,diagnostics);
    diagnostics.endpoint_cut=[&](Code code,HEPEnergyType kinetic,TimeType time) {
      // ParticleCut's public thresholds are global particle properties. This
      // application has no additional per-PID cut map or invisible cut.
      auto const energyPerParticle=is_nucleus(code)?kinetic/get_nucleus_A(code):kinetic;
      return energyPerParticle<get_kinetic_energy_propagation_threshold(code)||time>10_ms;
    };
    EMThinning thinning(emthin*energy*1_GeV,maxWeight,true);
    mountain::HomogeneousMountainRadioPropagator<MountainEnvironment> propagator(env,n,
        [&](Point const& p){return volume.contains(p);});
    RadioProcess<decltype(coreasDetector),CoREAS<decltype(coreasDetector),decltype(propagator)>,decltype(propagator)>
        coreas(coreasDetector,propagator);
    RadioProcess<decltype(zhsDetector),ZHS<decltype(zhsDetector),decltype(propagator)>,decltype(propagator)>
        zhs(zhsDetector,propagator);
    diagnostics.radio_merge=[&](auto const& waves){mountain::mergeMountainGpuRadioWaveforms(waves,coreasDetector,zhsDetector);};
    // CPU physics and output order are local to this application only.
    auto sequence=make_sequence(neutrinoInteraction,hadronicSequence,decay,em,
        continuousSequence,coreas,zhs,diagnostics,thinning,cut);
    tracking_line::ConvexPolyhedronTracking tracking;
    MountainStack stack;
    stack.addParticle(std::make_tuple(pid,energy*1_GeV-get_mass(pid),direction,injection,injectionTime));
    OutputManager output(outputPath,seed,configFile,false);
    output.add("mountain",diagnostics);
    output.add("CoREAS",coreas);output.add("ZHS",zhs);
    output.startOfLibrary();
    save(std::filesystem::path(outputPath)/"mountain_run.yaml",summary);
    auto const showerStart=std::chrono::steady_clock::now();
    if(backend=="proposal") {
      Cascade<decltype(tracking),decltype(sequence),OutputManager,MountainStack> cascade(env,tracking,sequence,output,stack);
      if(forced)cascade.forceInteraction();
      cascade.run();
    }
#if defined(CORSIKA8_WITH_KOKKOS_EM)
    else {
      namespace emgpu=corsika::gpu::em;
      using Policy=emgpu::GpuEmStepProcessPolicy;
      using Registry=emgpu::GpuEmStepProcessRegistry<
          emgpu::GpuEmStepProcessRegistration<decltype(neutrinoInteraction),Policy::InapplicableToRoutedEm>,
          emgpu::GpuEmStepProcessRegistration<decltype(hadronicSequence),Policy::InapplicableToRoutedEm>,
          emgpu::GpuEmStepProcessRegistration<decltype(decay),Policy::DeferredToCpu>,
          emgpu::GpuEmStepProcessRegistration<decltype(em),Policy::ReplacedOnDevice>,
          emgpu::GpuEmStepProcessRegistration<decltype(continuousSequence),Policy::ReplacedOnDevice>,
          emgpu::GpuEmStepProcessRegistration<decltype(coreas),Policy::ReplayedFromDeviceRecord>,
          emgpu::GpuEmStepProcessRegistration<decltype(zhs),Policy::ReplayedFromDeviceRecord>,
          emgpu::GpuEmStepProcessRegistration<mountain::Diagnostics,Policy::ReplayedFromDeviceRecord>,
          emgpu::GpuEmStepProcessRegistration<decltype(thinning),Policy::ReplacedOnDevice>,
          emgpu::GpuEmStepProcessRegistration<decltype(cut),Policy::ReplacedOnDevice>>;
      Registry::validateOrThrow<decltype(sequence)>();
      auto snapshot=emgpu::makeHomogeneousConvexSnapshot(density,volume);
      emgpu::GpuEmConfig gpuConfig;
      gpuConfig.device=device;gpuConfig.min_batch_size=batch;
      gpuConfig.resident_batch_limit=capacity;gpuConfig.memory_fraction=memoryFraction;
      gpuConfig.em_transport_cut_MeV=emcut/1_MeV;gpuConfig.muon_transport_cut_MeV=mucut/1_MeV;
      gpuConfig.physics_source=emgpu::GpuPhysicsSource::ProposalNative;
      gpuConfig.random_seed=seed;gpuConfig.shower_id=0;
      gpuConfig.thinning.enabled=emthin>0.;gpuConfig.thinning.threshold_GeV=emthin*energy;
      gpuConfig.thinning.maximum_weight=maxWeight;gpuConfig.thinning.erase_zero_weight=1;
      if(radioEnabled)gpuConfig.radio=mountain::makeHomogeneousMountainRadioConfig(n,cs,coreasDetector,zhsDetector,[&](Point const& p){return volume.contains(p);});
      accelerator::em::KokkosRuntimeConfig runtime;
      runtime.execution_backend=execution;runtime.threads=threads;runtime.device=device;
      accelerator::em::detail::KokkosEmRunSession session(runtime,auxiliaryCache);
      auto const stochasticCut=proposal::optimized_proposal_energy_cut(prodCut);
      auto begin=session.beginProposalNative(snapshot,gpuConfig,
          {energy*1000.,emcut/1_MeV,mucut/1_MeV,stochasticCut/1_MeV},
          em.nativeCalculatorViews(),continuous.nativeCalculatorViews());
      using Fallback=emgpu::ProposalCpuFallbackHandler<MountainStack,decltype(em),decltype(sequence),MountainEnvironment>;
      Fallback fallback(em,sequence,env,cs,seed,0,stochasticCut);
      using Backend=accelerator::em::KokkosEmBackend;
      using Router=emgpu::PhysicalAcceleratedEmRouter<MountainStack,Backend,Fallback,mountain::Diagnostics>;
      Router router(*begin.backend,cs,snapshot,fallback,diagnostics);
      router.setRetainRecords(false);router.setFailOnUnexpectedFallback(true);
      HybridCascade<decltype(tracking),decltype(sequence),OutputManager,MountainStack,Router>
          cascade(env,tracking,sequence,output,stack,router);
      if(forced)cascade.forceInteraction();
      cascade.run();
      auto const& stats=begin.backend->statistics();
      summary["accelerator"]["backend"]=stats.accelerator_backend;
      summary["accelerator"]["concurrency"]=stats.accelerator_concurrency;
      summary["accelerator"]["device"]=stats.accelerator_device_name;
      summary["accelerator"]["wavefronts"]=stats.wavefronts;
      summary["accelerator"]["particles_advanced"]=stats.particles_advanced;
      summary["accelerator"]["proposal_fallbacks"]=stats.proposal_fallbacks;
      summary["accelerator"]["specified_cpu_final_states"]=fallback.statistics().specified_interactions;
      summary["accelerator"]["queue_overflows"]=stats.queue_overflows;
      summary["accelerator"]["kernel_time_ms"]=stats.kernel_time_ms;
      summary["accelerator"]["radio_tracks"]=stats.radio.lepton_tracks;
      summary["accelerator"]["native_table_hash"]=emgpu::tables::toHex(stats.native_table_hash);
      summary["accelerator"]["auxiliary_cache_hash"]=emgpu::tables::toHex(stats.auxiliary_cache_hash);
      summary["accelerator"]["steps"]=diagnostics.getSummary()["accelerated_steps"];
      summary["accelerator"]["physics_source"]="proposal-native";
    }
#endif
    summary["shower_wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-showerStart).count();
    summary["neutrino"]=neutrinoInteraction.summary();
    summary["diagnostics"]=diagnostics.getSummary();
    output.endOfLibrary();
    summary["complete"]=true;
    summary["total_wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    save(std::filesystem::path(outputPath)/"mountain_run.yaml",summary);
    std::cout<<summary<<'\n';
    return 0;
  } catch(std::exception const& error) {
    std::cerr<<"c8_mountain_neutrino: "<<error.what()<<'\n';return 1;
  }
}
