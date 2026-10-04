#include "OverheadOptions.hpp"
// Isolated application: native C++ EGS4 e-/e+/gamma + unchanged C8 non-EM
// physics. Experiment-only harness; production main remains unmodified.
// Muon acceleration is opt-in, with the original scalar path retained for controls.
#include "Egs4C8HostHandler.hpp"
#include "Egs4C8Frontier.hpp"
#include "Egs4FeReference.hpp"
#include "Egs4EventPlan.hpp"
#include <corsika/accelerator/em/detail/KokkosEmRunSession.hpp>
#include <corsika/accelerator/em/PhysicalAcceleratedEmRouter.hpp>
#include <corsika/accelerator/em/common/ProposalCpuFallbackHandler.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>
#include <corsika/accelerator/radio/common/RadioSnapshotBuilder.hpp>
#include <corsika/framework/process/DynamicInteractionProcess.hpp>
#include <corsika/modules/thinning/EMThinning.hpp>
#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/framework/process/SwitchProcessSequence.hpp>
#include <corsika/accelerator/em/common/EnvironmentSnapshotBuilder.hpp>
#include <corsika/accelerator/em/common/CorsikaOutputSink.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/modules/Sibyll.hpp>
#include <corsika/modules/FLUKA.hpp>
#include <corsika/modules/PROPOSAL.hpp>
#include <corsika/modules/QGSJetII.hpp>
#include <corsika/modules/QGSJetIII.hpp>
#include <corsika/modules/sophia/InteractionModel.hpp>
#include <corsika/modules/proposal/ThresholdPhotoproductionModel.hpp>
#include <corsika/modules/proposal/HadronicInteractionModelFallback.hpp>
#include <corsika/modules/pythia8/Decay.hpp>
#include <corsika/modules/TAUOLA.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <corsika/modules/BetheBlochPDG.hpp>
#include <corsika/modules/ObservationPlane.hpp>
#include <corsika/modules/LongitudinalProfile.hpp>
#include <corsika/modules/writers/EnergyLossWriter.hpp>
#include <corsika/modules/writers/LongitudinalWriter.hpp>
#include <corsika/modules/writers/ProductionWriter.hpp>
#include <corsika/modules/writers/SubWriter.hpp>
#include <corsika/modules/radio/CoREAS.hpp>
#include <corsika/modules/radio/ZHS.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <corsika/modules/radio/propagators/TabulatedFlatAtmospherePropagator.hpp>
#include <corsika/output/OutputManager.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupTrajectory.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>
#include <CLI/CLI.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>

extern "C" void egs4_reset_sibyll_gaussian_cache();

namespace {
using namespace corsika;
namespace app=c7_egs4::application;
using MediumInterface=IMagneticFieldModel<IRefractiveIndexModel<IMediumPropertyModel<IMediumModel>>>;
template<class T>class ReferenceIndex:public GladstoneDaleRefractiveIndex<T> {
  bool matched_;double refractivity_;
public:
  template<class... Args>ReferenceIndex(double n,Point const& sea,bool matched,Args&&... args)
    :GladstoneDaleRefractiveIndex<T>(n,sea,std::forward<Args>(args)...),matched_(matched),refractivity_(n-1.){}
  double getRefractiveIndex(Point const& p)const override {
    if(!matched_)return GladstoneDaleRefractiveIndex<T>::getRefractiveIndex(p);
    auto rho0=(1222.6562/994186.38)*1_g/(1_cm*1_cm*1_cm);
    return 1.+refractivity_*this->getMassDensity(p)/rho0;
  }
};
template<class T>using ExtraMedium=UniformMagneticField<ReferenceIndex<MediumPropertyModel<T>>>;
using Env=Environment<MediumInterface>;
using ShowerStack=setup::HybridStack<Env>;
// DynamicInteractionProcess type-erases the existing models but privately
// inherits its process tag. Expose the public tag without adding histogram IO.
struct HostHadronProcess:InteractionProcess<HostHadronProcess> {
  DynamicInteractionProcess<ShowerStack>& model;
  explicit HostHadronProcess(DynamicInteractionProcess<ShowerStack>& m):model(m){}
  CrossSectionType getCrossSection(Code projectile,Code target,FourMomentum const& p,FourMomentum const& t)const {
    return model.getCrossSection(projectile,target,p,t);
  }
  void doInteraction(ShowerStack::stack_view_type& view,Code projectile,Code target,FourMomentum const& p,FourMomentum const& t) {
    model.doInteraction(view,projectile,target,p,t);
  }
};
struct UnusedWriter {};
struct NonEmAudit:ContinuousProcess<NonEmAudit>,SecondariesProcess<NonEmAudit> {
  std::uint64_t steps{},muon_steps{},hadron_steps{},em_daughters{},daughters{};
  template<class Particle,class Track>LengthType getMaxStepLength(Particle const&,Track const&)const {
    return std::numeric_limits<double>::infinity()*1_m;
  }
  template<class Particle>ProcessReturn doContinuous(Step<Particle> const& s,bool) {
    auto pid=s.getParticlePre().getPID();
    if(pid==Code::Electron||pid==Code::Positron||pid==Code::Photon)
      throw std::runtime_error("An EM particle escaped the native EGS4 router");
    // Match production Cascade::run(): finish when the stack drains, not after
    // a fixed total number of steps. A large shower legitimately exceeds the
    // old 3-million-step fixture cap. The coordinator retains its wall-time
    // deadline; routing, queue and output-overflow checks remain independent.
    ++steps;
    if(pid==Code::MuMinus||pid==Code::MuPlus)++muon_steps;
    if(is_hadron(pid))++hadron_steps;
    return ProcessReturn::Ok;
  }
  template<class View>void doSecondaries(View& view) {
    for(auto&& p:view){++daughters;auto id=p.getPID();
      if(id==Code::Electron||id==Code::Positron||id==Code::Photon)++em_daughters;}
  }
};
// C8's cut module is reused for new native non-EM daughters before transport.
// No fictitious track or physical energy update: the supplied path is zero.
template<class Cut>struct HostArrivalCut {
  Cut& cut;
  template<class Particle>bool operator()(Particle& p,app::HostRequest const&) {
    StraightTrajectory track(Line(p.getPosition(),p.getDirection()*constants::c),0_s);
    Step step(p,track);auto result=cut.doContinuous(step,false);
    if(result==ProcessReturn::ParticleAbsorbed)return false;
    if(result!=ProcessReturn::Ok)throw std::runtime_error("Unexpected native host arrival cut result");
    return true;
  }
};
}
int main(int argc,char** argv) {
  try {
    CLI::App cli{"Isolated native C++ EGS4 EM + existing C8 hadronic/muon application"};
    std::string table,output_path;int pdg=2212,source_count=1;double energy_GeV=10.,height_m=5000.,stepfc=1.;
    app::FeReference reference;reference.options(cli);
    std::uint64_t seed=731;bool host_reference=false,force_interaction=false,force_decay=false;
    double thin_threshold=0.,thin_weight=1.;std::string capture,frontier;unsigned worker=0;
    int queue_capacity=65536,frontier_batch=0;double gpu_memory_fraction=0.;std::string radio_backend="kokkos";
    std::string muon_backend="cpu";
    std::string profile_backend="cpu",queue_workspace="fresh";
    std::string event_plan;double event_timeout=3600.;std::uint64_t initialization_seed=731;
    cli.add_option("--initialization-seed",initialization_seed,"Fixed model-initialization RNG, independent of event order (event-plan mode only)");
    cli.add_option("--event-plan",event_plan,"Fixed-physics campaign; resident devices, separate outputs, continuous C8 host RNG streams");
    cli.add_option("--event-timeout",event_timeout)->check(CLI::PositiveNumber);
    bool split_rare_kernels=false;
    cli.add_flag("--split-rare-kernels",split_rare_kernels,"Separate rare final states from the native transport kernel (experimental)");
    cli.add_option("--profile-backend",profile_backend,"kokkos: resident native EM profile/deposition; cpu: per-step reference")
      ->check(CLI::IsMember({"cpu","kokkos"}));
    cli.add_option("--queue-workspace",queue_workspace,"reuse: retain device scratch buffers; fresh: allocation reference")
      ->check(CLI::IsMember({"fresh","reuse"}));
    double requested_muon_cut_GeV=-1.;
    cli.add_option("--muon-cut-GeV",requested_muon_cut_GeV,"Explicit muon kinetic cut for transport controls; Fe default remains 0.05 GeV")
      ->check(CLI::PositiveNumber);
    cli.add_option("--muon-backend",muon_backend,"cpu: frozen reference; kokkos: existing C8 PROPOSAL muons with immediate EGS4 daughter handoff")
      ->check(CLI::IsMember({"cpu","kokkos"}));
    cli.add_option("--radio-backend",radio_backend,"kokkos: resident projection on the selected EM execution space; cpu: scalar reference")
      ->check(CLI::IsMember({"kokkos","cpu"}));
    cli.add_option("--queue-capacity",queue_capacity,"Maximum resident native histories (overflow fails without dropping particles)")
      ->check(CLI::Range(8192,std::numeric_limits<int>::max()/5));
    cli.add_option("--gpu-memory-fraction",gpu_memory_fraction,"CUDA working-set budget as fraction of total memory, bounded by free memory; not a utilization target")
      ->check(CLI::Range(.01,1.));
    cli.add_option("--frontier-batch",frontier_batch,"Unstarted EM roots fed together; zero selects the budget-aware default")
      ->check(CLI::Range(0,std::numeric_limits<int>::max()/10));
    cli.add_option("--thin-threshold-GeV",thin_threshold)->check(CLI::NonNegativeNumber);
    cli.add_option("--thin-max-weight",thin_weight)->check(CLI::PositiveNumber);
    cli.add_option("--capture-frontier",capture);cli.add_option("--frontier",frontier);
    cli.add_option("--worker-id",worker)->check(CLI::Range(1,255));
    cli.add_option("--table",table)->required();cli.add_option("-f,--output",output_path)->required();
    cli.add_option("--pdg",pdg);cli.add_option("--energy-GeV",energy_GeV)->check(CLI::PositiveNumber);
    cli.add_option("--height-m",height_m)->check(CLI::Range(1100.01,112750.));
    cli.add_option("--stepfc",stepfc)->check(CLI::PositiveNumber);cli.add_option("--seed",seed);
    cli.add_option("--source-count",source_count,"Coincident primary cohort for interface tests; 1 is an individual shower")->check(CLI::Range(1,1024));
    cli.add_flag("--host-reference",host_reference);cli.add_flag("--force-interaction",force_interaction);
    cli.add_flag("--force-decay",force_decay);CLI11_PARSE(cli,argc,argv);
    reference.defaults(cli,pdg,energy_GeV,height_m,thin_threshold,thin_weight);
    auto events=app::readEventPlan(event_plan,{seed,output_path,capture,frontier,{}});
    if(!event_plan.empty()) {
      seed=events.front().seed;output_path=events.front().output;
      capture=events.front().capture;frontier=events.front().frontier;
    }
    if(reference.enabled&&!cli.count("--queue-capacity"))queue_capacity=1048576;
    if(gpu_memory_fraction>0.&&host_reference&&capture.empty())
      throw std::invalid_argument("GPU memory fraction is not applicable to CPU EM workers");
    if(force_interaction&&force_decay)throw std::invalid_argument("Choose at most one forced primary action");
    if(!frontier.empty()&&(force_interaction||force_decay||!capture.empty()))
      throw std::invalid_argument("A worker imports roots without forced primary actions or prefix capture");
    if(!capture.empty()&&boost::filesystem::exists(capture))throw std::invalid_argument("Capture file already exists");
    auto primary=pdg==1000260560?get_nucleus_code(56,26):convert_from_PDG(static_cast<PDGCode>(pdg));
    if(energy_GeV*1_GeV<get_mass(primary))throw std::invalid_argument("Primary energy below rest mass");
    if((force_interaction||force_decay)&&(pdg==11||pdg==-11||pdg==22))
      throw std::invalid_argument("Forced scalar actions are forbidden for native EM primaries");
    if(boost::filesystem::exists(output_path))throw std::invalid_argument("Output directory already exists");
    logging::set_level(logging::level::err);auto begin=std::chrono::steady_clock::now();
    // Diagnostic only: retain constructor order, physics objects and RNG streams.
    std::vector<std::pair<char const*,double>> initialization_stages;
    auto initialization_mark=begin;
    auto initialized=[&](char const* stage) {
      auto now=std::chrono::steady_clock::now();
      initialization_stages.emplace_back(stage,std::chrono::duration<double>(now-initialization_mark).count());
      initialization_mark=now;
    };
    auto& rng=RNGManager<>::getInstance();
    for(auto name:{"cascade","pythia","proposal","sibyll","sophia","fluka","qgsjet","thinning"})rng.registerRandomStream(name);
    if(reference.enabled)for(auto name:{"qgsjetIII","epos","urqmd","primary_particle"})rng.registerRandomStream(name);
    // Per-worker C8 generator streams must not all restart with the same seed.
    // Native EM keys additionally include globally disjoint history IDs.
    auto host_seed=worker?gpu::em::detail::splitmix64(seed^worker):seed;
    // Keep model initialization independent of which event happens to be first;
    // single-event legacy semantics stay unchanged.
    rng.setSeed(event_plan.empty()?host_seed:initialization_seed);
    // Engine::setSeed changes the key only, not counter/cache. Retain pristine
    // engines in their registered stream domains; do not merely reseed a used RNG.
    auto const pristine_rngs=rng.getRngs();
    Env env;auto cs=env.getCoordinateSystem();auto R=constants::EarthRadius::Mean;
    Point center{cs,0_m,0_m,0_m},sea{cs,0_m,0_m,R},ground{cs,0_m,0_m,R+1100_m};
    std::array<double,3> B=reference.enabled?std::array<double,3>{27.232962e-6,0.,-49.332426e-6}:
      std::array<double,3>{2.e-5,1.e-5,-4.e-5};
    double n0=reference.enabled?1.000312:1.0003;
    MagneticFieldVector field{cs,{B[0]*1_T,B[1]*1_T,B[2]*1_T}};
    create_5layer_atmosphere<MediumInterface,ExtraMedium>(
      env,AtmosphereId::LinsleyUSStd,center,field,n0,sea,reference.enabled,Medium::AirDry1Atm);
    double theta=reference.enabled?M_PI/3.:0.,phi=reference.enabled?225.*M_PI/180.:0.;
    DirectionVector direction{cs,{sin(theta)*cos(phi),sin(theta)*sin(phi),-cos(theta)}};
    auto robs=R+1100_m,rinj=R+height_m*1_m;
    auto distance=-robs*cos(theta)+sqrt(rinj*rinj-robs*robs*sin(theta)*sin(theta));
    Point injection=reference.enabled?ground-direction*distance:Point{cs,0_m,0_m,R+height_m*1_m};
    Point top=reference.enabled?injection:Point{cs,0_m,0_m,R+112799_m};
    ShowerAxis axis=reference.enabled?ShowerAxis(top,(ground-top)*1.2,env):ShowerAxis(top,ground-top,env,false,3000);
    auto dX=(reference.enabled?5.:1.)*1_g/square(1_cm);
    EnergyLossWriter deposition(axis,dX);
    LongitudinalWriter profile(axis,dX,ProfileCrossingMode::Both);
    LongitudinalProfile<SubWriter<decltype(profile)>> longprof(profile);
    ObservationPlane<setup::Tracking> observation(Plane(ground,DirectionVector{cs,{0.,0.,1.}}),DirectionVector{cs,{1.,0.,0.}});
    ObserverCollection<TimeDomainObserver> coreas_detector,zhs_detector;
    if(!c7_egs4::overhead::noAntennas()) {
    if(reference.enabled) {
      auto positions=reference.positions();
      for(std::size_t i=0;i<positions.size();++i) {
        auto p=positions[i];Point point{cs,p[0]*1_m,p[1]*1_m,R+p[2]*1_m};
        auto trigger=(injection-point).getNorm()/constants::c-200_ns;
        coreas_detector.addObserver(TimeDomainObserver("CoREAS_Antenna_"+std::to_string(i),point,cs,trigger,400_ns,5_GHz,trigger));
      }
    } else for(double distance:{50.,100.,200.}) {
      TimeDomainObserver observer("antenna_"+std::to_string(int(distance)),Point{cs,distance*1_m,0_m,R+1100_m},cs,
        -10000_ns,100000_ns,2.e8_Hz,0_ns);
      coreas_detector.addObserver(observer);zhs_detector.addObserver(observer);
    }
    }
    auto propagator=make_tabulated_flat_atmosphere_radio_propagator(env,top,reference.enabled?sea:ground,reference.enabled?1_m:100_m);
    // The native backend owns all EM energy cuts and stopped-positron handling.
    // C8 non-EM cuts and models are reused; no C8 cut removes an EM daughter first.
    auto muon_cut=reference.enabled?.05_GeV:.01_GeV,tau_cut=reference.enabled?.3_GeV:.01_GeV;
    if(cli.count("--muon-cut-GeV"))muon_cut=requested_muon_cut_GeV*1_GeV;
    ParticleCut<SubWriter<decltype(deposition)>> cut(0_GeV,0_GeV,.3_GeV,muon_cut,tau_cut,true,deposition);
    for(auto id:proposal::tracked)set_energy_production_threshold(id,.5_MeV);
    initialized("geometry_and_radio");
    auto sibyll=std::make_shared<corsika::sibyll::Interaction>(get_all_elements_in_universe(env),setup::C7trackedParticles);
    DynamicInteractionProcess<ShowerStack> hadron_he=reference.enabled?
      DynamicInteractionProcess<ShowerStack>{std::make_shared<corsika::qgsjetIII::Interaction>()}:
      DynamicInteractionProcess<ShowerStack>{sibyll};
    corsika::fluka::Interaction hadron_le(get_all_elements_in_universe(env));
    HostHadronProcess counted_he{hadron_he};
    auto hadrons=make_select([](auto const& p){return p.getEnergyNN()<80_GeV;},hadron_le,counted_he);
    corsika::sophia::InteractionModel sophia_model;proposal::ThresholdPhotoproductionModel threshold;
    proposal::HadronicInteractionModelFallback photon_le(sophia_model,threshold);
    proposal::LazyHadronicInteractionModel<corsika::qgsjetII::InteractionModel> qgsjet;
    proposal::HadronicInteractionModelFallback photon_he(sibyll->getHadronInteractionModel(),qgsjet);
    proposal::HadronicPhotonModel photons(photon_le,photon_he,80_GeV);
    initialized("hadronic_models");
    // Unmodified C8 PROPOSAL objects are retained exclusively for non-EM leptons.
    // They still initialize their usual tables; initialization is timed separately.
    proposal::Interaction muon_interactions(env,photon_le,photon_he,80_GeV);
    initialized("proposal_interactions");
    proposal::ContinuousProcess<SubWriter<decltype(deposition)>> muon_continuous(env,deposition);
    initialized("proposal_continuous");
    auto is_mu_tau=[](auto const& p){auto id=p.getPID();return id==Code::MuMinus||id==Code::MuPlus||id==Code::TauMinus||id==Code::TauPlus;};
    NullModel null;
    auto muons=make_select(is_mu_tau,muon_interactions,null);
    auto muon_losses=make_select(is_mu_tau,muon_continuous,null);
    BetheBlochPDG<SubWriter<decltype(deposition)>> hadron_losses(deposition);
    auto continuous=make_select([](auto const& p){return is_hadron(p.getPID());},hadron_losses,muon_losses);
    corsika::pythia8::Decay pythia_decay;
    corsika::tauola::Decay tau_decay(corsika::tauola::Helicity::LeftHanded);
    auto decay=make_select([](auto const& p){return p.getPID()==Code::TauMinus||p.getPID()==Code::TauPlus;},tau_decay,pythia_decay);
    EMThinning thinning{thin_threshold*1_GeV,thin_weight,true};
    // C8 owns the shared Kokkos lifetime when its muon path is selected.
    // Construct it before EGS4 and destroy it after all EGS4 device views.
    auto environment=gpu::em::makeCorsika7AtmosphereSnapshot(AtmosphereId::LinsleyUSStd,
      {0.,0.,0.},3,(R+1100_m)/1_m,B,.2);
    initialized("host_sequence");
    accelerator::em::detail::KokkosEmRunSession muon_session;
    std::unique_ptr<app::Session> native_session; // destroyed before muon_session/Kokkos
    app::GpuMemoryBudget persistent_budget;
    app::Configuration immutable_config;
    for(std::size_t event_index=0;event_index<events.size();++event_index) {
    auto const& event=events[event_index];
    double const event_wait_s=app::awaitEvent(event,event_timeout);
    if(event_index) {
      begin=std::chrono::steady_clock::now();initialization_mark=begin;
      initialization_stages={{"geometry_and_radio",0.},{"hadronic_models",0.},
        {"proposal_interactions",0.},{"proposal_continuous",0.},{"host_sequence",0.}};
    }else {
      auto wait=std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(event_wait_s));
      begin+=wait;initialization_mark+=wait;
    }
    seed=event.seed;output_path=event.output;capture=event.capture;frontier=event.frontier;
    host_seed=worker?gpu::em::detail::splitmix64(seed^worker):seed;
    if(!event_plan.empty()&&event_index==0) {
      // Like original C8's multi-shower loop, host model/RNG state continues
      // across the campaign. FLUKA has state beyond its injected uniform RNG:
      // reseeding only that RNG does NOT reproduce isolated later events.
      // Reset once before the campaign, never partially rewind between events.
      // GPU per-history keys/output accumulators remain event-specific.
      // Assign values in place: model callbacks retain references to map nodes.
      for(auto const& [name,engine]:pristine_rngs)rng.getRandomStream(name)=engine;
      rng.setSeed(host_seed);
      // Fortran adapters capture a COPY of the C8 engine, plus a 16-value
      // buffer. Reset both through their public setter; reseeding RNGManager
      // alone does not reset these generators between events.
      corsika::connect_random_stream("sibyll",::sibyll23d::set_rng_function);
      egs4_reset_sibyll_gaussian_cache();
      corsika::connect_random_stream("fluka",::fluka::set_rng_function);
      corsika::connect_random_stream("sophia",::sophia::set_rng_function);
      corsika::connect_random_stream("qgsjet",::QGSJetII04::set_rng_function);
      if(reference.enabled)corsika::connect_random_stream("qgsjetIII",::QGSJetIII::set_rng_function);
    }
    { // All event consumers/routers are destroyed before EVENT_DONE is published.
    // Propagation tables and observer geometry stay resident; output processes
    // own an independent library/shower ID and diagnostic summary per event.
    RadioProcess<decltype(coreas_detector),CoREAS<decltype(coreas_detector),decltype(propagator)>,decltype(propagator)> coreas(coreas_detector,propagator);
    RadioProcess<decltype(zhs_detector),ZHS<decltype(zhs_detector),decltype(propagator)>,decltype(propagator)> zhs(zhs_detector,propagator);
    NonEmAudit audit;
    auto sequence=make_sequence(hadrons,decay,muons,audit,continuous,coreas,zhs,longprof,observation,thinning,cut);
    auto secondary_hooks=make_sequence(audit,thinning,cut);
    HostArrivalCut<decltype(cut)> arrival{cut};ShowerStack stack;
    app::HostHandler<ShowerStack,Env,decltype(photons),decltype(secondary_hooks),decltype(arrival)>
      handler(env,cs,photons,secondary_hooks,arrival);
    auto const photon_energy_before=photons.energyLedgerStatistics().weighted_energy_residual_GeV;
    accelerator::em::KokkosEmBackend* proposal_backend=nullptr;
    if(muon_backend=="kokkos"&&capture.empty()) {
      gpu::em::GpuEmConfig m;
      m.min_batch_size=64;m.resident_batch_limit=4096;m.memory_fraction=.05;
      m.random_seed=seed;m.shower_id=1;m.em_transport_cut_MeV=.5;
      m.muon_transport_cut_MeV=muon_cut/1_MeV;m.thinning={thin_threshold,thin_weight,thin_threshold>0.,true};
      m.resident_cross_species=false; // daughters must never stay in PROPOSAL
      accelerator::em::detail::KokkosSessionBeginResult result;
      try {
        result=muon_session.beginProposalNative(environment,m,
          {energy_GeV*1000.,.5,muon_cut/1_MeV,.5},
          muon_interactions.nativeCalculatorViews(),muon_continuous.nativeCalculatorViews());
      }catch(...) {
        if(auto table=muon_session.loadedNativeTable()) {
          for(auto const& col:table->total_rate_columns)
            std::cerr<<"PROPOSAL total-rate domain pid="<<col.pdg_id<<" low_MeV="<<col.spline.axis.low<<'\n';
          for(auto const& col:table->utility_columns)
            std::cerr<<"PROPOSAL range domain pid="<<col.pdg_id<<" low_MeV="<<col.spline.axis.low
              <<" lower_limit_MeV="<<col.lower_energy_limit_MeV<<'\n';
        }
        throw;
      }
      if(!result.muon_transport_available)throw std::runtime_error("Existing C8 backend cannot transport muons");
      proposal_backend=result.backend;
    }
    initialized("proposal_device_session");
    if(!native_session)native_session=std::make_unique<app::Session>(table);
    auto& session=*native_session;app::Configuration cfg;
    initialized("native_egs4_session");
    if(radio_backend=="kokkos")cfg.radio=gpu::radio::makeGpuRadioConfig(env,top,reference.enabled?sea:ground,
      reference.enabled?1_m:100_m,coreas_detector,zhs_detector,true,!reference.enabled);
    app::GpuMemoryBudget memory_budget;
    if(gpu_memory_fraction>0.&&capture.empty()) {
      memory_budget=event_index?persistent_budget:session.gpuMemoryBudget(gpu_memory_fraction,cfg.radio);
      persistent_budget=memory_budget;
      if(cli.count("--queue-capacity")&&queue_capacity>memory_budget.queue_capacity)
        throw std::length_error("Explicit queue capacity exceeds the GPU memory budget");
      if(!cli.count("--queue-capacity"))queue_capacity=memory_budget.queue_capacity;
    }
    if(!frontier_batch)frontier_batch=reference.enabled?
      (memory_budget.queue_capacity?std::min(65536,queue_capacity/32):4096):64;
    if(frontier_batch>queue_capacity/2)throw std::length_error("Frontier batch exceeds half of queue capacity");
    if(memory_budget.queue_capacity)std::cerr<<std::setprecision(6)
      <<"Native GPU budget: fraction="<<gpu_memory_fraction<<" total_MiB="<<memory_budget.total_bytes/1048576.
      <<" free_MiB="<<memory_budget.free_bytes/1048576.<<" working_MiB="<<memory_budget.working_bytes/1048576.
      <<" reserve_MiB="<<memory_budget.reserve_bytes/1048576.<<" bytes_per_history="<<memory_budget.bytes_per_history
      <<" queue_capacity="<<queue_capacity<<" frontier_batch="<<frontier_batch<<" (demand allocated)\n";
    cfg.environment=environment;
    cfg.reuse_queue_workspace=queue_workspace=="reuse";
    cfg.split_rare_kernels=split_rare_kernels;
    if(profile_backend=="kokkos") {
      auto& p=cfg.profile;p.enabled=p.accumulate_on_device=true;
      p.crossing_mode=profile.getCrossingMode();p.output_bin_count=deposition.GetNBins();
      if(p.output_bin_count!=profile.getNBins())throw std::runtime_error("Native profile writer bin mismatch");
      p.output_bin_width_g_per_cm2=dX/(1_g/square(1_cm));p.energy_loss_threshold_g_per_cm2=1.e-4;
      p.fixed_point_weight_limit=2.*source_count*energy_GeV/.0005;
      p.fixed_point_energy_limit_GeV=std::max(2.*source_count*energy_GeV,1.);
      auto start=axis.getStart().getCoordinates(cs);auto direction=axis.getDirection().getComponents(cs);
      for(int j=0;j<3;++j){p.axis_start_position_m[j]=start[j]/1_m;p.axis_direction[j]=direction[j].magnitude();}
      p.axis_step_length_m=axis.getSteplength()/1_m;
      for(auto x:axis.getGrammageSupport())p.axis_grammage_g_per_cm2.push_back(x/(1_g/square(1_cm)));
    }
    cfg.earth_radius_m=R/1_m;cfg.seed=seed;cfg.shower_id=1;cfg.queue_capacity=queue_capacity;cfg.maximum_waves=200000;
    cfg.stepfc=stepfc;cfg.electron_mass_MeV=get_mass(Code::Electron)/1_MeV;
    cfg.thinning={thin_threshold,thin_weight,thin_threshold>0.?1U:0U,1U};
    // One mass convention at the C++/C8 boundary, including nearly-rest recoil
    // hadrons. The algorithms take these masses as explicit inputs.
    auto mass=[](int id){return get_mass(convert_from_PDG(static_cast<PDGCode>(id)))/1_MeV;};
    cfg.muon_mass_MeV=mass(13);cfg.charged_pion_mass_MeV=mass(211);cfg.neutral_pion_mass_MeV=mass(111);
    cfg.proton_mass_MeV=mass(2212);cfg.neutron_mass_MeV=mass(2112);
    cfg.phi_mass_MeV=mass(333);cfg.omega_mass_MeV=mass(223);cfg.rho_mass_MeV=mass(113);
    cfg.charged_kaon_mass_MeV=mass(321);cfg.long_kaon_mass_MeV=mass(130);cfg.short_kaon_mass_MeV=mass(310);cfg.eta_mass_MeV=mass(221);
    cfg.electron_total_cut_MeV=cfg.electron_mass_MeV+.5;cfg.photon_cut_MeV=.5;
    cfg.resolve_rare_vertices=cfg.native_photonuclear_vertices=cfg.native_prompt_rho_decay=cfg.native_prompt_resonance_decay=true;
    cfg.retain_across_showers=!event_plan.empty();
    if(!event_index)immutable_config=cfg;
    else {cfg=immutable_config;cfg.seed=seed;cfg.resume_retained_shower=session.hasRetainedShower();}
    ProductionWriter production(axis,dX);UnusedWriter interaction;
    gpu::em::CorsikaOutputSink sink(cs,deposition,profile,production,observation,interaction,coreas,zhs,true,cfg.radio.enabled);
    double discarded=0.;std::uint64_t prompt=0;app::OutputCallbacks callbacks;
    callbacks.profile=[&](auto const& result){sink.onGpuProfile(result);};
    std::uint64_t written_steps=0;auto last_progress=std::chrono::steady_clock::now();
    callbacks.step=[&](auto const& r){
      sink.onStep(r);
      if(++written_steps%100000==0&&reference.enabled) {
        auto now=std::chrono::steady_clock::now();
        if(now-last_progress>=std::chrono::seconds(30)) {
          auto progress=session.progressStatistics();
          std::cerr<<"Native Fe progress: worker="<<worker<<" written_steps="<<written_steps
            <<" active="<<session.activeParticles()
            <<" radio_backend="<<progress.radio_execution_space<<" radio_batches="<<progress.radio_projection_batches
            <<" radio_cuda_kernel_ms="<<progress.radio_kernel_ms
            <<" elapsed_s="<<std::chrono::duration<double>(now-begin).count()<<'\n';last_progress=now;
        }
      }
    };callbacks.radio=[&](auto const& r){sink.onRadioTrack(r);};
    callbacks.radio_waveforms=[&](auto const& waveforms,std::uint64_t tracks){sink.onGpuRadioWaveforms(waveforms,tracks);};
    callbacks.observation=[&](auto const& r){sink.onObservation(r);};
    callbacks.discarded=[&](double E,unsigned){discarded+=E;};callbacks.prompt_decay=[&](auto const&){++prompt;};
    auto region=[](auto const* node){if(!node||!node->hasModelProperties())throw std::runtime_error("Native EM outside air");return 3;};
    using MuonFallback=gpu::em::ProposalCpuFallbackHandler<ShowerStack,decltype(muon_interactions),decltype(sequence),Env>;
    // GPU tables are exported from this exact interaction object. Resolve its
    // existing calculator by the selected interaction hash: the transport cut
    // is NOT the optimized PROPOSAL stochastic/continuous table boundary.
    // A forced .5 MeV calculator here changes that hash for muon radiative
    // channels. No alternate-cut calculator or resampled process is needed.
    MuonFallback muon_fallback(muon_interactions,sequence,env,cs,seed,1);
    app::MuonOutputSink<decltype(sink)> muon_sink(sink);
    using MuonRouter=gpu::em::AcceleratedEmRouterImpl<ShowerStack,MuonFallback,decltype(muon_sink),app::ProposalMuonBackend>;
    std::unique_ptr<app::ProposalMuonBackend> muon_adapter;
    std::unique_ptr<MuonRouter> muon_router;
    if(proposal_backend) {
      muon_adapter=std::make_unique<app::ProposalMuonBackend>(*proposal_backend);
      muon_router=std::make_unique<MuonRouter>(*muon_adapter,cs,cfg.environment,muon_fallback,muon_sink);
      muon_router->setRetainRecords(false);
    }
    app::FrontierRouter<ShowerStack,decltype(handler),decltype(region),MuonRouter> router(
      session,cfg,callbacks,cs,handler,region,stack,host_reference,capture,frontier,worker,frontier_batch,
      muon_router.get(),muon_adapter.get(),muon_backend=="kokkos");
    OutputManager output(output_path,seed,"isolated native C++ EGS4; unchanged C8 hadrons/muons",false);
    output.add("energyloss",deposition);output.add("profile",profile);output.add("particles",observation);
    output.add("CoREAS",coreas);output.add("ZHS",zhs);output.startOfLibrary();
    setup::Tracking tracking(.2);
    for(int i=0;frontier.empty()&&i<source_count;++i)stack.addParticle(std::make_tuple(primary,energy_GeV*1_GeV-get_mass(primary),direction,injection,0_ns));
    HybridCascade<setup::Tracking,decltype(sequence),OutputManager,ShowerStack,decltype(router)> cascade(env,tracking,sequence,output,stack,router);
    if(force_interaction)cascade.forceInteraction();if(force_decay)cascade.forceDecay();
    initialized("application_outputs");
    auto ready=std::chrono::steady_clock::now();cascade.run();auto finished=std::chrono::steady_clock::now();output.endOfLibrary();
    if(stack.getEntries()!=0||router.pending())throw std::runtime_error("Queues did not drain");
    for(auto const& entry:cascade.timingStatistics().scalar_steps_by_pdg)
      if(entry.first==11||entry.first==-11||entry.first==22)throw std::runtime_error("Native EM routed to scalar");
    auto const& native=router.statistics();auto const& host=handler.statistics();
    gpu::em::PhysicalAcceleratedEmRouterStatistics muon_stats;
    if(muon_router)muon_stats=muon_router->statistics();
    std::ofstream record(output_path+"/native_egs4_run.json");record<<std::setprecision(17)
      <<"{\n  \"status\": \"completed\",\n  \"primary_pdg\": "<<pdg<<",\n  \"primary_energy_GeV\": "<<energy_GeV
      <<",\n  \"seed\": "<<seed<<",\n  \"height_m\": "<<height_m<<",\n  \"stepfc\": "<<stepfc
      <<",\n  \"persistent_process\": "<<(!event_plan.empty()?"true":"false")
      <<",\n  \"model_initialization_seed\": "<<(event_plan.empty()?host_seed:initialization_seed)
      <<",\n  \"host_campaign_seed\": "<<(worker?gpu::em::detail::splitmix64(events.front().seed^worker):events.front().seed)
      <<",\n  \"process_pid\": "<<getpid()<<",\n  \"event_index\": "<<event_index
      <<",\n  \"event_wait_s\": "<<event_wait_s<<",\n  \"native_session_reused\": "<<(cfg.resume_retained_shower?"true":"false")
      <<",\n  \"event_rng_policy\": \""<<(event_plan.empty()?"legacy standalone":"campaign-continuous C8 host streams; event-specific GPU keys; reproduce campaign prefix, not isolated later seeds")<<"\""
      <<",\n  \"source_count\": "<<source_count<<",\n  \"mass_convention\": \"C8 ParticleProperties\""
      <<",\n  \"queue_capacity\": "<<queue_capacity
      <<",\n  \"radio_backend_requested\": \""<<radio_backend<<"\""
      <<",\n  \"profile_backend_requested\": \""<<profile_backend<<"\""
      <<",\n  \"queue_workspace\": \""<<queue_workspace<<"\""
      <<",\n  \"profile_execution_space\": \""<<native.profile_execution_space<<"\""
      <<",\n  \"profile_accumulated_steps\": "<<native.profile_steps
      <<",\n  \"profile_fixed_point_overflows\": "<<native.profile_fixed_point_overflows
      <<",\n  \"profile_invalid_records\": "<<native.profile_invalid_records
      <<",\n  \"host_output_records\": "<<native.host_output_records
      <<",\n  \"host_output_bytes\": "<<native.host_output_bytes
      <<",\n  \"queue_workspace_allocations\": "<<native.queue_workspace_allocations
      <<",\n  \"queue_workspace_reuses\": "<<native.queue_workspace_reuses
      <<",\n  \"queue_processed_records\": "<<native.queue_processed_records
      <<",\n  \"queue_peak_active\": "<<native.queue_peak_active
      <<",\n  \"queue_advance_wall_ms\": "<<native.queue_advance_wall_ms
      <<",\n  \"queue_append_calls\": "<<native.queue_append_calls
      <<",\n  \"queue_append_records\": "<<native.queue_append_records
      <<",\n  \"queue_append_copied_active_records\": "<<native.queue_append_copied_active_records
      <<",\n  \"queue_append_wall_ms\": "<<native.queue_append_wall_ms
      <<",\n  \"split_rare_kernels_requested\": "<<(split_rare_kernels?"true":"false")
      <<",\n  \"queue_split_rare_waves\": "<<native.queue_split_rare_waves
      <<",\n  \"queue_split_rare_records\": "<<native.queue_split_rare_records
      <<",\n  \"queue_timing_scope\": \"Host elapsed advance including allocation, copies, kernels and synchronization; excludes radio/profile/output consumers\""
      <<",\n  \"muon_backend_requested\": \""<<muon_backend<<"\""
#ifdef EGS4_STABLE_MUON_IONIZATION
      <<",\n  \"experimental_stable_muon_ionization\": true"
#else
      <<",\n  \"experimental_stable_muon_ionization\": false"
#endif
      <<",\n  \"proposal_muon_steps\": "<<(muon_adapter?muon_adapter->muonSteps():0)
      <<",\n  \"proposal_muon_waves\": "<<(muon_adapter?muon_adapter->muonWaves():0)
      <<",\n  \"proposal_muon_em_handoffs\": "<<(muon_adapter?muon_adapter->emTransfers():0)
      <<",\n  \"proposal_muon_execution_space\": \""<<(proposal_backend?proposal_backend->statistics().accelerator_backend:"not_initialized")<<"\""
      <<",\n  \"proposal_muon_cpu_decays\": "<<muon_stats.forced_cpu_decays_executed
      <<",\n  \"proposal_muon_specified_cpu_final_states\": "<<muon_stats.specified_cpu_final_states
      <<",\n  \"proposal_muon_generic_cpu_steps\": "<<muon_stats.cpu_fallback_steps_executed
      <<",\n  \"proposal_muon_memory_spills\": "<<muon_stats.particles_returned_for_cpu_memory_spill
      <<",\n  \"proposal_muon_backend_wall_ms\": "<<muon_stats.lepton_backend_wall_time_ms
      <<",\n  \"radio_execution_space\": \""<<native.radio_execution_space<<"\""
      <<",\n  \"native_radio_tracks\": "<<native.radio_tracks
      <<",\n  \"radio_projected_tracks\": "<<native.radio.lepton_tracks
      <<",\n  \"radio_projection_batches\": "<<native.radio_projection_batches
      <<",\n  \"radio_track_observer_pairs\": "<<native.radio.track_observer_pairs
      <<",\n  \"radio_coreas_contributions\": "<<native.radio.coreas_contributions
      <<",\n  \"radio_zhs_contributions\": "<<native.radio.zhs_contributions
      <<",\n  \"radio_fixed_point_overflows\": "<<native.radio.fixed_point_overflows
      <<",\n  \"radio_cuda_kernel_ms\": "<<native.radio_kernel_ms
      <<",\n  \"radio_projection_wall_ms\": "<<native.radio_projection_wall_ms
      <<",\n  \"radio_timing_scope\": \"CUDA events around resident track conversion and CoREAS/ZHS kernels; no CPU callback radio when kokkos is selected\""
      <<",\n  \"frontier_batch\": "<<frontier_batch
      <<",\n  \"gpu_memory_fraction\": "<<gpu_memory_fraction
      <<",\n  \"gpu_memory_plan_active\": "<<(memory_budget.queue_capacity?"true":"false")
      <<",\n  \"gpu_total_bytes\": "<<memory_budget.total_bytes<<",\n  \"gpu_free_bytes_at_plan\": "<<memory_budget.free_bytes
      <<",\n  \"gpu_requested_bytes\": "<<memory_budget.requested_bytes<<",\n  \"gpu_working_budget_bytes\": "<<memory_budget.working_bytes
      <<",\n  \"gpu_reserve_bytes\": "<<memory_budget.reserve_bytes<<",\n  \"gpu_bytes_per_history_bound\": "<<memory_budget.bytes_per_history
      <<",\n  \"atmosphere\": \"LinsleyUSStd\",\n  \"observation_height_m\": 1100,\n  \"sea_level_refractive_index\": "<<n0
      <<",\n  \"magnetic_field_T\": ["<<B[0]<<","<<B[1]<<","<<B[2]<<"]"
      <<",\n  \"fe_reference\": "<<(reference.enabled?"true":"false")<<",\n  \"zenith_deg\": "<<(reference.enabled?60:0)
      <<",\n  \"azimuth_deg\": "<<(reference.enabled?225:0)<<",\n  \"antenna_count\": "<<(c7_egs4::overhead::noAntennas()?0:(reference.enabled?110:3))
      <<",\n  \"profile_bin_g_cm2\": "<<(reference.enabled?5:1)<<",\n  \"profile_crossings\": \"both\""
      <<",\n  \"refractivity_normalization\": \""<<(reference.enabled?"common_sea_level_density":"legacy_layer_reference")<<"\""
      <<",\n  \"native_electron_kinetic_cut_MeV\": 0.5,\n  \"native_photon_cut_MeV\": 0.5,\n  \"hadron_kinetic_cut_GeV\": 0.3,\n  \"muon_kinetic_cut_GeV\": "<<muon_cut/1_GeV
      <<",\n  \"hadronic_models\": \"C8 "<<(reference.enabled?"QGSJet-III":"SIBYLL 2.3d")<<" / FLUKA; transition 80 GeV\",\n  \"muon_transport\": \"unchanged C8 PROPOSAL\",\n  \"muon_polarization\": \"existing C8 treatment; C7 scalar polarization not mapped to spin\""
      <<",\n  \"forced_interaction\": "<<(force_interaction?"true":"false")<<",\n  \"forced_decay\": "<<(force_decay?"true":"false")
      <<",\n  \"execution_space\": \""<<native.execution_space<<"\",\n  \"unthinned\": "<<(thin_threshold>0.?"false":"true")
      <<",\n  \"thin_threshold_GeV\": "<<thin_threshold<<",\n  \"thin_max_weight\": "<<thin_weight
      <<",\n  \"thinning_hillas\": "<<native.thinning_hillas<<",\n  \"thinning_statistical\": "<<native.thinning_statistical
      <<",\n  \"thinning_removed\": "<<native.thinning_removed<<",\n  \"prefix_only\": "<<(!capture.empty()?"true":"false")
      <<",\n  \"worker_id\": "<<worker<<",\n  \"scalar_em_steps\": 0,\n  \"scalar_steps\": "<<audit.steps
      <<",\n  \"host_generator_seed\": "<<(worker?gpu::em::detail::splitmix64(events.front().seed^worker):events.front().seed)
      <<",\n  \"event_key_seed\": "<<host_seed<<",\n  \"frontier_roots_consumed\": "<<router.importedRoots()
      <<",\n  \"scalar_muon_steps\": "<<audit.muon_steps<<",\n  \"scalar_hadron_steps\": "<<audit.hadron_steps
      <<",\n  \"c8_em_daughters\": "<<audit.em_daughters<<",\n  \"native_injections\": "<<native.injections
      <<",\n  \"native_steps\": "<<native.steps<<",\n  \"native_children\": "<<native.children<<",\n  \"native_host_requests\": "<<native.host_requests
      <<",\n  \"host_muons_returned\": "<<host.transported_muons<<",\n  \"host_hadrons_returned\": "<<host.transported_hadrons
      <<",\n  \"host_photonuclear_vertices\": "<<host.selected_photonuclear<<",\n  \"host_arrival_cuts\": "<<host.arrival_cuts
      <<",\n  \"native_target_rest_energy_GeV\": "<<native.target_rest_energy_GeV<<",\n  \"host_photonuclear_energy_residual_GeV\": "<<photons.energyLedgerStatistics().weighted_energy_residual_GeV-photon_energy_before
      <<",\n  \"native_discarded_available_energy_GeV\": "<<discarded<<",\n  \"native_prompt_decays\": "<<prompt
      <<",\n  \"initialization_s\": "<<std::chrono::duration<double>(ready-begin).count();
    record<<",\n  \"initialization_stages_s\": {";
    for(std::size_t i=0;i<initialization_stages.size();++i)
      record<<(i?", ":"")<<'\"'<<initialization_stages[i].first<<"\": "<<initialization_stages[i].second;
    auto const& phase=cascade.timingStatistics();
    record<<"}"
      <<",\n  \"hybrid_total_s\": "<<phase.total_run_time_ms/1000.
      <<",\n  \"shower_output_start_s\": "<<phase.output_start_time_ms/1000.
      <<",\n  \"shower_output_end_s\": "<<phase.output_end_time_ms/1000.
      <<",\n  \"shower_transport_s\": "<<(phase.total_run_time_ms-phase.output_start_time_ms-phase.output_end_time_ms)/1000.
      <<",\n  \"shower_wall_s\": "<<std::chrono::duration<double>(finished-ready).count()<<"\n}\n";
    record.close();if(!record)throw std::runtime_error("Could not write completion record");
    std::cout<<"completed output="<<output_path<<" native_steps="<<native.steps<<" scalar_muon_steps="<<audit.muon_steps
      <<" scalar_hadron_steps="<<audit.hadron_steps<<" scalar_em_steps=0\n";
    }
    if(!event_plan.empty())app::completeEvent(event,event_index);
    }
    return 0;
  }catch(std::exception const& e){std::cerr<<"Incomplete isolated shower: "<<e.what()<<'\n';return 1;}
}
