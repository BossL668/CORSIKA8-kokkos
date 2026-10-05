#pragma once

#include <corsika/modules/EGS4.hpp>
#include <corsika/modules/egs4/HostHandler.hpp>
#include <corsika/modules/egs4/Router.hpp>
#include <corsika/accelerator/em/common/EnvironmentSnapshotBuilder.hpp>
#include <corsika/accelerator/em/common/CorsikaOutputSink.hpp>
#include <corsika/accelerator/em/common/ProposalCpuFallbackHandler.hpp>
#include <corsika/accelerator/em/PhysicalAcceleratedEmRouter.hpp>
#include <corsika/accelerator/em/detail/KokkosEmRunSession.hpp>
#include <corsika/accelerator/radio/common/RadioSnapshotBuilder.hpp>
#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/output/BaseOutput.hpp>
#include <corsika/output/ShowerSummary.hpp>
#include "../air_shower_multigpu/Frontier.hpp"

namespace corsika::applications::air_shower {
namespace egs4 = ::c7_egs4::application;
namespace egs4_detail {
// Preserve all non-EM cuts. EGS4, not the scalar cut, owns EM termination
// (including positrons already below the transport cut at their birth).
template<class Cut> struct NonEmCut : ContinuousProcess<NonEmCut<Cut>>,
    SecondariesProcess<NonEmCut<Cut>> {
  Cut& original;
  explicit NonEmCut(Cut& c):original(c) {}
  template<class P,class T> LengthType getMaxStepLength(P const&,T const&)const {
    return std::numeric_limits<double>::infinity()*1_m;
  }
  template<class P> ProcessReturn doContinuous(Step<P>& s,bool limited) {
    if(is_em(s.getParticlePre().getPID()))throw std::logic_error("EGS4 EM reached scalar transport");
    return original.doContinuous(s,limited);
  }
  template<class P> bool arrival(P& p,egs4::HostRequest const&) {
    if(is_em(p.getPID()))return true;
    StraightTrajectory track(Line(p.getPosition(),p.getDirection()*constants::c),0_s);
    Step step(p,track);auto result=original.doContinuous(step,false);
    if(result!=ProcessReturn::Ok&&result!=ProcessReturn::ParticleAbsorbed)
      throw std::runtime_error("Unexpected non-EM arrival cut");
    return result==ProcessReturn::Ok;
  }
  template<class V> void doSecondaries(V& v) {
    for(auto p=v.begin();p!=v.end();++p)if(!arrival(p,{}))p.erase();
  }
};
class Report final:public BaseOutput {
  YAML::Node config_;mutable ShowerSummary summary_;
public:
  explicit Report(GpuCliOptions const& o) {
    config_["em_backend"]="egs4";config_["stepfc"]=o.egs4_stepfc;
    config_["table_sha256"]=egs4::embeddedTableSha256();
    config_["table_storage"]="embedded at build time";
    config_["non_em"]="shared C8 models; PROPOSAL muons";
    config_["native_em_production_profile_complete"]=false;
  }
  void startOfLibrary(boost::filesystem::path const& p)final{summary_.open(p);setInit(true);}
  void startOfShower(unsigned i)final{summary_.event(i)["status"]="in_progress";summary_.flush();}
  void endOfShower(unsigned)final{summary_.flush();}
  void endOfLibrary()final{summary_.flush();setInit(false);}
  YAML::Node getConfig()const final{return config_;}
  YAML::Node getSummary()const final{return summary_.snapshot();}
  void writeSummary(boost::filesystem::path const& p)const final{summary_.writeSummary(p);}
  void failed(unsigned i,std::string const& message){summary_.event(i)["status"]="incomplete";summary_.event(i)["reason"]=message;summary_.flush();}
  template<class Timing>void completed(unsigned i,egs4::RunStatistics const& s,Timing const& t,
      double discarded,bool prefix=false) {
    for(auto const& entry:t.scalar_steps_by_pdg)
      if(entry.first==11||entry.first==-11||entry.first==22)throw std::runtime_error("EM escaped EGS4 router");
    auto n=summary_.event(i);n["status"]="completed";n["prefix_capture"]=prefix;
    n["scalar_em_steps"]=0;n["native_steps"]=s.steps;n["execution_space"]=s.execution_space;
    n["radio_execution_space"]=s.radio_execution_space;n["radio_tracks"]=s.radio_tracks;
    n["radio_kernel_ms"]=s.radio_kernel_ms;n["radio_projection_batches"]=s.radio_projection_batches;
    n["radio_fixed_point_overflows"]=s.radio.fixed_point_overflows;
    n["profile_execution_space"]=s.profile_execution_space;
    n["profile_fixed_point_overflows"]=s.profile_fixed_point_overflows;
    n["profile_invalid_records"]=s.profile_invalid_records;
    n["discarded_available_energy_GeV"]=discarded;summary_.flush();
  }
};
}

// Uses the SAME application-owned environment, models, cuts, observers, writers
// and event settings as PROPOSAL. Only the EM router/termination is replaced.
// No Fe fixture, duplicate CLI, private atmosphere or private antenna defaults.
class Egs4RunSession {
  accelerator::em::detail::KokkosEmRunSession muons_;
  std::unique_ptr<egs4::Session> native_; // dies before muons_ / Kokkos
  egs4_detail::Report report_;
  int capacity_{};
  double previous_threshold_{-1.},previous_weight_{-1.};
  static accelerator::em::KokkosRuntimeConfig runtime(GpuCliOptions const& o) {
    accelerator::em::KokkosRuntimeConfig c;c.device=o.kokkos_device;
    c.threads=o.kokkos_num_threads;c.execution_backend=o.kokkos_execution;return c;
  }
public:
  explicit Egs4RunSession(GpuCliOptions const& o):muons_(runtime(o),o.gpu_aux_cache_dir),report_(o) {}
  template<class Output>void registerOutput(Output& o){o.add("native_egs4",report_);}
  template<class Event,class Context>void run(GpuCliOptions const& options,Event const& event,Context const& c) {
    auto const& g=c.geometry;auto const& o=c.outputs;auto const& p=c.processes;
    auto const& e=c.execution;auto const& d=c.diagnostics;
    using Stack=std::remove_reference_t<decltype(e.stack)>;
    using Env=std::remove_reference_t<decltype(g.environment)>;
    using Tracking=std::remove_reference_t<decltype(e.tracking)>;
    try {
      egs4_detail::NonEmCut<std::remove_reference_t<decltype(p.cut)>> cut(p.cut);
      auto sequence=make_sequence(p.stack_inspector,p.neutrino_primary,p.hadron_sequence,p.decay_sequence,
        p.em_cascade,p.production_process,p.em_continuous,p.coreas,p.zhs,p.longitudinal_process,
        o.observation,o.interactions,p.thinning,cut);
      if(!multigpu::activeFrontierOutput.empty()) {
        // Never advance prefix EM with PROPOSAL before handing it to EGS4.
        multigpu::CaptureRouter capture(multigpu::activeFrontierOutput,g.coordinates,
          std::numeric_limits<double>::max());
        HybridCascade<Tracking,decltype(sequence),std::remove_reference_t<decltype(e.output)>,Stack,decltype(capture)>
          cascade(g.environment,e.tracking,sequence,e.output,e.stack,capture);
        e.configure_primary(cascade);cascade.run();
        report_.completed(event.output_shower_id,{},cascade.timingStatistics(),0.,true);return;
      }
      auto environment=gpu::em::makeCorsika7AtmosphereSnapshot(event.atmosphere_id,
        {0.,0.,0.},3,g.observation_height_m,g.magnetic_field_T,event.maximum_magnetic_deflection_rad);
      environment.observation_plane_point_m[0]=g.core_x_m;
      environment.observation_plane_point_m[1]=g.core_y_m;
      double const threshold=event.em_thinning_fraction*event.primary_total_energy/1_GeV;
      gpu::em::GpuEmConfig m;m.min_batch_size=64;m.resident_batch_limit=4096;m.memory_fraction=.05;
      m.random_seed=event.seed;m.shower_id=event.output_shower_id;
      m.deterministic=options.gpu_deterministic;
      m.detailed_stage_timing=options.gpu_detailed_stage_timing;
      m.em_transport_cut_MeV=event.em_cut/1_MeV;m.muon_transport_cut_MeV=event.muon_cut/1_MeV;
      m.thinning={threshold,event.maximum_weight,threshold>0.,!event.keep_zero_weight_particles};
      m.resident_cross_species=false;
      auto muon=muons_.beginProposalNative(environment,m,
        {event.configured_primary_energy_max/1_MeV,event.em_cut/1_MeV,event.muon_cut/1_MeV,
         proposal::optimized_proposal_energy_cut(event.production_threshold)/1_MeV},
        p.em_cascade.nativeCalculatorViews(),p.em_continuous_proposal.nativeCalculatorViews());
      if(!muon.muon_transport_available)throw std::runtime_error("PROPOSAL muon backend unavailable");
      if(!native_)native_=egs4::makeEmbeddedSession();
      egs4::Configuration cfg;cfg.environment=environment;cfg.earth_radius_m=constants::EarthRadius::Mean/1_m;
      cfg.seed=event.seed;cfg.shower_id=event.output_shower_id;cfg.stepfc=options.egs4_stepfc;
      cfg.maximum_waves=std::numeric_limits<int>::max();cfg.reuse_queue_workspace=true;
      if(options.radio_backend=="kokkos") {
        cfg.radio=gpu::radio::makeGpuRadioConfig(g.environment,
          g.injection_position,g.surface,g.propagation_step,o.coreas_detector,o.zhs_detector);
        cfg.radio.fixed_point_field_limit_V_per_m=options.gpu_radio_field_limit;
        cfg.radio.deterministic=options.gpu_deterministic;
        cfg.radio.track_diagnostics=options.gpu_radio_track_diagnostics;
      }
      if(!capacity_) {
        auto execution=accelerator::em::resolveKokkosExecutionBackend(options.kokkos_execution);
        capacity_=execution=="cuda"?native_->gpuMemoryBudget(options.gpu_memory_fraction,cfg.radio).queue_capacity:1048576;
        if(options.gpu_resident_batch_limit) {
          if(options.gpu_resident_batch_limit>std::size_t(capacity_))
            throw std::length_error("Resident batch limit exceeds the EGS4 queue memory budget");
          capacity_=int(options.gpu_resident_batch_limit);
        }
      }
      cfg.queue_capacity=capacity_;
      cfg.thinning=m.thinning;
      auto mass=[](int id){return get_mass(convert_from_PDG(static_cast<PDGCode>(id)))/1_MeV;};
      cfg.electron_mass_MeV=mass(11);cfg.electron_total_cut_MeV=mass(11)+event.em_cut/1_MeV;
      cfg.photon_cut_MeV=event.em_cut/1_MeV;cfg.muon_mass_MeV=mass(13);
      cfg.charged_pion_mass_MeV=mass(211);cfg.neutral_pion_mass_MeV=mass(111);
      cfg.proton_mass_MeV=mass(2212);cfg.neutron_mass_MeV=mass(2112);
      cfg.phi_mass_MeV=mass(333);cfg.omega_mass_MeV=mass(223);cfg.rho_mass_MeV=mass(113);
      cfg.charged_kaon_mass_MeV=mass(321);cfg.long_kaon_mass_MeV=mass(130);cfg.short_kaon_mass_MeV=mass(310);cfg.eta_mass_MeV=mass(221);
      cfg.resolve_rare_vertices=cfg.native_photonuclear_vertices=cfg.native_prompt_rho_decay=cfg.native_prompt_resonance_decay=true;
      auto& projection=cfg.profile;projection.enabled=projection.accumulate_on_device=true;
      projection.crossing_mode=o.longitudinal.getCrossingMode();projection.output_bin_count=o.energy_loss.GetNBins();
      if(projection.output_bin_count!=o.longitudinal.getNBins())throw std::runtime_error("EGS4 output grid mismatch");
      projection.output_bin_width_g_per_cm2=g.depth_step/(1_g/square(1_cm));
      projection.fixed_point_weight_limit=2.*event.configured_primary_energy_max/event.em_cut;
      projection.fixed_point_energy_limit_GeV=std::max(2.*event.configured_primary_energy_max/1_GeV,1.);
      auto start=g.shower_axis.getStart().getCoordinates(g.coordinates);
      auto direction=g.shower_axis.getDirection().getComponents(g.coordinates);
      for(int j=0;j<3;++j){projection.axis_start_position_m[j]=start[j]/1_m;projection.axis_direction[j]=direction[j].magnitude();}
      projection.axis_step_length_m=g.shower_axis.getSteplength()/1_m;
      for(auto x:g.shower_axis.getGrammageSupport())projection.axis_grammage_g_per_cm2.push_back(x/(1_g/square(1_cm)));
      cfg.retain_across_showers=true;
      cfg.resume_retained_shower=native_->hasRetainedShower()&&threshold==previous_threshold_&&event.maximum_weight==previous_weight_;
      gpu::em::CorsikaOutputSink sink(g.coordinates,o.energy_loss,o.longitudinal,o.production,o.observation,o.interactions,p.coreas,p.zhs,true,cfg.radio.enabled);
      double discarded=0.;egs4::OutputCallbacks callbacks;
      callbacks.step=[&](auto const& r){sink.onStep(r);};callbacks.profile=[&](auto const& r){sink.onGpuProfile(r);};
      callbacks.radio=[&](auto const& r){sink.onRadioTrack(r);};
      callbacks.radio_waveforms=[&](auto const& r,std::uint64_t n){sink.onGpuRadioWaveforms(r,n);};
      callbacks.observation=[&](auto const& r){sink.onObservation(r);};
      callbacks.discarded=[&](double x,unsigned){discarded+=x;};callbacks.prompt_decay=[](auto const&){};
      auto secondaries=make_sequence(p.production_process,p.thinning,cut);
      auto arrival=[&](auto& particle,auto const& request){return cut.arrival(particle,request);};
      proposal::HadronicPhotonModel photons(d.photo_low,d.photo_high,event.high_energy_hadronic_threshold);
      egs4::HostHandler<Stack,Env,decltype(photons),decltype(secondaries),decltype(arrival)> host(g.environment,g.coordinates,photons,secondaries,arrival);
      using Fallback=gpu::em::ProposalCpuFallbackHandler<Stack,std::remove_reference_t<decltype(p.em_cascade)>,decltype(sequence),Env>;
      Fallback fallback(p.em_cascade,sequence,g.environment,g.coordinates,event.seed,event.output_shower_id);
      egs4::MuonOutputSink<decltype(sink)> muon_sink(sink);
      egs4::ProposalMuonBackend muon_adapter(*muon.backend);
      using MuonRouter=gpu::em::AcceleratedEmRouterImpl<Stack,Fallback,decltype(muon_sink),egs4::ProposalMuonBackend>;
      MuonRouter muon_router(muon_adapter,g.coordinates,environment,fallback,muon_sink);muon_router.setRetainRecords(false);
      auto region=[](auto const* node){if(!node||!node->hasModelProperties())throw std::runtime_error("EGS4 outside air");return 3;};
      using Router=egs4::Router<Stack,decltype(host),decltype(region),MuonRouter>;
      auto batch=std::min(options.gpu_min_batch,std::size_t(capacity_/2));
      multigpu::BufferedFrontierRouter<Router> router(multigpu::ForwardRouterArguments{},g.coordinates,batch,
        *native_,cfg,callbacks,g.coordinates,host,region,batch,
        accelerator::em::resolveKokkosExecutionBackend(options.kokkos_execution)=="openmp",
        &muon_router,&muon_adapter);
      HybridCascade<Tracking,decltype(sequence),std::remove_reference_t<decltype(e.output)>,Stack,decltype(router)>
        cascade(g.environment,e.tracking,sequence,e.output,e.stack,router);
      e.configure_primary(cascade);cascade.run();
      if(!e.stack.isEmpty()||router.pending())throw std::runtime_error("EGS4 queues did not drain");
      report_.completed(event.output_shower_id,router.statistics(),cascade.timingStatistics(),discarded);
      previous_threshold_=threshold;previous_weight_=event.maximum_weight;
    }catch(std::exception const& error){report_.failed(event.output_shower_id,error.what());throw;}
  }
};
}
