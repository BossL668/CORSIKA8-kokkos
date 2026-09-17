/* Full backend coordinator test; real PROPOSAL, no CUDA physics substitutes. */
#include <corsika/accelerator/em/KokkosEmBackend.hpp>
#include <corsika/accelerator/em/common/ConvexEnvironmentSnapshot.hpp>
#include "CooperativeLeptonFixture.hpp"
#include <iostream>
#include <set>
#include <cmath>
#ifdef C8_COOPERATIVE_CUPTI
#include "CooperativeActivityTrace.hpp"
#endif
using namespace corsika;
namespace em=gpu::em;
void require(bool c,char const* m){if(!c)throw std::runtime_error(m);}
int main(int argc,char** argv) {
  try {
    if(argc<2 || argc>6)throw std::invalid_argument("expected existing PROPOSAL cache [OpenMP threads] [subshowers] [device memory fraction] [adaptive|openmp-cuda]");
    bool independent=argc>=4 && std::string(argv[3])=="subshowers";
    if(argc>=4 && !independent)throw std::invalid_argument("unknown diagnostic mode");
    auto fixture=loadCooperativeLeptonFixture(argv[1]);
    accelerator::em::KokkosRuntimeConfig runtime;
    runtime.execution_backend="cuda-openmp";
    if(argc==6) {
      auto policy=std::string(argv[5]);
      require(policy=="adaptive" || policy=="openmp-cuda","unknown scheduling policy");
      if(policy=="openmp-cuda") {
        require(independent,"CPU-primary fixture must use independent subshowers");
        runtime.execution_backend=policy;
      } else runtime.cooperative_policy=policy;
    }
    runtime.threads=argc>=3?std::stoi(argv[2]):16;
    require(runtime.threads>0,"OpenMP threads must be positive");
#ifdef C8_COOPERATIVE_CUPTI
    std::uint64_t host_begin=0,host_end=0;
    bool trace_subshowers=false;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> host_windows;
    runtime.cooperative_slice_observer=[&](bool start){
      if(trace_subshowers) {
        if(start) {
          require(host_windows.size()<16384,"host trace capacity exceeded");
          host_windows.emplace_back(c8_overlap_trace::timestamp(),0);
        } else host_windows.back().second=c8_overlap_trace::timestamp();
        return;
      }
      if(start && !host_begin)host_begin=c8_overlap_trace::timestamp();
      else if(!start && host_begin && !host_end)host_end=c8_overlap_trace::timestamp();
    };
#endif
    accelerator::em::KokkosEmBackend backend(runtime);
    require(backend.batchIndependentSpecifiedFallbacks() ==
                (runtime.cooperative_policy == "adaptive" ||
                 runtime.execution_backend == "openmp-cuda"),
            "specified fallback batching leaked into the legacy policy");
    struct Box{
      std::vector<geometry_detail::ConvexPlane> planes;
      auto const& exportPlanes()const{return planes;}
    } box{{{1,0,0,1000},{-1,0,0,1000},{0,1,0,1000},{0,-1,0,1000},
           {0,0,1,1000},{0,0,-1,1000}}};
    auto environment=em::makeHomogeneousConvexSnapshot(.001,box);
    em::GpuEmConfig config;
    config.random_seed=26091022;config.shower_id=0;
    config.em_transport_cut_MeV=.4;config.muon_transport_cut_MeV=300.;
    config.min_batch_size=4096;config.resident_batch_limit=32768;
    // CPU-primary protects its entire resident arena before sharing surplus.
    // This bounded fixture therefore uses 4096+4096 original e+- inputs to
    // exercise both real backends without inflating particle count or memory.
    // Legacy/adaptive fixtures retain their original 32768-particle arena.
    if(runtime.execution_backend=="openmp-cuda")config.resident_batch_limit=4096;
    // Keep the original default; small-memory devices may explicitly grant
    // enough budget for this fixed-size diagnostic arena, never bypass it.
    config.memory_fraction=argc>=5?std::stod(argv[4]):.10;
    config.resident_cross_species=true;
    auto& p=config.profile_projection;
    p.enabled=true;p.accumulate_on_device=true;p.axis_direction[0]=1.;
    p.axis_step_length_m=1.;p.output_bin_count=1000;
    p.output_bin_width_g_per_cm2=.1;p.fixed_point_weight_limit=1.e8;
    p.fixed_point_energy_limit_GeV=1.e8;
    for(int i=0;i<=1000;++i)p.axis_grammage_g_per_cm2.push_back(i*.1);
    auto& radio=config.radio;
    radio.enabled=true;radio.deterministic=true;radio.track_diagnostics=true;
    radio.propagation.homogeneous_refractive_index=1.0003;
    for(int i=0;i<4;++i) {
      gpu::radio::RadioObserverSnapshot o;
      o.position_m[0]=20.+i*5.;o.position_m[1]=10.;
      o.start_time_s=0.;o.sample_rate_Hz=1.e9;o.number_of_bins=4096;
      o.duration_s=4095.e-9;
      radio.coreas_observers.push_back(o);radio.zhs_observers.push_back(o);
    }
    backend.initialize(environment,fixture.tables,fixture.auxiliary,config);
    std::size_t retained_bytes=0;
    for(unsigned event=0;event<2;++event) {
      if(event){config.shower_id=event;backend.beginShower(em::makeGpuEmShowerConfig(config));}
      std::vector<em::EmParticleState> leptons(8192),photons;
      for(std::size_t i=0;i<leptons.size();++i){
        auto& q=leptons[i];q.pid=i%2?11:-11;q.energy_GeV=.0015+.0001*(i%10);
        q.history_id=i+1;q.direction[0]=1.;q.weight=1.;q.generation=1;
      }
      // Closed accelerator-boundary audit: returned fallback energy is an
      // outgoing flux, not discarded energy. No thinning in this fixture.
      long double initial_energy=0.,returned_energy=0.;
      for(auto const& q:leptons)initial_energy+=
          static_cast<long double>(q.weight)*q.energy_GeV;
      std::set<std::uint64_t> terminal;
      std::uint64_t next=1000000;
      auto check=[&](auto const& v){for(auto const& a:v)
        require(terminal.insert(a.particle.history_id).second,"duplicate terminal history");};
      auto checkFallback=[&](auto const& v){check(v);for(auto const& a:v)
        returned_energy+=static_cast<long double>(a.particle.weight)*a.particle.energy_GeV;};
      std::size_t rounds=0;
      if(independent) {
#ifdef C8_COOPERATIVE_CUPTI
        if(event==1){trace_subshowers=true;c8_overlap_trace::start("ResidentLepton");}
#endif
        accelerator::em::detail::SubshowerCallbacks cb;
        cb.reserve_histories=[&](std::uint64_t n){auto first=next;next+=n;return first;};
        cb.photons=[&](em::ResidentPhotonCascadeResult&& r,double) {
          check(r.observations);checkFallback(r.fallback_events);
          require(r.cpu_spill_particles.empty(),"unexpected photon subshower spill");
          require(r.remaining_photons.empty() && r.electromagnetic_secondaries.empty(),
                  "subshower descendants escaped their owner");
        };
        cb.leptons=[&](em::ResidentLeptonCascadeResult&& r,double) {
          check(r.observations);checkFallback(r.fallback_events);
          require(r.cpu_spill_particles.empty() && r.decay_candidates.empty(),"unexpected lepton subshower spill/decay");
          require(r.remaining_leptons.empty() && r.generated_photons.empty(),
                  "subshower descendants escaped their owner");
        };
        do {
          require(++rounds<20000,"independent subshowers did not drain");
          auto accepted=backend.advanceIndependentSubshowers(leptons,cb);
          leptons.erase(leptons.begin(),leptons.begin()+accepted);
        } while(!leptons.empty() || backend.pendingPhotonCount() || backend.pendingLeptonCount());
#ifdef C8_COOPERATIVE_CUPTI
        if(event==1){c8_overlap_trace::finish(host_windows);trace_subshowers=false;}
#endif
        auto const& s=backend.statistics().cooperative;
        require(s.independent_subshowers && s.subshower_cuda_submissions==s.subshower_cuda_commits,
                "subshower submissions not committed exactly once");
        if(runtime.execution_backend=="openmp-cuda")
          require(s.cuda_completion_mailbox_capacity==4 && s.maximum_cuda_packet_calls<=4,
                  "CPU-primary real-EM packet exceeded its four-call result boundary");
        std::cout<<"subshower_epochs="<<s.subshower_openmp_epochs
                 <<" maximum_host_epochs_per_cuda_job="<<s.maximum_host_epochs_per_cuda_job
                 <<" cuda_commits="<<s.subshower_cuda_commits<<std::endl;
      } else
      while(!leptons.empty()||!photons.empty()||backend.pendingPhotonCount()||backend.pendingLeptonCount()){
        require(++rounds<3000,"cooperative queues did not drain");
        if(!photons.empty()||backend.pendingPhotonCount()){
          auto n=photons.size()+backend.pendingPhotonCount();
          auto result=backend.runPhotonWavefront(photons,next,4,4096);
          next+=2*n*4;
          check(result.observations);checkFallback(result.fallback_events);
          require(result.cpu_spill_particles.empty(),"unexpected bounded fixture spill");
          photons=std::move(result.remaining_photons);
          leptons.insert(leptons.end(),result.electromagnetic_secondaries.begin(),result.electromagnetic_secondaries.end());
        }
        if(!leptons.empty()||backend.pendingLeptonCount()){
          auto n=leptons.size()+backend.pendingLeptonCount();
          auto count=std::max<std::uint64_t>(3*n,1U<<20);
#ifdef C8_COOPERATIVE_CUPTI
          // Warm first event exercises allocation/JIT before overlap timing.
          bool trace=event==1 && rounds==1;
          if(trace){host_begin=host_end=0;c8_overlap_trace::start("ResidentLepton");}
#endif
          auto result=backend.runLeptonWavefront(leptons,next,8,next+count,4096);
#ifdef C8_COOPERATIVE_CUPTI
          if(trace)c8_overlap_trace::finish(host_begin,host_end);
#endif
          next+=count;
          check(result.observations);checkFallback(result.fallback_events);
          require(result.cpu_spill_particles.empty(),"unexpected bounded fixture spill");
          require(result.decay_candidates.empty(),"electron fixture produced muon decay");
          leptons=std::move(result.remaining_leptons);
          photons.insert(photons.end(),result.generated_photons.begin(),result.generated_photons.end());
        }
      }
      auto profile=backend.downloadProfile();
      auto wave=backend.downloadRadioWaveforms();
      auto const& stats=backend.statistics();
      require(stats.accelerator_backend == runtime.execution_backend,"wrong reported priority backend");
      if (runtime.cooperative_policy == "adaptive" && runtime.threads > 1) {
        require(stats.cooperative.host_profile_shards > 1 &&
                stats.cooperative.host_profile_shards <= 256 &&
                stats.cooperative.host_profile_shard_bytes > 0 &&
                stats.cooperative.host_profile_shard_bytes <= 16 * 1024 * 1024,
                "missing/unbounded adaptive host profile shards after event reset");
      } else require(stats.cooperative.host_profile_shards == 0 &&
                     stats.cooperative.host_profile_shard_bytes == 0,
                     "legacy/one-thread cooperative path unexpectedly enabled profile shards");
      require(stats.cooperative.cuda_input_particles>0&&stats.cooperative.openmp_input_particles>0,
              "one endpoint performed no transport");
      require(stats.cooperative.independent_drivers &&
              stats.cooperative.independent_joint_calls>0,"no independent joint execution");
      require(profile.steps>0&&profile.invalid_records==0&&profile.fixed_point_overflows==0,
              "invalid cooperative profile");
      auto const source=initial_energy+profile.weighted_medium_rest_mass_input_GeV+
          profile.weighted_mass_convention_correction_GeV;
      auto const sink=returned_energy+profile.weighted_deposited_energy_GeV+
          profile.weighted_cut_rest_mass_energy_GeV+
          profile.weighted_observed_total_energy_GeV+
          profile.weighted_escaped_total_energy_GeV+
          profile.weighted_unwritten_photoelectric_binding_energy_GeV-
          profile.weighted_observation_cut_overlap_energy_GeV;
      auto const relative=std::abs(source-sink)/source;
      std::cout<<"energy_source_GeV="<<static_cast<double>(source)
               <<" energy_sink_GeV="<<static_cast<double>(sink)
               <<" fallback_flux_GeV="<<static_cast<double>(returned_energy)
               <<" energy_relative_residual="<<static_cast<double>(relative)<<std::endl;
      require(relative<=1.e-4L,"independent accelerator energy flux does not close");
      require(stats.radio.coreas_contributions>0&&stats.radio.zhs_contributions>0,
              "no full-radio contributions");
      require(wave.coreas.size()==4&&wave.zhs.size()==4,"lost observer output");
      require(backend.downloadProfile().energy_loss_GeV==profile.energy_loss_GeV,
              "profile download committed twice");
      require(backend.downloadRadioWaveforms().zhs[0].x==wave.zhs[0].x,
              "radio download committed twice");
      auto storage=stats.physical_workspace_bytes+stats.cooperative.openmp_workspace_bytes;
      if(event)require(storage==retained_bytes,"workspace grew across identical showers");
      retained_bytes=storage;
      std::cout<<"event="<<event<<" rounds="<<rounds
               <<" cuda_inputs="<<stats.cooperative.cuda_input_particles
               <<" openmp_inputs="<<stats.cooperative.openmp_input_particles
               <<" joint_calls="<<stats.cooperative.independent_joint_calls
               <<" endpoint_window_overlap_ms="<<stats.cooperative.endpoint_window_overlap_ms
               <<" cuda_tail_wait_ms="<<stats.cooperative.cuda_finished_before_host_ms
               <<" coreas="<<stats.radio.coreas_contributions
               <<" zhs="<<stats.radio.zhs_contributions
               <<" profile_steps="<<profile.steps<<" workspace_bytes="<<storage<<std::endl;
    }
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
