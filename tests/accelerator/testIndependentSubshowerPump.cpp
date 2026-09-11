/* Scheduling/ownership test doubles, not an electromagnetic physics oracle. */
#include <corsika/accelerator/em/detail/IndependentSubshowerPump.hpp>
#include <iostream>
#include <set>
using namespace corsika::accelerator::em::detail;
namespace em=corsika::gpu::em;
void require(bool value,char const* message) {if(!value)throw std::runtime_error(message);}
struct ControlledEndpoint {
  std::array<std::vector<em::EmParticleState>,2> pending;
  bool gpu{},fail{};
  unsigned lifetime=8;
  std::shared_future<void> permit;
  std::atomic<bool> entered{false};
  std::atomic<bool> executing{false};
  std::thread::id owner{};
  std::size_t host_capacity{2048},pending_limit{std::numeric_limits<std::size_t>::max()};
  std::size_t last_input{},bounded_input_calls{};
  void setCooperativePendingInputLimit(std::size_t n) {
    access();pending_limit=n;
    if(n!=std::numeric_limits<std::size_t>::max())++bounded_input_calls;
  }
  void access() const {
    require(!executing.load() || owner==std::this_thread::get_id(),
            "coordinator accessed an in-flight endpoint");
  }
  std::size_t maximumResidentPhotonBatchSize() const {access();return gpu?8192:host_capacity;}
  std::size_t maximumResidentLeptonBatchSize() const {return maximumResidentPhotonBatchSize();}
  // Immutable configuration is allowed; queue state is not.
  std::size_t minimumBatchSize() const {return 4096;}
  std::size_t pendingPhotonCount() const {access();return pending[0].size();}
  std::size_t pendingLeptonCount() const {access();return pending[1].size();}
  std::vector<em::EmParticleState> takeCooperativePending(bool photons,std::size_t n) {
    access();
    auto& p=pending[photons?0:1];require(n<=p.size(),"steal beyond pending prefix");
    std::vector<em::EmParticleState> result(p.begin(),p.begin()+n);
    p.erase(p.begin(),p.begin()+n);return result;
  }
  template<class Result> Result run(std::vector<em::EmParticleState> input,unsigned k) {
    auto self=std::this_thread::get_id();
    if(owner==std::thread::id{})owner=self;
    require(owner==self,"endpoint moved between driver threads");
    executing=true;
    struct Release {std::atomic<bool>& value;~Release(){value=false;}} guard{executing};
    if(!entered.exchange(true) && permit.valid())
      require(permit.wait_for(std::chrono::seconds(3))==std::future_status::ready,
              "GPU gate timed out: host progress was blocked behind GPU");
    if(fail)throw std::runtime_error("controlled endpoint failure");
    auto n=std::min({maximumResidentPhotonBatchSize()-input.size(),pending[k].size(),pending_limit});
    auto resident=takeCooperativePending(k==0,n);
    input.insert(input.end(),resident.begin(),resident.end());
    last_input=input.size();
    Result r;r.completed=true;r.input_particles=input.size();r.wavefronts=1;
    r.transport_records=input.size();
    if(spill_once) {
      spill_once=false;r.cpu_spill_particles=std::move(input);
      if(refuse_allocation){r.wavefronts=0;r.workspace_limit_checkpoint=true;}
      // Deliberate oversized mock packet: test storage pressure, not physics.
      for(std::size_t i=0;i<spill_children;++i) {
        em::EmParticleState child;child.pid=11;child.history_id=100000+i;
        child.energy_GeV=1.;child.weight=1.;r.cpu_spill_particles.push_back(child);
      }
      return r;
    }
    for(auto p:input) {
      if(++p.step_id<lifetime) {p.pid=k==0?11:22;pending[1-k].push_back(p);}
      else {em::ObservationRecord o;o.particle=p;r.observations.push_back(o);}
    }
    return r;
  }
  em::ResidentPhotonCascadeResult runPhotonWavefront(
      std::vector<em::EmParticleState> const& p,std::uint64_t,std::size_t,std::size_t) {
    return run<em::ResidentPhotonCascadeResult>(p,0);
  }
  bool spill_once{};
  std::size_t minimum_seen{};
  bool refuse_allocation{};
  std::size_t spill_children{};
  em::ResidentLeptonCascadeResult runLeptonWavefront(
      std::vector<em::EmParticleState> const& p,std::uint64_t,std::size_t,std::uint64_t,std::size_t minimum) {
    minimum_seen=minimum;
    return run<em::ResidentLeptonCascadeResult>(p,1);
  }
};
std::vector<em::EmParticleState> roots(std::size_t n) {
  std::vector<em::EmParticleState> p(n);
  for(std::size_t i=0;i<n;++i){p[i].pid=11;p[i].history_id=i+1;p[i].energy_GeV=1.;p[i].weight=1.;}
  return p;
}
int main() {
  try {
    require(CooperativeHostPolicy::capacity(20)==2048,"laptop arena changed");
    require(CooperativeHostPolicy::capacity(130)==16384,"130-thread arena did not scale");
    require(CooperativeHostPolicy::capacity(100000)==16384,"arena lacks upper bound");
    require(CooperativeHostPolicy::initialShare(130)>.4,"server starts with laptop share");
    auto main=std::this_thread::get_id();
    ControlledEndpoint gpu,cpu;gpu.gpu=true;
    std::promise<void> release;gpu.permit=release.get_future().share();
    em::CooperativeEmStatistics stats;
    IndependentSubshowerPump<ControlledEndpoint> pump(gpu,cpu,stats,[]{});
    std::set<std::uint64_t> completed;
    std::uint64_t next=1000000;
    SubshowerCallbacks cb;
    cb.reserve_histories=[&](std::uint64_t n){require(std::this_thread::get_id()==main,"off-thread lease");auto first=next;next+=n;return first;};
    auto consume=[&](auto&& r,double) {
      require(std::this_thread::get_id()==main,"off-thread output");
      require(r.cpu_spill_particles.empty(),"unexpected fixture spill");
      for(auto const& o:r.observations)
        require(completed.insert(o.particle.history_id).second,"duplicate terminal identity");
    };
    cb.photons=consume;cb.leptons=consume;
    auto input=roots(8192);
    require(pump.advance(input,cb)==input.size(),"initial input not accepted");
    for(int i=0;i<4;++i) pump.advance({},cb);
    require(stats.subshower_openmp_epochs>=4 && pump.inFlight(),"CPU did not advance independently");
    require(stats.subshower_cuda_commits==0,"blocked GPU result was committed");
    release.set_value();
    for(int round=0;!pump.idle();++round){require(round<3000,"queues did not drain");pump.advance({},cb);}
    pump.requireIdle();
    require(completed.size()==input.size(),"lost terminal particles");
    require(stats.maximum_host_epochs_per_cuda_job>=4,"missing multi-epoch progress evidence");
    require(stats.subshower_cuda_commits==stats.subshower_cuda_submissions,"uncommitted CUDA result");
    require(gpu.owner!=main && cpu.owner==main,"endpoint ownership violated");

    // GPU deliberately blocked, but its coordinator-owned unsent tail can
    // feed an empty CPU repeatedly. No pending GPU state may be inspected.
    ControlledEndpoint waitinggpu,waitingcpu;waitinggpu.gpu=true;
    waitinggpu.lifetime=waitingcpu.lifetime=1;
    std::promise<void> waitingrelease;waitinggpu.permit=waitingrelease.get_future().share();
    em::CooperativeEmStatistics waitingstats;
    IndependentSubshowerPump<ControlledEndpoint> waiting(waitinggpu,waitingcpu,waitingstats,[]{});
    completed.clear();auto large=roots(16384);
    require(waiting.advance(large,cb)==large.size(),"queued-tail input rejected");
    require(waiting.ready(),"safe waiting work not advertised while GPU is blocked");
    for(int i=0;i<5;++i)waiting.advance({},cb);
    require(waitingstats.subshower_cuda_commits==0 && waiting.inFlight(),"GPU gate bypassed");
    require(waitingstats.subshower_inflight_waiting_particles>0 && completed.size()>2048,
            "CPU starved despite safe unsubmitted GPU tail");
    waitingrelease.set_value();
    for(int i=0;!waiting.idle();++i){require(i<3000,"waiting-tail queues did not drain");waiting.advance({},cb);}
    require(completed.size()==large.size(),"in-flight waiting transfer lost/duplicated identities");

    // New arrivals replenish before both CPU species queues become empty.
    ControlledEndpoint feedgpu,feedcpu;feedgpu.gpu=true;
    em::CooperativeEmStatistics feedstats;
    IndependentSubshowerPump<ControlledEndpoint> feed(feedgpu,feedcpu,feedstats,[]{});
    completed.clear();feed.advance(input,cb);
    auto arrivals=roots(1024);for(auto& p:arrivals)p.history_id+=input.size();
    require(feed.advance(arrivals,cb)==arrivals.size(),"refill input rejected");
    require(feedstats.subshower_proactive_refills>0,"CPU only refilled after empty");
    for(int i=0;!feed.idle();++i){require(i<3000,"refill queues did not drain");feed.advance({},cb);}
    require(completed.size()==input.size()+arrivals.size(),"refill lost/duplicated identities");

    // Only scheduling epochs shorten: a deliberately slow CUDA job with no
    // safe waiting work is allowed to finish, then its next quantum adapts.
    ControlledEndpoint slowgpu,slowcpu;slowgpu.gpu=true;
    slowgpu.lifetime=slowcpu.lifetime=1;
    std::promise<void> slowrelease;slowgpu.permit=slowrelease.get_future().share();
    em::CooperativeEmStatistics slowstats;
    IndependentSubshowerPump<ControlledEndpoint> slow(slowgpu,slowcpu,slowstats,[]{});
    completed.clear();slow.advance(input,cb);
    auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(140);
    while(std::chrono::steady_clock::now()<until)slow.advance({},cb);
    require(slowstats.subshower_cuda_commits==0,"slow kernel interrupted");
    slowrelease.set_value();
    for(int i=0;!slow.idle();++i){require(i<3000,"slow queues did not drain");slow.advance({},cb);}
    require(slowstats.subshower_cuda_epoch_reductions>0 &&
            slowstats.subshower_gpu_lepton_wave_limit<1024,"starved-host quantum did not adapt");
    require(completed.size()==input.size(),"quantum adaptation lost/duplicated identity");

    // Both directions use unchanged particle identity/state at safe boundaries.
    ControlledEndpoint tailgpu,tailcpu;tailgpu.gpu=true;tailgpu.lifetime=1;
    tailcpu.pending[1]=roots(2048);
    em::CooperativeEmStatistics tailstats;
    IndependentSubshowerPump<ControlledEndpoint> tail(tailgpu,tailcpu,tailstats,[]{});
    completed.clear();
    for(int i=0;!tail.idle();++i){require(i<3000,"tail did not drain");tail.advance({},cb);}
    require(completed.size()==2048 && tailstats.subshower_tail_migrations>0,"tail migration lost particles");

    // Both endpoint arenas deliberately overflow once. The ownership moves
    // back to bounded waiting queues, not to scalar physics, without changing
    // particle identity or producing a second terminal observation.
    ControlledEndpoint spillgpu,spillcpu;spillgpu.gpu=true;
    spillgpu.spill_once=spillcpu.spill_once=true;
    em::CooperativeEmStatistics spillstats;
    IndependentSubshowerPump<ControlledEndpoint> spilling(spillgpu,spillcpu,spillstats,[]{});
    completed.clear();
    require(spilling.advance(input,cb)==input.size(),"spill fixture input rejected");
    for(int i=0;!spilling.idle();++i){require(i<3000,"spill queues did not drain");spilling.advance({},cb);}
    require(completed.size()==input.size(),"requeued spill lost/duplicated a particle");
    require(spillstats.subshower_requeued_spill_particles==input.size(),"spill requeue count incorrect");

    ControlledEndpoint fullgpu,fullcpu;fullgpu.gpu=true;
    fullcpu.spill_once=true;fullcpu.spill_children=50000;
    em::CooperativeEmStatistics fullstats;
    IndependentSubshowerPump<ControlledEndpoint> full(fullgpu,fullcpu,fullstats,[]{});
    auto fullcb=cb;
    std::size_t scalar_spills=0;
    auto consume_full=[&](auto&& result,double) {
      for(auto const& p:result.cpu_spill_particles) {
        require(completed.insert(p.history_id).second,"duplicate overflow spill");
        ++scalar_spills;
      }
      for(auto const& o:result.observations)
        require(completed.insert(o.particle.history_id).second,"duplicate overflow terminal");
    };
    fullcb.photons=consume_full;fullcb.leptons=consume_full;completed.clear();
    require(full.advance(input,fullcb)==input.size(),"full-queue input rejected");
    for(int i=0;!full.idle();++i){require(i<3000,"full queues did not drain");full.advance({},fullcb);}
    require(completed.size()==input.size()+50000,"overflow lost particles");
    require(fullstats.subshower_cross_endpoint_spill_particles>0 && scalar_spills>0,
            "fixture did not exercise cross-end spill and both-full scalar handoff");
    require(fullstats.subshower_host_queue_peak_bytes<=
                8*2*(8192+2048)*sizeof(em::EmParticleState),"unbounded queue allocation");

    ControlledEndpoint nogpu,nocpu;nogpu.gpu=true;
    nogpu.spill_once=nocpu.spill_once=nogpu.refuse_allocation=nocpu.refuse_allocation=true;
    em::CooperativeEmStatistics nostats;
    IndependentSubshowerPump<ControlledEndpoint> refused(nogpu,nocpu,nostats,[]{});
    completed.clear();scalar_spills=0;
    refused.advance(input,fullcb);
    for(int i=0;!refused.idle();++i){require(i<3000,"zero-progress retry loop");refused.advance({},fullcb);}
    require(completed.size()==input.size() && scalar_spills==input.size() &&
            nostats.subshower_requeued_spill_particles==0,"allocation failure did not escape");

    // Adaptive discovery accepts a bounded prefix, including when the first
    // wave contains both kinds. A deliberately delayed GPU must not stop the
    // CPU or expose endpoint-owned queues to the coordinator.
    for(bool block_gpu:{false,true}) {
      ControlledEndpoint agpu,acpu;agpu.gpu=true;acpu.lifetime=agpu.lifetime=12;
      std::promise<void> gate;
      if(block_gpu)agpu.permit=gate.get_future().share();
      em::CooperativeEmStatistics astats;
      IndependentSubshowerPump<ControlledEndpoint> adaptive(agpu,acpu,astats,[]{},{},130,true);
      auto mixed=roots(19000);
      for(std::size_t i=0;i<mixed.size();i+=2)mixed[i].pid=22;
      auto total=mixed.size();
      completed.clear();
      for(int round=0;!mixed.empty() || !adaptive.idle();++round) {
        require(round<5000,"adaptive queues did not drain");
        auto n=adaptive.advance(mixed,cb);
        require(n<=mixed.size(),"accepted beyond bounded prefix");
        mixed.erase(mixed.begin(),mixed.begin()+n);
        if(block_gpu && round==4) {
          require(astats.subshower_openmp_epochs>=4,"adaptive CPU blocked on GPU");
          require(astats.subshower_cuda_commits==0,"adaptive committed blocked GPU");
          gate.set_value();
        }
      }
      adaptive.requireIdle();
      require(agpu.minimum_seen==1 && acpu.minimum_seen==1,
              "adaptive split tail inherits original staging threshold");
      require(completed.size()==total,"adaptive lost or duplicated terminal particles");
      require(astats.adaptive_policy && astats.subshower_cuda_commits==
              astats.subshower_cuda_submissions,"adaptive unfinished result");
      require(astats.adaptive_observations[0][0]+astats.adaptive_observations[0][1]>0 &&
              astats.adaptive_observations[1][0]+astats.adaptive_observations[1][1]>0,
              "adaptive did not calibrate both endpoints");
    }

    // A large host arena must not auto-refill a deliberately short CPU job.
    // Only the selected prefix is executed; the rest stays on its endpoint.
    ControlledEndpoint prefixgpu,prefixcpu;prefixgpu.gpu=true;
    prefixcpu.host_capacity=65536;prefixcpu.lifetime=prefixgpu.lifetime=1;
    prefixcpu.pending[1]=roots(12000);
    em::CooperativeEmStatistics prefixstats;
    IndependentSubshowerPump<ControlledEndpoint> prefix(prefixgpu,prefixcpu,prefixstats,[]{},{},20,true);
    completed.clear();prefix.advance({},cb);
    require(prefixcpu.last_input<=2048 && prefixcpu.bounded_input_calls==1,
            "resident auto-fill defeated adaptive input quantum");
    require(prefixcpu.pending_limit==std::numeric_limits<std::size_t>::max(),
            "CPU pending-input bound leaked after job");
    for(int i=0;!prefix.idle();++i){require(i<5000,"bounded-prefix queue did not drain");prefix.advance({},cb);}
    require(completed.size()==12000,"bounded input lost pending particles");
    require(cpu.bounded_input_calls==0,"legacy independent CPU input behavior changed");

    // Restore the per-call input limit even when the OpenMP endpoint fails.
    ControlledEndpoint failgpu,failcpu;failgpu.gpu=true;failcpu.fail=true;
    failcpu.pending[1]=roots(12000);failcpu.host_capacity=65536;
    em::CooperativeEmStatistics failstats;
    IndependentSubshowerPump<ControlledEndpoint> failpump(failgpu,failcpu,failstats,[]{},{},20,true);
    bool failed_prefix=false;
    try{failpump.advance({},cb);}catch(std::runtime_error const&){failed_prefix=true;}
    require(failed_prefix && failcpu.pending_limit==std::numeric_limits<std::size_t>::max(),
            "exception leaked bounded-input state");

    // Fail closed on a reused/overlapping global lease, while safely draining
    // the other endpoint rather than leaving a worker holding destroyed data.
    ControlledEndpoint badgpu,badcpu;badgpu.gpu=true;
    em::CooperativeEmStatistics badstats;
    IndependentSubshowerPump<ControlledEndpoint> bad(badgpu,badcpu,badstats,[]{});
    cb.reserve_histories=[](std::uint64_t){return std::uint64_t{1000000};};
    bool rejected=false;
    try {bad.advance(input,cb);}catch(std::logic_error const&){rejected=true;}
    require(rejected,"overlapping lease accepted");
    rejected=false;
    try {bad.requireIdle();}catch(std::logic_error const&){rejected=true;}
    require(rejected,"failed pump allowed finalization");
    // A device exception and an output-consumer exception both close the
    // event. Destruction must drain the worker without touching freed input.
    for(bool consumer_failure:{false,true}) {
      ControlledEndpoint errorgpu,errorcpu;errorgpu.gpu=true;
      errorgpu.fail=!consumer_failure;
      em::CooperativeEmStatistics errorstats;
      IndependentSubshowerPump<ControlledEndpoint> error(errorgpu,errorcpu,errorstats,[]{});
      SubshowerCallbacks errorcb;
      std::uint64_t lease=1000000;
      errorcb.reserve_histories=[&](std::uint64_t n){auto first=lease;lease+=n;return first;};
      auto result=[&](auto&&,double){if(consumer_failure)throw std::runtime_error("controlled output failure");};
      errorcb.photons=result;errorcb.leptons=result;
      rejected=false;
      try {
        error.advance(input,errorcb);
        for(int i=0;i<1000 && !error.idle();++i)error.advance({},errorcb);
      }catch(std::runtime_error const&){rejected=true;}
      require(rejected,"endpoint/output exception was lost");
      rejected=false;
      try{error.requireIdle();}catch(std::logic_error const&){rejected=true;}
      require(rejected,"failed endpoint/output allowed finalization");
    }
    std::cout<<"PASS: independent cross-species progress, 4+ CPU epochs during blocked GPU, "
                "exact-once output, coordinator ownership, bounded tail migration, lease failure cleanup\n";
  } catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
}
