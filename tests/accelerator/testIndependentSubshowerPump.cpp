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
  bool retain_species{};
  std::shared_future<void> permit;
  std::atomic<bool> entered{false};
  std::atomic<bool> executing{false};
  std::thread::id owner{};
  std::size_t host_capacity{2048},gpu_capacity{8192};
  std::size_t pending_limit{std::numeric_limits<std::size_t>::max()};
  std::size_t last_input{},bounded_input_calls{};
  std::function<void()> on_call;
  std::size_t diagnostic_padding{};
  std::vector<std::uint64_t> received_histories;
  std::vector<unsigned> received_kinds;
  std::array<std::vector<std::size_t>,2> received_input_sizes;
  std::array<std::vector<std::pair<std::size_t,std::size_t>>,2> received_wave_requests;
  std::vector<std::size_t> received_lepton_wave_limits;
  void setCooperativePendingInputLimit(std::size_t n) {
    access();pending_limit=n;
    if(n!=std::numeric_limits<std::size_t>::max())++bounded_input_calls;
  }
  void access() const {
    require(!executing.load() || owner==std::this_thread::get_id(),
            "coordinator accessed an in-flight endpoint");
  }
  std::size_t maximumResidentPhotonBatchSize() const {access();return gpu?gpu_capacity:host_capacity;}
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
    if(on_call)on_call();
    if(!entered.exchange(true) && permit.valid())
      require(permit.wait_for(std::chrono::seconds(3))==std::future_status::ready,
              "GPU gate timed out: host progress was blocked behind GPU");
    if(fail)throw std::runtime_error("controlled endpoint failure");
    auto n=std::min({maximumResidentPhotonBatchSize()-input.size(),pending[k].size(),pending_limit});
    auto resident=takeCooperativePending(k==0,n);
    input.insert(input.end(),resident.begin(),resident.end());
    last_input=input.size();
    received_kinds.push_back(k);
    received_input_sizes[k].push_back(input.size());
    Result r;r.completed=true;r.input_particles=input.size();r.wavefronts=1;
    r.interaction_records.resize(diagnostic_padding);
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
      if(++p.step_id<lifetime) {
        auto destination=retain_species?k:1-k;
        p.pid=destination==0?22:11;pending[destination].push_back(p);
      }
      else {em::ObservationRecord o;o.particle=p;r.observations.push_back(o);}
    }
    return r;
  }
  em::ResidentPhotonCascadeResult runPhotonWavefront(
      std::vector<em::EmParticleState> const& p,std::uint64_t first,std::size_t waves,std::size_t minimum) {
    received_histories.push_back(first);
    received_wave_requests[0].emplace_back(waves,minimum);
    return run<em::ResidentPhotonCascadeResult>(p,0);
  }
  bool spill_once{};
  std::size_t minimum_seen{};
  std::size_t lepton_input_seen{};
  bool refuse_allocation{};
  std::size_t spill_children{};
  em::ResidentLeptonCascadeResult runLeptonWavefront(
      std::vector<em::EmParticleState> const& p,std::uint64_t first,std::size_t waves,std::uint64_t,std::size_t minimum) {
    received_histories.push_back(first);
    received_wave_requests[1].emplace_back(waves,minimum);
    received_lepton_wave_limits.push_back(waves);
    minimum_seen=minimum;
    auto result=run<em::ResidentLeptonCascadeResult>(p,1);
    lepton_input_seen=result.input_particles;
    return result;
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
    require(stats.subshower_cuda_autonomous_continuations==0 &&
                stats.maximum_cuda_packet_calls<=1 &&
                gpu.received_histories.size()==stats.subshower_cuda_commits,
            "original GPU-primary path unexpectedly gained autonomous packets");

    // CPU-primary v3: preserve a complete primary arena and grant the helper
    // only a useful batch. Neither backend gets a time-derived input prefix.
    require(CpuPrimarySubshowerPolicy::gpuShare(0,0)==1./16.,"cold helper share changed");
    require(CpuPrimarySubshowerPolicy::gpuShare(1000,10)<.01,"weak helper oversubscribed");
    require(CpuPrimarySubshowerPolicy::gpuShare(10,1000)>.99,"fast helper share cannot grow");
    require(CpuPrimarySubshowerPolicy::hostReserve(65536,4096)==65536,
            "primary reserve does not protect a complete standalone arena");
    ControlledEndpoint primarygpu,primarycpu;primarygpu.gpu=true;
    primarycpu.host_capacity=8192;
    std::promise<void> primaryrelease;primarygpu.permit=primaryrelease.get_future().share();
    em::CooperativeEmStatistics primarystats;
    IndependentSubshowerPump<ControlledEndpoint> primary(primarygpu,primarycpu,primarystats,
        []{}, {},130,false,64U<<20,true);
    completed.clear();auto primaryinput=roots(24576);
    for(std::size_t i=0;i<primaryinput.size();i+=2)primaryinput[i].pid=22;
    require(primary.advance(primaryinput,cb)==primaryinput.size(),"CPU-primary input rejected");
    require(primarystats.cuda_input_particles==4096,
            "primary refill did not grant one useful auxiliary batch");
    require(primarycpu.last_input==primarycpu.host_capacity,
            "auxiliary grant fragmented the first primary arena");
    for(int i=0;i<4;++i)primary.advance({},cb);
    require(primarystats.subshower_openmp_epochs>=4 && primary.inFlight(),
            "CPU-primary progress waits for auxiliary GPU");
    require(primarystats.subshower_cuda_commits==0,"CPU-primary GPU gate bypassed");
    require(primarycpu.bounded_input_calls==0,"CPU-primary inherited the small adaptive prefix");
    primaryrelease.set_value();
    for(int i=0;!primary.idle();++i){require(i<3000,"CPU-primary queues did not drain");primary.advance({},cb);}
    primary.requireIdle();
    require(completed.size()==primaryinput.size(),"CPU-primary identities lost or duplicated");
    require(primarystats.subshower_cuda_commits==primarystats.subshower_cuda_submissions,
            "CPU-primary has uncommitted results");
    require(primarygpu.bounded_input_calls==0,"simple helper inherited adaptive input slicing");
    for(auto n:primarygpu.received_lepton_wave_limits)
      require(n==CpuPrimarySubshowerPolicy::auxiliary_lepton_waves,"auxiliary wave quantum changed");
    for(auto n:primarycpu.received_lepton_wave_limits)
      require(n==CpuPrimarySubshowerPolicy::host_lepton_waves,"primary host wave quantum changed");
    std::uint64_t primarywork=0;
    for(auto const& e:primarystats.adaptive_transport_records)for(auto n:e)primarywork+=n;
    require(primarywork==primaryinput.size()*primarycpu.lifetime,"CPU-primary step accounting");
    for(unsigned kind=0;kind<2;++kind)
      for(std::size_t i=0;i<primarycpu.received_input_sizes[kind].size();++i)
        require(primarycpu.received_wave_requests[kind][i].second==
            CpuPrimarySubshowerPolicy::hostCheckpointMinimum(primarycpu.received_input_sizes[kind][i],primarycpu.minimumBatchSize()),
            "CPU-primary checkpoint minimum does not match owned initial front");
    for(auto count:{std::size_t{1},std::size_t{3},std::size_t{4},std::size_t{256},
                    std::size_t{4095},std::size_t{4096},std::size_t{16383},
                    std::size_t{16384},std::size_t{65536}}) {
      auto minimum=CpuPrimarySubshowerPolicy::hostCheckpointMinimum(count,4096);
      require(minimum>=1 && minimum<=count && minimum<=4096,"invalid CPU tail threshold");
      if(count>=16384)require(minimum==4096,"large CPU front threshold changed");
      if(count>=4 && count<16384)require(minimum==count/4,"small CPU front still uses oversized threshold");
    }
    std::uint64_t histogram_calls=0;
    for(auto count:primarystats.adaptive_job_input_histogram[1])histogram_calls+=count;
    require(histogram_calls==primarystats.subshower_openmp_epochs,"CPU input histogram lost/duplicated calls");
    std::cout<<"PASS: CPU-primary full-arena mixed-PID ownership and independent progress during blocked GPU\n";

    // CPU-primary v6: bounded deferral, no cuts or changed history identity.
    // Test the pure chooser symmetrically, including nonstandard capacities.
    for(unsigned tiny:{0U,1U}) {
      std::array<std::size_t,2> counts{8192,8192},capacity{8192,8192};
      std::array<unsigned,2> deferred{};counts[tiny]=1;
      for(unsigned i=0;i<8;++i)
        require(CpuPrimarySubshowerPolicy::selectHostSpecies(tiny,counts,capacity,4096,deferred)==1-tiny,
                "tiny CPU species not deferred to useful other front");
      require(CpuPrimarySubshowerPolicy::selectHostSpecies(tiny,counts,capacity,4096,deferred)==tiny,
              "tiny CPU species starved beyond bounded deferrals");
      require(deferred[tiny]==0,"served species retained stale deferral debt");
      counts[1-tiny]=0;
      require(CpuPrimarySubshowerPolicy::selectHostSpecies(tiny,counts,capacity,4096,deferred)==tiny,
              "lone CPU tail waited for nonexistent future input");
      counts={4095,4095};
      require(CpuPrimarySubshowerPolicy::selectHostSpecies(tiny,counts,capacity,4096,deferred)==tiny,
              "two small species unnecessarily changed order");
      counts={4096,4096};
      require(CpuPrimarySubshowerPolicy::selectHostSpecies(tiny,counts,capacity,4096,deferred)==tiny,
              "two useful fronts lost alternating service");
      capacity={2048,2048};counts={2048,2048};
      require(CpuPrimarySubshowerPolicy::selectHostSpecies(tiny,counts,capacity,4096,deferred)==tiny,
              "full small-capacity arena treated as tiny");
      counts[tiny]=0;deferred[tiny]=8;
      require(CpuPrimarySubshowerPolicy::selectHostSpecies(tiny,counts,capacity,4096,deferred)==1-tiny && deferred[tiny]==0,
              "empty species not drained/reset correctly");
    }
    for(bool primary_mode:{false,true}) {
      ControlledEndpoint coalgpu,coalcpu;coalgpu.gpu=true;
      coalcpu.host_capacity=8192;coalcpu.lifetime=12;coalcpu.retain_species=true;
      coalcpu.pending[0]=roots(1);coalcpu.pending[0][0].pid=22;
      coalcpu.pending[1]=roots(8192);
      for(auto& p:coalcpu.pending[1])p.history_id+=10;
      em::CooperativeEmStatistics coalstats;completed.clear();
      IndependentSubshowerPump<ControlledEndpoint> coal(coalgpu,coalcpu,coalstats,
          []{},{},130,false,64U<<20,primary_mode);
      for(int i=0;!coal.idle();++i) {
        require(i<500,"bounded CPU coalescing did not drain");coal.advance({},cb);
      }
      coal.requireIdle();
      require(completed.size()==8193,"CPU coalescing lost/duplicated identities");
      if(primary_mode) {
        require(coalcpu.received_kinds.size()>=9,"starvation fixture too short");
        for(unsigned i=0;i<8;++i) require(coalcpu.received_kinds[i]==1,"large front interrupted too early");
        require(coalcpu.received_kinds[8]==0,"tiny species not served by ninth choice");
        require(coalstats.adaptive_species_coalesces[1]>0 && coalstats.subshower_cuda_submissions==0,
                "coalescing altered ownership or was not measured");
        require(coalstats.openmp_input_particles==8193*12,"CPU coalescing duplicated/lost steps");
      } else {
        require(coalcpu.received_kinds.front()==0 && coalstats.adaptive_species_coalesces[1]==0,
                "old GPU-primary path changed species selection");
      }
    }
    std::cout<<"PASS: CPU-primary bounded species coalescing, tail drain and unchanged GPU-primary selection\n";

    // Exercise both species at the reserve boundary. Terminal one-wave
    // fixtures deliberately generate no descendants: any helper input here
    // must have been carved out of these original primary staging batches.
    for(auto helper_capacity:{std::size_t{2048},std::size_t{8192}}) {
      auto helper_minimum=std::min(helper_capacity,std::size_t{4096});
      auto share_boundary=std::size_t{8192}+helper_minimum;
      for(auto per_species:{std::size_t{100},std::size_t{8192},
                            share_boundary-1,share_boundary}) {
        ControlledEndpoint reservegpu,reservecpu;reservegpu.gpu=true;
        reservegpu.gpu_capacity=helper_capacity;
        reservecpu.host_capacity=8192;
        reservegpu.lifetime=reservecpu.lifetime=1;
        em::CooperativeEmStatistics reservestats;
        IndependentSubshowerPump<ControlledEndpoint> reserve(reservegpu,reservecpu,
            reservestats,[]{},{},130,false,64U<<20,true);
        auto reserveinput=roots(2*per_species);completed.clear();
        for(std::size_t i=0;i<reserveinput.size();i+=2)reserveinput[i].pid=22;
        require(reserve.advance(reserveinput,cb)==reserveinput.size(),
                "bounded primary reserve input rejected");
        require(reservecpu.last_input==std::min(per_species,reservecpu.host_capacity),
                "GPU assistance reduced the first primary batch");
        for(int i=0;!reserve.idle();++i) {
          require(i<100,"primary reserve queues did not drain");reserve.advance({},cb);
        }
        reserve.requireIdle();
        for(auto const& calls:reservecpu.received_input_sizes)
          require(!calls.empty() && calls.front()==std::min(per_species,reservecpu.host_capacity),
                  "helper fragmented a species-specific primary staging batch");
        if(per_species<share_boundary) {
          require(reservestats.subshower_cuda_submissions==0,
                  "sub-minimum primary surplus was assigned to the helper");
          require(reservestats.openmp_input_particles==reserveinput.size(),
                  "small original front did not remain entirely on primary CPU");
        } else {
          require(reservestats.cuda_input_particles>=helper_minimum,
                  "a full primary arena plus useful helper surplus left GPU idle");
          for(auto const& calls:reservegpu.received_input_sizes)
            for(auto count:calls)require(count>=helper_minimum,"helper launched a tiny original input grant");
        }
        require(completed.size()==reserveinput.size(),"reserve boundary lost/duplicated identities");
        require(reservestats.openmp_input_particles+reservestats.cuda_input_particles==
                    reserveinput.size(),"reserve boundary duplicated transport work");
      }
    }

    // Large incoming lists must retain their unaccepted suffix at the caller.
    // Re-presenting precisely that suffix tests bounded backpressure while a
    // blocked auxiliary job cannot impede independent primary progress.
    ControlledEndpoint pressuregpu,pressurecpu;pressuregpu.gpu=true;
    pressurecpu.host_capacity=8192;pressuregpu.lifetime=pressurecpu.lifetime=1;
    std::promise<void> pressurerelease;pressuregpu.permit=pressurerelease.get_future().share();
    em::CooperativeEmStatistics pressurestats;
    IndependentSubshowerPump<ControlledEndpoint> pressure(pressuregpu,pressurecpu,
        pressurestats,[]{},{},130,false,64U<<20,true);
    auto pressureinput=roots(100000);completed.clear();
    for(std::size_t i=0;i<pressureinput.size();i+=2)pressureinput[i].pid=22;
    auto accepted=pressure.advance(pressureinput,cb);
    require(accepted>0 && accepted<pressureinput.size(),"CPU-primary input lacks bounded backpressure");
    pressureinput.erase(pressureinput.begin(),pressureinput.begin()+accepted);
    auto const before=completed.size();
    pressure.advance({},cb);
    require(completed.size()>before && pressure.inFlight() &&
                pressurestats.subshower_cuda_commits==0,
            "primary backpressure progress waited for blocked auxiliary GPU");
    pressurerelease.set_value();
    for(int i=0;!pressureinput.empty() || !pressure.idle();++i) {
      require(i<3000,"CPU-primary backpressure did not drain");
      auto n=pressure.advance(pressureinput,cb);
      require(n<=pressureinput.size(),"accepted beyond caller-owned suffix");
      pressureinput.erase(pressureinput.begin(),pressureinput.begin()+n);
    }
    pressure.requireIdle();
    require(completed.size()==100000,"CPU-primary backpressure lost/duplicated identities");
    require(pressurestats.openmp_input_particles+pressurestats.cuda_input_particles==100000,
            "CPU-primary backpressure duplicated input transport");
    require(pressurestats.subshower_host_queue_peak_bytes<=
                8*2*(8192+8192)*sizeof(em::EmParticleState),
            "CPU-primary backpressure allocated unbounded coordinator queues");

    // Small work is allowed to finish on the primary CPU: no synthetic GPU
    // work is created just to report utilization on both endpoints.
    ControlledEndpoint smallgpu,smallcpu;smallgpu.gpu=true;
    em::CooperativeEmStatistics smallstats;
    IndependentSubshowerPump<ControlledEndpoint> smallprimary(smallgpu,smallcpu,smallstats,
        []{}, {},20,false,64U<<20,true);
    completed.clear();auto smallinput=roots(100);
    require(smallprimary.advance(smallinput,cb)==100,"small primary front rejected");
    for(int i=0;!smallprimary.idle();++i){require(i<100,"small primary front stuck");smallprimary.advance({},cb);}
    require(completed.size()==100 && smallstats.subshower_cuda_submissions==0,
            "small CPU-primary front was forced onto GPU");

    // Native scalar-router order: consume owned residents before filling the
    // arena from newly staged particles of the same species. A full resident
    // front must not be displaced just because new arrivals exist.
    for(bool start_with_photons:{false,true}) {
      ControlledEndpoint ordergpu,ordercpu;ordergpu.gpu=true;
      ordercpu.host_capacity=8192;ordercpu.lifetime=2;
      em::CooperativeEmStatistics orderstats;
      IndependentSubshowerPump<ControlledEndpoint> ordered(ordergpu,ordercpu,
          orderstats,[]{},{},130,false,64U<<20,true);
      auto resident_roots=roots(8192),new_roots=roots(4095);
      for(auto& p:resident_roots)p.pid=start_with_photons?22:11;
      for(auto& p:new_roots){p.pid=start_with_photons?11:22;p.history_id+=20000;}
      completed.clear();
      require(ordered.advance(resident_roots,cb)==resident_roots.size(),
              "resident-order initial roots rejected");
      require(completed.empty(),"resident-order fixture prematurely terminated");
      require(ordered.advance(new_roots,cb)==new_roots.size(),
              "resident-order staged arrivals rejected");
      require(completed.size()==8192 && *completed.begin()==1 && *completed.rbegin()==8192,
              "new staged arrivals displaced the full previously resident CPU front");
      require(ordercpu.last_input==8192 && orderstats.subshower_cuda_submissions==0,
              "resident-order fixture changed arena size or assigned a sub-minimum GPU grant");
      for(int i=0;!ordered.idle();++i){require(i<100,"resident-order queues did not drain");ordered.advance({},cb);}
      ordered.requireIdle();
      require(completed.size()==12287 && orderstats.openmp_input_particles==2*12287,
              "resident-first refill lost/duplicated particles or work");
    }

    // CPU-primary auxiliary work may continue for four complete calls while
    // the main thread is genuinely blocked INSIDE the first OpenMP call.
    // Explicit promises make this an ownership/progress test, not timing luck.
    for(bool fail_followup:{false,true}) {
      ControlledEndpoint packetgpu,packetcpu;packetgpu.gpu=true;
      packetcpu.host_capacity=8192;packetgpu.lifetime=packetcpu.lifetime=8;
      packetgpu.pending[1]=roots(8192);packetcpu.pending[1]=roots(8192);
      for(auto& p:packetcpu.pending[1])p.history_id+=20000;
      std::promise<void> cpu_entered,gpu_fourth;
      packetgpu.permit=cpu_entered.get_future().share();
      packetcpu.permit=gpu_fourth.get_future().share();
      unsigned gpu_calls=0,cpu_calls=0;
      packetgpu.on_call=[&]{
        if(++gpu_calls==4){gpu_fourth.set_value();if(fail_followup)packetgpu.fail=true;}
      };
      packetcpu.on_call=[&]{if(++cpu_calls==1)cpu_entered.set_value();};
      em::CooperativeEmStatistics packetstats;
      IndependentSubshowerPump<ControlledEndpoint> packet(packetgpu,packetcpu,
          packetstats,[]{},{},130,false,64U<<20,true);
      completed.clear();bool threw=false;
      try {
        for(int i=0;!packet.idle();++i){require(i<5000,"CPU-primary fixed packet did not drain");packet.advance({},cb);}
      }catch(std::runtime_error const&){threw=true;}
      require(gpu_calls>=4 && threw==fail_followup,
              "CPU-primary auxiliary did not reach its fourth call or hid a followup failure");
      if(fail_followup) {
        bool unhealthy=false;
        try{packet.requireHealthy();}catch(std::logic_error const&){unhealthy=true;}
        require(unhealthy,"failed CPU-primary packet can report complete output");
      } else {
        packet.requireIdle();
        require(completed.size()==16384,"CPU-primary fixed packet lost/duplicated identities");
        require(packetstats.cuda_completion_mailbox_capacity==4 &&
                    packetstats.maximum_cuda_packet_calls==4 &&
                    packetstats.cuda_continuation_stops[3]>0 &&
                    packetstats.subshower_cuda_autonomous_continuations>=3,
                "CPU-primary packet failed its actual four-call boundary");
        auto const& leases=packetgpu.received_histories;
        for(std::size_t i=1;i<leases.size();++i)
          require(leases[i]>leases[i-1],"CPU-primary cross-species continuation reused an earlier history bank");
        for(unsigned k=0;k<2;++k)for(auto request:packetgpu.received_wave_requests[k])
          require(request.first==(k==0?16:8) && request.second==1,
                  "CPU-primary auxiliary continuation changed its fixed wave/minimum policy");
        std::uint64_t stops=0;for(auto n:packetstats.cuda_continuation_stops)stops+=n;
        require(stops+packetstats.subshower_cuda_autonomous_continuations==
                    packetstats.subshower_cuda_commits &&
                    packetstats.subshower_cuda_submissions==packetstats.subshower_cuda_commits,
                "CPU-primary packet commits or stop accounting is incomplete");
      }
    }

    // A completed primary front requests a safe return AFTER the current GPU
    // call, never by inspecting/modifying the in-flight endpoint queue.
    ControlledEndpoint handoffgpu,handoffcpu;handoffgpu.gpu=true;
    handoffcpu.host_capacity=8192;handoffgpu.lifetime=8;handoffcpu.lifetime=1;
    handoffgpu.pending[1]=roots(4096);handoffcpu.pending[1]=roots(4096);
    for(auto& p:handoffcpu.pending[1])p.history_id+=20000;
    std::promise<void> handoff_entered,handoff_release;
    handoffcpu.permit=handoff_entered.get_future().share();
    auto handoff_gate=handoff_release.get_future();
    unsigned handoff_calls=0;
    handoffgpu.on_call=[&]{if(++handoff_calls==1){
      handoff_entered.set_value();
      require(handoff_gate.wait_for(std::chrono::seconds(3))==std::future_status::ready,
              "CPU-primary handoff test blocked before main-thread progress");
    }};
    em::CooperativeEmStatistics handoffstats;
    IndependentSubshowerPump<ControlledEndpoint> handoff(handoffgpu,handoffcpu,
        handoffstats,[]{},{},130,false,64U<<20,true);
    completed.clear();handoff.advance({},cb);
    require(completed.size()==4096 && handoff.inFlight() && handoffstats.subshower_cuda_commits==0,
            "primary completion waited for in-flight auxiliary work");
    handoff_release.set_value();
    for(int i=0;!handoff.idle();++i){require(i<5000,"CPU-primary handoff did not drain");handoff.advance({},cb);}
    handoff.requireIdle();
    require(handoffstats.cuda_continuation_stops[4]>0 && handoff_calls==1 &&
                handoffstats.subshower_cuda_autonomous_continuations==0 && completed.size()==8192,
            "auxiliary ignored the primary's between-call handoff or lost states");

    // Shared retained-result budget: already owned particle states survive a
    // one-call packet boundary even when diagnostics exceed the soft budget.
    ControlledEndpoint retainedgpu,retainedcpu;retainedgpu.gpu=true;
    retainedcpu.host_capacity=8192;retainedgpu.lifetime=retainedcpu.lifetime=4;
    retainedgpu.pending[1]=roots(8192);retainedcpu.pending[1]=roots(8192);
    for(auto& p:retainedcpu.pending[1])p.history_id+=20000;
    retainedgpu.diagnostic_padding=1024;
    em::CooperativeEmStatistics retainedstats;
    IndependentSubshowerPump<ControlledEndpoint> retained(retainedgpu,retainedcpu,
        retainedstats,[]{},{},130,false,1,true);
    completed.clear();
    for(int i=0;!retained.idle();++i){require(i<5000,"CPU-primary retention boundary stuck");retained.advance({},cb);}
    retained.requireIdle();
    require(completed.size()==16384 && retainedstats.cuda_continuation_stops[2]>0 &&
                retainedstats.maximum_cuda_packet_calls==1 &&
                retainedstats.subshower_cuda_autonomous_continuations==0,
            "CPU-primary packet exceeded retention budget or lost identities");

    ControlledEndpoint emptygpu,emptycpu;emptygpu.gpu=true;
    emptycpu.host_capacity=8192;emptygpu.lifetime=emptycpu.lifetime=1;
    emptygpu.pending[1]=roots(4096);emptycpu.pending[1]=roots(4096);
    for(auto& p:emptycpu.pending[1])p.history_id+=20000;
    em::CooperativeEmStatistics emptystats;
    IndependentSubshowerPump<ControlledEndpoint> empty_packet(emptygpu,emptycpu,
        emptystats,[]{},{},130,false,64U<<20,true);
    completed.clear();
    for(int i=0;!empty_packet.idle();++i){require(i<5000,"empty CPU-primary packet did not terminate");empty_packet.advance({},cb);}
    empty_packet.requireIdle();
    require(completed.size()==8192 && emptystats.cuda_continuation_stops[0]>0 &&
                emptystats.subshower_cuda_autonomous_continuations==0,
            "empty resident GPU queue generated spurious continuations");

    // Zero-progress packets return existing spill states to the scalar owner;
    // another species remains resident, so this explicitly tests reason 1
    // rather than satisfying the earlier empty-resident termination branch.
    ControlledEndpoint stalledgpu,stalledcpu;stalledgpu.gpu=true;
    stalledcpu.host_capacity=8192;stalledgpu.lifetime=stalledcpu.lifetime=2;
    stalledgpu.pending[1]=roots(4096);stalledgpu.pending[0]=roots(4096);
    for(auto& p:stalledgpu.pending[0]){p.pid=22;p.history_id+=10000;}
    stalledcpu.pending[1]=roots(8192);
    for(auto& p:stalledcpu.pending[1])p.history_id+=20000;
    stalledgpu.spill_once=stalledgpu.refuse_allocation=true;
    em::CooperativeEmStatistics stalledstats;
    IndependentSubshowerPump<ControlledEndpoint> stalled(stalledgpu,stalledcpu,
        stalledstats,[]{},{},130,false,64U<<20,true);
    auto stalled_cb=cb;std::size_t zero_progress_spills=0;
    auto consume_stalled=[&](auto&& result,double elapsed){
      for(auto const& p:result.cpu_spill_particles){
        require(completed.insert(p.history_id).second,"duplicate CPU-primary zero-progress spill");
        ++zero_progress_spills;
      }
      result.cpu_spill_particles.clear();consume(std::move(result),elapsed);
    };
    stalled_cb.photons=consume_stalled;stalled_cb.leptons=consume_stalled;
    completed.clear();
    for(int i=0;!stalled.idle();++i){require(i<5000,"CPU-primary zero-progress retry loop");stalled.advance({},stalled_cb);}
    stalled.requireIdle();
    require(completed.size()==16384 && zero_progress_spills==4096 &&
                stalledstats.cuda_continuation_stops[1]>0,
            "zero-progress CPU-primary call retried autonomously or discarded its spill");
    std::cout<<"PASS: CPU-primary resident-first refill, fixed four-call auxiliary packets, handoff, retention and exception boundaries\n";

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
      require((agpu.minimum_seen==1 || agpu.minimum_seen==4096) &&
              acpu.minimum_seen==std::min<std::size_t>(4096,std::max<std::size_t>(1,acpu.lepton_input_seen/4)),
              "unexpected wavefront checkpoint threshold");
      require(completed.size()==total,"adaptive lost or duplicated terminal particles");
      require(astats.adaptive_policy && astats.subshower_cuda_commits==
              astats.subshower_cuda_submissions,"adaptive unfinished result");
      require(astats.adaptive_observations[0][0]+astats.adaptive_observations[0][1]>0 &&
              astats.adaptive_observations[1][0]+astats.adaptive_observations[1][1]>0,
              "adaptive did not calibrate both endpoints");
      std::uint64_t steps=0,jobs=0;
      for(auto const& endpoint:astats.adaptive_transport_records)for(auto n:endpoint)steps+=n;
      for(auto const& endpoint:astats.adaptive_job_input_histogram)for(auto n:endpoint)jobs+=n;
      require(steps==total*12,"physical step accounting lost/doubled completed work");
      require(jobs==astats.subshower_cuda_commits+astats.subshower_openmp_epochs,
              "job histogram differs from exact-once completions");
      std::uint64_t reasons=0;
      for(auto const& endpoint:astats.adaptive_completion_reasons)
        for(auto const& kind:endpoint)for(auto count:kind)reasons+=count;
      require(reasons==jobs,"completion reasons do not cover each job exactly once");
    }

    // The coordinator is deliberately BLOCKED inside an OpenMP call until
    // the GPU's fourth resident call starts. This cannot pass by merely
    // collecting/relaunching two GPU jobs sequentially on the main thread.
    for(bool fail_followup:{false,true}) {
      ControlledEndpoint autonomous_gpu,blocked_cpu;autonomous_gpu.gpu=true;
      autonomous_gpu.lifetime=blocked_cpu.lifetime=8;
      autonomous_gpu.pending[1]=roots(8192);blocked_cpu.pending[1]=roots(8192);
      for(auto& p:blocked_cpu.pending[1])p.history_id+=20000;
      std::promise<void> cpu_entered,gpu_continued;
      autonomous_gpu.permit=cpu_entered.get_future().share();
      blocked_cpu.permit=gpu_continued.get_future().share();
      unsigned gpu_calls=0,cpu_calls=0;
      autonomous_gpu.on_call=[&] {
        ++gpu_calls;
        // Give the first one-wave mock call a known minimum cost. The next
        // lepton lease must shrink from its uncalibrated 64-wave request.
        // Without this delay a test of the DIRECTION of learning depends on
        // machine speed/sanitizer overhead. This is not a timing benchmark.
        if(gpu_calls==1)std::this_thread::sleep_for(std::chrono::milliseconds(25));
        if(gpu_calls==4) {
          gpu_continued.set_value();
          if(fail_followup)autonomous_gpu.fail=true;
        }
      };
      blocked_cpu.on_call=[&] {if(++cpu_calls==1)cpu_entered.set_value();};
      em::CooperativeEmStatistics mailbox_stats;
      IndependentSubshowerPump<ControlledEndpoint> mailbox(
          autonomous_gpu,blocked_cpu,mailbox_stats,[]{},{},20,true);
      completed.clear();bool threw=false;
      try {
        for(int i=0;!mailbox.idle();++i) {
          require(i<5000,"autonomous mailbox failed to drain");mailbox.advance({},cb);
        }
      } catch(std::runtime_error const&) {threw=true;}
      require(threw==fail_followup,"autonomous followup failure was hidden or unexpectedly thrown");
      require(gpu_calls>=4,"GPU did not continue while coordinator was blocked");
      if(fail_followup) {
        bool unhealthy=false;
        try {mailbox.requireHealthy();}catch(std::logic_error const&){unhealthy=true;}
        require(unhealthy,"failed second packet can produce complete output");
      } else {
        mailbox.requireIdle();
        require(completed.size()==16384,"mailbox lost/duplicated terminal states");
        require(mailbox_stats.subshower_cuda_autonomous_continuations>0,
                "autonomous continuation not accounted");
        auto const& leases_waves=autonomous_gpu.received_lepton_wave_limits;
        require(leases_waves.size()>=2 && leases_waves[1]<leases_waves[0],
                "autonomous GPU reused a stale wave estimate instead of learning completed work");
        std::cout<<"private GPU cost learning: lepton wave lease "
                 <<leases_waves[0]<<" -> "<<leases_waves[1]<<'\n';
        require(mailbox_stats.maximum_cuda_packet_calls>=4 &&
                mailbox_stats.maximum_cuda_packet_calls<=mailbox_stats.cuda_completion_mailbox_capacity,
                "autonomous packet is truncated or exceeds its bound");
        auto const& leases=autonomous_gpu.received_histories;
        for(std::size_t i=1;i<leases.size();++i)
          require(leases[i]>leases[i-1],"alternating species reused an earlier history bank");
        std::uint64_t packets=0;
        for(auto n:mailbox_stats.cuda_continuation_stops)packets+=n;
        require(packets+mailbox_stats.subshower_cuda_autonomous_continuations==
                    mailbox_stats.subshower_cuda_commits,"continuation stop reasons miss a packet");
        require(mailbox_stats.subshower_cuda_commits==mailbox_stats.subshower_cuda_submissions,
                "mailbox call accounting differs from exactly-once commits");
        require(mailbox_stats.cuda_completion_buffer_delay_ms>=mailbox_stats.cuda_result_service_delay_ms,
                "GPU busy time confused with mailbox idle time");
      }
    }

    // A retained diagnostic result already exceeding the soft extra-storage
    // budget must return, not let the driver accumulate more such results.
    ControlledEndpoint memory_gpu,memory_cpu;memory_gpu.gpu=true;
    memory_gpu.pending[1]=roots(8192);memory_cpu.pending[1]=roots(8192);
    for(auto& p:memory_cpu.pending[1])p.history_id+=20000;
    memory_gpu.diagnostic_padding=1024;
    memory_gpu.lifetime=memory_cpu.lifetime=4;
    em::CooperativeEmStatistics memory_stats;
    IndependentSubshowerPump<ControlledEndpoint> memory_pump(
        memory_gpu,memory_cpu,memory_stats,[]{},{},20,true,1);
    completed.clear();
    for(int i=0;!memory_pump.idle();++i){require(i<5000,"memory-bounded mailbox did not drain");memory_pump.advance({},cb);}
    require(completed.size()==16384,"memory boundary discarded/duplicated particles");
    require(memory_stats.cuda_continuation_stops[2]>0 &&
            memory_stats.subshower_cuda_autonomous_continuations==0 &&
            memory_stats.maximum_cuda_packet_calls==1,
            "driver exceeded retained-memory continuation boundary");

    // Returning work to the scalar coordinator does not create a dependency
    // for the GPU's independent resident particles. Model foreground scalar
    // work by keeping this thread outside advance(), not in an OpenMP call.
    // CUDA must progress without invoking output/history callbacks off-thread.
    ControlledEndpoint foreground_gpu,foreground_cpu;foreground_gpu.gpu=true;
    foreground_gpu.pending[1]=roots(256);foreground_gpu.lifetime=8;
    std::promise<void> foreground_continued;
    auto foreground_signal=foreground_continued.get_future();
    unsigned foreground_calls=0;
    foreground_gpu.on_call=[&]{if(++foreground_calls==4)foreground_continued.set_value();};
    em::CooperativeEmStatistics foreground_stats;
    IndependentSubshowerPump<ControlledEndpoint> foreground_pump(
        foreground_gpu,foreground_cpu,foreground_stats,[]{},{},20,true);
    bool foreground_busy=true;
    auto foreground_cb=cb;
    foreground_cb.yield_to_scalar=[&]{return foreground_busy;};
    completed.clear();foreground_pump.advance({},foreground_cb);
    require(foreground_signal.wait_for(std::chrono::milliseconds(500))==std::future_status::ready,
            "GPU stalled during independent scalar foreground work");
    require(!foreground_cpu.entered && foreground_stats.subshower_openmp_epochs==0,
            "scalar-yield test unexpectedly ran OpenMP");
    foreground_busy=false;
    for(int i=0;!foreground_pump.idle();++i){require(i<5000,"scalar foreground queue did not drain");foreground_pump.advance({},foreground_cb);}
    require(completed.size()==256 && foreground_stats.subshower_cuda_autonomous_continuations>=3,
            "scalar foreground continuation lost identities or accounting");
    require(foreground_stats.subshower_cuda_foreground_packets>0 &&
            foreground_stats.subshower_cuda_foreground_continuations>=3,
            "scalar foreground submissions were not identified");

    // The real router sets its scalar-yield flag FROM the result consumer.
    // Exercise that transition, rather than only a flag set before advance.
    ControlledEndpoint returned_gpu,returned_cpu;returned_gpu.gpu=true;
    returned_gpu.pending[1]=roots(256);returned_gpu.lifetime=8;
    std::promise<void> returned_continued;
    auto returned_signal=returned_continued.get_future();
    unsigned returned_calls=0;
    returned_gpu.on_call=[&]{if(++returned_calls==5)returned_continued.set_value();};
    em::CooperativeEmStatistics returned_stats;
    IndependentSubshowerPump<ControlledEndpoint> returned_pump(
        returned_gpu,returned_cpu,returned_stats,[]{},{},20,true);
    bool returned_busy=false,returned_armed=true;
    auto returned_cb=cb;
    auto consume_and_yield=[&](auto&& result,double elapsed) {
      consume(std::move(result),elapsed);
      if(returned_armed){returned_busy=true;returned_armed=false;}
    };
    returned_cb.photons=consume_and_yield;returned_cb.leptons=consume_and_yield;
    returned_cb.yield_to_scalar=[&]{return returned_busy;};
    completed.clear();
    for(int i=0;!returned_busy;++i){require(i<5000,"initial scalar return not consumed");returned_pump.advance({},returned_cb);}
    require(returned_signal.wait_for(std::chrono::milliseconds(500))==std::future_status::ready,
            "consumer-generated scalar return stalled GPU continuation");
    returned_busy=false;
    for(int i=0;!returned_pump.idle();++i){require(i<5000,"returned scalar queue did not drain");returned_pump.advance({},returned_cb);}
    require(completed.size()==256 && returned_stats.subshower_cuda_foreground_packets>0 &&
            returned_stats.subshower_cuda_foreground_continuations>=3,
            "consumer-generated foreground lost identities or continuation accounting");

    // The call cap is independent of queue length and elapsed-time estimates.
    // Keep the coordinator blocked until call 16 while making each mock call
    // small; this tests a real packet boundary, not a post-run counter clamp.
    ControlledEndpoint capped_gpu,capped_cpu;capped_gpu.gpu=true;
    capped_gpu.lifetime=capped_cpu.lifetime=64;
    capped_gpu.pending[1]=roots(256);capped_cpu.pending[1]=roots(256);
    for(auto& p:capped_cpu.pending[1])p.history_id+=20000;
    std::promise<void> cap_cpu_entered,cap_gpu_ready;
    capped_gpu.permit=cap_cpu_entered.get_future().share();
    capped_cpu.permit=cap_gpu_ready.get_future().share();
    unsigned cap_gpu_calls=0,cap_cpu_calls=0;
    capped_gpu.on_call=[&]{if(++cap_gpu_calls==16)cap_gpu_ready.set_value();};
    capped_cpu.on_call=[&]{if(++cap_cpu_calls==1)cap_cpu_entered.set_value();};
    em::CooperativeEmStatistics cap_stats;
    IndependentSubshowerPump<ControlledEndpoint> capped(
        capped_gpu,capped_cpu,cap_stats,[]{},{},20,true);
    completed.clear();
    for(int i=0;!capped.idle();++i){require(i<5000,"call-capped driver did not drain");capped.advance({},cb);}
    require(completed.size()==512,"call cap discarded/duplicated identities");
    require(cap_stats.cuda_continuation_stops[3]>0 && cap_stats.maximum_cuda_packet_calls==16,
            "driver did not stop at the actual 16-call packet boundary");

    // A single ordinary call may overrun the soft horizon, but no further
    // call is allowed in that packet. No kernel preemption or particle cut.
    ControlledEndpoint timed_gpu,timed_cpu;timed_gpu.gpu=true;
    timed_gpu.pending[1]=roots(256);timed_cpu.pending[1]=roots(256);
    for(auto& p:timed_cpu.pending[1])p.history_id+=20000;
    timed_gpu.lifetime=timed_cpu.lifetime=4;
    std::promise<void> timed_call_finished;
    timed_cpu.permit=timed_call_finished.get_future().share();
    unsigned timed_calls=0;
    timed_gpu.on_call=[&]{
      if(++timed_calls==1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        timed_call_finished.set_value();
      }
    };
    em::CooperativeEmStatistics timed_stats;
    IndependentSubshowerPump<ControlledEndpoint> timed(
        timed_gpu,timed_cpu,timed_stats,[]{},{},20,true);
    completed.clear();
    for(int i=0;!timed.idle();++i){require(i<5000,"time-bounded driver did not drain");timed.advance({},cb);}
    require(completed.size()==512 && timed_stats.cuda_continuation_stops[5]>0,
            "soft time horizon lost states or failed to stop continuation");

    // The entire continuation bank must fit BEFORE any GPU work is launched.
    // Exercise addition overflow, not just overlapping first-history values.
    ControlledEndpoint overflow_gpu,overflow_cpu;overflow_gpu.gpu=true;
    overflow_gpu.pending[1]=roots(256);overflow_cpu.pending[1]=roots(256);
    em::CooperativeEmStatistics overflow_stats;
    IndependentSubshowerPump<ControlledEndpoint> overflow_pump(
        overflow_gpu,overflow_cpu,overflow_stats,[]{},{},20,true);
    auto overflow_cb=cb;unsigned overflow_requests=0;
    overflow_cb.reserve_histories=[&](std::uint64_t){
      return ++overflow_requests==1?std::uint64_t{1000000}:
          std::numeric_limits<std::uint64_t>::max()-1;
    };
    bool overflow_rejected=false;
    try{overflow_pump.advance({},overflow_cb);}catch(std::overflow_error const&){overflow_rejected=true;}
    require(overflow_rejected && !overflow_gpu.entered && overflow_stats.subshower_cuda_submissions==0,
            "overflowed continuation history bank reached the GPU");

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
    require(prefixcpu.minimum_seen==512,"large CPU front still drains to a single-particle tail");
    require(prefixcpu.pending_limit==std::numeric_limits<std::size_t>::max(),
            "CPU pending-input bound leaked after job");
    for(int i=0;!prefix.idle();++i){require(i<5000,"bounded-prefix queue did not drain");prefix.advance({},cb);}
    require(completed.size()==12000,"bounded input lost pending particles");
    require(cpu.bounded_input_calls==0,"legacy independent CPU input behavior changed");

    ControlledEndpoint coal_gpu,coal_cpu;coal_gpu.gpu=true;
    coal_cpu.host_capacity=65536;coal_cpu.lifetime=coal_gpu.lifetime=1;
    coal_cpu.pending[1]=roots(12000);
    auto lone=roots(1);lone[0].pid=22;lone[0].history_id=20000;
    coal_cpu.pending[0]=lone;
    em::CooperativeEmStatistics coal_stats;
    IndependentSubshowerPump<ControlledEndpoint> coalesced(coal_gpu,coal_cpu,coal_stats,[]{},{},20,true);
    completed.clear();
    for(int i=0;!coalesced.idle();++i){require(i<5000,"coalesced small tail starved");coalesced.advance({},cb);}
    require(completed.size()==12001,"coalescing lost/duplicated a particle");
    require(coal_stats.adaptive_species_coalesces[1]>0,"tiny species submitted ahead of a useful batch");

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
                "4+ autonomous GPU calls during blocked CPU, bounded result memory, "
                "16-call/time boundaries, scalar-foreground progress, per-call GPU cost learning, ordered history banks/overflow rejection, "
                "exact-once output and exception cleanup\n";
  } catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
}
