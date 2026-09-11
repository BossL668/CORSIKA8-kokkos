/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#pragma once
#include <corsika/accelerator/em/detail/CooperativeBackendMerge.hpp>
#include <corsika/accelerator/em/detail/IndependentEndpointDriver.hpp>
#include <corsika/accelerator/em/detail/CooperativeHostPolicy.hpp>
#include <corsika/accelerator/em/detail/AdaptiveSubshowerControl.hpp>
#include <corsika/accelerator/em/detail/SubshowerCallbacks.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <type_traits>
#include <variant>
#include <vector>

namespace corsika::accelerator::em::detail {
// Endpoint is deliberately a template: the host scheduling/ownership contract
// is tested with controlled endpoints without CUDA, physics or timing guesses.
// Only the CUDA endpoint runs on driver_. CPU, leases and consumers stay here.
template<class Endpoint> class IndependentSubshowerPump {
  using Particle = gpu::em::EmParticleState;
  using Photon = gpu::em::ResidentPhotonCascadeResult;
  using Lepton = gpu::em::ResidentLeptonCascadeResult;
  using Clock = std::chrono::steady_clock;
  using Statistics = gpu::em::CooperativeEmStatistics;
  struct Peer {
    std::array<std::vector<Particle>, 2> waiting;
    std::array<std::size_t, 2> resident{}, capacity{};
    unsigned next_kind{};
    std::size_t count(unsigned kind) const {
      return resident[kind] + waiting[kind].size();
    }
    bool empty() const { return !count(0) && !count(1); }
  };
  struct Job {
    unsigned kind{};
    std::vector<Particle> input;
    std::size_t count{}, waves{}, minimum{};
    std::size_t resident_input{};
    bool bounded_input{};
    std::uint64_t first{}, limit{}, sequence{};
  };
  struct Completion {
    std::variant<Photon, Lepton> result;
    std::array<std::size_t, 2> resident;
    Job identity;
    Clock::time_point begin, end;
  };
 public:
  IndependentSubshowerPump(Endpoint& gpu, Endpoint& cpu, Statistics& statistics,
                           std::function<void()> select_device,
                           std::function<void(bool)> observe_host = {},
                           std::size_t host_threads = 20,bool adaptive = false)
      : gpu_(gpu), cpu_(cpu), stats_(statistics),
        select_device_(std::move(select_device)), observe_host_(std::move(observe_host)),
        initial_host_share_(adaptive?.5:CooperativeHostPolicy::initialShare(host_threads)),
        adaptive_(adaptive) {
    for (unsigned e=0; e<2; ++e) {
      auto& b=endpoint(e); auto& p=peers_[e];
      p.capacity={b.maximumResidentPhotonBatchSize(), b.maximumResidentLeptonBatchSize()};
      p.resident={b.pendingPhotonCount(), b.pendingLeptonCount()};
      for (auto n:p.capacity)
        if (!n || n>std::numeric_limits<std::size_t>::max()/4)
          throw std::length_error("Invalid independent subshower capacity");
    }
    stats_.independent_subshowers=true;
    stats_.adaptive_policy=adaptive_;
    stats_.openmp_batch_target=peers_[1].capacity[1];
    stats_.subshower_initial_host_share=initial_host_share_;
  }
  ~IndependentSubshowerPump() { driver_.waitIdle(); }
  IndependentSubshowerPump(IndependentSubshowerPump const&)=delete;
  IndependentSubshowerPump& operator=(IndependentSubshowerPump const&)=delete;

  // Never read an endpoint queue or statistics while its driver is using it.
  // A submitted result remains pending even if all its particles terminated.
  std::size_t pending(unsigned kind) const noexcept {
    return peers_[0].count(kind)+peers_[1].count(kind)+
        (flight_.valid() && flight_kind_==kind ? flight_count_ : 0);
  }
  bool idle() const noexcept {
    return !flight_.valid() && peers_[0].empty() && peers_[1].empty();
  }
  bool inFlight() const noexcept { return flight_.valid(); }
  bool ready() const noexcept {
    return !peers_[1].empty() || canFeedHostWhileInFlight() ||
        (flight_.valid() ? gpuReady() : !peers_[0].empty());
  }
  void requireHealthy() const {
    if (failed_) throw std::logic_error("Independent subshower failed; cannot complete output");
  }
  void requireIdle() const {
    requireHealthy();
    if (!idle()) throw std::logic_error("Independent subshower has uncommitted work");
  }

  // Accept a bounded prefix, service any ready GPU result, and execute ONE
  // host epoch. Return without joining an unfinished GPU job. The caller may
  // service scalar fallback between calls. No callback is saved by the pump.
  std::size_t advance(std::vector<Particle> const& input,
                      SubshowerCallbacks const& callbacks) {
    requireHealthy();
    if (advancing_) throw std::logic_error("Reentrant independent subshower progress");
    if (!callbacks.reserve_histories || !callbacks.photons || !callbacks.leptons)
      throw std::invalid_argument("Independent subshower callbacks are incomplete");
    advancing_=true;
    auto start=Clock::now();
    try {
      if (gpuReady()) collect(callbacks);
      auto accepted=enqueue(input);
      rebalance();
      if (!flight_.valid() && !peers_[0].empty()) launch(callbacks);
      if (!yield(callbacks) && !peers_[1].empty()) {
        auto job=prepare(1, callbacks);
        bool concurrent=flight_.valid();
        auto begin=Clock::now();
        if (observe_host_) observe_host_(true);
        Completion result;
        try { result=execute(cpu_, std::move(job)); }
        catch (...) { if(observe_host_) observe_host_(false); throw; }
        if (observe_host_) observe_host_(false);
        auto end=Clock::now();
        auto duration=milliseconds(begin,end);
        ++stats_.openmp_slices;
        if (duration > 2.) ++stats_.oversized_slices;
        stats_.openmp_wall_ms+=duration;
        stats_.maximum_slice_ms=std::max(stats_.maximum_slice_ms,duration);
        if (concurrent) {
          ++stats_.independent_joint_calls;
          ++stats_.slices_while_cuda_pending;
          ++host_epochs_in_flight_;
          stats_.maximum_host_epochs_per_cuda_job=std::max(
              stats_.maximum_host_epochs_per_cuda_job,host_epochs_in_flight_);
          auto a=gpu_begin_ns_.load(std::memory_order_acquire);
          auto b=gpu_end_ns_.load(std::memory_order_acquire);
          if (a) {
            auto lo=std::max(nanoseconds(begin),a);
            auto hi=std::min(nanoseconds(end),b?b:nanoseconds(end));
            auto overlap=std::max<std::int64_t>(0,hi-lo)*1.e-6;
            stats_.endpoint_window_overlap_ms+=overlap;
            stats_.openmp_while_cuda_pending_ms+=overlap;
          }
        }
        commit(1,std::move(result),callbacks);
      }
      if (gpuReady()) collect(callbacks);
      if (!flight_.valid() && !peers_[0].empty() && !yield(callbacks)) launch(callbacks);
      // No runnable host work: bounded blocking poll avoids a busy spin. It
      // does not wait for the entire GPU batch, and never waits with CPU work.
      if (flight_.valid() && peers_[1].empty() && !yield(callbacks)) {
        auto wait=Clock::now();
        flight_.wait_for(std::chrono::milliseconds(1));
        auto duration=milliseconds(wait,Clock::now());
        stats_.coordinator_idle_wait_ms+=duration;
        flight_host_starved_ms_+=duration;
      }
      stats_.joint_wall_ms+=milliseconds(start,Clock::now());
      recordQueuePeak();
      advancing_=false;
      return accepted;
    } catch (...) {
      failed_=true;
      driver_.waitIdle(); // error cleanup only, never successful batch rendezvous
      advancing_=false;
      throw;
    }
  }
 private:
  Endpoint& endpoint(unsigned e) {return e==0?gpu_:cpu_;}
  static bool yield(SubshowerCallbacks const& c) {return c.yield_to_scalar && c.yield_to_scalar();}
  static double milliseconds(Clock::time_point a,Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
  }
  static std::int64_t nanoseconds(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
  }
  bool gpuReady() const noexcept {
    return flight_.valid() && flight_.wait_for(std::chrono::seconds(0))==std::future_status::ready;
  }
  std::size_t enqueue(std::vector<Particle> const& input) {
    std::array<std::size_t,2> incoming{};
    for(auto const& p:input) {
      if(p.pid==22) ++incoming[0];
      else if(gpu::em::isChargedLeptonPid(p.pid)) ++incoming[1];
      else throw std::invalid_argument("Unsupported independent subshower PID");
    }
    if(adaptive_) return enqueueAdaptive(input);
    auto keep=gpu_.minimumBatchSize(); // immutable after initialization
    std::array<std::size_t,2> host_quota{};
    for(unsigned k=0;k<2;++k) {
      // Refill below one epoch before the CPU goes empty. Its descendants
      // still stay on their owning endpoint; only unassigned input is split.
      if(peers_[1].count(k)<peers_[1].capacity[k]) {
        auto total=incoming[k]+peers_[0].count(k)+
            (flight_.valid() && flight_kind_==k?flight_count_:0);
        double share=rates_[k][0]>0 && rates_[k][1]>0?
            rates_[k][1]/(rates_[k][0]+rates_[k][1]):initial_host_share_;
        host_quota[k]=total>keep?std::min({incoming[k],total-keep,
            2*peers_[1].capacity[k]-peers_[1].count(k),
            std::max<std::size_t>(256,incoming[k]*share)}):0;
        if(host_quota[k] && !peers_[1].empty()) ++stats_.subshower_proactive_refills;
      }
    }
    std::size_t accepted=0;
    for(auto const& p:input) {
      unsigned k=p.pid==22?0:1;
      auto room=[&](unsigned e) {return peers_[e].waiting[k].size()<2*peers_[e].capacity[k];};
      unsigned e=host_quota[k] && room(1)?1:0;
      if(!room(e)) {e=1-e;if(!room(e))break;}
      peers_[e].waiting[k].push_back(p);
      if(e==1 && host_quota[k])--host_quota[k];
      ++accepted;
    }
    return accepted;
  }
  Job prepare(unsigned e,SubshowerCallbacks const& callbacks) {
    auto& p=peers_[e];
    unsigned k=p.next_kind;
    if(!p.count(k)) k=1-k;
    if(!p.count(k)) throw std::logic_error("Submitting an empty subshower epoch");
    p.next_kind=1-k; // each endpoint alternates its own photon/lepton queues
    Job job;job.kind=k;job.sequence=++submitted_[e];
    auto target=adaptive_ && e==1?control_.hostTargetMs(flight_.valid()):
        AdaptiveSubshowerControl::epoch_target_ms;
    auto capacity=adaptive_ && e==1?control_.inputLimit(e,k,p.capacity[k],target):
        p.capacity[k];
    auto external=std::min(capacity,p.waiting[k].size());
    job.input.assign(p.waiting[k].begin(),p.waiting[k].begin()+external);
    p.waiting[k].erase(p.waiting[k].begin(),p.waiting[k].begin()+external);
    auto resident=std::min(capacity-external,p.resident[k]);
    job.resident_input=resident;
    job.bounded_input=adaptive_ && e==1;
    job.count=external+resident;
    p.resident[k]-=resident; // ownership moves to the job, not a second queue entry
    job.waves=adaptive_?control_.waves(e,k,job.count,target):
        (k==0?16:(e==0?gpu_lepton_waves_:8));
    // Split tails must not return after every single wave merely because
    // each half is smaller than the original unsplit 4096-input threshold.
    // The measured epoch controller already supplies a bounded hand-off.
    job.minimum=adaptive_?1:(e==0?gpu_.minimumBatchSize():1);
    if(adaptive_) stats_.adaptive_wave_limit[e][k]=job.waves;
    auto n=static_cast<std::uint64_t>(job.count);
    if(n>std::numeric_limits<std::uint64_t>::max()/192)
      throw std::overflow_error("Subshower history request overflow");
    auto reserve=k==0?2*n*job.waves:std::max(3*n,std::min<std::uint64_t>(1U<<20,192*n));
    job.first=callbacks.reserve_histories(reserve);
    job.limit=cooperativeAdd(job.first,reserve);
    if(!job.first || job.first<last_history_limit_)
      throw std::logic_error("Overlapping/nonmonotonic subshower history lease");
    last_history_limit_=job.limit;
    if(e==0) stats_.cuda_input_particles+=n;
    else stats_.openmp_input_particles+=n;
    return job;
  }
  static Completion execute(Endpoint& backend,Job job) {
    // Limit resident input as well as the external prefix. Otherwise the
    // backend silently fills the remaining arena and defeats the time budget.
    // Restore even on an exception; legacy/single-endpoint calls are untouched.
    struct InputLimitScope {
      Endpoint& backend; bool active;
      ~InputLimitScope() {
        if(active)backend.setCooperativePendingInputLimit(std::numeric_limits<std::size_t>::max());
      }
    } input_scope{backend,job.bounded_input};
    if(job.bounded_input)backend.setCooperativePendingInputLimit(job.resident_input);
    Completion done;done.begin=Clock::now();
    if(job.kind==0) done.result=backend.runPhotonWavefront(job.input,job.first,job.waves,job.minimum);
    else done.result=backend.runLeptonWavefront(job.input,job.first,job.waves,job.limit,job.minimum);
    done.end=Clock::now();
    done.resident={backend.pendingPhotonCount(),backend.pendingLeptonCount()};
    std::visit([&](auto const& r) {
      if(r.input_particles!=job.count)
        throw std::logic_error("Subshower epoch input accounting mismatch");
    },done.result);
    if(job.kind==1 && std::get<Lepton>(done.result).secondary_history_ids_used>job.limit-job.first)
      throw std::logic_error("Subshower exceeded its history lease");
    // Release the input capture before returning a potentially large result.
    std::vector<Particle>().swap(job.input);
    done.identity=std::move(job);
    return done;
  }
  void launch(SubshowerCallbacks const& callbacks) {
    auto job=prepare(0,callbacks);
    flight_kind_=job.kind;flight_count_=job.count;
    host_epochs_in_flight_=0;
    flight_host_starved_ms_=0.;
    gpu_begin_ns_=0;gpu_end_ns_=0;
    flight_=driver_.submit([this,job=std::move(job)]() mutable {
      select_device_();
      gpu_begin_ns_.store(nanoseconds(Clock::now()),std::memory_order_release);
      auto done=execute(gpu_,std::move(job));
      gpu_end_ns_.store(nanoseconds(done.end),std::memory_order_release);
      return done;
    });
    ++stats_.subshower_cuda_submissions;
  }
  void collect(SubshowerCallbacks const& callbacks) {
    if(!gpuReady()) throw std::logic_error("Blocking subshower collect is forbidden");
    auto result=flight_.get();
    driver_.waitIdle(); // packaged_task teardown, after result is already ready
    stats_.cuda_driver_wall_ms+=milliseconds(result.begin,result.end);
    stats_.cuda_result_service_delay_ms+=milliseconds(result.end,Clock::now());
    auto duration=milliseconds(result.begin,result.end);
    if(adaptive_)control_.observeGpuDuration(duration);
    stats_.subshower_maximum_cuda_epoch_ms=std::max(
        stats_.subshower_maximum_cuda_epoch_ms,duration);
    // A slow GPU must reach a safe redistribution boundary when the CPU is
    // starved. Never interrupt a kernel or alter a particle's physical step.
    if(!adaptive_ && result.identity.kind==1) {
      if(flight_host_starved_ms_>5. && duration>100.) {
        auto target=static_cast<std::size_t>(result.identity.waves*100./duration);
        auto next=std::max<std::size_t>(16,std::min(gpu_lepton_waves_/2,target));
        if(next<gpu_lepton_waves_) ++stats_.subshower_cuda_epoch_reductions;
        gpu_lepton_waves_=next;
      } else if(flight_host_starved_ms_<1. && duration<50.) {
        gpu_lepton_waves_=std::min<std::size_t>(1024,2*gpu_lepton_waves_);
      }
    }
    stats_.subshower_gpu_lepton_wave_limit=gpu_lepton_waves_;
    ++stats_.subshower_cuda_commits;
    commit(0,std::move(result),callbacks);
  }
  void retain(unsigned e,unsigned k,std::vector<Particle>& values,std::vector<Particle>& spill) {
    auto& p=peers_[e];auto& queue=p.waiting[k];
    auto bound=4*p.capacity[k];
    if(queue.size()>bound) throw std::logic_error("Subshower queue exceeded its bound");
    auto n=std::min(values.size(),bound-queue.size());
    queue.insert(queue.end(),std::make_move_iterator(values.begin()),
                 std::make_move_iterator(values.begin()+n));
    // Explicit existing memory-spill path, not an unbounded host queue or a
    // particle cut. The router records and returns every excess state.
    spill.insert(spill.end(),std::make_move_iterator(values.begin()+n),
                 std::make_move_iterator(values.end()));
    values.clear();
  }
  void commit(unsigned e,Completion done,SubshowerCallbacks const& callbacks) {
    auto const& job=done.identity;
    if(job.sequence!=committed_[e]+1)
      throw std::logic_error("Duplicate/out-of-order subshower completion");
    ++committed_[e];
    peers_[e].resident=done.resident;
    auto elapsed=milliseconds(done.begin,done.end);
    std::visit([&](auto& result) {
      double rate=result.transport_records/std::max(elapsed,1.e-6);
      auto& average=rates_[job.kind][e];
      if(rate>0) average=average==0?rate:.8*average+.2*rate;
      if(adaptive_) {
        control_.observe(e,job.kind,job.count,result.wavefronts,job.waves,
                         result.transport_records,elapsed);
        auto const& s=control_.sample(e,job.kind);
        stats_.adaptive_records_per_ms[e][job.kind]=s.records_per_ms;
        stats_.adaptive_observations[e][job.kind]=s.observations;
      }
      using R=std::decay_t<decltype(result)>;
      if constexpr(std::is_same_v<R,Photon>) {
        retain(e,0,result.remaining_photons,result.cpu_spill_particles);
        retain(e,1,result.electromagnetic_secondaries,result.cpu_spill_particles);
        // A prelaunch allocation refusal made no progress: preserve the old
        // scalar escape route instead of retrying the same impossible batch.
        if(result.wavefronts) requeueSpills(e,result.cpu_spill_particles);
        ++stats_.photon_calls;
        callbacks.photons(std::move(result),elapsed);
      } else {
        retain(e,1,result.remaining_leptons,result.cpu_spill_particles);
        retain(e,0,result.generated_photons,result.cpu_spill_particles);
        if(result.wavefronts) requeueSpills(e,result.cpu_spill_particles);
        ++stats_.lepton_calls;
        callbacks.leptons(std::move(result),elapsed);
      }
    },done.result);
    if(e==1)++stats_.subshower_openmp_epochs;
  }
  void requeueSpills(unsigned owner,std::vector<Particle>& spill) {
    // An endpoint's cross-species arena being full is a storage event, not a
    // request for scalar physics. The completed packet owns these particles;
    // they are not also resident/in flight. Keep them on their endpoint first.
    // Only coordinator-owned waiting vectors are touched, even when the other
    // endpoint is active. Neither endpoint kernel reads these vectors.
    std::size_t remaining=0;
    for(auto& particle:spill) {
      unsigned k;
      if(particle.pid==22) k=0;
      else if(gpu::em::isChargedLeptonPid(particle.pid)) k=1;
      else throw std::logic_error("Non-EM particle in subshower memory spill");
      bool queued=false;
      for(unsigned e:{owner,1-owner}) {
        auto& peer=peers_[e];auto& queue=peer.waiting[k];
        if(queue.size()>=4*peer.capacity[k])continue;
        queue.push_back(std::move(particle)); // retain all random/history state
        ++stats_.subshower_requeued_spill_particles;
        if(e!=owner) {
          ++stats_.subshower_cross_endpoint_spill_particles;
          stats_.migration_bytes=cooperativeAdd(stats_.migration_bytes,sizeof(Particle));
        }
        queued=true;break;
      }
      if(!queued) {
        if(&spill[remaining]!=&particle)spill[remaining]=std::move(particle);
        ++remaining; // explicit old scalar spill policy only if BOTH queues full
      }
    }
    spill.resize(remaining);
  }
  bool canFeedHostWhileInFlight() const noexcept {
    if(!flight_.valid()) return false;
    for(unsigned k=0;k<2;++k)
      if(peers_[1].count(k)<peers_[1].capacity[k] &&
         peers_[0].waiting[k].size()>=256) return true;
    return false;
  }
  void rebalance() {
    if(adaptive_) {rebalanceAdaptive();return;}
    // Coordinator-owned waiting vectors are NOT read by the GPU driver:
    // prepare() moves a separate input capture into the submitted job.
    // Steal those safe particles without touching any live endpoint state.
    if(flight_.valid()) {
      for(unsigned k=0;k<2;++k) {
        auto& donor=peers_[0].waiting[k];auto& receiver=peers_[1];
        if(receiver.count(k)>=receiver.capacity[k] || donor.size()<256)continue;
        auto count=std::min(donor.size(),2*receiver.capacity[k]-receiver.count(k));
        receiver.waiting[k].insert(receiver.waiting[k].end(),
            std::make_move_iterator(donor.end()-count),std::make_move_iterator(donor.end()));
        donor.erase(donor.end()-count,donor.end());
        stats_.subshower_inflight_waiting_migrations++;
        stats_.subshower_inflight_waiting_particles+=count;
        stats_.migration_bytes=cooperativeAdd(stats_.migration_bytes,
            static_cast<std::uint64_t>(count*sizeof(Particle)));
      }
      return; // device-resident and in-flight particles remain exclusive
    }
    for(unsigned destination=0;destination<2;++destination) {
      auto source=1-destination;
      auto& donor=peers_[source];auto& receiver=peers_[destination];
      if(!receiver.empty() || donor.empty())continue;
      unsigned k=donor.count(0)>=donor.count(1)?0:1;
      auto keep=source==0?gpu_.minimumBatchSize():std::size_t{256};
      if(donor.count(k)<=keep+256)continue;
      auto count=std::min(receiver.capacity[k],(donor.count(k)-keep)/2);
      if(count<256)continue; // do not shuttle tiny tails merely to make both busy
      auto& from=donor.waiting[k];auto& to=receiver.waiting[k];
      auto host=std::min(count,from.size());
      to.insert(to.end(),std::make_move_iterator(from.end()-host),std::make_move_iterator(from.end()));
      from.erase(from.end()-host,from.end());
      auto device=count-host;
      if(device) {
        auto moved=endpoint(source).takeCooperativePending(k==0,device);
        if(moved.size()!=device)throw std::logic_error("Subshower migration count mismatch");
        to.insert(to.end(),std::make_move_iterator(moved.begin()),std::make_move_iterator(moved.end()));
        donor.resident[k]-=device;
      }
      stats_.migration_bytes=cooperativeAdd(stats_.migration_bytes,
          static_cast<std::uint64_t>(count*sizeof(Particle)));
      ++stats_.subshower_tail_migrations;
    }
  }
  void recordQueuePeak() {
    std::uint64_t bytes=0;
    for(auto const& p:peers_) for(auto const& q:p.waiting)
      bytes=cooperativeAdd(bytes,static_cast<std::uint64_t>(q.capacity()*sizeof(Particle)));
    stats_.subshower_host_queue_peak_bytes=std::max(stats_.subshower_host_queue_peak_bytes,bytes);
  }
  std::size_t outstanding(unsigned e,unsigned k) const {
    return peers_[e].count(k)+(e==0 && flight_.valid() && flight_kind_==k?flight_count_:0);
  }
  double estimatedLoad(unsigned e) const {
    double load=0.;
    for(unsigned k=0;k<2;++k)load+=outstanding(e,k)*control_.unitCost(e,k);
    return load;
  }
  bool adaptiveRoom(unsigned e,unsigned k) const {
    if(peers_[e].waiting[k].size()>=2*peers_[e].capacity[k]) return false;
    return control_.measured(e,k) ||
        outstanding(e,k)<std::min(peers_[e].capacity[k],
                                  AdaptiveSubshowerControl::discovery_particles);
  }
  std::size_t enqueueAdaptive(std::vector<Particle> const& input) {
    std::array<double,2> load{estimatedLoad(0),estimatedLoad(1)};
    std::size_t accepted=0;
    for(auto const& particle:input) {
      unsigned k=particle.pid==22?0:1;
      bool room0=adaptiveRoom(0,k),room1=adaptiveRoom(1,k);
      if(!room0 && !room1)break; // bounded prefix, no particles discarded
      unsigned e=!room0?1:!room1?0:
          (load[0]+control_.unitCost(0,k)<=load[1]+control_.unitCost(1,k)?0:1);
      peers_[e].waiting[k].push_back(particle);
      load[e]+=control_.unitCost(e,k);
      ++accepted;
    }
    return accepted;
  }
  void rebalanceAdaptive() {
    // At most one transfer per species per invocation. Waiting vectors are
    // coordinator-owned; resident GPU state is inspected only after collect.
    // Predict balancing benefit INCLUDING measured copy cost and hysteresis.
    for(unsigned k=0;k<2;++k) {
      double a=estimatedLoad(0),b=estimatedLoad(1);
      unsigned source=a>b?0:1,destination=1-source;
      auto& donor=peers_[source];auto& receiver=peers_[destination];
      if(!adaptiveRoom(destination,k))continue;
      auto available=donor.waiting[k].size()+
          (source==0 && flight_.valid()?0:donor.resident[k]);
      auto room=2*receiver.capacity[k]-std::min(2*receiver.capacity[k],receiver.waiting[k].size());
      if(!control_.measured(destination,k)) {
        auto bound=std::min(receiver.capacity[k],AdaptiveSubshowerControl::discovery_particles);
        auto owned=outstanding(destination,k);
        room=std::min(room,bound-std::min(bound,owned));
      }
      // Do not fragment a useful GPU staging batch merely to even two noisy
      // estimates while the CPU already has work. An idle CPU may still steal
      // a profitable tail, important on CPU-heavy servers with a slow GPU.
      if(source==0 && !receiver.empty()) {
        auto keep=std::min(available,gpu_.minimumBatchSize());
        available-=keep;
      }
      auto cap=std::min(available,room);
      if(!cap)continue;
      auto saving=std::abs(a-b);
      auto sum=control_.unitCost(source,k)+control_.unitCost(destination,k);
      auto predicted=std::min(static_cast<double>(cap),saving/sum);
      auto count=static_cast<std::size_t>(predicted);
      if(!count)continue;
      double source_load=source==0?a:b,destination_load=destination==0?a:b;
      double before=std::max(source_load,destination_load);
      double after=std::max(source_load-count*control_.unitCost(source,k),
                           destination_load+count*control_.unitCost(destination,k));
      double benefit=before-after;
      // Compare the predicted critical path, not only donor work removed.
      // Hysteresis prevents oscillatory migrations between two nonempty peers.
      if(benefit<=1.+2.*count*migration_ms_per_particle_+
          (receiver.empty()?0.:.1*before))continue;
      auto begin=Clock::now();
      auto& from=donor.waiting[k];auto& to=receiver.waiting[k];
      auto host=std::min(count,from.size());
      to.insert(to.end(),std::make_move_iterator(from.end()-host),
                         std::make_move_iterator(from.end()));
      from.erase(from.end()-host,from.end());
      auto device=count-host;
      if(device) {
        auto moved=endpoint(source).takeCooperativePending(k==0,device);
        if(moved.size()!=device)throw std::logic_error("Adaptive migration count mismatch");
        to.insert(to.end(),std::make_move_iterator(moved.begin()),
                           std::make_move_iterator(moved.end()));
        donor.resident[k]-=device;
      }
      auto elapsed=milliseconds(begin,Clock::now());
      migration_ms_per_particle_=.75*migration_ms_per_particle_+.25*elapsed/count;
      stats_.adaptive_migration_ms+=elapsed;
      stats_.migration_bytes=cooperativeAdd(stats_.migration_bytes,
          static_cast<std::uint64_t>(count*sizeof(Particle)));
      ++stats_.subshower_tail_migrations;
      if(flight_.valid()) {
        ++stats_.subshower_inflight_waiting_migrations;
        stats_.subshower_inflight_waiting_particles+=count;
      }
    }
  }
  Endpoint& gpu_; Endpoint& cpu_; Statistics& stats_;
  std::function<void()> select_device_;
  std::function<void(bool)> observe_host_;
  double initial_host_share_,flight_host_starved_ms_{};
  std::size_t gpu_lepton_waves_{1024};
  std::array<Peer,2> peers_;
  std::array<std::array<double,2>,2> rates_{};
  std::array<std::uint64_t,2> submitted_{},committed_{};
  std::uint64_t last_history_limit_{},host_epochs_in_flight_{};
  unsigned flight_kind_{}; std::size_t flight_count_{};
  bool advancing_{},failed_{},adaptive_{};
  AdaptiveSubshowerControl control_;
  double migration_ms_per_particle_{};
  std::atomic<std::int64_t> gpu_begin_ns_{0},gpu_end_ns_{0};
  std::future<Completion> flight_;
  IndependentEndpointDriver driver_; // drain before queues, callbacks and Views
};
} // namespace corsika::accelerator::em::detail
