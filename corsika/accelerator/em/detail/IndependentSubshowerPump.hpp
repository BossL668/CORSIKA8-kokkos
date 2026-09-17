/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#pragma once
#include <corsika/accelerator/em/detail/CooperativeBackendMerge.hpp>
#include <corsika/accelerator/em/detail/IndependentEndpointDriver.hpp>
#include <corsika/accelerator/em/detail/CooperativeHostPolicy.hpp>
#include <corsika/accelerator/em/detail/CpuPrimarySubshowerPolicy.hpp>
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
    std::array<unsigned,2> deferrals{};
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
    bool scalar_foreground_continuation{};
    std::size_t input_limit{};
    double target_ms{};
    std::uint64_t first{}, limit{}, sequence{};
  };
  struct Completion {
    std::variant<Photon, Lepton> result;
    std::array<std::size_t, 2> resident;
    Job identity;
    Clock::time_point begin, end;
  };
  // One in-flight driver packet, with time/call/memory bounded continuation.
  // All history ranges are reserved by the coordinator BEFORE submission.
  static constexpr std::size_t maximum_packet_calls=16;
  static constexpr std::size_t maximum_cpu_primary_packet_calls=4;
  struct Packet {
    std::vector<Completion> results;
    std::uint64_t retained_bytes{};
    unsigned stop_reason{6};
  };
  struct Continuation {
    std::uint64_t first{},stride{},limit{};
    std::array<std::uint64_t,2> lease{};
    std::array<std::size_t,2> capacity{};
    AdaptiveSubshowerControl control;
    double target_ms{};
    std::size_t maximum_calls{1};
    bool fixed_cpu_primary{};
    bool enabled{};
  };
 public:
  IndependentSubshowerPump(Endpoint& gpu, Endpoint& cpu, Statistics& statistics,
                           std::function<void()> select_device,
                           std::function<void(bool)> observe_host = {},
                           std::size_t host_threads = 20,bool adaptive = false,
                           std::uint64_t continuation_retention_bytes = 64U<<20,
                           bool cpu_primary = false)
      : gpu_(gpu), cpu_(cpu), stats_(statistics),
        select_device_(std::move(select_device)), observe_host_(std::move(observe_host)),
        initial_host_share_(cpu_primary?1.-CpuPrimarySubshowerPolicy::initial_gpu_share:
            adaptive?.5:CooperativeHostPolicy::initialShare(host_threads)),
        adaptive_(adaptive),cpu_primary_(cpu_primary),continuation_retention_bytes_(continuation_retention_bytes) {
    if (cpu_primary_ && adaptive_) throw std::invalid_argument("CPU-primary and experimental adaptive policies are mutually exclusive");
    if(!continuation_retention_bytes_)
      throw std::invalid_argument("Zero subshower continuation retention budget");
    control_.setHostInitialBatch(CooperativeHostPolicy::capacity(host_threads));
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
    if(adaptive_ || cpu_primary_) {
      stats_.cuda_completion_mailbox_capacity=cpu_primary_?
          maximum_cpu_primary_packet_calls:maximum_packet_calls;
      stats_.cuda_completion_retention_budget_bytes=continuation_retention_bytes_;
    }
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
      publishHostWork(callbacks);
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
        publishHostWork(callbacks);
      }
      if (gpuReady()) collect(callbacks);
      // collect() may itself return scalar work. Publish that state before
      // launching, so an empty OpenMP queue is not confused with an idle host.
      if(adaptive_ || cpu_primary_)publishHostWork(callbacks);
      if (!flight_.valid() && !peers_[0].empty() && (adaptive_ || cpu_primary_ || !yield(callbacks))) launch(callbacks);
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
  void publishHostWork(SubshowerCallbacks const& callbacks) {
    bool scalar=yield(callbacks);
    gpu_scalar_foreground_.store(scalar,std::memory_order_release);
    // Scalar fallback/stack work occupies the SAME coordinator as OpenMP.
    // Its unrelated particle histories do not require the CUDA queue to wait.
    // Request a checkpoint when the host genuinely needs fresh work instead.
    gpu_needs_handoff_.store((adaptive_ || cpu_primary_)?(peers_[1].empty() && !scalar):
        (peers_[1].empty() || scalar),std::memory_order_release);
  }
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
    if(cpu_primary_) return enqueueCpuPrimary(input, incoming);
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
    if(cpu_primary_ && e==1) {
      auto selected=CpuPrimarySubshowerPolicy::selectHostSpecies(
          p.next_kind, {p.count(0),p.count(1)}, p.capacity,
          cpu_.minimumBatchSize(), p.deferrals);
      if(selected!=k) ++stats_.adaptive_species_coalesces[1];
      k=selected;
    }
    auto target=adaptive_?control_.targetMs(e,k,flight_.valid()):
        AdaptiveSubshowerControl::epoch_target_ms;
    if(adaptive_) {
      auto useful=std::min(gpu_.minimumBatchSize(),p.capacity[k]);
      // Coalesce the small species while processing a useful batch of the
      // other species. No sleeping for an empty future queue; a lone tail
      // drains immediately. Eight deferrals bound species starvation.
      if(p.count(k)<useful && p.count(1-k)>=useful && p.deferrals[k]<8) {
        ++p.deferrals[k];k=1-k;
        ++stats_.adaptive_species_coalesces[e];
      }
      p.deferrals[k]=0;
    }
    if(adaptive_)target=control_.targetMs(e,k,flight_.valid());
    p.next_kind=1-k; // each endpoint alternates its own photon/lepton queues
    Job job;job.kind=k;job.sequence=++submitted_[e];
    auto capacity=adaptive_ && e==1?control_.inputLimit(e,k,p.capacity[k],target):p.capacity[k];
    // Match the standalone router's primary fill order: already resident
    // particles retain their arena slots; new waiting input uses the spare
    // capacity. In particular, do not evict a mature front with fresh input.
    auto protected_resident=cpu_primary_ && e==1?std::min(capacity,p.resident[k]):0;
    auto external=std::min(capacity-protected_resident,p.waiting[k].size());
    job.input.assign(p.waiting[k].begin(),p.waiting[k].begin()+external);
    p.waiting[k].erase(p.waiting[k].begin(),p.waiting[k].begin()+external);
    auto resident=std::min(capacity-external,p.resident[k]);
    job.resident_input=resident;
    job.bounded_input=adaptive_ && e==1;
    job.count=external+resident;
    job.input_limit=capacity;job.target_ms=target;
    p.resident[k]-=resident; // ownership moves to the job, not a second queue entry
    job.waves=adaptive_?control_.waves(e,k,job.count,target):
        (k==0?16:(e==0?gpu_lepton_waves_:8));
    if(cpu_primary_ && k==1) job.waves=e==1?
        CpuPrimarySubshowerPolicy::host_lepton_waves:CpuPrimarySubshowerPolicy::auxiliary_lepton_waves;
    // Split tails must not return after every single wave merely because
    // each half is smaller than the original unsplit 4096-input threshold.
    // The measured epoch controller already supplies a bounded hand-off.
    job.minimum=adaptive_?1:(e==0?gpu_.minimumBatchSize():1);
    if(cpu_primary_) job.minimum=e==0?1:
        CpuPrimarySubshowerPolicy::hostCheckpointMinimum(job.count,cpu_.minimumBatchSize());
    if(adaptive_ && e==0 && job.count>=gpu_.minimumBatchSize())
      // Newly generated other-species particles are not reflected in the
      // coordinator's count until return. Let a big front hand off its small
      // tail so those daughters can form the next useful front. A job which
      // STARTS small still drains with minimum=1, avoiding single-step tails.
      job.minimum=gpu_.minimumBatchSize();
    if(adaptive_ && e==1) {
      // An independent host epoch may now run >8 waves. Do not spend hundreds
      // of 130-thread barriers draining a vanishing front while other-species
      // daughters wait. Rejoin/coalesce at a useful native-sized boundary.
      // A small initial batch uses a proportional threshold, not an immediate
      // one-wave return. This is a lossless scheduler checkpoint, never a cut.
      job.minimum=std::min(gpu_.minimumBatchSize(),
                           std::max<std::size_t>(1,job.count/4));
    }
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
  static std::uint64_t retainedBytes(Completion const& result) {
    // Count allocated capacity, not size. A new call is allowed only while
    // the PREVIOUS results fit the retention budget: peak is bounded by that
    // budget plus one ordinary result, whose existing backend limits apply.
    std::uint64_t bytes=sizeof(Completion);
    auto add=[&](auto const& values) {
      using T=typename std::decay_t<decltype(values)>::value_type;
      if(values.capacity()>std::numeric_limits<std::uint64_t>::max()/sizeof(T))
        throw std::overflow_error("Subshower completion capacity overflow");
      bytes=cooperativeAdd(bytes,static_cast<std::uint64_t>(values.capacity()*sizeof(T)));
    };
    add(result.identity.input);
    std::visit([&](auto const& r) {
      add(r.cpu_spill_particles);add(r.fallback_events);add(r.observations);
      add(r.interaction_records);add(r.step_records);add(r.projected_step_records);
      add(r.final_state_records);
      if constexpr(std::is_same_v<std::decay_t<decltype(r)>,Photon>) {
        add(r.electromagnetic_secondaries);add(r.remaining_photons);
      } else {
        add(r.generated_photons);add(r.remaining_leptons);add(r.decay_candidates);
      }
    },result.result);
    return bytes;
  }
  void launch(SubshowerCallbacks const& callbacks) {
    auto job=prepare(0,callbacks);
    Continuation continuation;
    bool foreground=yield(callbacks);
    continuation.fixed_cpu_primary=cpu_primary_;
    continuation.maximum_calls=cpu_primary_?maximum_cpu_primary_packet_calls:maximum_packet_calls;
    continuation.enabled=(adaptive_ || cpu_primary_) && (!peers_[1].empty() || foreground);
    if(continuation.enabled) {
      continuation.capacity=peers_[0].capacity;
      if(adaptive_) {
        continuation.control=control_; // private snapshot; never shares mutable host estimates
        // The CPU's measured useful epoch sets the service horizon. This is
        // long on a CPU-heavy server, short when CUDA is the faster endpoint.
        continuation.target_ms=std::max(control_.targetMs(1,0,true),control_.targetMs(1,1,true));
      }
      // CPU-primary uses a fixed small packet, not a time/rate controller.
      // Every follow-up lease is reserved here; the driver never touches Stack.
      for(unsigned k=0;k<2;++k) {
        auto n=static_cast<std::uint64_t>(continuation.capacity[k]);
        if(n>std::numeric_limits<std::uint64_t>::max()/192)
          throw std::overflow_error("Subshower continuation history request overflow");
        auto per_call=k==0?32*n:std::max(3*n,std::min<std::uint64_t>(1U<<20,192*n));
        continuation.lease[k]=per_call;
        continuation.stride=std::max(continuation.stride,per_call);
      }
      // One ordered bank, NOT separate photon/lepton banks: repeated species
      // alternation must never allocate children from an earlier ID interval.
      if(continuation.stride>std::numeric_limits<std::uint64_t>::max()/(continuation.maximum_calls-1))
        throw std::overflow_error("Subshower continuation bank overflow");
      auto reserve=continuation.stride*(continuation.maximum_calls-1);
      continuation.first=callbacks.reserve_histories(reserve);
      continuation.limit=cooperativeAdd(continuation.first,reserve);
      if(!continuation.first || continuation.first<last_history_limit_)
        throw std::logic_error("Overlapping/nonmonotonic continuation history lease");
      last_history_limit_=continuation.limit;
    }
    flight_kind_=job.kind;flight_count_=job.count;
    host_epochs_in_flight_=0;
    flight_host_starved_ms_=0.;
    gpu_begin_ns_=0;gpu_end_ns_=0;
    flight_=driver_.submit([this,job=std::move(job),continuation=std::move(continuation)]() mutable {
      select_device_();
      gpu_begin_ns_.store(nanoseconds(Clock::now()),std::memory_order_release);
      Packet packet;packet.results.reserve(continuation.enabled?continuation.maximum_calls:1);
      packet.results.push_back(execute(gpu_,std::move(job)));
      packet.retained_bytes=(adaptive_ || cpu_primary_)?retainedBytes(packet.results.back()):0;
      // No coordinator-owned queue, callback or mutable estimate is touched.
      // The driver may learn from its own completed calls in its PRIVATE copy.
      // If the host needs work, hand over this lossless checkpoint instead of
      // extending a slow GPU tail. The horizon is checked between complete
      // calls, not kernel preemption or a particle cut.
      while(continuation.enabled) {
        auto const& last=packet.results.back();
        if(!last.resident[0] && !last.resident[1]) {packet.stop_reason=0;break;}
        if(!std::visit([](auto const& r){return r.wavefronts!=0;},last.result)) {packet.stop_reason=1;break;}
        if(packet.retained_bytes>=continuation_retention_bytes_) {packet.stop_reason=2;break;}
        if(packet.results.size()==continuation.maximum_calls) {packet.stop_reason=3;break;}
        if(gpu_needs_handoff_.load(std::memory_order_acquire)) {packet.stop_reason=4;break;}
        if(!continuation.fixed_cpu_primary &&
           milliseconds(packet.results.front().begin,last.end)>=continuation.target_ms) {packet.stop_reason=5;break;}
        // Refresh before planning the next autonomous call. Keeping the launch
        // snapshot unchanged can repeat a one/two-wave lease for all 16 calls,
        // although each short completed call demonstrates room for growth.
        // The coordinator independently observes each result exactly once at
        // commit; this private calibration never mutates global statistics.
        if(!continuation.fixed_cpu_primary) {
          std::visit([&](auto const& result) {
            auto const& id=last.identity;
            continuation.control.observe(0,id.kind,id.count,result.wavefronts,id.waves,
                result.transport_records,milliseconds(last.begin,last.end),
                id.input_limit,id.target_ms);
          },last.result);
        }
        unsigned k=1-last.identity.kind;
        if(!last.resident[k])k=1-k;
        auto useful=gpu_.minimumBatchSize();
        if(last.resident[k]<useful && last.resident[1-k]>=useful)k=1-k;
        Job follow;follow.kind=k;follow.sequence=cooperativeAdd(
            packet.results.front().identity.sequence,static_cast<std::uint64_t>(packet.results.size()));
        follow.scalar_foreground_continuation=gpu_scalar_foreground_.load(std::memory_order_acquire);
        follow.count=std::min(last.resident[k],continuation.capacity[k]);
        follow.resident_input=follow.count;follow.input_limit=continuation.capacity[k];
        if(continuation.fixed_cpu_primary) {
          follow.waves=k==0?16:CpuPrimarySubshowerPolicy::auxiliary_lepton_waves;
          follow.minimum=1;
        } else {
          follow.target_ms=continuation.control.targetMs(0,k,true);
          follow.waves=continuation.control.waves(0,k,follow.count,follow.target_ms);
          follow.minimum=follow.count>=useful?useful:1;
        }
        follow.first=cooperativeAdd(continuation.first,
            continuation.stride*static_cast<std::uint64_t>(packet.results.size()-1));
        follow.limit=cooperativeAdd(follow.first,continuation.lease[k]);
        if(follow.limit>continuation.limit)
          throw std::logic_error("Subshower continuation bank exhausted");
        packet.results.push_back(execute(gpu_,std::move(follow)));
        packet.retained_bytes=cooperativeAdd(packet.retained_bytes,retainedBytes(packet.results.back()));
      }
      gpu_end_ns_.store(nanoseconds(packet.results.back().end),std::memory_order_release);
      return packet;
    });
    ++stats_.subshower_cuda_submissions;
    if((adaptive_ || cpu_primary_) && foreground)++stats_.subshower_cuda_foreground_packets;
  }
  void collect(SubshowerCallbacks const& callbacks) {
    if(!gpuReady()) throw std::logic_error("Blocking subshower collect is forbidden");
    auto packet=flight_.get();
    driver_.waitIdle(); // packaged_task teardown, after result is already ready
    auto const packet_limit=cpu_primary_?maximum_cpu_primary_packet_calls:
        (adaptive_?maximum_packet_calls:1);
    if(packet.results.empty() || packet.results.size()>packet_limit)
      throw std::logic_error("Invalid independent completion mailbox size");
    auto received=Clock::now();
    // This is actual GPU idle time after the LAST autonomous call. The first
    // result may wait in the mailbox while the GPU is still doing useful work.
    stats_.cuda_result_service_delay_ms+=milliseconds(packet.results.back().end,received);
    if(adaptive_ || cpu_primary_) {
      stats_.cuda_completion_peak_bytes=std::max(stats_.cuda_completion_peak_bytes,packet.retained_bytes);
      stats_.maximum_cuda_packet_calls=std::max(stats_.maximum_cuda_packet_calls,
          static_cast<std::uint64_t>(packet.results.size()));
      ++stats_.cuda_continuation_stops.at(packet.stop_reason);
    }
    for(std::size_t i=1;i<packet.results.size();++i) {
      submitted_[0]=cooperativeAdd(submitted_[0],std::uint64_t{1});
      stats_.subshower_cuda_submissions=cooperativeAdd(stats_.subshower_cuda_submissions,std::uint64_t{1});
      stats_.subshower_cuda_autonomous_continuations=cooperativeAdd(
          stats_.subshower_cuda_autonomous_continuations,std::uint64_t{1});
      if(packet.results[i].identity.scalar_foreground_continuation)
        ++stats_.subshower_cuda_foreground_continuations;
      stats_.cuda_input_particles=cooperativeAdd(stats_.cuda_input_particles,
          static_cast<std::uint64_t>(packet.results[i].identity.count));
    }
    for(auto& result:packet.results) {
    stats_.cuda_driver_wall_ms+=milliseconds(result.begin,result.end);
    if(adaptive_ || cpu_primary_)stats_.cuda_completion_buffer_delay_ms+=milliseconds(result.end,received);
    auto duration=milliseconds(result.begin,result.end);
    if(adaptive_)control_.observeGpuDuration(duration);
    stats_.subshower_maximum_cuda_epoch_ms=std::max(
        stats_.subshower_maximum_cuda_epoch_ms,duration);
    // A slow GPU must reach a safe redistribution boundary when the CPU is
    // starved. Never interrupt a kernel or alter a particle's physical step.
    if(!adaptive_ && !cpu_primary_ && result.identity.kind==1) {
      if(flight_host_starved_ms_>5. && duration>100.) {
        auto target=static_cast<std::size_t>(result.identity.waves*100./duration);
        auto next=std::max<std::size_t>(16,std::min(gpu_lepton_waves_/2,target));
        if(next<gpu_lepton_waves_) ++stats_.subshower_cuda_epoch_reductions;
        gpu_lepton_waves_=next;
      } else if(flight_host_starved_ms_<1. && duration<50.) {
        gpu_lepton_waves_=std::min<std::size_t>(1024,2*gpu_lepton_waves_);
      }
    }
    stats_.subshower_gpu_lepton_wave_limit=cpu_primary_?
        CpuPrimarySubshowerPolicy::auxiliary_lepton_waves:gpu_lepton_waves_;
    ++stats_.subshower_cuda_commits;
    commit(0,std::move(result),callbacks);
    }
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
      if(cpu_primary_) {
        stats_.adaptive_transport_records[e][job.kind]=cooperativeAdd(
            stats_.adaptive_transport_records[e][job.kind],result.transport_records);
        stats_.adaptive_resident_wavefronts[e][job.kind]=cooperativeAdd(
            stats_.adaptive_resident_wavefronts[e][job.kind],static_cast<std::uint64_t>(result.wavefronts));
        auto count=job.count;std::size_t bin=0;
        while(count>1 && bin+1<stats_.adaptive_job_input_histogram[e].size()) {count>>=1;++bin;}
        ++stats_.adaptive_job_input_histogram[e][bin];
      }
      if(adaptive_) {
        control_.observe(e,job.kind,job.count,result.wavefronts,job.waves,
                         result.transport_records,elapsed,job.input_limit,job.target_ms);
        auto const& s=control_.sample(e,job.kind);
        stats_.adaptive_records_per_ms[e][job.kind]=s.records_per_ms;
        stats_.adaptive_observations[e][job.kind]=s.observations;
        stats_.adaptive_transport_records[e][job.kind]=cooperativeAdd(
            stats_.adaptive_transport_records[e][job.kind],result.transport_records);
        stats_.adaptive_resident_wavefronts[e][job.kind]=cooperativeAdd(
            stats_.adaptive_resident_wavefronts[e][job.kind],static_cast<std::uint64_t>(result.wavefronts));
        stats_.adaptive_input_limit[e][job.kind]=job.input_limit;
        stats_.adaptive_full_observations[e][job.kind]=s.full_observations;
        // One exclusive reason for every completed job. No unbounded trace.
        unsigned reason=5; // other: diagnostics must expose unexpected exits
        if(result.workspace_limit_checkpoint)reason=2;
        else if(result.below_minimum_batch_checkpoint)reason=1;
        else if(result.completed)reason=0;
        else if(result.wavefronts>=job.waves)reason=4;
        if constexpr(std::is_same_v<std::decay_t<decltype(result)>,Lepton>)
          if(result.history_range_exhausted)reason=3;
        ++stats_.adaptive_completion_reasons[e][job.kind][reason];
        auto value=job.count;std::size_t bin=0;
        while(value>1 && bin+1<stats_.adaptive_job_input_histogram[e].size()) {value>>=1;++bin;}
        ++stats_.adaptive_job_input_histogram[e][bin];
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
    if(cpu_primary_) {rebalanceCpuPrimary();return;}
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
  std::size_t enqueueCpuPrimary(std::vector<Particle> const& input,
                                std::array<std::size_t,2> incoming) {
    std::array<std::size_t,2> quota{};
    for(unsigned k=0;k<2;++k) {
      auto total=incoming[k]+peers_[1].count(k);
      auto keep=CpuPrimarySubshowerPolicy::hostReserve(peers_[1].capacity[k],cpu_.minimumBatchSize());
      auto capacity=peers_[0].capacity[k], already=outstanding(0,k);
      auto useful=CpuPrimarySubshowerPolicy::helperMinimum(capacity,gpu_.minimumBatchSize());
      if(total>keep && already<capacity && incoming[k]>=useful && total-keep>=useful)
        quota[k]=std::min({incoming[k],total-keep,2*capacity-already,
          std::max(useful,static_cast<std::size_t>(incoming[k]*
              CpuPrimarySubshowerPolicy::gpuShare(rates_[k][1],rates_[k][0])))});
    }
    std::size_t accepted=0;
    for(auto const& particle:input) {
      unsigned k=particle.pid==22?0:1;
      unsigned e=quota[k]?0:1;
      auto room=[&](unsigned side){return peers_[side].waiting[k].size()<2*peers_[side].capacity[k];};
      if(!room(e)) {
        // A full coordinator queue is backpressure, not permission to seed a
        // tiny helper job outside the quota above. The router retains the
        // unaccepted suffix; the CPU still executes its queued epoch below.
        if(e==1 || !room(1))break;
        e=1;
      }
      peers_[e].waiting[k].push_back(particle);
      if(e==0 && quota[k])--quota[k];
      ++accepted;
    }
    return accepted;
  }
  void moveCpuPrimaryWork(unsigned source,unsigned k,std::size_t count) {
    auto& donor=peers_[source];auto& receiver=peers_[1-source];
    auto n=std::min(count,donor.waiting[k].size());
    auto& from=donor.waiting[k];auto& to=receiver.waiting[k];
    to.insert(to.end(),std::make_move_iterator(from.end()-n),std::make_move_iterator(from.end()));
    from.erase(from.end()-n,from.end());
    if(count>n) {
      if(source==0 && flight_.valid())throw std::logic_error("CPU-primary attempted in-flight device migration");
      auto moved=endpoint(source).takeCooperativePending(k==0,count-n);
      if(moved.size()!=count-n)throw std::logic_error("CPU-primary migration count mismatch");
      to.insert(to.end(),std::make_move_iterator(moved.begin()),std::make_move_iterator(moved.end()));
      donor.resident[k]-=count-n;
    }
    stats_.migration_bytes=cooperativeAdd(stats_.migration_bytes,count*sizeof(Particle));
    ++stats_.subshower_tail_migrations;
    if(flight_.valid()) {
      ++stats_.subshower_inflight_waiting_migrations;
      stats_.subshower_inflight_waiting_particles+=count;
    }
  }
  void rebalanceCpuPrimary() {
    for(unsigned k=0;k<2;++k) {
      auto& host=peers_[1];auto& device=peers_[0];
      // Do not undo the just-computed throughput split merely because the
      // host batch is not full. Steal only when the primary has no work.
      if(host.empty()) {
        auto available=flight_.valid()?device.waiting[k].size():device.count(k);
        auto n=std::min(available,host.capacity[k]-host.count(k));
        if(n)moveCpuPrimaryWork(0,k,n);
      }
      // An empty helper takes only a measured share of primary surplus.
      // No predicted completion times or repeated balancing while both run.
      if(flight_.valid() || !device.empty())continue;
      auto keep=CpuPrimarySubshowerPolicy::hostReserve(host.capacity[k],cpu_.minimumBatchSize());
      auto useful=CpuPrimarySubshowerPolicy::helperMinimum(device.capacity[k],gpu_.minimumBatchSize());
      if(host.count(k)<=keep || host.count(k)-keep<useful)continue;
      auto surplus=host.count(k)-keep;
      auto n=std::min({device.capacity[k],surplus,std::max(useful,
          static_cast<std::size_t>(surplus*CpuPrimarySubshowerPolicy::gpuShare(rates_[k][1],rates_[k][0])))});
      if(n>=useful)moveCpuPrimaryWork(1,k,n);
    }
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
      // Transferring a handful of particles changes ownership but cannot
      // amortize a new launch/scan/radio batch. Do not oscillate tiny tails.
      if(count<256)continue;
      double source_load=source==0?a:b,destination_load=destination==0?a:b;
      double before=std::max(source_load,destination_load);
      double after=std::max(source_load-count*control_.unitCost(source,k),
                           destination_load+count*control_.unitCost(destination,k));
      double benefit=before-after;
      // Compare the predicted critical path, not only donor work removed.
      // Hysteresis prevents oscillatory migrations between two nonempty peers.
      if(benefit<=1.+2.*count*migration_ms_per_particle_+
          (receiver.empty()?0.:.25*before))continue;
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
  bool advancing_{},failed_{},adaptive_{},cpu_primary_{};
  std::uint64_t continuation_retention_bytes_;
  AdaptiveSubshowerControl control_;
  double migration_ms_per_particle_{};
  std::atomic<std::int64_t> gpu_begin_ns_{0},gpu_end_ns_{0};
  std::atomic<bool> gpu_needs_handoff_{true};
  std::atomic<bool> gpu_scalar_foreground_{false};
  std::future<Packet> flight_;
  IndependentEndpointDriver driver_; // drain before queues, callbacks and Views
};
} // namespace corsika::accelerator::em::detail
