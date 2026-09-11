/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#include <corsika/accelerator/em/detail/KokkosBackendInstance.hpp>
#include <corsika/accelerator/em/detail/CooperativeBackendMerge.hpp>
#include <corsika/accelerator/em/detail/IndependentEndpointDriver.hpp>
#include <corsika/accelerator/em/detail/IndependentSubshowerPump.hpp>
#include <corsika/accelerator/em/detail/CooperativeDriverAffinity.hpp>
#include <cuda_runtime_api.h>
#include <array>
#include <chrono>
#include <exception>
#include <optional>

namespace corsika::accelerator::em::detail {
namespace {
using namespace gpu::em;
using Clock = std::chrono::steady_clock;

// Air-shower host coordinator. Neither endpoint sees Stack/SecondaryView,
// writers or scalar physics models. Only the CUDA endpoint's wavefront call
// runs on a dedicated driver; OpenMP stays on its initialization thread.
class CooperativeBackend final : public KokkosBackendInstance {
 public:
  explicit CooperativeBackend(KokkosRuntimeConfig config)
      : owner_(ownerConfig(config)), slice_observer_(config.cooperative_slice_observer),
        adaptive_(config.cooperative_policy=="adaptive") {
    config.cooperative_owner = false;
    config.runtime_lease = owner_.shareCooperativeLifetime();
    config.execution_backend = "cuda";
    gpu_ = makeCudaBackendInstance(config);
    config.execution_backend = "openmp";
    cpu_ = makeOpenMPBackendInstance(config);
    info_ = gpu_->runtimeInfo();
    info_.backend = "cuda-openmp";
    info_.openmp = true;
  }
  ~CooperativeBackend() override { subshowers_.reset(); driver_.waitIdle(); }
  void initialize(EnvironmentSnapshot const& environment,
                  tables::ProposalNativeTableSet const& table,
                  tables::ProposalNativeAuxData const& auxiliary,
                  GpuEmConfig const& config) override {
    requireIdle();
    if(subshowers_) subshowers_->requireIdle();
    if (!config.deterministic || (config.radio.enabled && !config.radio.deterministic))
      throw std::invalid_argument("Cooperative transport requires deterministic accumulators");
    if (config.profile_projection.enabled && !config.profile_projection.accumulate_on_device)
      throw std::invalid_argument("Cooperative profile requires endpoint integer accumulation");
    // Construct the small CPU workspace first, before any CUDA work is in flight.
    auto host = config;
    if (!adaptive_) {
    host.min_batch_size = 256;
    host.resident_batch_limit = CooperativeHostPolicy::capacity(
        static_cast<std::size_t>(info_.host_threads));
    // Independent bounded host queues, not one arena per OpenMP thread.
    // This is the queue budget; tables, radio and result storage are additional.
    host.memory_fraction = std::min(config.memory_fraction, 0.70 / 16.);
    }
    // Adaptive mode inherits the standalone OpenMP arena and queue budgets,
    // including any explicit user safety cap. No per-thread duplication.
    cpu_->initialize(environment, table, auxiliary, host);
    cpu_->prepareCooperativeWorkspace();
    gpu_->initialize(environment, table, auxiliary, config);
    // Allocation/preparation is serialized before either driver starts.
    // Otherwise a cold CUDA arena can delay its first kernels until after
    // the entire first OpenMP batch has already finished.
    gpu_->prepareCooperativeWorkspace();
    resetEvent();
  }
  void beginShower(AcceleratedEmShowerConfig const& config) override {
    requireIdle();
    if (pendingPhotonCount() || pendingLeptonCount())
      throw std::logic_error("Cooperative beginShower with unconsumed particles");
    cpu_->beginShower(config);
    gpu_->beginShower(config);
    resetEvent();
  }
  bool canTransport(EmParticleState const& p) const override {
    return gpu_->canTransport(p) && cpu_->canTransport(p);
  }
  bool hasProposalTable() const override { return gpu_->hasProposalTable() && cpu_->hasProposalTable(); }
  std::size_t minimumBatchSize() const override { return gpu_->minimumBatchSize(); }
  std::size_t maximumResidentPhotonBatchSize() const override {
    return gpu_->maximumResidentPhotonBatchSize()+cpu_->maximumResidentPhotonBatchSize();
  }
  std::size_t maximumResidentLeptonBatchSize() const override {
    return gpu_->maximumResidentLeptonBatchSize()+cpu_->maximumResidentLeptonBatchSize();
  }
  std::size_t maximumResidentInputBatchSize() const override {
    return gpu_->maximumResidentInputBatchSize(); // never reduce the GPU arena
  }
  std::size_t pendingPhotonCount() const noexcept override {
    if(subshowers_) return subshowers_->pending(0);
    return eligible(*gpu_, true)+eligible(*cpu_, true);
  }
  std::size_t pendingLeptonCount() const noexcept override {
    if(subshowers_) return subshowers_->pending(1);
    return eligible(*gpu_, false)+eligible(*cpu_, false);
  }

  bool independentSubshowersEnabled() const noexcept override {return true;}
  bool independentSubshowersReady() const noexcept override {
    return subshowers_ && subshowers_->ready();
  }
  std::size_t advanceIndependentSubshowers(std::vector<EmParticleState> const& input,
                                          SubshowerCallbacks const& callbacks) override {
    CallScope scope(*this);
    if(!subshowers_) {
      subshowers_=std::make_unique<IndependentSubshowerPump<KokkosBackendInstance>>(
          *gpu_,*cpu_,cooperative_,[device=info_.device,
                                  affinity=driver_affinity_,restored=false]() mutable {
            if(!restored) {affinity();restored=true;}
            if(cudaSetDevice(device)!=cudaSuccess)
              throw std::runtime_error("Independent CUDA driver could not select its device");
          },slice_observer_,static_cast<std::size_t>(info_.host_threads),adaptive_);
    }
    return subshowers_->advance(input,callbacks);
  }

  ResidentPhotonCascadeResult runPhotonWavefront(
      std::vector<EmParticleState> const& input, std::uint64_t first,
      std::size_t waves, std::size_t minimum) override {
    CallScope scope(*this);
    if(subshowers_) throw std::logic_error("Cannot mix synchronous and independent subshower calls");
    auto split = partition(input, true);
    if (!waves || waves > std::numeric_limits<std::uint64_t>::max()/2/
                              std::max<std::size_t>(1,split.total()))
      throw std::overflow_error("Cooperative photon history reservation overflow");
    auto host_range = 2*split.cpu_count*waves;
    auto limit = cooperativeAdd(first,2*split.total()*waves);
    auto gpu_first = cooperativeAdd(first,host_range);
    cpu_->setCooperativePendingInputLimit(split.cpu_pending);
    auto results = runIndependent<ResidentPhotonCascadeResult>(split, true,
        [&] { return gpu_->runPhotonWavefront(split.gpu,gpu_first,waves,minimum); },
        [&] { return cpu_->runPhotonWavefront(split.cpu,first,std::min(waves,HostEpochWaves),1); });
    auto device=std::move(results.first);
    mergeCooperative(device,std::move(results.second));
    if (device.input_particles!=split.total() || gpu_first>limit)
      throw std::logic_error("Cooperative photon commit lost/duplicated an input");
    ++cooperative_.photon_calls;
    return device;
  }

  ResidentLeptonCascadeResult runLeptonWavefront(
      std::vector<EmParticleState> const& input, std::uint64_t first,
      std::size_t waves, std::uint64_t limit, std::size_t minimum) override {
    CallScope scope(*this);
    if(subshowers_) throw std::logic_error("Cannot mix synchronous and independent subshower calls");
    auto split=partition(input,false);
    if (!waves || limit<first || split.total()>(limit-first)/3)
      throw std::overflow_error("Cooperative lepton history reservation too small");
    auto available=limit-first;
    // Disjoint checked ID ranges; exhaustion returns remaining particles to
    // the coordinator. It never reallocates IDs from the other in-flight end.
    auto host_range=split.cpu_count?std::min<std::uint64_t>(
        192*split.cpu_count,available-3*split.gpu_count):0;
    auto gpu_first=cooperativeAdd(first,host_range);
    cpu_->setCooperativePendingInputLimit(split.cpu_pending);
    auto results = runIndependent<ResidentLeptonCascadeResult>(split, false,
        [&] { return gpu_->runLeptonWavefront(split.gpu,gpu_first,waves,limit,minimum); },
        [&] { return cpu_->runLeptonWavefront(split.cpu,first,std::min(waves,HostEpochWaves),gpu_first,1); });
    auto device=std::move(results.first);
    mergeCooperative(device,std::move(results.second));
    if (device.input_particles!=split.total() || device.secondary_history_ids_used>available)
      throw std::logic_error("Cooperative lepton commit lost/duplicated an input");
    ++cooperative_.lepton_calls;
    return device;
  }

  BackendCapabilities capabilities() const override { return gpu_->capabilities(); }
  AcceleratedEmStatistics const& statistics() const override {
    if(subshowers_) {
      subshowers_->requireHealthy();
      if(subshowers_->inFlight()) throw std::logic_error("Cannot snapshot live endpoint statistics");
    }
    stats_=mergeCooperativeStatistics(gpu_->statistics(),cpu_->statistics());
    stats_.accelerator_backend="cuda-openmp";
    stats_.accelerator_openmp=true;
    stats_.cooperative=cooperative_;
    stats_.cooperative.openmp_workspace_bytes=cpu_->statistics().physical_workspace_bytes;
    return stats_;
  }
  bool gpuProfileEnabled() const noexcept override { return gpu_->gpuProfileEnabled(); }
  GpuProfileResult downloadProfile() override {
    requireIdle();
    if(subshowers_) subshowers_->requireIdle();
    if(!profile_) {
      auto a=gpu_->downloadFixedProfile(identity_,CooperativeEndpoint::Cuda);
      auto b=cpu_->downloadFixedProfile(identity_,CooperativeEndpoint::OpenMP);
      CooperativeProfileMerge merged(identity_,a.config);
      merged.commit(a);merged.commit(b);
      profile_=decodeFixedProfile(merged.take());
    }
    return *profile_;
  }
  bool gpuRadioEnabled() const noexcept override { return gpu_->gpuRadioEnabled(); }
  gpu::radio::GpuRadioWaveforms downloadRadioWaveforms() override {
    requireIdle();
    if(subshowers_) subshowers_->requireIdle();
    if(!radio_) {
      auto a=gpu_->downloadFixedRadio(identity_,CooperativeEndpoint::Cuda);
      auto b=cpu_->downloadFixedRadio(identity_,CooperativeEndpoint::OpenMP);
      radio::detail::CooperativeRadioMerge merged(identity_,a.config);
      merged.commit(a);merged.commit(b);
      radio_=radio::detail::decodeFixedRadio(merged.take());
    }
    return *radio_;
  }
  std::optional<GpuFirstInteractionSnapshot> downloadFirstInteractionSnapshot() override {
    if(subshowers_) subshowers_->requireIdle();
    auto a=gpu_->downloadFirstInteractionSnapshot(),b=cpu_->downloadFirstInteractionSnapshot();
    if(a && b) throw std::logic_error("Cooperative first interaction has two owners");
    return a?a:b;
  }
  KokkosRuntimeInfo const& runtimeInfo() const noexcept override { return info_; }
  // Oracle/stage APIs explicitly use one endpoint, never pretend to test both.
  EmInteractionBatchResult selectInteractionsForValidation(std::vector<EmParticleState> const& p) override {return gpu_->selectInteractionsForValidation(p);}
  PhotonTransportBatchResult transportPhotonsForValidation(std::vector<EmInteractionRecord> const& p) override {return gpu_->transportPhotonsForValidation(p);}
  LeptonTransportBatchResult transportLeptonsForValidation(std::vector<EmInteractionRecord> const& p) override {return gpu_->transportLeptonsForValidation(p);}
  LeptonVertexSelectionBatchResult selectLeptonVerticesForValidation(std::vector<EmInteractionRecord> const& p) override {return gpu_->selectLeptonVerticesForValidation(p);}
  EmFinalStateBatchResult generatePhotonFinalStatesForValidation(std::vector<EmInteractionRecord> const& p,std::uint64_t h) override {return gpu_->generatePhotonFinalStatesForValidation(p,h);}
  BremsFinalStateBatchResult generateLeptonFinalStatesForValidation(std::vector<EmInteractionRecord> const& p,std::uint64_t h) override {return gpu_->generateLeptonFinalStatesForValidation(p,h);}
  gpu::radio::GpuRadioWaveforms projectRadioForValidation(std::vector<LeptonTransportRecord> const&) override {throw std::logic_error("Use endpoint radio oracle, not cooperative accumulator");}
  void setCooperativeProgress(std::function<bool()>) override {throw std::logic_error("Nested cooperative coordinator is forbidden");}
  void prepareCooperativeWorkspace() override {throw std::logic_error("Cooperative workspace is already prepared");}
  void setCooperativePendingInputLimit(std::size_t) override {throw std::logic_error("Nested cooperative coordinator is forbidden");}
  std::vector<EmParticleState> takeCooperativePending(bool,std::size_t) override {throw std::logic_error("Nested cooperative migration is forbidden");}
  FixedProfileSnapshot downloadFixedProfile(std::string const&,CooperativeEndpoint) override {throw std::logic_error("Cooperative output already contains both endpoints");}
  radio::detail::FixedRadioSnapshot downloadFixedRadio(std::string const&,CooperativeEndpoint) override {throw std::logic_error("Cooperative output already contains both endpoints");}
 private:
  // A scheduling checkpoint, not a particle cut or a GPU-wait time slice.
  // Bound the longest host epoch so a growing CPU cascade can be rebalanced.
  static constexpr std::size_t HostEpochWaves=8;
  struct SliceObservation {
    std::function<void(bool)> const& observer;
    bool active;
    SliceObservation(std::function<void(bool)> const& f,bool pending)
        :observer(f),active(pending && bool(f)) {if(active)observer(true);}
    ~SliceObservation(){if(active)observer(false);}
  };
  static KokkosRuntimeConfig ownerConfig(KokkosRuntimeConfig c) {
    if(c.cooperative_owner || c.runtime_lease)
      throw std::invalid_argument("Cooperative backend must own its process runtime");
    c.execution_backend="cuda";c.cooperative_owner=true;return c;
  }
  static std::size_t eligible(KokkosBackendInstance const& b,bool photons) noexcept {
    return std::min(photons?b.pendingPhotonCount():b.pendingLeptonCount(),
                   photons?b.maximumResidentPhotonBatchSize():b.maximumResidentLeptonBatchSize());
  }
  struct Split {
    std::vector<EmParticleState> gpu,cpu;
    std::size_t gpu_count{},cpu_count{},cpu_pending{};
    std::size_t total() const {return gpu_count+cpu_count;}
  };
  // Bounded independent calls. Join before transferring result ownership or
  // unwinding inputs, Views and runtime leases. No shared Stack/writer access.
  template<class Result,class DeviceCall,class HostCall>
  std::pair<Result,Result> runIndependent(Split const& split,bool photons,
                                        DeviceCall deviceCall,HostCall hostCall) {
    struct Timed { Result result; Clock::time_point begin,end; };
    auto epoch=Clock::now();
    std::future<Timed> device;
    if(split.gpu_count) {
      device=driver_.submit([&,deviceCall] {
        if(cudaSetDevice(info_.device)!=cudaSuccess)
          throw std::runtime_error("Independent CUDA driver could not select its device");
        auto begin=Clock::now();
        auto result=deviceCall();
        return Timed{std::move(result),begin,Clock::now()};
      });
    }
    Result host; host.completed=true;
    auto hostBegin=Clock::now(),hostEnd=hostBegin;
    try {
      if(split.cpu_count) {
        SliceObservation observation(slice_observer_,true);
        hostBegin=Clock::now();
        host=hostCall(); // independent of CUDA wait callbacks
        hostEnd=Clock::now();
      }
    } catch(...) {
      if(device.valid()) device.wait();
      driver_.waitIdle();
      throw;
    }
    Timed gpu; gpu.result.completed=true;gpu.begin=gpu.end=epoch;
    try { if(device.valid()) gpu=device.get(); }
    catch(...) { driver_.waitIdle(); throw; }
    driver_.waitIdle();
    auto ms=[](auto a,auto b) {
      return std::chrono::duration<double,std::milli>(b-a).count();
    };
    double cpuMs=ms(hostBegin,hostEnd),gpuMs=ms(gpu.begin,gpu.end);
    auto& rates=throughput_[photons?0:1];
    if(split.cpu_count) {
      ++cooperative_.openmp_slices;
      cooperative_.openmp_wall_ms+=cpuMs;
      cooperative_.maximum_slice_ms=std::max(cooperative_.maximum_slice_ms,cpuMs);
      double rate=split.cpu_count/std::max(cpuMs,1.e-6);
      rates[1]=rates[1]==0?rate:.8*rates[1]+.2*rate;
    }
    if(split.gpu_count) {
      cooperative_.cuda_driver_wall_ms+=gpuMs;
      double rate=split.gpu_count/std::max(gpuMs,1.e-6);
      rates[0]=rates[0]==0?rate:.8*rates[0]+.2*rate;
    }
    if(split.gpu_count && split.cpu_count) {
      ++cooperative_.independent_joint_calls;
      cooperative_.endpoint_window_overlap_ms+=std::max(0.,
          ms(std::max(gpu.begin,hostBegin),std::min(gpu.end,hostEnd)));
      cooperative_.cuda_finished_before_host_ms+=std::max(0.,ms(gpu.end,hostEnd));
      cooperative_.host_finished_before_cuda_ms+=std::max(0.,ms(hostEnd,gpu.end));
    }
    cooperative_.joint_wall_ms+=ms(epoch,Clock::now());
    return {std::move(gpu.result),std::move(host)};
  }
  Split partition(std::vector<EmParticleState> const& input,bool photons) {
    auto gp=eligible(*gpu_,photons),cp=eligible(*cpu_,photons);
    auto gc=photons?gpu_->maximumResidentPhotonBatchSize():gpu_->maximumResidentLeptonBatchSize();
    auto cc=photons?cpu_->maximumResidentPhotonBatchSize():cpu_->maximumResidentLeptonBatchSize();
    if(input.size()>gc-gp+cc-cp)
      throw std::length_error("Cooperative input exceeds endpoint capacity");
    auto preserved=std::min(minimumBatchSize(),gc);
    auto const& rates=throughput_[photons?0:1];
    // Match predicted completion times including existing resident backlog.
    // The host is an execution peer, not limited to GPU wait opportunities.
    double cpuTarget=256.;
    if(rates[0]>0 && rates[1]>0)
      cpuTarget=(input.size()+gp+cp)*rates[1]/(rates[0]+rates[1]);
    auto target=static_cast<std::size_t>(std::clamp(cpuTarget,0.,double(cc)));
    cooperative_.openmp_batch_target=target;
    auto desired=std::min(target>cp?target-cp:0,cc-cp);
    // Do not bootstrap all small primary EM batches on the much slower
    // host before any throughput estimate exists. Preserve useful CUDA
    // fronts; peer execution does not imply equal particle counts.
    auto share=input.size()+gp>preserved?
        std::min({desired,input.size(),input.size()+gp-preserved}):std::size_t{0};
    share=std::max(share,input.size()>gc-gp?input.size()-(gc-gp):0);
    Split s;
    s.gpu.assign(input.begin(),input.end()-share);
    s.cpu.assign(input.end()-share,input.end());
    // Idle CPU may borrow a bounded stable prefix only at a safe boundary.
    // Preserve >= min-batch on CUDA and never touch in-flight work.
    auto actual=photons?gpu_->pendingPhotonCount():gpu_->pendingLeptonCount();
    if(cp==0 && share==0 && actual==gp && gp>preserved) {
      auto count=std::min(desired,gp-preserved);
      s.cpu=gpu_->takeCooperativePending(photons,count);
      gp-=count;
      cooperative_.migration_bytes=cooperativeAdd(cooperative_.migration_bytes,
          static_cast<std::uint64_t>(count*sizeof(EmParticleState)));
    }
    // Symmetric safe-boundary migration: an idle CUDA end may take CPU
    // cross-species backlog. Do not refill a truncated eligible prefix in
    // this call: that would consume more histories than the caller reserved.
    auto cpuActual=photons?cpu_->pendingPhotonCount():cpu_->pendingLeptonCount();
    if(gp+s.gpu.size()==0 && cp && cpuActual==cp) {
      double fraction=rates[0]>0 && rates[1]>0?
          rates[0]/(rates[0]+rates[1]):.75;
      auto count=std::min({gc,cp,std::max<std::size_t>(1,cp*fraction)});
      auto moved=cpu_->takeCooperativePending(photons,count);
      s.gpu.insert(s.gpu.end(),std::make_move_iterator(moved.begin()),
                    std::make_move_iterator(moved.end()));
      cp-=count;
      cooperative_.migration_bytes=cooperativeAdd(cooperative_.migration_bytes,
          static_cast<std::uint64_t>(count*sizeof(EmParticleState)));
    }
    s.cpu_pending=cp;s.cpu_count=cp+s.cpu.size();s.gpu_count=gp+s.gpu.size();
    cooperative_.cuda_input_particles+=s.gpu_count;
    cooperative_.openmp_input_particles+=s.cpu_count;
    return s;
  }
  void requireIdle() const {
    if(failed_) throw std::logic_error("Cooperative shower failed; outputs cannot be completed");
    if(active_)throw std::logic_error("Cooperative call is already active");
  }
  struct CallScope {
    CooperativeBackend& b;
    int exceptions{std::uncaught_exceptions()};
    explicit CallScope(CooperativeBackend& backend):b(backend) {
      b.requireIdle();
      if(b.profile_ || b.radio_) throw std::logic_error("Cooperative event outputs are sealed");
      b.active_=true;
    }
    ~CallScope(){
      b.cpu_->setCooperativePendingInputLimit(std::numeric_limits<std::size_t>::max());
      b.active_=false;
      if(std::uncaught_exceptions()>exceptions) b.failed_=true;
    }
  };
  void resetEvent() {
    if(subshowers_) subshowers_->requireIdle();
    subshowers_.reset();
    cooperative_={};cooperative_.enabled=true;
    cooperative_.adaptive_policy=adaptive_;
    cooperative_.independent_drivers=true;
    profile_.reset();radio_.reset();throughput_={};
    identity_="cooperative-shower-"+std::to_string(gpu_->statistics().shower_ordinal);
  }
  // Runtime outlives all Views, accumulators, stream tickets and borrowed leases.
  std::function<void()> driver_affinity_=captureCooperativeDriverAffinity();
  KokkosRuntime owner_;
  std::unique_ptr<KokkosBackendInstance> gpu_,cpu_;
  KokkosRuntimeInfo info_;
  std::function<void(bool)> slice_observer_;
  mutable AcceleratedEmStatistics stats_;
  CooperativeEmStatistics cooperative_;
  std::optional<GpuProfileResult> profile_;
  std::optional<gpu::radio::GpuRadioWaveforms> radio_;
  std::string identity_;
  bool active_{},failed_{},adaptive_{};
  std::array<std::array<double,2>,2> throughput_{};
  std::unique_ptr<IndependentSubshowerPump<KokkosBackendInstance>> subshowers_;
  IndependentEndpointDriver driver_; // destroyed/drained before endpoints
};
} // namespace
std::unique_ptr<KokkosBackendInstance> makeCooperativeBackendInstance(KokkosRuntimeConfig const& c) {
  return std::make_unique<CooperativeBackend>(c);
}
} // namespace corsika::accelerator::em::detail
