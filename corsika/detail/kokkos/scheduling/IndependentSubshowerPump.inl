/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/scheduling/CooperativeBackendMerge.hpp>
#include <corsika/kokkos/scheduling/IndependentEndpointDriver.hpp>
#include <corsika/kokkos/scheduling/CooperativeHostPolicy.hpp>
#include <corsika/kokkos/scheduling/CpuPrimarySubshowerPolicy.hpp>
#include <corsika/kokkos/scheduling/AdaptiveSubshowerControl.hpp>
#include <corsika/kokkos/scheduling/SubshowerCallbacks.hpp>
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

namespace corsika::kokkos {

  // Endpoint is deliberately a template: the host scheduling/ownership contract
  // is tested with controlled endpoints without CUDA, physics or timing guesses.
  // Only the CUDA endpoint runs on driver_. CPU, leases and consumers stay here.
  template <class TEndpoint>
  class IndependentSubshowerPump {
    using particle_type = ParticleState;
    using photon_result_type = PhotonCascadeResult;
    using lepton_result_type = LeptonCascadeResult;
    using clock_type = std::chrono::steady_clock;
    using statistics_type = CooperativeStatistics;
    struct Peer {
      std::array<std::vector<particle_type>, 2> waiting_;
      std::array<std::size_t, 2> resident_{}, capacity_{};
      unsigned nextKind_{};
      std::array<unsigned, 2> deferrals_{};
      std::size_t count(unsigned kind) const {
        return resident_[kind] + waiting_[kind].size();
      }
      bool empty() const { return !count(0) && !count(1); }
    };
    struct Job {
      unsigned kind_{};
      std::vector<particle_type> input_;
      std::size_t count_{}, waves_{}, minimum_{};
      std::size_t residentInput_{};
      bool boundedInput_{};
      bool scalarForegroundContinuation_{};
      std::size_t inputLimit_{};
      double targetMs_{};
      std::uint64_t first_{}, limit_{}, sequence_{};
    };
    struct Completion {
      std::variant<photon_result_type, lepton_result_type> result_;
      std::array<std::size_t, 2> resident_;
      Job identity_;
      clock_type::time_point begin_, end_;
    };
    // One in-flight driver packet, with time/call/memory bounded continuation.
    // All history ranges are reserved by the coordinator BEFORE submission.
    static constexpr std::size_t MaximumPacketCalls = 16;
    static constexpr std::size_t MaximumCpuPrimaryPacketCalls = 4;
    struct Packet {
      std::vector<Completion> results_;
      std::uint64_t retainedBytes_{};
      unsigned stopReason_{6};
    };
    struct Continuation {
      std::uint64_t first_{}, stride_{}, limit_{};
      std::array<std::uint64_t, 2> lease_{};
      std::array<std::size_t, 2> capacity_{};
      AdaptiveSubshowerControl control_;
      double targetMs_{};
      std::size_t maximumCalls_{1};
      bool fixedCpuPrimary_{};
      bool enabled_{};
    };

  public:
    IndependentSubshowerPump(TEndpoint& gpu, TEndpoint& cpu, statistics_type& statistics,
                             std::function<void()> selectDevice,
                             std::function<void(bool)> observeHost = {},
                             std::size_t hostThreads = 20, bool adaptive = false,
                             std::uint64_t continuationRetentionBytes = 64U << 20,
                             bool cpuPrimary = false)
        : gpu_(gpu)
        , cpu_(cpu)
        , stats_(statistics)
        , selectDevice_(std::move(selectDevice))
        , observeHost_(std::move(observeHost))
        , initialHostShare_(
              cpuPrimary
                  ? 1. - CpuPrimarySubshowerPolicy::InitialGpuShare
                  : adaptive ? .5 : CooperativeHostPolicy::initialShare(hostThreads))
        , adaptive_(adaptive)
        , cpuPrimary_(cpuPrimary)
        , continuationRetentionBytes_(continuationRetentionBytes) {
      if (cpuPrimary_ && adaptive_)
        throw std::invalid_argument(
            "CPU-primary and experimental adaptive policies are mutually exclusive");
      if (!continuationRetentionBytes_)
        throw std::invalid_argument("Zero subshower continuation retention budget");
      control_.setHostInitialBatch(CooperativeHostPolicy::capacity(hostThreads));
      for (unsigned e = 0; e < 2; ++e) {
        auto& b = endpoint(e);
        auto& p = peers_[e];
        p.capacity_ = {b.maximumResidentPhotonBatchSize(),
                       b.maximumResidentLeptonBatchSize()};
        p.resident_ = {b.pendingPhotonCount(), b.pendingLeptonCount()};
        for (auto n : p.capacity_)
          if (!n || n > std::numeric_limits<std::size_t>::max() / 4)
            throw std::length_error("Invalid independent subshower capacity");
      }
      stats_.independentSubshowers_ = true;
      stats_.adaptivePolicy_ = adaptive_;
      if (adaptive_ || cpuPrimary_) {
        stats_.cudaCompletionMailboxCapacity_ =
            cpuPrimary_ ? MaximumCpuPrimaryPacketCalls : MaximumPacketCalls;
        stats_.cudaCompletionRetentionBudgetBytes_ = continuationRetentionBytes_;
      }
      stats_.openmpBatchTarget_ = peers_[1].capacity_[1];
      stats_.subshowerInitialHostShare_ = initialHostShare_;
    }
    ~IndependentSubshowerPump() { driver_.waitIdle(); }
    IndependentSubshowerPump(IndependentSubshowerPump const&) = delete;
    IndependentSubshowerPump& operator=(IndependentSubshowerPump const&) = delete;

    // Never read an endpoint queue or statistics while its driver is using it.
    // A submitted result remains pending even if all its particles terminated.
    std::size_t pending(unsigned kind) const noexcept {
      return peers_[0].count(kind) + peers_[1].count(kind) +
             (flight_.valid() && flightKind_ == kind ? flightCount_ : 0);
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
      if (failed_)
        throw std::logic_error("Independent subshower failed; cannot complete output");
    }
    void requireIdle() const {
      requireHealthy();
      if (!idle()) throw std::logic_error("Independent subshower has uncommitted work");
    }

    // Serialize a host radio submission with the CUDA driver, without taking
    // ownership of its pending transport result or advancing any histories.
    void waitCudaDriver() {
      requireHealthy();
      driver_.waitIdle();
    }

    // Accept a bounded prefix, service any ready GPU result, and execute ONE
    // host epoch. Return without joining an unfinished GPU job. The caller may
    // service scalar fallback between calls. No callback is saved by the pump.
    std::size_t advance(std::vector<particle_type> const& input,
                        SubshowerCallbacks const& callbacks) {
      requireHealthy();
      if (advancing_) throw std::logic_error("Reentrant independent subshower progress");
      if (!callbacks.reserveHistories_ || !callbacks.photons_ || !callbacks.leptons_)
        throw std::invalid_argument("Independent subshower callbacks are incomplete");
      advancing_ = true;
      auto start = clock_type::now();
      try {
        if (gpuReady()) collect(callbacks);
        auto accepted = enqueue(input);
        rebalance();
        publishHostWork(callbacks);
        if (!flight_.valid() && !peers_[0].empty()) launch(callbacks);
        if (!yield(callbacks) && !peers_[1].empty()) {
          auto job = prepare(1, callbacks);
          bool concurrent = flight_.valid();
          auto begin = clock_type::now();
          if (observeHost_) observeHost_(true);
          Completion result;
          try {
            result = execute(cpu_, std::move(job));
          } catch (...) {
            if (observeHost_) observeHost_(false);
            throw;
          }
          if (observeHost_) observeHost_(false);
          auto end = clock_type::now();
          auto duration = milliseconds(begin, end);
          ++stats_.openmpSlices_;
          if (duration > 2.) ++stats_.oversizedSlices_;
          stats_.openmpWallMs_ += duration;
          stats_.maximumSliceMs_ = std::max(stats_.maximumSliceMs_, duration);
          if (concurrent) {
            ++stats_.independentJointCalls_;
            ++stats_.slicesWhileCudaPending_;
            ++hostEpochsInFlight_;
            stats_.maximumHostEpochsPerCudaJob_ =
                std::max(stats_.maximumHostEpochsPerCudaJob_, hostEpochsInFlight_);
            auto a = gpuBeginNs_.load(std::memory_order_acquire);
            auto b = gpuEndNs_.load(std::memory_order_acquire);
            if (a) {
              auto lo = std::max(nanoseconds(begin), a);
              auto hi = std::min(nanoseconds(end), b ? b : nanoseconds(end));
              auto overlap = std::max<std::int64_t>(0, hi - lo) * 1.e-6;
              stats_.endpointWindowOverlapMs_ += overlap;
              stats_.openmpWhileCudaPendingMs_ += overlap;
            }
          }
          commit(1, std::move(result), callbacks);
          publishHostWork(callbacks);
        }
        if (gpuReady()) collect(callbacks);
        // collect() may itself return scalar work. Publish that state before
        // launching, so an empty OpenMP queue is not confused with an idle host.
        if (adaptive_ || cpuPrimary_) publishHostWork(callbacks);
        if (!flight_.valid() && !peers_[0].empty() &&
            (adaptive_ || cpuPrimary_ || !yield(callbacks)))
          launch(callbacks);
        // No runnable host work: bounded blocking poll avoids a busy spin. It
        // does not wait for the entire GPU batch, and never waits with CPU work.
        if (flight_.valid() && peers_[1].empty() && !yield(callbacks)) {
          auto wait = clock_type::now();
          flight_.wait_for(std::chrono::milliseconds(1));
          auto duration = milliseconds(wait, clock_type::now());
          stats_.coordinatorIdleWaitMs_ += duration;
          flightHostStarvedMs_ += duration;
        }
        stats_.jointWallMs_ += milliseconds(start, clock_type::now());
        recordQueuePeak();
        advancing_ = false;
        return accepted;
      } catch (...) {
        failed_ = true;
        driver_.waitIdle(); // error cleanup only, never successful batch rendezvous
        advancing_ = false;
        throw;
      }
    }

  private:
    TEndpoint& endpoint(unsigned e) { return e == 0 ? gpu_ : cpu_; }
    static bool yield(SubshowerCallbacks const& c) {
      return c.yieldToScalar_ && c.yieldToScalar_();
    }
    void publishHostWork(SubshowerCallbacks const& callbacks) {
      bool scalar = yield(callbacks);
      gpuScalarForeground_.store(scalar, std::memory_order_release);
      // Scalar fallback/stack work occupies the SAME coordinator as OpenMP.
      // Its unrelated particle histories do not require the CUDA queue to wait.
      // Request a checkpoint when the host genuinely needs fresh work instead.
      gpuNeedsHandoff_.store((adaptive_ || cpuPrimary_) ? (peers_[1].empty() && !scalar)
                                                        : (peers_[1].empty() || scalar),
                             std::memory_order_release);
    }
    static double milliseconds(clock_type::time_point a, clock_type::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    }
    static std::int64_t nanoseconds(clock_type::time_point t) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch())
          .count();
    }
    bool gpuReady() const noexcept {
      return flight_.valid() &&
             flight_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
    std::size_t enqueue(std::vector<particle_type> const& input) {
      std::array<std::size_t, 2> incoming{};
      for (auto const& p : input) {
        if (p.pid_ == 22)
          ++incoming[0];
        else if (is_charged_lepton_pid(p.pid_))
          ++incoming[1];
        else
          throw std::invalid_argument("Unsupported independent subshower PID");
      }
      if (cpuPrimary_) return enqueueCpuPrimary(input, incoming);
      if (adaptive_) return enqueueAdaptive(input);
      auto keep = gpu_.minimumBatchSize(); // immutable after initialization
      std::array<std::size_t, 2> hostQuota{};
      for (unsigned k = 0; k < 2; ++k) {
        // Refill below one epoch before the CPU goes empty. Its descendants
        // still stay on their owning endpoint; only unassigned input is split.
        if (peers_[1].count(k) < peers_[1].capacity_[k]) {
          auto total = incoming[k] + peers_[0].count(k) +
                       (flight_.valid() && flightKind_ == k ? flightCount_ : 0);
          double share = rates_[k][0] > 0 && rates_[k][1] > 0
                             ? rates_[k][1] / (rates_[k][0] + rates_[k][1])
                             : initialHostShare_;
          hostQuota[k] = total > keep
                             ? std::min({incoming[k], total - keep,
                                         2 * peers_[1].capacity_[k] - peers_[1].count(k),
                                         std::max<std::size_t>(256, incoming[k] * share)})
                             : 0;
          if (hostQuota[k] && !peers_[1].empty()) ++stats_.subshowerProactiveRefills_;
        }
      }
      std::size_t accepted = 0;
      for (auto const& p : input) {
        unsigned k = p.pid_ == 22 ? 0 : 1;
        auto room = [&](unsigned e) {
          return peers_[e].waiting_[k].size() < 2 * peers_[e].capacity_[k];
        };
        unsigned e = hostQuota[k] && room(1) ? 1 : 0;
        if (!room(e)) {
          e = 1 - e;
          if (!room(e)) break;
        }
        peers_[e].waiting_[k].push_back(p);
        if (e == 1 && hostQuota[k]) --hostQuota[k];
        ++accepted;
      }
      return accepted;
    }
    Job prepare(unsigned e, SubshowerCallbacks const& callbacks) {
      auto& p = peers_[e];
      unsigned k = p.nextKind_;
      if (!p.count(k)) k = 1 - k;
      if (!p.count(k)) throw std::logic_error("Submitting an empty subshower epoch");
      if (cpuPrimary_ && e == 1) {
        auto selected = CpuPrimarySubshowerPolicy::selectHostSpecies(
            p.nextKind_, {p.count(0), p.count(1)}, p.capacity_, cpu_.minimumBatchSize(),
            p.deferrals_);
        if (selected != k) ++stats_.adaptiveSpeciesCoalesces_[1];
        k = selected;
      }
      auto target = adaptive_ ? control_.targetMs(e, k, flight_.valid())
                              : AdaptiveSubshowerControl::EpochTargetMs;
      if (adaptive_) {
        auto useful = std::min(gpu_.minimumBatchSize(), p.capacity_[k]);
        // Coalesce the small species while processing a useful batch of the
        // other species. No sleeping for an empty future queue; a lone tail
        // drains immediately. Eight deferrals bound species starvation.
        if (p.count(k) < useful && p.count(1 - k) >= useful && p.deferrals_[k] < 8) {
          ++p.deferrals_[k];
          k = 1 - k;
          ++stats_.adaptiveSpeciesCoalesces_[e];
        }
        p.deferrals_[k] = 0;
      }
      if (adaptive_) target = control_.targetMs(e, k, flight_.valid());
      p.nextKind_ = 1 - k; // each endpoint alternates its own photon/lepton queues
      Job job;
      job.kind_ = k;
      job.sequence_ = ++submitted_[e];
      auto capacity = adaptive_ && e == 1
                          ? control_.inputLimit(e, k, p.capacity_[k], target)
                          : p.capacity_[k];
      // Match the standalone router's primary fill order: already resident
      // particles retain their arena slots; new waiting input uses the spare
      // capacity. In particular, do not evict a mature front with fresh input.
      auto protectedResident =
          cpuPrimary_ && e == 1 ? std::min(capacity, p.resident_[k]) : 0;
      auto external = std::min(capacity - protectedResident, p.waiting_[k].size());
      job.input_.assign(p.waiting_[k].begin(), p.waiting_[k].begin() + external);
      p.waiting_[k].erase(p.waiting_[k].begin(), p.waiting_[k].begin() + external);
      auto resident = std::min(capacity - external, p.resident_[k]);
      job.residentInput_ = resident;
      job.boundedInput_ = adaptive_ && e == 1;
      job.count_ = external + resident;
      job.inputLimit_ = capacity;
      job.targetMs_ = target;
      p.resident_[k] -= resident; // ownership moves to the job, not a second queue entry
      job.waves_ = adaptive_ ? control_.waves(e, k, job.count_, target)
                             : (k == 0 ? 16 : (e == 0 ? gpuLeptonWaves_ : 8));
      if (cpuPrimary_ && k == 1)
        job.waves_ = e == 1 ? CpuPrimarySubshowerPolicy::HostLeptonWaves
                            : CpuPrimarySubshowerPolicy::AuxiliaryLeptonWaves;
      // Split tails must not return after every single wave merely because
      // each half is smaller than the original unsplit 4096-input threshold.
      // The measured epoch controller already supplies a bounded hand-off.
      job.minimum_ = adaptive_ ? 1 : (e == 0 ? gpu_.minimumBatchSize() : 1);
      if (cpuPrimary_)
        job.minimum_ = e == 0 ? 1
                              : CpuPrimarySubshowerPolicy::hostCheckpointMinimum(
                                    job.count_, cpu_.minimumBatchSize());
      if (adaptive_ && e == 0 && job.count_ >= gpu_.minimumBatchSize())
        // Newly generated other-species particles are not reflected in the
        // coordinator's count until return. Let a big front hand off its small
        // tail so those daughters can form the next useful front. A job which
        // STARTS small still drains with minimum=1, avoiding single-step tails.
        job.minimum_ = gpu_.minimumBatchSize();
      if (adaptive_ && e == 1) {
        // An independent host epoch may now run >8 waves. Do not spend hundreds
        // of 130-thread barriers draining a vanishing front while other-species
        // daughters wait. Rejoin/coalesce at a useful native-sized boundary.
        // A small initial batch uses a proportional threshold, not an immediate
        // one-wave return. This is a lossless scheduler checkpoint, never a cut.
        job.minimum_ =
            std::min(gpu_.minimumBatchSize(), std::max<std::size_t>(1, job.count_ / 4));
      }
      if (adaptive_) stats_.adaptiveWaveLimit_[e][k] = job.waves_;
      auto n = static_cast<std::uint64_t>(job.count_);
      if (n > std::numeric_limits<std::uint64_t>::max() / 192)
        throw std::overflow_error("Subshower history request overflow");
      auto reserve = k == 0 ? 2 * n * job.waves_
                            : std::max(3 * n, std::min<std::uint64_t>(1U << 20, 192 * n));
      job.first_ = callbacks.reserveHistories_(reserve);
      job.limit_ = cooperative_add(job.first_, reserve);
      if (!job.first_ || job.first_ < lastHistoryLimit_)
        throw std::logic_error("Overlapping/nonmonotonic subshower history lease");
      lastHistoryLimit_ = job.limit_;
      if (e == 0)
        stats_.cudaInputParticles_ += n;
      else
        stats_.openmpInputParticles_ += n;
      return job;
    }
    static Completion execute(TEndpoint& backend, Job job) {
      // Limit resident input as well as the external prefix. Otherwise the
      // backend silently fills the remaining arena and defeats the time budget.
      // Restore even on an exception; legacy/single-endpoint calls are untouched.
      struct InputLimitScope {
        TEndpoint& backend_;
        bool active_;
        ~InputLimitScope() {
          if (active_)
            backend_.setCooperativePendingInputLimit(
                std::numeric_limits<std::size_t>::max());
        }
      } inputScope{backend, job.boundedInput_};
      if (job.boundedInput_) backend.setCooperativePendingInputLimit(job.residentInput_);
      Completion done;
      done.begin_ = clock_type::now();
      if (job.kind_ == 0)
        done.result_ =
            backend.runPhotonWavefront(job.input_, job.first_, job.waves_, job.minimum_);
      else
        done.result_ = backend.runLeptonWavefront(job.input_, job.first_, job.waves_,
                                                  job.limit_, job.minimum_);
      done.end_ = clock_type::now();
      done.resident_ = {backend.pendingPhotonCount(), backend.pendingLeptonCount()};
      std::visit(
          [&](auto const& r) {
            if (r.inputParticles_ != job.count_)
              throw std::logic_error("Subshower epoch input accounting mismatch");
          },
          done.result_);
      if (job.kind_ == 1 &&
          std::get<lepton_result_type>(done.result_).secondaryHistoryIdsUsed_ >
              job.limit_ - job.first_)
        throw std::logic_error("Subshower exceeded its history lease");
      // Release the input capture before returning a potentially large result.
      std::vector<particle_type>().swap(job.input_);
      done.identity_ = std::move(job);
      return done;
    }
    static std::uint64_t retainedBytes(Completion const& result) {
      // Count allocated capacity, not size. A new call is allowed only while
      // the PREVIOUS results fit the retention budget: peak is bounded by that
      // budget plus one ordinary result, whose existing backend limits apply.
      std::uint64_t bytes = sizeof(Completion);
      auto add = [&](auto const& values) {
        using value_type = typename std::decay_t<decltype(values)>::value_type;
        if (values.capacity() >
            std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
          throw std::overflow_error("Subshower completion capacity overflow");
        bytes = cooperative_add(
            bytes, static_cast<std::uint64_t>(values.capacity() * sizeof(value_type)));
      };
      add(result.identity_.input_);
      std::visit(
          [&](auto const& r) {
            add(r.cpuSpillParticles_);
            add(r.fallbackEvents_);
            add(r.observations_);
            add(r.interactionRecords_);
            add(r.stepRecords_);
            add(r.projectedStepRecords_);
            add(r.finalStateRecords_);
            if constexpr (std::is_same_v<std::decay_t<decltype(r)>, photon_result_type>) {
              add(r.electromagneticSecondaries_);
              add(r.remainingPhotons_);
            } else {
              add(r.generatedPhotons_);
              add(r.remainingLeptons_);
              add(r.decayCandidates_);
            }
          },
          result.result_);
      return bytes;
    }
    void launch(SubshowerCallbacks const& callbacks) {
      auto job = prepare(0, callbacks);
      Continuation continuation;
      bool foreground = yield(callbacks);
      continuation.fixedCpuPrimary_ = cpuPrimary_;
      continuation.maximumCalls_ =
          cpuPrimary_ ? MaximumCpuPrimaryPacketCalls : MaximumPacketCalls;
      continuation.enabled_ =
          (adaptive_ || cpuPrimary_) && (!peers_[1].empty() || foreground);
      if (continuation.enabled_) {
        continuation.capacity_ = peers_[0].capacity_;
        if (adaptive_) {
          continuation.control_ =
              control_; // private snapshot; never shares mutable host estimates
          // The CPU's measured useful epoch sets the service horizon. This is
          // long on a CPU-heavy server, short when CUDA is the faster endpoint.
          continuation.targetMs_ =
              std::max(control_.targetMs(1, 0, true), control_.targetMs(1, 1, true));
        }
        // CPU-primary uses a fixed small packet, not a time/rate controller.
        // Every follow-up lease is reserved here; the driver never touches Stack.
        for (unsigned k = 0; k < 2; ++k) {
          auto n = static_cast<std::uint64_t>(continuation.capacity_[k]);
          if (n > std::numeric_limits<std::uint64_t>::max() / 192)
            throw std::overflow_error("Subshower continuation history request overflow");
          auto perCall =
              k == 0 ? 32 * n
                     : std::max(3 * n, std::min<std::uint64_t>(1U << 20, 192 * n));
          continuation.lease_[k] = perCall;
          continuation.stride_ = std::max(continuation.stride_, perCall);
        }
        // One ordered bank, NOT separate photon/lepton banks: repeated species
        // alternation must never allocate children from an earlier ID interval.
        if (continuation.stride_ >
            std::numeric_limits<std::uint64_t>::max() / (continuation.maximumCalls_ - 1))
          throw std::overflow_error("Subshower continuation bank overflow");
        auto reserve = continuation.stride_ * (continuation.maximumCalls_ - 1);
        continuation.first_ = callbacks.reserveHistories_(reserve);
        continuation.limit_ = cooperative_add(continuation.first_, reserve);
        if (!continuation.first_ || continuation.first_ < lastHistoryLimit_)
          throw std::logic_error("Overlapping/nonmonotonic continuation history lease");
        lastHistoryLimit_ = continuation.limit_;
      }
      flightKind_ = job.kind_;
      flightCount_ = job.count_;
      hostEpochsInFlight_ = 0;
      flightHostStarvedMs_ = 0.;
      gpuBeginNs_ = 0;
      gpuEndNs_ = 0;
      flight_ = driver_.submit([this, job = std::move(job),
                                continuation = std::move(continuation)]() mutable {
        selectDevice_();
        gpuBeginNs_.store(nanoseconds(clock_type::now()), std::memory_order_release);
        Packet packet;
        packet.results_.reserve(continuation.enabled_ ? continuation.maximumCalls_ : 1);
        packet.results_.push_back(execute(gpu_, std::move(job)));
        packet.retainedBytes_ =
            (adaptive_ || cpuPrimary_) ? retainedBytes(packet.results_.back()) : 0;
        // No coordinator-owned queue, callback or mutable estimate is touched.
        // The driver may learn from its own completed calls in its PRIVATE copy.
        // If the host needs work, hand over this lossless checkpoint instead of
        // extending a slow GPU tail. The horizon is checked between complete
        // calls, not kernel preemption or a particle cut.
        while (continuation.enabled_) {
          auto const& last = packet.results_.back();
          if (!last.resident_[0] && !last.resident_[1]) {
            packet.stopReason_ = 0;
            break;
          }
          if (!std::visit([](auto const& r) { return r.wavefronts_ != 0; },
                          last.result_)) {
            packet.stopReason_ = 1;
            break;
          }
          if (packet.retainedBytes_ >= continuationRetentionBytes_) {
            packet.stopReason_ = 2;
            break;
          }
          if (packet.results_.size() == continuation.maximumCalls_) {
            packet.stopReason_ = 3;
            break;
          }
          if (gpuNeedsHandoff_.load(std::memory_order_acquire)) {
            packet.stopReason_ = 4;
            break;
          }
          if (!continuation.fixedCpuPrimary_ &&
              milliseconds(packet.results_.front().begin_, last.end_) >=
                  continuation.targetMs_) {
            packet.stopReason_ = 5;
            break;
          }
          // Refresh before planning the next autonomous call. Keeping the launch
          // snapshot unchanged can repeat a one/two-wave lease for all 16 calls,
          // although each short completed call demonstrates room for growth.
          // The coordinator independently observes each result exactly once at
          // commit; this private calibration never mutates global statistics.
          if (!continuation.fixedCpuPrimary_) {
            std::visit(
                [&](auto const& result) {
                  auto const& id = last.identity_;
                  continuation.control_.observe(
                      0, id.kind_, id.count_, result.wavefronts_, id.waves_,
                      result.transportRecords_, milliseconds(last.begin_, last.end_),
                      id.inputLimit_, id.targetMs_);
                },
                last.result_);
          }
          unsigned k = 1 - last.identity_.kind_;
          if (!last.resident_[k]) k = 1 - k;
          auto useful = gpu_.minimumBatchSize();
          if (last.resident_[k] < useful && last.resident_[1 - k] >= useful) k = 1 - k;
          Job follow;
          follow.kind_ = k;
          follow.sequence_ =
              cooperative_add(packet.results_.front().identity_.sequence_,
                              static_cast<std::uint64_t>(packet.results_.size()));
          follow.scalarForegroundContinuation_ =
              gpuScalarForeground_.load(std::memory_order_acquire);
          follow.count_ = std::min(last.resident_[k], continuation.capacity_[k]);
          follow.residentInput_ = follow.count_;
          follow.inputLimit_ = continuation.capacity_[k];
          if (continuation.fixedCpuPrimary_) {
            follow.waves_ = k == 0 ? 16 : CpuPrimarySubshowerPolicy::AuxiliaryLeptonWaves;
            follow.minimum_ = 1;
          } else {
            follow.targetMs_ = continuation.control_.targetMs(0, k, true);
            follow.waves_ =
                continuation.control_.waves(0, k, follow.count_, follow.targetMs_);
            follow.minimum_ = follow.count_ >= useful ? useful : 1;
          }
          follow.first_ =
              cooperative_add(continuation.first_,
                              continuation.stride_ *
                                  static_cast<std::uint64_t>(packet.results_.size() - 1));
          follow.limit_ = cooperative_add(follow.first_, continuation.lease_[k]);
          if (follow.limit_ > continuation.limit_)
            throw std::logic_error("Subshower continuation bank exhausted");
          packet.results_.push_back(execute(gpu_, std::move(follow)));
          packet.retainedBytes_ = cooperative_add(packet.retainedBytes_,
                                                  retainedBytes(packet.results_.back()));
        }
        gpuEndNs_.store(nanoseconds(packet.results_.back().end_),
                        std::memory_order_release);
        return packet;
      });
      ++stats_.subshowerCudaSubmissions_;
      if ((adaptive_ || cpuPrimary_) && foreground)
        ++stats_.subshowerCudaForegroundPackets_;
    }
    void collect(SubshowerCallbacks const& callbacks) {
      if (!gpuReady()) throw std::logic_error("Blocking subshower collect is forbidden");
      auto packet = flight_.get();
      driver_.waitIdle(); // packaged_task teardown, after result is already ready
      auto const packetLimit = cpuPrimary_ ? MaximumCpuPrimaryPacketCalls
                                           : (adaptive_ ? MaximumPacketCalls : 1);
      if (packet.results_.empty() || packet.results_.size() > packetLimit)
        throw std::logic_error("Invalid independent completion mailbox size");
      auto received = clock_type::now();
      // This is actual GPU idle time after the LAST autonomous call. The first
      // result may wait in the mailbox while the GPU is still doing useful work.
      stats_.cudaResultServiceDelayMs_ +=
          milliseconds(packet.results_.back().end_, received);
      if (adaptive_ || cpuPrimary_) {
        stats_.cudaCompletionPeakBytes_ =
            std::max(stats_.cudaCompletionPeakBytes_, packet.retainedBytes_);
        stats_.maximumCudaPacketCalls_ =
            std::max(stats_.maximumCudaPacketCalls_,
                     static_cast<std::uint64_t>(packet.results_.size()));
        ++stats_.cudaContinuationStops_.at(packet.stopReason_);
      }
      for (std::size_t i = 1; i < packet.results_.size(); ++i) {
        submitted_[0] = cooperative_add(submitted_[0], std::uint64_t{1});
        stats_.subshowerCudaSubmissions_ =
            cooperative_add(stats_.subshowerCudaSubmissions_, std::uint64_t{1});
        stats_.subshowerCudaAutonomousContinuations_ = cooperative_add(
            stats_.subshowerCudaAutonomousContinuations_, std::uint64_t{1});
        if (packet.results_[i].identity_.scalarForegroundContinuation_)
          ++stats_.subshowerCudaForegroundContinuations_;
        stats_.cudaInputParticles_ = cooperative_add(
            stats_.cudaInputParticles_,
            static_cast<std::uint64_t>(packet.results_[i].identity_.count_));
      }
      for (auto& result : packet.results_) {
        stats_.cudaDriverWallMs_ += milliseconds(result.begin_, result.end_);
        if (adaptive_ || cpuPrimary_)
          stats_.cudaCompletionBufferDelayMs_ += milliseconds(result.end_, received);
        auto duration = milliseconds(result.begin_, result.end_);
        if (adaptive_) control_.observeGpuDuration(duration);
        stats_.subshowerMaximumCudaEpochMs_ =
            std::max(stats_.subshowerMaximumCudaEpochMs_, duration);
        // A slow GPU must reach a safe redistribution boundary when the CPU is
        // starved. Never interrupt a kernel or alter a particle's physical step.
        if (!adaptive_ && !cpuPrimary_ && result.identity_.kind_ == 1) {
          if (flightHostStarvedMs_ > 5. && duration > 100.) {
            auto target =
                static_cast<std::size_t>(result.identity_.waves_ * 100. / duration);
            auto next = std::max<std::size_t>(16, std::min(gpuLeptonWaves_ / 2, target));
            if (next < gpuLeptonWaves_) ++stats_.subshowerCudaEpochReductions_;
            gpuLeptonWaves_ = next;
          } else if (flightHostStarvedMs_ < 1. && duration < 50.) {
            gpuLeptonWaves_ = std::min<std::size_t>(1024, 2 * gpuLeptonWaves_);
          }
        }
        stats_.subshowerGpuLeptonWaveLimit_ =
            cpuPrimary_ ? CpuPrimarySubshowerPolicy::AuxiliaryLeptonWaves
                        : gpuLeptonWaves_;
        ++stats_.subshowerCudaCommits_;
        commit(0, std::move(result), callbacks);
      }
    }
    void retain(unsigned e, unsigned k, std::vector<particle_type>& values,
                std::vector<particle_type>& spill) {
      auto& p = peers_[e];
      auto& queue = p.waiting_[k];
      auto bound = 4 * p.capacity_[k];
      if (queue.size() > bound)
        throw std::logic_error("Subshower queue exceeded its bound");
      auto n = std::min(values.size(), bound - queue.size());
      queue.insert(queue.end(), std::make_move_iterator(values.begin()),
                   std::make_move_iterator(values.begin() + n));
      // Explicit existing memory-spill path, not an unbounded host queue or a
      // particle cut. The router records and returns every excess state.
      spill.insert(spill.end(), std::make_move_iterator(values.begin() + n),
                   std::make_move_iterator(values.end()));
      values.clear();
    }
    void commit(unsigned e, Completion done, SubshowerCallbacks const& callbacks) {
      auto const& job = done.identity_;
      if (job.sequence_ != committed_[e] + 1)
        throw std::logic_error("Duplicate/out-of-order subshower completion");
      ++committed_[e];
      peers_[e].resident_ = done.resident_;
      auto elapsed = milliseconds(done.begin_, done.end_);
      std::visit(
          [&](auto& result) {
            double rate = result.transportRecords_ / std::max(elapsed, 1.e-6);
            auto& average = rates_[job.kind_][e];
            if (rate > 0) average = average == 0 ? rate : .8 * average + .2 * rate;
            if (cpuPrimary_) {
              stats_.adaptiveTransportRecords_[e][job.kind_] =
                  cooperative_add(stats_.adaptiveTransportRecords_[e][job.kind_],
                                  result.transportRecords_);
              stats_.adaptiveResidentWavefronts_[e][job.kind_] =
                  cooperative_add(stats_.adaptiveResidentWavefronts_[e][job.kind_],
                                  static_cast<std::uint64_t>(result.wavefronts_));
              auto count = job.count_;
              std::size_t bin = 0;
              while (count > 1 && bin + 1 < stats_.adaptiveJobInputHistogram_[e].size()) {
                count >>= 1;
                ++bin;
              }
              ++stats_.adaptiveJobInputHistogram_[e][bin];
            }
            if (adaptive_) {
              control_.observe(e, job.kind_, job.count_, result.wavefronts_, job.waves_,
                               result.transportRecords_, elapsed, job.inputLimit_,
                               job.targetMs_);
              auto const& s = control_.sample(e, job.kind_);
              stats_.adaptiveRecordsPerMs_[e][job.kind_] = s.recordsPerMs_;
              stats_.adaptiveObservations_[e][job.kind_] = s.observations_;
              stats_.adaptiveTransportRecords_[e][job.kind_] =
                  cooperative_add(stats_.adaptiveTransportRecords_[e][job.kind_],
                                  result.transportRecords_);
              stats_.adaptiveResidentWavefronts_[e][job.kind_] =
                  cooperative_add(stats_.adaptiveResidentWavefronts_[e][job.kind_],
                                  static_cast<std::uint64_t>(result.wavefronts_));
              stats_.adaptiveInputLimit_[e][job.kind_] = job.inputLimit_;
              stats_.adaptiveFullObservations_[e][job.kind_] = s.fullObservations_;
              // One exclusive reason for every completed job. No unbounded trace.
              unsigned reason = 5; // other: diagnostics must expose unexpected exits
              if (result.workspaceLimitCheckpoint_)
                reason = 2;
              else if (result.belowMinimumBatchCheckpoint_)
                reason = 1;
              else if (result.completed_)
                reason = 0;
              else if (result.wavefronts_ >= job.waves_)
                reason = 4;
              if constexpr (std::is_same_v<std::decay_t<decltype(result)>,
                                           lepton_result_type>)
                if (result.historyRangeExhausted_) reason = 3;
              ++stats_.adaptiveCompletionReasons_[e][job.kind_][reason];
              auto value = job.count_;
              std::size_t bin = 0;
              while (value > 1 && bin + 1 < stats_.adaptiveJobInputHistogram_[e].size()) {
                value >>= 1;
                ++bin;
              }
              ++stats_.adaptiveJobInputHistogram_[e][bin];
            }
            using result_type = std::decay_t<decltype(result)>;
            if constexpr (std::is_same_v<result_type, photon_result_type>) {
              retain(e, 0, result.remainingPhotons_, result.cpuSpillParticles_);
              retain(e, 1, result.electromagneticSecondaries_, result.cpuSpillParticles_);
              // A prelaunch allocation refusal made no progress: preserve the old
              // scalar escape route instead of retrying the same impossible batch.
              if (result.wavefronts_) requeueSpills(e, result.cpuSpillParticles_);
              ++stats_.photonCalls_;
              callbacks.photons_(std::move(result), elapsed);
            } else {
              retain(e, 1, result.remainingLeptons_, result.cpuSpillParticles_);
              retain(e, 0, result.generatedPhotons_, result.cpuSpillParticles_);
              if (result.wavefronts_) requeueSpills(e, result.cpuSpillParticles_);
              ++stats_.leptonCalls_;
              callbacks.leptons_(std::move(result), elapsed);
            }
          },
          done.result_);
      if (e == 1) ++stats_.subshowerOpenmpEpochs_;
    }
    void requeueSpills(unsigned owner, std::vector<particle_type>& spill) {
      // An endpoint's cross-species arena being full is a storage event, not a
      // request for scalar physics. The completed packet owns these particles;
      // they are not also resident/in flight. Keep them on their endpoint first.
      // Only coordinator-owned waiting vectors are touched, even when the other
      // endpoint is active. Neither endpoint kernel reads these vectors.
      std::size_t remaining = 0;
      for (auto& particle : spill) {
        unsigned k;
        if (particle.pid_ == 22)
          k = 0;
        else if (is_charged_lepton_pid(particle.pid_))
          k = 1;
        else
          throw std::logic_error("Non-EM particle in subshower memory spill");
        bool queued = false;
        for (unsigned e : {owner, 1 - owner}) {
          auto& peer = peers_[e];
          auto& queue = peer.waiting_[k];
          if (queue.size() >= 4 * peer.capacity_[k]) continue;
          queue.push_back(std::move(particle)); // retain all random/history state
          ++stats_.subshowerRequeuedSpillParticles_;
          if (e != owner) {
            ++stats_.subshowerCrossEndpointSpillParticles_;
            stats_.migrationBytes_ =
                cooperative_add(stats_.migrationBytes_, sizeof(particle_type));
          }
          queued = true;
          break;
        }
        if (!queued) {
          if (&spill[remaining] != &particle) spill[remaining] = std::move(particle);
          ++remaining; // explicit old scalar spill policy only if BOTH queues full
        }
      }
      spill.resize(remaining);
    }
    bool canFeedHostWhileInFlight() const noexcept {
      if (!flight_.valid()) return false;
      for (unsigned k = 0; k < 2; ++k)
        if (peers_[1].count(k) < peers_[1].capacity_[k] &&
            peers_[0].waiting_[k].size() >= 256)
          return true;
      return false;
    }
    void rebalance() {
      if (cpuPrimary_) {
        rebalanceCpuPrimary();
        return;
      }
      if (adaptive_) {
        rebalanceAdaptive();
        return;
      }
      // Coordinator-owned waiting vectors are NOT read by the GPU driver:
      // prepare() moves a separate input capture into the submitted job.
      // Steal those safe particles without touching any live endpoint state.
      if (flight_.valid()) {
        for (unsigned k = 0; k < 2; ++k) {
          auto& donor = peers_[0].waiting_[k];
          auto& receiver = peers_[1];
          if (receiver.count(k) >= receiver.capacity_[k] || donor.size() < 256) continue;
          auto count =
              std::min(donor.size(), 2 * receiver.capacity_[k] - receiver.count(k));
          receiver.waiting_[k].insert(receiver.waiting_[k].end(),
                                      std::make_move_iterator(donor.end() - count),
                                      std::make_move_iterator(donor.end()));
          donor.erase(donor.end() - count, donor.end());
          stats_.subshowerInflightWaitingMigrations_++;
          stats_.subshowerInflightWaitingParticles_ += count;
          stats_.migrationBytes_ =
              cooperative_add(stats_.migrationBytes_,
                              static_cast<std::uint64_t>(count * sizeof(particle_type)));
        }
        return; // device-resident and in-flight particles remain exclusive
      }
      for (unsigned destination = 0; destination < 2; ++destination) {
        auto source = 1 - destination;
        auto& donor = peers_[source];
        auto& receiver = peers_[destination];
        if (!receiver.empty() || donor.empty()) continue;
        unsigned k = donor.count(0) >= donor.count(1) ? 0 : 1;
        auto keep = source == 0 ? gpu_.minimumBatchSize() : std::size_t{256};
        if (donor.count(k) <= keep + 256) continue;
        auto count = std::min(receiver.capacity_[k], (donor.count(k) - keep) / 2);
        if (count < 256) continue; // do not shuttle tiny tails merely to make both busy
        auto& from = donor.waiting_[k];
        auto& to = receiver.waiting_[k];
        auto host = std::min(count, from.size());
        to.insert(to.end(), std::make_move_iterator(from.end() - host),
                  std::make_move_iterator(from.end()));
        from.erase(from.end() - host, from.end());
        auto device = count - host;
        if (device) {
          auto moved = endpoint(source).takeCooperativePending(k == 0, device);
          if (moved.size() != device)
            throw std::logic_error("Subshower migration count mismatch");
          to.insert(to.end(), std::make_move_iterator(moved.begin()),
                    std::make_move_iterator(moved.end()));
          donor.resident_[k] -= device;
        }
        stats_.migrationBytes_ =
            cooperative_add(stats_.migrationBytes_,
                            static_cast<std::uint64_t>(count * sizeof(particle_type)));
        ++stats_.subshowerTailMigrations_;
      }
    }
    void recordQueuePeak() {
      std::uint64_t bytes = 0;
      for (auto const& p : peers_)
        for (auto const& q : p.waiting_)
          bytes = cooperative_add(
              bytes, static_cast<std::uint64_t>(q.capacity() * sizeof(particle_type)));
      stats_.subshowerHostQueuePeakBytes_ =
          std::max(stats_.subshowerHostQueuePeakBytes_, bytes);
    }
    std::size_t outstanding(unsigned e, unsigned k) const {
      return peers_[e].count(k) +
             (e == 0 && flight_.valid() && flightKind_ == k ? flightCount_ : 0);
    }
    std::size_t enqueueCpuPrimary(std::vector<particle_type> const& input,
                                  std::array<std::size_t, 2> incoming) {
      std::array<std::size_t, 2> quota{};
      for (unsigned k = 0; k < 2; ++k) {
        auto total = incoming[k] + peers_[1].count(k);
        auto keep = CpuPrimarySubshowerPolicy::hostReserve(peers_[1].capacity_[k],
                                                           cpu_.minimumBatchSize());
        auto capacity = peers_[0].capacity_[k], already = outstanding(0, k);
        auto useful =
            CpuPrimarySubshowerPolicy::helperMinimum(capacity, gpu_.minimumBatchSize());
        if (total > keep && already < capacity && incoming[k] >= useful &&
            total - keep >= useful)
          quota[k] = std::min(
              {incoming[k], total - keep, 2 * capacity - already,
               std::max(useful, static_cast<std::size_t>(
                                    incoming[k] * CpuPrimarySubshowerPolicy::gpuShare(
                                                      rates_[k][1], rates_[k][0])))});
      }
      std::size_t accepted = 0;
      for (auto const& particle : input) {
        unsigned k = particle.pid_ == 22 ? 0 : 1;
        unsigned e = quota[k] ? 0 : 1;
        auto room = [&](unsigned side) {
          return peers_[side].waiting_[k].size() < 2 * peers_[side].capacity_[k];
        };
        if (!room(e)) {
          // A full coordinator queue is backpressure, not permission to seed a
          // tiny helper job outside the quota above. The router retains the
          // unaccepted suffix; the CPU still executes its queued epoch below.
          if (e == 1 || !room(1)) break;
          e = 1;
        }
        peers_[e].waiting_[k].push_back(particle);
        if (e == 0 && quota[k]) --quota[k];
        ++accepted;
      }
      return accepted;
    }
    void moveCpuPrimaryWork(unsigned source, unsigned k, std::size_t count) {
      auto& donor = peers_[source];
      auto& receiver = peers_[1 - source];
      auto n = std::min(count, donor.waiting_[k].size());
      auto& from = donor.waiting_[k];
      auto& to = receiver.waiting_[k];
      to.insert(to.end(), std::make_move_iterator(from.end() - n),
                std::make_move_iterator(from.end()));
      from.erase(from.end() - n, from.end());
      if (count > n) {
        if (source == 0 && flight_.valid())
          throw std::logic_error("CPU-primary attempted in-flight device migration");
        auto moved = endpoint(source).takeCooperativePending(k == 0, count - n);
        if (moved.size() != count - n)
          throw std::logic_error("CPU-primary migration count mismatch");
        to.insert(to.end(), std::make_move_iterator(moved.begin()),
                  std::make_move_iterator(moved.end()));
        donor.resident_[k] -= count - n;
      }
      stats_.migrationBytes_ =
          cooperative_add(stats_.migrationBytes_, count * sizeof(particle_type));
      ++stats_.subshowerTailMigrations_;
      if (flight_.valid()) {
        ++stats_.subshowerInflightWaitingMigrations_;
        stats_.subshowerInflightWaitingParticles_ += count;
      }
    }
    void rebalanceCpuPrimary() {
      for (unsigned k = 0; k < 2; ++k) {
        auto& host = peers_[1];
        auto& device = peers_[0];
        // Do not undo the just-computed throughput split merely because the
        // host batch is not full. Steal only when the primary has no work.
        if (host.empty()) {
          auto available = flight_.valid() ? device.waiting_[k].size() : device.count(k);
          auto n = std::min(available, host.capacity_[k] - host.count(k));
          if (n) moveCpuPrimaryWork(0, k, n);
        }
        // An empty helper takes only a measured share of primary surplus.
        // No predicted completion times or repeated balancing while both run.
        if (flight_.valid() || !device.empty()) continue;
        auto keep = CpuPrimarySubshowerPolicy::hostReserve(host.capacity_[k],
                                                           cpu_.minimumBatchSize());
        auto useful = CpuPrimarySubshowerPolicy::helperMinimum(device.capacity_[k],
                                                               gpu_.minimumBatchSize());
        if (host.count(k) <= keep || host.count(k) - keep < useful) continue;
        auto surplus = host.count(k) - keep;
        auto n =
            std::min({device.capacity_[k], surplus,
                      std::max(useful, static_cast<std::size_t>(
                                           surplus * CpuPrimarySubshowerPolicy::gpuShare(
                                                         rates_[k][1], rates_[k][0])))});
        if (n >= useful) moveCpuPrimaryWork(1, k, n);
      }
    }
    double estimatedLoad(unsigned e) const {
      double load = 0.;
      for (unsigned k = 0; k < 2; ++k)
        load += outstanding(e, k) * control_.unitCost(e, k);
      return load;
    }
    bool adaptiveRoom(unsigned e, unsigned k) const {
      if (peers_[e].waiting_[k].size() >= 2 * peers_[e].capacity_[k]) return false;
      return control_.measured(e, k) ||
             outstanding(e, k) < std::min(peers_[e].capacity_[k],
                                          AdaptiveSubshowerControl::DiscoveryParticles);
    }
    std::size_t enqueueAdaptive(std::vector<particle_type> const& input) {
      std::array<double, 2> load{estimatedLoad(0), estimatedLoad(1)};
      std::size_t accepted = 0;
      for (auto const& particle : input) {
        unsigned k = particle.pid_ == 22 ? 0 : 1;
        bool room0 = adaptiveRoom(0, k), room1 = adaptiveRoom(1, k);
        if (!room0 && !room1) break; // bounded prefix, no particles discarded
        unsigned e = !room0 ? 1
                            : !room1 ? 0
                                     : (load[0] + control_.unitCost(0, k) <=
                                                load[1] + control_.unitCost(1, k)
                                            ? 0
                                            : 1);
        peers_[e].waiting_[k].push_back(particle);
        load[e] += control_.unitCost(e, k);
        ++accepted;
      }
      return accepted;
    }
    void rebalanceAdaptive() {
      // At most one transfer per species per invocation. Waiting vectors are
      // coordinator-owned; resident GPU state is inspected only after collect.
      // Predict balancing benefit INCLUDING measured copy cost and hysteresis.
      for (unsigned k = 0; k < 2; ++k) {
        double a = estimatedLoad(0), b = estimatedLoad(1);
        unsigned source = a > b ? 0 : 1, destination = 1 - source;
        auto& donor = peers_[source];
        auto& receiver = peers_[destination];
        if (!adaptiveRoom(destination, k)) continue;
        auto available = donor.waiting_[k].size() +
                         (source == 0 && flight_.valid() ? 0 : donor.resident_[k]);
        auto room = 2 * receiver.capacity_[k] -
                    std::min(2 * receiver.capacity_[k], receiver.waiting_[k].size());
        if (!control_.measured(destination, k)) {
          auto bound = std::min(receiver.capacity_[k],
                                AdaptiveSubshowerControl::DiscoveryParticles);
          auto owned = outstanding(destination, k);
          room = std::min(room, bound - std::min(bound, owned));
        }
        // Do not fragment a useful GPU staging batch merely to even two noisy
        // estimates while the CPU already has work. An idle CPU may still steal
        // a profitable tail, important on CPU-heavy servers with a slow GPU.
        if (source == 0 && !receiver.empty()) {
          auto keep = std::min(available, gpu_.minimumBatchSize());
          available -= keep;
        }
        auto cap = std::min(available, room);
        if (!cap) continue;
        auto saving = std::abs(a - b);
        auto sum = control_.unitCost(source, k) + control_.unitCost(destination, k);
        auto predicted = std::min(static_cast<double>(cap), saving / sum);
        auto count = static_cast<std::size_t>(predicted);
        // Transferring a handful of particles changes ownership but cannot
        // amortize a new launch/scan/radio batch. Do not oscillate tiny tails.
        if (count < 256) continue;
        double sourceLoad = source == 0 ? a : b,
               destinationLoad = destination == 0 ? a : b;
        double before = std::max(sourceLoad, destinationLoad);
        double after =
            std::max(sourceLoad - count * control_.unitCost(source, k),
                     destinationLoad + count * control_.unitCost(destination, k));
        double benefit = before - after;
        // Compare the predicted critical path, not only donor work removed.
        // Hysteresis prevents oscillatory migrations between two nonempty peers.
        if (benefit <= 1. + 2. * count * migrationMsPerParticle_ +
                           (receiver.empty() ? 0. : .25 * before))
          continue;
        auto begin = clock_type::now();
        auto& from = donor.waiting_[k];
        auto& to = receiver.waiting_[k];
        auto host = std::min(count, from.size());
        to.insert(to.end(), std::make_move_iterator(from.end() - host),
                  std::make_move_iterator(from.end()));
        from.erase(from.end() - host, from.end());
        auto device = count - host;
        if (device) {
          auto moved = endpoint(source).takeCooperativePending(k == 0, device);
          if (moved.size() != device)
            throw std::logic_error("Adaptive migration count mismatch");
          to.insert(to.end(), std::make_move_iterator(moved.begin()),
                    std::make_move_iterator(moved.end()));
          donor.resident_[k] -= device;
        }
        auto elapsed = milliseconds(begin, clock_type::now());
        migrationMsPerParticle_ = .75 * migrationMsPerParticle_ + .25 * elapsed / count;
        stats_.adaptiveMigrationMs_ += elapsed;
        stats_.migrationBytes_ =
            cooperative_add(stats_.migrationBytes_,
                            static_cast<std::uint64_t>(count * sizeof(particle_type)));
        ++stats_.subshowerTailMigrations_;
        if (flight_.valid()) {
          ++stats_.subshowerInflightWaitingMigrations_;
          stats_.subshowerInflightWaitingParticles_ += count;
        }
      }
    }
    TEndpoint& gpu_;
    TEndpoint& cpu_;
    statistics_type& stats_;
    std::function<void()> selectDevice_;
    std::function<void(bool)> observeHost_;
    double initialHostShare_, flightHostStarvedMs_{};
    std::size_t gpuLeptonWaves_{1024};
    std::array<Peer, 2> peers_;
    std::array<std::array<double, 2>, 2> rates_{};
    std::array<std::uint64_t, 2> submitted_{}, committed_{};
    std::uint64_t lastHistoryLimit_{}, hostEpochsInFlight_{};
    unsigned flightKind_{};
    std::size_t flightCount_{};
    bool advancing_{}, failed_{}, adaptive_{}, cpuPrimary_{};
    std::uint64_t continuationRetentionBytes_;
    AdaptiveSubshowerControl control_;
    double migrationMsPerParticle_{};
    std::atomic<std::int64_t> gpuBeginNs_{0}, gpuEndNs_{0};
    std::atomic<bool> gpuNeedsHandoff_{true};
    std::atomic<bool> gpuScalarForeground_{false};
    std::future<Packet> flight_;
    IndependentEndpointDriver driver_; // drain before queues, callbacks and Views
  };

} // namespace corsika::kokkos
