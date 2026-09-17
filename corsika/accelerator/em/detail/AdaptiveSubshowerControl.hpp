/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace corsika::accelerator::em::detail {
// Host-only scheduling estimates. No particle/RNG/physics state is consulted.
// Keep the particle quantum separate from the completion-service interval.
// A sparse tail is not a measurement of a full batch: feeding its fixed call
// overhead back into the width estimate caused the v3 small-job feedback loop.
class AdaptiveSubshowerControl {
 public:
  struct Sample {
    double records_per_ms{}, ms_per_input_wave{};
    double calibrated_records{}, calibrated_ms{}, calibrated_input_waves{};
    std::uint64_t observations{};
    std::size_t last_waves{1};
    std::size_t last_inputs{2048};
    std::size_t input_target{};
    std::uint64_t full_observations{};
    unsigned short_full_jobs{}, oversized_full_jobs{};
  };
  static constexpr std::size_t discovery_particles=4096;
  static constexpr double epoch_target_ms=200.;
  void setHostInitialBatch(std::size_t count) { host_initial_batch_=std::max<std::size_t>(1,count); }

  Sample const& sample(unsigned e,unsigned k) const {return samples_[e][k];}
  bool measured(unsigned e,unsigned k) const {return sample(e,k).observations!=0;}
  // For an unmeasured channel borrow the other endpoint's estimate, not a
  // hardware-brand/thread-count prior. Each channel still gets bounded probes.
  double unitCost(unsigned e,unsigned k) const {
    auto rate=sample(e,k).records_per_ms;
    if(!(rate>0.)) rate=sample(1-e,k).records_per_ms;
    return rate>0.?1./rate:1.;
  }
  void observe(unsigned e,unsigned k,std::size_t inputs,std::size_t waves,
               std::size_t requested,std::uint64_t records,double elapsed_ms,
               std::size_t input_limit=0,double target_ms=epoch_target_ms) {
    if(!inputs || !waves || !records || !(elapsed_ms>0.) ||
       !std::isfinite(elapsed_ms)) return; // failed allocation is not calibration
    auto& s=samples_[e][k];
    // Width calibration requires a full input prefix. Throughput calibration
    // does NOT require filling the maximum memory arena (often >400k states).
    // Otherwise a GPU channel which never fills that arena learns only the
    // final sparse tails. 4096 is a useful probe, not a resident capacity cap.
    bool full=!input_limit || inputs>=input_limit-input_limit/8;
    auto useful=std::min(input_limit?input_limit:inputs,discovery_particles);
    bool calibrated=inputs>=useful-useful/8;
    if(calibrated && !s.full_observations) {
      s.calibrated_records=s.calibrated_ms=s.calibrated_input_waves=0.;
    }
    if(calibrated || !s.full_observations) {
      // Weight the rates by measured work/time, not by the number of calls.
      // Thousands of sub-ms jobs must not outweigh a large useful epoch.
      auto decay=std::exp(-elapsed_ms/500.);
      s.calibrated_records=decay*s.calibrated_records+records;
      s.calibrated_ms=decay*s.calibrated_ms+elapsed_ms;
      s.calibrated_input_waves=decay*s.calibrated_input_waves+
          static_cast<double>(inputs)*waves;
      s.records_per_ms=s.calibrated_records/s.calibrated_ms;
      s.ms_per_input_wave=s.calibrated_ms/s.calibrated_input_waves;
    }
    if(calibrated) ++s.full_observations;
    if(e==1 && full && input_limit) {
      if(!s.input_target)s.input_target=input_limit;
      // First reduce the wave lease when a call is too long. Only a full
      // ONE-wave call can prove that the particle width itself is too large.
      if(waves==1 && elapsed_ms>1.5*target_ms) {
        s.short_full_jobs=0;
        if(++s.oversized_full_jobs>=2) {
          s.input_target=std::max<std::size_t>(std::min<std::size_t>(input_limit,256),input_limit/2);
          s.oversized_full_jobs=0;
        }
      } else if(waves>=2 && elapsed_ms/waves<target_ms/32.) {
        // A natural species checkpoint can end before the requested lease.
        // It still measures per-wave cost. Requiring all eight waves prevented
        // a 130-thread CPU from ever reaching its standalone input width.
        s.oversized_full_jobs=0;
        if(++s.short_full_jobs>=2) {
          auto maximum=std::numeric_limits<std::size_t>::max();
          s.input_target=input_limit>maximum/2?maximum:2*input_limit;
          s.short_full_jobs=0;
        }
      } else s.short_full_jobs=s.oversized_full_jobs=0;
    }
    s.last_waves=requested;
    s.last_inputs=inputs;
    ++s.observations;
  }
  // Hysteretic measured width, NOT target/(8*cost) at every dispatch. In
  // particular a short/sparse job cannot ratchet the next width down to 256.
  std::size_t inputLimit(unsigned e,unsigned k,std::size_t capacity,
                         double /*target_ms*/) const {
    if (!capacity) return 0;
    auto const& s=sample(e,k);
    return std::min(capacity,s.input_target?s.input_target:host_initial_batch_);
  }
  void observeGpuDuration(double elapsed_ms) {
    if (!(elapsed_ms>0.) || !std::isfinite(elapsed_ms)) return;
    gpu_duration_ms_=gpu_duration_ms_>0.?
        .75*gpu_duration_ms_+.25*elapsed_ms:elapsed_ms;
  }
  double hostTargetMs(bool cuda_in_flight) const {
    // Service is polled between bounded useful jobs, not enforced by making
    // every physical batch fit 2 ms. This remains a soft (not preemptive) limit.
    return cuda_in_flight?std::clamp(.125*gpu_duration_ms_,20.,100.):epoch_target_ms;
  }
  double targetMs(unsigned e,unsigned k,bool /*cuda_in_flight*/) const {
    auto mine=sample(e,k).records_per_ms,other=sample(1-e,k).records_per_ms;
    // Uncalibrated devices get equally bounded discovery work. Afterwards
    // the faster endpoint amortizes longer epochs; this is symmetric between
    // a many-core server and a GPU-heavy laptop, not a GPU-first prior.
    if(!(mine>0.) || !(other>0.))return epoch_target_ms;
    auto share=mine/(mine+other);
    // Both endpoints now have independent progress. The CUDA driver can
    // continue across species while OpenMP executes a useful epoch, with a
    // bounded history bank/mailbox/retention budget. Clamping the CPU lease
    // to 1/8 of ONE CUDA call was a synchronous-service constraint: it also
    // shortened the driver packet horizon, causing both sides to hand off
    // repeatedly. Keep the same measured symmetric work target, regardless
    // of whether a GPU packet happens to be in flight. The driver still
    // checks handoff, memory and time between complete calls; no kernel is
    // preempted, and this 50--800 ms target is soft, not a wall-time guarantee.
    return std::clamp(1000.*share*share,50.,800.);
  }
  std::size_t waves(unsigned e,unsigned k,std::size_t inputs,
                     double target_ms=epoch_target_ms) const {
    auto const& s=sample(e,k);
    if(!inputs) return 1;
    // Both instances call the SAME resident kernels. An arbitrary eight-wave
    // OpenMP ceiling is not a physical constraint; use the common limits.
    auto ceiling=k==0?std::size_t{16}:std::size_t{1024};
    // A bounded useful probe; one-wave cold probes systematically expose
    // launch/barrier cost as if it were transport cost.
    if(!s.observations)return std::min<std::size_t>(ceiling,e==1?8:64);
    auto estimate=target_ms/(inputs*s.ms_per_input_wave);
    // Check floating point BEFORE casting, including extreme small durations.
    estimate=std::clamp(estimate,1.,static_cast<double>(ceiling));
    auto growth=s.last_waves>ceiling/2?ceiling:2*s.last_waves;
    return std::min({ceiling,std::max<std::size_t>(1,growth),
                     static_cast<std::size_t>(estimate)});
  }
 private:
  std::array<std::array<Sample,2>,2> samples_{};
  double gpu_duration_ms_{100.};
  std::size_t host_initial_batch_{2048};
};
} // namespace corsika::accelerator::em::detail
