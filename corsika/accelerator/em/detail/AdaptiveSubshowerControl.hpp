/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace corsika::accelerator::em::detail {
// Host-only scheduling estimates. No particle/RNG/physics state is consulted.
// Throughput estimates are symmetric, but the OpenMP owner also services CUDA
// completions. Equal 200 ms leases starve that service, not the CPU kernels.
class AdaptiveSubshowerControl {
 public:
  struct Sample {
    double records_per_ms{}, ms_per_input_wave{};
    std::uint64_t observations{};
    std::size_t last_waves{1};
    std::size_t last_inputs{2048};
  };
  static constexpr std::size_t discovery_particles=4096;
  static constexpr double epoch_target_ms=200.;

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
               std::size_t requested,std::uint64_t records,double elapsed_ms) {
    if(!inputs || !waves || !records || !(elapsed_ms>0.) ||
       !std::isfinite(elapsed_ms)) return; // failed allocation is not calibration
    auto& s=samples_[e][k];
    auto smooth=[&](double old,double now) {
      return s.observations?.75*old+.25*now:now;
    };
    s.records_per_ms=smooth(s.records_per_ms,records/elapsed_ms);
    s.ms_per_input_wave=smooth(s.ms_per_input_wave,
        elapsed_ms/static_cast<double>(inputs)/waves);
    s.last_waves=requested;
    s.last_inputs=inputs;
    ++s.observations;
  }
  // This is an execution quantum, NOT an arena/queue capacity or a CPU share.
  // Eight waves amortize host kernels; if even one full-arena wave is too slow,
  // reduce its input prefix instead of blocking the coordinator indefinitely.
  std::size_t inputLimit(unsigned e,unsigned k,std::size_t capacity,
                         double target_ms) const {
    if (!capacity) return 0;
    auto const& s=sample(e,k);
    if (!s.observations) return std::min<std::size_t>(capacity,2048);
    auto estimate=target_ms/(8.*s.ms_per_input_wave);
    auto floor=std::min<std::size_t>(capacity,256);
    estimate=std::clamp(estimate,static_cast<double>(floor),
                        static_cast<double>(capacity));
    auto growth=s.last_inputs>capacity/2?capacity:2*s.last_inputs;
    return std::max(floor,std::min(growth,static_cast<std::size_t>(estimate)));
  }
  void observeGpuDuration(double elapsed_ms) {
    if (!(elapsed_ms>0.) || !std::isfinite(elapsed_ms)) return;
    gpu_duration_ms_=gpu_duration_ms_>0.?
        .75*gpu_duration_ms_+.25*elapsed_ms:elapsed_ms;
  }
  double hostTargetMs(bool cuda_in_flight) const {
    // Aim for <=5% completion-service latency. A slow T400 permits larger
    // host quanta; a fast GPU gets more frequent service. Neither endpoint's
    // thread count, queue capacity nor aggregate work share is restricted.
    return cuda_in_flight?std::clamp(.05*gpu_duration_ms_,2.,10.):epoch_target_ms;
  }
  std::size_t waves(unsigned e,unsigned k,std::size_t inputs,
                     double target_ms=epoch_target_ms) const {
    auto const& s=sample(e,k);
    if(!s.observations || !inputs) return 1; // measure before a long device lease
    auto ceiling=k==0?std::size_t{16}:std::size_t{1024};
    auto estimate=target_ms/(inputs*s.ms_per_input_wave);
    // Check floating point BEFORE casting, including extreme small durations.
    estimate=std::clamp(estimate,1.,static_cast<double>(ceiling));
    return std::min({ceiling,std::max<std::size_t>(1,2*s.last_waves),
                     static_cast<std::size_t>(estimate)});
  }
 private:
  std::array<std::array<Sample,2>,2> samples_{};
  double gpu_duration_ms_{100.};
};
} // namespace corsika::accelerator::em::detail
