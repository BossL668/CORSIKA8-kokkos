/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace corsika::kokkos {

  // Host-only scheduling estimates. No particle/RNG/physics state is consulted.
  // Keep the particle quantum separate from the completion-service interval.
  // A sparse tail is not a measurement of a full batch: feeding its fixed call
  // overhead back into the width estimate caused the v3 small-job feedback loop.
  class AdaptiveSubshowerControl {
  public:
    struct Sample {
      double recordsPerMs_{}, msPerInputWave_{};
      double calibratedRecords_{}, calibratedMs_{}, calibratedInputWaves_{};
      std::uint64_t observations_{};
      std::size_t lastWaves_{1};
      std::size_t lastInputs_{2048};
      std::size_t inputTarget_{};
      std::uint64_t fullObservations_{};
      unsigned shortFullJobs_{}, oversizedFullJobs_{};
    };
    static constexpr std::size_t DiscoveryParticles = 4096;
    static constexpr double EpochTargetMs = 200.;
    void setHostInitialBatch(std::size_t count);

    Sample const& sample(unsigned e, unsigned k) const;
    bool measured(unsigned e, unsigned k) const;
    // For an unmeasured channel borrow the other endpoint's estimate, not a
    // hardware-brand/thread-count prior. Each channel still gets bounded probes.
    double unitCost(unsigned e, unsigned k) const;
    void observe(unsigned e, unsigned k, std::size_t inputs, std::size_t waves,
                 std::size_t requested, std::uint64_t records, double elapsedMs,
                 std::size_t inputLimit = 0, double targetMs = EpochTargetMs);
    // Hysteretic measured width, NOT target/(8*cost) at every dispatch. In
    // particular a short/sparse job cannot ratchet the next width down to 256.
    std::size_t inputLimit(unsigned e, unsigned k, std::size_t capacity,
                           double /*target_ms*/) const;
    void observeGpuDuration(double elapsedMs);
    double hostTargetMs(bool cudaInFlight) const;
    double targetMs(unsigned e, unsigned k, bool /*cuda_in_flight*/) const;
    std::size_t waves(unsigned e, unsigned k, std::size_t inputs,
                      double targetMs = EpochTargetMs) const;

  private:
    std::array<std::array<Sample, 2>, 2> samples_{};
    double gpuDurationMs_{100.};
    std::size_t hostInitialBatch_{2048};
  };

} // namespace corsika::kokkos
#include <corsika/detail/kokkos/scheduling/AdaptiveSubshowerControl.inl>
