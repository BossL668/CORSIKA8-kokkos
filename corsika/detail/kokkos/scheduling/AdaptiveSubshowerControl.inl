/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  inline void AdaptiveSubshowerControl::setHostInitialBatch(std::size_t count) {
    hostInitialBatch_ = std::max<std::size_t>(1, count);
  }

  inline AdaptiveSubshowerControl::Sample const& AdaptiveSubshowerControl::sample(
      unsigned e, unsigned k) const {
    return samples_[e][k];
  }

  inline bool AdaptiveSubshowerControl::measured(unsigned e, unsigned k) const {
    return sample(e, k).observations_ != 0;
  }

  inline double AdaptiveSubshowerControl::unitCost(unsigned e, unsigned k) const {
    auto rate = sample(e, k).recordsPerMs_;
    if (!(rate > 0.)) rate = sample(1 - e, k).recordsPerMs_;
    return rate > 0. ? 1. / rate : 1.;
  }

  inline void AdaptiveSubshowerControl::observe(unsigned e, unsigned k,
                                                std::size_t inputs, std::size_t waves,
                                                std::size_t requested,
                                                std::uint64_t records, double elapsedMs,
                                                std::size_t inputLimit, double targetMs) {
    if (!inputs || !waves || !records || !(elapsedMs > 0.) || !std::isfinite(elapsedMs))
      return; // failed allocation is not calibration
    auto& s = samples_[e][k];
    // Width calibration requires a full input prefix. Throughput calibration
    // does NOT require filling the maximum memory arena (often >400k states).
    // Otherwise a GPU channel which never fills that arena learns only the
    // final sparse tails. 4096 is a useful probe, not a resident capacity cap.
    bool full = !inputLimit || inputs >= inputLimit - inputLimit / 8;
    auto useful = std::min(inputLimit ? inputLimit : inputs, DiscoveryParticles);
    bool calibrated = inputs >= useful - useful / 8;
    if (calibrated && !s.fullObservations_) {
      s.calibratedRecords_ = s.calibratedMs_ = s.calibratedInputWaves_ = 0.;
    }
    if (calibrated || !s.fullObservations_) {
      // Weight the rates by measured work/time, not by the number of calls.
      // Thousands of sub-ms jobs must not outweigh a large useful epoch.
      auto decay = std::exp(-elapsedMs / 500.);
      s.calibratedRecords_ = decay * s.calibratedRecords_ + records;
      s.calibratedMs_ = decay * s.calibratedMs_ + elapsedMs;
      s.calibratedInputWaves_ =
          decay * s.calibratedInputWaves_ + static_cast<double>(inputs) * waves;
      s.recordsPerMs_ = s.calibratedRecords_ / s.calibratedMs_;
      s.msPerInputWave_ = s.calibratedMs_ / s.calibratedInputWaves_;
    }
    if (calibrated) ++s.fullObservations_;
    if (e == 1 && full && inputLimit) {
      if (!s.inputTarget_) s.inputTarget_ = inputLimit;
      // First reduce the wave lease when a call is too long. Only a full
      // ONE-wave call can prove that the particle width itself is too large.
      if (waves == 1 && elapsedMs > 1.5 * targetMs) {
        s.shortFullJobs_ = 0;
        if (++s.oversizedFullJobs_ >= 2) {
          s.inputTarget_ = std::max<std::size_t>(std::min<std::size_t>(inputLimit, 256),
                                                 inputLimit / 2);
          s.oversizedFullJobs_ = 0;
        }
      } else if (waves >= 2 && elapsedMs / waves < targetMs / 32.) {
        // A natural species checkpoint can end before the requested lease.
        // It still measures per-wave cost. Requiring all eight waves prevented
        // a 130-thread CPU from ever reaching its standalone input width.
        s.oversizedFullJobs_ = 0;
        if (++s.shortFullJobs_ >= 2) {
          auto maximum = std::numeric_limits<std::size_t>::max();
          s.inputTarget_ = inputLimit > maximum / 2 ? maximum : 2 * inputLimit;
          s.shortFullJobs_ = 0;
        }
      } else
        s.shortFullJobs_ = s.oversizedFullJobs_ = 0;
    }
    s.lastWaves_ = requested;
    s.lastInputs_ = inputs;
    ++s.observations_;
  }

  inline std::size_t AdaptiveSubshowerControl::inputLimit(unsigned e, unsigned k,
                                                          std::size_t capacity,
                                                          double /*target_ms*/) const {
    if (!capacity) return 0;
    auto const& s = sample(e, k);
    return std::min(capacity, s.inputTarget_ ? s.inputTarget_ : hostInitialBatch_);
  }

  inline void AdaptiveSubshowerControl::observeGpuDuration(double elapsedMs) {
    if (!(elapsedMs > 0.) || !std::isfinite(elapsedMs)) return;
    gpuDurationMs_ =
        gpuDurationMs_ > 0. ? .75 * gpuDurationMs_ + .25 * elapsedMs : elapsedMs;
  }

  inline double AdaptiveSubshowerControl::hostTargetMs(bool cudaInFlight) const {
    // Service is polled between bounded useful jobs, not enforced by making
    // every physical batch fit 2 ms. This remains a soft (not preemptive) limit.
    return cudaInFlight ? std::clamp(.125 * gpuDurationMs_, 20., 100.) : EpochTargetMs;
  }

  inline double AdaptiveSubshowerControl::targetMs(unsigned e, unsigned k,
                                                   bool /*cuda_in_flight*/) const {
    auto mine = sample(e, k).recordsPerMs_, other = sample(1 - e, k).recordsPerMs_;
    // Uncalibrated devices get equally bounded discovery work. Afterwards
    // the faster endpoint amortizes longer epochs; this is symmetric between
    // a many-core server and a GPU-heavy laptop, not a GPU-first prior.
    if (!(mine > 0.) || !(other > 0.)) return EpochTargetMs;
    auto share = mine / (mine + other);
    // Both endpoints now have independent progress. The CUDA driver can
    // continue across species while OpenMP executes a useful epoch, with a
    // bounded history bank/mailbox/retention budget. Clamping the CPU lease
    // to 1/8 of ONE CUDA call was a synchronous-service constraint: it also
    // shortened the driver packet horizon, causing both sides to hand off
    // repeatedly. Keep the same measured symmetric work target, regardless
    // of whether a GPU packet happens to be in flight. The driver still
    // checks handoff, memory and time between complete calls; no kernel is
    // preempted, and this 50--800 ms target is soft, not a wall-time guarantee.
    return std::clamp(1000. * share * share, 50., 800.);
  }

  inline std::size_t AdaptiveSubshowerControl::waves(unsigned e, unsigned k,
                                                     std::size_t inputs,
                                                     double targetMs) const {
    auto const& s = sample(e, k);
    if (!inputs) return 1;
    // Both instances call the SAME resident kernels. An arbitrary eight-wave
    // OpenMP ceiling is not a physical constraint; use the common limits.
    auto ceiling = k == 0 ? std::size_t{16} : std::size_t{1024};
    // A bounded useful probe; one-wave cold probes systematically expose
    // launch/barrier cost as if it were transport cost.
    if (!s.observations_) return std::min<std::size_t>(ceiling, e == 1 ? 8 : 64);
    auto estimate = targetMs / (inputs * s.msPerInputWave_);
    // Check floating point BEFORE casting, including extreme small durations.
    estimate = std::clamp(estimate, 1., static_cast<double>(ceiling));
    auto growth = s.lastWaves_ > ceiling / 2 ? ceiling : 2 * s.lastWaves_;
    return std::min(
        {ceiling, std::max<std::size_t>(1, growth), static_cast<std::size_t>(estimate)});
  }
} // namespace corsika::kokkos
