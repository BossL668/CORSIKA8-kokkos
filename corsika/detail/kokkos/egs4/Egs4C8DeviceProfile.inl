/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4C8AirOutput.inl>
#include <corsika/kokkos/egs4/Session.hpp>
#include <corsika/kokkos/output/ProfileAccumulatorStorage.hpp>
#include <corsika/kokkos/output/ProfileProjectionStorage.hpp>

namespace corsika::kokkos::egs4::application {
  // Reuse production geometry, crossing rules, fixed-point ledger and continuous
  // / endpoint deposit routines. Do not synthesize PROPOSAL interactions or cuts:
  // EGS4 already froze the exact total and endpoint energy in AirOutput.
  template <class TOutputs>
  struct AccumulateEgs4Profile {
    TOutputs outputs_;
    ::corsika::kokkos::ProfileProjectionView projection_;
    ::corsika::kokkos::ProfileAccumulatorView accumulator_;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
      if (!outputs_(i).hasStep_) return;
      auto const& r = outputs_(i).step_;
      namespace d = ::corsika::kokkos;
      using atomic_type = ::corsika::kokkos::ProfileAtomicOperations;
      double start = d::project_profile_grammage(projection_, r.startPositionM_);
      double end = d::project_profile_grammage(projection_, r.endPositionM_);
      if (!d::profile_finite(start) || !d::profile_finite(end) ||
          !d::profile_finite(r.weight_) || r.weight_ < 0. ||
          !d::profile_finite(r.depositedEnergyGeV_) ||
          !d::profile_finite(r.cutDepositedEnergyGeV_) || r.cutDepositedEnergyGeV_ < 0. ||
          r.depositedEnergyGeV_ < r.cutDepositedEnergyGeV_) {
        atomic_type::add(&accumulator_.counters_->invalidRecords_, 1ULL);
        return;
      }
      d::accumulate_particle_profile<atomic_type>(accumulator_, r.pid_, start, end,
                                                  r.weight_);
      d::accumulate_energy_profile<atomic_type>(
          accumulator_, accumulator_.energyLoss_, start, end,
          (r.depositedEnergyGeV_ - r.cutDepositedEnergyGeV_) * r.weight_);
      d::accumulate_point_energy_profile<atomic_type>(
          accumulator_, accumulator_.energyLoss_, end,
          r.cutDepositedEnergyGeV_ * r.weight_);
      atomic_type::add(&accumulator_.counters_->steps_, 1ULL);
    }
  };
  template <class TExec>
  class DeviceProfile {
    ::corsika::kokkos::ProfileAccumulatorStorage<TExec> accumulator_;
    ::corsika::kokkos::ProfileProjectionStorage<TExec> projection_;
    OutputCallbacks const& output_;
    double weightLimit_{}, energyLimit_{};

  public:
    DeviceProfile(::corsika::kokkos::ProfileProjectionConfig const& config,
                  OutputCallbacks const& output)
        : output_(output) {
      if (config.enabled_ && (!config.accumulateOnDevice_ || !output.profile_))
        throw std::invalid_argument(
            "Native device profile requires accumulation and a final consumer");
      projection_.initialize(config);
      accumulator_.initialize(config);
      weightLimit_ = config.fixedPointWeightLimit_;
      energyLimit_ = config.fixedPointEnergyLimitGeV_;
    }
    bool enabled() const { return accumulator_.enabled(); }
    void reset() { accumulator_.reset(weightLimit_, energyLimit_); }
    template <class TOutputs>
    void accumulate(TOutputs const& records) {
      if (!enabled()) return;
      accumulator_.requireOpenForAccumulation();
      Kokkos::parallel_for(
          "egs4_accumulate_resident_profile",
          Kokkos::RangePolicy<TExec>(0, records.extent(0)),
          AccumulateEgs4Profile<TOutputs>{records, projection_.deviceView(),
                                          accumulator_.deviceView()});
    }
    void finish(RunStatistics& stats) {
      if (!enabled()) return;
      auto result = accumulator_.download();
      stats.profileExecutionSpace_ = TExec::name();
      stats.profileSteps_ = result.steps_;
      stats.profileFixedPointOverflows_ = result.fixedPointOverflows_;
      stats.profileInvalidRecords_ = result.invalidRecords_;
      if (result.steps_ != stats.steps_ || result.fixedPointOverflows_ ||
          result.invalidRecords_)
        throw std::runtime_error(
            "Native device profile missing/invalid/overflowing records");
      output_.profile_(result);
    }
  };
} // namespace corsika::kokkos::egs4::application
