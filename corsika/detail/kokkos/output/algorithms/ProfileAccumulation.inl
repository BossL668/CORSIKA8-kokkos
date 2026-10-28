/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline bool profile_finite(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }
  CORSIKA_HOST_DEVICE inline double profile_abs(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fabs(value);
#else
    return std::fabs(value);
#endif
  }
  CORSIKA_HOST_DEVICE inline double profile_floor(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::floor(value);
#else
    return std::floor(value);
#endif
  }
  CORSIKA_HOST_DEVICE inline double profile_ceil(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::ceil(value);
#else
    return std::ceil(value);
#endif
  }
  CORSIKA_HOST_DEVICE inline long long profile_round(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return static_cast<long long>(::rint(value));
#else
    return static_cast<long long>(std::rint(value));
#endif
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline bool checked_profile_atomic_add(long long* address,
                                                             long long increment) {
    auto const previous = AtomicOperations::add(address, increment);
    return !((increment > 0 && previous > ProfileSignedIntegerMaximum - increment) ||
             (increment < 0 && previous < ProfileSignedIntegerMinimum - increment));
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void add_profile_fixed_point(long long* address,
                                                          double contribution,
                                                          double scale,
                                                          ProfileCounters* counters) {
    auto const scaled = contribution * scale;
    if (!profile_finite(scaled) || profile_abs(scaled) >= ProfileSignedIntegerLimit ||
        !checked_profile_atomic_add<AtomicOperations>(address, profile_round(scaled)))
      AtomicOperations::add(&counters->fixedPointOverflows_, 1ULL);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_particle_profile(
      ProfileAccumulatorView const& accumulator, std::int32_t pid, double start_grammage,
      double end_grammage, double weight) {
    using namespace corsika::kokkos;
    auto const crossings =
        corsika::longitudinal_profile_bins(start_grammage / accumulator.binWidthGPerCm2_,
                                           end_grammage / accumulator.binWidthGPerCm2_,
                                           accumulator.bins_, accumulator.crossingMode_);
    if (crossings.begin_ == crossings.end_) return;
    long long* profile = nullptr;
    if (pid == static_cast<std::int32_t>(EmPid::Photon))
      profile = accumulator.photons_;
    else if (pid == static_cast<std::int32_t>(EmPid::Electron))
      profile = accumulator.electrons_;
    else if (pid == static_cast<std::int32_t>(EmPid::Positron))
      profile = accumulator.positrons_;
    else if (pid == static_cast<std::int32_t>(EmPid::MuonMinus))
      profile = accumulator.muonsMinus_;
    else if (pid == static_cast<std::int32_t>(EmPid::MuonPlus))
      profile = accumulator.muonsPlus_;
    if (profile == nullptr) {
      AtomicOperations::add(&accumulator.counters_->invalidRecords_, 1ULL);
      return;
    }
    for (auto bin = crossings.begin_; bin < crossings.end_; ++bin)
      add_profile_fixed_point<AtomicOperations>(
          profile + bin, weight, accumulator.weightScale_, accumulator.counters_);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_muon_parent_production(
      ProfileAccumulatorView const& accumulator, double grammage, double parent_weight) {
    if (accumulator.bins_ == 0 || accumulator.muonParentProductions_ == nullptr) return;
    auto const bin_value = profile_floor(grammage / accumulator.binWidthGPerCm2_);
    if (!profile_finite(bin_value)) {
      AtomicOperations::add(&accumulator.counters_->invalidRecords_, 1ULL);
      return;
    }
    if (bin_value < 0.) return;
    auto const candidate = static_cast<std::size_t>(bin_value);
    auto const bin = candidate < accumulator.bins_ ? candidate : accumulator.bins_ - 1;
    add_profile_fixed_point<AtomicOperations>(accumulator.muonParentProductions_ + bin,
                                              parent_weight, accumulator.weightScale_,
                                              accumulator.counters_);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_energy_profile(
      ProfileAccumulatorView const& accumulator, long long* energy_loss,
      double start_grammage, double end_grammage, double weighted_deposit_GeV) {
    if (!(weighted_deposit_GeV > 0.) || accumulator.bins_ == 0) return;
    AtomicOperations::add(&accumulator.counters_->depositedSteps_, 1ULL);
    if (start_grammage > end_grammage) {
      auto const temporary = start_grammage;
      start_grammage = end_grammage;
      end_grammage = temporary;
    }
    auto const delta = end_grammage - start_grammage;
    auto const maximum_bin = static_cast<int>(accumulator.bins_ - 1);
    auto first = static_cast<int>(start_grammage / accumulator.binWidthGPerCm2_);
    first = first < 0 ? 0 : first;
    first = first > maximum_bin ? maximum_bin : first;
    if (delta < accumulator.energyLossThresholdGPerCm2_) {
      add_profile_fixed_point<AtomicOperations>(energy_loss + first, weighted_deposit_GeV,
                                                accumulator.energyScale_,
                                                accumulator.counters_);
      return;
    }
    auto last = static_cast<int>(end_grammage / accumulator.binWidthGPerCm2_);
    last = last < 0 ? 0 : last;
    last = last > maximum_bin ? maximum_bin : last;
    auto const density = weighted_deposit_GeV / delta;
    if (first == last) {
      add_profile_fixed_point<AtomicOperations>(energy_loss + first, density * delta,
                                                accumulator.energyScale_,
                                                accumulator.counters_);
      return;
    }
    add_profile_fixed_point<AtomicOperations>(
        energy_loss + first,
        density * (1. + static_cast<double>(first)) * accumulator.binWidthGPerCm2_ -
            density * start_grammage,
        accumulator.energyScale_, accumulator.counters_);
    add_profile_fixed_point<AtomicOperations>(
        energy_loss + last,
        density * end_grammage -
            density * static_cast<double>(last) * accumulator.binWidthGPerCm2_,
        accumulator.energyScale_, accumulator.counters_);
    for (auto bin = first + 1; bin < last; ++bin)
      add_profile_fixed_point<AtomicOperations>(
          energy_loss + bin, density * accumulator.binWidthGPerCm2_,
          accumulator.energyScale_, accumulator.counters_);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_point_energy_profile(
      ProfileAccumulatorView const& accumulator, long long* energy_loss, double grammage,
      double weighted_deposit_GeV, bool native_signed_cut) {
    if ((!(weighted_deposit_GeV > 0.) &&
         !(native_signed_cut && weighted_deposit_GeV < 0.)) ||
        accumulator.bins_ == 0)
      return;
    // Mirror EnergyLossWriter::write(Point, ...), including a zero dX threshold.
    // Reusing the segment overload with equal endpoints divides by zero when
    // the configured continuous-track threshold is zero.
    auto const maximum_bin = static_cast<int>(accumulator.bins_ - 1);
    auto bin = static_cast<int>(grammage / accumulator.binWidthGPerCm2_);
    bin = bin < 0 ? 0 : bin;
    bin = bin > maximum_bin ? maximum_bin : bin;
    AtomicOperations::add(&accumulator.counters_->depositedSteps_, 1ULL);
    add_profile_fixed_point<AtomicOperations>(energy_loss + bin, weighted_deposit_GeV,
                                              accumulator.energyScale_,
                                              accumulator.counters_);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_terminal_energy(
      ProfileAccumulatorView const& accumulator, double weighted_total_energy_GeV,
      bool observed) {
    if (!(weighted_total_energy_GeV >= 0.) ||
        !profile_finite(weighted_total_energy_GeV)) {
      AtomicOperations::add(&accumulator.counters_->invalidRecords_, 1ULL);
      return;
    }
    add_profile_fixed_point<AtomicOperations>(
        observed ? &accumulator.counters_->weightedObservedTotalEnergy_
                 : &accumulator.counters_->weightedEscapedTotalEnergy_,
        weighted_total_energy_GeV, accumulator.energyScale_, accumulator.counters_);
  }
  template <class AtomicOperations, class FinalStateRecord>
  CORSIKA_HOST_DEVICE inline void accumulate_profile_thinning(
      FinalStateRecord const& record, ProfileCounters* counters) {
    using namespace corsika::kokkos;
    auto const status = static_cast<EmThinningStatus>(record.thinningStatus_);
    if (status == EmThinningStatus::Hillas)
      AtomicOperations::add(&counters->thinningHillasVertices_, 1ULL);
    else if (status == EmThinningStatus::Statistical)
      AtomicOperations::add(&counters->thinningStatisticalVertices_, 1ULL);
    else if (status != EmThinningStatus::NotApplied)
      AtomicOperations::add(&counters->invalidRecords_, 1ULL);
    auto const original_multiplicity =
        record.processId_ == PhotoelectricProcessId
            ? 1U
            : record.processId_ == ElectronPairProcessId ? 3U : 2U;
    if (record.secondaryCount_ > original_multiplicity)
      AtomicOperations::add(&counters->invalidRecords_, 1ULL);
    else
      AtomicOperations::add(&counters->thinningParticlesDiscarded_,
                            static_cast<unsigned long long>(original_multiplicity -
                                                            record.secondaryCount_));
  }
  template <class TransportRecord>
  CORSIKA_HOST_DEVICE inline TransportRecord const* find_profile_transport_record(
      TransportRecord const* records, std::size_t count, std::uint64_t input_index) {
    std::size_t first = 0;
    std::size_t last = count;
    while (first < last) {
      auto const middle = first + (last - first) / 2;
      if (records[middle].inputIndex_ < input_index)
        first = middle + 1;
      else
        last = middle;
    }
    if (first >= count || records[first].inputIndex_ != input_index) return nullptr;
    return records + first;
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_photon_profile_step(
      ProfileProjectionView const& projection, ProfileAccumulatorView const& accumulator,
      PhotonTransportRecord const& record) {
    using namespace corsika::kokkos;
    auto const start = project_profile_grammage(projection, record.start_.positionM_);
    auto const end = project_profile_grammage(projection, record.end_.positionM_);
    accumulate_particle_profile<AtomicOperations>(accumulator, record.start_.pid_, start,
                                                  end, record.start_.weight_);
    accumulate_point_energy_profile<AtomicOperations>(
        accumulator, accumulator.energyLoss_, end,
        record.cutDepositedEnergyGeV_ * record.start_.weight_);
    AtomicOperations::add(&accumulator.counters_->steps_, 1ULL);
    if (record.limit_ == PhotonTransportLimit::ParticleCut) {
      AtomicOperations::add(&accumulator.counters_->photonCuts_, 1ULL);
      if (record.observationSurfaceReachedBeforeCut_ != 0U) {
        add_profile_fixed_point<AtomicOperations>(
            &accumulator.counters_->weightedObservationCutOverlapEnergy_,
            record.cutDepositedEnergyGeV_ * record.start_.weight_,
            accumulator.energyScale_, accumulator.counters_);
        accumulate_terminal_energy<AtomicOperations>(
            accumulator, record.end_.energyGeV_ * record.start_.weight_, true);
      }
    } else if (record.limit_ == PhotonTransportLimit::ObservationSurface) {
      accumulate_terminal_energy<AtomicOperations>(
          accumulator, record.end_.energyGeV_ * record.start_.weight_, true);
    } else if (record.limit_ == PhotonTransportLimit::EscapedEnvironment) {
      accumulate_terminal_energy<AtomicOperations>(
          accumulator, record.end_.energyGeV_ * record.start_.weight_, false);
    }
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_photon_profile_final_state(
      [[maybe_unused]] ProfileProjectionView const& projection,
      ProfileAccumulatorView const& accumulator,
      PhotonTransportRecord const* transport_records, std::size_t transport_count,
      PhotonFinalStateRecord const& record) {
    using namespace corsika::kokkos;
    accumulate_profile_thinning<AtomicOperations>(record, accumulator.counters_);
    add_profile_fixed_point<AtomicOperations>(
        &accumulator.counters_->weightedMassConventionCorrection_,
        record.weightedMassConventionCorrectionGeV_, accumulator.energyScale_,
        accumulator.counters_);
    auto const atomic_electron_process = record.processId_ == ComptonProcessId ||
                                         record.processId_ == PhotoelectricProcessId;
    if (!atomic_electron_process) return;
    auto const* transport = find_profile_transport_record(
        transport_records, transport_count, record.inputIndex_);
    if (transport == nullptr || transport->start_.historyId_ != record.parentHistoryId_ ||
        transport->interaction_.processId_ != record.processId_) {
      AtomicOperations::add(&accumulator.counters_->invalidRecords_, 1ULL);
      return;
    }
    add_profile_fixed_point<AtomicOperations>(
        &accumulator.counters_->weightedMediumRestMassInput_,
        ElectronMassGeV * transport->start_.weight_, accumulator.energyScale_,
        accumulator.counters_);
    if (record.processId_ != PhotoelectricProcessId) return;
    if (!profile_finite(record.energySplitFraction_) ||
        record.energySplitFraction_ < 0. || record.energySplitFraction_ > 1.) {
      AtomicOperations::add(&accumulator.counters_->invalidRecords_, 1ULL);
      return;
    }
    auto const deposit = transport->end_.energyGeV_ * (1. - record.energySplitFraction_) *
                         transport->start_.weight_;
    // Preserve scalar PROPOSAL's legacy dE/dX definition. Its photoelectric
    // secondary has this binding energy removed, but InteractionModel does
    // not pass it to EnergyLossWriter. Track it as an explicit closure sink.
    add_profile_fixed_point<AtomicOperations>(
        &accumulator.counters_->weightedUnwrittenPhotoelectricBindingEnergy_, deposit,
        accumulator.energyScale_, accumulator.counters_);
  }
  template <class AtomicOperations, bool AccumulateMoliereIterationCounters>
  CORSIKA_HOST_DEVICE inline void accumulate_lepton_profile_step(
      ProfileProjectionView const& projection, ProfileAccumulatorView const& accumulator,
      LeptonTransportRecord const& record) {
    using namespace corsika::kokkos;
    auto const native_cut = is_native_muon_pair_cut(record.start_);
    if (native_cut && (record.limit_ != LeptonTransportLimit::ParticleCut ||
                       record.continuousDepositedEnergyGeV_ != 0. ||
                       !is_native_muon_pair_cut_deposit(
                           record.start_.pid_, record.start_.reserved_,
                           record.end_.energyGeV_, record.cutDepositedEnergyGeV_))) {
      AtomicOperations::add(&accumulator.counters_->invalidRecords_, 1ULL);
      return;
    }
    auto const start = project_profile_grammage(projection, record.start_.positionM_);
    auto const end = project_profile_grammage(projection, record.end_.positionM_);
    accumulate_particle_profile<AtomicOperations>(accumulator, record.start_.pid_, start,
                                                  end, record.start_.weight_);
    auto* const energy_loss = is_muon_pid(record.start_.pid_)
                                  ? accumulator.muonEnergyLoss_
                                  : accumulator.energyLoss_;
    accumulate_energy_profile<AtomicOperations>(
        accumulator, energy_loss, start, end,
        record.continuousDepositedEnergyGeV_ * record.start_.weight_);
    accumulate_point_energy_profile<AtomicOperations>(
        accumulator, energy_loss, end,
        record.cutDepositedEnergyGeV_ * record.start_.weight_, native_cut);
    if (native_cut)
      add_profile_fixed_point<AtomicOperations>(
          &accumulator.counters_->weightedNativeMuonPairCutEnergy_,
          record.cutDepositedEnergyGeV_ * record.start_.weight_, accumulator.energyScale_,
          accumulator.counters_);
    AtomicOperations::add(&accumulator.counters_->steps_, 1ULL);
    auto const limit = static_cast<std::int32_t>(record.limit_);
    if (record.limit_ == LeptonTransportLimit::MaterialBoundary)
      AtomicOperations::add(&accumulator.counters_->leptonMaterialBoundaries_, 1ULL);
    else if (limit < 0 || limit >= 8)
      AtomicOperations::add(&accumulator.counters_->invalidRecords_, 1ULL);
    else
      AtomicOperations::add(&accumulator.counters_->leptonLimits_[limit], 1ULL);
    if (record.limit_ == LeptonTransportLimit::ParticleCut) {
      if (record.interaction_.status_ != InteractionStatus::AtRestAnnihilation)
        add_profile_fixed_point<AtomicOperations>(
            &accumulator.counters_->weightedCutRestMassEnergy_,
            transport_mass_gev(record.start_.pid_) * record.start_.weight_,
            accumulator.energyScale_, accumulator.counters_);
      if (record.observationSurfaceReachedBeforeCut_ != 0U) {
        add_profile_fixed_point<AtomicOperations>(
            &accumulator.counters_->weightedObservationCutOverlapEnergy_,
            (record.cutDepositedEnergyGeV_ + transport_mass_gev(record.start_.pid_)) *
                record.start_.weight_,
            accumulator.energyScale_, accumulator.counters_);
        accumulate_terminal_energy<AtomicOperations>(
            accumulator, record.end_.energyGeV_ * record.start_.weight_, true);
      }
    } else if (record.limit_ == LeptonTransportLimit::ObservationSurface) {
      accumulate_terminal_energy<AtomicOperations>(
          accumulator, record.end_.energyGeV_ * record.start_.weight_, true);
    } else if (record.limit_ == LeptonTransportLimit::EscapedEnvironment) {
      accumulate_terminal_energy<AtomicOperations>(
          accumulator, record.end_.energyGeV_ * record.start_.weight_, false);
    }
    AtomicOperations::add(&accumulator.counters_->moliereTrials_, 1ULL);
    if constexpr (AccumulateMoliereIterationCounters) {
      AtomicOperations::add(
          &accumulator.counters_->moliereNewtonIterations_,
          static_cast<unsigned long long>(record.multipleScatteringIterations_));
      AtomicOperations::maximum(
          &accumulator.counters_->moliereMaxNewtonIterations_,
          static_cast<unsigned long long>(record.multipleScatteringIterations_));
    }
    if (record.multipleScatteringApplied_ != 0)
      AtomicOperations::add(&accumulator.counters_->moliereDeflections_, 1ULL);
    else if (record.multipleScatteringStatus_ ==
             static_cast<std::uint32_t>(MoliereStatus::NoDeflection))
      AtomicOperations::add(&accumulator.counters_->moliereZeroDeflections_, 1ULL);
  }
  template <class AtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_lepton_profile_final_state(
      ProfileProjectionView const& projection, ProfileAccumulatorView const& accumulator,
      LeptonTransportRecord const* transport_records, std::size_t transport_count,
      LeptonFinalStateRecord const& record) {
    using namespace corsika::kokkos;
    accumulate_profile_thinning<AtomicOperations>(record, accumulator.counters_);
    add_profile_fixed_point<AtomicOperations>(
        &accumulator.counters_->weightedMassConventionCorrection_,
        record.weightedMassConventionCorrectionGeV_, accumulator.energyScale_,
        accumulator.counters_);
    auto const atomic_electron_process = record.processId_ == IonizationProcessId ||
                                         record.processId_ == AnnihilationProcessId;
    if (!atomic_electron_process) return;
    auto const* transport = find_profile_transport_record(
        transport_records, transport_count, record.inputIndex_);
    if (transport == nullptr || transport->start_.historyId_ != record.parentHistoryId_) {
      AtomicOperations::add(&accumulator.counters_->invalidRecords_, 1ULL);
      return;
    }
    if (record.processId_ == IonizationProcessId && is_muon_pid(transport->start_.pid_)) {
      auto const vertex_grammage =
          project_profile_grammage(projection, transport->end_.positionM_);
      accumulate_muon_parent_production<AtomicOperations>(accumulator, vertex_grammage,
                                                          transport->start_.weight_);
    }
    add_profile_fixed_point<AtomicOperations>(
        &accumulator.counters_->weightedMediumRestMassInput_,
        (record.processId_ == AnnihilationProcessId &&
                 record.finalStateDrawId_ == AtRestAnnihilationPolarDrawId
             ? TransportElectronMassGeV
             : ElectronMassGeV) *
            transport->start_.weight_,
        accumulator.energyScale_, accumulator.counters_);
  }

} // namespace corsika::kokkos
