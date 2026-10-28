/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonTransportStatisticsFunctor<TExecutionSpace>::init(
      value_type& value) const {
    value.materialBoundaries_ = 0;
    for (std::size_t index = 0; index < LeptonTransportStatisticCount; ++index)
      value.values_[index] = 0;
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonTransportStatisticsFunctor<TExecutionSpace>::join(
      value_type& destination, value_type const& source) const {
    destination.materialBoundaries_ += source.materialBoundaries_;
    for (std::size_t index = 0; index < LeptonTransportStatisticCount; ++index) {
      if (index == LeptonTransportMoliereMaximumNewtonIterations)
        destination.values_[index] = destination.values_[index] > source.values_[index]
                                         ? destination.values_[index]
                                         : source.values_[index];
      else
        destination.values_[index] += source.values_[index];
    }
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void ResidentLeptonTransportStatisticsFunctor<TExecutionSpace>::
  operator()(std::size_t const source, value_type& value) const {

    if (counts_(source).step_ != 0) {
      auto const& step = transports_(source);
      value.values_[LeptonTransportInteractionVertices] +=
          step.limit_ == LeptonTransportLimit::InteractionCandidate;
      auto const limitIndex = static_cast<std::size_t>(step.limit_);
      if (limitIndex < 8) ++value.values_[LeptonTransportLimitBase + limitIndex];
      if (step.limit_ == LeptonTransportLimit::MaterialBoundary)
        ++value.materialBoundaries_;
      // Native CUDA records one Moliere trial for every completed lepton
      // transport step for which multiple scattering is enabled, including
      // zero-grammage boundary/observation steps.  The Kokkos resident path
      // invokes the same stage for every counts(source).step record, so use
      // that exact event definition instead of conditioning the metadata on
      // traversed grammage.
      ++value.values_[LeptonTransportMoliereTrials];
      value.values_[LeptonTransportMoliereDeflections] +=
          step.multipleScatteringApplied_ != 0;
      value.values_[LeptonTransportMoliereZeroDeflections] +=
          step.multipleScatteringStatus_ ==
          static_cast<std::uint16_t>(MoliereStatus::NoDeflection);
      value.values_[LeptonTransportMoliereNewtonIterations] +=
          step.multipleScatteringIterations_;
      auto const iterations =
          static_cast<std::uint64_t>(step.multipleScatteringIterations_);
      if (iterations > value.values_[LeptonTransportMoliereMaximumNewtonIterations])
        value.values_[LeptonTransportMoliereMaximumNewtonIterations] = iterations;
    }
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonTransportStatisticsFunctor<TExecutionSpace>::final(
      value_type& value) const {
    callStatistics_(LeptonCallMaterialBoundaries) += value.materialBoundaries_;
    for (std::size_t index = 0; index < LeptonTransportStatisticCount; ++index) {
      auto& accumulated = callStatistics_(LeptonCallTransportStatisticBase + index);
      if (index == LeptonTransportMoliereMaximumNewtonIterations)
        accumulated =
            accumulated > value.values_[index] ? accumulated : value.values_[index];
      else
        accumulated += value.values_[index];
    }
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonTransportStatisticsProfileFunctor<TExecutionSpace>::init(
      value_type& value) const {
    statistics_.init(value);
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonTransportStatisticsProfileFunctor<TExecutionSpace>::join(
      value_type& destination, value_type const& source) const {
    statistics_.join(destination, source);
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonTransportStatisticsProfileFunctor<TExecutionSpace>::final(
      value_type& value) const {
    statistics_.final(value);
    // The reduction has already computed these two integer diagnostics for
    // exactly the same stable packed step ledger.  Commit one aggregate per
    // wavefront instead of issuing two contended global atomics per lepton.
    ProfileAtomicOperations::add(&accumulator_.counters_->moliereNewtonIterations_,
                                 value.values_[LeptonTransportMoliereNewtonIterations]);
    ProfileAtomicOperations::maximum(
        &accumulator_.counters_->moliereMaxNewtonIterations_,
        value.values_[LeptonTransportMoliereMaximumNewtonIterations]);
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonTransportStatisticsProfileFunctor<TExecutionSpace>::operator()(
      std::size_t const source, value_type& value) const {
    statistics_(source, value);
    if (source < stepCount_)
      accumulate_lepton_profile_step<ProfileAtomicOperations, false>(
          projection_, accumulator_, steps_(source));
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonInteractionStatisticsFunctor<TExecutionSpace>::init(
      value_type& value) const {
    for (std::size_t index = 0; index < LeptonInteractionStatisticCount; ++index)
      value.values_[index] = 0;
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonInteractionStatisticsFunctor<TExecutionSpace>::join(
      value_type& destination, value_type const& source) const {
    for (std::size_t index = 0; index < LeptonInteractionStatisticCount; ++index)
      destination.values_[index] += source.values_[index];
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonInteractionStatisticsFunctor<TExecutionSpace>::operator()(
      std::size_t const candidate, value_type& value) const {

    // Vertex and final-state storage is intentionally left untouched by
    // the transport stage.  It is valid only for compacted candidates that
    // reached a selected physical interaction.
    if (vertices_(candidate).interactionFlag_ == 0) return;
    auto const& final = finalStates_(candidate);
    value.values_[LeptonInteractionLpmSuppressions] += final.suppressionFlag_;
    value.values_[LeptonInteractionGpuFinalStates] += final.recordFlag_;
    if (final.recordFlag_ != 0) {
      value.values_[LeptonInteractionPhysicalSecondaries] += final.childCount_;
      value.values_[LeptonInteractionBremsFinalStates] += final.bremsFlag_;
      value.values_[LeptonInteractionAnnihilationFinalStates] += final.annihilationFlag_;
      value.values_[LeptonInteractionIonizationFinalStates] += final.ionizationFlag_;
      value.values_[LeptonInteractionElectronPairFinalStates] += final.electronPairFlag_;
      auto const status =
          static_cast<EmThinningStatus>(final.parameters_.thinningStatus_);
      value.values_[LeptonInteractionThinningHillasVertices] +=
          status == EmThinningStatus::Hillas;
      value.values_[LeptonInteractionThinningStatisticalVertices] +=
          status == EmThinningStatus::Statistical;
      auto const originalMultiplicity =
          final.parameters_.processId_ == ElectronPairProcessId ? 3U : 2U;
      if (final.childCount_ <= originalMultiplicity)
        value.values_[LeptonInteractionThinningParticlesDiscarded] +=
            originalMultiplicity - final.childCount_;
    }
    value.values_[LeptonInteractionBremsLpmTrials] +=
        final.bremsFlag_ + final.bremsSuppressionFlag_;
    value.values_[LeptonInteractionBremsLpmSuppressions] += final.bremsSuppressionFlag_;
    value.values_[LeptonInteractionElectronPairLpmTrials] +=
        final.electronPairFlag_ + final.electronPairSuppressionFlag_;
    value.values_[LeptonInteractionElectronPairLpmSuppressions] +=
        final.electronPairSuppressionFlag_;
    value.values_[LeptonInteractionElectronPairRejectionTrials] +=
        final.electronPairRejectionTrials_;
    value.values_[LeptonInteractionElectronPairZeroWeightSamples] +=
        final.electronPairZeroWeightFlag_;
    value.values_[LeptonInteractionElectronPairRejectionFallbacks] +=
        final.electronPairRejectionFallbackFlag_;
    value.values_[LeptonInteractionElectronPairEnvelopeViolations] +=
        final.electronPairEnvelopeViolationFlag_;
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonInteractionStatisticsFunctor<TExecutionSpace>::final(
      value_type& value) const {
    for (std::size_t index = 0; index < LeptonInteractionStatisticCount; ++index)
      callStatistics_(LeptonCallInteractionStatisticBase + index) += value.values_[index];
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonInteractionStatisticsProfileFunctor<TExecutionSpace>::init(
      value_type& value) const {
    statistics_.init(value);
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonInteractionStatisticsProfileFunctor<TExecutionSpace>::join(
      value_type& destination, value_type const& source) const {
    statistics_.join(destination, source);
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonInteractionStatisticsProfileFunctor<TExecutionSpace>::final(
      value_type& value) const {
    statistics_.final(value);
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void
  ResidentLeptonInteractionStatisticsProfileFunctor<TExecutionSpace>::operator()(
      std::size_t const interactionIndex, value_type& value) const {
    statistics_(interactionIndex, value);
    if (interactionIndex < recordCount_)
      accumulate_lepton_profile_final_state<ProfileAtomicOperations>(
          projection_, accumulator_, steps_.data(), stepCount_,
          records_(interactionIndex));
  }

#ifdef KOKKOS_ENABLE_OPENMP
  inline void HostShardedLeptonProfileFunctor::operator()(std::size_t const source,
                                                          value_type& value) const {
    statistics_(source, value);
    if (source < stepCount_) {
      auto const rank = static_cast<std::size_t>(Kokkos::OpenMP::impl_thread_pool_rank());
      auto const shard = rank < shardCount_ ? rank : rank % shardCount_;
      accumulate_lepton_profile_step<ProfileAtomicOperations, false>(
          projection_, shards_[shard].accumulator_, steps_(source));
    }
  }
#endif

  template <class TExecutionSpace, class TPolicy>
  void enqueue_lepton_profile_statistics(
      TPolicy const& policy,
      ResidentLeptonTransportStatisticsProfileFunctor<TExecutionSpace> const& functor,
      [[maybe_unused]] HostProfileShards const& shards) {
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<TExecutionSpace, Kokkos::OpenMP>) {
      if (shards.enabled()) {
        Kokkos::parallel_reduce(
            "c8_kokkos_resident_lepton_transport_statistics_host_sharded", policy,
            HostShardedLeptonProfileFunctor{functor, shards.entries(), shards.count()});
        return;
      }
    }
#endif
    Kokkos::parallel_reduce("c8_kokkos_resident_lepton_transport_statistics", policy,
                            functor);
  }

} // namespace corsika::kokkos
