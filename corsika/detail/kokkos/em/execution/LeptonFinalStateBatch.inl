/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  template <class TExecutionSpace>
  LeptonFinalStateExecutionResult generate_lepton_final_states_batch(
      BremsLpmSnapshot const lpmSnapshot, EmThinningConfig const thinning,
      std::vector<InteractionRecord> const& input, std::uint64_t const randomSeed,
      std::uint64_t const showerId, std::uint64_t const firstSecondaryHistoryId,
      TExecutionSpace const& execution) {
    using memory_space_type = typename TExecutionSpace::memory_space;
    using policy_type = Kokkos::RangePolicy<TExecutionSpace>;
    using classification_type = LeptonFinalStateClassification;
    using materialization_type = LeptonFinalStateMaterialization;
    LeptonFinalStateExecutionResult result{};
    result.batch_.inputInteractions_ = input.size();
    if (input.empty()) return result;
    if (firstSecondaryHistoryId == 0) {
      throw std::invalid_argument(
          "Kokkos lepton secondary history IDs must start above zero");
    }
    if (input.size() > std::numeric_limits<std::uint32_t>::max() / 3) {
      throw std::length_error("Kokkos lepton final-state batch exceeds 32-bit offsets");
    }

    auto const count = input.size();
    Kokkos::View<InteractionRecord*, memory_space_type> interactions(
        "c8_kokkos_lepton_final_state_input", count);
    Kokkos::View<classification_type*, memory_space_type> classifications(
        "c8_kokkos_lepton_final_state_classifications", count);
    Kokkos::View<std::uint32_t * [5], memory_space_type> offsets(
        "c8_kokkos_lepton_final_state_offsets", count);
    Kokkos::View<materialization_type*, memory_space_type> materialized(
        "c8_kokkos_lepton_final_state_materialized", count);
    Kokkos::View<LeptonFinalStateRecord*, memory_space_type> records(
        "c8_kokkos_lepton_final_state_records", count);
    Kokkos::View<ParticleState*, memory_space_type> secondaries(
        "c8_kokkos_lepton_final_state_secondaries", 3 * count);
    Kokkos::View<FallbackRequest*, memory_space_type> fallbacks(
        "c8_kokkos_lepton_final_state_fallbacks", count);
    Kokkos::View<InteractionRecord*, memory_space_type> continuations(
        "c8_kokkos_lepton_final_state_continuations", count);
    Kokkos::View<BremsLpmSuppressionRecord*, memory_space_type> suppressions(
        "c8_kokkos_lepton_final_state_suppressions", count);
    Kokkos::View<std::uint32_t*, memory_space_type> error(
        "c8_kokkos_lepton_final_state_error", 1);
    Kokkos::deep_copy(execution, error, std::uint32_t{0});

    auto hostInput = Kokkos::create_mirror_view(interactions);
    for (std::size_t i = 0; i < count; ++i) hostInput(i) = input[i];
    Kokkos::deep_copy(execution, interactions, hostInput);
    Kokkos::parallel_for(
        "c8_kokkos_classify_lepton_final_states", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          classifications(index) = classify_lepton_final_state(
              lpmSnapshot, thinning, interactions(index), randomSeed, showerId);
        });

    std::uint32_t totals[5]{};
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_children", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update, bool const final) {
          if (final) offsets(index, 0) = update;
          update += classifications(index).childCount_;
        },
        totals[0]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_records", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update, bool const final) {
          if (final) offsets(index, 1) = update;
          update += classifications(index).recordFlag_;
        },
        totals[1]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_fallbacks", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update, bool const final) {
          if (final) offsets(index, 2) = update;
          update += classifications(index).fallbackFlag_;
        },
        totals[2]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_continuations", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update, bool const final) {
          if (final) offsets(index, 3) = update;
          update += classifications(index).continuationFlag_;
        },
        totals[3]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_suppressions", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update, bool const final) {
          if (final) offsets(index, 4) = update;
          update += classifications(index).suppressionFlag_;
        },
        totals[4]);

    auto const electronMassGeV = lpmSnapshot.leptonMassMeV_ / 1000.;
    Kokkos::parallel_for(
        "c8_kokkos_materialize_lepton_final_states", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const classification = classifications(index);
          auto const output = materialize_lepton_final_state(
              interactions(index), classification, offsets(index, 0),
              firstSecondaryHistoryId, electronMassGeV);
          materialized(index) = output;
          if (output.error_ != 0) {
            Kokkos::atomic_compare_exchange(error.data(), std::uint32_t{0},
                                            output.error_);
            return;
          }
          if (output.hasRecord_) {
            records(offsets(index, 1)) = output.record_;
            for (std::uint32_t child = 0; child < output.secondaryCount_; ++child)
              secondaries(offsets(index, 0) + child) = output.secondaries_[child];
          } else if (output.hasFallback_) {
            fallbacks(offsets(index, 2)) = output.fallback_;
          } else if (output.hasContinuation_) {
            continuations(offsets(index, 3)) = output.continuation_;
          } else if (output.hasSuppression_) {
            suppressions(offsets(index, 4)) = output.suppression_;
          }
        });
    execution.fence("complete Kokkos lepton final states");

    auto hostError = Kokkos::create_mirror_view(error);
    Kokkos::deep_copy(execution, hostError, error);
    execution.fence("download Kokkos lepton final-state status");
    if (hostError(0) != 0)
      throw std::runtime_error("Kokkos lepton final-state materialization failed");
    if (totals[1] + totals[2] + totals[3] + totals[4] != count)
      throw std::runtime_error("Kokkos lepton final-state partition is not exhaustive");

    auto hostRecords = Kokkos::create_mirror_view(records);
    auto hostSecondaries = Kokkos::create_mirror_view(secondaries);
    auto hostFallbacks = Kokkos::create_mirror_view(fallbacks);
    auto hostContinuations = Kokkos::create_mirror_view(continuations);
    auto hostSuppressions = Kokkos::create_mirror_view(suppressions);
    auto hostClassifications = Kokkos::create_mirror_view(classifications);
    auto hostMaterialized = Kokkos::create_mirror_view(materialized);
    Kokkos::deep_copy(execution, hostRecords, records);
    Kokkos::deep_copy(execution, hostSecondaries, secondaries);
    Kokkos::deep_copy(execution, hostFallbacks, fallbacks);
    Kokkos::deep_copy(execution, hostContinuations, continuations);
    Kokkos::deep_copy(execution, hostSuppressions, suppressions);
    Kokkos::deep_copy(execution, hostClassifications, classifications);
    Kokkos::deep_copy(execution, hostMaterialized, materialized);
    execution.fence("download Kokkos lepton final states");

    result.batch_.gpuInteractions_ = totals[1];
    result.batch_.finalStateRecords_.resize(totals[1]);
    result.batch_.secondaries_.resize(totals[0]);
    result.batch_.fallbackEvents_.resize(totals[2]);
    result.batch_.continuations_.resize(totals[3]);
    result.batch_.lpmSuppressed_.resize(totals[4]);
    for (std::size_t i = 0; i < totals[1]; ++i)
      result.batch_.finalStateRecords_[i] = hostRecords(i);
    for (std::size_t i = 0; i < totals[0]; ++i)
      result.batch_.secondaries_[i] = hostSecondaries(i);
    for (std::size_t i = 0; i < totals[2]; ++i)
      result.batch_.fallbackEvents_[i] = hostFallbacks(i);
    for (std::size_t i = 0; i < totals[3]; ++i)
      result.batch_.continuations_[i] = hostContinuations(i);
    for (std::size_t i = 0; i < totals[4]; ++i)
      result.batch_.lpmSuppressed_[i] = hostSuppressions(i);
    for (std::size_t i = 0; i < count; ++i) {
      auto const& classification = hostClassifications(i);
      result.batch_.bremsInteractions_ += classification.bremsFlag_;
      result.batch_.annihilationInteractions_ += classification.annihilationFlag_;
      result.batch_.ionizationInteractions_ += classification.ionizationFlag_;
      result.batch_.electronPairInteractions_ += classification.electronPairFlag_;
      result.batch_.bremsLpmTrials_ +=
          classification.bremsFlag_ + classification.bremsSuppressionFlag_;
      result.batch_.bremsLpmSuppressions_ += classification.bremsSuppressionFlag_;
      result.batch_.electronPairLpmTrials_ +=
          classification.electronPairFlag_ + classification.electronPairSuppressionFlag_;
      result.batch_.electronPairLpmSuppressions_ +=
          classification.electronPairSuppressionFlag_;
      result.batch_.electronPairRejectionTrials_ +=
          classification.electronPairRejectionTrials_;
      result.batch_.electronPairZeroWeightSamples_ +=
          classification.electronPairZeroWeightFlag_;
      result.batch_.electronPairRejectionFallbacks_ +=
          classification.electronPairRejectionFallbackFlag_;
      result.batch_.electronPairEnvelopeViolations_ +=
          classification.electronPairEnvelopeViolationFlag_;
      auto const& output = hostMaterialized(i);
      if (!result.firstInteraction_ && output.hasFirstInteraction_)
        result.firstInteraction_ = output.firstInteraction_;
    }
    return result;
  }
} // namespace corsika::kokkos
