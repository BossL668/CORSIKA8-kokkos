/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include "StochasticReference.hpp"
#include "../continuous/ContinuousAdapters.hpp"
#include <corsika/kokkos/em/ProcessCapabilities.hpp>
#include <corsika/kokkos/em/random/Philox.hpp>
#include <corsika/kokkos/em/random/RandomDomains.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <catch2/catch_all.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace stochastic_test {
  namespace current = corsika::kokkos;
  namespace reference = stochastic_reference;
  using continuous_test::append;
  using continuous_test::Outcome;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
  inline void check(Outcome const& actual, Outcome const& expected) {
    REQUIRE(actual.valueCount_ <= 96);
    REQUIRE(actual.stateCount_ <= 128);
    REQUIRE(actual.valueCount_ == expected.valueCount_);
    REQUIRE(actual.stateCount_ == expected.stateCount_);
    auto compare_value = [](double a, double b) {
      if (std::isnan(b)) {
        REQUIRE(std::isnan(a));
      } else {
        REQUIRE(a == b);
        if (b == 0.) REQUIRE(std::signbit(a) == std::signbit(b));
      }
    };
    for (std::uint32_t index = 0; index < actual.valueCount_; ++index)
      compare_value(actual.values_[index], expected.values_[index]);
    for (std::uint32_t index = 0; index < actual.stateCount_; ++index)
      REQUIRE(actual.states_[index] == expected.states_[index]);
  }
  template <typename T>
  auto upload(std::vector<T> const& data) {
    Kokkos::View<T*, memory_space_type> result("stochastic_input", data.size());
    auto mirror = Kokkos::create_mirror_view(result);
    for (std::size_t index = 0; index < data.size(); ++index) mirror(index) = data[index];
    Kokkos::deep_copy(result, mirror);
    return result;
  }
  struct PolicyQuery {
    std::int32_t pid_{}, process_{}, reason_{}, status_{};
    double quantile_{}, time_{};
  };
  CORSIKA_HOST_DEVICE inline Outcome policy(PolicyQuery const& input) {
    Outcome result;
    auto const capability = current::process_capability(input.pid_, input.process_);
    append(result, capability);
    append(result, current::process_fallback_reason(capability));
    append(result, current::proposal_native_selection_requires_replay(
                       input.pid_, input.process_, input.quantile_));
    append(result, current::is_electron_or_positron_pid(input.pid_));
    append(result, current::is_muon_pid(input.pid_));
    append(result, current::is_charged_lepton_pid(input.pid_));
    append(result, current::exceeds_particle_cut_time(input.time_));
    auto const reason = static_cast<current::FallbackReason>(input.reason_);
    append(result, current::proposal_fallback_requires_selected_loss(reason));
    append(result, current::proposal_fallback_requires_interaction_vertex(reason));
    current::ParticleState particle{};
    particle.pid_ = input.pid_;
    particle.mediumId_ = 7;
    particle.generation_ = 3;
    particle.reserved_ = 1;
    particle.energyGeV_ = 6.;
    particle.positionM_[0] = -7.;
    particle.direction_[2] = 1.;
    particle.timeS_ = input.time_;
    particle.weight_ = 2.;
    particle.historyId_ = 0xfedcba9876543210ULL;
    particle.parentHistoryId_ = 17;
    particle.stepId_ = 0x98765432ffffffffULL;
    particle.environmentNode_ = 13;
    current::TableQuery query{current::TableQueryKind::LossFraction,
                              input.pid_,
                              input.process_,
                              0x9876543210abcdefULL,
                              100.,
                              input.quantile_};
    current::TableQueryResult value{
        static_cast<current::TableLookupStatus>(input.status_), 0x12345678U, -.25};
    append(result,
           current::make_table_fallback_event(
               particle, query, value, 0xabcdef0123456789ULL, 0x8888888888888888ULL));
    return result;
  }
  CORSIKA_HOST_DEVICE inline Outcome policy_reference(PolicyQuery const& input) {
    Outcome result;
    auto const capability = reference::gpuProcessCapability(input.pid_, input.process_);
    append(result, capability);
    append(result, reference::processFallbackReason(capability));
    append(result, reference::proposalNativeSelectionRequiresReplay(
                       input.pid_, input.process_, input.quantile_));
    append(result, reference::isElectronOrPositronPid(input.pid_));
    append(result, reference::isMuonPid(input.pid_));
    append(result, reference::isChargedLeptonPid(input.pid_));
    append(result, reference::exceedsParticleCutTime(input.time_));
    auto const reason = static_cast<reference::ProposalFallbackReason>(input.reason_);
    append(result, reference::proposalFallbackRequiresSelectedLoss(reason));
    append(result, reference::proposalFallbackRequiresInteractionVertex(reason));
    reference::EmParticleState particle{};
    particle.pid = input.pid_;
    particle.medium_id = 7;
    particle.generation = 3;
    particle.reserved = 1;
    particle.energy_GeV = 6.;
    particle.position_m[0] = -7.;
    particle.direction[2] = 1.;
    particle.time_s = input.time_;
    particle.weight = 2.;
    particle.history_id = 0xfedcba9876543210ULL;
    particle.parent_history_id = 17;
    particle.step_id = 0x98765432ffffffffULL;
    particle.environment_node = 13;
    reference::tables::TableQuery query{reference::tables::TableQueryKind::LossFraction,
                                        input.pid_,
                                        input.process_,
                                        0x9876543210abcdefULL,
                                        100.,
                                        input.quantile_};
    reference::tables::TableQueryResult value{
        static_cast<reference::tables::TableLookupStatus>(input.status_), 0x12345678U,
        -.25};
    append(result,
           reference::makeTableFallbackEvent(
               particle, query, value, 0xabcdef0123456789ULL, 0x8888888888888888ULL));
    return result;
  }
  CORSIKA_HOST_DEVICE inline Outcome random_result(current::RandomNumberKey const& key) {
    Outcome result;
    auto words = current::random_words(key);
    for (auto word : words.words_) append(result, word);
    for (std::uint32_t lane = 0; lane < 8; ++lane)
      append(result, current::uniform_open01(key, lane));
    append(result, current::uniform_open01(key, 0xffffffffU));
    return result;
  }
  CORSIKA_HOST_DEVICE inline Outcome random_reference(
      current::RandomNumberKey const& key) {
    Outcome result;
    reference::RandomNumberKey old{key.seed_,   key.showerId_,  key.historyId_,
                                   key.stepId_, key.processId_, key.drawId_};
    auto words = reference::randomWords(old);
    for (auto word : words.words) append(result, word);
    for (std::uint32_t lane = 0; lane < 8; ++lane)
      append(result, reference::uniformOpen01(old, lane));
    append(result, reference::uniformOpen01(old, 0xffffffffU));
    return result;
  }
} // namespace stochastic_test
