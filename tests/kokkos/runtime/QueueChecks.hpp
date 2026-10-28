/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <catch2/catch_test_macros.hpp>
#include <corsika/kokkos/data/ParticleState.hpp>
#include <tests/kokkos/runtime/QueueReference.hpp>
#include <vector>
namespace queue_checks {
  inline std::vector<queue_reference::EmParticleState> original(
      std::vector<corsika::kokkos::ParticleState> const& source) {
    std::vector<queue_reference::EmParticleState> result(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
      result[i].pid = source[i].pid_;
      result[i].medium_id = source[i].mediumId_;
      result[i].environment_node = source[i].environmentNode_;
      result[i].generation = source[i].generation_;
      result[i].reserved = source[i].reserved_;
      result[i].energy_GeV = source[i].energyGeV_;
      result[i].time_s = source[i].timeS_;
      result[i].weight = source[i].weight_;
      result[i].history_id = source[i].historyId_;
      result[i].parent_history_id = source[i].parentHistoryId_;
      result[i].step_id = source[i].stepId_;
      for (int k = 0; k < 3; ++k) result[i].position_m[k] = source[i].positionM_[k];
      for (int k = 0; k < 3; ++k) result[i].direction[k] = source[i].direction_[k];
    }
    return result;
  }
  inline void equal(std::vector<corsika::kokkos::ParticleState> const& current,
                    std::vector<queue_reference::EmParticleState> const& original,
                    std::vector<corsika::kokkos::ParticleState> const& expected) {
    REQUIRE(current.size() == expected.size());
    REQUIRE(original.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
      REQUIRE(current[i].pid_ == expected[i].pid_);
      REQUIRE(original[i].pid == expected[i].pid_);
      REQUIRE(current[i].mediumId_ == expected[i].mediumId_);
      REQUIRE(original[i].medium_id == expected[i].mediumId_);
      REQUIRE(current[i].environmentNode_ == expected[i].environmentNode_);
      REQUIRE(original[i].environment_node == expected[i].environmentNode_);
      REQUIRE(current[i].generation_ == expected[i].generation_);
      REQUIRE(original[i].generation == expected[i].generation_);
      REQUIRE(current[i].reserved_ == expected[i].reserved_);
      REQUIRE(original[i].reserved == expected[i].reserved_);
      REQUIRE(current[i].energyGeV_ == expected[i].energyGeV_);
      REQUIRE(original[i].energy_GeV == expected[i].energyGeV_);
      REQUIRE(current[i].timeS_ == expected[i].timeS_);
      REQUIRE(original[i].time_s == expected[i].timeS_);
      REQUIRE(current[i].weight_ == expected[i].weight_);
      REQUIRE(original[i].weight == expected[i].weight_);
      REQUIRE(current[i].historyId_ == expected[i].historyId_);
      REQUIRE(original[i].history_id == expected[i].historyId_);
      REQUIRE(current[i].parentHistoryId_ == expected[i].parentHistoryId_);
      REQUIRE(original[i].parent_history_id == expected[i].parentHistoryId_);
      REQUIRE(current[i].stepId_ == expected[i].stepId_);
      REQUIRE(original[i].step_id == expected[i].stepId_);
      for (int k = 0; k < 3; ++k) {
        REQUIRE(current[i].positionM_[k] == expected[i].positionM_[k]);
        REQUIRE(original[i].position_m[k] == expected[i].positionM_[k]);
      }
      for (int k = 0; k < 3; ++k) {
        REQUIRE(current[i].direction_[k] == expected[i].direction_[k]);
        REQUIRE(original[i].direction[k] == expected[i].direction_[k]);
      }
    }
  }
} // namespace queue_checks
