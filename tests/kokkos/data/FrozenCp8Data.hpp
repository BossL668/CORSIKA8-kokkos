/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

// Test-only declarations copied verbatim from the Git object, never from the
// reference working tree. Keep their legacy spelling and initializers intact.
// Commit: 7c24233bead442c136361dfb02b23450b17efbe6
// Path: corsika/accelerator/em/common/Types.hpp
// Blob: 6849cdf2d882aec795b1180e4e2aa1d3dc4a1e10
// Full-source SHA-256:
// b305f0dc85a4bff14985db2068ca6c17c4c7c5d76a2e8410d1d5631c53af4df4
// The namespace and minimal include above are fixture scaffolding.
namespace cp8_frozen {

  struct alignas(16) EmParticleState {
    std::int32_t pid{};
    std::int32_t medium_id{};
    std::uint32_t generation{};
    std::uint32_t reserved{};

    double energy_GeV{};
    double position_m[3]{};
    double direction[3]{};
    double time_s{};
    double weight{1.};

    std::uint64_t history_id{};
    std::uint64_t parent_history_id{};
    std::uint64_t step_id{};
    // Independent of medium_id: distinct logical nodes may share one model.
    // -3 denotes legacy/unassigned, -1 denotes outside the exported tree.
    std::int32_t environment_node{-3};
  };

  struct EmStepRecord {
    std::uint64_t history_id{};
    std::uint64_t step_id{};
    std::int32_t pid{};
    std::int32_t process_id{};
    double start_position_m[3]{};
    double end_position_m[3]{};
    double start_time_s{};
    double end_time_s{};
    double start_energy_GeV{};
    double end_energy_GeV{};
    double deposited_energy_GeV{};
    double weight{};
    // Included in deposited_energy_GeV, but written at the endpoint (ParticleCut),
    // not distributed along the preceding continuous-transport segment.
    double cut_deposited_energy_GeV{};
    std::uint32_t native_muon_pair_cut{};
  };

} // namespace cp8_frozen
