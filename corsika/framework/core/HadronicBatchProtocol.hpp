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
#include <stdexcept>
#include <type_traits>

namespace corsika {

  /**
   * Versioned wire records for a process-isolated hadronic final-state worker.
   *
   * FLUKA, SIBYLL, and similar Fortran generators own process-global COMMON
   * blocks.  They must therefore run in independent processes rather than in
   * threads sharing one address space.  These records deliberately contain no
   * CORSIKA pointers, units classes, stack iterators, or coordinate-system
   * objects.  All momenta use GeV and all directions are dimensionless.
   *
   * The first implementation targets local workers built from the same source
   * tree.  The endian marker and protocol version make ABI mismatches fail
   * explicitly instead of silently corrupting a shower.
   */
  inline constexpr std::uint32_t HadronicBatchMagic = 0x43384842U; // "C8HB"
  // Version 3 writes a response as three contiguous blocks: batch header,
  // interaction-result headers, and flattened secondaries.  This replaces the
  // version-2 header/secondaries alternation for every individual interaction.
  inline constexpr std::uint32_t HadronicBatchProtocolVersion = 3U;
  inline constexpr std::uint32_t HadronicBatchEndianMarker = 0x01020304U;

  enum class HadronicWorkerModel : std::uint8_t {
    Fluka = 1,
    Sibyll = 2,
    QgsJetII = 3,
    EposLhc = 4,
    Pythia8 = 5
  };

  enum class HadronicWorkerStatus : std::uint32_t {
    Success = 0,
    InvalidProtocol = 1,
    InvalidRequest = 2,
    UnsupportedProjectile = 3,
    UnsupportedTarget = 4,
    GeneratorFailure = 5,
    WorkerFailure = 6
  };

  struct HadronicRandomKey {
    std::uint64_t seed{};
    std::uint64_t shower_id{};
    std::uint64_t history_id{};
    std::uint64_t step_id{};
    std::uint32_t process_id{};
    std::uint32_t draw_domain{};
  };

  struct HadronicBatchHeader {
    std::uint32_t magic{HadronicBatchMagic};
    std::uint32_t protocol_version{HadronicBatchProtocolVersion};
    std::uint32_t endian_marker{HadronicBatchEndianMarker};
    std::uint32_t request_count{};
    std::uint64_t batch_id{};
  };

  struct HadronicInteractionRequest {
    std::uint32_t protocol_version{HadronicBatchProtocolVersion};
    HadronicWorkerModel model{HadronicWorkerModel::Fluka};
    std::array<std::uint8_t, 3> reserved{};
    std::uint64_t sequence_id{};
    HadronicRandomKey random_key{};
    std::int32_t projectile_pdg{};
    std::int32_t target_pdg{};
    // (E, px, py, pz), all in GeV, in the request coordinate frame.
    std::array<double, 4> projectile_four_momentum_GeV{};
    std::array<double, 4> target_four_momentum_GeV{};
  };

  struct HadronicInteractionResultHeader {
    std::uint32_t protocol_version{HadronicBatchProtocolVersion};
    HadronicWorkerStatus status{HadronicWorkerStatus::Success};
    std::uint64_t sequence_id{};
    std::uint32_t secondary_count{};
    // FLUKA requests random values in buffered blocks. This is the number supplied
    // by the worker callback, including values prefetched into FLUKA's buffer.
    std::uint32_t random_values_supplied{};
    // Generator call only. IPC, parent scheduling, target selection, and main-stack
    // commit are measured separately by the parent process.
    double final_state_time_ms{};
  };

  struct HadronicSecondaryRecord {
    std::int32_t pdg{};
    std::uint32_t reserved{};
    double kinetic_energy_GeV{};
    std::array<double, 3> direction{};
  };

  /**
   * Stable SplitMix64 combination used to seed a worker-local Philox stream.
   *
   * The returned seed depends only on the physical history key, never on the
   * worker number or dispatch order.  Rebalancing batches can therefore not
   * change an individual interaction's random sequence.
   */
  inline std::uint64_t deriveHadronicWorkerSeed(
      HadronicRandomKey const& key) noexcept {
    auto mix = [](std::uint64_t value) noexcept {
      value += 0x9e3779b97f4a7c15ULL;
      value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
      value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31U);
    };

    auto state = mix(key.seed);
    state = mix(state ^ key.shower_id);
    state = mix(state ^ key.history_id);
    state = mix(state ^ key.step_id);
    state = mix(
        state ^
        (static_cast<std::uint64_t>(key.process_id) << 32U) ^
        static_cast<std::uint64_t>(key.draw_domain));
    return state;
  }

  inline void validateHadronicBatchHeader(
      HadronicBatchHeader const& header) {
    if (header.magic != HadronicBatchMagic) {
      throw std::invalid_argument("Invalid hadronic batch magic");
    }
    if (header.protocol_version != HadronicBatchProtocolVersion) {
      throw std::invalid_argument(
          "Unsupported hadronic batch protocol version");
    }
    if (header.endian_marker != HadronicBatchEndianMarker) {
      throw std::invalid_argument(
          "Hadronic batch worker endian mismatch");
    }
  }

  inline void validateHadronicInteractionRequest(
      HadronicInteractionRequest const& request) {
    if (request.protocol_version != HadronicBatchProtocolVersion) {
      throw std::invalid_argument(
          "Unsupported hadronic interaction request version");
    }
    if (request.projectile_pdg == 0 || request.target_pdg == 0) {
      throw std::invalid_argument(
          "Hadronic interaction request contains an invalid PDG code");
    }
    auto validateFourMomentum = [](std::array<double, 4> const& p4) {
      if (!std::all_of(
              p4.begin(), p4.end(),
              [](double const value) { return std::isfinite(value); })) {
        throw std::invalid_argument(
            "Hadronic interaction request contains a non-finite four-momentum");
      }
      if (!(p4[0] > 0.)) {
        throw std::invalid_argument(
            "Hadronic interaction request energy must be positive");
      }
    };
    validateFourMomentum(request.projectile_four_momentum_GeV);
    validateFourMomentum(request.target_four_momentum_GeV);
  }

  static_assert(
      std::is_standard_layout_v<HadronicBatchHeader> &&
      std::is_trivially_copyable_v<HadronicBatchHeader>);
  static_assert(
      std::is_standard_layout_v<HadronicInteractionRequest> &&
      std::is_trivially_copyable_v<HadronicInteractionRequest>);
  static_assert(
      std::is_standard_layout_v<HadronicInteractionResultHeader> &&
      std::is_trivially_copyable_v<HadronicInteractionResultHeader>);
  static_assert(
      std::is_standard_layout_v<HadronicSecondaryRecord> &&
      std::is_trivially_copyable_v<HadronicSecondaryRecord>);

} // namespace corsika
