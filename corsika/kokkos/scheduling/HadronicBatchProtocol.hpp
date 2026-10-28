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

namespace corsika::kokkos {

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
    std::uint64_t seed_{};
    std::uint64_t showerId_{};
    std::uint64_t historyId_{};
    std::uint64_t stepId_{};
    std::uint32_t processId_{};
    std::uint32_t drawDomain_{};
  };

  struct HadronicBatchHeader {
    std::uint32_t magic_{HadronicBatchMagic};
    std::uint32_t protocolVersion_{HadronicBatchProtocolVersion};
    std::uint32_t endianMarker_{HadronicBatchEndianMarker};
    std::uint32_t requestCount_{};
    std::uint64_t batchId_{};
  };

  struct HadronicInteractionRequest {
    std::uint32_t protocolVersion_{HadronicBatchProtocolVersion};
    HadronicWorkerModel model_{HadronicWorkerModel::Fluka};
    std::array<std::uint8_t, 3> reserved_{};
    std::uint64_t sequenceId_{};
    HadronicRandomKey randomKey_{};
    std::int32_t projectilePdg_{};
    std::int32_t targetPdg_{};
    // (E, px, py, pz), all in GeV, in the request coordinate frame.
    std::array<double, 4> projectileFourMomentumGeV_{};
    std::array<double, 4> targetFourMomentumGeV_{};
  };

  struct HadronicInteractionResultHeader {
    std::uint32_t protocolVersion_{HadronicBatchProtocolVersion};
    HadronicWorkerStatus status_{HadronicWorkerStatus::Success};
    std::uint64_t sequenceId_{};
    std::uint32_t secondaryCount_{};
    // FLUKA requests random values in buffered blocks. This is the number supplied
    // by the worker callback, including values prefetched into FLUKA's buffer.
    std::uint32_t randomValuesSupplied_{};
    // Generator call only. IPC, parent scheduling, target selection, and main-stack
    // commit are measured separately by the parent process.
    double finalStateTimeMs_{};
  };

  struct HadronicSecondaryRecord {
    std::int32_t pdg_{};
    std::uint32_t reserved_{};
    double kineticEnergyGeV_{};
    std::array<double, 3> direction_{};
  };

  /**
   * Stable SplitMix64 combination used to seed a worker-local Philox stream.
   *
   * The returned seed depends only on the physical history key, never on the
   * worker number or dispatch order.  Rebalancing batches can therefore not
   * change an individual interaction's random sequence.
   */
  inline std::uint64_t derive_hadronic_worker_seed(HadronicRandomKey const& key) noexcept;

  inline void validate_hadronic_batch_header(HadronicBatchHeader const& header);

  inline void validate_hadronic_interaction_request(
      HadronicInteractionRequest const& request);

  static_assert(std::is_standard_layout_v<HadronicBatchHeader> &&
                std::is_trivially_copyable_v<HadronicBatchHeader>);
  static_assert(std::is_standard_layout_v<HadronicInteractionRequest> &&
                std::is_trivially_copyable_v<HadronicInteractionRequest>);
  static_assert(std::is_standard_layout_v<HadronicInteractionResultHeader> &&
                std::is_trivially_copyable_v<HadronicInteractionResultHeader>);
  static_assert(std::is_standard_layout_v<HadronicSecondaryRecord> &&
                std::is_trivially_copyable_v<HadronicSecondaryRecord>);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/HadronicBatchProtocol.inl>
