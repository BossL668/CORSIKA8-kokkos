/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

/*
 * POD-only disk/device types for scalar-to-CUDA decision replay.
 */
#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace corsika::diagnostics {

  inline constexpr std::array<char, 8> ReplayTapeMagic{'C', '8', 'R', 'P',
                                                       'T', '0', '0', '1'};
  inline constexpr std::uint32_t ReplayTapeVersion = 1;
  inline constexpr std::uint64_t ReplayEndianMarker = 0x0102030405060708ULL;

  enum class ReplayTransportLimit : std::int32_t {
    Continuous = 0,
    Geometry = 1,
    Interaction = 2,
    Decay = 3,
    ForcedInteraction = 4,
    ForcedDecay = 5,
  };

  struct ReplayRadioObserver {
    double positionM_[3]{};
    double startTimeS_{};
    double durationS_{};
    double sampleRateHz_{};
    double axisStartNs_{};
    std::uint64_t numberOfBins_{};
  };

  struct ReplayPropagationSnapshot {
    double minimumHeightM_{};
    double maximumHeightM_{};
    double stepM_{};
    double inverseStepPerM_{};
    double slopeRefractivityLower_{};
    double slopeIntegratedRefractivityLower_{};
    double slopeRefractivityUpper_{};
    double slopeIntegratedRefractivityUpper_{};
    std::vector<double> refractivity_{};
    std::vector<double> integratedRefractivity_{};
  };

  struct ReplayRadioSnapshot {
    ReplayPropagationSnapshot propagation_{};
    std::vector<ReplayRadioObserver> coreasObservers_{};
    std::vector<ReplayRadioObserver> zhsObservers_{};
  };

  struct ReplayTapeHeader {
    char magic_[8]{};
    std::uint32_t formatVersion_{};
    std::uint32_t headerBytes_{};
    std::uint64_t endianMarker_{};
    std::uint64_t recordBytes_{};
    std::uint64_t recordCount_{};
    std::uint64_t configuredShowers_{};
    std::uint64_t randomSeed_{};
    std::uint64_t refractivityCount_{};
    std::uint64_t integratedRefractivityCount_{};
    std::uint64_t coreasObserverCount_{};
    std::uint64_t zhsObserverCount_{};
    double minimumHeightM_{};
    double maximumHeightM_{};
    double propagationStepM_{};
    double inversePropagationStepPerM_{};
    double slopeRefractivityLower_{};
    double slopeIntegratedRefractivityLower_{};
    double slopeRefractivityUpper_{};
    double slopeIntegratedRefractivityUpper_{};
  };

  struct ReplayTransportStep {
    std::uint64_t shower_{};
    std::uint64_t ordinal_{};
    std::uint64_t stackIndex_{};
    std::int32_t pdg_{};
    ReplayTransportLimit transportLimit_{ReplayTransportLimit::Continuous};
    std::int32_t processReturn_{};
    std::int32_t reserved_{};
    double startPositionM_[3]{};
    double endPositionM_[3]{};
    double startDirection_[3]{};
    double endDirection_[3]{};
    double startTimeS_{};
    double endTimeS_{};
    double startTotalEnergyGeV_{};
    double endTotalEnergyGeV_{};
    double weight_{};
  };

  struct ReplayTapeData {
    ReplayTapeHeader header_{};
    ReplayRadioSnapshot radio_{};
    std::vector<ReplayTransportStep> steps_{};
  };

  static_assert(std::is_trivially_copyable_v<ReplayTapeHeader>);
  static_assert(std::is_trivially_copyable_v<ReplayRadioObserver>);
  static_assert(std::is_trivially_copyable_v<ReplayTransportStep>);

} // namespace corsika::diagnostics
