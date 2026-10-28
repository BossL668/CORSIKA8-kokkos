/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
namespace corsika::kokkos {
  struct InteractionRecord;
  struct PhotonFinalStateRecord;
  struct LeptonFinalStateRecord;
} // namespace corsika::kokkos
namespace corsika {
  /** Process trace retained independently of OutputManager, including incomplete showers.
   */
  class ReplayTrace {
  public:
    static ReplayTrace& instance();
    void open(std::filesystem::path const& path, std::string backend);
    bool enabled() const noexcept;
    void beginShower(unsigned int shower) noexcept;
    void recordProposalSelection(std::int32_t pdg, std::int32_t processId,
                                 std::uint64_t componentHash, double startEnergyGeV,
                                 double lossFraction, double xM, double yM, double zM,
                                 double nx, double ny, double nz, double timeS);
    void recordGpuStep(std::uint64_t historyId, std::uint64_t stepId, std::int32_t pdg,
                       std::int32_t processId, double startEnergyGeV, double endEnergyGeV,
                       double depositedEnergyGeV, double weight, double xM, double yM,
                       double zM, double timeS);
    void recordGpuInteraction(kokkos::InteractionRecord const& record);
    void recordGpuPhotonFinalState(kokkos::PhotonFinalStateRecord const& record,
                                   kokkos::InteractionRecord const* interaction);
    void recordGpuLeptonFinalState(kokkos::LeptonFinalStateRecord const& record,
                                   kokkos::InteractionRecord const* interaction);
    void close();

  private:
    ReplayTrace() = default;
    struct ExtraFields {
      double totalRateCm2PerG_{std::numeric_limits<double>::quiet_NaN()};
      double vertexTotalRateCm2PerG_{std::numeric_limits<double>::quiet_NaN()};
      double interactionGrammageGPerCm2_{std::numeric_limits<double>::quiet_NaN()};
      double distanceUniform_{std::numeric_limits<double>::quiet_NaN()};
      double processUniform_{std::numeric_limits<double>::quiet_NaN()};
      double proposalSelectionUniform_{std::numeric_limits<double>::quiet_NaN()};
      double lossQuantile_{std::numeric_limits<double>::quiet_NaN()};
      double finalStateUniform_{std::numeric_limits<double>::quiet_NaN()};
      double azimuthUniform_{std::numeric_limits<double>::quiet_NaN()};
      double auxiliaryUniform_{std::numeric_limits<double>::quiet_NaN()};
      double lpmSurvivalProbability_{std::numeric_limits<double>::quiet_NaN()};
      double lpmUniform_{std::numeric_limits<double>::quiet_NaN()};
      std::uint64_t distanceDrawId_{};
      std::uint64_t processDrawId_{};
      std::uint64_t proposalSelectionDrawId_{};
      std::uint64_t lossDrawId_{};
      std::uint64_t finalStateDrawId_{};
      std::uint64_t azimuthDrawId_{};
      std::uint64_t auxiliaryDrawId_{};
      std::uint64_t lpmDrawId_{};
      std::uint32_t secondaryCount_{};
    };
    void writeRow(char const* kind, std::uint64_t historyId, std::uint64_t stepId,
                  std::int32_t pdg, std::int32_t processId, std::uint64_t componentHash,
                  double startEnergyGeV, double endEnergyGeV, double lossFraction,
                  double depositedEnergyGeV, double weight, double xM, double yM,
                  double zM, double nx, double ny, double nz, double timeS);
    void writeRow(char const* kind, std::uint64_t historyId, std::uint64_t stepId,
                  std::int32_t pdg, std::int32_t processId, std::uint64_t componentHash,
                  double startEnergyGeV, double endEnergyGeV, double lossFraction,
                  double depositedEnergyGeV, double weight, double xM, double yM,
                  double zM, double nx, double ny, double nz, double timeS,
                  ExtraFields const& extra);
    std::ofstream output_{};
    std::string backend_{};
    unsigned int shower_{};
    std::uint64_t ordinal_{};
  };
} // namespace corsika
