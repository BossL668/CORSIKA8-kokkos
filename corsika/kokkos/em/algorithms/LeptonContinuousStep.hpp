/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/kokkos/em/FallbackRequest.hpp>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  /** Scalar C8 maximum fractional loss of total energy per continuous step. */
  inline constexpr double MaximumContinuousRelativeLoss = 0.10;

  /** Prepared transport mass/cuts and maximum grammage, or original diagnostics.
   * Energies are MeV and grammage is g/cm^2. This does not execute ParticleCut. */
  struct LeptonContinuousPreparation {
    double massMeV_{};
    double minimumEnergyMeV_{};
    double transportCutMeV_{};
    double maximumGrammageGPerCm2_{};
    std::uint32_t particleCut_{};
    std::uint32_t fallbackFlag_{};
    FallbackRequest fallback_{};
  };

  /** Post-step total energy and unweighted deposited energy, or diagnostics.
   * finalEnergyMeV_ is in MeV; depositedEnergyGeV_ is in GeV. */
  struct LeptonContinuousLoss {
    double finalEnergyMeV_{};
    double depositedEnergyGeV_{};
    std::uint32_t transportCutReached_{};
    std::uint32_t fallbackFlag_{};
    FallbackRequest fallback_{};
  };
  static_assert(std::is_trivially_copyable_v<LeptonContinuousPreparation>);
  static_assert(std::is_trivially_copyable_v<LeptonContinuousLoss>);

  namespace lepton_continuous_detail {
    CORSIKA_HOST_DEVICE inline bool finite(double const value);

    CORSIKA_HOST_DEVICE inline FallbackRequest make_transport_fallback(
        InteractionRecord const& interaction, FallbackReason const reason);

    CORSIKA_HOST_DEVICE inline FallbackRequest make_table_fallback(
        InteractionRecord const& interaction, TableLookupStatus const status,
        double const query_stage = 0., double const query_value = 0.);
  } // namespace lepton_continuous_detail

  /** Resolve mass, cuts and the original 10%-of-total-energy grammage limit.
   * Retains query order, strict cut comparisons and the existing tiny-negative
   * range-difference correction. The returned flag/request must be inspected;
   * no transport, RNG, scheduling or CPU fallback is performed here. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline LeptonContinuousPreparation
  prepare_lepton_continuous_step(ContinuousPhysicsView const& table,
                                 InteractionRecord const& interaction);

  /** Evaluate the original post-step range inversion and deposit accounting.
   * Supply the grammage of the completed step's endpoint chord, not the
   * start-tangent grammage used for earlier step-limit competition.
   * continuous_limit_won retains the original nonpositive-loss failure rule.
   * This consumes the preceding preparation without adding new validation. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline LeptonContinuousLoss
  evaluate_lepton_continuous_loss(ContinuousPhysicsView const& table,
                                  InteractionRecord const& interaction,
                                  LeptonContinuousPreparation const& preparation,
                                  double const traversed_grammage_g_per_cm2,
                                  bool const continuous_limit_won);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/LeptonContinuousStep.inl>
