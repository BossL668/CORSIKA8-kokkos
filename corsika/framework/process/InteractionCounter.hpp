/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/process/InteractionHistogram.hpp>
#include <corsika/framework/core/HadronicInteractionDeferral.hpp>
#include <corsika/framework/process/InteractionProcess.hpp>
#include <corsika/framework/geometry/FourVector.hpp>

#include <chrono>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika {

  namespace interaction_counter_detail {
    template <typename T, typename = void>
    struct CanAuditEnergy : std::false_type {};

    template <typename T>
    struct CanAuditEnergy<
        T, std::void_t<
               decltype(std::declval<T&>()
                            .getProjectile()
                            .getWeight()),
               decltype(std::begin(std::declval<T&>())),
               decltype(std::end(std::declval<T&>())),
               decltype((*std::begin(std::declval<T&>()))
                            .getEnergy()),
               decltype((*std::begin(std::declval<T&>()))
                            .getPID()),
               decltype((*std::begin(std::declval<T&>()))
                            .getWeight())>> : std::true_type {};
  } // namespace interaction_counter_detail

  struct InteractionTimingSample {
    std::uint64_t sequence_id{};
    Code projectile{Code::Unknown};
    Code target{Code::Unknown};
    HEPEnergyType kinetic_energy{};
    double final_state_time_ms{};
    bool deferred{};
  };

  /**
   * Four-momentum energy audit accumulated at a hadronic interaction boundary.
   *
   * All energies include rest mass and particle weights.  The residual is
   * defined as projectile + target - generated secondaries.  Keeping the
   * target term explicit is essential for a whole-cascade ledger: event
   * generators receive a material target at rest, so comparing secondaries to
   * the projectile alone creates an apparent energy source.
   */
  struct InteractionEnergyLedgerStatistics {
    std::uint64_t audited_interactions{};
    std::uint64_t deferred_interactions{};
    double weighted_projectile_total_energy_GeV{};
    double weighted_target_total_energy_GeV{};
    double weighted_effective_target_total_energy_GeV{};
    double weighted_target_isotope_correction_GeV{};
    std::uint64_t target_isotope_adjusted_interactions{};
    double weighted_secondary_total_energy_GeV{};
    double weighted_energy_residual_GeV{};
    double weighted_absolute_energy_residual_GeV{};
    double maximum_absolute_residual_GeV{};
    double weighted_isotope_corrected_energy_residual_GeV{};
    double weighted_absolute_isotope_corrected_energy_residual_GeV{};
    double maximum_absolute_isotope_corrected_residual_GeV{};
    std::uint64_t negative_isotope_corrected_residual_interactions{};
    double maximum_negative_isotope_corrected_residual_GeV{};
    std::uint64_t spacelike_isotope_corrected_residual_interactions{};
    double maximum_isotope_corrected_spacelike_excess_GeV{};
    double minimum_isotope_corrected_residual_mass_squared_GeV2{};
    double maximum_target_total_energy_GeV{};
    double weighted_momentum_residual_norm_GeV{};
    double maximum_momentum_residual_norm_GeV{};
    std::uint64_t negative_energy_residual_interactions{};
    std::uint64_t spacelike_residual_interactions{};
    double maximum_negative_energy_residual_GeV{};
    int maximum_negative_projectile_pdg{};
    int maximum_negative_target_pdg{};
    double maximum_negative_projectile_total_energy_GeV{};
    double maximum_negative_target_total_energy_GeV{};
    double maximum_negative_secondary_total_energy_GeV{};
    std::uint64_t maximum_negative_secondary_count{};
    double maximum_negative_baryon_number_residual{};
    double maximum_negative_charge_number_residual{};
    std::vector<int> maximum_negative_secondary_pdgs{};
    double maximum_spacelike_excess_GeV{};
    double minimum_residual_mass_squared_GeV2{};
    double maximum_timelike_residual_mass_GeV{};
    double maximum_positive_residual_beta{};
  };

  /**
   * @ingroup Processes
   * @{
   *
   * Wrapper around an InteractionProcess that fills histograms of the number
   * of calls to `doInteraction()` binned in projectile energy (both in
   * lab and center-of-mass frame) and species.
   *
   * Use by wrapping a normal InteractionProcess:
   * @code{.cpp}
   * InteractionProcess collision1;
   * InteractionClounter<collision1> counted_collision1;
   * @endcode
   */

  template <class TCountedProcess>
  class InteractionCounter
      : public InteractionProcess<InteractionCounter<TCountedProcess>> {

  public:
    InteractionCounter(TCountedProcess& process);

    /**
     * Wrapper around internal process doInteraction.
     */
    template <typename TSecondaryView>
    void doInteraction(TSecondaryView& view, Code const, Code const, FourMomentum const&,
                       FourMomentum const&);

    /**
     * Wrapper around internal process getCrossSection.
     */
    CrossSectionType getCrossSection(Code const, Code const, FourMomentum const&,
                                     FourMomentum const&) const;

    /**
     * returns the filles histograms.
     *
     * @return InteractionHistogram, which contains the histogram data
     */
    InteractionHistogram const& getHistogram() const;

    /**
     * Return the exact number of calls forwarded to doInteraction().
     *
     * This is deliberately independent of histogram serialization, so run
     * metadata can prove which low-/high-energy model generated final states.
     */
    std::uint64_t getCount() const;

    /**
     * Per-interaction final-state generation timings in execution order.
     *
     * Tracking, continuous transport, target selection, doSecondaries(), and
     * output are deliberately excluded. These samples estimate only the work
     * that a process-isolated event-generator worker can accelerate.
     */
    std::vector<InteractionTimingSample> const& getTimingSamples() const;

    double getTotalFinalStateTimeMs() const;

    InteractionEnergyLedgerStatistics const&
    getEnergyLedgerStatistics() const;

  private:
    TCountedProcess& process_;
    InteractionHistogram histogram_;
    std::uint64_t count_{};
    double total_final_state_time_ms_{};
    std::vector<InteractionTimingSample> timing_samples_;
    InteractionEnergyLedgerStatistics energy_ledger_statistics_{};
  };

  //! @}

} // namespace corsika

#include <corsika/detail/framework/process/InteractionCounter.inl>
