/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/process/InteractionHistogram.hpp>
#include <corsika/framework/process/InteractionProcess.hpp>
#include <corsika/framework/geometry/FourVector.hpp>

#include <chrono>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika {

  /** Default execution policy: keep final-state generation synchronous.
   * An explicitly selected policy may capture an interaction after histogram
   * accounting. Distinct policies produce distinct counter types; merely
   * including an optional adapter cannot change a native counter's behavior.
   */
  struct NoInteractionDeferral {
    static bool tryPrepare(Code const, Code const, FourMomentum const&,
                           FourMomentum const&) noexcept;
  };

  namespace interaction_counter_detail {
    template <typename T, typename = void>
    struct CanAuditEnergy : std::false_type {};

    template <typename T>
    struct CanAuditEnergy<
        T, std::void_t<decltype(std::declval<T&>().getProjectile().getWeight()),
                       decltype(std::begin(std::declval<T&>())),
                       decltype(std::end(std::declval<T&>())),
                       decltype((*std::begin(std::declval<T&>())).getEnergy()),
                       decltype((*std::begin(std::declval<T&>())).getPID()),
                       decltype((*std::begin(std::declval<T&>())).getWeight())>>
        : std::true_type {};
  } // namespace interaction_counter_detail

  struct InteractionTimingSample {
    std::uint64_t sequenceId_{};
    Code projectile_{Code::Unknown};
    Code target_{Code::Unknown};
    HEPEnergyType kineticEnergy_{};
    double finalStateTimeMs_{};
    bool deferred_{};
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
    std::uint64_t auditedInteractions_{};
    std::uint64_t deferredInteractions_{};
    double weightedProjectileTotalEnergyGeV_{};
    double weightedTargetTotalEnergyGeV_{};
    double weightedEffectiveTargetTotalEnergyGeV_{};
    double weightedTargetIsotopeCorrectionGeV_{};
    std::uint64_t targetIsotopeAdjustedInteractions_{};
    double weightedSecondaryTotalEnergyGeV_{};
    double weightedEnergyResidualGeV_{};
    double weightedAbsoluteEnergyResidualGeV_{};
    double maximumAbsoluteResidualGeV_{};
    double weightedIsotopeCorrectedEnergyResidualGeV_{};
    double weightedAbsoluteIsotopeCorrectedEnergyResidualGeV_{};
    double maximumAbsoluteIsotopeCorrectedResidualGeV_{};
    std::uint64_t negativeIsotopeCorrectedResidualInteractions_{};
    double maximumNegativeIsotopeCorrectedResidualGeV_{};
    std::uint64_t spacelikeIsotopeCorrectedResidualInteractions_{};
    double maximumIsotopeCorrectedSpacelikeExcessGeV_{};
    double minimumIsotopeCorrectedResidualMassSquaredGeV2_{};
    double maximumTargetTotalEnergyGeV_{};
    double weightedMomentumResidualNormGeV_{};
    double maximumMomentumResidualNormGeV_{};
    std::uint64_t negativeEnergyResidualInteractions_{};
    std::uint64_t spacelikeResidualInteractions_{};
    double maximumNegativeEnergyResidualGeV_{};
    int maximumNegativeProjectilePdg_{};
    int maximumNegativeTargetPdg_{};
    double maximumNegativeProjectileTotalEnergyGeV_{};
    double maximumNegativeTargetTotalEnergyGeV_{};
    double maximumNegativeSecondaryTotalEnergyGeV_{};
    std::uint64_t maximumNegativeSecondaryCount_{};
    double maximumNegativeBaryonNumberResidual_{};
    double maximumNegativeChargeNumberResidual_{};
    std::vector<int> maximumNegativeSecondaryPdgs_{};
    double maximumSpacelikeExcessGeV_{};
    double minimumResidualMassSquaredGeV2_{};
    double maximumTimelikeResidualMassGeV_{};
    double maximumPositiveResidualBeta_{};
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

  template <class TCountedProcess, typename TDeferral = NoInteractionDeferral>
  class InteractionCounter
      : public InteractionProcess<InteractionCounter<TCountedProcess, TDeferral>> {

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

    /** Release consumed samples only; retain cumulative counts and ledgers. */
    void releaseTimingSamples() {
      std::vector<InteractionTimingSample>{}.swap(timingSamples_);
    }

    double getTotalFinalStateTimeMs() const;

    InteractionEnergyLedgerStatistics const& getEnergyLedgerStatistics() const;

  private:
    TCountedProcess& process_;
    InteractionHistogram histogram_;
    std::uint64_t count_{};
    double totalFinalStateTimeMs_{};
    std::vector<InteractionTimingSample> timingSamples_;
    InteractionEnergyLedgerStatistics energyLedgerStatistics_{};
  };

  //! @}

} // namespace corsika

#include <corsika/detail/framework/process/InteractionCounter.inl>
