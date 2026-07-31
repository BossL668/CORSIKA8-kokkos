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
#include <vector>

namespace corsika {

  struct InteractionTimingSample {
    std::uint64_t sequence_id{};
    Code projectile{Code::Unknown};
    Code target{Code::Unknown};
    HEPEnergyType kinetic_energy{};
    double final_state_time_ms{};
    bool deferred{};
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

  private:
    TCountedProcess& process_;
    InteractionHistogram histogram_;
    std::uint64_t count_{};
    double total_final_state_time_ms_{};
    std::vector<InteractionTimingSample> timing_samples_;
  };

  //! @}

} // namespace corsika

#include <corsika/detail/framework/process/InteractionCounter.inl>
