/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

/**
  @file SwitchProcessSequence.hpp
 **/

#include <corsika/framework/process/BaseProcess.hpp>
#include <corsika/framework/process/ProcessTraits.hpp>
#include <corsika/framework/process/BoundaryCrossingProcess.hpp>
#include <corsika/framework/process/ContinuousProcess.hpp>
#include <corsika/framework/process/ContinuousProcessIndex.hpp>
#include <corsika/framework/process/ContinuousProcessStepLength.hpp>
#include <corsika/framework/process/DecayProcess.hpp>
#include <corsika/framework/process/InteractionProcess.hpp>
#include <corsika/framework/process/ProcessReturn.hpp>
#include <corsika/framework/process/SecondariesProcess.hpp>
#include <corsika/framework/process/StackProcess.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/geometry/FourVector.hpp>

#include <cmath>
#include <limits>
#include <type_traits>
#include <tuple>

namespace corsika {

  /**
     @ingroup Processes
     @{

     Class to switch between two process branches

     A compile-time static list of processes that uses an internal
     TCondition class to switch between different versions of processes
     (or process sequence).

     TSequence and USequence must be derived from BaseProcess and are
     both references if possible (lvalue), otherwise (rvalue) they are
     just classes. This allows us to handle both, rvalue as well as
     lvalue Processes in the SwitchProcessSequence.
     Please use the `corsika::make_select(condition, sequence, alt_sequence)`
     factory function for best results.

     TCondition has to implement a `bool operator()(Particle const&)`. Note:
     TCondition may absolutely also use random numbers to sample between
     its results. This can be used to achieve arbitrarily smooth
     transition or mixtures of processes.

     Warning: do not put StackProcess into a SwitchProcessSequence
     since this makes no sense. The StackProcess acts on an entire
     particle stack and not on indiviidual particles.

     Template parameters:
      @tparam TCondition selector functor/function
      @tparam TSequence is of type BaseProcess, either a dedicatd process, or a
  ProcessSequence
      @tparam USequence is of type BaseProcess, either a dedicatd process, or a
  ProcessSequence
      @tparam IndexFirstProcess to count and index each Process in the entire
  process-chain
      @tparam IndexOfProcess1 index of TSequence (counting of Process)
      @tparam IndexOfProcess2 index of USequence (counting of Process)

     See also class ProcessSequence.
  **/

  template <typename TCondition, typename TSequence, typename USequence,
            int IndexFirstProcess = 0,
            int IndexOfProcess1 = count_processes<TSequence, IndexFirstProcess>::count,
            int IndexOfProcess2 = count_processes<USequence, IndexOfProcess1>::count>
  class SwitchProcessSequence
      : public BaseProcess<SwitchProcessSequence<TCondition, TSequence, USequence>> {

    using process1_type = typename std::decay_t<TSequence>;
    using process2_type = typename std::decay_t<USequence>;

    // make sure only BaseProcess types TSequence/2 are passed
    static_assert(is_process_v<process1_type>,
                  "can only use process derived from BaseProcess in "
                  "SwitchProcessSequence, for Process 1");
    static_assert(is_process_v<process2_type>,
                  "can only use process derived from BaseProcess in "
                  "SwitchProcessSequence, for Process 2");

    // make sure none of TSequence/2 is a StackProcess
    static_assert(!std::is_base_of_v<StackProcess<process1_type>, process1_type>,
                  "cannot use StackProcess in SwitchProcessSequence, for Process 1");
    static_assert(!std::is_base_of_v<StackProcess<process2_type>, process2_type>,
                  "cannot use StackProcess in SwitchProcessSequence, for Process 2");

    // if TSequence/2 are already ProcessSequences, make sure they do not contain
    // any StackProcess
    static_assert(!contains_stack_process_v<process1_type>,
                  "cannot use StackProcess in SwitchProcessSequence, remove from "
                  "ProcessSequence 1");
    static_assert(!contains_stack_process_v<process2_type>,
                  "cannot use StackProcess in SwitchProcessSequence, remove from "
                  "ProcessSequence 2");

  public:
    // resource management
    SwitchProcessSequence() = delete; // only initialized objects
    SwitchProcessSequence(SwitchProcessSequence const&) = default;
    SwitchProcessSequence(SwitchProcessSequence&&) = default;
    SwitchProcessSequence& operator=(SwitchProcessSequence const&) = default;
    ~SwitchProcessSequence() = default;

    static bool const is_process_sequence = true;
    static bool const is_switch_process_sequence = true;

    /**
     * Only valid user constructor will create fully initialized object.
     *
     * SwitchProcessSequence supports and encourages move semantics. You can
     * use object, l-value references or r-value references to
     * construct sequences.
     *
     * @param sel functor to switch between branch A and B
     * @param in_A process branch A
     * @param in_B process branch B
     */
    SwitchProcessSequence(TCondition sel, TSequence in_A, USequence in_B)
        : select_(sel)
        , A_(in_A)
        , B_(in_B) {}

    template <typename TParticle>
    ProcessReturn doBoundaryCrossing(TParticle& particle,
                                     typename TParticle::node_type const& from,
                                     typename TParticle::node_type const& to);

    template <typename TParticle>
    ProcessReturn doContinuous(Step<TParticle>& particle,
                               ContinuousProcessIndex const limitId);

    template <typename TSecondaries>
    void doSecondaries(TSecondaries& vS);

    template <typename TParticle, typename TTrack>
    ContinuousProcessStepLength getMaxStepLength(TParticle& particle, TTrack& vTrack);

    template <typename TParticle>
    CrossSectionType getCrossSection(TParticle const& projectile, Code const targetId,
                                     FourMomentum const& targetP4) const;

    template <typename TSecondaryView, typename TRNG>
    ProcessReturn selectInteraction(TSecondaryView& view,
                                    FourMomentum const& projectileP4,
                                    NuclearComposition const& composition, TRNG& rng,
                                    CrossSectionType const cx_select,
                                    CrossSectionType cx_sum = CrossSectionType::zero());

    template <typename TParticle>
    TimeType getLifetime(TParticle&& particle) {
      return 1. / getInverseLifetime(particle);
    }

    template <typename TParticle>
    InverseTimeType getInverseLifetime(TParticle&& particle);

    // select decay process
    template <typename TSecondaryView>
    ProcessReturn selectDecay(
        TSecondaryView& view, [[maybe_unused]] InverseTimeType decay_inv_select,
        [[maybe_unused]] InverseTimeType decay_inv_sum = InverseTimeType::zero());

    /**
     * static counter to uniquely index (count) all ContinuousProcess in switch sequence.
     */
    static size_t constexpr getNumberOfProcesses() { return numberOfProcesses_; }

    /** Borrow both alternatives without evaluating the condition or copying models.
     * The returned references must not outlive this switch or its borrowed models.
     */
    auto getProcesses() & { return std::tie(A_, B_); }

#ifdef CORSIKA_UNIT_TESTING
    TCondition getCondition() const { return select_; }
    TSequence getSequence() const { return A_; }
    USequence getAltSequence() const { return B_; }
#endif

  private:
    TCondition select_; /// selector functor to switch between branch a and b, this is a
                        /// reference, if possible

    TSequence A_; /// process branch a, this is a reference, if possible
    USequence B_; /// process branch b, this is a reference, if possible

    static unsigned int constexpr numberOfProcesses_ = IndexOfProcess2; // static counter
  };

  /**
    the functin `make_select(select, proc1, proc1)` assembles many
    BaseProcesses, and ProcessSequences into a SwitchProcessSequence,
    all combinatorics are allowed.

    @param selector must provide `bool operator()(Particle const&) const`
    @param vA needs to derive from BaseProcess or ProcessSequence
    @param vB needs to derive from BaseProcess or ProcessSequence
   **/

  template <typename TCondition, typename TSequence, typename USequence,
            typename = std::enable_if_t<is_process_v<typename std::decay_t<TSequence>> &&
                                        is_process_v<typename std::decay_t<USequence>>>>
  SwitchProcessSequence<TCondition, TSequence, USequence> make_select(
      TCondition&& selector, TSequence&& vA, USequence&& vB) {
    return SwitchProcessSequence<TCondition, TSequence, USequence>(selector, vA, vB);
  }

  //! @}

} // namespace corsika

#include <corsika/detail/framework/process/SwitchProcessSequence.inl>
