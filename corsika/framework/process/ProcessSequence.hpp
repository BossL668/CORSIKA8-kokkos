/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

/**
 * @file ProcessSequence.hpp
 */

#include <corsika/framework/process/BaseProcess.hpp>
#include <corsika/framework/process/ProcessTraits.hpp>
#include <corsika/framework/process/BoundaryCrossingProcess.hpp>
#include <corsika/framework/process/ContinuousProcess.hpp>
#include <corsika/framework/process/ContinuousProcessIndex.hpp>
#include <corsika/framework/process/ContinuousProcessStepLength.hpp>
#include <corsika/framework/process/DecayProcess.hpp>
#include <corsika/framework/process/InteractionProcess.hpp>
#include <corsika/framework/process/CascadeEquationsProcess.hpp>
#include <corsika/framework/process/ProcessReturn.hpp>
#include <corsika/framework/process/SecondariesProcess.hpp>
#include <corsika/framework/process/StackProcess.hpp>
#include <corsika/framework/process/NullModel.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>

#include <corsika/framework/geometry/FourVector.hpp>
#include <tuple>

namespace corsika {

  class COMBoost;           // fwd-decl
  class NuclearComposition; // fwd-decl

  /**
   * count_processes traits specialization to increase process count by
   * getNumberOfProcesses(). This is used to statically count processes in the sequence.
   */
  template <typename TProcess, int N>
  struct count_processes<
      TProcess, N,
      typename std::enable_if_t<is_process_v<std::decay_t<TProcess>> &&
                                std::decay_t<TProcess>::is_process_sequence>> {
    static size_t constexpr count = N + std::decay_t<TProcess>::getNumberOfProcesses();
  };

  /**
   * @defgroup Processes Physics Processes and Modules
   *
   * Physics processes in CORSIKA 8 are clustered in ProcessSequence and
   * SwitchProcessSequence containers. The former is a mere (ordered) collection, while
   * the latter has the option to switch between two alternative ProcessSequences.
   *
   * Depending on the type of data to act on and on the allowed actions of processes
   * there are several interface options:
   * - InteractionProcess
   * - DecayProcess
   * - ContinuousProcess
   * - StackProcess
   * - SecondariesProcess
   * - BoundaryCrossingProcess
   *
   * And all processes (including ProcessSequence and SwitchProcessSequence) are derived
   * from BaseProcess.
   *
   * Processes of any type (e.g. p1, p2, p3,...) can be assembled into a ProcessSequence
   * using the `make_sequence` factory function.
   *
   * @code
   *   auto sequence1 = make_sequence(p1, p2, p3);
   *   auto sequence2 = make_sequence(p4, p5, p6, p7);
   *   auto sequence3 = make_sequence(sequence1, sequemce2, p8, p9);
   * @endcode
   *
   * Note, if the order of processes
   * matters, the order of occurence
   * in the ProcessSequence determines
   * the executiion order.
   *
   * SecondariesProcess alyways act on
   * new secondaries produced (i.e. in
   * InteractionProcess and
   * DecayProcess) in the scope of
   * their ProcessSequence. For
   * example if i1 and i2 are
   * InteractionProcesses and s1 is a
   * SecondariesProcess, then:
   *
   * @code{.cpp}
   *   auto sequence = make_sequence(i1, make_sequence(i2, s1))
   * @endcode
   *
   * will result in s1 acting only on
   * the particles produced by i2 and
   * not by i1. This can be very
   * useful, e.g. to fine tune thinning.
   *
   * A special type of ProcessSequence
   * is SwitchProcessSequence, which
   * has two branches and a functor
   * that can select between these two
   * branches.
   *
   * @code{.cpp}
   *   auto sequence = make_switch(sequence1, sequence2, selector);
   * @endcode
   *
   * where the only requirement to
   * `selector` is that it
   * provides a `SwitchResult operator()(Particle const& particle) const` method. Thus,
   * based on the dynamic properties
   * of `particle` the functor
   * can make its decision. This is
   * clearly important for switching
   * between low-energy and
   * high-energy models, but not
   * limited to this. The selection
   * can even be done with a lambda
   * function.
   *
   * @class ProcessSequence
   * @ingroup Processes
   *
   *   Definition of a static process list/sequence.
   *
   *   A compile time static list of processes. The compiler will
   *   generate a new type based on template logic containing all the
   *   elements provided by the user.
   *
   *   TProcess1 and TProcess2 must both be derived from BaseProcess,
   *   and are both references if possible (lvalue), otherwise (rvalue)
   *   they are just classes. This allows us to handle both, rvalue as
   *   well as lvalue Processes in the ProcessSequence.
   *
   *   (For your potential interest,
   *   the static version of the
   *   ProcessSequence and all Process
   *   types are based on the CRTP C++
   *   design pattern).
   *
   *  Template parameters:
   *  @tparam TProcess1 is of type BaseProcess, either a dedicatd process, or a
   *           ProcessSequence
   *  @tparam TProcess2 is of type BaseProcess, either a dedicatd process, or a
   *           ProcessSequence
   *  @tparam IndexFirstProcess to count and index each Process in the entire
   *          process-chain. The offset is the starting value for this ProcessSequence
   *  @tparam IndexOfProcess1 index of TProcess1 (counting of Process)
   *  @tparam IndexOfProcess2 index of TProcess2 (counting of Process)
   */

  template <typename TProcess1, typename TProcess2 = NullModel,
            int ProcessIndexOffset = 0,
            int IndexOfProcess1 = corsika::count_processes<
                TProcess1,
                corsika::count_processes<TProcess2, ProcessIndexOffset>::count>::count,
            int IndexOfProcess2 =
                corsika::count_processes<TProcess2, ProcessIndexOffset>::count>
  class ProcessSequence : public BaseProcess<ProcessSequence<TProcess1, TProcess2>> {

    using process1_type = typename std::decay_t<TProcess1>;
    using process2_type = typename std::decay_t<TProcess2>;

  public:
    // resource management
    ProcessSequence() = delete; // only initialized objects
    ProcessSequence(ProcessSequence const&) = default;
    ProcessSequence(ProcessSequence&&) = default;
    ProcessSequence& operator=(ProcessSequence const&) = default;
    ~ProcessSequence() = default;

    static bool const is_process_sequence = true;

    /**
     * Only valid user constructor will create fully initialized object.
     *
     * ProcessSequence supports and encourages move semantics. You can
     * use object, l-value references or r-value references to
     * construct sequences.
     *
     * @param in_A BaseProcess or switch/process list
     * @param in_B BaseProcess or switch/process list
     */
    ProcessSequence(TProcess1 in_A, TProcess2 in_B);

    /**
     * List of all BoundaryProcess.
     *
     * @tparam TParticle
     * @param particle The particle.
     * @param from Volume the particle is exiting.
     * @param to Volume the particle is entering.
     * @return ProcessReturn
     */
    template <typename TParticle>
    ProcessReturn doBoundaryCrossing(TParticle& particle,
                                     typename TParticle::node_type const& from,
                                     typename TParticle::node_type const& to);

    template <typename TParticle>
    ProcessReturn doContinuous(Step<TParticle>& step,
                               ContinuousProcessIndex const limitID);

    /**
     * Process all secondaries in TSecondaries.
     *
     * The seondaries produced by other processes and accessible via TSecondaries
     * are processed by all SecondariesProcesse via a call here.
     *
     * @tparam TSecondaries
     * @param vS
     */
    template <typename TSecondaries>
    void doSecondaries(TSecondaries& vS);

    /**
     * The processes of type StackProcess do have an internal counter,
     * so they can be exectuted only each N steps. Often these are
     * "maintenacne processes" that do not need to run after each
     * single step of the simulations. In the CheckStep function it is
     * tested if either A_ or B_ are StackProcess and if they are due
     * for execution.
     */
    bool checkStep();

    /**
     * Execute the StackProcess-es in the ProcessSequence.
     */
    template <typename TStack>
    void doStack(TStack& stack);

    /**
     * Execute the CascadeEquationsProcess-es in the ProcessSequence.
     */
    template <typename TStack>
    void doCascadeEquations(TStack& stack);

    /**
     * Init the CascadeEquationsProcess-es in the ProcessSequence.
     */
    void initCascadeEquations();

    /**
     * Calculate the maximum allowed length of the next tracking step, based on all
     * ContinuousProcess-es.
     *
     * The maximum allowed step length is the minimum of the allowed track lenght over all
     * ContinuousProcess-es in the ProcessSequence.
     *
     * @tparam TParticle particle type.
     * @tparam TTrack the trajectory type.
     * @param particle The particle data object.
     * @param track The track data object.
     * @return ContinuousProcessStepLength which contains the step length itself in
     *          LengthType, and a unique identifier of the related ContinuousProcess.
     */

    template <typename TParticle, typename TTrack>
    ContinuousProcessStepLength getMaxStepLength(TParticle&& particle, TTrack&& vTrack);

    /**
     * @brief Calculates the cross section of a projectile with a target.
     *
     * @tparam TParticle
     * @param projectile
     * @param targetId
     * @param targetP4
     * @return CrossSectionType
     */
    template <typename TParticle>
    CrossSectionType getCrossSection(TParticle const& projectile, Code const targetId,
                                     FourMomentum const& targetP4) const;

    template <typename TParticle>
    TimeType getLifetime(TParticle& particle) {
      return 1. / getInverseLifetime(particle);
    }

    /**
     * Selects one concrete InteractionProcess and samples a target nucleus from
     * the material.
     *
     * The selectInteraction method statically loops over all active InteractionProcess
     * and calculates the material-weighted cross section for all of them. In an iterative
     * way those cross sections are summed up. The random number cx_select, uniformely
     * drawn from the cross section before energy losses, is used to discriminate the
     * selected sub-process here. If the cross section after the step smaller than it was
     * before, there is a non-zero probability that the particle survives and no
     * interaction takes place. This method becomes imprecise when cross section rise with
     * falling energies.
     *
     * If a sub-process was selected, the target nucleus is selected from the material
     * (weighted with cross section). The interaction is then executed.
     *
     * @tparam TSecondaryView Object type as storage for new secondary particles.
     * @tparam TRNG Object type to produce random numbers.
     * @param view Object to store new secondary particles.
     * @param projectileP4 The four momentum of the projectile.
     * @param composition The environment/material composition.
     * @param rng Random number object.
     * @param cx_select Drawn random numer, uniform between [0, cx_initial]
     * @param cx_sum For interal use, to sum up cross section contributions.
     * @return ProcessReturn
     */
    template <typename TSecondaryView, typename TRNG>
    inline ProcessReturn selectInteraction(
        TSecondaryView&& view, FourMomentum const& projectileP4,
        NuclearComposition const& composition, TRNG&& rng,
        CrossSectionType const cx_select,
        CrossSectionType cx_sum = CrossSectionType::zero());

    template <typename TParticle>
    InverseTimeType getInverseLifetime(TParticle&& particle);

    // select decay process
    template <typename TSecondaryView>
    ProcessReturn selectDecay(TSecondaryView&& view, InverseTimeType decay_inv_select,
                              InverseTimeType decay_inv_sum = InverseTimeType::zero());

    /**
     * static counter to uniquely index (count) all ContinuousProcess in switch sequence.
     */
    static size_t constexpr getNumberOfProcesses() { return numberOfProcesses_; }

    /** Borrow the two immediate components without executing or copying them.
     * References remain valid while this sequence and its borrowed processes live.
     * This accessor does not flatten nested process scopes.
     */
    auto getProcesses() & { return std::tie(A_, B_); }

#ifdef CORSIKA_UNIT_TESTING
    TProcess1 getProcess1() const { return A_; }
    TProcess2 getProcess2() const { return B_; }
#endif

  private:
    TProcess1 A_; //! process/list A, this is a reference, if possible
    TProcess2 B_; //! process/list B, this is a reference, if possible

    static size_t constexpr numberOfProcesses_ = IndexOfProcess1; // static counter
  };

  /**
   * @fn make_sequence
   * @ingroup Processes
   *
   * Factory function to create a ProcessSequence.
   *
   * to construct ProcessSequences in a flexible and dynamic way the
   * `sequence` factory functions are provided.
   *
   * Any objects of type
   *  - BaseProcess
   *  - ContinuousProcess and
   *  - InteractionProcess/DecayProcess
   *  - StackProcess
   *  - SecondariesProcess
   *  - BoundaryCrossingProcess
   *
   * can be assembled into a ProcessSequence, all
   * combinatorics are allowed.
   *
   * The sequence function checks that all its arguments are all of
   * types derived from BaseProcess. Also the ProcessSequence itself
   * is derived from type BaseProcess.
   *
   * @tparam TProcesses parameter pack with objects of type BaseProcess
   * @tparam TProcess1 another BaseProcess
   * @param vA needs to derive from BaseProcess
   * @param vB paramter-pack, needs to derive BaseProcess
   */
  template <typename... TProcesses, typename TProcess1>
  ProcessSequence<TProcess1, decltype(make_sequence(std::declval<TProcesses>()...))>
  make_sequence(TProcess1&& vA, TProcesses&&... vBs) {
    return ProcessSequence<TProcess1,
                           decltype(make_sequence(std::declval<TProcesses>()...))>(
        vA, make_sequence(std::forward<TProcesses>(vBs)...));
  }

  /**
   * @fn make_sequence
   * @ingroup Processes
   *
   * Factory function to create ProcessSequence.
   *
   * specialization for two input objects (no paramter pack in vB).
   *
   * @tparam TProcess1 another BaseProcess
   * @tparam TProcess2 another BaseProcess
   * @param vA needs to derive from BaseProcess
   * @param vB needs to derive BaseProcess
   */
  template <typename TProcess1, typename TProcess2>
  ProcessSequence<TProcess1, TProcess2> make_sequence(TProcess1&& vA, TProcess2&& vB) {
    return ProcessSequence<TProcess1, TProcess2>(vA, vB);
  }

  /**
   * @fn make_sequence
   * @ingroup Processes
   *
   * Factory function to create ProcessSequence from a single BaseProcess.
   *
   * also allow a single Process in ProcessSequence, accompany by
   * `NullModel`.
   *
   * @tparam TProcess1 another BaseProcess
   * @param vA needs to derive from BaseProcess
   */
  template <typename TProcess>
  ProcessSequence<TProcess, NullModel> make_sequence(TProcess&& vA) {
    return ProcessSequence<TProcess, NullModel>(vA, NullModel());
  }

} // namespace corsika

#include <corsika/detail/framework/process/ProcessSequence.inl>
