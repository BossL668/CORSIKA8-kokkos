/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika {

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack>
  inline Cascade<TTracking, TProcessList, TOutput, TStack>::Cascade(
      Environment<medium_interface_type> const& env, TTracking& tr, TProcessList& pl,
      TOutput& out, TStack& stack)
      : sequence_(pl)
      , output_(out)
      , stack_(stack)
      , stepper_(env, tr, pl, stack) {
    CORSIKA_LOG_INFO(c8_ascii_);
    CORSIKA_LOG_INFO("This is CORSIKA {}.{}.{}.{}", CORSIKA_RELEASE_NUMBER,
                     CORSIKA_MAJOR_NUMBER, CORSIKA_MINOR_NUMBER, CORSIKA_PATCH_NUMBER);
    CORSIKA_LOG_INFO(
        "The C8 author list can be found at: "
        "https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/-/wikis/"
        "Current-CORSIKA-8-author-list");
    CORSIKA_LOG_INFO("Tracking algorithm: {} (version {})", TTracking::getName(),
                     TTracking::getVersion());
    if constexpr (stack_view_type::has_event) {
      CORSIKA_LOG_INFO("Stack - with full cascade HISTORY.");
    }
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack>
  inline void Cascade<TTracking, TProcessList, TOutput, TStack>::run() {

    // trigger the start of the outputs for this shower
    output_.startOfShower();

    setNodes(); // put each particle on stack in correct environment volume

    while (!stack_.isEmpty()) {

      sequence_.initCascadeEquations();

      while (!stack_.isEmpty()) {
        CORSIKA_LOG_TRACE("Stack: {}", stack_.asString());
        count_++;
        auto pNext = stack_.getNextParticle();

        CORSIKA_LOG_TRACE(
            "============== next particle : count={}, pid={}"
            ", stack entries={}"
            ", stack deleted={}",
            count_, pNext.getPID(), stack_.getEntries(), stack_.getErased());

        stepper_.advance(pNext);
        sequence_.doStack(stack_);
      }

      // do cascade equations, which can put new particles on Stack,
      // thus, the double loop
      sequence_.doCascadeEquations(stack_);
    }

    // indicate end of shower
    output_.endOfShower();
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack>
  inline void Cascade<TTracking, TProcessList, TOutput, TStack>::setNodes() {
    stepper_.setNodes();
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack>
  inline void Cascade<TTracking, TProcessList, TOutput, TStack>::forceInteraction() {
    stepper_.forceInteraction();
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack>
  inline void Cascade<TTracking, TProcessList, TOutput, TStack>::forceDecay() {
    stepper_.forceDecay();
  }

} // namespace corsika
