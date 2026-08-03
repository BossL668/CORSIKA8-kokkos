/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/core/HadronicInteractionDeferral.hpp>
#include <corsika/framework/core/Step.hpp>

#include <corsika/framework/process/ContinuousProcessIndex.hpp>
#include <corsika/framework/process/ContinuousProcessStepLength.hpp>

#include <corsika/framework/random/ExponentialDistribution.hpp>
#include <corsika/framework/random/UniformRealDistribution.hpp>

#include <corsika/framework/stack/SecondaryView.hpp>
#include <corsika/validation/CudaDecisionReplayTape.hpp>

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>

namespace corsika {

  template <typename TTracking, typename TProcessList, typename TStack>
  inline ScalarCascadeStepper<TTracking, TProcessList, TStack>::ScalarCascadeStepper(
      Environment<medium_interface_type> const& env, TTracking& tr, TProcessList& pl,
      TStack& stack)
      : environment_(env)
      , tracking_(tr)
      , sequence_(pl)
      , stack_(stack) {}

  template <typename TTracking, typename TProcessList, typename TStack>
  inline void
  ScalarCascadeStepper<TTracking, TProcessList, TStack>::forceInteraction() {
    forceInteraction_ = true;
    if (forceDecay_) {
      CORSIKA_LOG_ERROR("Cannot set forceInteraction when forceDecay is already set");
      throw std::runtime_error(
          "Cannot set forceInteraction when forceDecay is already set");
    }
  }

  template <typename TTracking, typename TProcessList, typename TStack>
  inline void ScalarCascadeStepper<TTracking, TProcessList, TStack>::forceDecay() {
    forceDecay_ = true;
    if (forceInteraction_) {
      CORSIKA_LOG_ERROR("Cannot set forceDecay when forceInteraction is already set");
      throw std::runtime_error(
          "Cannot set forceDecay when forceInteraction is already set");
    }
  }

  template <typename TTracking, typename TProcessList, typename TStack>
  inline ScalarCascadeAdvanceResult
  ScalarCascadeStepper<TTracking, TProcessList, TStack>::advance(
      particle_type& particle) {
    using PhaseClock = std::chrono::steady_clock;
    ScalarCascadePhaseTimingStatistics*
        phase_statistics = nullptr;
    if (detailed_phase_timing_enabled_) {
      auto const pdg = static_cast<std::int32_t>(
          get_PDG(particle.getPID()));
      phase_statistics =
          &detailed_phase_timing_[pdg];
      ++phase_statistics->steps;
    }
    auto phaseStart = [&]() {
      return phase_statistics
                 ? PhaseClock::now()
                 : PhaseClock::time_point{};
    };
    auto recordPhase =
        [&](double ScalarCascadePhaseTimingStatistics::*member,
            PhaseClock::time_point const start) {
          if (phase_statistics) {
            phase_statistics->*member +=
                std::chrono::duration<double, std::milli>(
                    PhaseClock::now() - start)
                    .count();
          }
        };

    // determine the volume where the particle is (last) known to be
    auto phase_start = phaseStart();
    auto const* currentLogicalNode = particle.getNode();

    // assert that particle stays outside void Universe if it has no
    // model properties set
    assert((currentLogicalNode != &*environment_.getUniverse() ||
            environment_.getUniverse()->hasModelProperties()) &&
           "FATAL: The environment model has no valid properties set!");

    NuclearComposition const& composition =
        currentLogicalNode->getModelProperties().getNuclearComposition();

    // determine projectile
    HEPEnergyType const Elab = particle.getEnergy();
    FourMomentum const projectileP4{Elab, particle.getMomentum()};

    // determine combined full inelastic cross section of the particles in the material
    auto const targetMomentum = MomentumVector{
        particle.getMomentum().getCoordinateSystem(), {0_GeV, 0_GeV, 0_GeV}};

    auto const xs_function = [&](Code const targetId) -> CrossSectionType {
      FourMomentum const targetP4{get_mass(targetId), targetMomentum};
      return sequence_.getCrossSection(particle, targetId, targetP4);
    };

    CrossSectionType const total_cx_pre = composition.getWeightedSum(xs_function);
    recordPhase(
        &ScalarCascadePhaseTimingStatistics::
            cross_section_time_ms,
        phase_start);

    if (forceInteraction_) {
      CORSIKA_LOG_TRACE("forced interaction!");
      forceInteraction_ = false; // just one (first) interaction
      stack_view_type secondaries(particle);
      interaction(secondaries, projectileP4, composition, total_cx_pre);
      if (auto* context =
              HadronicInteractionDeferralContext::current();
          context != nullptr &&
          context->hasPrepared()) {
        return ScalarCascadeAdvanceResult::
            DeferredHadronicInteraction;
      }
      sequence_.doSecondaries(secondaries);
      particle.erase(); // primary particle is done
      return ScalarCascadeAdvanceResult::Completed;
    }

    if (forceDecay_) {
      CORSIKA_LOG_TRACE("forced decay!");
      forceDecay_ = false; // just one decay
      stack_view_type secondaries(particle);
      decay(secondaries, sequence_.getInverseLifetime(particle));
      if (secondaries.getSize() == 1 && secondaries.getProjectile().getPID() ==
                                            secondaries.getNextParticle().getPID()) {
        throw std::runtime_error(
            fmt::format("Particle {} decays into itself!",
                        get_name(secondaries.getProjectile().getPID())));
      }
      sequence_.doSecondaries(secondaries);
      particle.erase(); // primary particle is done
      return ScalarCascadeAdvanceResult::Completed;
    }

    // calculate interaction length in medium
    phase_start = phaseStart();
    GrammageType const total_lambda =
        (composition.getAverageMassNumber() * constants::u) / total_cx_pre;

    // sample random exponential step length in grammage
    ExponentialDistribution expDist{total_lambda};
    GrammageType const next_interact = expDist(rng_);

    CORSIKA_LOG_DEBUG("total_lambda={} g/cm2, next_interact={} g/cm2",
                      double(total_lambda / 1_g * 1_cm * 1_cm),
                      double(next_interact / 1_g * 1_cm * 1_cm));

    // determine combined total inverse decay time
    InverseTimeType const total_inv_lifetime_pre = sequence_.getInverseLifetime(particle);

    // sample random exponential decay time
    ExponentialDistribution expDistDecay(1 / total_inv_lifetime_pre);
    TimeType const next_decay = expDistDecay(rng_);

    CORSIKA_LOG_DEBUG("total_lifetime={} ns, next_decay={} ns",
                      (1 / total_inv_lifetime_pre) / 1_ns, next_decay / 1_ns);

    // convert next_decay from time to length [m]
    LengthType const distance_decay = next_decay * particle.getMomentum().getNorm() /
                                      particle.getEnergy() * constants::c;
    recordPhase(
        &ScalarCascadePhaseTimingStatistics::
            distance_sampling_time_ms,
        phase_start);

    // determine geometric tracking
    phase_start = phaseStart();
    auto [track, nextVol] = tracking_.getTrack(particle);
    auto geomMaxLength = track.getLength(1);

    // convert next_step from grammage to length
    LengthType const distance_interact =
        currentLogicalNode->getModelProperties().getArclengthFromGrammage(track,
                                                                          next_interact);
    recordPhase(
        &ScalarCascadePhaseTimingStatistics::
            tracking_time_ms,
        phase_start);

    // determine the maximum geometric step length
    phase_start = phaseStart();
    ContinuousProcessStepLength const continuousMaxStep =
        sequence_.getMaxStepLength(particle, track);
    LengthType const continuous_max_dist = continuousMaxStep;
    recordPhase(
        &ScalarCascadePhaseTimingStatistics::
            continuous_limit_time_ms,
        phase_start);

    // take minimum of geometry, interaction, decay for next step
    LengthType const min_discrete = std::min(distance_interact, distance_decay);
    LengthType const min_non_continuous = std::min(min_discrete, geomMaxLength);
    LengthType const min_distance = std::min(min_non_continuous, continuous_max_dist);

    bool const isContinuous = continuous_max_dist < min_non_continuous;

    // inform ContinuousProcesses (if applicable) that it is responsible for step-limit
    // this would become simpler if we follow the idea of Max to enumerate ALL types of
    // processes. Then non-continuous are included and no further logic is needed to
    // distinguish between continuous and non-continuous limit.
    auto const limitingId = isContinuous ? continuousMaxStep : ContinuousProcessIndex{};
    // // the current step IS limited by a known continuous process

    CORSIKA_LOG_DEBUG(
        "transport particle by : {} m "
        "Medium transition after: {} m "
        "Decay after: {} m "
        "Interaction after: {} m "
        "Continuous limit: {} m ",
        min_distance / 1_m, geomMaxLength / 1_m, distance_decay / 1_m,
        distance_interact / 1_m, continuous_max_dist / 1_m);

    // move particle along the trajectory to new position
    // also update momentum/direction/time
    track.setLength(min_distance);

    Step step{particle, track};

    auto const replay_limit = [&]() {
      using validation::ReplayTransportLimit;
      if (isContinuous) {
        return ReplayTransportLimit::Continuous;
      }
      if (geomMaxLength < min_discrete) {
        return ReplayTransportLimit::Geometry;
      }
      if (distance_interact < distance_decay) {
        return ReplayTransportLimit::Interaction;
      }
      return ReplayTransportLimit::Decay;
    }();

    // apply all continuous processes on particle + track
    phase_start = phaseStart();
    auto const continuous_result =
        sequence_.doContinuous(step, limitingId);
    recordPhase(
        &ScalarCascadePhaseTimingStatistics::
            continuous_process_time_ms,
        phase_start);
    validation::CudaDecisionReplayTape::instance().recordTransportStep(
        step, replay_limit, continuous_result);
    if (continuous_result == ProcessReturn::ParticleAbsorbed) {
      CORSIKA_LOG_DEBUG("Cascade: delete absorbed particle PID={} E={} GeV",
                        particle.getPID(), particle.getEnergy() / 1_GeV);
      if (particle.isErased()) {
        CORSIKA_LOG_WARN(
            "Particle marked as Absorbed in doContinuous, but prematurely erased. This "
            "may be bug. Check.");
      } else {
        particle.erase();
      }
      return ScalarCascadeAdvanceResult::Completed;
      // particle is gone
    }
    particle.setTime(step.getTimePost());
    particle.setPosition(step.getPositionPost());
    particle.setDirection(step.getDirectionPost());
    particle.setKineticEnergy(step.getEkinPost());

    if (isContinuous) {
      return ScalarCascadeAdvanceResult::Completed;
    }

    CORSIKA_LOG_DEBUG("discrete process before geometric limit ? {}",
                      ((min_distance < geomMaxLength) ? "yes" : "no"));

    if (geomMaxLength < min_discrete) {
      // geometric / tracking limit
      phase_start = phaseStart();

      if (nextVol != currentLogicalNode) {
        // boundary crossing, step is limited by volume boundary

        CORSIKA_LOG_DEBUG("volume boundary crossing to {}", fmt::ptr(nextVol));

        if (nextVol == environment_.getUniverse().get()) {
          CORSIKA_LOG_DEBUG(
              "particle left physics world, is now in unknown space -> delete");
          particle.erase();
        }
        particle.setNode(nextVol);
        /*
          doBoundary may delete the particle (or not)

          caveat: any changes to particle, or even the production
          of new secondaries is currently not passed to ParticleCut,
          thus, particles outside the desired phase space may be produced.

          \todo: this must be fixed.
        */

        sequence_.doBoundaryCrossing(particle, *currentLogicalNode, *nextVol);
        if (phase_statistics) {
          ++phase_statistics->geometry_boundary_crossings;
        }
        recordPhase(
            &ScalarCascadePhaseTimingStatistics::
                geometry_boundary_time_ms,
            phase_start);
        return ScalarCascadeAdvanceResult::Completed;
      }

      CORSIKA_LOG_DEBUG("step limit reached (e.g. deflection). nothing further happens.");

      // Final numerical containment sanity check. Keep this in every build: it is
      // part of the original scalar execution path, and removing even an apparently
      // read-only floating-point/geometry call can change a fixed-seed shower after
      // compiler optimization and chaotic cascade amplification.
      {
        auto const* numericalNodeAfterStep =
            environment_.getUniverse()->getContainingNode(particle.getPosition());
        CORSIKA_LOG_TRACE(
            "Geometry check: numericalNodeAfterStep={} currentLogicalNode={}",
            fmt::ptr(numericalNodeAfterStep), fmt::ptr(currentLogicalNode));
        if (numericalNodeAfterStep != currentLogicalNode) {
          CORSIKA_LOG_DEBUG(
              "expect to be in node currentLogicalNode={} but are in "
              "numericalNodeAfterStep={}. Continue, but without guarantee.",
              fmt::ptr(currentLogicalNode), fmt::ptr(numericalNodeAfterStep));
        }
      }
      // we did not cross any volume boundary

      // step length limit
      if (phase_statistics) {
        ++phase_statistics->geometry_step_limits;
      }
      recordPhase(
          &ScalarCascadePhaseTimingStatistics::
              geometry_step_limit_time_ms,
          phase_start);
      return ScalarCascadeAdvanceResult::Completed;
    }

    // interaction or decay to happen in this step
    // the outcome of decay or interaction MAY be a) new particles in
    // secondaries, b) the projectile particle deleted (or
    // changed)

    stack_view_type secondaries{particle};

    /*
      Create SecondaryView object on Stack. The data container
      remains untouched and identical, and 'projectile' is identical
      to 'particle' above this line. However,
      projectile.addSecondaries populate the SecondaryView, which can
      then be used afterwards for further processing. Thus: it is
      important to use projectile/view (and not particle) for Interaction,
      and Decay!
    */

    FourMomentum const projectileP4Post{particle.getEnergy(), particle.getMomentum()};

    bool eraseParticle =
        false; // only erase original particle if it decayed or interacted

    if (distance_interact < distance_decay) {
      if (phase_statistics) {
        ++phase_statistics->interactions;
      }
      auto const interaction_start = phaseStart();
      auto const interactionResult =
          interaction(
              secondaries, projectileP4Post,
              composition, total_cx_pre);
      recordPhase(
          &ScalarCascadePhaseTimingStatistics::
              interaction_time_ms,
          interaction_start);
      if (auto* context =
              HadronicInteractionDeferralContext::current();
          context != nullptr &&
          context->hasPrepared()) {
        return ScalarCascadeAdvanceResult::
            DeferredHadronicInteraction;
      }
      eraseParticle =
          isInteracted(interactionResult);
    } else {
      if (phase_statistics) {
        ++phase_statistics->decays;
      }
      auto const decay_start = phaseStart();
      [[maybe_unused]] auto projectile = secondaries.getProjectile();
      if (decay(secondaries, total_inv_lifetime_pre) == ProcessReturn::Decayed) {
        eraseParticle = true;
        if (secondaries.getSize() == 1 &&
            projectile.getPID() == secondaries.getNextParticle().getPID()) {
          throw std::runtime_error(fmt::format("Particle {} decays into itself!",
                                               get_name(projectile.getPID())));
        }
      }
      recordPhase(
          &ScalarCascadePhaseTimingStatistics::
              decay_time_ms,
          decay_start);
    }

    if (eraseParticle) {
      // doSecondaries() makes sense only if there was an actual event
      if (phase_statistics) {
        ++phase_statistics->secondary_processing_calls;
      }
      auto const secondary_start = phaseStart();
      sequence_.doSecondaries(secondaries);
      recordPhase(
          &ScalarCascadePhaseTimingStatistics::
              secondary_processing_time_ms,
          secondary_start);
      particle.erase();
    }
    return ScalarCascadeAdvanceResult::Completed;
  }

  template <typename TTracking, typename TProcessList, typename TStack>
  inline void
  ScalarCascadeStepper<TTracking, TProcessList, TStack>::
      commitDeferredHadronicInteraction(
          particle_type& particle,
          std::vector<HadronicSecondaryRecord> const&
              records) {
    if (particle.isErased()) {
      throw std::logic_error(
          "Cannot commit a deferred hadronic interaction to an erased projectile");
    }
    stack_view_type secondaries{particle};
    auto const& coordinateSystem =
        particle.getMomentum().getCoordinateSystem();
    for (auto const& record : records) {
      if (!std::isfinite(record.kinetic_energy_GeV) ||
          record.kinetic_energy_GeV < 0. ||
          !std::all_of(
              record.direction.begin(),
              record.direction.end(),
              [](double const value) {
                return std::isfinite(value);
              })) {
        throw std::runtime_error(
            "Cannot commit an invalid hadronic worker secondary");
      }
      auto const pid =
          convert_from_PDG(
              static_cast<PDGCode>(record.pdg));
      DirectionVector direction{
          coordinateSystem,
          {record.direction[0], record.direction[1],
           record.direction[2]}};
      secondaries.addSecondary(
          std::make_tuple(
              pid,
              record.kinetic_energy_GeV * 1_GeV,
              direction));
    }
    setEventType(
        secondaries,
        history::EventType::Interaction);
    sequence_.doSecondaries(secondaries);
    particle.erase();
  }

  template <typename TTracking, typename TProcessList, typename TStack>
  inline ProcessReturn ScalarCascadeStepper<TTracking, TProcessList, TStack>::decay(
      stack_view_type& view, InverseTimeType initial_inv_decay_time) {
    CORSIKA_LOG_DEBUG("decay");

    // one option is that decay_time is now larger (less
    // probability for decay) than it was before the step, thus,
    // no decay might actually occur and is allowed

    UniformRealDistribution<InverseTimeType> uniDist(initial_inv_decay_time);
    const auto sample_process = uniDist(rng_);

    auto const returnCode = sequence_.selectDecay(view, sample_process);
    if (returnCode != ProcessReturn::Decayed) {
      CORSIKA_LOG_ERROR("Particle {} did not decay!",
                        get_name(view.getProjectile().getPID()));
    }
    setEventType(view, history::EventType::Decay);
    return returnCode;
  }

  template <typename TTracking, typename TProcessList, typename TStack>
  inline ProcessReturn ScalarCascadeStepper<TTracking, TProcessList, TStack>::interaction(
      stack_view_type& view, FourMomentum const& projectileP4,
      NuclearComposition const& composition,
      CrossSectionType const initial_cross_section) {

    CORSIKA_LOG_DEBUG("collide");

    // one option is that cross section is now smaller (less
    // probability for collision) than it was before the step, thus,
    // no interaction might actually occur and is allowed

    UniformRealDistribution<CrossSectionType> uniDist(initial_cross_section);
    CrossSectionType const sample_process_by_cx = uniDist(rng_);
    auto const returnCode = sequence_.selectInteraction(view, projectileP4, composition,
                                                        rng_, sample_process_by_cx);
    if (returnCode != ProcessReturn::Interacted) {
      CORSIKA_LOG_DEBUG("Particle did not interact!");
    }
    setEventType(view, history::EventType::Interaction);
    return returnCode;
  }

  template <typename TTracking, typename TProcessList, typename TStack>
  inline void ScalarCascadeStepper<TTracking, TProcessList, TStack>::setNodes() {
    std::for_each(stack_.begin(), stack_.end(), [&](auto& p) {
      auto const* numericalNode =
          environment_.getUniverse()->getContainingNode(p.getPosition());
      p.setNode(numericalNode);
    });
  }

  template <typename TTracking, typename TProcessList, typename TStack>
  inline void ScalarCascadeStepper<TTracking, TProcessList, TStack>::setEventType(
      stack_view_type& view, [[maybe_unused]] history::EventType eventType) {
    if constexpr (stack_view_type::has_event) {
      for (auto&& sec : view) { sec.getEvent()->setEventType(eventType); }
    }
  }

} // namespace corsika
