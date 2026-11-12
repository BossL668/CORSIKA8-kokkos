/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/framework/process/ProcessSelection.hpp>
#include <corsika/framework/process/ProcessTraits.hpp>
#include <corsika/framework/process/SwitchProcessSequence.hpp>

namespace corsika {
  template <typename T>
  class StackInspector;
  template <typename T>
  class LongitudinalProfile;
  template <typename T>
  class ProductionProfile;
  template <typename T>
  class ParticleCut;
  template <typename T, typename U>
  class ObservationPlane;
  template <typename T, typename U>
  class InteractionWriter;
  template <typename T, typename U>
  class InteractionCounter;
  class EMThinning;
  namespace pythia8 {
    class NeutrinoInteraction;
  }
  namespace proposal {
    template <typename T>
    class ContinuousProcess;
  }
} // namespace corsika

namespace corsika::kokkos::application::process_selection {
  template <typename T>
  struct Inspector : std::false_type {};
  template <typename T>
  struct Inspector<::corsika::StackInspector<T>> : std::true_type {};
  template <typename T>
  struct Longitudinal : std::false_type {};
  template <typename T>
  struct Longitudinal<::corsika::LongitudinalProfile<T>> : std::true_type {};
  template <typename T>
  struct Production : std::false_type {};
  template <typename T>
  struct Production<::corsika::ProductionProfile<T>> : std::true_type {};
  template <typename T>
  struct Cut : std::false_type {};
  template <typename T>
  struct Cut<::corsika::ParticleCut<T>> : std::true_type {};
  template <typename T>
  struct Observation : std::false_type {};
  template <typename T, typename U>
  struct Observation<::corsika::ObservationPlane<T, U>> : std::true_type {};
  template <typename T>
  struct Interactions : std::false_type {};
  template <typename T, typename U>
  struct Interactions<::corsika::InteractionWriter<T, U>> : std::true_type {};
  template <typename T>
  struct Counter : std::false_type {};
  template <typename T, typename U>
  struct Counter<::corsika::InteractionCounter<T, U>> : std::true_type {};
  template <typename T>
  struct ProposalContinuous : std::false_type {};
  template <typename T>
  struct ProposalContinuous<::corsika::proposal::ContinuousProcess<T>> : std::true_type {
  };
  template <typename T>
  struct Decay : std::bool_constant<is_decay_process_v<T>> {};
  template <typename T>
  struct Thinning : std::is_same<T, ::corsika::EMThinning> {};
  template <typename T>
  struct Neutrino : std::is_same<T, ::corsika::pythia8::NeutrinoInteraction> {};

  template <typename T>
  struct HadronSwitch : std::false_type {};
  template <typename TC, typename TA, typename TB, int Start, int First, int Second>
  struct HadronSwitch<::corsika::SwitchProcessSequence<TC, TA, TB, Start, First, Second>>
      : std::bool_constant<contains_process_v<Counter, std::remove_reference_t<TA>> &&
                           contains_process_v<Counter, std::remove_reference_t<TB>>> {};
  template <typename T>
  struct DecaySwitch : std::false_type {};
  template <typename TC, typename TA, typename TB, int Start, int First, int Second>
  struct DecaySwitch<::corsika::SwitchProcessSequence<TC, TA, TB, Start, First, Second>>
      : std::bool_constant<contains_process_v<Decay, std::remove_reference_t<TA>> &&
                           contains_process_v<Decay, std::remove_reference_t<TB>>> {};
  template <typename T>
  struct ContinuousSwitch : std::false_type {};
  template <typename TC, typename TA, typename TB, int Start, int First, int Second>
  struct ContinuousSwitch<
      ::corsika::SwitchProcessSequence<TC, TA, TB, Start, First, Second>>
      : std::bool_constant<
            contains_process_v<ProposalContinuous, std::remove_reference_t<TA>> ||
            contains_process_v<ProposalContinuous, std::remove_reference_t<TB>>> {};
} // namespace corsika::kokkos::application::process_selection
