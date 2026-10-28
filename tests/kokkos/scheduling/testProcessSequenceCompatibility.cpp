/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/ProcessSequenceCompatibility.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <corsika/modules/thinning/EMThinning.hpp>
#include <corsika/modules/proposal/ContinuousProcess.hpp>
#include <corsika/modules/proposal/InteractionModel.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#ifdef CORSIKA_PRODUCTION_SCHEDULING_REFERENCE
#include "ProductionRegistry.hpp"
#endif

using namespace corsika;
namespace cp = corsika::kokkos;
namespace {
  struct Continuous : ContinuousProcess<Continuous> {};
  struct Secondaries : SecondariesProcess<Secondaries> {};
  struct Interaction : InteractionProcess<Interaction> {};
  struct Decay : DecayProcess<Decay> {};
  struct Boundary : BoundaryCrossingProcess<Boundary> {};
  struct StackOperation : StackProcess<StackOperation> {};
  struct CascadeEquations : CascadeEquationsProcess<CascadeEquations> {};
  struct Unused {};
  struct Condition {};
  using sequence_type = ProcessSequence<
      StackOperation,
      ProcessSequence<ProcessSequence<Continuous&, Secondaries const&>,
                      SwitchProcessSequence<
                          Condition, ProcessSequence<Interaction, Decay>, Boundary>>>;
  template <unsigned Mask, unsigned Bit, typename T>
  using selected_type = std::conditional_t<(Mask & Bit) != 0, T, Unused>;
  template <typename T>
  using registration_type =
      cp::EmStepProcessRegistration<T, cp::EmStepProcessPolicy::DeferredToCpu>;
  template <unsigned Mask>
  using registry_type = cp::EmStepProcessRegistry<
      registration_type<selected_type<Mask, 1, Continuous>>,
      registration_type<selected_type<Mask, 2, Secondaries>>,
      registration_type<selected_type<Mask, 4, Interaction>>,
      registration_type<selected_type<Mask, 8, Decay>>,
      registration_type<selected_type<Mask, 16, Boundary>>,
      registration_type<selected_type<Mask, 32, StackOperation>>>;
  template <typename R, typename S>
  constexpr auto counts() {
    return std::array<std::size_t, 6>{
        R::template unregisteredContinuousProcessCount<S>(),
        R::template unregisteredSecondariesProcessCount<S>(),
        R::template unregisteredInteractionProcessCount<S>(),
        R::template unregisteredDecayProcessCount<S>(),
        R::template unregisteredBoundaryProcessCount<S>(),
        R::template unregisteredStackProcessCount<S>()};
  }
  template <unsigned Mask>
  void check_mask() {
    CAPTURE(Mask);
    using r = registry_type<Mask>;
    auto c = counts<r, sequence_type const&>();
    for (unsigned j = 0; j < 6; ++j) CHECK(c[j] == ((Mask & (1U << j)) ? 0 : 1));
    CHECK(r::template compatible<sequence_type>() == (Mask == 63));
    if constexpr (Mask == 63) {
      CHECK_NOTHROW(r::template validateOrThrow<sequence_type>());
    } else {
      CHECK_THROWS_AS(r::template validateOrThrow<sequence_type>(), std::runtime_error);
    }
#ifdef CORSIKA_PRODUCTION_SCHEDULING_REFERENCE
    namespace old = production_scheduling_reference;
    using original = old::GpuEmStepProcessRegistry<
        old::GpuEmStepProcessRegistration<selected_type<Mask, 1, Continuous>,
                                          old::GpuEmStepProcessPolicy::DeferredToCpu>,
        old::GpuEmStepProcessRegistration<selected_type<Mask, 2, Secondaries>,
                                          old::GpuEmStepProcessPolicy::DeferredToCpu>,
        old::GpuEmStepProcessRegistration<selected_type<Mask, 4, Interaction>,
                                          old::GpuEmStepProcessPolicy::DeferredToCpu>,
        old::GpuEmStepProcessRegistration<selected_type<Mask, 8, Decay>,
                                          old::GpuEmStepProcessPolicy::DeferredToCpu>,
        old::GpuEmStepProcessRegistration<selected_type<Mask, 16, Boundary>,
                                          old::GpuEmStepProcessPolicy::DeferredToCpu>,
        old::GpuEmStepProcessRegistration<selected_type<Mask, 32, StackOperation>,
                                          old::GpuEmStepProcessPolicy::DeferredToCpu>>;
    CHECK((c == counts<original, sequence_type>()));
    CHECK(r::template compatible<sequence_type>() ==
          original::template compatible<sequence_type>());
#endif
    if constexpr (Mask < 63) check_mask<Mask + 1>();
  }
} // namespace

TEST_CASE(
    "All six process categories are checked in nested sequences and both switch branches",
    "[process_registry]") {
  check_mask<0>();
  using repeated = ProcessSequence<Continuous, ProcessSequence<Continuous, Continuous>>;
  CHECK(cp::EmStepProcessRegistry<>::unregisteredContinuousProcessCount<repeated>() == 3);
  using whole = cp::EmStepProcessRegistry<registration_type<sequence_type>>;
  CHECK(whole::compatible<sequence_type>());
  // Production permits explicit registration of an entire subtree.
  // This is a declaration of responsibility, not proof of an implementation.
  CHECK(whole::registrationCount() == 1);
  CHECK(cp::EmStepProcessRegistry<>::compatible<NullModel>());
  CHECK(cp::EmStepProcessRegistry<>::compatible<CascadeEquations>());
}

TEST_CASE(
    "Process policies remain explicit and native PROPOSAL cut and thinning require "
    "registration",
    "[process_registry]") {
  using policies = cp::EmStepProcessRegistry<
      cp::EmStepProcessRegistration<Continuous,
                                    cp::EmStepProcessPolicy::ReplacedOnDevice>,
      cp::EmStepProcessRegistration<Secondaries,
                                    cp::EmStepProcessPolicy::ReplayedFromDeviceRecord>,
      cp::EmStepProcessRegistration<Interaction, cp::EmStepProcessPolicy::DeferredToCpu>,
      cp::EmStepProcessRegistration<Decay,
                                    cp::EmStepProcessPolicy::InapplicableToRoutedEm>,
      cp::EmStepProcessRegistration<Boundary, cp::EmStepProcessPolicy::DiagnosticOnly>,
      cp::EmStepProcessRegistration<StackOperation,
                                    cp::EmStepProcessPolicy::DeferredToCpu>>;
  static_assert(policies::compatible<sequence_type>());
  CHECK(policies::registrationCount() == 6);
  CHECK(policies::replacedOnDeviceCount() == 1);
  CHECK(policies::replayedFromDeviceRecordCount() == 1);
  CHECK(policies::deferredToCpuCount() == 2);
  CHECK(policies::inapplicableToRoutedEmCount() == 1);
  CHECK(policies::diagnosticOnlyCount() == 1);
  using native =
      ProcessSequence<proposal::ContinuousProcess<>,
                      ProcessSequence<proposal::InteractionModel<NullModel, NullModel>,
                                      ProcessSequence<ParticleCut<>, EMThinning>>>;
  using registered = cp::EmStepProcessRegistry<
      registration_type<proposal::ContinuousProcess<>>,
      registration_type<proposal::InteractionModel<NullModel, NullModel>>,
      registration_type<ParticleCut<>>, registration_type<EMThinning>>;
  CHECK_FALSE(cp::EmStepProcessRegistry<>::compatible<native>());
  CHECK_THROWS(cp::EmStepProcessRegistry<>::validateOrThrow<native>());
  CHECK_NOTHROW(registered::validateOrThrow<native>());
  using missing_thinning = cp::EmStepProcessRegistry<
      registration_type<proposal::ContinuousProcess<>>,
      registration_type<proposal::InteractionModel<NullModel, NullModel>>,
      registration_type<ParticleCut<>>>;
  CHECK(missing_thinning::unregisteredSecondariesProcessCount<native>() == 1);
  CHECK_THROWS(missing_thinning::validateOrThrow<native>());
}
