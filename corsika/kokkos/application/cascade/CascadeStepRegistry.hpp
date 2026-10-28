/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/em/ProcessSequenceCompatibility.hpp>
namespace corsika::kokkos::application {
  // A type-policy registry; it does not reorder or replace make_sequence.
  template <typename TStackInspector, typename TNeutrinoPrimary, typename THadronSequence,
            typename TDecaySequence, typename TEmCascade, typename TEmContinuous,
            typename TCoreas, typename TZhs, typename TLongitudinalProfile,
            typename TObservationLevel, typename TProductionProfileProcess,
            typename TInteractionWriter, typename TThinning, typename TCut>
  using cascade_step_registry_type = corsika::kokkos::EmStepProcessRegistry<
      corsika::kokkos::EmStepProcessRegistration<
          TStackInspector, corsika::kokkos::EmStepProcessPolicy::DiagnosticOnly>,
      corsika::kokkos::EmStepProcessRegistration<
          TNeutrinoPrimary, corsika::kokkos::EmStepProcessPolicy::InapplicableToRoutedEm>,
      corsika::kokkos::EmStepProcessRegistration<
          THadronSequence, corsika::kokkos::EmStepProcessPolicy::InapplicableToRoutedEm>,
      corsika::kokkos::EmStepProcessRegistration<
          TDecaySequence, corsika::kokkos::EmStepProcessPolicy::DeferredToCpu>,
      corsika::kokkos::EmStepProcessRegistration<
          TEmCascade, corsika::kokkos::EmStepProcessPolicy::ReplacedOnDevice>,
      corsika::kokkos::EmStepProcessRegistration<
          TEmContinuous, corsika::kokkos::EmStepProcessPolicy::ReplacedOnDevice>,
      corsika::kokkos::EmStepProcessRegistration<
          TCoreas, corsika::kokkos::EmStepProcessPolicy::ReplayedFromDeviceRecord>,
      corsika::kokkos::EmStepProcessRegistration<
          TZhs, corsika::kokkos::EmStepProcessPolicy::ReplayedFromDeviceRecord>,
      corsika::kokkos::EmStepProcessRegistration<
          TLongitudinalProfile,
          corsika::kokkos::EmStepProcessPolicy::ReplayedFromDeviceRecord>,
      corsika::kokkos::EmStepProcessRegistration<
          TObservationLevel,
          corsika::kokkos::EmStepProcessPolicy::ReplayedFromDeviceRecord>,
      corsika::kokkos::EmStepProcessRegistration<
          TProductionProfileProcess,
          corsika::kokkos::EmStepProcessPolicy::ReplayedFromDeviceRecord>,
      corsika::kokkos::EmStepProcessRegistration<
          TInteractionWriter,
          corsika::kokkos::EmStepProcessPolicy::ReplayedFromDeviceRecord>,
      corsika::kokkos::EmStepProcessRegistration<
          TThinning, corsika::kokkos::EmStepProcessPolicy::ReplacedOnDevice>,
      corsika::kokkos::EmStepProcessRegistration<
          TCut, corsika::kokkos::EmStepProcessPolicy::ReplacedOnDevice>>;

} // namespace corsika::kokkos::application
