/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/process/InteractionCounter.hpp>
#include <corsika/kokkos/scheduling/HadronicInteractionDeferral.hpp>
#include <type_traits>

namespace corsika::fluka {
  class InteractionModel;
  class Interaction;
} // namespace corsika::fluka

namespace corsika::kokkos {

  /** Explicit policy for process-isolated FLUKA final-state generation.
   * A scheduling scope must be active for capture. Merely including this header
   * cannot change an ordinary InteractionCounter's behavior in any translation unit.
   */
  struct FlukaInteractionDeferral {
    static bool tryPrepare(Code const projectileId, Code const targetId,
                           FourMomentum const& projectileP4,
                           FourMomentum const& targetP4);
  };

  /** Select the FLUKA policy for the native model or its ProcessSequence wrapper. */
  template <typename TProcess,
            std::enable_if_t<std::is_same_v<TProcess, fluka::InteractionModel> ||
                                 std::is_same_v<TProcess, fluka::Interaction>,
                             int> = 0>
  InteractionCounter<TProcess, FlukaInteractionDeferral> make_fluka_interaction_counter(
      TProcess& process);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/FlukaInteractionDeferral.inl>
