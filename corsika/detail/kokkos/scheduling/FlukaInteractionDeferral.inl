/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  inline bool FlukaInteractionDeferral::tryPrepare(Code const projectileId,
                                                   Code const targetId,
                                                   FourMomentum const& projectileP4,
                                                   FourMomentum const& targetP4) {
    if (auto* context = HadronicInteractionDeferralContext::current();
        context != nullptr &&
        context->tryPrepare(HadronicWorkerModel::Fluka, projectileId, targetId,
                            projectileP4, targetP4)) {
      return true;
    }
    return false;
  }

  template <typename TProcess,
            std::enable_if_t<std::is_same_v<TProcess, fluka::InteractionModel> ||
                                 std::is_same_v<TProcess, fluka::Interaction>,
                             int>>
  inline InteractionCounter<TProcess, FlukaInteractionDeferral>
  make_fluka_interaction_counter(TProcess& process) {
    return InteractionCounter<TProcess, FlukaInteractionDeferral>{process};
  }

} // namespace corsika::kokkos
