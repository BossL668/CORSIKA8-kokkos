/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/scheduling/FlukaInteractionDeferral.hpp>
#include <corsika/framework/process/InteractionCounter.hpp>

namespace counter_fixture {
  bool header_adapter_first(corsika::Code projectile, corsika::Code target,
                            corsika::FourMomentum const& projectileP4,
                            corsika::FourMomentum const& targetP4) {
    return corsika::kokkos::FlukaInteractionDeferral::tryPrepare(projectile, target,
                                                                 projectileP4, targetP4);
  }
} // namespace counter_fixture
