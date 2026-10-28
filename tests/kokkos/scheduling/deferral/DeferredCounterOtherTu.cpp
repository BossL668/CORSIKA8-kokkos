/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "SharedCounterModel.hpp"
#include <corsika/kokkos/scheduling/FlukaInteractionDeferral.hpp>
#include <numeric>

namespace counter_fixture {
  CounterSnapshot deferred_counter_other_tu(SharedModel& model,
                                            corsika::FourMomentum const& projectile,
                                            corsika::FourMomentum const& target) {
    corsika::InteractionCounter<SharedModel, corsika::kokkos::FlukaInteractionDeferral>
        counter(model);
    SharedView view;
    counter.doInteraction(view, corsika::Code::Proton, corsika::Code::Oxygen, projectile,
                          target);
    auto const& histogram = counter.getHistogram().labHist();
    return {counter.getCount(), counter.getEnergyLedgerStatistics().deferredInteractions_,
            std::accumulate(histogram.begin(), histogram.end(), 0)};
  }
} // namespace counter_fixture
