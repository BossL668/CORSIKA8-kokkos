/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika::kokkos::application {
  template <typename TPhotoHigh, typename TPhotoLow, typename THigh, typename TLow,
            typename TPool, typename TFallback>
  template <typename TH, typename TL>
  auto CascadeMonitors<TPhotoHigh, TPhotoLow, THigh, TLow, TPool, TFallback>::capture(
      TH const& photoHighBefore, TL const& photoLowBefore) const {
    using pool_statistics_type = std::decay_t<decltype(pool_->statistics())>;
    // Braced initialization preserves the old low/high read order. Snapshots
    // are values: later counter changes cannot mutate the event boundary.
    return CascadeBefore<TH, TL, pool_statistics_type>{
        photoHighBefore,
        photoLowBefore,
        low_.getCount(),
        high_.getCount(),
        low_.getTimingSamples().size(),
        high_.getTimingSamples().size(),
        pool_ ? pool_->statistics() : pool_statistics_type{}};
  }

} // namespace corsika::kokkos::application
