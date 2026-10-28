/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/kokkos/application/cascade/CascadeBefore.hpp>
#include <type_traits>
#include <boost/optional.hpp>

namespace corsika::kokkos::application {
  /** Borrow run-local diagnostics without owning or replacing the processes.
   * The referenced processes and optional worker pool must outlive this binding.
   * An absent pool denotes execution without worker-pool diagnostics.
   */
  template <typename TPhotoHigh, typename TPhotoLow, typename THigh, typename TLow,
            typename TPool, typename TFallback>
  struct CascadeMonitors {
    TPhotoHigh& photoHigh_;
    TPhotoLow& photoLow_;
    THigh& high_;
    TLow& low_;
    boost::optional<TPool&> pool_;
    TFallback const& fallback_;

    /** Capture counters before running an event, retaining the supplied photon
     * interaction snapshots and the current hadron and worker-pool counters.
     */
    template <typename TH, typename TL>
    auto capture(TH const& photoHighBefore, TL const& photoLowBefore) const;
  };
  template <typename TPhotoHigh, typename TPhotoLow, typename THigh, typename TLow,
            typename TPool, typename TFallback>
  CascadeMonitors(TPhotoHigh&, TPhotoLow&, THigh&, TLow&, boost::optional<TPool&>,
                  TFallback const&)
      ->CascadeMonitors<TPhotoHigh, TPhotoLow, THigh, TLow, TPool, TFallback>;

} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/CascadeMonitors.inl>
