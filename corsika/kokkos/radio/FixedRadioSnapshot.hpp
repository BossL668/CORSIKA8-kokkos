/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/RadioConfig.hpp>
#include <corsika/kokkos/radio/RadioProjectionData.hpp>
#include <corsika/kokkos/scheduling/CooperativeEndpoint.hpp>
#include <string>
#include <vector>

namespace corsika::kokkos::radio {
  /** Owned final integer waveforms for one shower endpoint; ZHS stores potential. */
  struct FixedRadioSnapshot {
    std::string showerIdentity_;
    CooperativeEndpoint endpoint_{};
    RadioConfig config_;
    std::vector<std::int64_t> coreas_, zhs_;
    RadioCounters counters_{};
  };
} // namespace corsika::kokkos::radio
