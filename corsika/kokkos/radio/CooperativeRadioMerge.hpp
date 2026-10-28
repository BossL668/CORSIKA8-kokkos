/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/FixedRadioSnapshot.hpp>
#include <corsika/kokkos/radio/RadioWaveforms.hpp>
#include <corsika/kokkos/output/FixedAccumulatorMerge.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace corsika::kokkos::radio {
  inline bool same_observers(std::vector<RadioObserverSnapshot> const& a,
                             std::vector<RadioObserverSnapshot> const& b);
  inline bool same_radio_physics(RadioConfig const& a, RadioConfig const& b);
  inline std::size_t radio_bins(std::vector<RadioObserverSnapshot> const& observers);
  inline void validate_fixed_radio(FixedRadioSnapshot const& s);
  inline RadioCounters merge_radio_counters(RadioCounters a, RadioCounters const& b);
  inline RadioWaveforms decode_fixed_radio(FixedRadioSnapshot const& s);
  /** Accept both endpoint snapshots once; reject mismatch/overflow atomically. */
  class CooperativeRadioMerge {
  public:
    CooperativeRadioMerge(std::string identity, RadioConfig config);
    void commit(FixedRadioSnapshot const& input);
    FixedRadioSnapshot take();

  private:
    std::string identity_;
    RadioConfig config_;
    FixedRadioSnapshot merged_;
    unsigned seen_{};
    bool finished_{};
  };
} // namespace corsika::kokkos::radio

#include <corsika/detail/kokkos/radio/CooperativeRadioMerge.inl>
