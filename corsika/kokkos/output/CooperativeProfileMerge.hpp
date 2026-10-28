/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <corsika/kokkos/output/ProfileProjectionConfig.hpp>
#include <corsika/kokkos/output/ProfileResult.hpp>
#include <corsika/kokkos/output/ProfileAccumulatorView.hpp>
#include <corsika/kokkos/output/FixedAccumulatorMerge.hpp>
#include <corsika/kokkos/scheduling/CooperativeEndpoint.hpp>
#include <corsika/kokkos/em/LeptonTransportRecord.hpp>

namespace corsika::kokkos {

  /** Owned integer profile and scales for one shower/endpoint contribution. */
  struct FixedProfileSnapshot {
    std::string showerIdentity_;
    CooperativeEndpoint endpoint_{};
    ProfileProjectionConfig config_;
    double weightUnits_{}, energyUnits_{};
    std::vector<std::int64_t> histograms_;
    ProfileCounters counters_{};
  };
  inline bool same_profile_config(ProfileProjectionConfig const& a,
                                  ProfileProjectionConfig const& b);
  inline void validate_fixed_profile(FixedProfileSnapshot const& s);
  inline ProfileCounters merge_profile_counters(ProfileCounters a,
                                                ProfileCounters const& b);
  inline ProfileResult decode_fixed_profile(FixedProfileSnapshot const& s);
  /** Accept each CPU/GPU endpoint once and combine matching integer profiles.
   * The shower identity, binning, crossing mode and scales must agree. take()
   * finalizes this merger; it does not launch or synchronize endpoint kernels.
   */
  class CooperativeProfileMerge {
  public:
    CooperativeProfileMerge(std::string identity, ProfileProjectionConfig config);
    void commit(FixedProfileSnapshot const& s);
    FixedProfileSnapshot take();

  private:
    std::string identity_;
    ProfileProjectionConfig config_;
    FixedProfileSnapshot merged_;
    unsigned seen_{};
    bool finished_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/output/CooperativeProfileMerge.inl>
