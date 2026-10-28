/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <corsika/kokkos/em/execution/BackendStatistics.hpp>
#include <corsika/kokkos/em/execution/PhotonCascadeResult.hpp>
#include <corsika/kokkos/em/execution/LeptonCascadeResult.hpp>

namespace corsika::kokkos {
  template <class T>
  T cooperative_add(T a, T b);
  inline void merge_cooperative(ProcessStatistics& a, ProcessStatistics b);
  inline void merge_cooperative(PhotonCascadeResult& a, PhotonCascadeResult b);
  inline void merge_cooperative(LeptonCascadeResult& a, LeptonCascadeResult b);
  inline BackendStatistics merge_cooperative_statistics(BackendStatistics a,
                                                        BackendStatistics const& b);
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/scheduling/CooperativeBackendMerge.inl>
