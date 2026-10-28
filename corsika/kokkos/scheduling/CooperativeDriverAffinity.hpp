/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <functional>

namespace corsika::kokkos {
  inline std::function<void()> capture_cooperative_driver_affinity();
}
#include <corsika/detail/kokkos/scheduling/CooperativeDriverAffinity.inl>
