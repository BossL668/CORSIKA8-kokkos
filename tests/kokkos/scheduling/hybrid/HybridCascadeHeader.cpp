/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/scheduling/HybridCascade.hpp>

static_assert(
    std::is_default_constructible_v<corsika::kokkos::HybridHadronicWorkerConfig>);
