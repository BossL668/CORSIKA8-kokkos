/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Annotate shared, dependency-free algorithms without requiring a Kokkos build.
// Inline remains explicit at each definition; this does not force inlining.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CORSIKA_HOST_DEVICE __host__ __device__
#else
#define CORSIKA_HOST_DEVICE
#endif
