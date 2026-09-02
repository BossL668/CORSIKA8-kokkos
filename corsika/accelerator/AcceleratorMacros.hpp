/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#if defined(CORSIKA8_WITH_KOKKOS) &&                              \
    (defined(CORSIKA8_KOKKOS_BACKEND_OPENMP) || defined(__CUDACC__) || \
     defined(__HIPCC__) || defined(CORSIKA8_KOKKOS_BACKEND_SYCL))
#include <Kokkos_Macros.hpp>
#define C8_ACCELERATOR_INLINE_FUNCTION KOKKOS_FUNCTION
#elif defined(__CUDACC__) || defined(__HIPCC__)
#define C8_ACCELERATOR_INLINE_FUNCTION __host__ __device__
#else
#define C8_ACCELERATOR_INLINE_FUNCTION
#endif
