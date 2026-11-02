/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <hip/hip_runtime.h>

// Keep the existing Random123 rounds and integer operations on HIP devices.
#ifndef RANDOM_ITERATOR_R123_CUDA_DEVICE
#define RANDOM_ITERATOR_R123_CUDA_DEVICE __host__ __device__
#endif
// As in the CUDA feature configuration, avoid host-only SIMD array overloads
// in a translation unit that also declares device-callable Random123 arrays.
#ifndef RANDOM_ITERATOR_R123_USE_SSE
#define RANDOM_ITERATOR_R123_USE_SSE 0
#endif
#ifndef RANDOM_ITERATOR_R123_USE_SSE4_1
#define RANDOM_ITERATOR_R123_USE_SSE4_1 0
#endif
#ifndef RANDOM_ITERATOR_R123_USE_SSE4_2
#define RANDOM_ITERATOR_R123_USE_SSE4_2 0
#endif
#ifndef RANDOM_ITERATOR_R123_USE_AES_NI
#define RANDOM_ITERATOR_R123_USE_AES_NI 0
#endif
#ifndef RANDOM_ITERATOR_R123_USE_X86INTRIN_H
#define RANDOM_ITERATOR_R123_USE_X86INTRIN_H 0
#endif
#if defined(__HIP_DEVICE_COMPILE__)
#ifndef RANDOM_ITERATOR_R123_USE_MULHILO64_CUDA_INTRIN
#define RANDOM_ITERATOR_R123_USE_MULHILO64_CUDA_INTRIN 1
#endif
#ifndef RANDOM_ITERATOR_R123_ASSERT
#define RANDOM_ITERATOR_R123_ASSERT(x) ((x) ? (void)0 : __builtin_trap())
#endif
#ifndef RANDOM_ITERATOR_R123_THROW
#define RANDOM_ITERATOR_R123_THROW(x) RANDOM_ITERATOR_R123_ASSERT(0)
#endif
#endif

#include "clangfeatures.h"
