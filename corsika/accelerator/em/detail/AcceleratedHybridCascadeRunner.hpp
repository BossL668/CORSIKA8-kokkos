/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// The implementation remains in its historical path for source compatibility.
// It no longer includes or assumes CudaEmBackend: the concrete backend is
// deduced once at the host wavefront boundary.
#include <corsika/gpu/em/detail/CudaHybridCascadeRunner.hpp>
