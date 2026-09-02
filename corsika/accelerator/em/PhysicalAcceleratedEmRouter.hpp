/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Compatibility bridge while the established router implementation retains
// its original include path.  The implementation is device neutral and its
// backend template parameter is deduced at host wavefront boundaries.
#include <corsika/gpu/em/PhysicalCudaEmRouter.hpp>
