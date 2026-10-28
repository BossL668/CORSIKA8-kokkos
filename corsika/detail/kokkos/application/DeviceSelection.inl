/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::application {
  inline bool DeviceSelection::hasMultipleDevices() const { return ids_.size() > 1; }
} // namespace corsika::kokkos::application
