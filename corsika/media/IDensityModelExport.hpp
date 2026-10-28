/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/media/DensityModelParameters.hpp>

namespace corsika {

  /**
   * Optional host-side interface for exact density-parameter export.
   *
   * No sampling, parameter fitting or device execution is performed by this
   * interface. An inherited exporter must remain consistent with the model's
   * density and grammage methods: a subclass changing those methods must also
   * provide a matching exporter, or must not be used through this interface.
   */
  class IDensityModelExport {
  public:
    virtual ~IDensityModelExport() = default;

    /**
     * Return the current stored density parameters with their units and frames.
     *
     * @returns An independent parameter value; no source model is modified.
     */
    [[nodiscard]] virtual DensityModelParameters exportDensity() const = 0;
  };

} // namespace corsika
