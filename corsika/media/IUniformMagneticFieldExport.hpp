/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/PhysicalGeometry.hpp>

namespace corsika {

  /**
   * Optional host-side interface for exporting a uniform magnetic field.
   *
   * The vector is copied from model state, not inferred from point samples. An
   * inherited exporter must describe the current field model; a subclass must not
   * retain this uniform export contract after replacing it with a varying field.
   */
  class IUniformMagneticFieldExport {
  public:
    virtual ~IUniformMagneticFieldExport() = default;

    /**
     * Return the stored field without converting units or coordinate systems.
     *
     * @returns The current uniform field vector, including its reference frame.
     */
    [[nodiscard]] virtual MagneticFieldVector exportUniformField() const = 0;
  };

} // namespace corsika
