/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/media/RefractivityModelParameters.hpp>

namespace corsika {

  /**
   * Optional host-side interface for exact refractive-index parameter export.
   *
   * Parameters are read from stored state without sampling or fitting. An inherited
   * exporter must remain consistent with getRefractiveIndex(); subclasses replacing
   * that model must also provide a matching exporter, or must not use this interface.
   */
  class IRefractivityModelExport {
  public:
    virtual ~IRefractivityModelExport() = default;

    /**
     * Return the current stored parameters without combining their factors.
     *
     * @returns A parameter value retaining the original density unit information.
     */
    [[nodiscard]] virtual RefractivityModelParameters exportRefractivity() const = 0;
  };

} // namespace corsika
