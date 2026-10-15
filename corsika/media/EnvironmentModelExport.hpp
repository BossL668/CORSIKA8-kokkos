/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>

namespace corsika {
  // Optional, initialization-only interfaces. These export model parameters,
  // not samples fitted at a few positions. The original physics methods remain
  // unchanged and unsupported user models cannot be silently approximated.
  enum class ExportDensityKind { Homogeneous, SlidingPlanarExponential, FlatExponential };
  struct ExportDensity {
    ExportDensityKind kind;
    MassDensityType rho0;
    LengthType lambda, reference_height;
    Point anchor;
    DirectionVector axis;
  };
  struct ExportRefractivity {
    enum class Kind { Uniform, GladstoneDale } kind;
    double value;
    // Store the original factor separately to preserve its evaluation order.
    InverseMassDensityType inverse_density;
  };
  struct IDensityExport {
    virtual ~IDensityExport()=default;
    virtual ExportDensity exportDensity() const=0;
  };
  struct IUniformFieldExport {
    virtual ~IUniformFieldExport()=default;
    virtual MagneticFieldVector exportUniformField() const=0;
  };
  struct IRefractivityExport {
    virtual ~IRefractivityExport()=default;
    virtual ExportRefractivity exportRefractivity() const=0;
  };
} // namespace corsika
