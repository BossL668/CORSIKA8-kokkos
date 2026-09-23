/* (c) Copyright 2026 CORSIKA Project. BSD-3-Clause; see LICENSE. */
#pragma once
#include <cmath>
#include <corsika/accelerator/AcceleratorMacros.hpp>

namespace corsika {
  // Free, unpolarized e+ e- at rest: two photons of mass-energy m_e,
  // back-to-back on an isotropic axis. The cut-endpoint approximation first
  // deposits the untracked positron kinetic energy locally. The target
  // electron contributes a separate m_e medium source, NOT deposited heat.
  // No positronium lifetime, binding correction or three-photon branch.
  C8_ACCELERATOR_INLINE_FUNCTION inline void positronAtRestDirection(
      double u, double azimuth_u, double* direction) {
    auto const cosine = 2. * u - 1.;
    auto const sine = ::sqrt((1. - cosine) * (1. + cosine));
    auto const phi = 6.283185307179586476925286766559 * azimuth_u;
    direction[0] = sine * ::cos(phi);
    direction[1] = sine * ::sin(phi);
    direction[2] = cosine;
  }
}
