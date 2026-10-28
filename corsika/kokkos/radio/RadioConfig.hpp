/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <corsika/kokkos/radio/RadioObserverSnapshot.hpp>
#include <corsika/kokkos/radio/FlatAtmosphereRadioSnapshot.hpp>

namespace corsika::kokkos::radio {
  /** Production radio configuration; constructing it neither allocates device storage
   * nor launches projection. Observer and trajectory coordinates must share a frame.
   * The default fixed-point accumulation and ZHS refinement retain production values. */
  struct RadioConfig {
    bool enabled_{};
    bool coreasEnabled_{true};
    bool zhsEnabled_{true};
    /**
     * Accumulate waveform samples as checked signed fixed-point integers.
     * Integer addition is associative and therefore independent of CUDA
     * warp/block scheduling. This is the production default.
     */
    bool deterministic_{true};
    /**
     * Absolute electric-field range reserved by the fixed-point
     * accumulator. ZHS potential samples use the equivalent range after
     * multiplication by each observer's sample rate.
     */
    double fixedPointFieldLimitVPerM_{1.};
    /**
     * Collect scalar-compatible electron/positron track diagnostics on the
     * device. This adds global reductions to every transport batch and is
     * therefore disabled for production runs unless explicitly requested.
     */
    bool trackDiagnostics_{};
    /**
     * Split every stock ZHS Fraunhofer subtrack into this many equal pieces.
     * The default preserves the released scalar/CUDA projection exactly.
     * Values above one are a validation control: they refine radio projection
     * without changing the transported shower or consuming random numbers.
     */
    std::uint32_t zhsSubtrackRefinement_{1};
    FlatAtmosphereRadioSnapshot propagation_{};
    std::vector<RadioObserverSnapshot> coreasObservers_{};
    std::vector<RadioObserverSnapshot> zhsObservers_{};
  };
} // namespace corsika::kokkos::radio
