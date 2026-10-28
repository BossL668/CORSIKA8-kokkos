/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::kokkos {

  /**
   * Pointer-free environment-model parameters in a caller-selected common frame.
   *
   * Density is in g/cm^3, lengths in metres, and magnetic-field components in
   * tesla. axis_ is copied without normalization. The reference refractivity and
   * inverse reference density remain separate factors; no model is evaluated by
   * this record. Homogeneous and radial density models retain their exported
   * placeholder anchor or axis even when the density law does not use them.
   *
   * mediumId_ is the native physical-medium enum value, not a topology model ID.
   * compositionBegin_ and compositionCount_ select a contiguous range in a
   * separately owned EnvironmentComponent array. This record owns no storage or
   * native object references. Re-export after changing source model parameters.
   * Its layout and padding are not a portable serialized representation.
   */
  struct EnvironmentModel {
    enum class Density : std::uint32_t {
      Invalid,
      Homogeneous,
      SlidingPlanarExponential,
      FlatExponential,
      ExponentialFirn
    };
    enum class Refractivity : std::uint32_t {
      Unavailable,
      Uniform,
      GladstoneDale,
      ExponentialFirn
    };

    Density densityKind_{Density::Invalid};
    double referenceDensityGPerCm3_{};
    //! Signed exponential scale length; zero for the native homogeneous model.
    double scaleLengthM_{};
    double inverseScaleLengthPerM_{};
    double referenceHeightM_{};
    double anchorM_[3]{};
    double axis_[3]{};
    double magneticFieldT_[3]{};
    Refractivity refractivityKind_{Refractivity::Unavailable};
    //! Uniform: refractive index n; Gladstone-Dale: reference refractivity n_ref - 1.
    double refractivityValue_{};
    //! Inverse reference density; zero for the native uniform-index model.
    double inverseReferenceDensityCm3PerG_{};
    std::int32_t mediumId_{-1};
    std::uint32_t compositionBegin_{};
    std::uint32_t compositionCount_{};
    //! Firn: reference density is the deep value; subtract this surface deficit.
    double densityDeficitGPerCm3_{};
    //! Firn: refractivityValue_ is the deep index; subtract this surface deficit.
    double refractiveIndexDeficit_{};
  };

} // namespace corsika::kokkos
