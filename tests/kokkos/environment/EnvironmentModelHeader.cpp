/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

// No preceding includes or native model dependencies.
#include <corsika/kokkos/environment/EnvironmentModel.hpp>

#include <cstdint>
#include <type_traits>

using model_type = corsika::kokkos::EnvironmentModel;
static_assert(std::is_standard_layout_v<model_type>);
static_assert(std::is_trivially_copyable_v<model_type>);
static_assert(std::is_aggregate_v<model_type>);
static_assert(std::is_same_v<std::underlying_type_t<model_type::Density>, std::uint32_t>);
static_assert(
    std::is_same_v<std::underlying_type_t<model_type::Refractivity>, std::uint32_t>);

namespace {
  constexpr bool has_default_values() {
    model_type const model{};
    if (model.densityKind_ != model_type::Density::Invalid ||
        model.refractivityKind_ != model_type::Refractivity::Unavailable ||
        model.mediumId_ != -1 || model.compositionBegin_ != 0 ||
        model.compositionCount_ != 0 || model.referenceDensityGPerCm3_ != 0. ||
        model.scaleLengthM_ != 0. || model.inverseScaleLengthPerM_ != 0. ||
        model.referenceHeightM_ != 0. || model.refractivityValue_ != 0. ||
        model.inverseReferenceDensityCm3PerG_ != 0.) {
      return false;
    }
    for (int component = 0; component < 3; ++component) {
      if (model.anchorM_[component] != 0. || model.axis_[component] != 0. ||
          model.magneticFieldT_[component] != 0.) {
        return false;
      }
    }
    return true;
  }
} // namespace

static_assert(has_default_values());
