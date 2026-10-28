/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentComponent.hpp>
#include <corsika/kokkos/environment/EnvironmentModel.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <array>
#include <type_traits>

namespace {
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using model_type = corsika::kokkos::EnvironmentModel;
  using component_type = corsika::kokkos::EnvironmentComponent;
  using density_type = model_type::Density;
  using refractivity_type = model_type::Refractivity;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif

  void check_model(model_type const& actual, model_type const& expected) {
    REQUIRE(actual.densityKind_ == expected.densityKind_);
    REQUIRE(actual.refractivityKind_ == expected.refractivityKind_);
    REQUIRE(actual.referenceDensityGPerCm3_ == expected.referenceDensityGPerCm3_);
    REQUIRE(actual.scaleLengthM_ == expected.scaleLengthM_);
    REQUIRE(actual.inverseScaleLengthPerM_ == expected.inverseScaleLengthPerM_);
    REQUIRE(actual.referenceHeightM_ == expected.referenceHeightM_);
    REQUIRE(actual.refractivityValue_ == expected.refractivityValue_);
    REQUIRE(actual.inverseReferenceDensityCm3PerG_ ==
            expected.inverseReferenceDensityCm3PerG_);
    REQUIRE(actual.mediumId_ == expected.mediumId_);
    REQUIRE(actual.compositionBegin_ == expected.compositionBegin_);
    REQUIRE(actual.compositionCount_ == expected.compositionCount_);
    REQUIRE(actual.densityDeficitGPerCm3_ == expected.densityDeficitGPerCm3_);
    REQUIRE(actual.refractiveIndexDeficit_ == expected.refractiveIndexDeficit_);
    for (int component = 0; component < 3; ++component) {
      CAPTURE(component);
      REQUIRE(actual.anchorM_[component] == expected.anchorM_[component]);
      REQUIRE(actual.axis_[component] == expected.axis_[component]);
      REQUIRE(actual.magneticFieldT_[component] == expected.magneticFieldT_[component]);
    }
  }
} // namespace

TEST_CASE("Model and composition records retain all fields on the execution device",
          "[model_transfer]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    int constexpr count = 257;
    int constexpr componentCount = 2 * count;
    std::array<density_type, 5> constexpr densityKinds{
        density_type::Invalid, density_type::Homogeneous,
        density_type::SlidingPlanarExponential, density_type::FlatExponential,
        density_type::ExponentialFirn};
    std::array<refractivity_type, 4> constexpr refractivityKinds{
        refractivity_type::Unavailable, refractivity_type::Uniform,
        refractivity_type::GladstoneDale, refractivity_type::ExponentialFirn};
    std::array<double, 5> constexpr densityChecks{-1., 11., 22., 33., 66.};
    std::array<double, 4> constexpr refractivityChecks{-2., 44., 55., 77.};
    Kokkos::View<model_type*, Kokkos::HostSpace> hostModels("host_models", count);
    Kokkos::View<component_type*, Kokkos::HostSpace> hostComponents("host_components",
                                                                    componentCount);
    // Distinct binary fractions also populate inactive fields. These records
    // test memory transfer and field access, not density or refraction laws.
    for (int index = 0; index < count; ++index) {
      auto& model = hostModels(index);
      model.densityKind_ = densityKinds[index % densityKinds.size()];
      model.refractivityKind_ = refractivityKinds[index % refractivityKinds.size()];
      model.referenceDensityGPerCm3_ = (index + 1) * 0.5;
      model.scaleLengthM_ = (index - 128) * 0.25;
      model.inverseScaleLengthPerM_ = (index - 64) * 0.125;
      model.referenceHeightM_ = (index + 3) * 0.0625;
      model.refractivityValue_ = (index + 5) * 0.03125;
      model.inverseReferenceDensityCm3PerG_ = (index + 7) * 0.015625;
      model.densityDeficitGPerCm3_ = (index + 9) * 0.0078125;
      model.refractiveIndexDeficit_ = (index + 11) * 0.00390625;
      model.mediumId_ = index % 9 - 1;
      model.compositionBegin_ = 2 * index;
      model.compositionCount_ = 2;
      for (int component = 0; component < 3; ++component) {
        model.anchorM_[component] = (index * 3 + component - 384) * 0.25;
        model.axis_[component] = (index * 3 + component - 192) * 0.125;
        model.magneticFieldT_[component] = (index * 3 + component + 1) * 0.0625;
      }
    }
    for (int index = 0; index < componentCount; ++index) {
      hostComponents(index) = {index - 257, (index + 1) * 0.0078125};
    }
    Kokkos::View<model_type*, memory_space_type> models("models", count);
    Kokkos::View<model_type*, memory_space_type> output("model_output", count);
    Kokkos::View<model_type*, memory_space_type> defaults("model_defaults", count);
    Kokkos::View<component_type*, memory_space_type> components("components",
                                                                componentCount);
    Kokkos::View<component_type*, memory_space_type> outputComponents("component_output",
                                                                      componentCount);
    Kokkos::View<component_type*, memory_space_type> defaultComponents(
        "component_defaults", componentCount);
    Kokkos::View<double * [9], Kokkos::LayoutRight, memory_space_type> checks(
        "model_checks", count);
    Kokkos::deep_copy(models, hostModels);
    Kokkos::deep_copy(components, hostComponents);
    Kokkos::deep_copy(checks, -999.);
    Kokkos::parallel_for(
        "model_fields", Kokkos::RangePolicy<execution_space_type>(0, count),
        KOKKOS_LAMBDA(int const index) {
          auto const source = models(index);
          output(count - 1 - index) = source;
          defaults(index) = model_type{};
          switch (source.densityKind_) {
            case density_type::Invalid:
              checks(index, 0) = -1.;
              break;
            case density_type::Homogeneous:
              checks(index, 0) = 11.;
              break;
            case density_type::SlidingPlanarExponential:
              checks(index, 0) = 22.;
              break;
            case density_type::FlatExponential:
              checks(index, 0) = 33.;
              break;
            case density_type::ExponentialFirn:
              checks(index, 0) = 66.;
              break;
          }
          switch (source.refractivityKind_) {
            case refractivity_type::Unavailable:
              checks(index, 1) = -2.;
              break;
            case refractivity_type::Uniform:
              checks(index, 1) = 44.;
              break;
            case refractivity_type::GladstoneDale:
              checks(index, 1) = 55.;
              break;
            case refractivity_type::ExponentialFirn:
              checks(index, 1) = 77.;
              break;
          }
          checks(index, 2) = source.referenceDensityGPerCm3_ + 2. * source.scaleLengthM_ +
                             3. * source.inverseScaleLengthPerM_ +
                             4. * source.referenceHeightM_ +
                             5. * source.refractivityValue_ +
                             6. * source.inverseReferenceDensityCm3PerG_;
          double anchor = 0.;
          double axis = 0.;
          double field = 0.;
          for (int component = 0; component < 3; ++component) {
            anchor += (component + 1) * source.anchorM_[component];
            axis += (component + 1) * source.axis_[component];
            field += (component + 1) * source.magneticFieldT_[component];
          }
          checks(index, 3) = anchor;
          checks(index, 4) = axis;
          checks(index, 5) = field;
          double code = 0.;
          double fraction = 0.;
          for (std::uint32_t offset = 0; offset < source.compositionCount_; ++offset) {
            auto const componentIndex = source.compositionBegin_ + offset;
            auto const component = components(componentIndex);
            outputComponents(componentCount - 1 - componentIndex) = component;
            defaultComponents(componentIndex) = component_type{};
            code += static_cast<double>(offset + 1) * component.code_;
            fraction += (offset + 1) * component.fraction_;
          }
          checks(index, 6) = code;
          checks(index, 7) = fraction;
          checks(index, 8) = source.mediumId_ + 2. * source.compositionBegin_ +
                             3. * source.compositionCount_;
        });
    Kokkos::fence("model transfer complete");
    Kokkos::View<model_type*, Kokkos::HostSpace> hostOutput("host_model_output", count);
    Kokkos::View<model_type*, Kokkos::HostSpace> hostDefaults("host_model_defaults",
                                                              count);
    Kokkos::View<component_type*, Kokkos::HostSpace> hostComponentOutput(
        "host_component_output", componentCount);
    Kokkos::View<component_type*, Kokkos::HostSpace> hostComponentDefaults(
        "host_component_defaults", componentCount);
    Kokkos::View<double * [9], Kokkos::LayoutRight, Kokkos::HostSpace> hostChecks(
        "host_model_checks", count);
    Kokkos::deep_copy(hostOutput, output);
    Kokkos::deep_copy(hostDefaults, defaults);
    Kokkos::deep_copy(hostComponentOutput, outputComponents);
    Kokkos::deep_copy(hostComponentDefaults, defaultComponents);
    Kokkos::deep_copy(hostChecks, checks);
    for (int index = 0; index < count; ++index) {
      CAPTURE(index);
      auto const& expected = hostModels(index);
      check_model(hostOutput(count - 1 - index), expected);
      check_model(hostDefaults(index), model_type{});
      REQUIRE(hostChecks(index, 0) == densityChecks[index % densityChecks.size()]);
      REQUIRE(hostChecks(index, 1) ==
              refractivityChecks[index % refractivityChecks.size()]);
      REQUIRE(hostChecks(index, 2) ==
              expected.referenceDensityGPerCm3_ + 2. * expected.scaleLengthM_ +
                  3. * expected.inverseScaleLengthPerM_ +
                  4. * expected.referenceHeightM_ + 5. * expected.refractivityValue_ +
                  6. * expected.inverseReferenceDensityCm3PerG_);
      double anchor = 0.;
      double axis = 0.;
      double field = 0.;
      for (int component = 0; component < 3; ++component) {
        anchor += (component + 1) * expected.anchorM_[component];
        axis += (component + 1) * expected.axis_[component];
        field += (component + 1) * expected.magneticFieldT_[component];
      }
      REQUIRE(hostChecks(index, 3) == anchor);
      REQUIRE(hostChecks(index, 4) == axis);
      REQUIRE(hostChecks(index, 5) == field);
      double code = 0.;
      double fraction = 0.;
      for (std::uint32_t offset = 0; offset < expected.compositionCount_; ++offset) {
        auto const componentIndex = expected.compositionBegin_ + offset;
        auto const component = hostComponents(componentIndex);
        code += static_cast<double>(offset + 1) * component.code_;
        fraction += (offset + 1) * component.fraction_;
      }
      REQUIRE(hostChecks(index, 6) == code);
      REQUIRE(hostChecks(index, 7) == fraction);
      REQUIRE(hostChecks(index, 8) == expected.mediumId_ +
                                          2. * expected.compositionBegin_ +
                                          3. * expected.compositionCount_);
    }
    for (int index = 0; index < componentCount; ++index) {
      CAPTURE(index);
      auto const& actual = hostComponentOutput(componentCount - 1 - index);
      REQUIRE(actual.code_ == hostComponents(index).code_);
      REQUIRE(actual.fraction_ == hostComponents(index).fraction_);
      REQUIRE(hostComponentDefaults(index).code_ == 0);
      REQUIRE(hostComponentDefaults(index).fraction_ == 0.);
    }
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}
