/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/ModelEvaluation.hpp>
#include <corsika/kokkos/environment/ModelConversion.hpp>
#include <corsika/framework/core/PhysicalConstants.hpp>
#include <corsika/framework/geometry/Sphere.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/FlatExponential.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/SlidingPlanarExponential.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>

#include "ModelEvaluationReference.hpp"

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

  using namespace corsika;
  using model_type = kokkos::EnvironmentModel;
  using model_interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;

  template <typename TModel>
  using uniform_extra_type =
      UniformRefractiveIndex<MediumPropertyModel<UniformMagneticField<TModel>>>;

  template <typename TModel>
  using gladstone_extra_type =
      GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<TModel>>>;

  void require_exact(double const actual, double const expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }

  void require_relative(double const actual, double const expected,
                        double const tolerance) {
    REQUIRE(std::isfinite(actual));
    REQUIRE(std::isfinite(expected));
    REQUIRE(std::abs(actual - expected) <=
            tolerance * std::max(std::abs(actual), std::abs(expected)));
  }

  void require_native(model_interface_type const& native, Point const& point,
                      CoordinateSystemPtr const& frame) {
    std::vector<kokkos::EnvironmentComponent> components;
    auto const model = kokkos::export_model(native, frame, components);
    auto const coordinates = point.getCoordinates(frame);
    double const position[3]{coordinates[0] / 1_m, coordinates[1] / 1_m,
                             coordinates[2] / 1_m};
    auto const actualDensity = kokkos::density(model, position);
    auto const expectedDensity =
        native.getMassDensity(point) / (1_g / (1_cm * 1_cm * 1_cm));
    auto const actualIndex = kokkos::refractive_index(model, position);
    auto const expectedIndex = native.getRefractiveIndex(point);
    CAPTURE(actualDensity, expectedDensity, actualIndex, expectedIndex);
    // Native quantities and frame transformations take a different rounding
    // path. Earth-radius coordinates also amplify small position roundoff.
    // These tolerances do not relax the separate exact production-oracle test.
    require_relative(actualDensity, expectedDensity, 5e-12);
    require_relative(actualIndex, expectedIndex,
                     16 * std::numeric_limits<double>::epsilon());
    auto const field = native.getMagneticField(point).getComponents(frame);
    for (int component = 0; component < 3; ++component) {
      require_relative(model.magneticFieldT_[component], field[component] / 1_T, 5e-14);
    }
  }

} // namespace

TEST_CASE("Point model values preserve frozen production expressions on the host",
          "[kokkos_model_evaluation]") {
  auto const queries = model_evaluation_reference::queries();
  for (std::size_t index = 0; index < queries.size(); ++index) {
    CAPTURE(index);
    auto const& query = queries[index];
    require_exact(kokkos::density(query.model_, query.positionM_),
                  model_evaluation_reference::density(query.model_, query.positionM_));
    require_exact(
        kokkos::refractive_index(query.model_, query.positionM_),
        model_evaluation_reference::refractive_index(query.model_, query.positionM_));
  }
}

TEST_CASE("Point evaluation keeps constant laws and exceptional arithmetic unchanged",
          "[kokkos_model_evaluation]") {
  double const nan = std::numeric_limits<double>::quiet_NaN();
  double const invalidPosition[3]{nan, nan, nan};
  model_type model;
  // Uniform/unavailable refractivity has no density precondition. Never evaluate
  // density itself, or Gladstone-Dale, with the default Invalid density tag.
  model.refractivityKind_ = model_type::Refractivity::Uniform;
  model.refractivityValue_ = -2.;
  require_exact(kokkos::refractive_index(model, invalidPosition), -2.);
  model.refractivityKind_ = model_type::Refractivity::Unavailable;
  REQUIRE(std::isnan(kokkos::refractive_index(model, invalidPosition)));
  model.densityKind_ = model_type::Density::Homogeneous;
  model.referenceDensityGPerCm3_ = -0.;
  require_exact(kokkos::density(model, invalidPosition), -0.);
  model.referenceDensityGPerCm3_ = -1.;
  require_exact(kokkos::density(model, invalidPosition), -1.);

  model.densityKind_ = model_type::Density::FlatExponential;
  model.referenceDensityGPerCm3_ = 1.;
  model.scaleLengthM_ = -10.;
  model.inverseScaleLengthPerM_ = -0.1;
  model.axis_[2] = 2.5;
  double position[3]{0., 0., 4.};
  require_exact(kokkos::density(model, position), ::exp(-1.));
  model.densityKind_ = model_type::Density::SlidingPlanarExponential;
  position[2] = 0.;
  model.referenceHeightM_ = 10.;
  require_exact(kokkos::density(model, position), ::exp(1.));
  // No normalization at the radial center, and no recomputation of inverse scale.
  model.inverseScaleLengthPerM_ = 0.;
  require_exact(kokkos::density(model, position), 1.);
  model.densityKind_ = model_type::Density::FlatExponential;
  model.axis_[2] = 1.;
  model.referenceHeightM_ = 0.;
  model.inverseScaleLengthPerM_ = 1.;
  position[2] = 1000.;
  REQUIRE(std::isinf(kokkos::density(model, position)));
  model.referenceDensityGPerCm3_ = 0.;
  REQUIRE(std::isnan(kokkos::density(model, position)));
  model.referenceDensityGPerCm3_ = 1.;
  position[2] = -1000.;
  require_exact(kokkos::density(model, position), 0.);
}

TEST_CASE(
    "Exported point laws agree with native composed media across coordinate systems",
    "[kokkos_model_evaluation]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {1_km, -2_km, 3_km});
  auto const rotated =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  auto const equalRotation =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  REQUIRE(rotated != equalRotation);
  REQUIRE(*rotated == *equalRotation);
  Point const anchor(rotated, 3_m, -4_m, 5_m);
  Point const reference(rotated, 3_m, -4_m, 25_m);
  DirectionVector const axis(rotated, {0.25, -0.5, 2.5});
  MagneticFieldVector const field(rotated, {11_uT, -22_uT, 33_uT});
  NuclearComposition const mixture({Code::Nitrogen, Code::Oxygen}, {0.8, 0.2});
  auto const rho = 1.25_kg / cube(1_m);
  for (auto const scale : {-800_m, 800_m}) {
    uniform_extra_type<HomogeneousMedium<model_interface_type>> homogeneous(
        1.000312, Medium::AirDry1Atm, field, rho, mixture);
    uniform_extra_type<FlatExponential<model_interface_type>> flat(
        1.000312, Medium::AirDry1Atm, field, anchor, axis, rho, scale, mixture);
    uniform_extra_type<SlidingPlanarExponential<model_interface_type>> sliding(
        1.000312, Medium::AirDry1Atm, field, anchor, rho, scale, mixture, 7_m);
    gladstone_extra_type<HomogeneousMedium<model_interface_type>> homogeneousGD(
        1.000312, reference, Medium::AirDry1Atm, field, rho, mixture);
    gladstone_extra_type<FlatExponential<model_interface_type>> flatGD(
        1.000312, reference, Medium::AirDry1Atm, field, anchor, axis, rho, scale,
        mixture);
    gladstone_extra_type<SlidingPlanarExponential<model_interface_type>> slidingGD(
        1.000312, reference, Medium::AirDry1Atm, field, anchor, rho, scale, mixture, 7_m);
    std::array<model_interface_type const*, 6> const models{
        &homogeneous, &flat, &sliding, &homogeneousGD, &flatGD, &slidingGD};
    std::array<Point, 4> const points{anchor, reference,
                                      Point(rotated, 303_m, -404_m, 705_m),
                                      Point(rotated, -97_m, 46_m, -20_m)};
    for (auto const* model : models) {
      for (auto const& point : points) {
        auto const densityBefore = model->getMassDensity(point);
        auto const indexBefore = model->getRefractiveIndex(point);
        for (auto const& frame : {root, translated, rotated, equalRotation}) {
          require_native(*model, point, frame);
        }
        REQUIRE(model->getMassDensity(point) == densityBefore);
        REQUIRE(model->getRefractiveIndex(point) == indexBefore);
      }
    }
  }
}

TEST_CASE("Five layer atmosphere point values match native density and refractivity",
          "[kokkos_model_evaluation]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {1_km, -2_km, 3_km});
  auto const frame =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  Point const center(frame, 0_m, 0_m, 0_m);
  Point const reference(frame, 0_m, 0_m, constants::EarthRadius::Mean);
  Environment<model_interface_type> environment;
  create_5layer_atmosphere<model_interface_type, gladstone_extra_type>(
      environment, AtmosphereId::LinsleyUSStd, center, 1.000312, reference,
      Medium::AirDry1Atm, MagneticFieldVector(frame, {11_uT, -22_uT, 33_uT}));
  unsigned layers{};
  environment.getUniverse()->walk([&](auto const& node) {
    if (!node.hasModelProperties()) { return; }
    ++layers;
    auto const outer = dynamic_cast<Sphere const&>(node.getVolume()).getRadius();
    auto const inner =
        node.getChildNodes().empty()
            ? constants::EarthRadius::Mean
            : dynamic_cast<Sphere const&>(node.getChildNodes()[0]->getVolume())
                  .getRadius();
    for (double const fraction : {0.1, 0.5, 0.9}) {
      Point const point(frame, 0_m, 0_m, inner + fraction * (outer - inner));
      for (auto const& commonFrame : {root, translated, frame}) {
        require_native(node.getModelProperties(), point, commonFrame);
      }
    }
  });
  REQUIRE(layers == 5);
}
