/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/ModelGrammage.hpp>
#include <corsika/kokkos/environment/ModelConversion.hpp>
#include <corsika/framework/core/PhysicalConstants.hpp>
#include <corsika/framework/geometry/Sphere.hpp>
#include <corsika/framework/geometry/StraightTrajectory.hpp>
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

#include "ModelGrammageReference.hpp"

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
    if (!std::isfinite(expected)) {
      require_exact(actual, expected);
      return;
    }
    REQUIRE(std::isfinite(actual));
    REQUIRE(std::abs(actual - expected) <=
            tolerance * std::max(std::abs(actual), std::abs(expected)));
  }

  void require_native(model_interface_type const& native, StraightTrajectory const& track,
                      CoordinateSystemPtr const& frame) {
    std::vector<kokkos::EnvironmentComponent> composition;
    auto const model = kokkos::export_model(native, frame, composition);
    auto const position = track.getPosition(0);
    auto const direction = track.getDirection(0);
    auto const coordinates = position.getCoordinates(frame);
    auto const components = direction.getComponents(frame);
    double const point[3]{coordinates[0] / 1_m, coordinates[1] / 1_m,
                          coordinates[2] / 1_m};
    double const axis[3]{components[0].magnitude(), components[1].magnitude(),
                         components[2].magnitude()};
    auto const nativeGrammage = native.getIntegratedGrammage(track);
    double const expected = nativeGrammage / (1_g / (1_cm * 1_cm));
    double const actual = kokkos::grammage(model, point, axis, track.getLength() / 1_m);
    CAPTURE(actual, expected);
    // Native unit/vector arithmetic follows a different rounding path, notably
    // when normalizing the radial axis. Exact migration is tested separately.
    require_relative(actual, expected, 5e-11);
    for (auto const grammage : {nativeGrammage, 1_g / (1_cm * 1_cm)}) {
      double const target = grammage / (1_g / (1_cm * 1_cm));
      double const nativeLength = native.getArclengthFromGrammage(track, grammage) / 1_m;
      double const length = kokkos::arclength(model, point, axis, target);
      CAPTURE(target, length, nativeLength);
      require_relative(length, nativeLength, 5e-11);
    }
    REQUIRE(native.getIntegratedGrammage(track) == nativeGrammage);
  }

} // namespace

TEST_CASE("Grammage and arclength preserve frozen production expressions",
          "[kokkos_model_grammage]") {
  auto const queries = model_grammage_reference::queries();
  for (std::size_t index = 0; index < queries.size(); ++index) {
    CAPTURE(index);
    auto const& query = queries[index];
    require_exact(kokkos::grammage(query.model_, query.positionM_, query.direction_,
                                   query.lengthM_),
                  model_grammage_reference::grammage(query.model_, query.positionM_,
                                                     query.direction_, query.lengthM_));
    require_exact(
        kokkos::arclength(query.model_, query.positionM_, query.direction_,
                          query.grammageGPerCm2_),
        model_grammage_reference::arclength(query.model_, query.positionM_,
                                            query.direction_, query.grammageGPerCm2_));
  }
}

TEST_CASE("Grammage retains exact branch boundaries and original IEEE behavior",
          "[kokkos_model_grammage]") {
  double const nan = std::numeric_limits<double>::quiet_NaN();
  double const inf = std::numeric_limits<double>::infinity();
  double const invalid[3]{nan, nan, nan};
  auto query =
      model_grammage_reference::make_query(model_type::Density::FlatExponential, -1., 1.);
  query.model_.referenceDensityGPerCm3_ = 1.;
  // A valid kind with unusable numeric fields still returns +0 before reading
  // position/direction/density. This does not assign physics to Invalid kinds.
  auto broken = query.model_;
  broken.referenceDensityGPerCm3_ = nan;
  require_exact(kokkos::grammage(broken, invalid, invalid, -0.), 0.);
  double const tangent[3]{1., 0., 0.};
  require_exact(kokkos::grammage(query.model_, query.positionM_, tangent, 2.), 200.);
  require_exact(kokkos::arclength(query.model_, query.positionM_, tangent, 200.), 2.);
  double const below = std::nextafter(100., 0.);
  double const above = std::nextafter(100., inf);
  REQUIRE(below * .01 < 1.);
  REQUIRE(above * .01 > 1.);
  REQUIRE(std::isfinite(
      kokkos::arclength(query.model_, query.positionM_, query.direction_, below)));
  require_exact(kokkos::arclength(query.model_, query.positionM_, query.direction_, 100.),
                inf);
  require_exact(
      kokkos::arclength(query.model_, query.positionM_, query.direction_, above), inf);
  require_exact(kokkos::arclength(query.model_, query.positionM_, query.direction_, nan),
                inf);

  // The threshold is projection == 0, not abs(projection) < 1e-14.
  query.model_.scaleLengthM_ = 1.;
  query.model_.inverseScaleLengthPerM_ = 1.;
  double const tinyProjection[3]{1., 0., 1e-16};
  double const length = 1e16;
  double const actual =
      kokkos::grammage(query.model_, query.positionM_, tinyProjection, length);
  require_relative(actual, 1e18 * ::expm1(1.), 4e-15);
  REQUIRE(actual != length * 100.);
  double const nonunit[3]{0., 0., 2.};
  require_relative(kokkos::grammage(query.model_, query.positionM_, nonunit, 1.),
                   50. * ::expm1(2.), 4e-15);

  query.model_.densityKind_ = model_type::Density::SlidingPlanarExponential;
  query.model_.referenceHeightM_ = 0.;
  double const center[3]{0., 0., 0.};
  REQUIRE(std::isnan(kokkos::grammage(query.model_, center, query.direction_, 1.)));
  require_exact(kokkos::arclength(query.model_, center, query.direction_, 0.), inf);
  // At radius 1, a tangent retains constant starting density even as the true
  // radius would change along the line: the local plane is frozen at the start.
  double const start[3]{0., 0., 1.};
  query.model_.referenceHeightM_ = 1.;
  require_exact(kokkos::grammage(query.model_, start, tangent, 10.), 1000.);
}

TEST_CASE("Small arguments retain expm1 log1p accuracy and inverse consistency",
          "[kokkos_model_grammage]") {
  for (double const scale : {-8000., 8000.}) {
    for (double const projection : {-1., -1e-12, 0., 1e-12, 1.}) {
      auto query = model_grammage_reference::make_query(
          model_type::Density::FlatExponential, scale, projection);
      for (double const length : {1e-12, 1e-6, 1., 100.}) {
        long double const rho = query.model_.referenceDensityGPerCm3_;
        long double const z = static_cast<long double>(projection) * length / scale;
        long double const expected =
            rho * length * 100.L * (z == 0.L ? 1.L : std::expm1(z) / z);
        double const value =
            kokkos::grammage(query.model_, query.positionM_, query.direction_, length);
        CAPTURE(scale, projection, length, value);
        require_relative(value, static_cast<double>(expected), 8e-15);
        double const recovered =
            kokkos::arclength(query.model_, query.positionM_, query.direction_, value);
        require_relative(recovered, length, 8e-15);
        long double const target = rho * length * 100.L;
        long double const w = target * 0.01L * projection / (scale * rho);
        long double const inverse =
            target / (rho * 100.L) * (w == 0.L ? 1.L : std::log1p(w) / w);
        require_relative(kokkos::arclength(query.model_, query.positionM_,
                                           query.direction_, static_cast<double>(target)),
                         static_cast<double>(inverse), 8e-15);
      }
    }
  }
}

TEST_CASE("Exported single model path laws agree with native straight trajectories",
          "[kokkos_model_grammage]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {1_km, -2_km, 3_km});
  auto const frame =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  auto const equalFrame =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  Point const anchor(frame, 3_m, -4_m, 5_m);
  Point const start(frame, 3_m, -4_m, 30_m);
  DirectionVector const axis(frame, {0., 0., 2.5});
  MagneticFieldVector const field(frame, {11_uT, -22_uT, 33_uT});
  NuclearComposition const mixture({Code::Nitrogen, Code::Oxygen}, {0.8, 0.2});
  for (auto const scale : {-800_m, 800_m}) {
    uniform_extra_type<HomogeneousMedium<model_interface_type>> homogeneous(
        1.000312, Medium::AirDry1Atm, field, 1.25_kg / cube(1_m), mixture);
    uniform_extra_type<FlatExponential<model_interface_type>> flat(
        1.000312, Medium::AirDry1Atm, field, anchor, axis, 1.25_kg / cube(1_m), scale,
        mixture);
    uniform_extra_type<SlidingPlanarExponential<model_interface_type>> sliding(
        1.000312, Medium::AirDry1Atm, field, anchor, 1.25_kg / cube(1_m), scale, mixture,
        7_m);
    std::array<model_interface_type const*, 3> const models{&homogeneous, &flat,
                                                            &sliding};
    for (auto const& velocity :
         {VelocityVector(frame, {1_m / second, 0_m / second, 0_m / second}),
          VelocityVector(frame, {0_m / second, 0_m / second, 1_m / second}),
          VelocityVector(frame, {0_m / second, 0_m / second, -1_m / second}),
          VelocityVector(frame, {1_m / second, 0_m / second, 1e-12 * 1_m / second}),
          VelocityVector(frame, {1_m / second, 2_m / second, 3_m / second})}) {
      for (auto const duration : {0_s, 1_s, 20_s}) {
        StraightTrajectory const track(Line(start, velocity), duration);
        for (auto const* model : models) {
          for (auto const& common : {root, translated, frame, equalFrame}) {
            require_native(*model, track, common);
          }
        }
      }
    }
  }
}

TEST_CASE("Five layer atmosphere retains native frozen axis grammage and inverse",
          "[kokkos_model_grammage]") {
  auto const root = get_root_CoordinateSystem();
  auto const frame = make_rotation(make_translation(root, {1_km, -2_km, 3_km}),
                                   QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
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
    Point const start(frame, 0_m, 0_m, (inner + outer) * 0.5);
    for (auto const& velocity :
         {VelocityVector(frame, {100_m / second, 0_m / second, 0_m / second}),
          VelocityVector(frame, {0_m / second, 0_m / second, 100_m / second}),
          VelocityVector(frame, {0_m / second, 0_m / second, -100_m / second})}) {
      StraightTrajectory const track(Line(start, velocity), 1_s);
      for (auto const& common : {root, frame}) {
        require_native(node.getModelProperties(), track, common);
      }
    }
  });
  REQUIRE(layers == 5);
}
