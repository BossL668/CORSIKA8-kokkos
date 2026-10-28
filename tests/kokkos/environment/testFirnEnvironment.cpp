/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "FirnProbe.hpp"
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/geometry/SeparationPlane.hpp>
#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotData.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/ExponentialFirn.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/modules/tracking/TrackingStraight.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
  using namespace corsika;
  namespace cp = corsika::kokkos;
  using interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  using medium_type =
      UniformMagneticField<MediumPropertyModel<ExponentialFirn<interface_type>>>;
  using environment_type = Environment<interface_type>;

  struct Particle {
    Point position_;
    MomentumVector momentum_;
    Point const& getPosition() const { return position_; }
    MomentumVector const& getMomentum() const { return momentum_; }
    HEPEnergyType getEnergy() const { return 1_GeV; }
  };

  NuclearComposition ice() {
    return NuclearComposition({Code::Hydrogen, Code::Oxygen}, {2. / 3., 1. / 3.});
  }

  void add_firn(environment_type& environment, CoordinateSystemPtr const& frame,
                ExponentialFirnProfile const& profile) {
    auto child = environment_type::createNode<SeparationPlane>(
        Plane(Point(frame, 0_m, 0_m, profile.surfaceHeight_),
              DirectionVector(frame, {0., 0., 1.})));
    child->setModelProperties<medium_type>(MagneticFieldVector(frame, {0_T, 0_T, 0_T}),
                                           Medium::WaterIce, frame, ice(), profile);
    environment.getUniverse()->addChild(std::move(child));
  }

  StraightTrajectory line(Point const& start, DirectionVector const& direction,
                          double length) {
    return StraightTrajectory(Line(start, direction * (1_m / 1_s)), length * 1_s);
  }

  double relative(double first, double second) {
    return first == second ? 0. : std::abs(first - second) / std::abs(second);
  }
} // namespace

TEST_CASE("Firn snapshots reproduce native queries in translated and rotated frames",
          "[firn][environment]") {
#if defined(FIRN_EXPECT_CUDA)
  REQUIRE(std::string(firn_test::execution_name()) == "Cuda");
#elif defined(FIRN_EXPECT_OPENMP)
  REQUIRE(std::string(firn_test::execution_name()) == "OpenMP");
#else
  REQUIRE_FALSE(std::string(firn_test::execution_name()).empty());
#endif
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {130_m, -270_m, 1000_m});
  auto const rotated =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, .7);
  std::vector<CoordinateSystemPtr> const frames{root, translated, rotated};
  ExponentialFirnProfile standard;
  auto uniform = standard;
  uniform.refractiveIndexDeficit_ = 0.;
  auto custom = standard;
  custom.scaleDepth_ = 31_m;
  custom.surfaceHeight_ = -40_m;
  custom.refractiveIndexDeficit_ = .2;
  std::vector<ExponentialFirnProfile> const profiles{standard, uniform, custom};
  std::size_t count = 0;
  double maximumDensity = 0., maximumIndex = 0., maximumDepth = 0., maximumLength = 0.;
  for (auto const& frame : frames) {
    for (auto const& profile : profiles) {
      environment_type environment;
      add_firn(environment, frame, profile);
      cp::EnvironmentSnapshot<interface_type> const snapshot(*environment.getUniverse(),
                                                             root);
      REQUIRE(snapshot.getModels().size() == 1);
      auto const& exported = snapshot.getModels()[0];
      REQUIRE(exported.densityKind_ == cp::EnvironmentModel::Density::ExponentialFirn);
      REQUIRE(exported.refractivityKind_ ==
              cp::EnvironmentModel::Refractivity::ExponentialFirn);
      REQUIRE(exported.mediumId_ == static_cast<int>(Medium::WaterIce));
      REQUIRE(snapshot.getComposition().size() == 2);
      REQUIRE(exported.scaleLengthM_ == profile.scaleDepth_ / 1_m);
      REQUIRE(exported.referenceHeightM_ == profile.surfaceHeight_ / 1_m);
      REQUIRE(exported.densityDeficitGPerCm3_ ==
              profile.getDensityDeficit() / (1_g / cube(1_cm)));
      REQUIRE(exported.refractiveIndexDeficit_ == profile.refractiveIndexDeficit_);
      std::vector<firn_test::Query> queries;
      std::vector<std::array<double, 6>> expected;
      for (double depth : {1.e-6, .01, 1., 25., 200., 1000., 5000.}) {
        for (double directionZ : {-1., -.8, -1.e-15, 0., 1.e-15, .6, 1.}) {
          for (double length : {0., 1.e-8, .125, 5., 50.}) {
            if (directionZ > 0. && length * directionZ > .9 * depth) { continue; }
            Point const start(frame, .25_m, -.5_m, profile.surfaceHeight_ - depth * 1_m);
            DirectionVector const direction(
                frame, {std::sqrt(1. - directionZ * directionZ), 0., directionZ});
            auto const* node = environment.getUniverse()->getContainingNode(start);
            REQUIRE(node != environment.getUniverse().get());
            auto const& model = node->getModelProperties();
            // The selected C8 baseline returns (entry, infinity) for a half-space,
            // including upward travel. Preserve this contract; the surface is
            // separately enforced by the firn grammage domain.
            Particle const particle{start, direction * 1_GeV};
            auto const nativeBoundary =
                tracking_line::Tracking::intersect(particle, *node);
            REQUIRE(std::isinf(nativeBoundary.getExit() / 1_s));
            auto const trajectory = line(start, direction, length);
            double const x =
                model.getIntegratedGrammage(trajectory) / (1_g / square(1_cm));
            firn_test::Query query;
            auto position = start.getCoordinates(root);
            auto components = direction.getComponents(root);
            for (int axis = 0; axis < 3; ++axis) {
              query.positionM_[axis] = position[axis] / 1_m;
              query.direction_[axis] = components[axis].magnitude();
            }
            query.lengthM_ = length;
            query.grammageGPerCm2_ = x;
            queries.push_back(query);
            expected.push_back(
                {model.getMassDensity(start) / (1_g / cube(1_cm)),
                 model.getRefractiveIndex(start), x,
                 model.getArclengthFromGrammage(trajectory, x * (1_g / square(1_cm))) /
                     1_m,
                 directionZ, depth});
          }
        }
      }
      auto const results = firn_test::probe(cp::bind_environment_data(snapshot), queries);
      REQUIRE(results.size() == queries.size());
      for (std::size_t index = 0; index < results.size(); ++index) {
        CAPTURE(index, profile.scaleDepth_ / 1_m, queries[index].lengthM_);
        auto const& result = results[index];
        auto const& reference = expected[index];
        maximumDensity =
            std::max(maximumDensity, relative(result.values_[0], reference[0]));
        maximumIndex = std::max(maximumIndex, relative(result.values_[1], reference[1]));
        maximumDepth = std::max(maximumDepth, relative(result.values_[2], reference[2]));
        maximumLength =
            std::max(maximumLength, relative(result.values_[3], reference[3]));
        for (int column = 0; column < 4; ++column) {
          REQUIRE(relative(result.values_[column], reference[column]) < 2.e-11);
        }
        REQUIRE(result.values_[4] == 1.);
        REQUIRE(result.values_[6] == static_cast<int>(Medium::WaterIce));
        for (int status = 0; status < 3; ++status) {
          REQUIRE(result.statuses_[status] ==
                  static_cast<int>(firn_test::Status::Success));
        }
        REQUIRE(result.statuses_[3] ==
                static_cast<int>(firn_test::Status::NoForwardIntersection));
        REQUIRE(result.values_[5] == 0.); // NoForwardIntersection has no usable distance.
        ++count;
      }
    }
  }
  CORSIKA_LOG_INFO(
      "firn queries={} max relative density={} index={} grammage={} inverse={}", count,
      maximumDensity, maximumIndex, maximumDepth, maximumLength);
}

TEST_CASE("Firn keeps the original surface and infinite-candidate contracts",
          "[firn][sentinel]") {
  auto const frame = get_root_CoordinateSystem();
  ExponentialFirnProfile profile;
  medium_type medium(MagneticFieldVector(frame, {0_T, 0_T, 0_T}), Medium::WaterIce, frame,
                     ice(), profile);
  auto const infinity = std::numeric_limits<double>::infinity();
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  Point const start(frame, 0_m, 0_m, -200_m);
  DirectionVector const down(frame, {.8, 0., -.6});
  auto const infinite = line(start, down, infinity);
  auto const finite = line(start, down, 37.25);
  auto const target = medium.getIntegratedGrammage(finite);
  REQUIRE(std::isinf(medium.getIntegratedGrammage(infinite) / (1_g / square(1_cm))));
  REQUIRE(std::abs(medium.getArclengthFromGrammage(infinite, target) / 1_m - 37.25) <
          1.e-11);
  REQUIRE_THROWS(medium.getMassDensity(Point(frame, 0_m, 0_m, 1_m)));
  REQUIRE_THROWS(medium.getRefractiveIndex(Point(frame, 0_m, 0_m, 1_m)));
  REQUIRE(medium.getArclengthFromGrammage(finite, -1_g / square(1_cm)) < 0_m);
  REQUIRE_THROWS(medium.getArclengthFromGrammage(finite, nan * 1_g / square(1_cm)));
  REQUIRE_THROWS(medium.getIntegratedGrammage(
      line(start, DirectionVector(frame, {0., 0., 1.}), 201.)));
  environment_type environment;
  add_firn(environment, frame, profile);
  cp::EnvironmentSnapshot<interface_type> const snapshot(*environment.getUniverse(),
                                                         frame);
  std::vector<firn_test::Query> queries(9);
  for (auto& query : queries) {
    query.positionM_[2] = -200.;
    query.lengthM_ = 37.25;
    query.grammageGPerCm2_ = 100.;
  }
  queries[0].lengthM_ = infinity;
  queries[0].direction_[0] = .8;
  queries[0].direction_[2] = -.6;
  queries[0].grammageGPerCm2_ = target / (1_g / square(1_cm));
  queries[1].lengthM_ = -1.;
  queries[2].lengthM_ = nan;
  queries[3].positionM_[2] = 1.;
  queries[4].direction_[2] = 1.;
  queries[4].lengthM_ = 201.;
  queries[4].grammageGPerCm2_ = 1.e6;
  queries[5].grammageGPerCm2_ = -1.;
  queries[6].grammageGPerCm2_ = nan;
  queries[7].positionM_[2] = nan;
  queries[8].direction_[2] = nan;
  auto const results = firn_test::probe(cp::bind_environment_data(snapshot), queries);
  REQUIRE(std::isinf(results[0].values_[2]));
  REQUIRE(std::abs(results[0].values_[3] - 37.25) < 1.e-11);
  REQUIRE(std::isnan(results[1].values_[2]));
  REQUIRE(std::isnan(results[2].values_[2]));
  REQUIRE(std::isnan(results[3].values_[0]));
  REQUIRE(std::isnan(results[3].values_[1]));
  REQUIRE(results[3].statuses_[0] ==
          static_cast<int>(firn_test::Status::InvalidSnapshot));
  REQUIRE(std::isnan(results[4].values_[2]));
  REQUIRE(std::isinf(results[4].values_[3]));
  REQUIRE(results[4].statuses_[2] ==
          static_cast<int>(firn_test::Status::GrammageOutOfRange));
  REQUIRE(std::isnan(results[5].values_[3]));
  REQUIRE(std::isnan(results[6].values_[3]));
  REQUIRE(results[7].values_[0] == profile.deepDensity_ / (1_g / cube(1_cm)));
  REQUIRE(results[7].values_[1] == profile.deepRefractiveIndex_);
  REQUIRE(std::isinf(results[7].values_[2]));
  REQUIRE(std::isinf(results[8].values_[2]));
  auto invalidProfile = profile;
  invalidProfile.scaleDepth_ = infinity * 1_m;
  medium_type invalid(MagneticFieldVector(frame, {0_T, 0_T, 0_T}), Medium::WaterIce,
                      frame, ice(), invalidProfile);
  std::vector<cp::EnvironmentComponent> composition;
  REQUIRE_THROWS_AS(cp::export_model(invalid, frame, composition), std::invalid_argument);
  REQUIRE(composition.empty());
}

TEST_CASE("Native firn inverse preserves the signed PROPOSAL range contract",
          "[firn][sentinel]") {
  auto const frame = get_root_CoordinateSystem();
  Point const start(frame, 0_m, 0_m, -200_m);
  ExponentialFirnProfile profile;
  medium_type medium(MagneticFieldVector(frame, {0_T, 0_T, 0_T}), Medium::WaterIce, frame,
                     ice(), profile);
  for (double directionZ : {-1., -.6, 0., .6, 1.}) {
    DirectionVector const direction(
        frame, {std::sqrt(1. - directionZ * directionZ), 0., directionZ});
    auto const trajectory = line(start, direction, 1.);
    for (double length : {1.e-8, .01, 1., 10.}) {
      auto const backward = line(start, -direction, length);
      auto const magnitude = medium.getIntegratedGrammage(backward);
      auto const result = medium.getArclengthFromGrammage(trajectory, -magnitude);
      CAPTURE(directionZ, length);
      REQUIRE(relative(result / 1_m, -length) < 1.e-12);
    }
  }
}

TEST_CASE("A bounded firn volume preserves native C8 exit distances",
          "[firn][boundary]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {130_m, -270_m, 1000_m});
  auto const rotated =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, .7);
  std::size_t count = 0;
  double maximum = 0.;
  for (auto const& frame : std::vector<CoordinateSystemPtr>{root, translated, rotated}) {
    environment_type environment;
    auto const center = make_translation(frame, {0_m, 0_m, -5000_m});
    auto child = environment_type::createNode<Box>(center, 100000_m, 100000_m, 5000_m);
    child->setModelProperties<medium_type>(MagneticFieldVector(frame, {0_T, 0_T, 0_T}),
                                           Medium::WaterIce, frame, ice());
    environment.getUniverse()->addChild(std::move(child));
    cp::EnvironmentSnapshot<interface_type> const snapshot(*environment.getUniverse(),
                                                           root);
    std::vector<firn_test::Query> queries;
    std::vector<std::array<double, 3>> expected;
    for (double depth : {.01, 1., 25., 200., 1000., 5000.}) {
      for (double directionZ : {-1., -.6, .6, 1.}) {
        Point const start(frame, .25_m, -.5_m, -depth * 1_m);
        DirectionVector const direction(
            frame, {std::sqrt(1. - directionZ * directionZ), 0., directionZ});
        auto const* node = environment.getUniverse()->getContainingNode(start);
        REQUIRE(node != environment.getUniverse().get());
        Particle const particle{start, direction * 1_GeV};
        auto const intersection = tracking_line::Tracking::intersect(particle, *node);
        double const distance = intersection.getExit() * constants::c / 1_m;
        REQUIRE(std::isfinite(distance));
        REQUIRE(distance > 0.);
        auto const& medium = node->getModelProperties();
        auto const trajectory = line(start, direction, .75 * distance);
        double const grammage =
            medium.getIntegratedGrammage(trajectory) / (1_g / square(1_cm));
        firn_test::Query query;
        auto const position = start.getCoordinates(root);
        auto const components = direction.getComponents(root);
        for (int axis = 0; axis < 3; ++axis) {
          query.positionM_[axis] = position[axis] / 1_m;
          query.direction_[axis] = components[axis].magnitude();
        }
        query.lengthM_ = .75 * distance;
        query.grammageGPerCm2_ = grammage;
        queries.push_back(query);
        expected.push_back({distance, grammage, .75 * distance});
      }
    }
    auto const results = firn_test::probe(cp::bind_environment_data(snapshot), queries);
    for (std::size_t index = 0; index < results.size(); ++index) {
      CAPTURE(index);
      auto const& result = results[index];
      REQUIRE(result.values_[4] == 1.);
      for (int status : result.statuses_) {
        REQUIRE(status == static_cast<int>(firn_test::Status::Success));
      }
      auto const error = relative(result.values_[5], expected[index][0]);
      maximum = std::max(maximum, error);
      REQUIRE(error < 2.e-10);
      REQUIRE(relative(result.values_[2], expected[index][1]) < 2.e-11);
      REQUIRE(relative(result.values_[3], expected[index][2]) < 2.e-11);
      ++count;
    }
  }
  CORSIKA_LOG_INFO("firn bounded exits={} max relative distance={}", count, maximum);
}
