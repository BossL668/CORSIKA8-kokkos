/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/egs4/AtmosphereExport.hpp>
#include <corsika/kokkos/environment/EnvironmentPreparation.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotData.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <catch2/catch_all.hpp>

namespace {
  using namespace corsika;
  using interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  template <typename TModel>
  using extra_type =
      GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<TModel>>>;
} // namespace

TEST_CASE("EGS4 atmosphere comes from the actual C8 nodes", "[egs4][environment]") {
  auto const frame = get_root_CoordinateSystem();
  auto const radius = constants::EarthRadius::Mean / 1_m;
  Environment<interface_type> environment;
  create_5layer_atmosphere<interface_type, extra_type>(
      environment, AtmosphereId::LinsleyUSStd, Point(frame, 0_m, 0_m, 0_m), 1.000312,
      Point(frame, 0_m, 0_m, constants::EarthRadius::Mean), Medium::AirDry1Atm,
      MagneticFieldVector(frame, 20_uT, 0_T, -40_uT));
  auto const snapshot = kokkos::prepare_environment_snapshot(environment, frame);
  auto view = kokkos::bind_environment_data(*snapshot);
  kokkos::TransportGeometryConfig geometry;
  geometry.observation_.pointM_[2] = radius + 1100.;
  geometry.observation_.normal_[2] = 1.;
  auto const air = kokkos::egs4::export_atmosphere(view, geometry, radius);
  REQUIRE(air.numberOfLayers_ == 5);
  REQUIRE(air.observationPlanePointM_[2] == radius + 1100.);
  for (unsigned layer = 0; layer < 5; ++layer) {
    auto const node = 5 - layer;
    auto const& model = view.models_[view.nodes_[node].modelIndex_];
    REQUIRE(air.atmosphereLayers_[layer].outerRadiusM_ == view.shapes_[node].radiusM_);
    REQUIRE(air.atmosphereLayers_[layer].densityParameterA_ ==
            model.referenceDensityGPerCm3_);
    REQUIRE(air.atmosphereLayers_[layer].densityParameterB_ == model.referenceHeightM_);
    REQUIRE(air.atmosphereLayers_[layer].densityParameterC_ == model.scaleLengthM_);
    REQUIRE(air.magneticFieldT_[0] == model.magneticFieldT_[0]);
  }
  SECTION("Additional exclusions are rejected, not ignored") {
    view.exclusionCount_ = 1;
    REQUIRE_THROWS_AS(kokkos::egs4::export_atmosphere(view, geometry, radius),
                      std::invalid_argument);
  }
  SECTION("Non-air and custom composition are rejected") {
    auto components = snapshot->getComposition();
    view.composition_ = components.data();
    components[0].fraction_ = .5;
    REQUIRE_THROWS_AS(kokkos::egs4::export_atmosphere(view, geometry, radius),
                      std::invalid_argument);
  }
  SECTION("Different magnetic fields are rejected") {
    auto models = snapshot->getModels();
    view.models_ = models.data();
    models[0].magneticFieldT_[0] += 1.e-5;
    REQUIRE_THROWS_AS(kokkos::egs4::export_atmosphere(view, geometry, radius),
                      std::invalid_argument);
  }
}
