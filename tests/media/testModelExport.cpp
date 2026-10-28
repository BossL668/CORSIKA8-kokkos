/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/media/IDensityModelExport.hpp>
#include <corsika/media/IRefractivityModelExport.hpp>
#include <corsika/media/IUniformMagneticFieldExport.hpp>
#include <corsika/framework/geometry/CoordinateSystem.hpp>
#include <corsika/framework/geometry/Line.hpp>
#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/ExponentialRefractiveIndex.hpp>
#include <corsika/media/FlatExponential.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/IMediumModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/media/SlidingPlanarExponential.hpp>
#include <corsika/media/SlidingPlanarTabular.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

  using namespace corsika;
  using density_kind = DensityModelParameters::Kind;
  using refractivity_kind = RefractivityModelParameters::Kind;
  using model_interface = IRefractiveIndexModel<IMagneticFieldModel<IMediumModel>>;

  template <typename T>
  using atmosphere_extras =
      GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<T>>>;

  NuclearComposition composition() {
    return NuclearComposition({Code::Nitrogen, Code::Oxygen}, {0.8, 0.2});
  }

  void require_same_point(Point const& actual, Point const& expected) {
    REQUIRE(actual.getCoordinateSystem() == expected.getCoordinateSystem());
    for (int component = 0; component < 3; ++component) {
      REQUIRE(actual.getCoordinates()[component] == expected.getCoordinates()[component]);
    }
  }

  template <typename TDimension>
  void require_same_vector(Vector<TDimension> const& actual,
                           Vector<TDimension> const& expected) {
    REQUIRE(actual.getCoordinateSystem() == expected.getCoordinateSystem());
    for (int component = 0; component < 3; ++component) {
      REQUIRE(actual.getComponents()[component] == expected.getComponents()[component]);
    }
  }

  std::vector<double> native_results(IMediumModel const& medium, Point const& start) {
    std::vector<double> result;
    auto const frame = start.getCoordinateSystem();
    auto const densityUnit = 1_g / cube(1_cm);
    auto const grammageUnit = 1_g / square(1_cm);
    result.push_back(medium.getMassDensity(start) / densityUnit);
    for (auto const duration : {0_s, 1_s, 20_s}) {
      for (auto const& velocity :
           {VelocityVector(frame, {1_m / second, 0_m / second, 0_m / second}),
            VelocityVector(frame, {0_m / second, 0_m / second, 1_m / second}),
            VelocityVector(frame, {0_m / second, 0_m / second, -1_m / second}),
            VelocityVector(frame, {1_m / second, 2_m / second, 3_m / second})}) {
        StraightTrajectory const track(Line(start, velocity), duration);
        result.push_back(medium.getIntegratedGrammage(track) / grammageUnit);
        result.push_back(medium.getArclengthFromGrammage(track, 3. * grammageUnit) /
                         meter);
      }
    }
    return result;
  }

  void require_native_unchanged(IMediumModel const& medium, Point const& start) {
    auto const before = native_results(medium, start);
    auto const* exporter = dynamic_cast<IDensityModelExport const*>(&medium);
    REQUIRE(exporter != nullptr);
    REQUIRE_NOTHROW(static_cast<void>(exporter->exportDensity()));
    auto const after = native_results(medium, start);
    REQUIRE(after == before);
  }

  class IndependentMedium : public IMediumModel {
  public:
    MassDensityType getMassDensity(Point const&) const override {
      return 1_g / cube(1_cm);
    }
    GrammageType getIntegratedGrammage(BaseTrajectory const& track) const override {
      return track.getLength() * (1_g / cube(1_cm));
    }
    LengthType getArclengthFromGrammage(BaseTrajectory const&,
                                        GrammageType const grammage) const override {
      return grammage / (1_g / cube(1_cm));
    }
    NuclearComposition const& getNuclearComposition() const override {
      return composition_;
    }

  private:
    NuclearComposition const composition_{composition()};
  };

  // These overrides add observation counters only. Their density/index laws remain
  // exactly the inherited laws, so the inherited export contracts remain valid.
  template <typename T>
  class CountingDensity : public HomogeneousMedium<T> {
  public:
    using HomogeneousMedium<T>::HomogeneousMedium;
    mutable unsigned calls_{};

    MassDensityType getMassDensity(Point const& point) const override {
      ++calls_;
      return HomogeneousMedium<T>::getMassDensity(point);
    }
  };

  template <typename T>
  class CountingRefractivity : public UniformRefractiveIndex<T> {
  public:
    using UniformRefractiveIndex<T>::UniformRefractiveIndex;
    mutable unsigned refractiveCalls_{};

    double getRefractiveIndex(Point const& point) const override {
      ++refractiveCalls_;
      return UniformRefractiveIndex<T>::getRefractiveIndex(point);
    }
  };

} // namespace

TEST_CASE("Homogeneous parameter export retains units and composition",
          "[model_export]") {
  auto const root = get_root_CoordinateSystem();
  auto const density = 1.234_g / cube(1_cm);
  auto const mixture = composition();
  HomogeneousMedium<IMediumModel> const model(density, mixture);
  IMediumModel const& native = model;
  auto const* exporter = dynamic_cast<IDensityModelExport const*>(&native);
  REQUIRE(exporter != nullptr);
  auto const parameters = exporter->exportDensity();
  REQUIRE(parameters.kind_ == density_kind::Homogeneous);
  REQUIRE(parameters.referenceDensity_ == density);
  REQUIRE(parameters.scaleLength_ == 0_m);
  REQUIRE(parameters.referenceHeight_ == 0_m);
  require_same_point(parameters.anchor_, Point(root, 0_m, 0_m, 0_m));
  require_same_vector(parameters.axis_, DirectionVector(root, {0., 0., 1.}));
  REQUIRE(native.getNuclearComposition().getComponents() == mixture.getComponents());
  REQUIRE(native.getNuclearComposition().getFractions() == mixture.getFractions());
  REQUIRE(native.getNuclearComposition().getHash() == mixture.getHash());
  require_native_unchanged(native, Point(root, 0_m, 0_m, 0_m));
}

TEST_CASE("Flat density export preserves signed scale and a nonunit axis",
          "[model_export]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {100_m, -200_m, 300_m});
  auto const rotated =
      make_rotation(root, QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  Point const anchor(translated, 10_m, 20_m, 30_m);
  DirectionVector const axis(rotated, {0., 0., 2.5});
  auto const density = 1.23_kg / cube(1_m);
  for (auto const scale : {-250_m, 250_m}) {
    FlatExponential<IMediumModel> const model(anchor, axis, density, scale,
                                              composition());
    auto const parameters = model.exportDensity();
    REQUIRE(parameters.kind_ == density_kind::FlatExponential);
    REQUIRE(parameters.referenceDensity_ == density);
    REQUIRE(parameters.scaleLength_ == scale);
    REQUIRE(parameters.referenceHeight_ == 0_m);
    REQUIRE(model.getReferenceDensity() == density);
    REQUIRE(model.getScaleLength() == scale);
    REQUIRE(model.getReferenceHeight() == 0_m);
    require_same_point(parameters.anchor_, anchor);
    require_same_vector(parameters.axis_, axis);
    REQUIRE(parameters.axis_.getNorm().magnitude() == 2.5);
    require_native_unchanged(model, Point(translated, 12_m, 23_m, 34_m));
    require_same_point(model.getAnchorPoint(), anchor);
  }
}

TEST_CASE("Sliding density export preserves its reference height and anchor",
          "[model_export]") {
  auto const root = get_root_CoordinateSystem();
  auto const frame = make_translation(root, {5_km, -8_km, 3_km});
  Point const anchor(frame, 1_m, 2_m, 3_m);
  auto const density = 1.23_kg / cube(1_m);
  for (auto const scale : {-400_m, 400_m}) {
    SlidingPlanarExponential<IMediumModel> const model(anchor, density, scale,
                                                       composition(), 13_m);
    auto const parameters = model.exportDensity();
    REQUIRE(parameters.kind_ == density_kind::SlidingPlanarExponential);
    REQUIRE(parameters.referenceDensity_ == density);
    REQUIRE(parameters.scaleLength_ == scale);
    REQUIRE(parameters.referenceHeight_ == 13_m);
    REQUIRE(model.getReferenceDensity() == density);
    REQUIRE(model.getScaleLength() == scale);
    REQUIRE(model.getReferenceHeight() == 13_m);
    require_same_point(parameters.anchor_, anchor);
    require_same_vector(parameters.axis_, DirectionVector(frame, {0., 0., 1.}));
    REQUIRE(model.getMassDensity(Point(frame, 1_m, 2_m, 16_m)) == density);
    require_native_unchanged(model, Point(frame, 1_m, 2_m, 16_m));
  }
}

TEST_CASE("Parameter export does not validate or repair raw native model state",
          "[model_export]") {
  auto const root = get_root_CoordinateSystem();
  Point const anchor(root, 1_m, 2_m, 3_m);
  DirectionVector const zeroAxis(root, {0., 0., 0.});
  auto const densityUnit = 1_kg / cube(1_m);
  // Native construction permits these inputs. Only export them here: evaluating
  // a zero-scale or nonpositive-density physics model is deliberately out of scope.
  for (double const densityValue : {0., -0., -1.25}) {
    auto const density = densityValue * densityUnit;
    HomogeneousMedium<IMediumModel> const homogeneous(density, composition());
    auto const homogeneousParameters = homogeneous.exportDensity();
    REQUIRE(homogeneousParameters.referenceDensity_ == density);
    REQUIRE(std::signbit(homogeneousParameters.referenceDensity_ / densityUnit) ==
            std::signbit(densityValue));
    for (double const scaleValue : {0., -0., -10.}) {
      auto const scale = scaleValue * meter;
      FlatExponential<IMediumModel> const flat(anchor, zeroAxis, density, scale,
                                               composition());
      SlidingPlanarExponential<IMediumModel> const sliding(anchor, density, scale,
                                                           composition(), -7_m);
      for (auto const& parameters : {flat.exportDensity(), sliding.exportDensity()}) {
        REQUIRE(parameters.referenceDensity_ == density);
        REQUIRE(parameters.scaleLength_ == scale);
        REQUIRE(std::signbit(parameters.referenceDensity_ / densityUnit) ==
                std::signbit(densityValue));
        REQUIRE(std::signbit(parameters.scaleLength_ / meter) ==
                std::signbit(scaleValue));
        require_same_point(parameters.anchor_, anchor);
      }
      require_same_vector(flat.exportDensity().axis_, zeroAxis);
      REQUIRE(sliding.exportDensity().referenceHeight_ == -7_m);
    }
  }
}

TEST_CASE("Optional exporters cross cast through composed native model interfaces",
          "[model_export]") {
  auto const root = get_root_CoordinateSystem();
  auto const fieldFrame =
      make_rotation(root, QuantityVector<dimensionless_d>{0., 1., 0.}, 0.3);
  MagneticFieldVector const field(fieldFrame, {10_uT, -20_uT, 30_uT});
  auto const density = 1.2_kg / cube(1_m);
  using first_type =
      UniformRefractiveIndex<UniformMagneticField<HomogeneousMedium<model_interface>>>;
  using second_type =
      UniformMagneticField<UniformRefractiveIndex<HomogeneousMedium<model_interface>>>;
  first_type first(1.0003, field, density, composition());
  second_type second(field, 1.0003, density, composition());
  std::array<model_interface*, 2> const models{&first, &second};
  for (auto* model : models) {
    IMediumModel const& base = *model;
    auto const* densityExporter = dynamic_cast<IDensityModelExport const*>(&base);
    auto const* fieldExporter = dynamic_cast<IUniformMagneticFieldExport const*>(&base);
    auto const* refractiveExporter = dynamic_cast<IRefractivityModelExport const*>(&base);
    REQUIRE(densityExporter != nullptr);
    REQUIRE(fieldExporter != nullptr);
    REQUIRE(refractiveExporter != nullptr);
    REQUIRE(densityExporter->exportDensity().referenceDensity_ == density);
    require_same_vector(fieldExporter->exportUniformField(), field);
    auto const refractivity = refractiveExporter->exportRefractivity();
    REQUIRE(refractivity.kind_ == refractivity_kind::Uniform);
    REQUIRE(refractivity.value_ == 1.0003);
    REQUIRE(refractivity.inverseReferenceDensity_ == InverseMassDensityType::zero());
    require_native_unchanged(base, Point(root, 0_m, 0_m, 0_m));
    REQUIRE(model->getRefractiveIndex(Point(root, 4_m, 5_m, 6_m)) == 1.0003);
    require_same_vector(model->getMagneticField(Point(root, 4_m, 5_m, 6_m)), field);
  }

  auto const oldField = first.exportUniformField();
  auto const oldIndex = first.exportRefractivity();
  MagneticFieldVector const replacement(root, {-5_uT, 6_uT, 7_uT});
  first.setMagneticField(replacement);
  first.setRefractiveIndex(1.4);
  require_same_vector(first.exportUniformField(), replacement);
  REQUIRE(first.exportRefractivity().value_ == 1.4);
  require_same_vector(oldField, field);
  REQUIRE(oldIndex.value_ == 1.0003);
}

TEST_CASE("Gladstone Dale export keeps the original two factors separate",
          "[model_export]") {
  auto const root = get_root_CoordinateSystem();
  auto const frame = make_translation(root, {100_m, -200_m, 300_m});
  Point const anchor(frame, 0_m, 0_m, 0_m);
  Point const reference(frame, 0_m, 0_m, 15_m);
  auto const density = 1.25_kg / cube(1_m);
  double const referenceIndex = 1.000312;
  MagneticFieldVector const field(root, {10_uT, -20_uT, 30_uT});
  using flat_type = GladstoneDaleRefractiveIndex<
      UniformMagneticField<FlatExponential<model_interface>>>;
  using sliding_type = UniformMagneticField<
      GladstoneDaleRefractiveIndex<SlidingPlanarExponential<model_interface>>>;
  flat_type flat(referenceIndex, reference, field, anchor,
                 DirectionVector(frame, {0., 0., 1.}), density, -800_m, composition());
  sliding_type sliding(field, referenceIndex, reference, anchor, density, -800_m,
                       composition(), 5_m);
  std::array<model_interface const*, 2> const models{&flat, &sliding};
  for (auto const* model : models) {
    auto const* exporter = dynamic_cast<IRefractivityModelExport const*>(model);
    REQUIRE(exporter != nullptr);
    std::array<double, 3> const before{
        model->getRefractiveIndex(reference),
        model->getRefractiveIndex(Point(frame, 0_m, 0_m, 30_m)),
        model->getRefractiveIndex(Point(frame, 4_m, 5_m, 20_m))};
    auto const parameters = exporter->exportRefractivity();
    REQUIRE(parameters.kind_ == refractivity_kind::GladstoneDale);
    REQUIRE(parameters.value_ == referenceIndex - 1.);
    REQUIRE(parameters.inverseReferenceDensity_ == 1. / model->getMassDensity(reference));
    std::array<Point, 3> const points{reference, Point(frame, 0_m, 0_m, 30_m),
                                      Point(frame, 4_m, 5_m, 20_m)};
    for (std::size_t index = 0; index < points.size(); ++index) {
      double const originalOrder =
          parameters.value_ * (model->getMassDensity(points[index]) *
                               parameters.inverseReferenceDensity_) +
          1.;
      REQUIRE(originalOrder == before[index]);
      REQUIRE(model->getRefractiveIndex(points[index]) == before[index]);
    }
  }
}

TEST_CASE("Export reads stored parameters without evaluating density or refractivity",
          "[model_export]") {
  auto const root = get_root_CoordinateSystem();
  auto const density = 1.2_kg / cube(1_m);
  using interface_type = IRefractiveIndexModel<IMediumModel>;
  CountingRefractivity<CountingDensity<interface_type>> uniform(1.0003, density,
                                                                composition());
  REQUIRE(uniform.calls_ == 0);
  REQUIRE(uniform.refractiveCalls_ == 0);
  REQUIRE(uniform.exportDensity().referenceDensity_ == density);
  REQUIRE(uniform.exportRefractivity().value_ == 1.0003);
  REQUIRE(uniform.calls_ == 0);
  REQUIRE(uniform.refractiveCalls_ == 0);

  GladstoneDaleRefractiveIndex<CountingDensity<interface_type>> gladstone(
      1.000312, Point(root, 0_m, 0_m, 0_m), density, composition());
  // Native construction evaluates the reference density once. Export must not do so.
  REQUIRE(gladstone.calls_ == 1);
  REQUIRE(gladstone.exportDensity().referenceDensity_ == density);
  REQUIRE(gladstone.exportRefractivity().value_ == 1.000312 - 1.);
  REQUIRE(gladstone.calls_ == 1);
}

TEST_CASE("Independent unsupported models do not acquire optional export contracts",
          "[model_export]") {
  auto const root = get_root_CoordinateSystem();
  IndependentMedium const independent;
  IMediumModel const& native = independent;
  REQUIRE(dynamic_cast<IDensityModelExport const*>(&native) == nullptr);
  REQUIRE(dynamic_cast<IUniformMagneticFieldExport const*>(&native) == nullptr);
  REQUIRE(dynamic_cast<IRefractivityModelExport const*>(&native) == nullptr);
  REQUIRE_FALSE(
      (std::is_base_of_v<IDensityModelExport, SlidingPlanarTabular<IMediumModel>>));

  using exponential_type =
      ExponentialRefractiveIndex<HomogeneousMedium<IRefractiveIndexModel<IMediumModel>>>;
  exponential_type const exponential(1.0003, -1. / 8000_m, Point(root, 0_m, 0_m, 0_m),
                                     6371_km, 1.2_kg / cube(1_m), composition());
  IMediumModel const& exponentialBase = exponential;
  REQUIRE(dynamic_cast<IDensityModelExport const*>(&exponentialBase) != nullptr);
  REQUIRE(dynamic_cast<IRefractivityModelExport const*>(&exponentialBase) == nullptr);
}

TEST_CASE("Five layer atmosphere exports actual composed model parameters",
          "[model_export]") {
  auto const root = get_root_CoordinateSystem();
  auto const frame = make_translation(root, {1_km, 2_km, -3_km});
  Point const center(frame, 0_m, 0_m, 0_m);
  Point const reference(frame, 0_m, 0_m, constants::EarthRadius::Mean);
  MagneticFieldVector const field(frame, {0_uT, 50_uT, 0_uT});
  using interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  Environment<interface_type> environment;
  create_5layer_atmosphere<interface_type, atmosphere_extras>(
      environment, AtmosphereId::LinsleyUSStd, center, 1.000312, reference,
      Medium::AirDry1Atm, field);

  unsigned homogeneousCount{};
  unsigned exponentialCount{};
  unsigned emptyCount{};
  auto const& expected =
      atmosphereParameterList[static_cast<std::uint8_t>(AtmosphereId::LinsleyUSStd)];
  environment.getUniverse()->walk([&](auto const& node) {
    if (!node.hasModelProperties()) {
      ++emptyCount;
      return;
    }
    auto const& native = node.getModelProperties();
    auto const* densityExporter = dynamic_cast<IDensityModelExport const*>(&native);
    auto const* fieldExporter = dynamic_cast<IUniformMagneticFieldExport const*>(&native);
    auto const* refractiveExporter =
        dynamic_cast<IRefractivityModelExport const*>(&native);
    REQUIRE(densityExporter != nullptr);
    REQUIRE(fieldExporter != nullptr);
    REQUIRE(refractiveExporter != nullptr);
    auto const parameters = densityExporter->exportDensity();
    if (parameters.kind_ == density_kind::Homogeneous) {
      ++homogeneousCount;
      REQUIRE(parameters.referenceDensity_ ==
              expected[4].offset / expected[4].scaleHeight);
    } else {
      REQUIRE(parameters.kind_ == density_kind::SlidingPlanarExponential);
      ++exponentialCount;
      require_same_point(parameters.anchor_, center);
      REQUIRE(parameters.referenceHeight_ == constants::EarthRadius::Mean);
      bool found{};
      for (std::size_t index = 0; index < 4; ++index) {
        if (parameters.scaleLength_ == -expected[index].scaleHeight) {
          found = true;
          REQUIRE(parameters.referenceDensity_ ==
                  expected[index].offset / expected[index].scaleHeight);
        }
      }
      REQUIRE(found);
    }
    require_same_vector(fieldExporter->exportUniformField(), field);
    auto const refractivity = refractiveExporter->exportRefractivity();
    REQUIRE(refractivity.kind_ == refractivity_kind::GladstoneDale);
    REQUIRE(refractivity.value_ == 1.000312 - 1.);
    REQUIRE(refractivity.inverseReferenceDensity_ ==
            1. / native.getMassDensity(reference));
    REQUIRE(native.getNuclearComposition().getComponents() ==
            standardAirComposition.getComponents());
    REQUIRE(native.getNuclearComposition().getFractions() ==
            standardAirComposition.getFractions());
    REQUIRE(native.getMedium() == Medium::AirDry1Atm);
    require_native_unchanged(native, reference);
  });
  REQUIRE(homogeneousCount == 1);
  REQUIRE(exponentialCount == 4);
  REQUIRE(emptyCount == 1);
}
