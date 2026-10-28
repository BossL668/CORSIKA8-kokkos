/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/ModelConversion.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/FlatExponential.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IDensityModelExport.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/IMediumModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/IRefractivityModelExport.hpp>
#include <corsika/media/IUniformMagneticFieldExport.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/SlidingPlanarExponential.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

  using namespace corsika;
  using model_type = kokkos::EnvironmentModel;
  using component_type = kokkos::EnvironmentComponent;
  using model_interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;

  template <typename TModel>
  using uniform_extra_type =
      UniformRefractiveIndex<MediumPropertyModel<UniformMagneticField<TModel>>>;

  template <typename TModel>
  using gladstone_extra_type =
      GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<TModel>>>;

  using uniform_model_type = uniform_extra_type<HomogeneousMedium<model_interface_type>>;

  NuclearComposition composition() {
    // The native constructor permits this total without normalizing the entries.
    return NuclearComposition({Code::Oxygen, Code::Nitrogen, Code::Oxygen},
                              {0.1, 0.6995, 0.201});
  }

  std::vector<component_type> prefix() { return {{-17, 0.125}, {44, -0.}}; }

  void require_double(double const actual, double const expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }

  void require_components(std::vector<component_type> const& actual,
                          std::vector<component_type> const& expected) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
      CAPTURE(index);
      REQUIRE(actual[index].code_ == expected[index].code_);
      require_double(actual[index].fraction_, expected[index].fraction_);
    }
  }

  void require_model(model_type const& actual, model_type const& expected) {
    REQUIRE(actual.densityKind_ == expected.densityKind_);
    REQUIRE(actual.refractivityKind_ == expected.refractivityKind_);
    REQUIRE(actual.mediumId_ == expected.mediumId_);
    REQUIRE(actual.compositionBegin_ == expected.compositionBegin_);
    REQUIRE(actual.compositionCount_ == expected.compositionCount_);
    require_double(actual.referenceDensityGPerCm3_, expected.referenceDensityGPerCm3_);
    require_double(actual.scaleLengthM_, expected.scaleLengthM_);
    require_double(actual.inverseScaleLengthPerM_, expected.inverseScaleLengthPerM_);
    require_double(actual.referenceHeightM_, expected.referenceHeightM_);
    require_double(actual.refractivityValue_, expected.refractivityValue_);
    require_double(actual.inverseReferenceDensityCm3PerG_,
                   expected.inverseReferenceDensityCm3PerG_);
    for (int component = 0; component < 3; ++component) {
      CAPTURE(component);
      require_double(actual.anchorM_[component], expected.anchorM_[component]);
      require_double(actual.axis_[component], expected.axis_[component]);
      require_double(actual.magneticFieldT_[component],
                     expected.magneticFieldT_[component]);
    }
  }

  /* Independent transcription of the field conversions in fixed production
   * 7c24233b, FullEnvironmentTreeBuilder.hpp. Only record member names and explicit
   * kind mapping differ. In particular, mutable local copies retain the original
   * Point/Vector conversion overloads. This is not a density evaluator.
   */
  template <typename TModel>
  model_type production_record(TModel const& native, CoordinateSystemPtr const& frame,
                               std::size_t const begin) {
    auto const* d = dynamic_cast<IDensityModelExport const*>(&native);
    auto const* b = dynamic_cast<IUniformMagneticFieldExport const*>(&native);
    auto const* n = dynamic_cast<IRefractivityModelExport const*>(&native);
    REQUIRE(d != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(n != nullptr);
    auto density = d->exportDensity();
    auto field = b->exportUniformField();
    auto refr = n->exportRefractivity();
    model_type result;
    switch (density.kind_) {
      case DensityModelParameters::Kind::Homogeneous:
        result.densityKind_ = model_type::Density::Homogeneous;
        break;
      case DensityModelParameters::Kind::SlidingPlanarExponential:
        result.densityKind_ = model_type::Density::SlidingPlanarExponential;
        break;
      case DensityModelParameters::Kind::FlatExponential:
        result.densityKind_ = model_type::Density::FlatExponential;
        break;
      default:
        throw std::logic_error("Invalid oracle density kind");
    }
    result.referenceDensityGPerCm3_ =
        density.referenceDensity_ / (1_g / (1_cm * 1_cm * 1_cm));
    result.scaleLengthM_ = density.scaleLength_ / 1_m;
    result.inverseScaleLengthPerM_ =
        result.scaleLengthM_ != 0. ? 1. / result.scaleLengthM_ : 0.;
    result.referenceHeightM_ = density.referenceHeight_ / 1_m;
    auto anchor = density.anchor_.getCoordinates(frame);
    auto axis = density.axis_.getComponents(frame);
    auto magnetic = field.getComponents(frame);
    for (int component = 0; component < 3; ++component) {
      result.anchorM_[component] = anchor[component] / 1_m;
      result.axis_[component] = axis[component].magnitude();
      result.magneticFieldT_[component] = magnetic[component] / 1_T;
    }
    result.refractivityKind_ = refr.kind_ == RefractivityModelParameters::Kind::Uniform
                                   ? model_type::Refractivity::Uniform
                                   : model_type::Refractivity::GladstoneDale;
    result.refractivityValue_ = refr.value_;
    result.inverseReferenceDensityCm3PerG_ =
        refr.inverseReferenceDensity_ * (1_g / (1_cm * 1_cm * 1_cm));
    result.mediumId_ = static_cast<std::int32_t>(native.getMedium());
    result.compositionBegin_ = static_cast<std::uint32_t>(begin);
    result.compositionCount_ =
        static_cast<std::uint32_t>(native.getNuclearComposition().getSize());
    return result;
  }

  template <typename TModel>
  void require_failed_unchanged(TModel const& native, CoordinateSystemPtr const& frame) {
    auto components = prefix();
    auto const before = components;
    REQUIRE_THROWS_AS(kokkos::export_model(native, frame, components),
                      std::invalid_argument);
    require_components(components, before);
  }

  // Synthetic descriptors exercise the converter contract only. Overridden
  // descriptors are never used to claim agreement with this fixture's physics.
  class ControlledModel : public uniform_model_type {
  public:
    explicit ControlledModel(CoordinateSystemPtr const& frame)
        : uniform_model_type(1.0003, Medium::AirDry1Atm,
                             corsika::MagneticFieldVector(frame, {1_uT, -2_uT, 3_uT}),
                             1.2_kg / cube(1_m), composition()) {}

    std::optional<DensityModelParameters> density_;
    std::optional<corsika::MagneticFieldVector> field_;
    std::optional<RefractivityModelParameters> refractivity_;

    DensityModelParameters exportDensity() const override {
      return density_ ? *density_ : uniform_model_type::exportDensity();
    }
    corsika::MagneticFieldVector exportUniformField() const override {
      return field_ ? *field_ : uniform_model_type::exportUniformField();
    }
    RefractivityModelParameters exportRefractivity() const override {
      return refractivity_ ? *refractivity_ : uniform_model_type::exportRefractivity();
    }
  };

  class IndependentDensity : public model_interface_type {
  public:
    MassDensityType getMassDensity(Point const&) const override {
      throw std::logic_error("Converter must not evaluate density");
    }
    GrammageType getIntegratedGrammage(BaseTrajectory const&) const override {
      throw std::logic_error("Converter must not integrate density");
    }
    LengthType getArclengthFromGrammage(BaseTrajectory const&,
                                        GrammageType const) const override {
      throw std::logic_error("Converter must not invert grammage");
    }
    NuclearComposition const& getNuclearComposition() const override { return mixture_; }

  private:
    NuclearComposition mixture_{composition()};
  };

  template <typename TModel>
  class CountingDensity : public HomogeneousMedium<TModel> {
  public:
    using HomogeneousMedium<TModel>::HomogeneousMedium;
    mutable unsigned densityCalls_{};
    MassDensityType getMassDensity(Point const& point) const override {
      ++densityCalls_;
      return HomogeneousMedium<TModel>::getMassDensity(point);
    }
  };

} // namespace

TEST_CASE("Numeric models reproduce production conversions in different frames",
          "[kokkos_model_conversion]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {1_km, -2_km, 3_km});
  auto const rotated =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  auto const sameRotation =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  auto const nested = make_translation(rotated, {4_m, 5_m, -6_m});
  REQUIRE(rotated != sameRotation);
  REQUIRE(*rotated == *sameRotation);
  Point const anchor(nested, 3_m, -4_m, 5_m);
  Point const reference(nested, 3_m, -4_m, 25_m);
  DirectionVector const axis(rotated, {0., 0., 2.5});
  MagneticFieldVector const field(rotated, {11_uT, -22_uT, 33_uT});
  auto const rho = 1.25_kg / cube(1_m);
  auto const mixture = composition();
  uniform_model_type homogeneous(1.000312, Medium::AirDry1Atm, field, rho, mixture);
  uniform_extra_type<FlatExponential<model_interface_type>> flat(
      1.000312, Medium::AirDry1Atm, field, anchor, axis, rho, -800_m, mixture);
  uniform_extra_type<SlidingPlanarExponential<model_interface_type>> sliding(
      1.000312, Medium::AirDry1Atm, field, anchor, rho, -900_m, mixture, 7_m);
  gladstone_extra_type<HomogeneousMedium<model_interface_type>> homogeneousGD(
      1.000312, reference, Medium::AirDry1Atm, field, rho, mixture);
  gladstone_extra_type<FlatExponential<model_interface_type>> flatGD(
      1.000312, reference, Medium::AirDry1Atm, field, anchor, axis, rho, -800_m, mixture);
  gladstone_extra_type<SlidingPlanarExponential<model_interface_type>> slidingGD(
      1.000312, reference, Medium::AirDry1Atm, field, anchor, rho, -900_m, mixture, 7_m);
  std::array<model_interface_type const*, 6> const models{
      &homogeneous, &flat, &sliding, &homogeneousGD, &flatGD, &slidingGD};
  for (auto const* model : models) {
    auto const* densityExporter = dynamic_cast<IDensityModelExport const*>(model);
    auto const* fieldExporter = dynamic_cast<IUniformMagneticFieldExport const*>(model);
    REQUIRE(densityExporter != nullptr);
    REQUIRE(fieldExporter != nullptr);
    auto const beforeDensity = densityExporter->exportDensity();
    auto const beforeField = fieldExporter->exportUniformField();
    for (auto const& frame : {root, translated, rotated, sameRotation, nested}) {
      auto components = prefix();
      auto const expected = production_record(*model, frame, components.size());
      auto const actual = kokkos::export_model(*model, frame, components);
      require_model(actual, expected);
      REQUIRE(components.size() == prefix().size() + mixture.getSize());
    }
    auto const afterDensity = densityExporter->exportDensity();
    auto const afterField = fieldExporter->exportUniformField();
    REQUIRE(afterDensity.anchor_.getCoordinateSystem() ==
            beforeDensity.anchor_.getCoordinateSystem());
    REQUIRE(afterDensity.axis_.getCoordinateSystem() ==
            beforeDensity.axis_.getCoordinateSystem());
    REQUIRE(afterField.getCoordinateSystem() == beforeField.getCoordinateSystem());
    for (int component = 0; component < 3; ++component) {
      REQUIRE(afterDensity.anchor_.getCoordinates()[component] ==
              beforeDensity.anchor_.getCoordinates()[component]);
      REQUIRE(afterDensity.axis_.getComponents()[component] ==
              beforeDensity.axis_.getComponents()[component]);
      REQUIRE(afterField.getComponents()[component] ==
              beforeField.getComponents()[component]);
    }
  }
}

TEST_CASE("Composition appends native codes and ordered unnormalized number fractions",
          "[kokkos_model_conversion]") {
  auto const root = get_root_CoordinateSystem();
  auto const mixture = composition();
  MagneticFieldVector const field(root, {0_T, 0_T, 0_T});
  uniform_model_type const model(1.0003, Medium::AirDry1Atm, field, 1.2_kg / cube(1_m),
                                 mixture);
  auto components = prefix();
  auto expected = components;
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    auto const begin = components.size();
    auto const record = kokkos::export_model(model, root, components);
    REQUIRE(record.compositionBegin_ == begin);
    REQUIRE(record.compositionCount_ == mixture.getSize());
    REQUIRE(record.mediumId_ == static_cast<std::int32_t>(Medium::AirDry1Atm));
    for (std::size_t index = 0; index < mixture.getSize(); ++index) {
      expected.push_back({static_cast<std::int32_t>(mixture.getComponents()[index]),
                          mixture.getFractions()[index]});
    }
    require_components(components, expected);
  }
  // Nucleus codes can coincide numerically with PDG codes; protons exercise
  // the distinction between the native enum and a PDG conversion.
  NuclearComposition const protonComposition({Code::Proton}, {1.});
  uniform_model_type const protonModel(1.0003, Medium::AirDry1Atm, field,
                                       1.2_kg / cube(1_m), protonComposition);
  std::vector<component_type> protonComponents;
  auto const protonRecord = kokkos::export_model(protonModel, root, protonComponents);
  REQUIRE(protonRecord.compositionCount_ == 1);
  REQUIRE(protonComponents[0].code_ == static_cast<std::int32_t>(Code::Proton));
  REQUIRE(protonComponents[0].code_ != static_cast<std::int32_t>(get_PDG(Code::Proton)));
  REQUIRE(components[2].code_ == components[4].code_);
  REQUIRE(components[2].fraction_ != components[4].fraction_);
  REQUIRE(mixture.getFractions()[0] + mixture.getFractions()[1] +
              mixture.getFractions()[2] !=
          1.);
  // The native constructor checks the total, not positivity. Preserve raw input
  // here without asserting that a signed-fraction material is physically valid.
  NuclearComposition const signedMixture({Code::Oxygen, Code::Nitrogen, Code::Oxygen},
                                         {-0.1, 0.8995, 0.201});
  uniform_model_type const signedModel(1.0003, Medium::AirDry1Atm, field,
                                       1.2_kg / cube(1_m), signedMixture);
  auto const signedRecord = kokkos::export_model(signedModel, root, components);
  for (std::size_t index = 0; index < signedMixture.getSize(); ++index) {
    require_double(components[signedRecord.compositionBegin_ + index].fraction_,
                   signedMixture.getFractions()[index]);
  }
}

TEST_CASE("Numeric conversion preserves raw parameters and the production zero branch",
          "[kokkos_model_conversion]") {
  auto const root = get_root_CoordinateSystem();
  ControlledModel model(root);
  auto density = model.exportDensity();
  density.referenceDensity_ = -0. * (1_g / cube(1_cm));
  density.referenceHeight_ = -7_m;
  density.anchor_ = Point(root, -0_m, 2_m, 3_m);
  density.axis_ = DirectionVector(root, {0., -0., 2.5});
  for (auto const kind : {DensityModelParameters::Kind::Homogeneous,
                          DensityModelParameters::Kind::FlatExponential,
                          DensityModelParameters::Kind::SlidingPlanarExponential}) {
    density.kind_ = kind;
    for (double const scale : {0., -0., -800., 800.}) {
      density.scaleLength_ = scale * meter;
      model.density_ = density;
      auto components = prefix();
      auto const actual = kokkos::export_model(model, root, components);
      require_model(actual, production_record(model, root, prefix().size()));
      if (scale == 0.) {
        REQUIRE(actual.inverseScaleLengthPerM_ == 0.);
        REQUIRE_FALSE(std::signbit(actual.inverseScaleLengthPerM_));
      }
      REQUIRE(actual.axis_[2] == 2.5);
    }
  }
  // Export does not declare these models physically valid or evaluate their laws.
  for (double const value :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(), -2.}) {
    density.referenceDensity_ = value * (1_g / cube(1_cm));
    density.scaleLength_ = value * meter;
    model.density_ = density;
    model.field_ = MagneticFieldVector(root, {value * tesla, 0_T, 0_T});
    model.refractivity_ =
        RefractivityModelParameters{RefractivityModelParameters::Kind::GladstoneDale,
                                    value, value / (1_g / cube(1_cm))};
    auto components = prefix();
    auto const actual = kokkos::export_model(model, root, components);
    require_model(actual, production_record(model, root, prefix().size()));
  }
}

TEST_CASE("Numeric model conversion rejects missing exporters without appending",
          "[kokkos_model_conversion]") {
  auto const root = get_root_CoordinateSystem();
  MagneticFieldVector const field(root, {0_T, 50_uT, 0_T});
  uniform_extra_type<IndependentDensity> const missingDensity(1.0003, Medium::AirDry1Atm,
                                                              field);
  require_failed_unchanged(missingDensity, root);
  using no_field_interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMediumModel>>;
  using no_refractivity_interface_type =
      IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>;
  UniformRefractiveIndex<
      MediumPropertyModel<HomogeneousMedium<no_field_interface_type>>> const
      missingField(1.0003, Medium::AirDry1Atm, 1.2_kg / cube(1_m), composition());
  require_failed_unchanged(missingField, root);
  MediumPropertyModel<
      UniformMagneticField<HomogeneousMedium<no_refractivity_interface_type>>> const
      missingRefractivity(Medium::AirDry1Atm, field, 1.2_kg / cube(1_m), composition());
  require_failed_unchanged(missingRefractivity, root);
}

TEST_CASE("Invalid model kinds and null frames preserve the destination composition",
          "[kokkos_model_conversion]") {
  auto const root = get_root_CoordinateSystem();
  CoordinateSystemPtr const missing;
  ControlledModel model(root);
  require_failed_unchanged(model, missing);
  auto density = model.exportDensity();
  auto const originalDensity = density;
  density.kind_ = static_cast<DensityModelParameters::Kind>(99);
  model.density_ = density;
  require_failed_unchanged(model, root);
  model.density_.reset();
  auto refractivity = model.exportRefractivity();
  refractivity.kind_ = static_cast<RefractivityModelParameters::Kind>(99);
  model.refractivity_ = refractivity;
  require_failed_unchanged(model, root);
  model.refractivity_.reset();
  density = originalDensity;
  density.anchor_ = Point(missing, 0_m, 0_m, 0_m);
  model.density_ = density;
  require_failed_unchanged(model, root);
  density = originalDensity;
  density.axis_ = DirectionVector(missing, {0., 0., 1.});
  model.density_ = density;
  require_failed_unchanged(model, root);
  model.density_.reset();
  model.field_ = MagneticFieldVector(missing, {0_T, 0_T, 1_T});
  require_failed_unchanged(model, root);
}

TEST_CASE("Reexport updates mutable native settings without changing earlier records",
          "[kokkos_model_conversion]") {
  auto const root = get_root_CoordinateSystem();
  auto const frame = make_translation(root, {1_km, 2_km, -3_km});
  uniform_model_type model(1.0003, Medium::AirDry1Atm,
                           MagneticFieldVector(frame, {1_uT, 2_uT, 3_uT}),
                           1.2_kg / cube(1_m), composition());
  auto components = prefix();
  auto const expectedBefore = production_record(model, root, components.size());
  auto const before = kokkos::export_model(model, root, components);
  auto const firstComponents = components;
  model.setMagneticField(MagneticFieldVector(root, {-4_uT, 5_uT, 6_uT}));
  model.setRefractiveIndex(1.35);
  auto const expectedAfter = production_record(model, frame, components.size());
  auto const after = kokkos::export_model(model, frame, components);
  require_model(before, expectedBefore);
  require_model(after, expectedAfter);
  REQUIRE(before.refractivityValue_ == 1.0003);
  REQUIRE(after.refractivityValue_ == 1.35);
  std::vector<component_type> const preserved(
      components.begin(), components.begin() + firstComponents.size());
  require_components(preserved, firstComponents);
}

TEST_CASE("Numeric model conversion copies stored factors without density evaluation",
          "[kokkos_model_conversion]") {
  auto const root = get_root_CoordinateSystem();
  MagneticFieldVector const field(root, {0_T, 50_uT, 0_T});
  auto const rho = 1.2_kg / cube(1_m);
  uniform_extra_type<CountingDensity<model_interface_type>> uniform(
      1.0003, Medium::AirDry1Atm, field, rho, composition());
  gladstone_extra_type<CountingDensity<model_interface_type>> gladstone(
      1.000312, Point(root, 0_m, 0_m, 0_m), Medium::AirDry1Atm, field, rho,
      composition());
  REQUIRE(uniform.densityCalls_ == 0);
  REQUIRE(gladstone.densityCalls_ == 1);
  std::vector<component_type> components;
  auto const uniformRecord = kokkos::export_model(uniform, root, components);
  auto const gladstoneRecord = kokkos::export_model(gladstone, root, components);
  REQUIRE(uniform.densityCalls_ == 0);
  REQUIRE(gladstone.densityCalls_ == 1);
  REQUIRE(uniformRecord.refractivityKind_ == model_type::Refractivity::Uniform);
  REQUIRE(uniformRecord.inverseReferenceDensityCm3PerG_ == 0.);
  REQUIRE(gladstoneRecord.refractivityKind_ == model_type::Refractivity::GladstoneDale);
  require_double(gladstoneRecord.refractivityValue_, 1.000312 - 1.);
  require_double(gladstoneRecord.inverseReferenceDensityCm3PerG_,
                 (1. / rho) * (1_g / (1_cm * 1_cm * 1_cm)));
}

TEST_CASE("Actual five layer models convert independently of topology and geometry",
          "[kokkos_model_conversion]") {
  auto const root = get_root_CoordinateSystem();
  auto const frame = make_translation(root, {1_km, -2_km, 3_km});
  Point const center(frame, 0_m, 0_m, 0_m);
  Point const reference(frame, 0_m, 0_m, constants::EarthRadius::Mean);
  Environment<model_interface_type> environment;
  create_5layer_atmosphere<model_interface_type, gladstone_extra_type>(
      environment, AtmosphereId::LinsleyUSStd, center, 1.000312, reference,
      Medium::AirDry1Atm, MagneticFieldVector(frame, {0_T, 50_uT, 0_T}));
  std::vector<component_type> components;
  unsigned homogeneousCount{};
  unsigned slidingCount{};
  unsigned absentCount{};
  environment.getUniverse()->walk([&](auto const& node) {
    if (!node.hasModelProperties()) {
      ++absentCount;
      return;
    }
    auto const& native = node.getModelProperties();
    auto const expected = production_record(native, root, components.size());
    auto const actual = kokkos::export_model(native, root, components);
    require_model(actual, expected);
    if (actual.densityKind_ == model_type::Density::Homogeneous) {
      ++homogeneousCount;
    } else {
      REQUIRE(actual.densityKind_ == model_type::Density::SlidingPlanarExponential);
      ++slidingCount;
    }
    for (std::size_t index = 0; index < actual.compositionCount_; ++index) {
      auto const& entry = components[actual.compositionBegin_ + index];
      REQUIRE(entry.code_ ==
              static_cast<std::int32_t>(standardAirComposition.getComponents()[index]));
      require_double(entry.fraction_, standardAirComposition.getFractions()[index]);
    }
  });
  REQUIRE(homogeneousCount == 1);
  REQUIRE(slidingCount == 4);
  REQUIRE(absentCount == 1);
}
