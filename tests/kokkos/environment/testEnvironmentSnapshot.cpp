/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentSnapshot.hpp>
#include <corsika/framework/geometry/Box.hpp>
#include <corsika/framework/geometry/SeparationPlane.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/FlatExponential.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IDensityModelExport.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/IRefractivityModelExport.hpp>
#include <corsika/media/IUniformMagneticFieldExport.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/SlidingPlanarExponential.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <corsika/media/Universe.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

  using namespace corsika;
  using model_interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  using environment_type = Environment<model_interface_type>;
  using node_type = environment_type::BaseNodeType;
  using snapshot_type = kokkos::EnvironmentSnapshot<model_interface_type>;
  using model_type = kokkos::EnvironmentModel;
  using shape_type = kokkos::EnvironmentShape;

  template <typename TModel>
  using uniform_extra_type =
      UniformRefractiveIndex<MediumPropertyModel<UniformMagneticField<TModel>>>;

  template <typename TModel>
  using gladstone_extra_type =
      GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<TModel>>>;

  using uniform_model_type = uniform_extra_type<HomogeneousMedium<model_interface_type>>;

  NuclearComposition composition() {
    // Native number fractions are ordered, repeated and not renormalized.
    return NuclearComposition({Code::Oxygen, Code::Nitrogen, Code::Oxygen},
                              {0.1, 0.6995, 0.201});
  }

  auto sphere(CoordinateSystemPtr const& frame, LengthType const radius = 1_m) {
    return environment_type::createNode<Sphere>(Point(frame, 3_m, -4_m, 5_m), radius);
  }

  auto set_uniform(node_type& node, CoordinateSystemPtr const& frame) {
    return node.setModelProperties<uniform_model_type>(
        1.000312, Medium::AirDry1Atm, MagneticFieldVector(frame, {11_uT, -22_uT, 33_uT}),
        1.25_kg / cube(1_m), composition());
  }

  void require_double(double const actual, double const expected) {
    REQUIRE(actual == expected);
    if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
  }

  // Independent parameter expressions from production 7c24233b,
  // EnvironmentTreeBuilder.hpp. Shape and model conversion entry points are not
  // called by the oracle; the original const Point/Vector overloads are retained.
  shape_type production_shape(IVolume const& volume, CoordinateSystemPtr const& frame) {
    shape_type result;
    auto point = [&](Point const& value, double(&out)[3]) {
      auto const coordinates = value.getCoordinates(frame);
      for (int axis = 0; axis < 3; ++axis) { out[axis] = coordinates[axis] / 1_m; }
    };
    auto vector = [&](DirectionVector const& value, double(&out)[3]) {
      auto const components = value.getComponents(frame);
      for (int axis = 0; axis < 3; ++axis) { out[axis] = components[axis].magnitude(); }
    };
    if (dynamic_cast<Universe const*>(&volume)) {
      result.kind_ = shape_type::Kind::Universe;
    } else if (auto const* native = dynamic_cast<Sphere const*>(&volume)) {
      result.kind_ = shape_type::Kind::Sphere;
      point(native->getCenter(), result.centerM_);
      result.radiusM_ = native->getRadius() / 1_m;
    } else if (auto const* native = dynamic_cast<Box const*>(&volume)) {
      result.kind_ = shape_type::Kind::Box;
      auto const boxFrame = native->getCoordinateSystem();
      point(Point(boxFrame, 0_m, 0_m, 0_m), result.centerM_);
      vector(DirectionVector(boxFrame, {1., 0., 0.}), result.axes_[0]);
      vector(DirectionVector(boxFrame, {0., 1., 0.}), result.axes_[1]);
      vector(DirectionVector(boxFrame, {0., 0., 1.}), result.axes_[2]);
      result.halfExtentM_[0] = native->getX() / 1_m;
      result.halfExtentM_[1] = native->getY() / 1_m;
      result.halfExtentM_[2] = native->getZ() / 1_m;
    } else if (auto const* native = dynamic_cast<SeparationPlane const*>(&volume)) {
      result.kind_ = shape_type::Kind::SeparationPlane;
      point(native->getPlane().getCenter(), result.centerM_);
      vector(native->getPlane().getNormal(), result.normal_);
    } else {
      throw std::logic_error("Unsupported shape in snapshot oracle");
    }
    return result;
  }

  void require_shape(shape_type const& actual, shape_type const& expected) {
    REQUIRE(actual.kind_ == expected.kind_);
    require_double(actual.radiusM_, expected.radiusM_);
    for (int axis = 0; axis < 3; ++axis) {
      CAPTURE(axis);
      require_double(actual.centerM_[axis], expected.centerM_[axis]);
      require_double(actual.halfExtentM_[axis], expected.halfExtentM_[axis]);
      require_double(actual.normal_[axis], expected.normal_[axis]);
      for (int component = 0; component < 3; ++component) {
        require_double(actual.axes_[axis][component], expected.axes_[axis][component]);
      }
    }
  }

  // Independent transcription of production FullEnvironmentTreeBuilder.hpp.
  // These mutable descriptor copies deliberately use the production conversion
  // overloads, without changing the native objects or evaluating their physics.
  model_type production_model(model_interface_type const& native,
                              CoordinateSystemPtr const& frame, std::size_t const begin) {
    auto const* densityExporter = dynamic_cast<IDensityModelExport const*>(&native);
    auto const* fieldExporter = dynamic_cast<IUniformMagneticFieldExport const*>(&native);
    auto const* refractivityExporter =
        dynamic_cast<IRefractivityModelExport const*>(&native);
    REQUIRE(densityExporter != nullptr);
    REQUIRE(fieldExporter != nullptr);
    REQUIRE(refractivityExporter != nullptr);
    auto density = densityExporter->exportDensity();
    auto field = fieldExporter->exportUniformField();
    auto refractivity = refractivityExporter->exportRefractivity();
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
        throw std::logic_error("Unsupported density in snapshot oracle");
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
    result.refractivityKind_ =
        refractivity.kind_ == RefractivityModelParameters::Kind::Uniform
            ? model_type::Refractivity::Uniform
            : model_type::Refractivity::GladstoneDale;
    result.refractivityValue_ = refractivity.value_;
    result.inverseReferenceDensityCm3PerG_ =
        refractivity.inverseReferenceDensity_ * (1_g / (1_cm * 1_cm * 1_cm));
    result.mediumId_ = static_cast<std::int32_t>(native.getMedium());
    result.compositionBegin_ = static_cast<std::uint32_t>(begin);
    result.compositionCount_ =
        static_cast<std::uint32_t>(native.getNuclearComposition().getSize());
    return result;
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
      require_double(actual.anchorM_[component], expected.anchorM_[component]);
      require_double(actual.axis_[component], expected.axis_[component]);
      require_double(actual.magneticFieldT_[component],
                     expected.magneticFieldT_[component]);
    }
  }

  void require_native(snapshot_type const& snapshot, node_type const& root) {
    std::vector<node_type const*> preorder;
    root.walk([&](auto const& node) { preorder.push_back(&node); });
    auto nativeIndex = [&](node_type const* pointer) -> std::int32_t {
      if (!pointer) { return -1; }
      auto const found = std::find(preorder.begin(), preorder.end(), pointer);
      REQUIRE(found != preorder.end());
      return static_cast<std::int32_t>(found - preorder.begin());
    };
    auto const& topology = snapshot.getTopology();
    REQUIRE(topology.getNodes().size() == preorder.size());
    REQUIRE(snapshot.getShapes().size() == preorder.size());
    std::vector<model_interface_type const*> models;
    std::size_t childOffset{}, excludedOffset{}, compositionOffset{};
    for (std::size_t index = 0; index < preorder.size(); ++index) {
      CAPTURE(index);
      auto const& native = *preorder[index];
      auto const& node = topology.getNodes()[index];
      REQUIRE(native.hasVolume());
      REQUIRE(topology.getNativeNode(static_cast<std::int32_t>(index)).get_ptr() ==
              &native);
      REQUIRE(topology.getNodeIndex(*(&native)) == static_cast<std::int32_t>(index));
      REQUIRE(node.parentIndex_ == nativeIndex(native.getParent()));
      REQUIRE(node.childOffset_ == childOffset);
      REQUIRE(node.childCount_ == native.getChildNodes().size());
      REQUIRE(node.excludedOffset_ == excludedOffset);
      REQUIRE(node.excludedCount_ == native.getExcludedNodes().size());
      for (auto const& child : native.getChildNodes()) {
        REQUIRE(topology.getChildren().at(childOffset++) == nativeIndex(child.get()));
      }
      for (auto const* excluded : native.getExcludedNodes()) {
        REQUIRE(topology.getExclusions().at(excludedOffset++) == nativeIndex(excluded));
      }
      require_shape(snapshot.getShapes()[index],
                    production_shape(native.getVolume(), snapshot.getCoordinateSystem()));
      if (!native.hasModelProperties()) {
        REQUIRE(node.modelIndex_ == -1);
        continue;
      }
      auto const* model = &native.getModelProperties();
      auto const found = std::find(models.begin(), models.end(), model);
      auto const modelIndex = static_cast<std::int32_t>(found - models.begin());
      REQUIRE(node.modelIndex_ == modelIndex);
      REQUIRE(topology.getNativeModel(modelIndex).get_ptr() == model);
      REQUIRE(topology.getModelIndex(*(model)) == modelIndex);
      if (found != models.end()) { continue; }
      models.push_back(model);
      auto const expected =
          production_model(*model, snapshot.getCoordinateSystem(), compositionOffset);
      require_model(snapshot.getModels().at(modelIndex), expected);
      auto const& mixture = model->getNuclearComposition();
      for (std::size_t component = 0; component < mixture.getSize(); ++component) {
        auto const& actual = snapshot.getComposition().at(compositionOffset++);
        REQUIRE(actual.code_ ==
                static_cast<std::int32_t>(mixture.getComponents()[component]));
        require_double(actual.fraction_, mixture.getFractions()[component]);
      }
    }
    REQUIRE(snapshot.getModels().size() == models.size());
    REQUIRE(snapshot.getComposition().size() == compositionOffset);
    REQUIRE(topology.getChildren().size() == childOffset);
    REQUIRE(topology.getExclusions().size() == excludedOffset);
  }

  class CountingModel : public uniform_model_type {
  public:
    using uniform_model_type::uniform_model_type;
    mutable unsigned densityExports_{};
    mutable unsigned fieldExports_{};
    mutable unsigned refractivityExports_{};

    DensityModelParameters exportDensity() const override {
      ++densityExports_;
      return uniform_model_type::exportDensity();
    }
    corsika::MagneticFieldVector exportUniformField() const override {
      ++fieldExports_;
      return uniform_model_type::exportUniformField();
    }
    RefractivityModelParameters exportRefractivity() const override {
      ++refractivityExports_;
      return uniform_model_type::exportRefractivity();
    }
    MassDensityType getMassDensity(Point const&) const override {
      throw std::logic_error("Snapshot must not evaluate density");
    }
    GrammageType getIntegratedGrammage(BaseTrajectory const&) const override {
      throw std::logic_error("Snapshot must not integrate density");
    }
    LengthType getArclengthFromGrammage(BaseTrajectory const&,
                                        GrammageType) const override {
      throw std::logic_error("Snapshot must not invert grammage");
    }
  };

  class IndependentDensity : public model_interface_type {
  public:
    MassDensityType getMassDensity(Point const&) const override {
      throw std::logic_error("Unsupported model must not be sampled");
    }
    GrammageType getIntegratedGrammage(BaseTrajectory const&) const override {
      throw std::logic_error("Unsupported model must not be integrated");
    }
    LengthType getArclengthFromGrammage(BaseTrajectory const&,
                                        GrammageType) const override {
      throw std::logic_error("Unsupported model must not be inverted");
    }
    NuclearComposition const& getNuclearComposition() const override { return mixture_; }

  private:
    NuclearComposition mixture_{composition()};
  };

  class UnsupportedVolume : public IVolume {
  public:
    bool contains(Point const&) const override {
      throw std::logic_error("Snapshot must not sample containment");
    }
  };

} // namespace

TEST_CASE("Empty universe snapshot retains its node and common coordinate system",
          "[kokkos_environment_snapshot]") {
  environment_type environment;
  auto const frame = make_translation(get_root_CoordinateSystem(), {1_m, 2_m, -3_m});
  snapshot_type const snapshot(*environment.getUniverse(), frame);
  REQUIRE(snapshot.getCoordinateSystem() == frame);
  require_native(snapshot, *environment.getUniverse());
  REQUIRE(snapshot.getShapes().size() == 1);
  REQUIRE(snapshot.getShapes()[0].kind_ == shape_type::Kind::Universe);
  REQUIRE(snapshot.getModels().empty());
  REQUIRE(snapshot.getComposition().empty());
}

TEST_CASE("Full snapshot preserves preorder shapes and directed repeated exclusions",
          "[kokkos_environment_snapshot]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {1_km, -2_km, 3_km});
  auto const rotated =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  auto const equalRotation =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  REQUIRE(rotated != equalRotation);
  REQUIRE(*rotated == *equalRotation);
  environment_type environment;
  auto outer = sphere(translated, 100_m);
  auto first = environment_type::createNode<Box>(rotated, 2_m, -3_m, 0_m);
  auto nested = sphere(rotated, 0_m);
  auto second = environment_type::createNode<SeparationPlane>(
      Plane(Point(translated, 4_m, 5_m, 6_m), DirectionVector(rotated, {0., 0., 7.})));
  auto third = sphere(root, -30_m);
  auto const* firstPointer = first.get();
  auto const* secondPointer = second.get();
  auto const* thirdPointer = third.get();
  first->excludeOverlapWith(third);
  first->excludeOverlapWith(second);
  first->excludeOverlapWith(third);
  second->excludeOverlapWith(first);
  first->addChild(std::move(nested));
  outer->addChild(std::move(first));
  outer->addChild(std::move(second));
  outer->addChild(std::move(third));
  environment.getUniverse()->addChild(std::move(outer));
  for (auto const& frame : {root, translated, rotated, equalRotation}) {
    snapshot_type const snapshot(*environment.getUniverse(), frame);
    REQUIRE(snapshot.getCoordinateSystem() == frame);
    require_native(snapshot, *environment.getUniverse());
    REQUIRE(snapshot.getTopology().getChildren() ==
            std::vector<std::int32_t>{1, 2, 4, 5, 3});
    REQUIRE(snapshot.getTopology().getExclusions() ==
            std::vector<std::int32_t>{5, 4, 5, 2});
    REQUIRE(snapshot.getTopology().getNodeIndex(*(firstPointer)) == 2);
    REQUIRE(snapshot.getTopology().getNodeIndex(*(secondPointer)) == 4);
    REQUIRE(snapshot.getTopology().getNodeIndex(*(thirdPointer)) == 5);
    REQUIRE(snapshot.getModels().empty());
  }
  // No shape, coordinate-system or ownership mutation during export.
  REQUIRE(firstPointer->getExcludedNodes() ==
          std::vector<node_type const*>{thirdPointer, secondPointer, thirdPointer});
  auto const& box = dynamic_cast<Box const&>(firstPointer->getVolume());
  REQUIRE(box.getCoordinateSystem() == rotated);
  REQUIRE(box.getX() == 2_m);
  REQUIRE(box.getY() == -3_m);
  REQUIRE(box.getZ() == 0_m);
}

TEST_CASE("Snapshot deduplicates model pointers but not equal values or compositions",
          "[kokkos_environment_snapshot]") {
  auto const frame = get_root_CoordinateSystem();
  environment_type environment;
  auto first = sphere(frame);
  auto missing = sphere(frame);
  auto sharedNode = sphere(frame);
  auto distinct = sphere(frame);
  auto const* missingPointer = missing.get();
  auto const shared = set_uniform(*first, frame);
  sharedNode->setModelProperties(shared);
  auto const sameValues = set_uniform(*distinct, frame);
  first->addChild(std::move(missing));
  environment.getUniverse()->addChild(std::move(first));
  environment.getUniverse()->addChild(std::move(sharedNode));
  environment.getUniverse()->addChild(std::move(distinct));
  snapshot_type const snapshot(*environment.getUniverse(), frame);
  require_native(snapshot, *environment.getUniverse());
  REQUIRE(snapshot.getModels().size() == 2);
  REQUIRE(snapshot.getComposition().size() == 2 * composition().getSize());
  REQUIRE(snapshot.getTopology().getModelIndex(*(shared.get())) == 0);
  REQUIRE(snapshot.getTopology().getModelIndex(*(sameValues.get())) == 1);
  REQUIRE(snapshot.getTopology()
              .getNodes()[snapshot.getTopology().getNodeIndex(*(missingPointer))]
              .modelIndex_ == -1);
  REQUIRE(snapshot.getModels()[0].compositionBegin_ == 0);
  REQUIRE(snapshot.getModels()[1].compositionBegin_ == composition().getSize());
  REQUIRE(snapshot.getComposition()[0].code_ == snapshot.getComposition()[2].code_);
  require_double(snapshot.getComposition()[1].fraction_, 0.6995);
}

TEST_CASE("Snapshot exports each shared model once without density evaluation",
          "[kokkos_environment_snapshot]") {
  auto const root = get_root_CoordinateSystem();
  auto const sourceFrame = make_translation(root, {2_km, -3_km, 4_km});
  environment_type environment;
  auto const model = std::static_pointer_cast<CountingModel>(
      environment.getUniverse()->setModelProperties<CountingModel>(
          1.000312, Medium::AirDry1Atm,
          MagneticFieldVector(sourceFrame, {1_uT, 2_uT, 3_uT}), 1.25_kg / cube(1_m),
          composition()));
  for (int child = 0; child < 5; ++child) {
    auto node = sphere(sourceFrame);
    node->setModelProperties(model);
    environment.getUniverse()->addChild(std::move(node));
  }
  snapshot_type const first(*environment.getUniverse(), root);
  REQUIRE(model->densityExports_ == 1);
  REQUIRE(model->fieldExports_ == 1);
  REQUIRE(model->refractivityExports_ == 1);
  REQUIRE(first.getModels().size() == 1);
  REQUIRE(first.getComposition().size() == composition().getSize());
  auto copy = first;
  auto moved = std::move(copy);
  REQUIRE(model->densityExports_ == 1);
  REQUIRE(model->fieldExports_ == 1);
  REQUIRE(model->refractivityExports_ == 1);
  REQUIRE(moved.getCoordinateSystem() == root);
  REQUIRE(moved.getTopology().getNativeModel(0).get_ptr() == model.get());
  REQUIRE(moved.getModels().data() != first.getModels().data());
  REQUIRE(moved.getShapes().data() != first.getShapes().data());
  REQUIRE(moved.getComposition().data() != first.getComposition().data());
  require_model(moved.getModels()[0], first.getModels()[0]);
  auto const field = model->uniform_model_type::exportUniformField();
  REQUIRE(field.getCoordinateSystem() == sourceFrame);
  REQUIRE(field.getComponents()[0] == 1_uT);
  // A new numeric snapshot is required after changing a native model. Existing
  // numeric arrays remain independent; the old native map is not used here.
  auto const original = first.getModels()[0];
  model->setMagneticField(MagneticFieldVector(sourceFrame, {5_uT, -6_uT, 7_uT}));
  snapshot_type const second(*environment.getUniverse(), root);
  REQUIRE(model->densityExports_ == 2);
  REQUIRE(model->fieldExports_ == 2);
  REQUIRE(model->refractivityExports_ == 2);
  require_model(first.getModels()[0], original);
  REQUIRE(second.getModels()[0].magneticFieldT_[0] == 5_uT / 1_T);
}

TEST_CASE("Snapshot assembles mixed model kinds in one frame without rebasing sources",
          "[kokkos_environment_snapshot]") {
  auto const root = get_root_CoordinateSystem();
  auto const frame = make_rotation(make_translation(root, {1_km, -2_km, 3_km}),
                                   QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  Point const anchor(frame, 3_m, -4_m, 5_m);
  Point const reference(frame, 3_m, -4_m, 25_m);
  DirectionVector const axis(frame, {0., 0., 2.5});
  MagneticFieldVector const field(frame, {11_uT, -22_uT, 33_uT});
  environment_type environment;
  auto homogeneous = sphere(frame);
  auto flat = sphere(frame);
  auto sliding = sphere(frame);
  set_uniform(*homogeneous, frame);
  using flat_model_type = uniform_extra_type<FlatExponential<model_interface_type>>;
  using sliding_model_type =
      gladstone_extra_type<SlidingPlanarExponential<model_interface_type>>;
  auto const flatModel =
      std::static_pointer_cast<flat_model_type>(flat->setModelProperties<flat_model_type>(
          1.000312, Medium::AirDry1Atm, field, anchor, axis, 1.25_kg / cube(1_m), -800_m,
          composition()));
  auto const slidingModel = std::static_pointer_cast<sliding_model_type>(
      sliding->setModelProperties<sliding_model_type>(
          1.000312, reference, Medium::AirDry1Atm, field, anchor, 1.25_kg / cube(1_m),
          -900_m, composition(), 7_m));
  auto const flatBefore = flatModel->exportDensity();
  auto const slidingBefore = slidingModel->exportDensity();
  auto const densityBefore = slidingModel->getMassDensity(reference);
  auto const refractivityBefore = slidingModel->getRefractiveIndex(reference);
  environment.getUniverse()->addChild(std::move(homogeneous));
  environment.getUniverse()->addChild(std::move(flat));
  environment.getUniverse()->addChild(std::move(sliding));
  snapshot_type const snapshot(*environment.getUniverse(), root);
  require_native(snapshot, *environment.getUniverse());
  REQUIRE(snapshot.getModels().size() == 3);
  REQUIRE(snapshot.getModels()[0].densityKind_ == model_type::Density::Homogeneous);
  REQUIRE(snapshot.getModels()[1].densityKind_ == model_type::Density::FlatExponential);
  REQUIRE(snapshot.getModels()[2].densityKind_ ==
          model_type::Density::SlidingPlanarExponential);
  REQUIRE(snapshot.getModels()[2].refractivityKind_ ==
          model_type::Refractivity::GladstoneDale);
  REQUIRE(snapshot.getModels()[2].inverseReferenceDensityCm3PerG_ != 0.);
  REQUIRE(snapshot.getModels()[2].referenceHeightM_ == 7.);
  REQUIRE(slidingModel->getMassDensity(reference) == densityBefore);
  REQUIRE(slidingModel->getRefractiveIndex(reference) == refractivityBefore);
  auto const flatAfter = flatModel->exportDensity();
  auto const slidingAfter = slidingModel->exportDensity();
  REQUIRE(flatAfter.anchor_.getCoordinateSystem() ==
          flatBefore.anchor_.getCoordinateSystem());
  REQUIRE(flatAfter.axis_.getCoordinateSystem() ==
          flatBefore.axis_.getCoordinateSystem());
  REQUIRE(slidingAfter.anchor_.getCoordinateSystem() ==
          slidingBefore.anchor_.getCoordinateSystem());
  REQUIRE(flatModel->exportUniformField().getCoordinateSystem() == frame);
  for (int component = 0; component < 3; ++component) {
    REQUIRE(flatAfter.anchor_.getCoordinates()[component] ==
            flatBefore.anchor_.getCoordinates()[component]);
    REQUIRE(flatAfter.axis_.getComponents()[component] ==
            flatBefore.axis_.getComponents()[component]);
    REQUIRE(slidingAfter.anchor_.getCoordinates()[component] ==
            slidingBefore.anchor_.getCoordinates()[component]);
  }
}

TEST_CASE("Full five layer atmosphere snapshot retains every shape model and composition",
          "[kokkos_environment_snapshot]") {
  auto const root = get_root_CoordinateSystem();
  auto const frame = make_translation(root, {1_km, -2_km, 3_km});
  Point const center(frame, 0_m, 0_m, 0_m);
  Point const reference(frame, 0_m, 0_m, constants::EarthRadius::Mean);
  environment_type environment;
  create_5layer_atmosphere<model_interface_type, gladstone_extra_type>(
      environment, AtmosphereId::LinsleyUSStd, center, 1.000312, reference,
      Medium::AirDry1Atm, MagneticFieldVector(frame, {0_T, 50_uT, 0_T}));
  snapshot_type const snapshot(*environment.getUniverse(), root);
  require_native(snapshot, *environment.getUniverse());
  REQUIRE(snapshot.getTopology().getNodes().size() == 6);
  REQUIRE(snapshot.getModels().size() == 5);
  REQUIRE(snapshot.getComposition().size() == 5 * standardAirComposition.getSize());
  unsigned homogeneousCount{}, exponentialCount{};
  for (std::size_t index = 1; index < snapshot.getShapes().size(); ++index) {
    auto const expectedIndex = static_cast<std::int32_t>(index - 1);
    REQUIRE(snapshot.getShapes()[index].kind_ == shape_type::Kind::Sphere);
    REQUIRE(snapshot.getTopology().getNodes()[index].parentIndex_ == expectedIndex);
    auto const& model = snapshot.getModels()[index - 1];
    REQUIRE(snapshot.getTopology().getNodes()[index].modelIndex_ == expectedIndex);
    REQUIRE(model.refractivityKind_ == model_type::Refractivity::GladstoneDale);
    if (model.densityKind_ == model_type::Density::Homogeneous) {
      ++homogeneousCount;
    } else {
      REQUIRE(model.densityKind_ == model_type::Density::SlidingPlanarExponential);
      ++exponentialCount;
    }
  }
  REQUIRE(homogeneousCount == 1);
  REQUIRE(exponentialCount == 4);
}

TEST_CASE("Snapshot construction rejects invalid input without publishing partial arrays",
          "[kokkos_environment_snapshot]") {
  auto const frame = get_root_CoordinateSystem();
  environment_type valid;
  auto validChild = sphere(frame);
  set_uniform(*validChild, frame);
  valid.getUniverse()->addChild(std::move(validChild));
  auto published = std::make_unique<snapshot_type>(*valid.getUniverse(), frame);
  auto const* previous = published.get();
  auto const previousModel = published->getModels()[0];
  environment_type invalid;
  // Valid content preceding the failure must not escape as a partial snapshot.
  auto first = sphere(frame);
  set_uniform(*first, frame);
  invalid.getUniverse()->addChild(std::move(first));
  node_type const* exportRoot = invalid.getUniverse().get();
  CoordinateSystemPtr commonFrame = frame;
  node_type missingRootVolume;
  auto external = sphere(frame);
  SECTION("null common coordinate system") { commonFrame.reset(); }
  SECTION("root without volume") {
    REQUIRE_FALSE(missingRootVolume.hasVolume());
    exportRoot = &missingRootVolume;
  }
  SECTION("descendant without volume") {
    invalid.getUniverse()->addChild(std::make_unique<node_type>());
    REQUIRE_FALSE(invalid.getUniverse()->getChildNodes().back()->hasVolume());
  }
  SECTION("unsupported volume after valid content") {
    invalid.getUniverse()->addChild(environment_type::createNode<UnsupportedVolume>());
  }
  SECTION("nonfinite shape parameter") {
    invalid.getUniverse()->addChild(
        sphere(frame, std::numeric_limits<double>::infinity() * 1_m));
  }
  SECTION("model without exact density export") {
    auto unsupported = sphere(frame);
    unsupported->setModelProperties<uniform_extra_type<IndependentDensity>>(
        1.000312, Medium::AirDry1Atm, MagneticFieldVector(frame, {0_T, 0_T, 0_T}));
    invalid.getUniverse()->addChild(std::move(unsupported));
  }
  SECTION("external exclusion") { invalid.getUniverse()->excludeOverlapWith(external); }
  SECTION("null exclusion") {
    node_type::VTNUPtr nullNode;
    invalid.getUniverse()->excludeOverlapWith(nullNode);
  }
  SECTION("partial subtree") {
    exportRoot = invalid.getUniverse()->getChildNodes()[0].get();
  }
  REQUIRE_THROWS_AS(published = std::make_unique<snapshot_type>(*exportRoot, commonFrame),
                    std::invalid_argument);
  REQUIRE(published.get() == previous);
  require_model(published->getModels()[0], previousModel);
  require_native(*published, *valid.getUniverse());
  // A failed export leaves native ownership/model assignments intact and does
  // not poison a subsequent independent complete export.
  REQUIRE(invalid.getUniverse()->getChildNodes()[0]->getParent() ==
          invalid.getUniverse().get());
  REQUIRE(invalid.getUniverse()->getChildNodes()[0]->hasModelProperties());
  snapshot_type const recovered(*valid.getUniverse(), frame);
  require_native(recovered, *valid.getUniverse());
}
