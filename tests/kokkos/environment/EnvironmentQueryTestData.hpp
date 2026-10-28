/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include <corsika/kokkos/environment/EnvironmentView.hpp>
#include "EnvironmentQueryDataset.hpp"
#include "EnvironmentQueryReference.hpp"
#include "LeptonBoundaryReference.hpp"
#include "UniformMagneticFieldReference.hpp"

#include <corsika/kokkos/environment/EnvironmentQuery.hpp>
#include <corsika/kokkos/environment/ObservationPlane.hpp>
#include <corsika/kokkos/environment/EnvironmentLeptonBoundary.hpp>
#include <corsika/kokkos/environment/UniformMagneticField.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <utility>

namespace environment_query_test {

  namespace reference = environment_query_reference;

  struct Outcome {
    reference::NodeQuery layer_;
    reference::BoundaryQuery boundary_;
    reference::IntegralQuery grammage_, arclength_, competition_;
    reference::ObservationQuery observation_;
    std::int32_t modelIndex_{-1}, mediumId_{-1};
    std::uint32_t modelCount_{}, componentCount_{}, modelComponentCount_{};
    double fieldT_[3]{};
    double fractionSum_{};
    std::int32_t firstComponentCode_{};
    bool hasModel_{}, sharedModel_{};
  };

  struct IntegrationOutcome {
    Outcome query_;
    corsika::kokkos::MagneticStepLimitResult limit_;
    corsika::kokkos::LeptonBoundary curved_;
    reference::IntegralQuery traversed_, recovered_;
    corsika::kokkos::MagneticAdvanceResult advance_;
  };

  template <typename TEnvironmentView>
  KOKKOS_INLINE_FUNCTION Outcome actual(TEnvironmentView const& view,
                                        Query const& query) {
    using namespace corsika::kokkos;
    EnvironmentQueryConfig config{query.maximumDeflection_};
    Outcome result;
    auto const layer = query_environment(
        view, config, query.positionM_,
        query.includeDirection_ ? query.direction_ : nullptr, query.logicalNode_);
    result.layer_ = {static_cast<reference::Status>(layer.status_), layer.nodeIndex_,
                     layer.radiusM_, layer.densityGPerCm3_};
    auto const boundary = distance_to_environment_boundary(
        view, config, query.positionM_, query.direction_, query.logicalNode_);
    result.boundary_ = {static_cast<reference::Status>(boundary.status_),
                        boundary.nodeIndex_, boundary.distanceM_, boundary.radiusM_,
                        boundary.nextNode_};
    auto const integral =
        environment_grammage(view, config, query.integralNode_, query.positionM_,
                             query.direction_, query.lengthM_);
    result.grammage_ = {static_cast<reference::Status>(integral.status_),
                        integral.reserved_, integral.value_};
    auto const inverse =
        environment_arclength(view, config, query.integralNode_, query.positionM_,
                              query.direction_, query.grammageGPerCm2_);
    result.arclength_ = {static_cast<reference::Status>(inverse.status_),
                         inverse.reserved_, inverse.value_};
    auto const competition =
        geometry_competition_grammage(view, config, query.integralNode_, query.positionM_,
                                      query.direction_, query.lengthM_);
    result.competition_ = {static_cast<reference::Status>(competition.status_),
                           competition.reserved_, competition.value_};
    ObservationPlane plane;
    for (int axis = 0; axis < 3; ++axis) {
      plane.pointM_[axis] = query.pointM_[axis];
      plane.normal_[axis] = query.normal_[axis];
    }
    auto const observation = intersect_observation_plane_straight(
        plane, query.positionM_, query.direction_, query.maximumObservationM_);
    result.observation_ = {static_cast<reference::ObservationStatus>(observation.status_),
                           observation.reserved_, observation.distanceM_,
                           observation.residualM_};
    result.modelCount_ = view.getModelCount();
    result.componentCount_ = view.getComponentCount();
    auto const model = view.getModel(query.integralNode_);
    result.hasModel_ = model != nullptr;
    result.sharedModel_ =
        view.getModel(1) != nullptr && view.getModel(1) == view.getModel(4);
    if (model != nullptr) {
      result.modelIndex_ = view.getTree().getModelIndex(query.integralNode_);
      result.mediumId_ = model->mediumId_;
      result.modelComponentCount_ = model->compositionCount_;
      for (int axis = 0; axis < 3; ++axis) {
        result.fieldT_[axis] = model->magneticFieldT_[axis];
      }
      for (std::uint32_t index = 0; index < model->compositionCount_; ++index) {
        auto const& component = view.getComponent(model->compositionBegin_ + index);
        result.fractionSum_ += component.fraction_;
        if (index == 0) { result.firstComponentCode_ = component.code_; }
      }
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION Outcome expected(reference::FullView view, Query const& query) {
    view.maximumDeflection_ = query.maximumDeflection_;
    Outcome result;
    result.layer_ = reference::query(view, query.positionM_,
                                     query.includeDirection_ ? query.direction_ : nullptr,
                                     query.logicalNode_);
    result.boundary_ =
        reference::boundary(view, query.positionM_, query.direction_, query.logicalNode_);
    result.grammage_ = reference::grammage(view, query.integralNode_, query.positionM_,
                                           query.direction_, query.lengthM_);
    result.arclength_ = reference::arclength(view, query.integralNode_, query.positionM_,
                                             query.direction_, query.grammageGPerCm2_);
    result.competition_ = reference::competition(
        view, query.integralNode_, query.positionM_, query.direction_, query.lengthM_);
    result.observation_ =
        reference::observation(query.pointM_, query.normal_, query.positionM_,
                               query.direction_, query.maximumObservationM_);
    result.modelCount_ = view.modelCount_;
    result.componentCount_ = view.compositionCount_;
    auto const model = view.model(query.integralNode_);
    result.hasModel_ = model != nullptr;
    result.sharedModel_ = view.model(1) != nullptr && view.model(1) == view.model(4);
    if (model != nullptr) {
      result.modelIndex_ = view.tree_.nodes_[query.integralNode_].modelIndex_;
      result.mediumId_ = model->mediumId_;
      result.modelComponentCount_ = model->compositionCount_;
      for (int axis = 0; axis < 3; ++axis) {
        result.fieldT_[axis] = model->magneticFieldT_[axis];
      }
      for (std::uint32_t index = 0; index < model->compositionCount_; ++index) {
        auto const component = view.composition_[model->compositionBegin_ + index];
        result.fractionSum_ += component.fraction_;
        if (index == 0) { result.firstComponentCode_ = component.code_; }
      }
    }
    return result;
  }

  template <typename TEnvironmentView>
  KOKKOS_INLINE_FUNCTION IntegrationOutcome actual_integration(
      TEnvironmentView const& view, IntegrationQuery const& input, Query const& query) {
    using namespace corsika::kokkos;
    IntegrationOutcome result;
    result.query_ = actual(view, query);
    auto particle = input.particle_;
    particle.environmentNode_ = result.query_.layer_.node_;
    auto const model = view.getModel(particle.environmentNode_);
    if (model == nullptr) { return result; }
    result.limit_ = maximum_uniform_magnetic_step(
        particle, input.massGeV_, input.chargeNumber_, model->magneticFieldT_);
    result.curved_ =
        lepton_boundary(view.getTree(), particle, input.massGeV_, input.chargeNumber_,
                        model->magneticFieldT_, result.limit_);
    result.traversed_ = result.query_.grammage_;
    auto const inverse = environment_arclength(
        view, EnvironmentQueryConfig{}, particle.environmentNode_, particle.positionM_,
        particle.direction_, result.traversed_.value_);
    result.recovered_ = {static_cast<reference::Status>(inverse.status_),
                         inverse.reserved_, inverse.value_};
    double field[3]{};
    if (result.limit_.status_ != MagneticStepStatus::Linear) {
      for (int axis = 0; axis < 3; ++axis) { field[axis] = model->magneticFieldT_[axis]; }
    }
    result.advance_ = advance_uniform_magnetic_field(
        particle, input.massGeV_, input.chargeNumber_, field, result.recovered_.value_);
    return result;
  }

  KOKKOS_INLINE_FUNCTION IntegrationOutcome
  expected_integration(reference::FullView const& view, IntegrationQuery const& input,
                       Query const& query) {
    IntegrationOutcome result;
    result.query_ = expected(view, query);
    auto particle = input.particle_;
    particle.environmentNode_ = result.query_.layer_.node_;
    auto const model = view.model(particle.environmentNode_);
    if (model == nullptr) { return result; }
    auto const limit = uniform_magnetic_reference::maximumUniformMagneticStep(
        particle, input.massGeV_, input.chargeNumber_, model->magneticFieldT_);
    result.limit_ = {static_cast<corsika::kokkos::MagneticStepStatus>(limit.status),
                     limit.reserved, limit.distance_m, limit.gyroradius_m};
    straight_boundary_reference::Tree tree;
    tree.nodes_ = view.tree_.nodes_;
    tree.shapes_ = view.tree_.shapes_;
    tree.children_ = view.tree_.children_;
    tree.exclusions_ = view.tree_.exclusions_;
    auto const curved = lepton_boundary_reference::lepton_boundary(
        tree, particle, input.massGeV_, input.chargeNumber_, model->magneticFieldT_,
        result.limit_);
    result.curved_ = {curved.distanceM_, curved.nextNode_, curved.failed_};
    result.traversed_ = result.query_.grammage_;
    result.recovered_ =
        reference::arclength(view, particle.environmentNode_, particle.positionM_,
                             particle.direction_, result.traversed_.value_);
    double field[3]{};
    if (limit.status != uniform_magnetic_reference::MagneticStepStatus::Linear) {
      for (int axis = 0; axis < 3; ++axis) { field[axis] = model->magneticFieldT_[axis]; }
    }
    auto const advance = uniform_magnetic_reference::advanceUniformMagneticField(
        particle, input.massGeV_, input.chargeNumber_, field, result.recovered_.value_);
    result.advance_ = {static_cast<corsika::kokkos::MagneticStepStatus>(advance.status),
                       advance.reserved, advance.particle, advance.bend_parameter,
                       advance.chord_length_m};
    return result;
  }

  template <typename TNumber, typename TState>
  void compare(Outcome const& actual, Outcome const& expected, TNumber number,
               TState state) {
    state(actual.layer_.status_, expected.layer_.status_);
    state(actual.layer_.node_, expected.layer_.node_);
    number(actual.layer_.radiusM_, expected.layer_.radiusM_);
    number(actual.layer_.densityGPerCm3_, expected.layer_.densityGPerCm3_);
    state(actual.boundary_.status_, expected.boundary_.status_);
    state(actual.boundary_.node_, expected.boundary_.node_);
    state(actual.boundary_.nextNode_, expected.boundary_.nextNode_);
    number(actual.boundary_.distanceM_, expected.boundary_.distanceM_);
    number(actual.boundary_.radiusM_, expected.boundary_.radiusM_);
    for (auto pair : {std::pair{actual.grammage_, expected.grammage_},
                      std::pair{actual.arclength_, expected.arclength_},
                      std::pair{actual.competition_, expected.competition_}}) {
      state(pair.first.status_, pair.second.status_);
      state(pair.first.reserved_, pair.second.reserved_);
      number(pair.first.value_, pair.second.value_);
    }
    state(actual.observation_.status_, expected.observation_.status_);
    state(actual.observation_.reserved_, expected.observation_.reserved_);
    number(actual.observation_.distanceM_, expected.observation_.distanceM_);
    number(actual.observation_.residualM_, expected.observation_.residualM_);
    state(actual.modelIndex_, expected.modelIndex_);
    state(actual.mediumId_, expected.mediumId_);
    state(actual.modelCount_, expected.modelCount_);
    state(actual.componentCount_, expected.componentCount_);
    state(actual.modelComponentCount_, expected.modelComponentCount_);
    state(actual.firstComponentCode_, expected.firstComponentCode_);
    state(actual.hasModel_, expected.hasModel_);
    state(actual.sharedModel_, expected.sharedModel_);
    number(actual.fractionSum_, expected.fractionSum_);
    for (int axis = 0; axis < 3; ++axis) {
      number(actual.fieldT_[axis], expected.fieldT_[axis]);
    }
  }

  template <typename TNumber, typename TState>
  void compare_particle(particle_type const& actual, particle_type const& expected,
                        TNumber number, TState state) {
    state(actual.pid_, expected.pid_);
    state(actual.mediumId_, expected.mediumId_);
    state(actual.generation_, expected.generation_);
    state(actual.reserved_, expected.reserved_);
    state(actual.historyId_, expected.historyId_);
    state(actual.parentHistoryId_, expected.parentHistoryId_);
    state(actual.stepId_, expected.stepId_);
    state(actual.environmentNode_, expected.environmentNode_);
    number(actual.energyGeV_, expected.energyGeV_);
    number(actual.timeS_, expected.timeS_);
    number(actual.weight_, expected.weight_);
    for (int axis = 0; axis < 3; ++axis) {
      number(actual.positionM_[axis], expected.positionM_[axis]);
      number(actual.direction_[axis], expected.direction_[axis]);
    }
  }

  template <typename TNumber, typename TState>
  void compare(IntegrationOutcome const& actual, IntegrationOutcome const& expected,
               TNumber number, TState state) {
    compare(actual.query_, expected.query_, number, state);
    state(actual.limit_.status_, expected.limit_.status_);
    state(actual.limit_.reserved_, expected.limit_.reserved_);
    number(actual.limit_.distanceM_, expected.limit_.distanceM_);
    number(actual.limit_.gyroradiusM_, expected.limit_.gyroradiusM_);
    state(actual.curved_.nextNode_, expected.curved_.nextNode_);
    state(actual.curved_.failed_, expected.curved_.failed_);
    number(actual.curved_.distanceM_, expected.curved_.distanceM_);
    state(actual.traversed_.status_, expected.traversed_.status_);
    state(actual.traversed_.reserved_, expected.traversed_.reserved_);
    number(actual.traversed_.value_, expected.traversed_.value_);
    state(actual.recovered_.status_, expected.recovered_.status_);
    state(actual.recovered_.reserved_, expected.recovered_.reserved_);
    number(actual.recovered_.value_, expected.recovered_.value_);
    state(actual.advance_.status_, expected.advance_.status_);
    state(actual.advance_.reserved_, expected.advance_.reserved_);
    number(actual.advance_.bendParameter_, expected.advance_.bendParameter_);
    number(actual.advance_.chordLengthM_, expected.advance_.chordLengthM_);
    compare_particle(actual.advance_.particle_, expected.advance_.particle_, number,
                     state);
  }

  inline reference::FullView reference_view(Dataset const& data) {
    reference::FullView result;
    result.tree_ = {data.nodes_.data(), data.shapes_.data(), data.children_.data(),
                    data.exclusions_.data(),
                    static_cast<unsigned int>(data.nodes_.size())};
    result.models_ = data.models_.data();
    result.composition_ = data.components_.data();
    result.modelCount_ = data.models_.size();
    result.compositionCount_ = data.components_.size();
    result.childCount_ = data.children_.size();
    result.exclusionCount_ = data.exclusions_.size();
    return result;
  }

  template <typename TMemorySpace>
  struct Uploaded {
    Kokkos::View<node_type*, TMemorySpace> nodes_;
    Kokkos::View<shape_type*, TMemorySpace> shapes_;
    Kokkos::View<std::int32_t*, TMemorySpace> children_, exclusions_;
    Kokkos::View<model_type*, TMemorySpace> models_;
    Kokkos::View<component_type*, TMemorySpace> components_;

    template <typename TValue>
    static Kokkos::View<TValue*, TMemorySpace> upload(std::vector<TValue> const& source) {
      Kokkos::View<TValue*, Kokkos::HostSpace> host("environment_host_input",
                                                    source.size());
      for (std::size_t index = 0; index < source.size(); ++index) {
        host(index) = source[index];
      }
      Kokkos::View<TValue*, TMemorySpace> result("environment_query_input",
                                                 source.size());
      Kokkos::deep_copy(result, host);
      return result;
    }

    explicit Uploaded(Dataset const& source)
        : nodes_(upload(source.nodes_))
        , shapes_(upload(source.shapes_))
        , children_(upload(source.children_))
        , exclusions_(upload(source.exclusions_))
        , models_(upload(source.models_))
        , components_(upload(source.components_)) {}

    auto view() const {
      corsika::kokkos::EnvironmentTreeView<TMemorySpace> tree(nodes_, shapes_, children_,
                                                              exclusions_);
      return corsika::kokkos::EnvironmentView<TMemorySpace>(tree, models_, components_);
    }

    reference::FullView referenceView() const {
      reference::FullView result;
      result.tree_ = {nodes_.data(), shapes_.data(), children_.data(), exclusions_.data(),
                      static_cast<unsigned int>(nodes_.extent(0))};
      result.models_ = models_.data();
      result.composition_ = components_.data();
      result.modelCount_ = models_.extent(0);
      result.compositionCount_ = components_.extent(0);
      result.childCount_ = children_.extent(0);
      result.exclusionCount_ = exclusions_.extent(0);
      return result;
    }
  };

  inline Dataset simple(std::string name, model_type::Density densityKind =
                                              model_type::Density::Homogeneous) {
    Dataset result;
    result.name_ = name;
    result.nodes_.resize(2);
    result.shapes_.resize(2);
    result.nodes_[0].childCount_ = 1;
    result.children_ = {1};
    result.nodes_[1].parentIndex_ = 0;
    result.nodes_[1].modelIndex_ = 0;
    result.shapes_[0].kind_ = shape_type::Kind::Universe;
    result.shapes_[1].kind_ = shape_type::Kind::Sphere;
    result.shapes_[1].radiusM_ = 1.e6;
    model_type model;
    model.densityKind_ = densityKind;
    model.referenceDensityGPerCm3_ = .01;
    model.scaleLengthM_ = 100.;
    model.inverseScaleLengthPerM_ = .01;
    model.referenceHeightM_ = 10.;
    model.axis_[2] = 1.;
    model.magneticFieldT_[0] = 2.e-5;
    model.magneticFieldT_[1] = -3.e-5;
    model.magneticFieldT_[2] = 4.e-5;
    model.mediumId_ = 7;
    model.compositionCount_ = 3;
    result.models_ = {model};
    result.components_ = {{1000080160, .2}, {1000070140, .7}, {1000080160, .1}};
    return result;
  }

  inline std::vector<Dataset> datasets() {
    std::vector<Dataset> result;
    double const infinity = std::numeric_limits<double>::infinity();
    double const nan = std::numeric_limits<double>::quiet_NaN();
    for (auto kind :
         {model_type::Density::Homogeneous, model_type::Density::FlatExponential,
          model_type::Density::SlidingPlanarExponential}) {
      auto data =
          simple("density-" + std::to_string(static_cast<unsigned int>(kind)), kind);
      Query query;
      data.queries_.push_back(query);
      for (double value : {0., -0., -1., infinity, -infinity, nan}) {
        query = Query{};
        query.lengthM_ = value;
        query.grammageGPerCm2_ = value;
        query.category_ =
            value == 0. ? Category::Zero
                        : (value < 0. ? Category::Negative : Category::NonFiniteValue);
        query.crossDevice_ = false;
        data.queries_.push_back(query);
      }
      for (double value : {0., 2., infinity, nan}) {
        query = Query{};
        query.direction_[2] = value;
        query.category_ = Category::InvalidDirection;
        query.crossDevice_ = false;
        data.queries_.push_back(query);
        query.includeDirection_ = false;
        data.queries_.push_back(query);
      }
      for (double value : {nan, infinity, -infinity}) {
        query = Query{};
        query.positionM_[0] = value;
        query.category_ = Category::NonFinitePosition;
        query.crossDevice_ = false;
        data.queries_.push_back(query);
        query.lengthM_ = 0.;
        query.grammageGPerCm2_ = 0.;
        data.queries_.push_back(query);
      }
      for (double angle : {0., -.2, .5 * 3.141592653589793, infinity, nan}) {
        query = Query{};
        query.maximumDeflection_ = angle;
        query.category_ = Category::InvalidConfig;
        query.positionM_[0] = nan;
        query.crossDevice_ = false;
        data.queries_.push_back(query);
      }
      for (double projection : {0., -1.e-16, 1.e-16, -1.}) {
        query = Query{};
        query.direction_[0] = std::sqrt(1. - projection * projection);
        query.direction_[2] = projection;
        query.crossDevice_ = false;
        data.queries_.push_back(query);
      }
      result.push_back(data);
    }
    for (double density : {0., -0., -1., infinity, -infinity, nan}) {
      auto data = simple("invalid-density");
      data.models_[0].referenceDensityGPerCm3_ = density;
      Query query;
      query.category_ = Category::InvalidDensity;
      query.crossDevice_ = false;
      data.queries_.push_back(query);
      query.lengthM_ = 0.;
      query.grammageGPerCm2_ = -0.;
      data.queries_.push_back(query);
      result.push_back(data);
    }
    for (auto kind : {model_type::Density::FlatExponential,
                      model_type::Density::SlidingPlanarExponential}) {
      for (double scale : {1., -1.}) {
        auto data = simple("growing-overflow", kind);
        data.models_[0].scaleLengthM_ = scale;
        data.models_[0].inverseScaleLengthPerM_ = 1. / scale;
        Query query;
        query.lengthM_ = 1000.;
        query.direction_[2] = scale;
        query.category_ = Category::OverflowSentinel;
        query.crossDevice_ = false;
        data.queries_ = {query};
        result.push_back(data);
      }
    }
    for (int variant = 0; variant < 3; ++variant) {
      auto data = simple("rejected-overflow", variant == 2
                                                  ? model_type::Density::Homogeneous
                                                  : model_type::Density::FlatExponential);
      Query query;
      query.category_ = Category::RejectedSentinel;
      query.crossDevice_ = false;
      if (variant == 0) {
        data.models_[0].scaleLengthM_ = infinity;
        data.models_[0].inverseScaleLengthPerM_ = 1.;
        query.lengthM_ = 1000.;
      } else if (variant == 1) {
        data.models_[0].scaleLengthM_ = 1.e-200;
        data.models_[0].inverseScaleLengthPerM_ = 1.e-200;
        query.lengthM_ = 1.e308;
      } else {
        data.models_[0].referenceDensityGPerCm3_ = 1.e308;
        query.lengthM_ = 10.;
      }
      data.queries_ = {query};
      result.push_back(data);
    }
    auto mapped = simple("node-model-mapping");
    mapped.nodes_.resize(7);
    mapped.shapes_.resize(7);
    mapped.nodes_[0].childOffset_ = 0;
    mapped.nodes_[0].childCount_ = 1;
    mapped.nodes_[1].childOffset_ = 1;
    mapped.nodes_[1].childCount_ = 4;
    mapped.nodes_[2].parentIndex_ = 1;
    mapped.nodes_[2].childOffset_ = 5;
    mapped.nodes_[2].childCount_ = 1;
    mapped.nodes_[3].parentIndex_ = 2;
    mapped.children_ = {1, 2, 4, 5, 6, 3};
    mapped.shapes_[1].radiusM_ = 100.;
    mapped.models_.push_back(mapped.models_[0]);
    mapped.models_[1].referenceDensityGPerCm3_ = .02;
    mapped.models_.push_back(mapped.models_[0]);
    mapped.nodes_[2].modelIndex_ = 1;
    for (int index = 2; index < 7; ++index) {
      mapped.shapes_[index].kind_ = shape_type::Kind::Sphere;
      mapped.shapes_[index].radiusM_ = index == 3 ? 1. : 10.;
    }
    for (int index = 4; index < 7; ++index) {
      mapped.nodes_[index].parentIndex_ = 1;
      mapped.shapes_[index].centerM_[0] = index == 4 ? 30. : (index == 5 ? -30. : 60.);
    }
    mapped.nodes_[4].modelIndex_ = 0;
    mapped.nodes_[5].modelIndex_ = 2;
    mapped.nodes_[6].modelIndex_ = 99;
    for (int node : {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}) {
      Query query;
      query.positionM_[2] = 5.;
      query.logicalNode_ = node;
      query.integralNode_ = node == -3 ? 2 : node;
      query.category_ = Category::Assigned;
      mapped.queries_.push_back(query);
    }
    Query noModel;
    noModel.positionM_[2] = 0.;
    noModel.logicalNode_ = -3;
    noModel.integralNode_ = 3;
    noModel.category_ = Category::MissingModel;
    mapped.queries_.push_back(noModel);
    Query outside;
    outside.positionM_[0] = 200.;
    outside.logicalNode_ = 1;
    outside.category_ = Category::Assigned;
    mapped.queries_.push_back(outside);
    result.push_back(mapped);

    auto tie = simple("ordered-tie");
    tie.nodes_.resize(4);
    tie.shapes_.resize(4);
    tie.children_ = {1, 2};
    tie.exclusions_ = {3};
    tie.nodes_[1].childOffset_ = 1;
    tie.nodes_[1].childCount_ = 1;
    tie.nodes_[1].excludedCount_ = 1;
    tie.nodes_[2].parentIndex_ = 1;
    tie.nodes_[2].modelIndex_ = 0;
    tie.nodes_[3].parentIndex_ = 0;
    tie.nodes_[3].modelIndex_ = 0;
    for (int index : {2, 3}) {
      tie.shapes_[index].kind_ = shape_type::Kind::SeparationPlane;
      tie.shapes_[index].centerM_[0] = 2.;
      tie.shapes_[index].normal_[0] = -1.;
    }
    Query tied;
    tied.direction_[0] = 1.;
    tied.direction_[2] = 0.;
    tied.category_ = Category::Tie;
    tie.queries_ = {tied};
    result.push_back(tie);

    auto observations = simple("observation");
    Query base;
    base.positionM_[2] = 0.;
    base.pointM_[2] = 3.;
    base.category_ = Category::Observation;
    for (double limit : {infinity, 3., std::nextafter(3., 0.), 0., -1., -infinity, nan}) {
      auto query = base;
      query.maximumObservationM_ = limit;
      query.crossDevice_ = false;
      observations.queries_.push_back(query);
    }
    for (double direction : {0., -1., 2., std::numeric_limits<double>::denorm_min()}) {
      auto query = base;
      query.direction_[2] = direction;
      query.crossDevice_ = false;
      observations.queries_.push_back(query);
    }
    for (double normal : {0., 2., nan, infinity}) {
      auto query = base;
      query.normal_[2] = normal;
      query.crossDevice_ = false;
      observations.queries_.push_back(query);
    }
    for (double point : {0., -1., nan, infinity}) {
      auto query = base;
      query.pointM_[2] = point;
      query.crossDevice_ = false;
      observations.queries_.push_back(query);
    }
    auto overflow = base;
    overflow.positionM_[2] = -1.e308;
    overflow.pointM_[2] = 1.e308;
    overflow.crossDevice_ = false;
    observations.queries_.push_back(overflow);
    result.push_back(observations);
    auto empty = Dataset{};
    empty.name_ = "empty";
    Query invalid;
    invalid.category_ = Category::Empty;
    empty.queries_ = {invalid};
    result.push_back(empty);
    auto noModels = simple("no-model-storage");
    noModels.models_.clear();
    noModels.queries_ = {invalid};
    result.push_back(noModels);
    return result;
  }

  template <typename TNumber, typename TState>
  void compare(Query const& actual, Query const& expected, TNumber number, TState state) {
    state(actual.logicalNode_, expected.logicalNode_);
    state(actual.integralNode_, expected.integralNode_);
    state(actual.includeDirection_, expected.includeDirection_);
    state(actual.category_, expected.category_);
    state(actual.crossDevice_, expected.crossDevice_);
    number(actual.lengthM_, expected.lengthM_);
    number(actual.grammageGPerCm2_, expected.grammageGPerCm2_);
    number(actual.maximumDeflection_, expected.maximumDeflection_);
    number(actual.maximumObservationM_, expected.maximumObservationM_);
    for (int axis = 0; axis < 3; ++axis) {
      number(actual.positionM_[axis], expected.positionM_[axis]);
      number(actual.direction_[axis], expected.direction_[axis]);
      number(actual.pointM_[axis], expected.pointM_[axis]);
      number(actual.normal_[axis], expected.normal_[axis]);
    }
  }

  template <typename TNumber, typename TState>
  void compare(IntegrationQuery const& actual, IntegrationQuery const& expected,
               TNumber number, TState state) {
    compare_particle(actual.particle_, expected.particle_, number, state);
    number(actual.massGeV_, expected.massGeV_);
    number(actual.chargeNumber_, expected.chargeNumber_);
    number(actual.lengthM_, expected.lengthM_);
    state(actual.nativeNode_, expected.nativeNode_);
    number(actual.nativeDensityGPerCm3_, expected.nativeDensityGPerCm3_);
    for (int axis = 0; axis < 3; ++axis) {
      number(actual.nativeFieldT_[axis], expected.nativeFieldT_[axis]);
    }
  }

  template <typename TRequire>
  void check_semantics(Query const& query, Outcome const& result, TRequire require) {
    using Status = reference::Status;
    using PlaneStatus = reference::ObservationStatus;
    if (query.category_ == Category::OverflowSentinel) {
      require(result.grammage_.status_ == Status::GrammageOutOfRange);
      require(result.grammage_.value_ == std::numeric_limits<double>::infinity());
      require(result.competition_.status_ == Status::Success);
      require(result.competition_.value_ == std::numeric_limits<double>::infinity());
    }
    if (query.category_ == Category::RejectedSentinel) {
      require(result.grammage_.status_ == Status::GrammageOutOfRange);
      require(result.grammage_.value_ == std::numeric_limits<double>::infinity());
      require(result.competition_.status_ == Status::GrammageOutOfRange);
    }
    if (query.category_ == Category::Tie) {
      require(result.boundary_.status_ == Status::Success);
      require(result.boundary_.distanceM_ == 2.);
      require(result.boundary_.nextNode_ == 3);
    }
    if (query.category_ == Category::MissingModel) {
      require(result.layer_.status_ == Status::OutsideEnvironment);
      require(result.layer_.node_ == 3);
      require(!result.hasModel_);
    }
    if (query.category_ == Category::Assigned && query.logicalNode_ != -3) {
      require(result.layer_.node_ == query.logicalNode_);
      if (query.logicalNode_ == 1 || query.logicalNode_ == 4) {
        require(result.layer_.status_ == Status::Success);
        require(result.modelIndex_ == 0);
        require(result.mediumId_ == 7);
        require(result.sharedModel_);
      }
      if (query.logicalNode_ == 5) { require(result.modelIndex_ == 2); }
      if (query.logicalNode_ == 3 || query.logicalNode_ == 6) {
        require(!result.hasModel_);
      }
    }
    if (query.category_ == Category::InvalidConfig) {
      require(result.layer_.status_ == Status::InvalidSnapshot);
      require(result.grammage_.status_ == Status::NonFiniteInput);
    }
    if (query.category_ == Category::Observation && query.direction_[2] == 2. &&
        query.positionM_[2] == 0. && query.pointM_[2] == 3.) {
      require(result.layer_.status_ == Status::NonFiniteInput);
      require(result.observation_.status_ == PlaneStatus::Success);
      require(result.observation_.distanceM_ == 1.5);
    }
    if (query.category_ == Category::Observation && query.maximumObservationM_ == -1.) {
      require(result.observation_.status_ == PlaneStatus::NoForwardIntersection);
    }
    if (query.category_ == Category::Empty) {
      require(result.layer_.status_ == Status::InvalidSnapshot);
      require(result.grammage_.status_ == Status::NonFiniteInput);
      require(result.observation_.status_ == PlaneStatus::Success);
    }
  }

  template <typename TRequire>
  void check_integration(IntegrationQuery const& input, IntegrationOutcome const& result,
                         TRequire require) {
    using Status = reference::Status;
    using MagneticStatus = corsika::kokkos::MagneticStepStatus;
    require(result.query_.layer_.status_ == Status::Success);
    require(result.query_.layer_.node_ == input.nativeNode_);
    require(result.query_.modelIndex_ == input.nativeNode_ - 1);
    require(
        std::abs(result.query_.layer_.densityGPerCm3_ - input.nativeDensityGPerCm3_) <=
        2.e-12 * input.nativeDensityGPerCm3_);
    for (int axis = 0; axis < 3; ++axis) {
      require(result.query_.fieldT_[axis] == input.nativeFieldT_[axis]);
    }
    require(
        result.limit_.status_ ==
        (input.chargeNumber_ == 0. ? MagneticStatus::Linear : MagneticStatus::Success));
    require(result.query_.boundary_.status_ == Status::Success);
    require(!result.curved_.failed_);
    require(input.lengthM_ <= result.limit_.distanceM_);
    require(input.lengthM_ <= result.query_.boundary_.distanceM_);
    require(input.lengthM_ <= result.curved_.distanceM_);
    require(result.traversed_.status_ == Status::Success);
    require(std::isfinite(result.traversed_.value_) && result.traversed_.value_ > 0.);
    require(result.recovered_.status_ == Status::Success);
    require(std::isfinite(result.recovered_.value_));
    require(std::abs(result.recovered_.value_ - input.lengthM_) <=
            2.e-12 * input.lengthM_);
    require(result.advance_.status_ == MagneticStatus::Success ||
            result.advance_.status_ == MagneticStatus::Linear);
    require(result.advance_.chordLengthM_ > 0.);
    require(result.advance_.particle_.timeS_ > input.particle_.timeS_);
    require(result.advance_.particle_.environmentNode_ == input.nativeNode_);
    require(result.advance_.particle_.mediumId_ == input.particle_.mediumId_);
    require(result.advance_.particle_.pid_ == input.particle_.pid_);
    require(result.advance_.particle_.energyGeV_ == input.particle_.energyGeV_);
    require(result.advance_.particle_.weight_ == input.particle_.weight_);
    require(result.advance_.particle_.historyId_ == input.particle_.historyId_);
    require(result.advance_.particle_.parentHistoryId_ ==
            input.particle_.parentHistoryId_);
    require(result.advance_.particle_.generation_ == input.particle_.generation_);
    require(result.advance_.particle_.reserved_ == input.particle_.reserved_);
    require(result.advance_.particle_.stepId_ == input.particle_.stepId_);
  }

  template <typename TMemorySpace, typename TNumber, typename TState>
  void check_storage(Dataset const& data, Uploaded<TMemorySpace> const& upload,
                     TNumber number, TState state) {
    auto const nodes =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, upload.nodes_);
    auto const shapes =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, upload.shapes_);
    auto const children =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, upload.children_);
    auto const exclusions =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, upload.exclusions_);
    auto const models =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, upload.models_);
    auto const components =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, upload.components_);
    for (std::size_t index = 0; index < data.nodes_.size(); ++index) {
      auto const& a = nodes(index);
      auto const& b = data.nodes_[index];
      state(a.parentIndex_, b.parentIndex_);
      state(a.modelIndex_, b.modelIndex_);
      state(a.childOffset_, b.childOffset_);
      state(a.childCount_, b.childCount_);
      state(a.excludedOffset_, b.excludedOffset_);
      state(a.excludedCount_, b.excludedCount_);
      auto const& x = shapes(index);
      auto const& y = data.shapes_[index];
      state(x.kind_, y.kind_);
      number(x.radiusM_, y.radiusM_);
      for (int axis = 0; axis < 3; ++axis) {
        number(x.centerM_[axis], y.centerM_[axis]);
        number(x.halfExtentM_[axis], y.halfExtentM_[axis]);
        number(x.normal_[axis], y.normal_[axis]);
        for (int component = 0; component < 3; ++component) {
          number(x.axes_[axis][component], y.axes_[axis][component]);
        }
      }
    }
    for (std::size_t index = 0; index < data.children_.size(); ++index) {
      state(children(index), data.children_[index]);
    }
    for (std::size_t index = 0; index < data.exclusions_.size(); ++index) {
      state(exclusions(index), data.exclusions_[index]);
    }
    for (std::size_t index = 0; index < data.models_.size(); ++index) {
      auto const& a = models(index);
      auto const& b = data.models_[index];
      state(a.densityKind_, b.densityKind_);
      state(a.refractivityKind_, b.refractivityKind_);
      state(a.mediumId_, b.mediumId_);
      state(a.compositionBegin_, b.compositionBegin_);
      state(a.compositionCount_, b.compositionCount_);
      number(a.referenceDensityGPerCm3_, b.referenceDensityGPerCm3_);
      number(a.scaleLengthM_, b.scaleLengthM_);
      number(a.inverseScaleLengthPerM_, b.inverseScaleLengthPerM_);
      number(a.referenceHeightM_, b.referenceHeightM_);
      number(a.refractivityValue_, b.refractivityValue_);
      number(a.inverseReferenceDensityCm3PerG_, b.inverseReferenceDensityCm3PerG_);
      for (int axis = 0; axis < 3; ++axis) {
        number(a.anchorM_[axis], b.anchorM_[axis]);
        number(a.axis_[axis], b.axis_[axis]);
        number(a.magneticFieldT_[axis], b.magneticFieldT_[axis]);
      }
    }
    for (std::size_t index = 0; index < data.components_.size(); ++index) {
      state(components(index).code_, data.components_[index].code_);
      number(components(index).fraction_, data.components_[index].fraction_);
    }
  }

} // namespace environment_query_test
