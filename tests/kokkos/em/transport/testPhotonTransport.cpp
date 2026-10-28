/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "TransportChecks.hpp"
#include "PhotonTransportReference.hpp"
#include "../continuous/ContinuousTestData.hpp"
#include "../../environment/EnvironmentQueryTestData.hpp"
#include <corsika/kokkos/em/algorithms/PhotonTransportStep.hpp>
#include <array>
#include <map>

namespace photon_transport_test {
  namespace current = corsika::kokkos;
  namespace reference = photon_transport_reference;
  namespace environment = environment_query_test;
  using transport_test::append;
  using transport_test::check;
  using transport_test::execution_space_type;
  using transport_test::flatten;
  using transport_test::memory_space_type;
  using transport_test::Outcome;

  struct Input {
    current::InteractionRecord interaction_{};
    current::TransportGeometryConfig config_{};
    current::ExternalTransportBoundary external_{};
  };
  struct Result {
    Outcome record_, fallback_;
    std::uint32_t failed_{};
  };
  template <typename T>
  KOKKOS_INLINE_FUNCTION Result capture(T const& outcome) {
    Result result;
    if constexpr (std::is_same_v<T, current::PhotonTransportOutcome>) {
      result.record_ = flatten(outcome.record_);
      append(result.fallback_, outcome.fallback_);
      result.failed_ = outcome.fallbackFlag_;
    } else {
      result.record_ = flatten(outcome.record);
      append(result.fallback_, outcome.fallback);
      result.failed_ = outcome.fallback_flag;
    }
    return result;
  }
  KOKKOS_INLINE_FUNCTION reference::PhotonTransportOutcome expected(
      environment_query_reference::FullView view, Input const& input) {
    reference::EnvironmentSnapshot snapshot;
    snapshot.node_tree = view;
    snapshot.node_tree.maximumDeflection_ =
        input.config_.query_.maximumMagneticDeflectionRad_;
    for (int axis = 0; axis < 3; ++axis) {
      snapshot.earth_center_m[axis] = input.config_.earthCenterM_[axis];
      snapshot.point[axis] = input.config_.observation_.pointM_[axis];
      snapshot.normal[axis] = input.config_.observation_.normal_[axis];
    }
    reference::ExternalTransportBoundary external{input.external_.distanceM_,
                                                  input.external_.enabled_,
                                                  input.external_.disableObservation_};
    return reference::transportPhoton(
        snapshot, continuous_test::convert(input.interaction_), external);
  }
  void compare(Result const& actual, Result const& expected) {
    REQUIRE(actual.failed_ == expected.failed_);
    check(actual.record_, expected.record_);
    check(actual.fallback_, expected.fallback_);
  }

  Input input(environment::Query const& query) {
    Input result;
    result.interaction_ = continuous_test::interaction(22, 10000.);
    auto& particle = result.interaction_.particle_;
    for (int axis = 0; axis < 3; ++axis) {
      particle.positionM_[axis] = query.positionM_[axis];
      particle.direction_[axis] = query.direction_[axis];
      result.config_.observation_.pointM_[axis] = query.pointM_[axis];
      result.config_.observation_.normal_[axis] = query.normal_[axis];
      result.config_.earthCenterM_[axis] = 3. * axis - 2.;
    }
    particle.environmentNode_ = query.logicalNode_;
    result.interaction_.status_ = current::InteractionStatus::Selected;
    result.interaction_.interactionGrammageGPerCm2_ = query.grammageGPerCm2_;
    result.interaction_.deferredPhotonFallbackReason_ =
        static_cast<int>(current::FallbackReason::CpuOnlyProcess);
    result.interaction_.deferredPhotonTableStatus_ = 5;
    result.config_.query_.maximumMagneticDeflectionRad_ = query.maximumDeflection_;
    return result;
  }

  struct Counts {
    std::array<std::size_t, 6> limits_{};
    std::map<int, std::size_t> reasons_;
    std::size_t cases_{};
  };

  void run_dataset(environment::Dataset const& data, std::vector<Input> const& cases,
                   Counts& counts) {
    environment::Uploaded<Kokkos::HostSpace> hostGeometry(data);
    environment::Uploaded<memory_space_type> deviceGeometry(data);
    auto const hostView = hostGeometry.view();
    auto const deviceView = deviceGeometry.view();
    auto const oldHost = hostGeometry.referenceView();
    auto const oldDevice = deviceGeometry.referenceView();
    auto queries = transport_test::upload(cases);
    Kokkos::View<current::PhotonTransportOutcome*, memory_space_type> actual(
        "photon_steps", cases.size());
    Kokkos::View<Result*, memory_space_type> old("photon_reference", cases.size());
    Kokkos::parallel_for(
        "photon_steps", Kokkos::RangePolicy<execution_space_type>(0, cases.size()),
        KOKKOS_LAMBDA(std::size_t index) {
          auto const& request = queries(index);
          actual(index) = current::transport_photon(
              deviceView, request.config_, request.interaction_, request.external_);
          old(index) = capture(expected(oldDevice, request));
        });
    auto actualHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, actual);
    auto expectedHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, old);
    auto inputsAfter = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, queries);
    for (std::size_t index = 0; index < cases.size(); ++index) {
      INFO(data.name_ << " photon case " << index);
      auto const& request = cases[index];
      auto const scalar = current::transport_photon(
          hostView, request.config_, request.interaction_, request.external_);
      // Strict same-device production comparisons; no tolerance hides branch changes.
      compare(capture(scalar), capture(expected(oldHost, request)));
      compare(capture(actualHost(index)), expectedHost(index));
      Outcome before, after;
      append(before, request.interaction_);
      append(after, inputsAfter(index).interaction_);
      check(after, before);
      auto const& step = actualHost(index);
      if (step.fallbackFlag_) {
        ++counts.reasons_[static_cast<int>(step.fallback_.reason_)];
        REQUIRE(step.fallback_.interactionVertexReached_ == 0);
        REQUIRE(step.fallback_.randomDrawId_ == request.interaction_.lossDrawId_);
      } else {
        ++counts.limits_.at(static_cast<std::size_t>(step.record_.limit_));
        REQUIRE(step.record_.interaction_.deferredPhotonFallbackReason_ ==
                request.interaction_.deferredPhotonFallbackReason_);
        REQUIRE(step.record_.interaction_.deferredPhotonTableStatus_ ==
                request.interaction_.deferredPhotonTableStatus_);
        REQUIRE(step.record_.start_.historyId_ ==
                request.interaction_.particle_.historyId_);
        REQUIRE(step.record_.end_.historyId_ ==
                request.interaction_.particle_.historyId_);
      }
      ++counts.cases_;
    }
  }

  void check_exact_competition(Counts& counts) {
    auto data = environment::simple("exact ties and zero-length steps");
    data.shapes_[1].radiusM_ = 100.;
    environment::Query query;
    query.positionM_[2] = 0.;
    query.logicalNode_ = -3;
    query.pointM_[2] = 50.;
    query.grammageGPerCm2_ = 5.;
    auto base = input(query);
    std::vector<Input> cases;
    std::vector<current::PhotonTransportLimit> limits;
    std::vector<double> lengths;
    auto add = [&](Input const& value, current::PhotonTransportLimit limit,
                   double length) {
      cases.push_back(value);
      limits.push_back(limit);
      lengths.push_back(length);
    };
    using Limit = current::PhotonTransportLimit;
    add(base, Limit::Interaction, 5.);
    auto value = base;
    value.interaction_.interactionGrammageGPerCm2_ = 0.;
    add(value, Limit::Interaction, 0.);
    value = base;
    value.interaction_.interactionGrammageGPerCm2_ = 50.;
    add(value, Limit::Interaction, 50.); // interaction wins equality with observation
    value.interaction_.interactionGrammageGPerCm2_ = 200.;
    add(value, Limit::ObservationSurface, 50.);
    value.external_ = {50., true, false};
    add(value, Limit::ObservationSurface, 50.); // external must be strictly closer
    value.external_.distanceM_ = 25.;
    add(value, Limit::MaterialBoundary, 25.);
    value.external_ = {};
    value.config_.observation_.pointM_[2] = 100.;
    add(value, Limit::EscapedEnvironment, 100.); // tree wins observation equality
    value.interaction_.interactionGrammageGPerCm2_ = 100.;
    add(value, Limit::Interaction, 100.); // assigned node preserved at exact boundary
    value = base;
    value.interaction_.status_ = current::InteractionStatus::ParticleCut;
    add(value, Limit::ParticleCut, 0.);
    value = base;
    value.interaction_.particle_.timeS_ = .011;
    add(value, Limit::ParticleCut, 5.);
    value.interaction_.interactionGrammageGPerCm2_ = 200.;
    add(value, Limit::ParticleCut, 50.);
    run_dataset(data, cases, counts);
    environment::Uploaded<Kokkos::HostSpace> geometry(data);
    for (std::size_t index = 0; index < cases.size(); ++index) {
      auto const& request = cases[index];
      auto const result = current::transport_photon(
          geometry.view(), request.config_, request.interaction_, request.external_);
      REQUIRE(result.fallbackFlag_ == 0);
      auto const& record = result.record_;
      REQUIRE(record.limit_ == limits[index]);
      REQUIRE(record.distanceM_ == lengths[index]);
      REQUIRE(record.traversedGrammageGPerCm2_ == lengths[index]);
      REQUIRE(record.start_.environmentNode_ == 1);
      REQUIRE(record.end_.stepId_ ==
              record.start_.stepId_ + (record.limit_ == Limit::Interaction ? 0 : 1));
      REQUIRE(record.interaction_.interactionVertexReached_ ==
              (record.limit_ == Limit::Interaction ? 1 : 0));
      REQUIRE(record.observationSurfaceReachedBeforeCut_ == (index == 10 ? 1 : 0));
    }

    // A child interface transfers to its model rather than escaping the tree.
    data.name_ = "child material transition";
    data.nodes_.resize(3);
    data.shapes_.resize(3);
    data.nodes_[1].childOffset_ = 1;
    data.nodes_[1].childCount_ = 1;
    data.children_.push_back(2);
    data.nodes_[2].parentIndex_ = 1;
    data.nodes_[2].modelIndex_ = 1;
    data.shapes_[2].kind_ = current::EnvironmentShape::Kind::Sphere;
    data.shapes_[2].centerM_[2] = 50.;
    data.shapes_[2].radiusM_ = 20.;
    auto model = data.models_[0];
    model.mediumId_ = 73;
    model.referenceDensityGPerCm3_ = .02;
    data.models_.push_back(model);
    value = base;
    value.interaction_.status_ = current::InteractionStatus::NoDiscreteInteraction;
    run_dataset(data, {value}, counts);
    environment::Uploaded<Kokkos::HostSpace> nested(data);
    auto const crossing =
        current::transport_photon(nested.view(), value.config_, value.interaction_);
    REQUIRE(crossing.fallbackFlag_ == 0);
    REQUIRE(crossing.record_.limit_ == Limit::LayerBoundary);
    REQUIRE(crossing.record_.distanceM_ == 30.);
    REQUIRE(crossing.record_.end_.environmentNode_ == 2);
    REQUIRE(crossing.record_.end_.mediumId_ == 73);
    REQUIRE(crossing.record_.endDensityGPerCm3_ == .02);
  }

  void check_failed_inverse_and_vertex(Counts& counts) {
    // A decreasing exponential has a finite limiting total column. At that
    // exact column the native inverse is infinite; never commit that step.
    auto data = environment::simple("saturated inverse",
                                    current::EnvironmentModel::Density::FlatExponential);
    data.models_[0].referenceHeightM_ = 0.;
    data.models_[0].scaleLengthM_ = -100.;
    data.models_[0].inverseScaleLengthPerM_ = -.01;
    environment::Query query;
    query.positionM_[2] = 0.;
    query.grammageGPerCm2_ = 100.;
    auto value = input(query);
    value.external_.disableObservation_ = true;
    run_dataset(data, {value}, counts);
    environment::Uploaded<Kokkos::HostSpace> inverseGeometry(data);
    auto const inverse = current::transport_photon(inverseGeometry.view(), value.config_,
                                                   value.interaction_, value.external_);
    REQUIRE(inverse.fallbackFlag_ == 1);
    REQUIRE(inverse.fallback_.reason_ ==
            current::FallbackReason::AtmosphereInverseGrammageFailed);
    REQUIRE(std::isinf(inverse.fallback_.diagnosticValue0_));
    REQUIRE(inverse.record_.startLayerIndex_ == 1);

    // In a sliding-planar model the integral freezes the initial radial axis.
    // A very long tangential test step can reach an overflowing vertex density.
    // This is an existing diagnostic, not a reason to change the model here.
    data =
        environment::simple("nonfinite vertex density",
                            current::EnvironmentModel::Density::SlidingPlanarExponential);
    query.positionM_[2] = 10.;
    query.direction_[0] = 1.;
    query.direction_[2] = 0.;
    query.grammageGPerCm2_ = 1.e5;
    value = input(query);
    run_dataset(data, {value}, counts);
    environment::Uploaded<Kokkos::HostSpace> vertexGeometry(data);
    auto const vertex = current::transport_photon(vertexGeometry.view(), value.config_,
                                                  value.interaction_);
    REQUIRE(vertex.fallbackFlag_ == 1);
    REQUIRE(vertex.fallback_.reason_ ==
            current::FallbackReason::AtmosphereVertexLookupFailed);
    REQUIRE(vertex.record_.end_.positionM_[0] == 1.e5);
    REQUIRE(vertex.record_.interaction_.interactionVertexReached_ == 0);
  }
} // namespace photon_transport_test

TEST_CASE("Photon single steps preserve production NodeTree transport",
          "[transport][photon]") {
  using namespace photon_transport_test;
  current::Runtime runtime;
  Counts counts;
  check_exact_competition(counts);
  check_failed_inverse_and_vertex(counts);
  auto datasets = environment::datasets();
  datasets.push_back(environment::atmosphere());
  for (auto const& data : datasets) {
    std::vector<Input> cases;
    for (auto const& query : data.queries_) {
      auto base = input(query);
      for (auto status : {current::InteractionStatus::Selected,
                          current::InteractionStatus::NoDiscreteInteraction,
                          current::InteractionStatus::ParticleCut,
                          current::InteractionStatus::DistanceSampled}) {
        for (double depth : {query.grammageGPerCm2_, 0., -1., 1.e-12, 1.e8, HUGE_VAL,
                             std::numeric_limits<double>::quiet_NaN()}) {
          auto value = base;
          value.interaction_.status_ = status;
          value.interaction_.interactionGrammageGPerCm2_ = depth;
          cases.push_back(value);
          value.external_ = {1., true, true};
          cases.push_back(value);
          value.interaction_.particle_.timeS_ = .02;
          cases.push_back(value);
        }
      }
      base.interaction_.particle_.pid_ = 13;
      cases.push_back(base);
    }
    run_dataset(data, cases, counts);
  }
  for (auto count : counts.limits_) REQUIRE(count > 0);
  for (auto reason : {current::FallbackReason::UnsupportedParticle,
                      current::FallbackReason::InvalidFinalState,
                      current::FallbackReason::UnsupportedGeometry,
                      current::FallbackReason::AtmosphereGrammageFailed,
                      current::FallbackReason::AtmosphereInverseGrammageFailed,
                      current::FallbackReason::AtmosphereVertexLookupFailed}) {
    REQUIRE(counts.reasons_[static_cast<int>(reason)] > 0);
  }
  std::ofstream report("photon-transport-counts.txt");
  report << "same_device_production_cases " << counts.cases_ << '\n';
  for (std::size_t index = 0; index < counts.limits_.size(); ++index)
    report << "limit_" << index << ' ' << counts.limits_[index] << '\n';
  for (auto const& item : counts.reasons_)
    report << "fallback_" << item.first << ' ' << item.second << '\n';
}
