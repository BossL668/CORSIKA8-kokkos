/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <tests/kokkos/em/transport/TransportChecks.hpp>
#include <tests/kokkos/em/transport/LeptonTransportReference.hpp>
#include <tests/kokkos/em/continuous/ContinuousTestData.hpp>
#include <tests/kokkos/environment/EnvironmentQueryTestData.hpp>
#include <corsika/kokkos/em/algorithms/LeptonTransportStep.hpp>
#include <corsika/kokkos/em/execution/NativeTableStorage.hpp>
#include <array>
#include <map>

namespace lepton_transport_test {
  namespace current = corsika::kokkos;
  namespace reference = lepton_transport_reference;
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
    double emCutMeV_{.4}, muonCutMeV_{.4};
    bool applyMoliere_{true};
  };
  struct Result {
    current::LeptonTransportRecord record_{};
    current::FallbackRequest fallback_{};
    std::uint32_t state_{};
  };
  struct Expected {
    Outcome record_, fallback_;
    std::uint32_t state_{};
  };
  KOKKOS_INLINE_FUNCTION current::LeptonTransportRecord initial_record() {
    current::LeptonTransportRecord result;
    result.distanceM_ = -17.;
    result.end_.weight_ = 31.;
    return result;
  }
  template <typename TEnvironmentView>
  KOKKOS_INLINE_FUNCTION Result actual(current::ProposalNativeDeviceView const& native,
                                       TEnvironmentView const& view, Input const& input) {
    Result output;
    output.record_ = initial_record();
    output.fallback_.diagnosticValue0_ = -99.;
    current::ContinuousPhysicsView table(native, input.emCutMeV_, input.muonCutMeV_);
    output.state_ = current::transport_lepton(
        table, input.applyMoliere_, view, input.config_, input.interaction_,
        output.record_, output.fallback_, input.external_);
    return output;
  }
  KOKKOS_INLINE_FUNCTION Expected
  expected(native_table_reference::ProposalNativeDeviceView const& native,
           environment_query_reference::FullView const& view, Input const& input) {
    reference::EnvironmentSnapshot snapshot;
    snapshot.node_tree = view;
    snapshot.node_tree.maximumDeflection_ =
        input.config_.query_.maximumMagneticDeflectionRad_;
    for (int axis = 0; axis < 3; ++axis) {
      snapshot.earth_center_m[axis] = input.config_.earthCenterM_[axis];
      snapshot.point[axis] = input.config_.observation_.pointM_[axis];
      snapshot.normal[axis] = input.config_.observation_.normal_[axis];
    }
    continuous_test::Query query;
    query.emCutMeV_ = input.emCutMeV_;
    query.muonCutMeV_ = input.muonCutMeV_;
    auto record = transport_test::convert(initial_record());
    reference::ProposalFallbackEvent fallback;
    fallback.diagnostic_value0 = -99.;
    auto flag = reference::transportLepton(
        continuous_test::old_view(native, query), input.applyMoliere_, snapshot,
        continuous_test::convert(input.interaction_), record, fallback,
        {input.external_.distanceM_, input.external_.enabled_,
         input.external_.disableObservation_});
    Expected output;
    output.record_ = flatten(record);
    append(output.fallback_, fallback);
    output.state_ = flag;
    return output;
  }
  void compare(Result const& result, Expected const& old) {
    REQUIRE(old.state_ != 99); // Legacy sphere-layer branch must remain unreachable.
    REQUIRE(result.state_ == old.state_);
    check(flatten(result.record_), old.record_);
    Outcome fallback;
    append(fallback, result.fallback_);
    check(fallback, old.fallback_);
  }
  Input input(environment::Query const& query, int pid = 11, double energyMeV = 10000.) {
    Input result;
    result.interaction_ = continuous_test::interaction(pid, energyMeV);
    result.interaction_.status_ = current::InteractionStatus::DistanceSampled;
    result.interaction_.interactionGrammageGPerCm2_ = query.grammageGPerCm2_;
    result.interaction_.decayDistanceM_ = HUGE_VAL;
    auto& particle = result.interaction_.particle_;
    particle.environmentNode_ = query.logicalNode_;
    for (int axis = 0; axis < 3; ++axis) {
      particle.positionM_[axis] = query.positionM_[axis];
      particle.direction_[axis] = query.direction_[axis];
      result.config_.observation_.pointM_[axis] = query.pointM_[axis];
      result.config_.observation_.normal_[axis] = query.normal_[axis];
      result.config_.earthCenterM_[axis] = 2. * axis - 3.;
    }
    result.config_.query_.maximumMagneticDeflectionRad_ = query.maximumDeflection_;
    return result;
  }
} // namespace lepton_transport_test
