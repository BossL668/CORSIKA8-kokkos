/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <tests/kokkos/em/transport/TransportChecks.hpp>
#include <tests/kokkos/em/transport/PhotonTransportReference.hpp>
#include <tests/kokkos/em/continuous/ContinuousTestData.hpp>
#include <tests/kokkos/environment/EnvironmentQueryTestData.hpp>
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

} // namespace photon_transport_test
