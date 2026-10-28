/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include "SelectionReference.hpp"
#include "../stochastic/StochasticChecks.hpp"
#include "../tables/ProposalNativeQueryTestData.hpp"
#include <corsika/kokkos/em/algorithms/LeptonVertexSelection.hpp>

namespace selection_test {
  namespace current = corsika::kokkos;
  namespace reference = selection_reference;
  using stochastic_test::append;
  using stochastic_test::check;
  using stochastic_test::execution_space_type;
  using stochastic_test::memory_space_type;
  using stochastic_test::Outcome;
  using stochastic_test::upload;

  struct Input {
    current::InteractionRecord candidate_{};
    std::uint64_t seed_{12345}, shower_{987};
    double emCut_{.1}, muonCut_{10.};
    std::uint32_t source_{1};
    bool vertex_{}, invalidView_{};
  };
  inline Input input(std::int32_t pid, double energyMeV, bool vertex = false) {
    Input result;
    result.vertex_ = vertex;
    auto& c = result.candidate_;
    c.particle_.pid_ = pid;
    c.particle_.energyGeV_ = energyMeV / 1000.;
    c.particle_.mediumId_ = 7;
    c.particle_.historyId_ = 0xfedcba9876543210ULL;
    c.particle_.parentHistoryId_ = 19;
    c.particle_.stepId_ = 0x12345678abcdefULL;
    c.particle_.generation_ = 2;
    c.particle_.weight_ = 3.;
    c.particle_.timeS_ = 1.e-6;
    c.particle_.positionM_[0] = 123.;
    c.particle_.direction_[2] = -1.;
    c.particle_.environmentNode_ = 4;
    c.inputIndex_ = 0xaabbccddeeffULL;
    c.status_ = current::InteractionStatus::RequiresReselection;
    c.interactionVertexReached_ = vertex ? 1u : 0u;
    c.totalRateCm2PerG_ = 1.;
    c.processUniform_ = .25;
    c.processRandomProcessId_ = current::InteractionColumnRandomProcessId;
    c.processDrawId_ = current::InteractionColumnDrawId;
    c.distanceDrawId_ = current::InteractionDistanceDrawId;
    c.lossDrawId_ = current::InteractionLossDrawId;
    return result;
  }
  CORSIKA_HOST_DEVICE inline current::InteractionPhysicsView physics(
      current::ProposalNativeDeviceView native, Input const& input) {
    if (input.invalidView_) native = {};
    return current::InteractionPhysicsView(
        current::ContinuousPhysicsView(native, input.emCut_, input.muonCut_),
        input.source_);
  }
  CORSIKA_HOST_DEVICE inline reference::table::NativePhysicsView old_physics(
      native_table_reference::ProposalNativeDeviceView native, Input const& input) {
    if (input.invalidView_) native = {};
    reference::table::NativePhysicsView view{};
    view.proposal_native = native;
    view.em_transport_cut_MeV = input.emCut_;
    view.muon_transport_cut_MeV = input.muonCut_;
    view.physics_source = input.source_;
    return view;
  }
  CORSIKA_HOST_DEVICE inline Outcome evaluate(current::ProposalNativeDeviceView native,
                                              Input const& input) {
    Outcome result;
    auto const view = physics(native, input);
    if (input.vertex_) {
      auto const value = current::select_lepton_vertex(view, input.candidate_,
                                                       input.seed_, input.shower_);
      append(result, value.record_);
      append(result, value.fallback_);
      append(result, value.interactionFlag_);
      append(result, value.continuationFlag_);
      append(result, value.fallbackFlag_);
      append(result, value.nativeNewtonIterations_);
      append(result, value.nativeBisectionIterations_);
      append(result, value.nativeInverseFailures_);
    } else {
      auto const& c = input.candidate_;
      auto const value = current::select_discrete_interaction(
          view, c.particle_, c.inputIndex_, input.seed_, input.shower_);
      append(result, value.interaction_);
      append(result, value.fallback_);
      append(result, value.fallbackFlag_);
      append(result, value.nativeNewtonIterations_);
      append(result, value.nativeBisectionIterations_);
      append(result, value.nativeInverseFailures_);
    }
    return result;
  }
  CORSIKA_HOST_DEVICE inline Outcome expected(
      native_table_reference::ProposalNativeDeviceView native, Input const& input) {
    Outcome result;
    auto const view = old_physics(native, input);
    auto const c = continuous_test::convert(input.candidate_);
    if (input.vertex_) {
      auto const value =
          reference::selectLeptonVertex(view, c, input.seed_, input.shower_);
      append(result, value.record);
      append(result, value.fallback);
      append(result, value.interaction_flag);
      append(result, value.continuation_flag);
      append(result, value.fallback_flag);
      append(result, value.native_newton_iterations);
      append(result, value.native_bisection_iterations);
      append(result, value.native_inverse_failures);
    } else {
      auto const value = reference::selectDiscreteInteraction(
          view, c.particle, c.input_index, input.seed_, input.shower_);
      append(result, value.interaction);
      append(result, value.fallback);
      append(result, value.fallback_flag);
      append(result, value.native_newton_iterations);
      append(result, value.native_bisection_iterations);
      append(result, value.native_inverse_failures);
    }
    return result;
  }
  inline void compare_host(current::ProposalNativeTableSet const& table,
                           Input const& query) {
    native_query_test::HostReference old(table);
    check(evaluate(current::make_proposal_native_host_view(table), query),
          expected(old.view(table), query));
  }
  inline void compare_device(current::ProposalNativeDeviceView native,
                             native_table_reference::ProposalNativeDeviceView old,
                             std::vector<Input> const& queries) {
    auto inputs = upload(queries);
    Kokkos::View<Outcome * [2], Kokkos::LayoutRight, memory_space_type> outputs(
        "selection", queries.size());
    Kokkos::parallel_for(
        "selection_reference",
        Kokkos::RangePolicy<execution_space_type>(0, queries.size()),
        KOKKOS_LAMBDA(std::size_t index) {
          outputs(index, 0) = evaluate(native, inputs(index));
          outputs(index, 1) = expected(old, inputs(index));
        });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, outputs);
    for (std::size_t index = 0; index < queries.size(); ++index) {
      INFO("Device selection case " << index);
      check(host(index, 0), host(index, 1));
    }
  }
} // namespace selection_test
