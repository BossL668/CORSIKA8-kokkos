/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "ContinuousChecks.hpp"

TEST_CASE("Continuous contracts preserve cuts and fallback provenance",
          "[continuous_contracts]") {
  using namespace continuous_test;
  for (std::uint32_t count : {0u, 1u, 65535u, 65536u, 0xffffffffu}) {
    for (std::uint32_t other : {0u, 7u, 65536u, 0xffffffffu}) {
      auto const packed = current::pack_native_inverse_iterations(count, other);
      REQUIRE(packed == reference::tables::packNativeInverseIterations(count, other));
      current::TableQueryResult const value{current::TableLookupStatus::Success, packed,
                                            1.};
      REQUIRE(current::native_newton_iterations(value) == std::min(count, 65535u));
      REQUIRE(current::native_bisection_iterations(value) == std::min(other, 65535u));
    }
  }
  for (unsigned status = 0; status < 15; ++status) {
    auto const actual = current::proposal_fallback_reason(
        static_cast<current::TableLookupStatus>(status));
    auto const old = reference::proposalFallbackReason(
        static_cast<reference::tables::TableLookupStatus>(status));
    REQUIRE(static_cast<unsigned>(actual) == static_cast<unsigned>(old));
    for (bool native : {false, true}) {
      auto record = interaction(13, 1.e3);
      record.proposalSelectionRandomProcessId_ = native ? 444 : 0;
      Outcome first, second;
      append(first, current::make_process_fallback_event(record, actual));
      append(second, reference::makeProcessFallbackEvent(convert(record), old));
      check(first, second);
    }
  }
  current::ProposalNativeDeviceView const empty;
  native_table_reference::ProposalNativeDeviceView const oldEmpty;
  current::ContinuousPhysicsView const differentCuts(empty, .3, .8);
  for (int pid : {11, -11, 13, -13}) {
    exact(current::query_continuous_transport_cut(differentCuts, pid).value_,
          std::abs(pid) == 13 ? .8 : .3);
  }
  for (int pid : {11, -11, 13, -13, 22, 2212}) {
    for (auto op : {Operation::Mass, Operation::TransportMass, Operation::Cut,
                    Operation::Minimum, Operation::Range, Operation::Prepare}) {
      Query query;
      query.operation_ = op;
      query.interaction_ = interaction(pid, 1.e3);
      check(evaluate(empty, query), expected(oldEmpty, query));
    }
  }
  // Synthetic utility-only owner: no rate, total-rate or dE/dX columns are present.
  std::vector<current::NativeUtilityColumn> utility(1);
  utility[0].pdgId_ = 11;
  utility[0].particleMassMeV_ = .51099895;
  utility[0].lowerEnergyLimitMeV_ = .51099895;
  utility[0].spline_.axis_ = {current::NativeAxisType::Linear, 2, .51099895, 1000.,
                              999.48900105};
  utility[0].spline_.coefficientCount_ = 2;
  std::vector<double> values{0., 999.48900105}, derivatives{999.48900105, 999.48900105};
  current::ProposalNativeDeviceView native(
      {}, {}, {}, {}, {},
      current::NativeArrayView<current::NativeUtilityColumn, std::uint32_t>(utility), {},
      {}, {}, {}, {}, current::NativeArrayView<double, std::uint64_t>(values),
      current::NativeArrayView<double, std::uint64_t>(derivatives), {});
  native_table_reference::NativeUtilityColumn oldColumn =
      native_query_test::convert(utility[0]);
  native_table_reference::ProposalNativeDeviceView old;
  old.utility_columns = &oldColumn;
  old.utility_column_count = 1;
  old.cubic_values = values.data();
  old.cubic_node_derivatives = derivatives.data();
  old.cubic_coefficient_count = values.size();
  current::ContinuousPhysicsView view(native, .4, .4);
  REQUIRE(current::query_continuous_mass(view, 11).status_ ==
          current::TableLookupStatus::Success);
  REQUIRE(current::query_continuous_minimum_energy(view, 11).status_ ==
          current::TableLookupStatus::Success);
  for (double energy : {.6, .9109989, std::nextafter(.9109989, 0.),
                        std::nextafter(.9109989, INFINITY), 1., 10., 1000.}) {
    Query query;
    query.interaction_ = interaction(11, energy);
    for (auto op : {Operation::Mass, Operation::Minimum, Operation::Prepare,
                    Operation::AfterLoss}) {
      query.operation_ = op;
      check(evaluate(native, query), expected(old, query));
    }
  }
  auto const range = current::query_continuous_range(view, 11, 10.);
  REQUIRE(range.status_ == current::TableLookupStatus::Success);
  for (double grammage : {0., std::nextafter(range.value_, 0.), range.value_,
                          std::nextafter(range.value_, INFINITY), -.1}) {
    Query query;
    query.operation_ = Operation::AfterLoss;
    query.interaction_ = interaction(11, 10.);
    query.argument_ = grammage;
    check(evaluate(native, query), expected(old, query));
  }
  REQUIRE(
      current::query_energy_after_continuous_loss(view, 11, 10., range.value_).status_ ==
      current::TableLookupStatus::TransportCutReached);
  auto const record = interaction(11, 10.);
  auto const preparation = current::prepare_lepton_continuous_step(view, record);
  REQUIRE(preparation.fallbackFlag_ == 0);
  REQUIRE(preparation.maximumGrammageGPerCm2_ > 0.);
  auto const lossAtEndpoint = current::evaluate_lepton_continuous_loss(
      view, record, preparation, range.value_, false);
  REQUIRE(lossAtEndpoint.transportCutReached_ == 1);
  exact(lossAtEndpoint.finalEnergyMeV_, preparation.minimumEnergyMeV_);
  auto const noLoss =
      current::evaluate_lepton_continuous_loss(view, record, preparation, 0., true);
  REQUIRE(noLoss.fallbackFlag_ == 1);
  REQUIRE(noLoss.fallback_.reason_ == current::FallbackReason::InvalidTableQuery);
  exact(noLoss.fallback_.diagnosticValue0_, 10.);
  exact(noLoss.fallback_.diagnosticValue1_, 10.);
  exact(noLoss.fallback_.diagnosticValue2_, 0.);
  auto const invalid =
      current::evaluate_lepton_continuous_loss(view, record, preparation, -.1, false);
  REQUIRE(invalid.fallbackFlag_ == 1);
  REQUIRE(invalid.fallback_.diagnosticStatus_ ==
          static_cast<int>(current::TableLookupStatus::NonFiniteInput));
  exact(invalid.fallback_.diagnosticValue0_, 5.);
  exact(invalid.fallback_.diagnosticValue1_, -.1);
  for (bool won : {false, true}) {
    auto const oldView = old_view(old, Query{});
    auto const oldRecord = convert(record);
    auto const oldPreparation =
        reference::step::prepareLeptonContinuousStep(oldView, oldRecord);
    for (double amount :
         {0., -.1, range.value_, std::nextafter(range.value_, INFINITY)}) {
      Outcome actual, expected;
      append(actual, current::evaluate_lepton_continuous_loss(view, record, preparation,
                                                              amount, won));
      append(expected, reference::step::evaluateLeptonContinuousLoss(
                           oldView, oldRecord, oldPreparation, amount, won));
      check(actual, expected);
    }
  }
  // Set the cut from the already materialized kinetic energy: exact equality
  // and adjacent cuts test the strict comparison without a GeV roundtrip ambiguity.
  auto const equalityRecord = interaction(11, 1.);
  auto const kinetic = equalityRecord.particle_.energyGeV_ * 1000. -
                       1000. * current::TransportElectronMassGeV;
  for (double cut :
       {std::nextafter(kinetic, 0.), kinetic, std::nextafter(kinetic, INFINITY)}) {
    current::ContinuousPhysicsView const exactCut(native, cut, cut);
    auto const result = current::prepare_lepton_continuous_step(exactCut, equalityRecord);
    REQUIRE(result.fallbackFlag_ == 0);
    REQUIRE(result.particleCut_ == static_cast<unsigned>(kinetic < cut));
    auto forced = equalityRecord;
    forced.status_ = current::InteractionStatus::ParticleCut;
    REQUIRE(current::prepare_lepton_continuous_step(exactCut, forced).particleCut_ == 1);
  }
  // Finite but deliberately nonmonotone synthetic utility. Both queried ranges
  // are valid, isolating the original tiny-negative subtraction guard.
  std::vector<current::NativeUtilityColumn> roundedUtility = utility;
  roundedUtility[0].lowerEnergyLimitMeV_ = 1.;
  roundedUtility[0].spline_.axis_ = {current::NativeAxisType::Linear, 3, 1., 17., 8.};
  roundedUtility[0].spline_.coefficientCount_ = 3;
  std::vector<double> roundedValues{0., 1., 1.}, roundedDerivatives{0., 0., 0.};
  current::ProposalNativeDeviceView roundedNative(
      {}, {}, {}, {}, {},
      current::NativeArrayView<current::NativeUtilityColumn, std::uint32_t>(
          roundedUtility),
      {}, {}, {}, {}, {}, current::NativeArrayView<double, std::uint64_t>(roundedValues),
      current::NativeArrayView<double, std::uint64_t>(roundedDerivatives), {});
  current::ContinuousPhysicsView const rounded(roundedNative, 4., 4.);
  auto const roundedRecord = interaction(11, 17.);
  for (double delta : {8. * std::numeric_limits<double>::epsilon(), 1.e-6}) {
    roundedValues[1] = 1. + delta;
    auto const result = current::prepare_lepton_continuous_step(rounded, roundedRecord);
    if (delta < 1.e-10) {
      REQUIRE(result.fallbackFlag_ == 0);
      exact(result.maximumGrammageGPerCm2_, 0.);
    } else {
      REQUIRE(result.fallbackFlag_ == 1);
      REQUIRE(result.maximumGrammageGPerCm2_ < 0.);
      exact(result.fallback_.diagnosticValue2_, result.maximumGrammageGPerCm2_);
    }
  }
  // Missing utility fails before a malformed cutoff is consulted; no fake
  // positive-count/null-storage view is manufactured to reach undefined states.
  current::ContinuousPhysicsView const absent(
      empty, std::numeric_limits<double>::quiet_NaN(), 0.);
  auto const missing = current::prepare_lepton_continuous_step(absent, record);
  REQUIRE(missing.fallbackFlag_ == 1);
  exact(missing.fallback_.diagnosticValue0_, 1.);
  REQUIRE(missing.fallback_.reason_ == current::FallbackReason::ParticleTableMissing);
  for (unsigned status = 0; status < 10; ++status) {
    for (bool continuous : {false, true}) {
      current::NativeQueryResult const nativeResult{
          static_cast<current::NativeQueryStatus>(status), 65537, 999999, 1.25};
      native_table_reference::NativeQueryResult const referenceResult{
          static_cast<native_table_reference::NativeQueryStatus>(status), 65537, 999999,
          1.25};
      Outcome actual, expected;
      append(actual,
             current::table_query_detail::from_proposal_native(nativeResult, continuous));
      append(expected,
             reference::tables::detail::fromProposalNative(referenceResult, continuous));
      check(actual, expected);
    }
  }
}
