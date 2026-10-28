/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "TransportChecks.hpp"
#include "MoliereStepReference.hpp"
#include "../moliere/MoliereTestUtilities.hpp"
#include "../moliere/MoliereQueryTestData.hpp"
#include <corsika/kokkos/em/algorithms/MoliereStep.hpp>
#include <corsika/kokkos/em/execution/MoliereInterpolationStorage.hpp>
#include <array>
#include <limits>
#include <map>

namespace moliere_stage_test {
  namespace current = corsika::kokkos;
  namespace fixture = current::test;
  namespace reference = moliere_stage_reference;
  using transport_test::append;
  using transport_test::check;
  using transport_test::execution_space_type;
  using transport_test::flatten;
  using transport_test::memory_space_type;
  using transport_test::Outcome;
  struct Input {
    current::LeptonTransportRecord record_;
    std::uint64_t seed_{0x123456789abcdef0ULL}, showerId_{0xfedcba9876543210ULL};
    std::uint32_t state_{}, capacity_{4}, view_{2};
    bool direct_{}, enabled_{true}, muonAvailable_{true};
  };
  struct Result {
    current::LeptonTransportRecord record_;
    current::FallbackRequest fallback_;
    std::uint32_t state_{};
  };
  struct Expected {
    Outcome record_, fallback_;
    std::uint32_t state_{};
  };
  template <std::size_t Capacity>
  KOKKOS_INLINE_FUNCTION Result actual(current::MoliereSnapshot const& electron,
                                       current::MoliereSnapshot const& muon,
                                       current::MoliereInterpolationView const& view,
                                       Input const& input) {
    Result result;
    result.record_ = input.record_;
    result.fallback_.diagnosticValue0_ = -99.;
    result.fallback_.finalStateDrawId_ = 77;
    auto const selected = fixture::moliere_view_variant(view, input.view_);
    if (input.direct_) {
      result.state_ =
          current::apply_moliere_scattering<Capacity>(
              current::is_muon_pid(input.record_.start_.pid_) ? muon : electron, selected,
              input.enabled_, input.seed_, input.showerId_, result.record_,
              result.fallback_)
              ? 0u
              : 1u;
    } else {
      result.state_ = current::apply_moliere_scattering_stage<Capacity>(
          electron, muon, selected, input.muonAvailable_, input.seed_, input.showerId_,
          result.record_, result.fallback_, input.state_);
    }
    return result;
  }
  template <std::size_t Capacity>
  KOKKOS_INLINE_FUNCTION Expected
  expected(fixture::moliere_reference::MoliereSnapshot const& electron,
           fixture::moliere_reference::MoliereSnapshot const& muon,
           fixture::moliere_reference::MoliereInterpolationView const& view,
           Input const& input) {
    auto record = transport_test::convert(input.record_);
    continuous_reference::ProposalFallbackEvent fallback;
    fallback.diagnostic_value0 = -99.;
    fallback.final_state_draw_id = 77;
    Expected result;
    auto const selected = fixture::moliere_reference_view_variant(view, input.view_);
    if (input.direct_) {
      result.state_ =
          reference::applyMoliereScattering<Capacity>(
              stochastic_reference::isMuonPid(record.start.pid) ? muon : electron,
              selected, input.enabled_, input.seed_, input.showerId_, record, fallback)
              ? 0u
              : 1u;
    } else {
      result.state_ = reference::applyMoliereScatteringStage<Capacity>(
          electron, muon, selected, input.muonAvailable_, input.seed_, input.showerId_,
          record, fallback, input.state_);
    }
    result.record_ = flatten(record);
    append(result.fallback_, fallback);
    return result;
  }
  template <typename TSnapshot, typename TView>
  KOKKOS_INLINE_FUNCTION Result actual_dispatch(TSnapshot const& e, TSnapshot const& m,
                                                TView const& v, Input const& i) {
    if (i.capacity_ == 1) return actual<1>(e, m, v, i);
    if (i.capacity_ == 4) return actual<4>(e, m, v, i);
    return actual<current::MaxMoliereComponents>(e, m, v, i);
  }
  template <typename TSnapshot, typename TView>
  KOKKOS_INLINE_FUNCTION Expected expected_dispatch(TSnapshot const& e,
                                                    TSnapshot const& m, TView const& v,
                                                    Input const& i) {
    if (i.capacity_ == 1) return expected<1>(e, m, v, i);
    if (i.capacity_ == 4) return expected<4>(e, m, v, i);
    return expected<current::MaxMoliereComponents>(e, m, v, i);
  }
  void compare(Result const& actual, Expected const& old) {
    REQUIRE(actual.state_ == old.state_);
    check(flatten(actual.record_), old.record_);
    Outcome fallback;
    append(fallback, actual.fallback_);
    check(fallback, old.fallback_);
  }
  Input input(int pid, std::size_t index) {
    Input result;
    result.record_ = transport_test::make_lepton_payload(index + 1);
    result.record_.start_.pid_ = result.record_.end_.pid_ = pid;
    result.record_.start_.energyGeV_ = 10.;
    result.record_.end_.energyGeV_ = 9.;
    result.record_.start_.stepId_ = 0x12345678ffffffffULL + index;
    result.record_.end_.stepId_ = result.record_.start_.stepId_ + 1;
    result.record_.traversedGrammageGPerCm2_ = 1.;
    result.record_.limit_ = static_cast<current::LeptonTransportLimit>(index % 9);
    for (int axis = 0; axis < 3; ++axis) {
      result.record_.start_.direction_[axis] = axis == 2 ? 1. : 0.;
      result.record_.end_.direction_[axis] = axis == 0 ? .6 : axis == 2 ? .8 : 0.;
    }
    result.record_.multipleScatteringApplied_ = 0;
    result.record_.multipleScatteringFirstUniform_ = -7.;
    result.record_.multipleScatteringSecondUniform_ = -8.;
    result.record_.multipleScatteringAzimuthUniform_ = -9.;
    return result;
  }
} // namespace moliere_stage_test

TEST_CASE("Moliere stage preserves production random draws, outputs and fallback routing",
          "[kokkos][moliere-stage]") {
  using namespace moliere_stage_test;
  current::Runtime runtime;
  std::vector<Input> inputs;
  std::array<current::MoliereSnapshot, 2> snapshots{};
  current::MoliereInterpolationTable table{};
  current::MoliereInterpolationStorage<execution_space_type> storage;
  current::MoliereInterpolationView hostView, deviceView;
#ifdef CORSIKA_MOLIERE_STAGE_REAL_TABLES
  auto const data = fixture::make_moliere_dataset();
  snapshots = {data.snapshots_[0], data.snapshots_[1]};
  table = data.table_;
  storage.initialize(snapshots[0], data.cacheFile_);
  hostView = current::make_moliere_interpolation_view(
      current::NativeArrayView<current::MoliereCubicPolynomial>{table.polynomials_},
      current::NativeArrayView<double>{table.initialGuessDelta_});
  deviceView = storage.getDeviceView();
  for (int pid : {11, -11, 13, -13})
    for (unsigned view : {0u, 1u, 2u})
      for (unsigned capacity : {1u, 4u, 16u})
        for (double grammage : {0., 1.e-12, 1.e-4, .1, 1., 10.})
          for (double kinetic : {.01, 1., 1.e3, 1.e6}) {
            auto value = input(pid, inputs.size());
            value.capacity_ = capacity;
            value.view_ = view;
            value.record_.start_.energyGeV_ =
                (snapshots[current::is_muon_pid(pid)].particleMassMeV_ + kinetic) / 1000.;
            value.record_.end_.energyGeV_ = value.record_.start_.energyGeV_ * .9;
            value.record_.traversedGrammageGPerCm2_ = grammage;
            inputs.push_back(value);
            value.direct_ = true;
            inputs.push_back(value);
          }
  for (int pid : {11, -11, 13, -13})
    for (unsigned limit = 0; limit < 9; ++limit)
      for (unsigned seed = 0; seed < 8; ++seed) {
        auto value = input(pid, inputs.size());
        value.record_.limit_ = static_cast<current::LeptonTransportLimit>(limit);
        value.seed_ = seed;
        value.showerId_ = seed ^ 0x8000000000000000ULL;
        inputs.push_back(value);
      }
  for (int pid : {11, -11, 13, -13}) {
    for (unsigned kind = 0; kind < 5; ++kind) {
      auto value = input(pid, inputs.size());
      if (kind == 0) value.record_.traversedGrammageGPerCm2_ = -1.;
      if (kind == 1)
        value.record_.start_.energyGeV_ = std::numeric_limits<double>::quiet_NaN();
      if (kind == 2) value.record_.start_.energyGeV_ = 0.;
      if (kind == 3)
        value.record_.start_.direction_[0] = std::numeric_limits<double>::quiet_NaN();
      if (kind == 4)
        value.record_.end_.direction_[0] = std::numeric_limits<double>::infinity();
      inputs.push_back(value);
    }
  }
#endif
  for (int pid : {11, -11, 13, -13}) {
    for (unsigned state : {0u, 1u, 2u, 17u})
      for (bool available : {false, true}) {
        auto value = input(pid, inputs.size());
        value.state_ = state;
        value.muonAvailable_ = available;
        inputs.push_back(value);
      }
    auto value = input(pid, inputs.size());
    value.direct_ = true;
    value.enabled_ = false;
    inputs.push_back(value);
  }
  auto oldTable = fixture::moliere_reference_table(table);
  auto oldPolynomials = transport_test::upload(
      std::vector<fixture::moliere_reference::MoliereCubicPolynomial>(
          oldTable.polynomials.begin(), oldTable.polynomials.end()));
  auto oldGuesses = transport_test::upload(std::vector<double>(
      oldTable.initial_guess_delta.begin(), oldTable.initial_guess_delta.end()));
  auto oldHostView = fixture::moliere_reference::makeMoliereInterpolationView(
      oldTable.polynomials.data(), oldTable.initial_guess_delta.data());
  auto oldDeviceView = fixture::moliere_reference::makeMoliereInterpolationView(
      oldPolynomials.data(), oldGuesses.data());
  if (!storage.isInitialized()) oldHostView = oldDeviceView = {};
  auto e = snapshots[0], m = snapshots[1];
  auto oldE = fixture::moliere_reference_snapshot(e),
       oldM = fixture::moliere_reference_snapshot(m);
  auto queries = transport_test::upload(inputs);
  Kokkos::View<Result*, memory_space_type> output("moliere_stage", inputs.size());
  Kokkos::View<Result*, memory_space_type> reversed("moliere_reordered", inputs.size());
  Kokkos::View<Expected*, memory_space_type> oracle("moliere_reference", inputs.size());
  auto count = inputs.size();
  Kokkos::parallel_for(
      "moliere_stage", Kokkos::RangePolicy<execution_space_type>(0, count),
      KOKKOS_LAMBDA(std::size_t index) {
        output(index) = actual_dispatch(e, m, deviceView, queries(index));
        oracle(index) = expected_dispatch(oldE, oldM, oldDeviceView, queries(index));
        auto reverseIndex = count - 1 - index;
        reversed(reverseIndex) = actual_dispatch(e, m, deviceView, queries(reverseIndex));
      });
  auto out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  auto old = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, oracle);
  auto reordered = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, reversed);
  auto after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, queries);
  std::map<std::string, std::size_t> counts;
  for (std::size_t index = 0; index < count; ++index) {
    INFO("Moliere stage input " << index);
    auto const& value = inputs[index];
    compare(actual_dispatch(e, m, hostView, value),
            expected_dispatch(oldE, oldM, oldHostView, value));
    compare(out(index), old(index));
    compare(reordered(index), old(index));
    check(flatten(after(index).record_), flatten(value.record_));
    auto const& result = out(index);
    bool bypass =
        (!value.direct_ && value.state_ != 0) || (value.direct_ && !value.enabled_);
    if (bypass) {
      ++counts["bypass"];
      check(flatten(result.record_), flatten(value.record_));
      REQUIRE(result.fallback_.diagnosticValue0_ == -99.);
      REQUIRE(result.fallback_.finalStateDrawId_ == 77);
      REQUIRE(result.state_ == (value.direct_ || value.state_ == 2 ? 0u : value.state_));
    } else if (result.state_ == 1) {
      if (result.fallback_.reason_ ==
          current::FallbackReason::MoliereParametersUnavailable) {
        ++counts["missing_muon"];
        REQUIRE(current::is_muon_pid(value.record_.start_.pid_));
        REQUIRE_FALSE(value.muonAvailable_);
        check(flatten(result.record_), flatten(value.record_));
      } else {
        REQUIRE(result.fallback_.reason_ ==
                current::FallbackReason::MoliereSamplingFailed);
        if (result.fallback_.finalStateDrawId_ == current::MoliereFirstAngleDrawId) {
          ++counts["sampling_failed"];
          REQUIRE(result.fallback_.finalStateUniform_ ==
                  result.record_.multipleScatteringFirstUniform_);
        } else {
          ++counts["direction_failed"];
          REQUIRE(result.fallback_.finalStateDrawId_ == current::MoliereAzimuthDrawId);
          REQUIRE(result.fallback_.finalStateUniform_ ==
                  result.record_.multipleScatteringAzimuthUniform_);
        }
      }
    } else {
      if (result.record_.multipleScatteringApplied_)
        ++counts["deflection"];
      else
        ++counts["no_deflection"];
      bool vertex =
          !value.direct_ &&
          (value.record_.limit_ == current::LeptonTransportLimit::InteractionCandidate ||
           value.record_.limit_ == current::LeptonTransportLimit::DecayCandidate);
      Outcome a, b;
      append(a, result.record_.interaction_.particle_);
      append(b, vertex ? result.record_.end_ : value.record_.interaction_.particle_);
      check(a, b);
      ++counts[vertex ? "vertex_copy" : "vertex_preserved"];
    }
    if (!bypass && !(result.state_ == 1 &&
                     result.fallback_.reason_ ==
                         current::FallbackReason::MoliereParametersUnavailable)) {
      auto const& start = value.record_.start_;
      stochastic_reference::RandomNumberKey key{
          value.seed_,
          value.showerId_,
          start.historyId_,
          start.stepId_,
          stochastic_reference::ContinuousScatteringRandomProcessId,
          0};
      REQUIRE(result.record_.multipleScatteringFirstUniform_ ==
              stochastic_reference::uniformOpen01(key));
      key.draw_id = 1;
      REQUIRE(result.record_.multipleScatteringSecondUniform_ ==
              stochastic_reference::uniformOpen01(key));
      key.draw_id = 2;
      REQUIRE(result.record_.multipleScatteringAzimuthUniform_ ==
              stochastic_reference::uniformOpen01(key));
    }
  }
  REQUIRE(counts["bypass"] > 0);
  REQUIRE(counts["missing_muon"] > 0);
#ifdef CORSIKA_MOLIERE_STAGE_REAL_TABLES
  for (auto name : {"sampling_failed", "direction_failed", "deflection", "no_deflection",
                    "vertex_copy", "vertex_preserved"})
    REQUIRE(counts[name] > 0);
#else
  REQUIRE(counts["sampling_failed"] > 0);
#endif
  std::ofstream report("moliere-stage-counts.txt");
  report << "same_device_production_cases " << count << '\n';
  for (auto const& entry : counts) report << entry.first << ' ' << entry.second << '\n';
  REQUIRE(report.good());
}
