/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "FinalStateChecks.hpp"
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <vector>

namespace final_state_test {
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = execution_space_type::memory_space;

  void check(Payload const& actual, Payload const& expected) {
    REQUIRE(actual.reals_ == expected.reals_);
    REQUIRE(actual.integers_ == expected.integers_);
    REQUIRE(actual.reals_ <= 64);
    REQUIRE(actual.integers_ <= 64);
    for (unsigned index = 0; index < actual.integers_; ++index)
      REQUIRE(actual.integer_[index] == expected.integer_[index]);
    for (unsigned index = 0; index < actual.reals_; ++index) {
      if (std::isnan(expected.real_[index]))
        REQUIRE(std::isnan(actual.real_[index]));
      else {
        REQUIRE(actual.real_[index] == expected.real_[index]);
        REQUIRE(std::signbit(actual.real_[index]) == std::signbit(expected.real_[index]));
      }
    }
  }

  template <typename TCurrent, typename TReference>
  void check_record() {
    check(flatten(TCurrent{}), flatten(TReference{}));
    Kokkos::View<TCurrent*, memory_space_type> values("final-state records", 129);
    auto inputs = Kokkos::create_mirror_view(values);
    inputs(0) = TCurrent{};
    for (std::size_t index = 1; index < 129; ++index) populate(inputs(index), index);
    Kokkos::deep_copy(values, inputs);
    Kokkos::View<Payload * [2], Kokkos::LayoutRight, memory_space_type> outputs(
        "record comparison", 129);
    Kokkos::parallel_for(
        "final-state record roundtrip", Kokkos::RangePolicy<execution_space_type>(0, 129),
        KOKKOS_LAMBDA(std::size_t index) {
          TReference reference{};
          if (index) populate(reference, index);
          outputs(index, 0) = flatten(values(index));
          outputs(index, 1) = flatten(reference);
        });
    auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, outputs);
    auto returned = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
    for (std::size_t index = 0; index < 129; ++index) {
      TReference reference{};
      if (index) populate(reference, index);
      check(flatten(inputs(index)), flatten(reference));
      check(result(index, 0), result(index, 1));
      check(flatten(returned(index)), flatten(inputs(index)));
    }
  }

  struct Input {
    current::EmThinningConfig config_{};
    double parentEnergy_{10.};
    double parentWeight_{1.};
    double firstEnergy_{4.};
    double secondEnergy_{6.};
    double firstUniform_{.25};
    double secondUniform_{.75};
  };

  KOKKOS_INLINE_FUNCTION current::EmThinningResult evaluate(Input const& input) {
    return current::apply_em_thinning(
        input.config_, input.parentEnergy_, input.parentWeight_, input.firstEnergy_,
        input.secondEnergy_, input.firstUniform_, input.secondUniform_);
  }
  KOKKOS_INLINE_FUNCTION reference::EmThinningResult reference_result(
      Input const& input) {
    auto const& config = input.config_;
    return reference::applyEmThinning({config.thresholdGeV_, config.maximumWeight_,
                                       config.enabled_, config.eraseZeroWeight_},
                                      input.parentEnergy_, input.parentWeight_,
                                      input.firstEnergy_, input.secondEnergy_,
                                      input.firstUniform_, input.secondUniform_);
  }

  std::vector<Input> cases() {
    std::vector<Input> result;
    auto const almost_zero = std::nextafter(0., 1.);
    auto const almost_one = std::nextafter(1., 0.);
    for (unsigned enabled : {0u, 1u})
      for (unsigned erase : {0u, 1u})
        for (double threshold : {9., 10., 11.})
          for (double maximum : {1., 2., 10., 100.})
            for (double weight : {0., 1.e-300, 1., 2., 10.})
              for (double fraction : {1.e-200, .1, .4, .5, .9, almost_one})
                for (double first :
                     {almost_zero, .25, .4, std::nextafter(.4, 1.), .5, almost_one})
                  for (double second : {.1, .75, almost_one})
                    result.push_back({{threshold, maximum, enabled, erase},
                                      10.,
                                      weight,
                                      10. * fraction,
                                      10. * (1. - fraction),
                                      first,
                                      second});
    Input base{{10., 10., 1, 1}};
    // Probe each validity condition independently; also keep disabled validation.
    std::array<double Input::*, 6> inputs{&Input::parentEnergy_, &Input::parentWeight_,
                                          &Input::firstEnergy_,  &Input::secondEnergy_,
                                          &Input::firstUniform_, &Input::secondUniform_};
    for (unsigned enabled : {0u, 1u}) {
      base.config_.enabled_ = enabled;
      for (auto field : inputs)
        for (double invalid : {-1., -0., 0., 1., 2., HUGE_VAL, -HUGE_VAL,
                               std::numeric_limits<double>::quiet_NaN()}) {
          auto input = base;
          input.*field = invalid;
          result.push_back(input);
        }
      for (auto field : {&current::EmThinningConfig::thresholdGeV_,
                         &current::EmThinningConfig::maximumWeight_})
        for (double invalid :
             {-1., 0., HUGE_VAL, std::numeric_limits<double>::quiet_NaN()}) {
          auto input = base;
          input.config_.*field = invalid;
          result.push_back(input);
        }
    }
    // Overflow, underflow and unequal-energy statistical branches.
    result.push_back({{10., 10., 1, 1}, 10., 1., 1.e308, 1.e308, .25, .75});
    result.push_back({{10., 10., 1, 1}, 10., 0., 4., 6., .25, .75});
    result.push_back({{10., 10., 1, 0}, 10., 0., 4., 6., .25, .75});
    return result;
  }

  void check_thinning() {
    auto const inputs = cases();
    Kokkos::View<Input*, memory_space_type> device_inputs("thinning inputs",
                                                          inputs.size());
    auto host_inputs = Kokkos::create_mirror_view(device_inputs);
    for (std::size_t index = 0; index < inputs.size(); ++index)
      host_inputs(index) = inputs[index];
    Kokkos::deep_copy(device_inputs, host_inputs);
    Kokkos::View<Payload * [3], Kokkos::LayoutRight, memory_space_type> outputs(
        "thinning comparison", inputs.size());
    Kokkos::parallel_for(
        "thinning production reference",
        Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
        KOKKOS_LAMBDA(std::size_t index) {
          outputs(index, 0) = flatten(evaluate(device_inputs(index)));
          outputs(index, 1) = flatten(reference_result(device_inputs(index)));
        });
    Kokkos::parallel_for(
        "thinning reversed work order",
        Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
        KOKKOS_LAMBDA(std::size_t index) {
          auto const reversed = device_inputs.extent(0) - 1 - index;
          outputs(reversed, 2) = flatten(evaluate(device_inputs(reversed)));
        });
    auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, outputs);
    std::array<std::size_t, 4> status{};
    std::array<std::size_t, 4> masks{};
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      auto const value = evaluate(inputs[index]);
      check(flatten(value), flatten(reference_result(inputs[index])));
      check(result(index, 0), result(index, 1));
      check(result(index, 0), result(index, 2));
      ++status[static_cast<unsigned>(value.status_)];
      ++masks[value.keepMask_];
    }
    std::ofstream report("final-state-counts.txt");
    report << "same_device_production_cases " << inputs.size() << '\n';
    report << "record_types 7\nrecord_cases 903\n";
    for (std::size_t index = 0; index < 4; ++index) {
      REQUIRE(status[index] > 0);
      REQUIRE(masks[index] > 0);
      report << "status_" << index << ' ' << status[index] << '\n';
      report << "mask_" << index << ' ' << masks[index] << '\n';
    }
    // Explicit regression: erase by final weight, not selection bit.
    Input zero{{10., 10., 1, 1}, 10., 0.};
    REQUIRE(evaluate(zero).keepMask_ == 0);
    zero.config_.eraseZeroWeight_ = 0;
    REQUIRE(evaluate(zero).keepMask_ == 3);
  }

  void check_weighted_expectation() {
    // Deterministic stratified uniforms validate the physical expectation,
    // independently of the frozen implementation (no stochastic pass/fail).
    constexpr unsigned samples = 65536;
    for (double maximum : {2., 10.}) {
      double first_sum = 0., second_sum = 0.;
      for (unsigned index = 0; index < samples; ++index) {
        Input input{{10., maximum, 1, 1}};
        input.firstUniform_ = (index + .5) / samples;
        input.secondUniform_ = ((index * 25173u + 13849u) % samples + .5) / samples;
        auto const result = evaluate(input);
        first_sum += result.firstWeight_;
        second_sum += result.secondWeight_;
        if (maximum == 10.) {
          REQUIRE((result.keepMask_ == 1u || result.keepMask_ == 2u));
          REQUIRE(std::abs(4. * result.firstWeight_ + 6. * result.secondWeight_ - 10.) <
                  1.e-14);
        }
      }
      REQUIRE(std::abs(first_sum / samples - 1.) <= 3. / samples);
      REQUIRE(std::abs(second_sum / samples - 1.) <= 3. / samples);
    }
  }
} // namespace final_state_test

TEST_CASE("Final-state records and production thinning", "[kokkos][finalstate]") {
  corsika::kokkos::Runtime runtime;
#if defined(CORSIKA_EXPECT_CUDA)
  REQUIRE(std::string(Kokkos::DefaultExecutionSpace::name()) == "Cuda");
#elif defined(CORSIKA_EXPECT_OPENMP)
  REQUIRE(std::string(Kokkos::DefaultExecutionSpace::name()) == "OpenMP");
#endif
  using namespace final_state_test;
  static_assert(current::EmThinningFirstDrawId == reference::EmThinningFirstDrawId);
  static_assert(current::EmThinningSecondDrawId == reference::EmThinningSecondDrawId);
  check_record<current::FirstInteractionRecord, reference::GpuFirstInteractionSnapshot>();
  check_record<current::PhotonFinalStateRecord, reference::PhotonFinalStateRecord>();
  check_record<current::LeptonFinalStateRecord, reference::BremsFinalStateRecord>();
  check_record<current::PhotonPairLpmSuppressionRecord,
               reference::PhotonPairLpmSuppressionRecord>();
  check_record<current::BremsLpmSuppressionRecord,
               reference::BremsLpmSuppressionRecord>();
  check_record<current::EmThinningConfig, reference::EmThinningConfig>();
  check_record<current::EmThinningResult, reference::EmThinningResult>();
  check_thinning();
  check_weighted_expectation();
}
