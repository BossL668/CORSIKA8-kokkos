/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "MoliereQueryTestData.hpp"
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace corsika::kokkos::test {
  inline bool moliere_same_value(double actual, double expected) {
    if (std::isnan(expected)) return std::isnan(actual);
    return actual == expected &&
           (actual != 0. || std::signbit(actual) == std::signbit(expected));
  }

  inline void require_moliere_exact(MoliereOutcome const& actual,
                                    MoliereOutcome const& expected) {
    REQUIRE(actual.status_ == expected.status_);
    REQUIRE(actual.iterations_ == expected.iterations_);
    REQUIRE(actual.count_ == expected.count_);
    for (std::uint32_t index = 0; index < expected.count_; ++index) {
      INFO("output component " << index << " actual " << std::setprecision(17)
                               << actual.values_[index] << " old "
                               << expected.values_[index]);
      REQUIRE(moliere_same_value(actual.values_[index], expected.values_[index]));
    }
  }

  struct MoliereErrorReport {
    struct NativeError {
      std::size_t count_{};
      std::size_t nonfinite_{};
      double absolute_{};
      double relative_{};
      std::uint64_t ulp_{};
    };
    std::array<NativeError, 6> native_{};
    std::array<std::size_t, 9> operations_{};
    std::array<std::size_t, 6> angleStatuses_{};
    std::size_t total_{};
    std::size_t differences_{};
    std::uint32_t maximumIterations_{};

    void observe(MoliereQuery const& query, MoliereOutcome const& actual,
                 MoliereOutcome const& expected) {
      ++total_;
      ++operations_[static_cast<std::size_t>(query.operation_)];
      maximumIterations_ = std::max(maximumIterations_, actual.iterations_);
      bool different = actual.status_ != expected.status_ ||
                       actual.iterations_ != expected.iterations_ ||
                       actual.count_ != expected.count_;
      for (std::uint32_t index = 0; index < expected.count_; ++index)
        different |= !moliere_same_value(actual.values_[index], expected.values_[index]);
      if (different) ++differences_;
      if ((query.operation_ == MoliereOperation::Angle ||
           query.operation_ == MoliereOperation::Projection) &&
          actual.status_ < angleStatuses_.size())
        ++angleStatuses_[actual.status_];
      if (!query.nativeComparable_) return;
      auto& error = native_[query.snapshotIndex_ * 3 + query.viewIndex_];
      ++error.count_;
      auto const value = actual.values_[0];
      if (actual.status_ != 0 || !std::isfinite(value) ||
          !std::isfinite(query.nativeExpected_)) {
        ++error.nonfinite_;
        return;
      }
      auto const absolute = std::abs(value - query.nativeExpected_);
      error.absolute_ = std::max(error.absolute_, absolute);
      if (query.nativeExpected_ != 0.)
        error.relative_ =
            std::max(error.relative_, absolute / std::abs(query.nativeExpected_));
      std::uint64_t actualBits{}, expectedBits{};
      std::memcpy(&actualBits, &value, sizeof(value));
      std::memcpy(&expectedBits, &query.nativeExpected_, sizeof(value));
      if (value >= 0. && query.nativeExpected_ >= 0.)
        error.ulp_ =
            std::max(error.ulp_, actualBits > expectedBits ? actualBits - expectedBits
                                                           : expectedBits - actualBits);
    }

    void write(char const* backend) const {
      auto const* directory = std::getenv("CORSIKA_MOLIERE_TEST_CACHE");
      if (!directory || !*directory)
        throw std::runtime_error("Missing Moliere evidence path");
      std::ofstream output(std::filesystem::path(directory) /
                           (std::string(backend) + "-errors.json"));
      output << std::setprecision(17) << "{\n  \"queries\": " << total_
             << ",\n  \"same_device_differences\": " << differences_
             << ",\n  \"maximum_returned_iterations\": " << maximumIterations_
             << ",\n  \"shared_electron_table_for_muon\": true,"
                "\n  \"native_comparison_is_diagnostic_not_migration_tolerance\": true,"
                "\n  \"operations\": [";
      for (std::size_t index = 0; index < operations_.size(); ++index)
        output << (index ? ", " : "") << operations_[index];
      output << "],\n  \"angle_statuses\": [";
      for (std::size_t index = 0; index < angleStatuses_.size(); ++index)
        output << (index ? ", " : "") << angleStatuses_[index];
      output << "],\n  \"native\": [\n";
      for (std::size_t index = 0; index < native_.size(); ++index) {
        auto const& error = native_[index];
        output << "    {\"particle\": " << index / 3 << ", \"view\": " << index % 3
               << ", \"count\": " << error.count_
               << ", \"nonfinite_or_failed\": " << error.nonfinite_
               << ", \"max_absolute\": " << error.absolute_
               << ", \"max_relative_nonzero_reference\": " << error.relative_
               << ", \"max_ulp_nonnegative\": " << error.ulp_ << "}"
               << (index + 1 == native_.size() ? "\n" : ",\n");
      }
      output << "  ]\n}\n";
      if (!output) throw std::runtime_error("Cannot write Moliere numerical evidence");
    }
  };
} // namespace corsika::kokkos::test
