/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include "AuxiliaryModelQueryTestData.hpp"
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace corsika::kokkos::test {
  struct AuxiliaryErrorReport {
    struct Error {
      std::size_t count_{};
      double absolute_{};
      double relative_{};
      std::uint64_t ulp_{};
    };
    std::array<Error, 3> native_{};
    std::array<std::size_t, 5> statusCounts_{};
    std::array<std::uint32_t, 3> branchMasks_{};
    std::size_t total_{};
    std::size_t sameDeviceDifferences_{};

    void observe(AuxiliaryQuery const& query, AuxiliaryOutcome actual,
                 AuxiliaryOutcome expected) {
      ++total_;
      if (actual.status_ != expected.status_ || actual.reserved_ != expected.reserved_ ||
          (actual.value_ != expected.value_ &&
           !(std::isnan(actual.value_) && std::isnan(expected.value_))) ||
          (actual.value_ == 0. &&
           std::signbit(actual.value_) != std::signbit(expected.value_)))
        ++sameDeviceDifferences_;
      auto const operation = static_cast<std::size_t>(query.operation_);
      if (operation >= 3) return;
      if (actual.status_ < statusCounts_.size()) ++statusCounts_[actual.status_];
      branchMasks_[operation] |= query.branchMask_;
      if (!query.nativeComparable_) return;
      auto& error = native_[operation];
      ++error.count_;
      auto const absolute = std::abs(actual.value_ - query.nativeExpected_);
      error.absolute_ = std::max(error.absolute_, absolute);
      if (query.nativeExpected_ != 0.)
        error.relative_ =
            std::max(error.relative_, absolute / std::abs(query.nativeExpected_));
      std::uint64_t actualBits{}, expectedBits{};
      std::memcpy(&actualBits, &actual.value_, sizeof(actualBits));
      std::memcpy(&expectedBits, &query.nativeExpected_, sizeof(expectedBits));
      auto const difference = actualBits > expectedBits ? actualBits - expectedBits
                                                        : expectedBits - actualBits;
      error.ulp_ = std::max(error.ulp_, difference);
    }

    void write(char const* name) const {
      auto const* directory = std::getenv("CORSIKA_AUXILIARY_TEST_CACHE");
      if (!directory || !*directory)
        throw std::runtime_error("Missing auxiliary report directory");
      auto const path =
          std::filesystem::path(directory) / (std::string(name) + "-errors.json");
      std::ofstream stream(path);
      stream
          << std::setprecision(17) << "{\n  \"queries\": " << total_
          << ",\n  \"same_device_differences\": " << sameDeviceDifferences_
          << ",\n  \"native_probability_not_clamped\": true,\n  "
             "\"native_tolerance_coefficient\": "
          << 512. * std::numeric_limits<double>::epsilon()
          << ",\n  \"native_tolerance_rule\": \"coefficient * max(1, abs(reference))\","
          << "\n  \"native\": [\n";
      for (std::size_t index = 0; index < native_.size(); ++index) {
        auto const& error = native_[index];
        stream << "    {\"operation\": " << index << ", \"queries\": " << error.count_
               << ", \"max_absolute\": " << error.absolute_
               << ", \"max_relative_nonzero_reference\": " << error.relative_
               << ", \"max_ulp\": " << error.ulp_
               << ", \"branch_mask\": " << branchMasks_[index] << "}"
               << (index + 1 == native_.size() ? "\n" : ",\n");
      }
      stream << "  ],\n  \"lpm_status_counts\": [";
      for (std::size_t index = 0; index < statusCounts_.size(); ++index)
        stream << (index ? ", " : "") << statusCounts_[index];
      stream << "]\n}\n";
      if (!stream) throw std::runtime_error("Cannot write auxiliary numerical evidence");
    }
  };

  inline void require_auxiliary_exact(AuxiliaryOutcome actual,
                                      AuxiliaryOutcome expected) {
    REQUIRE(actual.status_ == expected.status_);
    REQUIRE(actual.reserved_ == expected.reserved_);
    if (std::isnan(expected.value_)) {
      REQUIRE(std::isnan(actual.value_));
    } else {
      REQUIRE(actual.value_ == expected.value_);
      if (expected.value_ == 0.)
        REQUIRE(std::signbit(actual.value_) == std::signbit(expected.value_));
    }
  }

  inline void require_auxiliary_native(AuxiliaryOutcome actual,
                                       AuxiliaryQuery const& query) {
    REQUIRE(actual.status_ == 0);
    REQUIRE(std::isfinite(query.nativeExpected_));
    // Direct library calls and compiled inline formulae may differ in rounding;
    // this does not relax the separate same-device exact migration comparison.
    auto const error = std::abs(actual.value_ - query.nativeExpected_);
    REQUIRE(error <= 512. * std::numeric_limits<double>::epsilon() *
                         std::max(1., std::abs(query.nativeExpected_)));
  }
} // namespace corsika::kokkos::test
