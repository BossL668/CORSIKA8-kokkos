/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include "ContinuousTestData.hpp"
#include <catch2/catch_all.hpp>
#include <fstream>
#include <iomanip>
#include <map>

namespace continuous_test {
  inline void exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
      return;
    }
    REQUIRE(actual == expected);
    if (expected == 0.) REQUIRE(std::signbit(actual) == std::signbit(expected));
  }
  inline void check(Outcome const& actual, Outcome const& expected) {
    REQUIRE(actual.valueCount_ <= 96);
    REQUIRE(actual.stateCount_ <= 128);
    compare(actual, expected, exact,
            [](auto first, auto second) { REQUIRE(first == second); });
  }
  inline void populate_records(Dataset& data) {
    for (auto& query : data.queries_) {
      auto const pid = query.interaction_.particle_.pid_;
      auto const energy = query.interaction_.particle_.energyGeV_;
      query.interaction_ = interaction(pid, energy * 1000.);
      query.interaction_.particle_.energyGeV_ = energy;
    }
    add_edge_queries(data);
  }
  struct Diagnostic {
    std::size_t count_{};
    double absolute_{}, relative_{};
  };
  inline void report(std::string const& device, Dataset const& data,
                     std::vector<Outcome> const& output) {
    REQUIRE(output.size() == data.queries_.size());
    std::array<std::size_t, 11> counts{};
    std::map<std::int32_t, std::array<std::size_t, 2>> continuousLimitCases;
    std::map<std::pair<unsigned, int>, Diagnostic> native;
    std::uint64_t newton{}, bisection{};
    for (std::size_t index = 0; index < data.queries_.size(); ++index) {
      auto const& query = data.queries_[index];
      auto const& result = output[index];
      ++counts[static_cast<std::size_t>(query.operation_)];
      if (query.continuousLimitWon_) {
        REQUIRE(query.continuousLimitWon_ == 1);
        REQUIRE(query.operation_ == Operation::Loss);
        Outcome prepShape, lossShape;
        append(prepShape, current::LeptonContinuousPreparation{});
        append(lossShape, current::LeptonContinuousLoss{});
        REQUIRE(result.valueCount_ == prepShape.valueCount_ + lossShape.valueCount_);
        REQUIRE(result.stateCount_ == prepShape.stateCount_ + lossShape.stateCount_);
        REQUIRE(result.states_[0] == 0); // Preparation did not reach ParticleCut.
        REQUIRE(result.states_[1] == 0); // Preparation succeeded.
        REQUIRE(result.states_[prepShape.stateCount_] == 0); // No endpoint cut.
        auto& cases = continuousLimitCases[query.interaction_.particle_.pid_];
        auto const initialEnergy = query.interaction_.particle_.energyGeV_ * 1000.;
        if (query.argument_ == 0.) {
          // Verify the actual device-returned failure and its complete copied
          // provenance, in addition to the independent production comparison.
          current::LeptonContinuousLoss rejected;
          rejected.finalEnergyMeV_ = initialEnergy;
          rejected.fallbackFlag_ = 1;
          rejected.fallback_ = current::make_process_fallback_event(
              query.interaction_, current::FallbackReason::InvalidTableQuery);
          rejected.fallback_.diagnosticValue0_ = initialEnergy;
          rejected.fallback_.diagnosticValue1_ = initialEnergy;
          rejected.fallback_.diagnosticValue2_ = 0.;
          Outcome wanted;
          append(wanted, rejected);
          for (std::uint32_t value = 0; value < wanted.valueCount_; ++value)
            exact(result.values_[prepShape.valueCount_ + value], wanted.values_[value]);
          for (std::uint32_t state = 0; state < wanted.stateCount_; ++state)
            REQUIRE(result.states_[prepShape.stateCount_ + state] ==
                    wanted.states_[state]);
          ++cases[0];
        } else {
          REQUIRE(std::isfinite(query.argument_));
          REQUIRE(query.argument_ > 0.);
          REQUIRE(result.states_[prepShape.stateCount_ + 1] == 0); // Loss succeeded.
          auto const finalEnergy = result.values_[prepShape.valueCount_];
          auto const deposit = result.values_[prepShape.valueCount_ + 1];
          REQUIRE(std::isfinite(finalEnergy));
          REQUIRE(finalEnergy > result.values_[0]); // Above transport rest mass.
          REQUIRE(finalEnergy < initialEnergy);
          REQUIRE(std::isfinite(deposit));
          REQUIRE(deposit > 0.);
          ++cases[1];
        }
      }
      if (query.operation_ == Operation::Chain) {
        Outcome prepShape, particleShape, lossShape;
        append(prepShape, current::LeptonContinuousPreparation{});
        append(particleShape, current::ParticleState{});
        append(lossShape, current::LeptonContinuousLoss{});
        REQUIRE(result.states_[0] == 0);
        REQUIRE(result.states_[1] == 0);
        REQUIRE(result.states_[prepShape.stateCount_] ==
                static_cast<unsigned>(current::MagneticStepStatus::Success));
        REQUIRE(result.valueCount_ == prepShape.valueCount_ + particleShape.valueCount_ +
                                          3 + lossShape.valueCount_);
        auto const chord =
            result.values_[prepShape.valueCount_ + particleShape.valueCount_ + 1];
        auto const grammage =
            result.values_[prepShape.valueCount_ + particleShape.valueCount_ + 2];
        REQUIRE(std::isfinite(chord));
        REQUIRE(chord > 0.);
        REQUIRE(std::isfinite(grammage));
        REQUIRE(grammage > 0.);
        auto const finalEnergy =
            result.values_[prepShape.valueCount_ + particleShape.valueCount_ + 3];
        auto const deposit =
            result.values_[prepShape.valueCount_ + particleShape.valueCount_ + 4];
        REQUIRE(std::isfinite(finalEnergy));
        REQUIRE(finalEnergy > 0.);
        REQUIRE(std::isfinite(deposit));
        REQUIRE(deposit > 0.);
        REQUIRE(
            result.states_[prepShape.stateCount_ + 2 + particleShape.stateCount_ + 1] ==
            0);
      }
      if (query.operation_ < Operation::Prepare && result.stateCount_ == 2) {
        newton += result.states_[1] & 0xffffu;
        bisection += result.states_[1] >> 16u;
      }
      if (!query.nativeKind_) continue;
      if (query.nativeKind_ == 3) {
        REQUIRE(result.states_[0] == 0); // particleCut
        REQUIRE(result.states_[1] == 0); // fallbackFlag
      } else {
        REQUIRE(result.states_[0] ==
                static_cast<std::uint32_t>(current::TableLookupStatus::Success));
      }
      double const actual =
          query.nativeKind_ == 3 ? result.values_[3] : result.values_[0];
      REQUIRE(std::isfinite(actual));
      REQUIRE(std::isfinite(query.nativeExpected_));
      auto const difference = std::abs(actual - query.nativeExpected_);
      auto& item = native[{query.nativeKind_, query.interaction_.particle_.pid_}];
      ++item.count_;
      item.absolute_ = std::max(item.absolute_, difference);
      item.relative_ = std::max(
          item.relative_, difference / std::max(std::abs(query.nativeExpected_),
                                                std::numeric_limits<double>::min()));
    }
    for (auto count : counts) REQUIRE(count > 0);
    REQUIRE(continuousLimitCases.size() == 4);
    std::array<std::size_t, 2> continuousLimitTotals{};
    for (std::int32_t pid : {11, -11, 13, -13}) {
      auto const found = continuousLimitCases.find(pid);
      REQUIRE(found != continuousLimitCases.end());
      for (std::size_t branch = 0; branch < continuousLimitTotals.size(); ++branch) {
        REQUIRE(found->second[branch] == 6);
        continuousLimitTotals[branch] += found->second[branch];
      }
    }
    REQUIRE(native.size() == 12);
    std::ofstream stream(cache_directory() + "/" + device + "-continuous.json");
    REQUIRE(stream.good());
    stream << std::setprecision(17);
    stream << "{\"queries\":" << data.queries_.size()
           << ",\"same_device_differences\":0,\"operation_counts\":[";
    for (std::size_t index = 0; index < counts.size(); ++index)
      stream << (index ? "," : "") << counts[index];
    stream << "],\"continuous_limit_won_cases\":{\"zero_grammage_fallback\":"
           << continuousLimitTotals[0]
           << ",\"positive_grammage_success\":" << continuousLimitTotals[1]
           << "},\"newton_iterations\":" << newton
           << ",\"bisection_iterations\":" << bisection;
    stream << ",\"table_content_sha256\":\"" << current::to_hex(data.table_.contentHash_)
           << "\"";
    stream << ",\"real_pids\":[11,-11,13,-13],\"fixture\":{"
           << "\"dndx_nodes\":32,\"utility_nodes\":64,"
           << "\"upper_energy_mev\":100000000,\"actual_c8_max_step\":true,"
           << "\"full_c8_do_continuous\":false}";
    stream
        << ",\"native_diagnostic_not_migration_tolerance\":true,\"native_diagnostics\":[";
    bool first = true;
    for (auto const& entry : native) {
      auto const& key = entry.first;
      auto const& value = entry.second;
      stream << (first ? "" : ",") << "{\"kind\":" << key.first
             << ",\"pid\":" << key.second << ",\"count\":" << value.count_
             << ",\"max_absolute\":" << value.absolute_
             << ",\"max_relative\":" << value.relative_ << "}";
      first = false;
    }
    stream << "]}\n";
    REQUIRE(stream.good());
  }
} // namespace continuous_test
