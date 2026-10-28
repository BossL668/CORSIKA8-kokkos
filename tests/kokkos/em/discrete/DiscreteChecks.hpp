/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include "DiscreteTestData.hpp"
#include <catch2/catch_all.hpp>
#include <fstream>
#include <map>
#include <set>

namespace discrete_test {
  inline void exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) REQUIRE(std::signbit(actual) == std::signbit(expected));
    }
  }
  inline void check(Outcome const& actual, Outcome const& wanted) {
    REQUIRE(actual.status_ == wanted.status_);
    REQUIRE(actual.iterations_ == wanted.iterations_);
    REQUIRE(actual.process_ == wanted.process_);
    REQUIRE(actual.component_ == wanted.component_);
    REQUIRE(actual.selected_ == wanted.selected_);
    exact(actual.value_, wanted.value_);
    exact(actual.total_, wanted.total_);
    exact(actual.residual_, wanted.residual_);
  }

  inline void report(std::string const& backend,
                     current::ProposalNativeTableSet const& table,
                     std::vector<Query> const& input,
                     std::vector<Outcome> const& output) {
    REQUIRE(input.size() == output.size());
    constexpr auto count = static_cast<std::size_t>(Operation::Count);
    std::array<std::size_t, count> calls{}, successes{};
    std::map<std::int32_t, std::size_t> splitSuccesses;
    std::set<std::int32_t> selectedParticles;
    std::size_t selected{}, absentAuxiliaryRates{}, newton{}, bisection{};
    for (std::size_t i = 0; i < input.size(); ++i) {
      auto const& query = input[i];
      auto const& result = output[i];
      auto const operation = static_cast<std::size_t>(query.operation_);
      ++calls[operation];
      bool const coordinate =
          query.operation_ == Operation::Encode || query.operation_ == Operation::Decode;
      if (result.status_ == (coordinate ? 1u : 0u)) ++successes[operation];
      newton += result.iterations_ & 0xffffu;
      bisection += result.iterations_ >> 16;
      bool const auxiliary = query.process_ == current::PhotonPairFinalStateProcessId ||
                             query.process_ == current::PhotonMuonPairFinalStateProcessId;
      if (query.operation_ == Operation::Rate && auxiliary) {
        REQUIRE(result.status_ ==
                static_cast<unsigned>(current::TableLookupStatus::ColumnNotFound));
        ++absentAuxiliaryRates;
      }
      if (query.operation_ == Operation::Loss && auxiliary && result.status_ == 0) {
        ++splitSuccesses[query.process_];
        REQUIRE(std::isfinite(result.value_));
        REQUIRE(result.value_ >= 0.);
        REQUIRE(result.value_ <= 1.);
      }
      if (query.operation_ == Operation::Select && result.selected_) {
        REQUIRE(result.selected_ == 1);
        REQUIRE(result.status_ == 0);
        REQUIRE(result.total_ > 0.);
        REQUIRE(result.residual_ >= 0.);
        REQUIRE(result.residual_ <= 1.);
        REQUIRE(result.process_ < 2000000000);
        selectedParticles.insert(query.pid_);
        ++selected;
      }
    }
    for (std::size_t i = 0; i < count; ++i) {
      INFO("Discrete operation " << i);
      REQUIRE(calls[i] > 0);
      REQUIRE(successes[i] > 0);
    }
    REQUIRE(selectedParticles == std::set<std::int32_t>{-13, -11, 11, 13, 22});
    REQUIRE(splitSuccesses[current::PhotonPairFinalStateProcessId] > 0);
    REQUIRE(splitSuccesses[current::PhotonMuonPairFinalStateProcessId] > 0);
    REQUIRE(absentAuxiliaryRates > 0);
    REQUIRE(newton > 0);
    REQUIRE(bisection > 0);
    std::ofstream file("discrete-" + backend + "-report.json");
    REQUIRE(file.good());
    file << "{\n  \"queries\": " << input.size()
         << ",\n  \"rate_columns\": " << table.dndxColumns_.size()
         << ",\n  \"selected\": " << selected
         << ",\n  \"auxiliary_rate_columns_absent\": " << absentAuxiliaryRates
         << ",\n  \"electron_pair_split_successes\": "
         << splitSuccesses[current::PhotonPairFinalStateProcessId]
         << ",\n  \"muon_pair_split_successes\": "
         << splitSuccesses[current::PhotonMuonPairFinalStateProcessId]
         << ",\n  \"newton_iterations\": " << newton
         << ",\n  \"bisection_iterations\": " << bisection << ",\n  \"operations\": [";
    for (std::size_t i = 0; i < count; ++i)
      file << (i ? "," : "") << "{\"calls\":" << calls[i]
           << ",\"successes\":" << successes[i] << "}";
    file << "]\n}\n";
    REQUIRE(file.good());
  }
} // namespace discrete_test
