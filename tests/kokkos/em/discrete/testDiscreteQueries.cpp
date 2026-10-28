/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "DiscreteChecks.hpp"
#include "../tables/NativeTableFixture.hpp"

TEST_CASE("Discrete host queries match fixed production", "[discrete]") {
  using namespace discrete_test;
  auto const table = current::test::make_native_tables();
  auto const hash = current::calculate_proposal_native_hash(table);
  auto const view = current::make_proposal_native_host_view(table);
  native_query_test::HostReference const owner(table);
  auto const old = owner.view(table);
  auto const input = queries(table);
  std::vector<Outcome> output;
  for (std::size_t i = 0; i < input.size(); ++i) {
    INFO("Host query " << i << ", operation "
                       << static_cast<unsigned>(input[i].operation_) << ", PID "
                       << input[i].pid_ << ", process " << input[i].process_
                       << ", energy " << input[i].energy_ << ", argument "
                       << input[i].argument_);
    auto const actual = evaluate(view, input[i]);
    check(actual, expected(old, input[i]));
    output.push_back(actual);
  }
  REQUIRE(current::calculate_proposal_native_hash(table) == hash);
  report("host", table, input, output);
}
