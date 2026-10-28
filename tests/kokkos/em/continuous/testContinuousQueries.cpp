/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "ContinuousChecks.hpp"

TEST_CASE("Continuous queries match fixed production and native diagnostics",
          "[continuous]") {
  using namespace continuous_test;
  auto data = make_dataset();
  populate_records(data);
  auto const view = current::make_proposal_native_host_view(data.table_);
  native_query_test::HostReference const owner(data.table_);
  auto const old = owner.view(data.table_);
  std::vector<Outcome> outcomes;
  for (std::size_t index = 0; index < data.queries_.size(); ++index) {
    INFO("Continuous query " << index << ", operation "
                             << static_cast<unsigned>(data.queries_[index].operation_));
    auto const actual = evaluate(view, data.queries_[index]);
    check(actual, expected(old, data.queries_[index]));
    outcomes.push_back(actual);
  }
  report("host", data, outcomes);
}
