/*
 * (c) Copyright 2023 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <catch2/catch_all.hpp>
#include <random>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <rng_impl.hpp> //from modules/common
IMPLEMENT_RNG(foobar)

TEST_CASE("uniform distribution") {
  // setup mockup RNG
  std::minstd_rand rng;
  std::uniform_real_distribution<double> distrib;

  ::foobar::set_rng_function([&](double* dest, std::size_t N) {
    while (N) { dest[--N] = distrib(rng); }
  });

  // test
  using namespace boost::accumulators;
  accumulator_set<double, stats<tag::mean, tag::moment<2>, tag::min, tag::max>> acc;

  for (int i = 0; i < 1'000'000; ++i) {
    auto const u = ::draw_std_rnd();
    acc(u);
  }

  using namespace Catch;

  CHECK(mean(acc) == Approx(.5).margin(0.001));
  CHECK(moment<2>(acc) == Approx(1. / 3.).margin(0.01)); // NB: not shifted by mean
  CHECK(min(acc) == Approx(0.).margin(0.001));
  CHECK(max(acc) == Approx(1.).margin(0.001));
}
