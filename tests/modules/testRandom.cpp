/*
 * (c) Copyright 2023 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <catch2/catch_all.hpp>
#ifdef WITH_FLUKA
#include <corsika/modules/FLUKA.hpp>
#endif
#include <corsika/modules/Epos.hpp>
#include <corsika/modules/Sibyll.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/modules/QGSJetII.hpp>
#include <corsika/modules/Sophia.hpp>
#include <corsika/modules/UrQMD.hpp>

using Catch::Approx;

static std::size_t first{};
void dummy_rng_func(double* dest, std::size_t N) {
  double constexpr factor = 1.e-6;
  for (std::size_t i = 0; i < N; ++i) { dest[i] = (first + i) * factor; }
}

TEST_CASE("set_rng") {
  first = 1;
  int dummy{};
  set_sophia_rng_function(dummy_rng_func);
  // call Sophia RNG, which uses injected function
  CHECK(rndm_(dummy) == 1 * 1.e-6);
  CHECK(rndm_(dummy) == 2 * 1.e-6);

  first = 20;
  set_epos_rng_function(dummy_rng_func);
  CHECK(epos::rangen_() == static_cast<float>(20 * 1.e-6));
  CHECK(epos::drangen_() == 21 * 1.e-6);

  first = 40;
  set_sibyll_rng_function(dummy_rng_func);
  CHECK(s_rndm_(dummy) == 40 * 1.e-6);
  CHECK(s_rndm_(dummy) == 41 * 1.e-6);

  first = 60;
  QGSJetII04::set_rng_function(dummy_rng_func);
  CHECK(QGSJetII04::qgran_(dummy) == 60 * 1.e-6);
  CHECK(QGSJetII04::qgran_(dummy) == 61 * 1.e-6);

  first = 80;
  set_urqmd_rng_function(dummy_rng_func);
  CHECK(urqmd::ranf_(dummy) == 80 * 1.e-6);
  CHECK(urqmd::ranf_(dummy) == 81 * 1.e-6);

  // Sophia is still untouched
  CHECK(rndm_(dummy) == 3 * 1.e-6);

#ifdef WITH_FLUKA
  first = 100;
  set_fluka_rng_function(dummy_rng_func);
  CHECK(fluka::flrndm_(dummy) == 100 * 1.e-6);
  CHECK(fluka::flrndm_(dummy) == 101 * 1.e-6);
#endif
}

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

  CHECK(mean(acc) == Approx(.5).margin(0.001));
  CHECK(moment<2>(acc) == Approx(1. / 3.).margin(0.01)); // NB: not shifted by mean
  CHECK(min(acc) == Approx(0.).margin(0.001));
  CHECK(max(acc) == Approx(1.).margin(0.001));
}
