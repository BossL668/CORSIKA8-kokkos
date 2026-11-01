/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/detail/kokkos/egs4/Egs4Radiative.inl>
#include <corsika/kokkos/em/random/Philox.hpp>
#include <catch2/catch_all.hpp>

namespace {
  namespace eg = corsika::kokkos::egs4;

  struct CounterStream {
    corsika::kokkos::RandomNumberKey key_;
    KOKKOS_INLINE_FUNCTION bool next(double& value) {
      value = corsika::kokkos::uniform_open01(key_);
      ++key_.drawId_;
      return true;
    }
  };

  struct Sample {
    eg::Secondaries secondaries_;
    std::uint64_t draws_{};
  };
} // namespace

TEST_CASE("EGS4 bremsstrahlung continues past a diagnostic trial limit", "[egs4]") {
  Kokkos::ScopeGuard kokkos;
  auto const tables = eg::read_air_tables(CORSIKA_EGS4_TEST_TABLE);
  auto const radiative = eg::radiative_table(tables);
  constexpr double mass = .51099895;
  bool const incident = tables.medium_.ap_ == .25;
  double const energy = incident ? .76239943104671881
                                : std::max(tables.medium_.ae_, mass + tables.medium_.ap_) + .001;
  eg::RadiativeInput const input{energy, mass, .0002, {0., 0., 1.}, 11};
  CounterStream random{{94010001, 38, 49731890, 0, 0x43474534U, 7}};
  auto boundedRandom = random;
  auto const bounded = eg::sample_bremsstrahlung(radiative, input, boundedRandom, 1);
  if (incident) {
    REQUIRE(bounded.status_ == eg::InteractionStatus::trial_limit);
    auto previousRandom = random;
    auto const previous = eg::sample_bremsstrahlung(radiative, input, previousRandom, 4096);
    REQUIRE(previous.status_ == eg::InteractionStatus::trial_limit);
  }
  auto const initial = random;
  auto const reference = eg::sample_bremsstrahlung(radiative, input, random);
  REQUIRE(reference.status_ == eg::InteractionStatus::success);
  REQUIRE(reference.count_ == 2);
  REQUIRE(reference.particle_[0].energyMeV_ + reference.particle_[1].energyMeV_ ==
          Catch::Approx(energy).margin(1.e-15));
  if (incident) REQUIRE(reference.trials_ == 5106);
  auto dispatchRandom = initial;
  auto const dispatched = eg::sample_radiative(radiative, input, dispatchRandom);
  REQUIRE(dispatched.status_ == eg::InteractionStatus::success);
  REQUIRE(dispatched.trials_ == reference.trials_);
  REQUIRE(dispatchRandom.key_.drawId_ == random.key_.drawId_);

  Kokkos::View<Sample*> result("egs4_brems_tail", 1);
  Kokkos::parallel_for("egs4_brems_tail", 1, KOKKOS_LAMBDA(int i) {
    auto deviceRandom = initial;
    result(i).secondaries_ = eg::sample_bremsstrahlung(radiative, input, deviceRandom);
    result(i).draws_ = deviceRandom.key_.drawId_;
  });
  auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  REQUIRE(host(0).secondaries_.status_ == eg::InteractionStatus::success);
  REQUIRE(host(0).secondaries_.trials_ == reference.trials_);
  REQUIRE(host(0).draws_ == random.key_.drawId_);
  for (int i = 0; i < 2; ++i) {
    REQUIRE(host(0).secondaries_.particle_[i].pdg_ == reference.particle_[i].pdg_);
    REQUIRE(host(0).secondaries_.particle_[i].energyMeV_ ==
            Catch::Approx(reference.particle_[i].energyMeV_).margin(1.e-14));
  }
}
