/*
 * (c) Copyright 2019 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/modules/Epos.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/random/RNGManager.hpp>

#include <SetupTestEnvironment.hpp>

#include <catch2/catch_all.hpp>
#include <tuple>

using namespace corsika;
using namespace corsika::epos;
using Catch::Approx;

using DummyEnvironmentInterface = IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>;
using DummyEnvironment = Environment<DummyEnvironmentInterface>;

TEST_CASE("EposBasics", "module,process") {

  logging::set_level(logging::level::info);

  SECTION("epos -> corsika") {
    CHECK(Code::Electron ==
          corsika::epos::convertFromEpos(corsika::epos::EposCode::Electron));
    CHECK(Code::Proton ==
          corsika::epos::convertFromEpos(corsika::epos::EposCode::Proton));
    CHECK_THROWS(corsika::epos::convertFromEpos(corsika::epos::EposCode::Unknown));
  }

  SECTION("corsika -> epos") {
    CHECK(corsika::epos::convertToEpos(Electron::code) ==
          corsika::epos::EposCode::Electron);
    // check if particle code is correct for common particles that interact (secret epos
    // knowledge)
    CHECK(corsika::epos::convertToEposRaw(Proton::code) == 1120);
    CHECK(corsika::epos::convertToEposRaw(PiPlus::code) == 120);
    CHECK(corsika::epos::convertToEposRaw(KPlus::code) == 130);
  }

  SECTION("canInteractInEpos") {
    CHECK(corsika::epos::canInteract(Code::Proton));
    CHECK(corsika::epos::canInteract(Code::Rho0));
    CHECK_FALSE(corsika::epos::canInteract(Code::Electron));
    CHECK(corsika::epos::canInteract(Code::Nucleus));
    CHECK(corsika::epos::canInteract(Code::Helium));
  }

  SECTION("cross-section type") {
    CHECK(corsika::epos::getEposXSCode(Code::Electron) == 0);
    CHECK(corsika::epos::getEposXSCode(Code::K0Long) == 3);
    CHECK(corsika::epos::getEposXSCode(Code::SigmaPlus) == 2);
    CHECK(corsika::epos::getEposXSCode(Code::KMinus) == 3);
    CHECK(corsika::epos::getEposXSCode(Code::PiMinus) == 1);
    CHECK(corsika::epos::getEposXSCode(Code::Proton) == 2);
    CHECK(corsika::epos::getEposXSCode(Code::Helium) == 2);
    CHECK(corsika::epos::getEposXSCode(Code::Nucleus) == 2);
  }

  SECTION("epos mass") {
    CHECK_FALSE(corsika::epos::getEposMass(Code::Electron) / 1_GeV == Approx(0));
    CHECK_THROWS(corsika::epos::getEposMass(Code::Unknown));
  }
}

#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>

#include <corsika/framework/core/PhysicalUnits.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>

#include <SetupTestEnvironment.hpp>
#include <SetupTestStack.hpp>

#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/media/UniformMagneticField.hpp>

template <typename TStackView>
auto sumMomentum(TStackView const& view, CoordinateSystemPtr const& vCS) {
  Vector<hepenergy_d> sum{vCS, 0_eV, 0_eV, 0_eV};
  for (auto const& p : view) { sum += p.getMomentum(); }
  return sum;
}

auto sqs2elab(HEPEnergyType const sqs, HEPEnergyType const ma, HEPEnergyType const mb) {
  return (sqs * sqs - ma * ma - mb * mb) / 2. / mb;
}

TEST_CASE("Epos", "modules") {

  logging::set_level(logging::level::info);

  RNGManager<>::getInstance().registerRandomStream("epos");
  InteractionModel model;

  auto [env, csPtr, nodePtr] = setup::testing::setup_environment(Code::Oxygen);
  auto const& cs = *csPtr;
  [[maybe_unused]] auto const& env_dummy = env;

  SECTION("InteractionInterface - isValid") {

    CHECK_FALSE(model.isValid(Code::Proton, Code::Electron, 100_GeV));
    CHECK(model.isValid(Code::Proton, Code::Hydrogen, 100_GeV));
    CHECK(model.isValid(Code::Proton, Code::Helium, 100_GeV));
    CHECK_FALSE(model.isValid(Code::Proton, Code::Iron, 10_EeV));
    CHECK_FALSE(model.isValid(Code::Proton, get_nucleus_code(240, 120), 10_EeV));
    CHECK(model.isValid(Code::Proton, Code::Oxygen, 100_GeV));
  }

  SECTION("InteractionInterface - getCrossSectionInelEla") {

    // hydrogen target == proton target == neutron target
    auto const [xs_prod_pp, xs_ela_pp] = model.getCrossSectionInelEla(
        Code::Proton, Code::Proton,
        {sqrt(static_pow<2>(100_GeV) + static_pow<2>(Proton::mass)),
         {cs, 100_GeV, 0_GeV, 0_GeV}},
        {Proton::mass, {cs, 0_GeV, 0_GeV, 0_GeV}});

    auto const [xs_prod_pn, xs_ela_pn] = model.getCrossSectionInelEla(
        Code::Proton, Code::Neutron,
        {sqrt(static_pow<2>(100_GeV) + static_pow<2>(Proton::mass)),
         {cs, 100_GeV, 0_GeV, 0_GeV}},
        {Neutron::mass, {cs, 0_GeV, 0_GeV, 0_GeV}});

    auto const [xs_prod_pHydrogen, xs_ela_pHydrogen] = model.getCrossSectionInelEla(
        Code::Proton, Code::Hydrogen,
        {sqrt(static_pow<2>(100_GeV) + static_pow<2>(Proton::mass)),
         {cs, 100_GeV, 0_GeV, 0_GeV}},
        {Hydrogen::mass, {cs, 0_GeV, 0_GeV, 0_GeV}});

    CHECK(xs_prod_pp == xs_prod_pHydrogen);
    CHECK(xs_prod_pp == xs_prod_pn);
    CHECK(xs_ela_pp == xs_ela_pHydrogen);
    CHECK(xs_ela_pn == xs_ela_pHydrogen);

    // invalid system
    auto const [xs_prod_0, xs_ela_0] = model.getCrossSectionInelEla(
        Code::Electron, Code::Electron,
        {sqrt(static_pow<2>(100_GeV) + static_pow<2>(Electron::mass)),
         {cs, 100_GeV, 0_GeV, 0_GeV}},
        {Electron::mass, {cs, 0_GeV, 0_GeV, 0_GeV}});
    CHECK(xs_prod_0 / 1_mb == Approx(0));
    CHECK(xs_ela_0 / 1_mb == Approx(0));
  }

  SECTION("InteractionModelInterface - hadron cross sections") {

    // p-p at 7TeV around 70mb according to LHC
    auto const xs_prod = model.getCrossSection(
        Code::Proton, Code::Proton,
        {3.5_TeV,
         {cs, sqrt(static_pow<2>(3.5_TeV) - static_pow<2>(Proton::mass)), 0_GeV, 0_GeV}},
        {3.5_TeV,
         {cs, -sqrt(static_pow<2>(3.5_TeV) - static_pow<2>(Proton::mass)), 0_GeV,
          0_GeV}});
    CHECK(xs_prod / 1_mb == Approx(70.7).margin(2.1));

    // pi-n at 7TeV
    auto const xs_prod1 = model.getCrossSection(
        Code::PiPlus, Code::Neutron,
        {3.5_TeV,
         {cs, sqrt(static_pow<2>(3.5_TeV) - static_pow<2>(PiPlus::mass)), 0_GeV, 0_GeV}},
        {3.5_TeV,
         {cs, -sqrt(static_pow<2>(3.5_TeV) - static_pow<2>(Neutron::mass)), 0_GeV,
          0_GeV}});
    CHECK(xs_prod1 / 1_mb == Approx(52.7).margin(2.1));

    // k-p at 7TeV
    auto const xs_prod2 = model.getCrossSection(
        Code::KPlus, Code::Proton,
        {3.5_TeV,
         {cs, sqrt(static_pow<2>(3.5_TeV) - static_pow<2>(KPlus::mass)), 0_GeV, 0_GeV}},
        {3.5_TeV,
         {cs, -sqrt(static_pow<2>(3.5_TeV) - static_pow<2>(Proton::mass)), 0_GeV,
          0_GeV}});
    CHECK(xs_prod2 / 1_mb == Approx(45.7).margin(2.1));
  }

  SECTION("InteractionInterface - nuclear cross sections") {

    auto const xs_prod = model.getCrossSection(
        Code::Proton, Code::Oxygen,
        {100_GeV,
         {cs, sqrt(static_pow<2>(100_GeV) - static_pow<2>(Proton::mass)), 0_GeV, 0_GeV}},
        {Oxygen::mass, {cs, 0_GeV, 0_GeV, 0_GeV}});
    CHECK(xs_prod / 1_mb == Approx(287.0).margin(5.1));

    auto const xs_prod2 = model.getCrossSection(
        Code::Nitrogen, Code::Oxygen,
        {400_GeV,
         {cs, sqrt(static_pow<2>(400_GeV) - static_pow<2>(Nitrogen::mass)), 0_GeV,
          0_GeV}},
        {Oxygen::mass, {cs, 0_GeV, 0_GeV, 0_GeV}});
    CHECK(xs_prod2 / 1_mb == Approx(1076.7).margin(3.1));
  }

  SECTION("InteractionInterface - invalid") {
    Code const pid = Code::Electron;
    HEPEnergyType const P0 = 10_TeV;
    auto [stack, viewPtr] = setup::testing::setup_stack(
        pid, P0, (DummyEnvironment::BaseNodeType* const)nodePtr, cs);
    test::StackView& view = *viewPtr;
    CHECK_THROWS(model.doInteraction(
        view, pid, Code::Oxygen,
        {sqrt(static_pow<2>(P0) + static_pow<2>(get_mass(pid))), {cs, P0, 0_GeV, 0_GeV}},
        {Oxygen::mass, {cs, 0_GeV, 0_GeV, 0_GeV}}));
  }

  SECTION("InteractionInterface - valid projectile target combinations") {

    HEPMomentumType const P0 = 10_TeV;
    Code const projectileId =
        GENERATE(Code::Proton, Code::PiPlus, Code::KPlus, Code::Iron);
    Code const targetId = GENERATE(Code::Proton, Code::Neutron, Code::Nitrogen);

    auto [stack, viewPtr] = setup::testing::setup_stack(
        projectileId, P0, (DummyEnvironment::BaseNodeType* const)nodePtr, cs);
    test::StackView& view = *viewPtr;

    // @todo This is very obscure since it fails for -O2, but for both clang and gcc ???
    model.doInteraction(
        view, projectileId, targetId,
        {calculate_total_energy(P0, get_mass(projectileId)), {cs, {P0, 0_eV, 0_eV}}},
        {get_mass(targetId), {cs, 0_GeV, 0_GeV, 0_GeV}});

    //  simply check if stack is not empty after the event. Energy and momentum
    //  conservation will be tested elsewhere
    CHECK(view.getSize() > 0);
  }

  SECTION("Decay config") {
    logging::set_level(logging::level::debug);

    InteractionModel model(std::set<Code>{Code::Proton, Code::PiPlus, Code::KPlus});
  }
}
