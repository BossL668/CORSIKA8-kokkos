/*
 * (c) Copyright 2025 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/modules/QGSJetIII.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/random/RNGManager.hpp>

#include <SetupTestEnvironment.hpp>

#include <catch2/catch_all.hpp>

#include <string>
#include <cstdlib>

using namespace corsika;
using Catch::Approx;

using DummyEnvironmentInterface = IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>;
using DummyEnvironment = Environment<DummyEnvironmentInterface>;

template <typename TStackView>
auto sumCharge(TStackView const& view) {
  int totalCharge = 0;
  for (auto const& p : view) { totalCharge += get_charge_number(p.getPID()); }
  return totalCharge;
}

template <typename TStackView>
auto sumMomentum(TStackView const& view, CoordinateSystemPtr const& vCS) {
  Vector<hepenergy_d> sum{vCS, 0_eV, 0_eV, 0_eV};
  for (auto const& p : view) { sum += p.getMomentum(); }
  return sum;
}

TEST_CASE("QgsjetIII", "[processes]") {

  logging::set_level(logging::level::info);
  RNGManager<>::getInstance().registerRandomStream("qgsjetIII");

  SECTION("Corsika -> QgsjetIII") {
    CHECK(corsika::qgsjetIII::convertToQgsjetIII(PiMinus::code) ==
          corsika::qgsjetIII::QgsjetIIICode::PiMinus);
    CHECK(corsika::qgsjetIII::convertToQgsjetIIIRaw(Proton::code) == 2);
  }

  SECTION("QgsjetIII -> Corsika") {
    CHECK(Code::PiPlus == corsika::qgsjetIII::convertFromQgsjetIII(
                              corsika::qgsjetIII::QgsjetIIICode::PiPlus));
    CHECK_THROWS(corsika::qgsjetIII::convertFromQgsjetIII(
        corsika::qgsjetIII::QgsjetIIICode::Unknown));
  }

  SECTION("Corsika -> QgsjetIII") {
    CHECK(corsika::qgsjetIII::convertToQgsjetIII(Code::PiMinus) ==
          corsika::qgsjetIII::QgsjetIIICode::PiMinus);
    CHECK(corsika::qgsjetIII::convertToQgsjetIIIRaw(Code::Proton) == 2);
  }

  SECTION("canInteractInQgsjetIII") {

    CHECK(corsika::qgsjetIII::canInteract(Code::Proton));
    CHECK(corsika::qgsjetIII::canInteract(Code::KPlus));
    CHECK(corsika::qgsjetIII::canInteract(Code::Nucleus));
    CHECK(corsika::qgsjetIII::canInteract(Code::Rho0));
    // CHECK(corsika::qgsjetIII::canInteract(Helium::getCode()));

    CHECK_FALSE(corsika::qgsjetIII::canInteract(Code::EtaC));
    CHECK_FALSE(corsika::qgsjetIII::canInteract(Code::SigmaC_2455_0));
  }

  SECTION("cross-section type") {

    CHECK(corsika::qgsjetIII::getQgsjetIIIXSCode(Code::Neutron) ==
          corsika::qgsjetIII::QgsjetIIIXSClass::Baryons);
    CHECK(corsika::qgsjetIII::getQgsjetIIIXSCode(Code::K0Long) ==
          corsika::qgsjetIII::QgsjetIIIXSClass::Kaons);
    CHECK(corsika::qgsjetIII::getQgsjetIIIXSCode(Code::Proton) ==
          corsika::qgsjetIII::QgsjetIIIXSClass::Baryons);
    CHECK(corsika::qgsjetIII::getQgsjetIIIXSCode(Code::PiMinus) ==
          corsika::qgsjetIII::QgsjetIIIXSClass::LightMesons);
    CHECK(corsika::qgsjetIII::getQgsjetIIIXSCode(Code::Helium) ==
          corsika::qgsjetIII::QgsjetIIIXSClass::Baryons);
  }

  SECTION("valid") {

    corsika::qgsjetIII::InteractionModel model;

    CHECK_FALSE(model.isValid(Code::Electron, Code::Proton, 1_TeV));
    CHECK_FALSE(model.isValid(Code::Proton, Code::Electron, 1_TeV));
    CHECK_FALSE(model.isValid(Code::Proton, Code::Proton, 1_GeV));

    CHECK(model.isValid(Code::Proton, Code::Helium, 1_TeV));
    CHECK_FALSE(model.isValid(Code::Proton, Code::Helium, 1_GeV));
  }
}

#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>

#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupTrajectory.hpp>

#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/NuclearComposition.hpp>

#include <SetupTestEnvironment.hpp>
#include <SetupTestStack.hpp>

TEST_CASE("QgsjetIIIInterface", "interaction,processes") {

  logging::set_level(logging::level::info);

  RNGManager<>::getInstance().registerRandomStream("qgsjetIII");

  auto [env, csPtr, nodePtr] = setup::testing::setup_environment(Code::Oxygen);
  auto const& cs = *csPtr;
  [[maybe_unused]] auto const& env_dummy = env;
  [[maybe_unused]] auto const& node_dummy = nodePtr;

  corsika::qgsjetIII::InteractionModel model;

  SECTION("cross-sections") {
    auto projCode = GENERATE(Code::PiPlus, Code::Proton, Code::K0Long, Code::Iron,
                             Code::Nitrogen, Code::Helium);
    auto targetCode = GENERATE(Code::Oxygen, Code::Nitrogen);
    auto projEnergy = GENERATE(1_PeV, 1e18_eV);

    auto momMagnitude = calculate_momentum(projEnergy, get_mass(projCode));
    MomentumVector const projMomentum{*csPtr, 0_eV, momMagnitude, 0_eV};

    REQUIRE(model.getCrossSection(
                projCode, targetCode, FourMomentum{projEnergy, projMomentum},
                FourMomentum{get_mass(targetCode), {*csPtr, 0_eV, 0_eV, 0_eV}}) /
                1_mb >
            0);
  }

  SECTION("InteractionInterface") {
    auto projCode = GENERATE(Code::PiPlus, Code::Proton, Code::K0Long, Code::Iron,
                             Code::Nitrogen, Code::Helium);
    auto targetCode = GENERATE(Code::Oxygen, Code::Nitrogen);
    auto projMomentum = is_nucleus(projCode)
                            ? GENERATE(1e20_eV, 1_PeV, 100_TeV, 10_TeV)
                            : GENERATE(1e20_eV, 1_PeV, 100_TeV, 10_TeV, 1_TeV, 100_GeV);

    auto [stackPtr, secViewPtr] = setup::testing::setup_stack(
        Code::Proton, projMomentum, (DummyEnvironment::BaseNodeType* const)nodePtr,
        *csPtr);
    test::StackView& view = *(secViewPtr.get());
    auto projectile = secViewPtr->getProjectile();
    auto const projectileMomentum = projectile.getMomentum();

    model.doInteraction(
        view, projCode, targetCode,
        FourMomentum{calculate_total_energy(projMomentum, get_mass(projCode)),
                     projectileMomentum},
        FourMomentum{get_mass(targetCode), MomentumVector{cs, {0_eV, 0_eV, 0_eV}}});

    /* **********************************
     As it turned out already twice (#291 and #307), the detailed output of
     QgsjetIII event generation depends on the gfortran version used. This is not reliable
     and cannot be tested in a unit test here. One related problem was already found
    (#291) and is realted to undefined behaviour in the evaluation of functions in logical
     expressions. It is not clear if #307 is the same issue.

     CHECK(view.getSize() == 14);
     CHECK(sumCharge(view) == 2);

     bit us again, lets set the size to max. size of stack in qgsjet
    *********************************** */
    auto const secMomSum = sumMomentum(view, projectileMomentum.getCoordinateSystem());
    CHECK((secMomSum - projectileMomentum).getNorm() / projectileMomentum.getNorm() ==
          Approx(0).margin(1e-2));
    CHECK(view.getSize() == Approx(95000).margin(94998));
  }

  SECTION("InteractionInterface Nuclei") {
    HEPEnergyType const P0 = 20100_GeV;
    MomentumVector const plab = MomentumVector(cs, {P0, 0_eV, 0_eV});
    Code const pid = get_nucleus_code(60, 30);
    auto [stackPtr, secViewPtr] = setup::testing::setup_stack(
        pid, P0, (DummyEnvironment::BaseNodeType* const)nodePtr, *csPtr);
    test::StackView& view = *(secViewPtr.get());

    HEPEnergyType const Elab = sqrt(static_pow<2>(P0) + static_pow<2>(get_mass(pid)));
    FourMomentum const projectileP4(Elab, plab);
    FourMomentum const targetP4(Oxygen::mass, MomentumVector(cs, {0_eV, 0_eV, 0_eV}));
    view.clear();

    model.doInteraction(view, pid, Code::Oxygen, projectileP4,
                        targetP4); // this also should produce some fragments
    CHECK(view.getSize() ==
          Approx(95000).margin(94998)); // this is not physics validation
    int countFragments = 0;
    for (auto const& sec : view) { countFragments += (is_nucleus(sec.getPID())); }
    CHECK(countFragments > 0); // this is not physics validation
  }

  SECTION("Heavy nuclei") {

    auto [stackPtr, secViewPtr] = setup::testing::setup_stack(
        get_nucleus_code(1000, 1000), 1100_GeV,
        (DummyEnvironment::BaseNodeType* const)nodePtr, *csPtr);
    test::StackView& view = *(secViewPtr.get());
    auto projectile = secViewPtr->getProjectile();
    auto const projectileMomentum = projectile.getMomentum();

    FourMomentum const aP4(100_GeV, {cs, 99_GeV, 0_GeV, 0_GeV});
    FourMomentum const bP4(1_TeV, {cs, 0.9_TeV, 0_GeV, 0_GeV});

    CHECK(model.getCrossSection(get_nucleus_code(10, 5), get_nucleus_code(1000, 500), aP4,
                                bP4) == 0_mb);
    CHECK(model.getCrossSection(Code::Nucleus, Code::Nucleus, aP4, bP4) == 0_mb);
    CHECK_THROWS(
        model.doInteraction(view, get_nucleus_code(1000, 500), Code::Oxygen, aP4, bP4));
  }

  SECTION("Allowed Particles") {
    HEPEnergyType const projMomentum = 500_GeV;
    // pi0 is internally converted into pi+/pi-
    // rho0 is internally converted into pi-/pi+
    // (anti-)Lambda is internally converted into (anti-)neutron
    auto pid = GENERATE(Code::Pi0, Code::Rho0, Code::Lambda, Code::LambdaBar);
    auto [stackPtr, secViewPtr] = setup::testing::setup_stack(
        Code::Proton, projMomentum, (DummyEnvironment::BaseNodeType* const)nodePtr,
        *csPtr);
    test::StackView& view = *(secViewPtr.get());
    auto projectile = secViewPtr->getProjectile();
    auto const projectileMomentum = projectile.getMomentum();

    CHECK_NOTHROW(model.doInteraction(
        view, pid, Code::Oxygen,
        FourMomentum{calculate_total_energy(projMomentum, get_mass(pid)),
                     projectileMomentum},
        FourMomentum{get_mass(Code::Oxygen), MomentumVector{cs, {0_eV, 0_eV, 0_eV}}}));

    CHECK(view.getSize() == Approx(70).margin(69));
  }
}
