/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <SetupTestEnvironment.hpp>

#include <corsika/media/Environment.hpp>
#include <corsika/media/LayeredSphericalAtmosphereBuilder.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>

#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>

#include <corsika/modules/CONEX.hpp>
#include <corsika/modules/Sibyll.hpp>
#include <corsika/modules/writers/WriterOff.hpp>

#include <corsika/framework/random/RNGManager.hpp>

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/utility/CorsikaFenv.hpp>

#include <corsika/setup/SetupC7trackedParticles.hpp>

#include <catch2/catch_all.hpp>

using namespace corsika;
using Catch::Approx;

using DummyEnvironmentInterface = IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>;
using DummyEnvironment = Environment<DummyEnvironmentInterface>;

const std::string refDataDir = std::string(REFDATADIR); // from cmake

template <typename T>
using MExtraEnvirnoment = MediumPropertyModel<UniformMagneticField<T>>;

struct DummyStack {};

extern "C" void inisibyll_();

TEST_CASE("CONEX") {

  logging::set_level(logging::level::info);

  RNGManager<>::getInstance().registerRandomStream("conex");
  RNGManager<>::getInstance().registerRandomStream("sibyll");
  RNGManager<>::getInstance().registerRandomStream("epos");

  feenableexcept(FE_INVALID);

  // setup environment, geometry
  DummyEnvironment env;
  CoordinateSystemPtr const& rootCS = env.getCoordinateSystem();
  Point const center{rootCS, 0_m, 0_m, 0_m};

  auto builder = make_layered_spherical_atmosphere_builder<
      DummyEnvironmentInterface, MExtraEnvirnoment>::create(center,
                                                            corsika::conex::earthRadius,
                                                            Medium::AirDry1Atm,
                                                            Vector{rootCS, 0_T, 50_mT,
                                                                   0_T});

  builder.setNuclearComposition(
      {{Code::Nitrogen, Code::Oxygen},
       {0.7847, 1. - 0.7847}}); // values taken from AIRES manual, Ar removed for now

  builder.addExponentialLayer(1222.6562_g / (1_cm * 1_cm), 994186.38_cm, 4_km);
  builder.addExponentialLayer(1144.9069_g / (1_cm * 1_cm), 878153.55_cm, 10_km);
  builder.addExponentialLayer(1305.5948_g / (1_cm * 1_cm), 636143.04_cm, 40_km);
  builder.addExponentialLayer(540.1778_g / (1_cm * 1_cm), 772170.16_cm, 100_km);
  builder.addLinearLayer(1_g / (1_cm * 1_cm), 1e9_cm, 112.8_km);

  builder.assemble(env);

  const HEPEnergyType E0 = 1_PeV;
  double thetaDeg = 60.;
  auto const thetaRad = thetaDeg / 180. * M_PI;

  auto const observationHeight = 1.4_km + corsika::conex::earthRadius;
  auto const injectionHeight = 112.75_km + corsika::conex::earthRadius;
  auto const t = -observationHeight * cos(thetaRad) +
                 sqrt(-static_pow<2>(sin(thetaRad) * observationHeight) +
                      static_pow<2>(injectionHeight));
  Point const showerCore{rootCS, 0_m, 0_m, observationHeight};
  Point const injectionPos =
      showerCore + DirectionVector{rootCS, {-sin(thetaRad), 0, cos(thetaRad)}} * t;

  ShowerAxis const showerAxis{injectionPos, (showerCore - injectionPos) * 1.02, env};

  inisibyll_();

  EnergyLossWriter<WriterOff> w1(showerAxis);
  LongitudinalWriter<WriterOff> w2(showerAxis);
  CONEXhybrid<decltype(w1), decltype(w2)> conex(center, showerAxis, t, injectionHeight,
                                                E0, get_PDG(Code::Proton), w1, w2);
  // initialize writers
  w1.startOfLibrary("test");
  w1.startOfShower(0);
  w2.startOfLibrary("test");
  w2.startOfShower(0);
  // init conex
  conex.initCascadeEquations();

  HEPEnergyType const Eem{1_PeV};
  auto const momentum = showerAxis.getDirection() * Eem;

  auto const emPosition = showerCore + showerAxis.getDirection() * (-20_km);

  CORSIKA_LOG_DEBUG("position injection: {} {}",
                    injectionPos.getCoordinates(conex.getObserverCS()),
                    injectionPos.getCoordinates(rootCS));
  CORSIKA_LOG_DEBUG("position core: {} {}",
                    showerCore.getCoordinates(conex.getObserverCS()),
                    showerCore.getCoordinates(rootCS));
  CORSIKA_LOG_DEBUG("position EM: {} {}",
                    emPosition.getCoordinates(conex.getObserverCS()),
                    emPosition.getCoordinates(rootCS));

  conex.addParticle(Code::Proton, Eem, Proton::mass, emPosition, momentum.normalized(),
                    0_s);
  // supperimpose a photon
  auto const momentumPhoton = showerAxis.getDirection() * 1_TeV;
  conex.addParticle(Code::Photon, 1_TeV, 0_eV, emPosition, momentumPhoton.normalized(),
                    0_s);
  DummyStack stack;
  conex.doCascadeEquations(stack);

  CHECK(w1.getEnergyLost() / 1_TeV == Approx(1.0).epsilon(0.1));

  auto const cfg = conex.getConfig();
  CHECK(cfg.size() == 0);
}
