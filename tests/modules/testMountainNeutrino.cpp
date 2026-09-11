/* Opt-in mountain CC module contracts; no stock neutrino process is modified. */
#include <corsika/modules/neutrino/MountainNeutrinoInteraction.hpp>
#include <corsika/modules/neutrino/TransportedLeptons.hpp>
#include "../../applications/detail/mountain/TerrainPrimaries.hpp"

#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>

#include <catch2/catch_all.hpp>

#include <limits>

using namespace corsika;
using namespace corsika::units::si;
using Catch::Approx;

TEST_CASE("Terrain named primary admission", "[mountain][neutrino]") {
  auto const& p = corsika::applications::terrain::primaries();
  CHECK(p.at("nu_tau") == Code::NuTau);
  CHECK(p.at("anti_nu_tau") == Code::NuTauBar);
  CHECK(p.at("nu_mu") == Code::NuMu);
  CHECK(p.at("anti_nu_mu") == Code::NuMuBar);
  CHECK(p.at("tau_minus") == Code::TauMinus);
  CHECK(p.at("tau_plus") == Code::TauPlus);
  CHECK(p.at("mu_minus") == Code::MuMinus);
  CHECK(p.at("mu_plus") == Code::MuPlus);
  CHECK(p.at("proton") == Code::Proton);
  CHECK_THROWS(p.at("unimplemented-particle"));
}

TEST_CASE("Mountain CC retains all outgoing charged lepton flavors", "[mountain][neutrino][tau]") {
  auto stable = neutrino::withTransportedTaus(setup::C7trackedParticles);
  REQUIRE(stable.count(Code::TauMinus) == 1);
  REQUIRE(stable.count(Code::TauPlus) == 1);
  CHECK(setup::C7trackedParticles.count(Code::TauMinus) == 0);
  CHECK(stable.size() == setup::C7trackedParticles.size()+2);
  auto& manager = RNGManager<>::getInstance();
  manager.registerRandomStream("pythia");
  auto cs = get_root_CoordinateSystem();
  for (int pdg : {12,-12,14,-14,16,-16}) for (double energy : {1.e4,1.e5}) {
    CAPTURE(pdg,energy);
    manager.setSeed(79110+std::abs(pdg)+(pdg<0?10:0)+(energy>1.e4?1:0));
    neutrino::AuditedPythiaNeutrinoFinalState gen(stable);
    auto event = gen.generate(convert_from_PDG(static_cast<PDGCode>(pdg)),get_nucleus_code(28,14),
        FourMomentum{energy*1_GeV,MomentumVector(cs,{0_GeV,0_GeV,energy*1_GeV})});
    auto& a = event.audit;
    int const charged = pdg>0 ? pdg-1 : pdg+1;
    CHECK(a.selectedOutgoingLeptonPdg == charged);
    CHECK(a.finalStatePdgCounts[charged] >= 1);
    CHECK(a.finalChargeE == a.initialChargeE);
    CHECK(a.corsikaMaxComponentRelativeResidual < 1.e-3);
    CHECK(a.inelasticityY > 0.);
    CHECK(a.inelasticityY < 1.);
  }
  // Reproduce the original mountain diagnosis using the same vertex RNG seed.
  manager.setSeed(79126);
  neutrino::AuditedPythiaNeutrinoFinalState old(setup::C7trackedParticles);
  auto legacy=old.generate(Code::NuTau,get_nucleus_code(28,14),
      FourMomentum{1.e4_GeV,MomentumVector(cs,{0_GeV,0_GeV,1.e4_GeV})});
  CHECK(legacy.audit.finalStatePdgCounts[15] == 0);
}

TEST_CASE("Mountain neutrino CTW scope and rates", "[mountain][neutrino]") {
  using namespace corsika::neutrino;
  for (int const pdg : {12, -12, 14, -14, 16, -16}) {
    auto const pid = convert_from_PDG(static_cast<PDGCode>(pdg));
    CHECK_NOTHROW(MountainNeutrinoInteraction::validatePrimary(pid, 1.e4));
    CHECK_NOTHROW(MountainNeutrinoInteraction::validatePrimary(pid, 1.e12));
    CHECK_THROWS(MountainNeutrinoInteraction::validatePrimary(pid, 9999.));
    CHECK_THROWS(MountainNeutrinoInteraction::validatePrimary(pid, 1.01e12));
    CHECK_THROWS(MountainNeutrinoInteraction::validatePrimary(
        pid, std::numeric_limits<double>::quiet_NaN()));
    CHECK(ctw2011CrossSectionCm2(pid, 1.e4) > 0.);
    CHECK(ctw2011CrossSectionCm2(pid, 1.e12) > 0.);
    CHECK(ctw2011CrossSectionCm2(pid, 9999.) == 0.);
    CHECK(ctw2011CrossSectionCm2(pid, 1.01e12) == 0.);
  }
  CHECK_THROWS(MountainNeutrinoInteraction::validatePrimary(Code::Proton, 1.e6));
  CHECK(ctw2011CrossSectionCm2(Code::Proton, 1.e6) == 0.);
  CHECK(ctw2011CrossSectionCm2(Code::NuE, 1.e6) ==
        Approx(7.2878172702e-34).epsilon(1.e-10));
  CHECK(targetMassNumber(get_nucleus_code(28, 14)) == 28.);
  CHECK(targetMassNumber(Code::Oxygen) == 16.);
  CHECK(targetMassNumber(Code::Proton) == 1.);
  CHECK(targetMassNumber(Code::Electron) == 0.);
}

TEST_CASE("Mountain CC natural versus forced per-target rate contract",
          "[mountain][neutrino]") {
  using namespace corsika::neutrino;
  RNGManager<>::getInstance().registerRandomStream("pythia");
  auto const cs = get_root_CoordinateSystem();
  FourMomentum const projectile{1.e6_GeV,
                               MomentumVector(cs, {0_GeV, 0_GeV, 1.e6_GeV})};
  auto const oxygen = Code::Oxygen;
  auto const silicon = get_nucleus_code(28, 14);
  FourMomentum const oxygenP4{get_mass(oxygen),
                             MomentumVector(cs, {0_GeV, 0_GeV, 0_GeV})};
  FourMomentum const siliconP4{get_mass(silicon),
                              MomentumVector(cs, {0_GeV, 0_GeV, 0_GeV})};
  MountainNeutrinoInteraction natural(setup::C7trackedParticles,
                                      SamplingMode::NaturalCC);
  MountainNeutrinoInteraction forced(setup::C7trackedParticles,
                                     SamplingMode::ForcedVertexCC);
  auto const sigmaO = natural.getCrossSection(Code::NuE, oxygen,
                                             projectile, oxygenP4);
  auto const sigmaSi = natural.getCrossSection(Code::NuE, silicon,
                                              projectile, siliconP4);
  CHECK(sigmaSi / sigmaO == Approx(28. / 16.));
  CHECK(forced.getCrossSection(Code::NuE, oxygen, projectile, oxygenP4) /
            sigmaO == Approx(1.));
  CHECK(forced.getCrossSection(Code::NuE, silicon, projectile, siliconP4) /
            sigmaSi == Approx(1.));
  CHECK_FALSE(natural.summary()["forced_vertex_is_conditional"].as<bool>());
  CHECK(forced.summary()["forced_vertex_is_conditional"].as<bool>());
  CHECK_FALSE(forced.summary()["event_rate_weight_provided"].as<bool>());
  natural.clearRecords();
  CHECK(natural.records().empty());
}

TEST_CASE("Mountain CC audited Pythia vertex on silicon",
          "[mountain][neutrino][pythia]") {
  auto& manager = RNGManager<>::getInstance();
  manager.registerRandomStream("pythia");
  manager.setSeed(87001);
  auto const cs = get_root_CoordinateSystem();
  FourMomentum const projectile{1.e4_GeV,
                               MomentumVector(cs, {0_GeV, 0_GeV, 1.e4_GeV})};
  corsika::neutrino::AuditedPythiaNeutrinoFinalState generator(
      setup::C7trackedParticles);
  auto const event = generator.generate(Code::NuE, get_nucleus_code(28, 14),
                                        projectile);
  CHECK_FALSE(event.secondaries.empty());
  CHECK(event.audit.hardOutgoingLeptonCandidateCount == 1);
  CHECK(event.audit.selectedOutgoingLeptonPdg == 11);
  CHECK(event.audit.finalChargeE == event.audit.initialChargeE);
  CHECK(event.audit.corsikaMaxComponentRelativeResidual < 1.e-4);
  CHECK(event.audit.q2GeV2 >= 25. * (1. - 1.e-9));
  CHECK(event.audit.inelasticityY > 0.);
  CHECK(event.audit.inelasticityY < 1.);
}
