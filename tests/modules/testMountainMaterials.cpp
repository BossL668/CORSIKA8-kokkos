#include <catch2/catch_all.hpp>
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <corsika/modules/transport/MaterialProposalEnvironment.hpp>
#include <corsika/modules/transport/MaterialHadronLoss.hpp>
#include "../../applications/detail/mountain/TerrainMaterialConfig.hpp"
#include <iostream>
using namespace corsika;
using namespace corsika::units::si;
using Catch::Approx;
namespace api=corsika::interfaces;
namespace app=corsika::applications::terrain;

TEST_CASE("Mountain presets reproduce the supplied research cards", "[material]") {
  auto cards=YAML::LoadFile(std::string(MATERIAL_SOURCE_DIR)+"/documentation/materials.yaml");
  for(auto card:cards["material_cards"]) {
    auto m=api::materialPreset(card["id"].as<std::string>());
    CHECK(m.density_kg_m3==card["transport"]["density_kg_m3"].as<double>());
    CHECK(m.refractive_index==Approx(card["radio"]["refractive_index"].as<double>()).epsilon(1.e-15));
    CHECK(m.field_attenuation_length_m==card["radio"]["field_attenuation_length_m"].as<double>());
    std::map<std::pair<int,int>,double> expected,actual;
    for(auto n:card["transport"]["nuclei"]) expected[{n["Z"].as<int>(),n["A"].as<int>()}]=n["number_fraction"].as<double>();
    for(auto n:m.nuclei) actual[{n.Z,n.A}]=n.number_fraction;
    REQUIRE(expected.size()==actual.size());
    for(auto n:expected) CHECK(actual.at(n.first)==n.second);
    auto roundtrip=app::readMaterial(app::materialConfig(m));
    CHECK(roundtrip.composition().getHash()==m.composition().getHash());
    CHECK(api::transportKey(roundtrip.transportProperties())==api::transportKey(m.transportProperties()));
    CHECK(roundtrip.field_attenuation_length_m==m.field_attenuation_length_m);
  }
  CHECK(api::materialPreset("SiO2").refractive_index==std::sqrt(5.));
  CHECK(api::materialPreset("Calcite").ZOverA()==.5);
  auto silica=api::materialPreset("SiO2");
  CHECK(silica.massFractions()[0]==Approx(28./60.));
  CHECK(silica.massFractions()[1]==Approx(32./60.));
}

TEST_CASE("NIST density shifts and mixture ionisation use selected material", "[material]") {
  for(auto name:{"SiO2","Calcite"}) {
    auto m=api::materialPreset(name); auto d=m.transportProperties();
    auto ref=corsika::mediumData(m.reference_medium);
    double ratio=m.density_kg_m3*.001/ref.corrected_density_;
    CHECK(d.Ieff_==ref.Ieff_); CHECK(d.corrected_density_==m.density_kg_m3*.001);
    for(double bg:{.1,1.,10.,100.,1.e4,1.e8})
      CHECK(api::densityEffect(d,bg)==Approx(api::densityEffect(ref,bg*std::sqrt(ratio))).margin(1.e-13));
    CHECK(api::hadronStoppingPower(d,938.27208816,1938.27208816,1)>1.);
    CHECK(api::hadronStoppingPower(d,938.27208816,1938.27208816,0)==0.);
  }
  auto g=api::materialPreset("Granite"); auto d=g.transportProperties();
  // Independent elemental I inputs from the bundled NIST records (eV).
  std::map<int,double> I{{1,19.2},{6,78.},{8,95.},{11,149.},{12,156.},
                        {13,166.},{14,173.},{19,190.},{20,191.},{26,286.}};
  double logI=0.,electrons=0.;
  for(auto n:g.nuclei) { logI+=n.number_fraction*n.Z*std::log(I.at(n.Z)); electrons+=n.number_fraction*n.Z; }
  CHECK(d.Ieff_==Approx(std::exp(logI/electrons)).epsilon(1.e-14));
  CHECK(api::densityEffect(d,std::pow(10.,d.x0_))==Approx(0.).margin(1.e-13));
  CHECK(d.Ieff_!=corsika::mediumData(Medium::StandardRock).Ieff_);
  auto doubled=g; doubled.density_kg_m3*=2.;
  CHECK(doubled.transportProperties().Cbar_==Approx(d.Cbar_-std::log(2.)));
}

TEST_CASE("Material view passes resolved data and rejects ambiguous calculator keys", "[material]") {
  terrain::Environment env; auto cs=env.getCoordinateSystem();
  auto add=[&](api::MaterialModel const& m,double x) {
    auto n=env.createNode<Sphere>(Point(cs,x*1_m,0_m,0_m),1_m);
    n->setModelProperties(m.makeModel<terrain::Interface>(cs)); env.getUniverse()->addChild(std::move(n));
  };
  auto silica=api::materialPreset("SiO2"); add(silica,0.);
  api::MaterialProposalEnvironment view(env); int count=0;
  view.getUniverse()->walk([&](auto const& n) {
    if(!n.hasModelProperties()) return;
    auto p=n.getModelProperties(); auto const& d=mediumData(p.getMedium());
    CHECK(d.corrected_density_==2.65); CHECK(d.name_==silica.id); ++count;
  });
  CHECK(count==1);
  auto optical=silica; optical.refractive_index=3.; optical.field_attenuation_length_m=30.; add(optical,3.);
  CHECK_NOTHROW(api::MaterialProposalEnvironment(env)); // optical edits do not alter EM tables
  auto dense=silica; dense.density_kg_m3*=1.1; add(dense,6.);
  CHECK_THROWS(api::MaterialProposalEnvironment(env));
}

TEST_CASE("Invalid and unsupported material cards fail before tables are built", "[material]") {
  auto m=api::materialPreset("SiO2");
  m.nuclei[0].number_fraction=-1.; CHECK_THROWS(m.validate());
  m=api::materialPreset("SiO2"); m.nuclei[0].number_fraction=.5; CHECK_THROWS(m.validate());
  m=api::materialPreset("SiO2"); m.nuclei[0].Z=13; CHECK_THROWS(m.transportProperties());
  m=api::materialPreset("SiO2"); m.field_attenuation_length_m=NAN; CHECK_THROWS(m.validate());
  m.field_attenuation_length_m=INFINITY; CHECK_NOTHROW(m.validate());
  CHECK_THROWS(app::readMaterial(YAML::Load("{preset: SiO2, radio: {frequency_dependent_attenuation_enabled: true}}")));
  CHECK_THROWS(app::readMaterial(YAML::Load("{preset: SiO2, radio: {attenuation_length: 10}}")));
  CHECK_THROWS(app::readMaterial(YAML::Load("{transport: {composition_basis: mass_fraction}}")));
  auto explicitCard=app::materialConfig(api::materialPreset("SiO2"));
  explicitCard["transport"]["ionisation"]["reference_density_kg_m3"]=NAN;
  CHECK_THROWS(app::readMaterial(explicitCard));
  CHECK_THROWS(app::readMaterial(YAML::Load("{transport: {ionisation: {model: bragg_sternheimer, I_eV: 200}}}")));
  CHECK_THROWS(api::materialPreset("Carbonate")); // no universal carbonate recipe
}
