/*
 * (c) Copyright 2019 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/modules/Epos.hpp>
#include <corsika/modules/Sibyll.hpp>
#include <corsika/modules/QGSJetII.hpp>
#include <corsika/modules/QGSJetIII.hpp>
#include <corsika/modules/Pythia8.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>

#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>

#include <corsika/framework/random/RNGManager.hpp>

#include <corsika/setup/SetupC7trackedParticles.hpp>
#include <corsika/modules/Random.hpp>

#include <fstream>
#include <string>
#include <tuple>

using namespace corsika;

/**
 * Calculates the inelastic p-p cross section and the production cross section for
 * p-Oxygen for a given interaction model using the C8 interface getCrossSection(ID1, ID2,
 * P41, P42) where ID? and P4? are the particle Id and 4-momenta of projectile and target
 * respectively. The input to getCrossSection can be ANY frame of reference. Internally
 * the boost to the CM is calculated and the correct cross section is returned.
 * calculate_cross_sections allows to switch between the CM and lab configurations.
 *
 * output is a table in ACSCII file cross-sections-<model-name>.txt
 *
 * Energy ranges from 100GeV to 10^12 GeV (10^11 eV to 10^21 eV) in the lab. frame
 * table format is: "# i, E_lab (GeV), E_cm per nucleon (GeV), cross section p-p (mb),
 * production "
 *
 * @param model the interaction model to use
 * @param model_name string with model name
 *
 * @return a tuple of: inelastic cross section, elastic cross section
 */

template <typename TModel>
void calculate_cross_sections(TModel& model, std::string const& model_name) {

  std::stringstream fname;
  fname << "cross-sections-" << model_name << ".txt";
  std::ofstream out(fname.str());
  out << "# cross section table for " << model_name << "\n";
  out << "# i, P_lab (GeV), E_cm per nucleon (GeV), cross section p-p (mb), cross "
         "section pi-p (mb), production "
         "cross section p-O16 (mb), production cross section p-N14 (mb), production "
         "cross section p-Ar40 (mb), cross section pi-O16 (mb), production cross "
         "section, pi-N14 (mb), production, cross section pi-Ar40 (mb), cross section "
         "He-O16 (mb), production cross section, He-N14 (mb), production, cross section "
         "He-Ar40 (mb)\n";

  CoordinateSystemPtr const& cs = get_root_CoordinateSystem();

  float const amin = 2;
  float const amax = 11;
  int const nebins = 15;
  float const da = (amax - amin) / (float)nebins;

  for (int i = 0; i < nebins; ++i) {

    corsika::units::si::HEPMomentumType const setP = pow(10, da * i + amin) * 1_GeV;
    CrossSectionType xs_prod_hNuc[3][3] = {};
    CrossSectionType xs_prod_hp[3] = {};
    HEPEnergyType setE, comEnn;
    int l = -1;
    for (auto projId : {Code::Proton, Code::PiPlus, Code::Helium}) {
      ++l;
      setE = calculate_total_energy(setP, get_mass(projId));
      comEnn = corsika::calculate_com_energy(setE, get_mass(projId),
                                             corsika::constants::nucleonMass);

      // four momenta in lab frame
      corsika::FourMomentum p4Proj(setE, MomentumVector(cs, {0_eV, 0_eV, setP}));
      corsika::FourMomentum p4protonTarg(Proton::mass,
                                         MomentumVector(cs, {0_eV, 0_eV, 0_eV}));

      // had-p
      auto const [xs_prod, xs_ela] =
          model->getCrossSectionInelEla(projId, Code::Proton, p4Proj, p4protonTarg);
      xs_prod_hp[l] = xs_prod;

      int k = -1;
      for (auto targNuc : {Code::Oxygen, Code::Nitrogen, Code::Argon}) {
        ++k;
        FourMomentum const p4nucTarg = corsika::FourMomentum(
            get_mass(targNuc), MomentumVector(cs, {0_eV, 0_eV, 0_eV}));
        xs_prod_hNuc[l][k] = model->getCrossSection(projId, targNuc, p4Proj, p4nucTarg);
      }
    }
    CORSIKA_LOG_INFO(
        "Elab={:15.2f} GeV,Ecom={:15.2f} GeV, sig-p-p={:6.2f} mb, sig-pi-p={:6.2f} mb, "
        "p-O={:6.2f} mb, p-N={:6.2f} mb, p-Ar={:6.2f} mb "
        "pi-O={:6.2f} mb, pi-N={:6.2f} mb, pi-Ar={:6.2f} mb "
        "He-O={:6.2f} mb, He-N={:6.2f} mb, He-Ar={:6.2f} mb",
        setP / 1_GeV, comEnn / 1_GeV, xs_prod_hp[0] / 1_mb, xs_prod_hp[1] / 1_mb,
        xs_prod_hNuc[0][0] / 1_mb, xs_prod_hNuc[0][1] / 1_mb, xs_prod_hNuc[0][2] / 1_mb,
        xs_prod_hNuc[1][0] / 1_mb, xs_prod_hNuc[1][1] / 1_mb, xs_prod_hNuc[1][2] / 1_mb,
        xs_prod_hNuc[2][0] / 1_mb, xs_prod_hNuc[2][1] / 1_mb, xs_prod_hNuc[2][2] / 1_mb);

    out << i << " " << setP / 1_GeV << " " << comEnn / 1_GeV << " "
        << xs_prod_hp[0] / 1_mb << " " << xs_prod_hp[1] / 1_mb << " "
        << xs_prod_hNuc[0][0] / 1_mb << " " << xs_prod_hNuc[0][1] / 1_mb << " "
        << xs_prod_hNuc[0][2] / 1_mb << " " << xs_prod_hNuc[1][0] / 1_mb << " "
        << xs_prod_hNuc[1][1] / 1_mb << " " << xs_prod_hNuc[1][2] / 1_mb << " "
        << " " << xs_prod_hNuc[2][0] / 1_mb << " " << xs_prod_hNuc[2][1] / 1_mb << " "
        << xs_prod_hNuc[2][2] / 1_mb << "\n";
  }
  out.close();
  std::cout << "wrote cross section table for " << model_name
            << " in file: " << fname.str() << std::endl;
}

int main(int argc, char** argv) {

  logging::set_level(logging::level::info);

  if (argc != 2) {
    std::cout << "usage: check <interaction model> \n valid models are: sibyll, "
                 "epos, qgsjetII, qgsjetIII pythia8"
              << std::endl;
    return 1;
  }
  std::string int_model_name = std::string(argv[1]);

  if (int_model_name == "sibyll") {
    RNGManager<>::getInstance().registerRandomStream("sibyll");
    auto model = std::make_shared<corsika::sibyll::InteractionModel>(
        std::set<Code>{Code::Proton, Code::Oxygen, Code::Nitrogen},
        corsika::setup::C7trackedParticles);
    calculate_cross_sections(model, int_model_name);
  } else if (int_model_name == "pythia8") {
    RNGManager<>::getInstance().registerRandomStream("pythia");
    auto model = std::make_shared<corsika::pythia8::InteractionModel>();
    calculate_cross_sections(model, int_model_name);

  } else if (int_model_name == "epos") {
    RNGManager<>::getInstance().registerRandomStream("epos");
    auto model = std::make_shared<corsika::epos::InteractionModel>(
        corsika::setup::C7trackedParticles);
    calculate_cross_sections(model, int_model_name);
  } else if (int_model_name == "qgsjetII") {
    RNGManager<>::getInstance().registerRandomStream("qgsjet");
    auto model = std::make_shared<corsika::qgsjetII::InteractionModel>();
    calculate_cross_sections(model, int_model_name);
  } else if (int_model_name == "qgsjetIII") {
    RNGManager<>::getInstance().registerRandomStream("qgsjetIII");
    auto model = std::make_shared<corsika::qgsjetIII::InteractionModel>();
    calculate_cross_sections(model, int_model_name);    
  } else {
    std::cout << "interaction model should be: sibyll, epos, qgsjet or pythia"
              << std::endl;
    return 1;
  }
}
