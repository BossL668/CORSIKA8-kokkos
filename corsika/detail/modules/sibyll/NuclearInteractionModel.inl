/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/media/Environment.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/modules/Random.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/utility/COMBoost.hpp>
#include <corsika/framework/core/Logging.hpp>

#include <nuclib.hpp>

namespace corsika::sibyll {

  template <typename TNucleonModel>
  inline NuclearInteractionModel<TNucleonModel>::NuclearInteractionModel(
      TNucleonModel& hadint, std::set<Code> const& nuccomp)
      : hadronicInteraction_(hadint) {

    // initialize nuclib
    // TODO: make sure this does not overlap with sibyll
    corsika::connect_random_stream("sibyll", ::sibyll23d::set_rng_function);
    ::sibyll23d::nuc_nuc_ini_();

    // initialize cross sections
    initializeNuclearCrossSections(nuccomp);
  }

  template <typename TNucleonModel>
  inline NuclearInteractionModel<TNucleonModel>::~NuclearInteractionModel() {
    CORSIKA_LOGGER_DEBUG(logger_, "nuclear interactions handled by Sibyll n={}", count_);
  }

  template <typename TNucleonModel>
  inline bool constexpr NuclearInteractionModel<TNucleonModel>::isValid(
      Code const projectileId, Code const targetId, HEPEnergyType const sqrtSnn) const {

    // also depends on underlying model, for Proton/Neutron projectile
    if (!hadronicInteraction_.isValid(Code::Proton, targetId, sqrtSnn)) { return false; }

    // projectile limits:
    if (!is_nucleus(projectileId)) { return false; }
    unsigned int projectileA = get_nucleus_A(projectileId);
    if (projectileA > getMaxNucleusAProjectile() || projectileA < 2) { return false; }
    return true;
  } // namespace corsika::sibyll

  template <typename TNucleonModel>
  inline void NuclearInteractionModel<TNucleonModel>::printCrossSectionTable(
      Code const pCode) const {
    if (!hadronicInteraction_.isValid(Code::Proton, pCode, 100_GeV)) { // LCOV_EXCL_START
      CORSIKA_LOGGER_WARN(logger_, "Invalid target type {} for hadron interaction model.",
                          pCode);
      return;
    } // LCOV_EXCL_STOP

    int const k = targetComponentsIndex_.at(pCode);
    Code const pNuclei[] = {Code::Helium, Code::Lithium7, Code::Oxygen,
                            Code::Neon,   Code::Argon,    Code::Iron};

    std::ostringstream table;
    table << "Nuclear CrossSectionTable pCode=" << pCode << " :\n en/A ";
    for (auto& j : pNuclei) table << std::setw(9) << j;
    table << "\n";

    // loop over energy bins
    for (unsigned int i = 0; i < getNEnergyBins(); ++i) {
      table << " " << i << "  ";

      for (auto& n : pNuclei) {
        auto const j = get_nucleus_A(n);
        table << " " << std::setprecision(5) << std::setw(8)
              << cnucsignuc_.sigma[j - 1][k][i];
      }
      table << "\n";
    }
    CORSIKA_LOGGER_DEBUG(logger_, table.str());
  }

  template <typename TNucleonModel>
  inline void NuclearInteractionModel<TNucleonModel>::initializeNuclearCrossSections(
      std::set<Code> const& allElementsInUniverse) {

    CORSIKA_LOGGER_DEBUG(logger_, "initializing nuclear cross sections...");

    // loop over target components, at most 4!!
    int k = -1;
    for (Code const ptarg : allElementsInUniverse) {
      ++k;
      CORSIKA_LOGGER_DEBUG(logger_, "init target component: {} A={}", ptarg,
                           get_nucleus_A(ptarg));
      int const ib = get_nucleus_A(ptarg);
      if (!hadronicInteraction_.isValid(Code::Proton, ptarg, 100_GeV)) {
        CORSIKA_LOGGER_WARN(
            logger_, "Invalid target type {} for hadron interaction model.", ptarg);
        continue;
      }
      targetComponentsIndex_.insert(std::pair<Code, int>(ptarg, k));
      // loop over energies, fNEnBins log. energy bins
      for (size_t i = 0; i < getNEnergyBins(); ++i) {
        // hard coded energy grid, has to be aligned to definition in signuc2!!, no
        // comment..
        HEPEnergyType const Ecm = pow(10., 1. + 1. * i) * 1_GeV;
        // head-on pp collision:
        HEPEnergyType const EcmHalve = Ecm / 2;
        HEPMomentumType const pcm =
            sqrt(EcmHalve * EcmHalve - Proton::mass * Proton::mass);
        CoordinateSystemPtr cs = get_root_CoordinateSystem();
        FourMomentum projectileP4(EcmHalve, {cs, pcm, 0_eV, 0_eV});
        FourMomentum targetP4(EcmHalve, {cs, -pcm, 0_eV, 0_eV});
        // get p-p cross sections
        if (!hadronicInteraction_.isValid(Code::Proton, Code::Proton, Ecm)) {
          throw std::runtime_error("invalid (projectile,target,ecm) combination");
        }
        auto const [siginel, sigela] = hadronicInteraction_.getCrossSectionInelEla(
            Code::Proton, Code::Proton, projectileP4, targetP4);
        double const dsig = siginel / 1_mb;
        double const dsigela = sigela / 1_mb;
        // loop over projectiles, mass numbers from 2 to fMaxNucleusAProjectile
        CORSIKA_LOGGER_TRACE(logger_, "Ecm={} siginel={} sigela={}", Ecm / 1_GeV, dsig,
                             dsigela);
        for (size_t j = 1; j < gMaxNucleusAProjectile_; ++j) {
          const int jj = j + 1;
          double sig_out, dsig_out, sigqe_out, dsigqe_out;
          sigma_mc_(jj, ib, dsig, dsigela, gNSample_, sig_out, dsig_out, sigqe_out,
                    dsigqe_out);
          // write to table
          cnucsignuc_.sigma[j][k][i] = sig_out;
          cnucsignuc_.sigqe[j][k][i] = sigqe_out;
          CORSIKA_LOGGER_TRACE(logger_, "nuc A={} sig={} qe={}", j, sig_out, sigqe_out);
        }
      }
    }
    CORSIKA_LOGGER_DEBUG(logger_, "cross sections for {} components initialized!",
                         targetComponentsIndex_.size());
    for (auto& ptarg : allElementsInUniverse) { printCrossSectionTable(ptarg); }
  }

  template <typename TNucleonModel>
  inline CrossSectionType NuclearInteractionModel<TNucleonModel>::readCrossSectionTable(
      int const ia, Code const pTarget, HEPEnergyType const elabnuc) const {
    CORSIKA_LOGGER_DEBUG(logger_, "ia={}, target={}, ElabNuc={} GeV", ia, pTarget,
                         elabnuc / 1_GeV);
    int const ib = targetComponentsIndex_.at(pTarget) + 1; // table index in fortran
    auto const ECoMNuc = sqrt(2. * constants::nucleonMass * elabnuc);
    CORSIKA_LOGGER_DEBUG(logger_, "sqrtSnn= {} GeV", ECoMNuc / 1_GeV);
    if (ECoMNuc < getMinEnergyPerNucleonCoM() || ECoMNuc > getMaxEnergyPerNucleonCoM()) {
      CORSIKA_LOGGER_WARN(
          logger_,
          "nucleon-nucleon energy outside range! sqrtSnn={}GeV (limits: {} .. {} GeV)",
          ECoMNuc / 1_GeV, getMinEnergyPerNucleonCoM() / 1_GeV,
          getMaxEnergyPerNucleonCoM() / 1_GeV);
      // throw std::runtime_error("energy outside tabulated range!");
    }
    double const e0 = elabnuc / 1_GeV;
    double sig;
    CORSIKA_LOGGER_DEBUG(logger_, "ReadCrossSectionTable: {} {} {}", ia, ib, e0);
    signuc2_(ia, ib, e0, sig);
    CORSIKA_LOGGER_DEBUG(logger_, "ReadCrossSectionTable: sig={}", sig);
    return sig * 1_mb;
  }

  template <typename TNucleonModel>
  CrossSectionType inline NuclearInteractionModel<TNucleonModel>::getCrossSection(
      Code const projectileId, Code const targetId, FourMomentum const& projectileP4,
      FourMomentum const& targetP4) const {
    CORSIKA_LOGGER_DEBUG(logger_, "projectile: E={}, p3={} \n target: E={}, p3={}",
                         projectileP4.getTimeLikeComponent() / 1_GeV,
                         projectileP4.getSpaceLikeComponents() / 1_GeV,
                         targetP4.getTimeLikeComponent() / 1_GeV,
                         targetP4.getSpaceLikeComponents() / 1_GeV);
    // check if projectile and target are nuclei!
    if (!is_nucleus(projectileId) || !is_nucleus(targetId)) {
      return CrossSectionType::zero();
    }

    // calculate sqrt(Snn) (only works if projectile and target are nuclei)
    HEPEnergyType const sqrtSnn =
        (projectileP4 / get_nucleus_A(projectileId) + targetP4 / get_nucleus_A(targetId))
            .getNorm();

    CORSIKA_LOG_DEBUG("proj={}, targ={}, sqrtSNN={}GeV", projectileId, targetId,
                      sqrtSnn / 1_GeV);
    if (!isValid(projectileId, targetId, sqrtSnn)) { return CrossSectionType::zero(); }

    // lab-frame energy per projectile nucleon as required by signuc2()
    HEPEnergyType const LabEnergyPerNuc = calculate_lab_energy(
        static_pow<2>(sqrtSnn), get_mass(projectileId) / get_nucleus_A(projectileId),
        get_mass(targetId) / get_nucleus_A(targetId));
    auto const sigProd =
        readCrossSectionTable(get_nucleus_A(projectileId), targetId, LabEnergyPerNuc);
    CORSIKA_LOGGER_DEBUG(logger_, "cross section (mb): sqrtSnn={} sig={}",
                         sqrtSnn / 1_GeV, sigProd / 1_mb);
    return sigProd;
  }

  template <typename TNucleonModel>
  template <typename TSecondaryView>
  inline void NuclearInteractionModel<TNucleonModel>::doInteraction(
      TSecondaryView& view, Code const projectileId, Code const targetId,
      FourMomentum const& projectileP4, FourMomentum const& targetP4) {

    // model is only designed for projectile nuclei. Collisions are broken down into
    // "nucleon-target" collisions.
    if (!is_nucleus(projectileId)) {
      throw std::runtime_error("Can only handle nuclear projectiles.");
    }
    size_t const projectileA = get_nucleus_A(projectileId);

    // this is center-of-mass for projectile_nucleon - target
    FourMomentum const nucleonP4 = projectileP4 / projectileA;
    HEPEnergyType const sqrtSnucleon = (nucleonP4 + targetP4).getNorm();
    if (!isValid(projectileId, targetId, sqrtSnucleon)) {
      throw std::runtime_error("Invalid projectile/target/energy combination.");
    }
    // projectile is always nucleus!
    // Elab corresponding to sqrtSnucleon -> fixed target projectile
    COMBoost const boost(nucleonP4, targetP4);

    CORSIKA_LOGGER_DEBUG(logger_, "pId={} tId={} sqrtSnucleon={}GeV Aproj={}",
                         projectileId, targetId, sqrtSnucleon / 1_GeV, projectileA);
    count_++;

    // lab. momentum per projectile nucleon
    HEPMomentumType const pNucleonLab = nucleonP4.getSpaceLikeComponents().getNorm();
    // nucleon momentum in direction of CM motion (lab system)
    MomentumVector const p3NucleonLab(boost.getRotatedCS(), {0_GeV, 0_GeV, pNucleonLab});

    /*
      FOR NOW: allow nuclei with A<18 or protons/nucleon only.
      when medium composition becomes more complex, approximations will have to be
      allowed air in atmosphere also contains some Argon.
    */
    int kATarget = -1;
    size_t targetA = 1;
    if (is_nucleus(targetId)) {
      kATarget = get_nucleus_A(targetId);
      targetA = kATarget;
    } else if (targetId == Code::Proton || targetId == Code::Neutron ||
               targetId == Code::Hydrogen) {
      kATarget = 1;
    }
    CORSIKA_LOGGER_DEBUG(logger_, "nuclib target code: {}", kATarget);

    // end of target sampling

    // superposition
    CORSIKA_LOGGER_DEBUG(logger_, "sampling nuc. multiple interaction structure.. ");
    // get nucleon-nucleon cross section
    // (needed to determine number of nucleon-nucleon scatterings)
    auto const protonId = Code::Proton;
    auto const [prodCrossSection, elaCrossSection] =
        hadronicInteraction_.getCrossSectionInelEla(
            protonId, protonId, nucleonP4,
            targetP4 / targetA); // todo check, wrong RU
    double const sigProd = prodCrossSection / 1_mb;
    double const sigEla = elaCrossSection / 1_mb;
    // sample number of interactions (only input variables, output in common cnucms)
    // nuclear multiple scattering according to glauber (r.i.p.)
    int_nuc_(kATarget, projectileA, sigProd, sigEla);

    CORSIKA_LOGGER_DEBUG(logger_,
                         "number of nucleons in target           : {}\n"
                         "number of wounded nucleons in target   : {}\n"
                         "number of nucleons in projectile       : {}\n"
                         "number of wounded nucleons in project. : {}\n"
                         "number of inel. nuc.-nuc. interactions : {}\n"
                         "number of elastic nucleons in target   : {}\n"
                         "number of elastic nucleons in project. : {}\n"
                         "impact parameter: {}",
                         kATarget, cnucms_.na, projectileA, cnucms_.nb, cnucms_.ni,
                         cnucms_.nael, cnucms_.nbel, cnucms_.b);

    // calculate fragmentation
    CORSIKA_LOGGER_DEBUG(logger_, "calculating nuclear fragments..");
    // number of interactions
    // include elastic
    int const nElasticNucleons = cnucms_.nbel;
    int const nInelNucleons = cnucms_.nb;
    int const nIntProj = nInelNucleons + nElasticNucleons;
    double const impactPar = cnucms_.b; // only needed to avoid passing common var.
    int nFragments = 0;
    // number of fragments is limited to 60
    int AFragments[60];
    // call fragmentation routine
    // input: target A, projectile A, number of int. nucleons in projectile, impact
    // parameter (fm) output: nFragments, AFragments in addition the momenta ar stored
    // in pf in common fragments, neglected
    fragm_(kATarget, projectileA, nIntProj, impactPar, nFragments, AFragments);

    // this should not occur but well :)  (LCOV_EXCL_START)
    if (nFragments > (int)getMaxNFragments()) {
      throw std::runtime_error("Number of nuclear fragments in NUCLIB exceeded!");
    }
    // (LCOV_EXCL_STOP)

    CORSIKA_LOGGER_DEBUG(logger_, "number of fragments: {}", nFragments);
    CORSIKA_LOGGER_DEBUG(logger_, "adding nuclear fragments to particle stack..");
    // put nuclear fragments on corsika stack
    for (int j = 0; j < nFragments; ++j) {
      CORSIKA_LOGGER_DEBUG(logger_, "fragment {}: A={} px={} py={} pz={}", j,
                           AFragments[j], fragments_.ppp[j][0], fragments_.ppp[j][1],
                           fragments_.ppp[j][2]);
      auto const nuclA = AFragments[j];
      // get Z from stability line
      auto const nuclZ = int(nuclA / 2.15 + 0.7);

      // TODO: do we need to catch single nucleons??
      Code const specCode = (nuclA == 1 ?
                                        // TODO: sample neutron or proton
                                 Code::Proton
                                        : get_nucleus_code(nuclA, nuclZ));
      HEPMassType const mass = get_mass(specCode);

      CORSIKA_LOGGER_DEBUG(logger_, "adding fragment: {}", get_name(specCode));
      CORSIKA_LOGGER_DEBUG(logger_, "A,Z: {}, {}", nuclA, nuclZ);
      CORSIKA_LOGGER_DEBUG(logger_, "mass: {} GeV", mass / 1_GeV);

      // CORSIKA 7 way
      // spectators inherit momentum from original projectile
      auto const p3lab = p3NucleonLab * nuclA;

      HEPEnergyType const Ekin = sqrt(p3lab.getSquaredNorm() + mass * mass) - mass;

      CORSIKA_LOGGER_DEBUG(logger_, "fragment momentum {}",
                           p3lab.getComponents() / 1_GeV);
      view.addSecondary(std::make_tuple(specCode, Ekin, p3lab.normalized()));
    }

    // add elastic nucleons to corsika stack
    // TODO: the elastic interaction could be external like the inelastic interaction,
    // e.g. use existing ElasticModel
    CORSIKA_LOGGER_DEBUG(logger_,
                         "adding elastically scattered nucleons to particle stack..");
    for (int j = 0; j < nElasticNucleons; ++j) {
      // TODO: sample proton or neutron
      Code const elaNucCode = Code::Proton;

      // CORSIKA 7 way
      // elastic nucleons inherit momentum from original projectile
      // neglecting momentum transfer in interaction
      auto const p3lab = p3NucleonLab;

      HEPEnergyType const mass = get_mass(elaNucCode);
      HEPEnergyType const Ekin = sqrt(p3lab.getSquaredNorm() + mass * mass) - mass;

      view.addSecondary(std::make_tuple(elaNucCode, Ekin, p3lab.normalized()));
    }

    // add inelastic interactions
    CORSIKA_LOGGER_DEBUG(logger_, "calculate inelastic nucleon-nucleon interactions..");
    for (int j = 0; j < nInelNucleons; ++j) {
      // TODO: sample neutron or proton
      auto const pCode = Code::Proton;
      HEPEnergyType const mass = get_mass(pCode);
      HEPEnergyType const Ekin = sqrt(p3NucleonLab.getSquaredNorm() + mass * mass) - mass;

      // temporarily add to stack, will be removed after interaction in DoInteraction
      CORSIKA_LOGGER_DEBUG(logger_, "inelastic interaction no. {}", j);
      typename TSecondaryView::inner_stack_value_type nucleonStack;
      Point const pDummy(boost.getOriginalCS(), {0_m, 0_m, 0_m});
      TimeType const tDummy = 0_ns;
      auto inelasticNucleon = nucleonStack.addParticle(
          std::make_tuple(pCode, Ekin, p3NucleonLab.normalized(), pDummy, tDummy));
      inelasticNucleon.setNode(view.getProjectile().getNode());

      // create inelastic interaction for each nucleon
      CORSIKA_LOGGER_TRACE(logger_, "calling HadronicInteraction...");
      // create new StackView for each of the nucleons
      TSecondaryView nucleon_secondaries(inelasticNucleon);
      // all inner hadronic event generator
      hadronicInteraction_.doInteraction(nucleon_secondaries, pCode, targetId, nucleonP4,
                                         targetP4);
      for (const auto& pSec : nucleon_secondaries) {

        auto const p3lab = pSec.getMomentum();
        Code const pid = pSec.getPID();
        HEPEnergyType const mass = get_mass(pid);
        HEPEnergyType const Ekin = sqrt(p3lab.getSquaredNorm() + mass * mass) - mass;
        view.addSecondary(std::make_tuple(pid, Ekin, p3lab.normalized()));
      }
    }
  }
} // namespace corsika::sibyll
