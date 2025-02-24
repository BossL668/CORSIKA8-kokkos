/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/modules/epos/InteractionModel.hpp>
#include <corsika/modules/epos/EposStack.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/utility/COMBoost.hpp>
#include <corsika/modules/Random.hpp>
#include <corsika/framework/utility/CorsikaData.hpp>

#include <epos-public.hpp>
#include <datadir.hpp>

#include <string>
#include <tuple>
#include <cmath>

namespace corsika::epos {

  inline InteractionModel::InteractionModel(std::set<Code> vList,
                                            std::string const& dataPath,
                                            bool const epos_printout_on)
      : data_path_(dataPath)
      , epos_listing_(epos_printout_on) {
    // initialize Eposlhc
    corsika::connect_random_stream(RNG_, ::EPOS_LHC::set_rng_function);
    if (!isInitialized_) {
      isInitialized_ = true;
      if (dataPath == "") {
        data_path_ = (std::string(corsika_data("EPOS").c_str()) + "/").c_str();
      }
      initialize();
      if (vList.empty()) {
        CORSIKA_LOGGER_DEBUG(logger_,
                             "set all particles known to CORSIKA stable inside EPOS..");
        setParticleListStable(get_all_particles());
      } else {
        CORSIKA_LOGGER_DEBUG(logger_, "set specific particles stable inside EPOS..");
        setParticleListStable(vList);
      }
    }
  }

  inline void InteractionModel::setParticleListStable(std::set<Code> vPartList) const {
    for (auto& p : vPartList) {
      int const eid = convertToEposRaw(p);
      if (eid != 0) {
        // LCOV_EXCL_START
        // this is only a safeguard against messing up the epos internals by initializing
        // more than once.
        unsigned int const n_particles_stable_epos =
            ::EPOS_LHC::nodcy_->nrnody; // avoid waring -Wsign-compare
        if (n_particles_stable_epos < ::EPOS_LHC::mxnody) {
          CORSIKA_LOGGER_TRACE(logger_, "setting {} with EposId={} stable inside EPOS.",
                               p, eid);
          ::EPOS_LHC::nodcy_->nrnody = ::EPOS_LHC::nodcy_->nrnody + 1;
          ::EPOS_LHC::nodcy_->nody[::EPOS_LHC::nodcy_->nrnody - 1] = eid;
        } else {
          CORSIKA_LOGGER_ERROR(logger_, "List of stable particles too long for Epos!");
          throw std::runtime_error("Epos initialization error!");
        }
        // LCOV_EXCL_STOP
      } else {
        CORSIKA_LOG_TRACE(
            "particle conversion Corsika-->Epos not known for {}. Using {}. Setting "
            "unstable in Epos!",
            p, eid);
      }
    }
    CORSIKA_LOGGER_DEBUG(logger_, "set {} particles stable inside Epos",
                         ::EPOS_LHC::nodcy_->nrnody);
  }

  inline bool InteractionModel::isValid(Code const projectileId, Code const targetId,
                                        HEPEnergyType const sqrtS) const {
    //! eposlhc only accepts nuclei with X<=A<=Y as targets, or protons aka Hydrogen or
    //! neutrons (p,n == nucleon)
    if (!is_nucleus(targetId) && targetId != Code::Neutron && targetId != Code::Proton) {
      return false;
    }
    if (is_nucleus(targetId) && (get_nucleus_A(targetId) >= get_nucleus_A(maxNucleus_))) {
      return false;
    }
    if ((minEnergyCoM_ > sqrtS) || (sqrtS > maxEnergyCoM_)) { return false; }
    if (!epos::canInteract(projectileId)) { return false; }
    return true;
  }

  inline void InteractionModel::initialize() const {

    CORSIKA_LOGGER_DEBUG(logger_, "initializing...");

    // corsika7 ini
    int iarg = 0;
    ::EPOS_LHC::aaset_(iarg);

    // debug output settings
    ::EPOS_LHC::prnt1_->ish = 0;
    ::EPOS_LHC::prnt3_->iwseed = 0; // 1: printout seeds, 0: off
    ::EPOS_LHC::files_->ifch = 6;   // output unit, 6: screen

    // dummy set seeds for random number generator in epos. need to fool epos checks...
    // we will use external generator
    ::EPOS_LHC::cseed_->seedi = 1;
    ::EPOS_LHC::cseed_->seedj = 1;
    ::EPOS_LHC::cseed_->seedc = 1;

    ::EPOS_LHC::enrgy_->egymin = minEnergyCoM_ / 1_GeV; // 6.;
    ::EPOS_LHC::enrgy_->egymax = maxEnergyCoM_ / 1_GeV; // 2.e6;

    ::EPOS_LHC::lhcparameters_();

    ::EPOS_LHC::hadr6_->isigma = 0; // do not show cross section
    ::EPOS_LHC::hadr6_->isetcs = 3; /*  !option to obtain pomeron parameters
      ! 0.....determine parameters but do not use Kfit
      ! 1.....determine parameters and use Kfit
      ! else..get from table
      !         should be sufficiently detailed
      !          say iclegy1=1,iclegy2=99
      !         table is always done, more or less detailed!!!
      !and option to use cross section tables
      ! 2....tabulation
      ! 3....simulation
                               */
    ::EPOS_LHC::cjinti_->ionudi =
        1; // !include quasi elastic events but strict calculation of xs
    ::EPOS_LHC::cjinti_->iorsce = 0; // !color exchange turned on(1) or off(0)
    ::EPOS_LHC::cjinti_->iorsdf = 3; //  !droplet formation turned on(>0) or off(0)
    ::EPOS_LHC::cjinti_->iorshh =
        0; //    !other hadron-hadron int. turned on(1) or off(0)

    ::EPOS_LHC::othe1_->istore = 0; // do not produce epos output file
    ::EPOS_LHC::nucl6_->infragm =
        2; // 0: keep free nucleons in fragmentation,1: one fragment, 2: fragmentation

    ::EPOS_LHC::othe2_->iframe = 11; // cms frame

    // decay settings
    // activate decays in epos for particles defined by set_stable/set_unstable
    // ::EPOS_LHC::othe2_->.idecay = 0; // no decays in epos

    // set paths to tables in corsika data
    using EPOSDatadir = datadir<500>;
    EPOSDatadir BASE(data_path_);
    strcpy(::EPOS_LHC::fname_->fnnx, BASE.data());
    ::EPOS_LHC::nfname_->nfnnx = BASE.length();

    EPOSDatadir TL(data_path_ + "epos.initl");
    strcpy(::EPOS_LHC::fname_->fnii, TL.data());
    ::EPOS_LHC::nfname_->nfnii = TL.length();

    EPOSDatadir EV(data_path_ + "epos.iniev");
    strcpy(::EPOS_LHC::fname_->fnie, EV.data());
    ::EPOS_LHC::nfname_->nfnie = EV.length();

    EPOSDatadir RJ(data_path_ + "epos.inirj"); // lhcparameters adds ".lhc"
    strcpy(::EPOS_LHC::fname_->fnrj, RJ.data());
    ::EPOS_LHC::nfname_->nfnrj = RJ.length();

    EPOSDatadir CS(data_path_ + "epos.inics"); // lhcparameters adds ".lhc"
    strcpy(::EPOS_LHC::fname_->fncs, CS.data());
    ::EPOS_LHC::nfname_->nfncs = CS.length();

    // initializes maximum energy and mass
    initializeEventCoM(
        maxNucleus_, get_nucleus_A(maxNucleus_), get_nucleus_Z(maxNucleus_), maxNucleus_,
        get_nucleus_A(maxNucleus_), get_nucleus_Z(maxNucleus_), maxEnergyCoM_);
  }

  inline void InteractionModel::initializeEventCoM(Code const idBeam, int const iBeamA,
                                                   int const iBeamZ, Code const idTarget,
                                                   int const iTargetA, int const iTargetZ,
                                                   HEPEnergyType const EcmNN) const {
    CORSIKA_LOGGER_TRACE(logger_,
                         "initialize event in CoM frame!"
                         " Ecm={}",
                         EcmNN);
    ::EPOS_LHC::lept1_->engy = -1.;
    ::EPOS_LHC::enrgy_->ecms = -1.;
    ::EPOS_LHC::enrgy_->elab = -1.;
    ::EPOS_LHC::enrgy_->ekin = -1.;
    ::EPOS_LHC::hadr1_->pnll = -1.;

    ::EPOS_LHC::enrgy_->ecms = EcmNN / 1_GeV; // -> c.m.s. frame

    CORSIKA_LOGGER_TRACE(logger_, "inside EPOS: Ecm={}, Elab={}",
                         ::EPOS_LHC::enrgy_->ecms, ::EPOS_LHC::enrgy_->elab);

    configureParticles(idBeam, iBeamA, iBeamZ, idTarget, iTargetA, iTargetZ);
    ::EPOS_LHC::ainit_();
  }

  inline void InteractionModel::configureParticles(Code const idBeam, int const iBeamA,
                                                   int const iBeamZ, Code const idTarget,
                                                   int const iTargetA,
                                                   int const iTargetZ) const {
    CORSIKA_LOGGER_TRACE(logger_,
                         "setting "
                         "Beam={}, "
                         "BeamA={}, "
                         "BeamZ={}, "
                         "Target={}, "
                         "TargetA={}, "
                         "TargetZ={} ",
                         idBeam, iBeamA, iBeamZ, idTarget, iTargetA, iTargetZ);

    if (is_nucleus(idBeam)) {
      ::EPOS_LHC::hadr25_->idprojin = convertToEposRaw(Code::Proton);
      ::EPOS_LHC::nucl1_->laproj = iBeamZ;
      ::EPOS_LHC::nucl1_->maproj = iBeamA;
    } else {
      ::EPOS_LHC::hadr25_->idprojin = convertToEposRaw(idBeam);
      ::EPOS_LHC::nucl1_->laproj = -1;
      ::EPOS_LHC::nucl1_->maproj = 1;
    }

    if (is_nucleus(idTarget)) {
      ::EPOS_LHC::hadr25_->idtargin = convertToEposRaw(Code::Proton);
      ::EPOS_LHC::nucl1_->matarg = iTargetA;
      ::EPOS_LHC::nucl1_->latarg = iTargetZ;
    } else if (idTarget == Code::Proton || idTarget == Code::Hydrogen) {
      ::EPOS_LHC::hadr25_->idtargin = convertToEposRaw(Code::Proton);
      ::EPOS_LHC::nucl1_->matarg = 1;
      ::EPOS_LHC::nucl1_->latarg = -1;
    } else if (idTarget == Code::Neutron) {
      ::EPOS_LHC::hadr25_->idtargin = convertToEposRaw(Code::Neutron);
      ::EPOS_LHC::nucl1_->matarg = 1;
      ::EPOS_LHC::nucl1_->latarg = -1;
    }

    CORSIKA_LOGGER_TRACE(logger_,
                         "inside EPOS: "
                         "Id beam={}, "
                         "Z beam={}, "
                         "A beam={}, "
                         "XS beam={}, "
                         "Id target={}, "
                         "Z target={}, "
                         "A target={}, "
                         "XS target={} ",
                         ::EPOS_LHC::hadr25_->idprojin, ::EPOS_LHC::nucl1_->laproj,
                         ::EPOS_LHC::nucl1_->maproj, ::EPOS_LHC::had10_->iclpro,
                         ::EPOS_LHC::hadr25_->idtargin, ::EPOS_LHC::nucl1_->latarg,
                         ::EPOS_LHC::nucl1_->matarg, ::EPOS_LHC::had10_->icltar);
  }

  inline InteractionModel::~InteractionModel() {
    CORSIKA_LOGGER_DEBUG(logger_, "n={} ", count_);
  }

  inline std::tuple<CrossSectionType, CrossSectionType>
  InteractionModel::calcCrossSectionCoM(Code const BeamId, int const BeamA,
                                        int const BeamZ, Code const TargetId,
                                        int const TargetA, int const TargetZ,
                                        const HEPEnergyType EnergyCOM) const {
    CORSIKA_LOGGER_DEBUG(logger_,
                         "calcCrossSection: input:"
                         " beamId={}, beamA={}, beamZ={}"
                         " target={}, targetA={}, targetZ={}"
                         " Ecm={:4.3f} GeV,",
                         BeamId, BeamA, BeamZ, TargetId, TargetA, TargetZ,
                         EnergyCOM / 1_GeV);

    const int iBeam = epos::getEposXSCode(
        BeamId); // 0 (can not interact, 1: proton-like, 2: pion-like, 3:kaon-like)

    CORSIKA_LOGGER_TRACE(logger_,
                         "projectile cross section type={} "
                         "(0: cannot interact, 1:pion, 2:baryon, 3:kaon)",
                         iBeam);
    // reset beam particle // (1: pion-like, 2: proton-like, 3:kaon-like)
    if (iBeam == 1)
      initializeEventCoM(Code::PiPlus, BeamA, BeamZ, TargetId, TargetA, TargetZ,
                         EnergyCOM);
    else if (iBeam == 2)
      initializeEventCoM(Code::Proton, BeamA, BeamZ, TargetId, TargetA, TargetZ,
                         EnergyCOM);
    else if (iBeam == 3)
      initializeEventCoM(Code::KPlus, BeamA, BeamZ, TargetId, TargetA, TargetZ,
                         EnergyCOM);

    double sigProd, sigEla = 0;
    float sigTot1, sigProd1, sigCut1 = 0;
    if (!is_nucleus(TargetId) && !is_nucleus(BeamId)) {
      sigProd = ::EPOS_LHC::hadr5_->sigine;
      sigEla = ::EPOS_LHC::hadr5_->sigela;
    } else {
      // calculate from model, SLOW:
      float sigQEla1 = 0; // target fragmentation/excitation
      ::EPOS_LHC::crseaaepos_(sigTot1, sigProd1, sigCut1, sigQEla1);
      sigProd = sigProd1;
      // sigEla not properly defined here
    }
    CORSIKA_LOGGER_DEBUG(logger_,
                         "calcCrossSectionCoM: output:"
                         " sigProd={} mb,"
                         " sigEla={} mb",
                         sigProd, sigEla);

    return std::make_tuple(sigProd * 1_mb, sigEla * 1_mb);
  }

  inline std::tuple<CrossSectionType, CrossSectionType>
  InteractionModel::readCrossSectionTableLab(Code const BeamId, int const BeamA,
                                             int const BeamZ, Code const TargetId,
                                             HEPEnergyType const EnergyLab) const {
    CORSIKA_LOGGER_DEBUG(logger_,
                         "readCrossSectionTableLab: input: "
                         "beamId={}, "
                         "beamA={}, "
                         "beamZ={} "
                         "targetId={}, "
                         "ELab={:12.2f} GeV,",
                         BeamId, BeamA, BeamZ, TargetId, EnergyLab / 1_GeV);

    // read cross section from epos internal tables
    int Abeam = 0;
    float Ekin = -1;

    if (is_nucleus(BeamId)) {
      Abeam = BeamA;
      // kinetic energy per nucleon
      Ekin = (EnergyLab / Abeam - constants::nucleonMass) / 1_GeV;
    } else {
      ::EPOS_LHC::hadr2_->idproj = convertToEposRaw(BeamId);
      int const iBeam = epos::getEposXSCode(
          BeamId); // 0 (can not interact, 1: pion-like, 2: proton-like, 3:kaon-like)
      CORSIKA_LOGGER_TRACE(logger_,
                           "readCrossSectionTableLab: projectile cross section type={} "
                           "(0: cannot interact, 1:pion, 2:baryon, 3:kaon)",
                           iBeam);

      ::EPOS_LHC::had10_->iclpro = iBeam;
      Abeam = 1;
      Ekin = (EnergyLab - get_mass(BeamId)) / 1_GeV;
    }

    int Atarget = 1;
    if (is_nucleus(TargetId)) { Atarget = get_nucleus_A(TargetId); }

    int iMode = 3; // 0: air, >0 not air

    CORSIKA_LOGGER_TRACE(logger_,
                         "readCrossSectionTableLab: inside Epos "
                         "beamId={}, beamXS={}",
                         ::EPOS_LHC::hadr2_->idproj, ::EPOS_LHC::had10_->iclpro);

    CORSIKA_LOGGER_TRACE(logger_,
                         "readCrossSectionTableLab: calling Epos cross section with"
                         "Ekin = {}, Abeam = {}, Atarget = {}, iMode = {}",
                         Ekin, Abeam, Atarget, iMode);

    // cross section from table, FAST
    float sigProdEpos = ::EPOS_LHC::eposcrse_(Ekin, Abeam, Atarget, iMode);
    // sig-el from analytic calculation, no fast
    float sigElaEpos = ::EPOS_LHC::eposelacrse_(Ekin, Abeam, Atarget, iMode);

    CORSIKA_LOGGER_TRACE(logger_,
                         "readCrossSectionTableLab: result: sigProd = {}, sigEla = {}",
                         sigProdEpos, sigElaEpos);

    return std::make_tuple(sigProdEpos * 1_mb, sigElaEpos * 1_mb);
  }

  inline std::tuple<CrossSectionType, CrossSectionType>
  InteractionModel::getCrossSectionInelEla(Code const projectileId, Code const targetId,
                                           FourMomentum const& projectileP4,
                                           FourMomentum const& targetP4) const {
    auto const sqrtS2 = (projectileP4 + targetP4).getNormSqr();
    auto const sqrtS = sqrt(sqrtS2);

    if (!isValid(projectileId, targetId, sqrtS)) {
      return {CrossSectionType::zero(), CrossSectionType::zero()};
    }
    HEPEnergyType const Elab = (sqrtS2 - static_pow<2>(get_mass(projectileId)) -
                                static_pow<2>(get_mass(targetId))) /
                               (2 * get_mass(targetId));
    int beamA = 1;
    int beamZ = 1;
    if (is_nucleus(projectileId)) {
      beamA = get_nucleus_A(projectileId);
      beamZ = get_nucleus_Z(projectileId);
    }

    CORSIKA_LOGGER_DEBUG(logger_,
                         "getCrossSectionLab: input:"
                         " beamId={}, beamA={}, beamZ={}"
                         " target={}"
                         " ELab={:4.3f} GeV, sqrtS={}",
                         projectileId, beamA, beamZ, targetId, Elab / 1_GeV,
                         sqrtS / 1_GeV);
    return readCrossSectionTableLab(projectileId, beamA, beamZ, targetId, Elab);
  }

  template <typename TSecondaryView>
  inline void InteractionModel::doInteraction(TSecondaryView& view,
                                              Code const projectileId,
                                              Code const targetId,
                                              FourMomentum const& projectileP4,
                                              FourMomentum const& targetP4) {

    count_ = count_ + 1;

    // define nucleon-nucleon center-of-mass frame
    auto const projectileP4NN =
        projectileP4 / (is_nucleus(projectileId) ? get_nucleus_A(projectileId) : 1);
    auto const targetP4NN =
        targetP4 / (is_nucleus(targetId) ? get_nucleus_A(targetId) : 1);
    auto const SNN = (projectileP4NN + targetP4NN).getNormSqr();
    HEPEnergyType const sqrtSNN = sqrt(SNN);
    if (!isValid(projectileId, targetId, sqrtSNN)) {
      throw std::runtime_error("invalid projectile/target/energy combination.");
    }
    HEPEnergyType const Elab = (SNN - static_pow<2>(get_mass(projectileId)) -
                                static_pow<2>(get_mass(targetId))) /
                               (2 * get_mass(targetId));

    // system of initial-state
    COMBoost const boost(projectileP4NN, targetP4NN);

    auto const& originalCS = boost.getOriginalCS();
    auto const& csPrime =
        boost.getRotatedCS(); // z is along the CM motion (projectile, in Cascade)

    CORSIKA_LOGGER_DEBUG(logger_,
                         "doInteraction: interaction, projectile id={}, E={}, p3={} ",
                         projectileId, projectileP4.getTimeLikeComponent(),
                         projectileP4.getSpaceLikeComponents());
    CORSIKA_LOGGER_DEBUG(
        logger_, "doInteraction: projectile per-nucleon ENN={}, p3NN={} ",
        projectileP4NN.getTimeLikeComponent(), projectileP4NN.getSpaceLikeComponents());
    CORSIKA_LOGGER_DEBUG(
        logger_, "doInteraction: interaction, target id={}, E={}, p3={} ", targetId,
        targetP4.getTimeLikeComponent(), targetP4.getSpaceLikeComponents());
    CORSIKA_LOGGER_DEBUG(logger_, "doInteraction: target per-nucleon ENN={}, p3NN={} ",
                         targetP4NN.getTimeLikeComponent(),
                         targetP4NN.getSpaceLikeComponents());
    CORSIKA_LOGGER_DEBUG(logger_, "doInteraction: Elab={}, sqrtSNN={} ", Elab, sqrtSNN);

    int beamA = 1;
    int beamZ = 1;
    if (is_nucleus(projectileId)) {
      beamA = get_nucleus_A(projectileId);
      beamZ = get_nucleus_Z(projectileId);
      CORSIKA_LOGGER_DEBUG(logger_, "projectile: A={}, Z={} ", beamA, beamZ);
    }

    // // from corsika7 interface
    // // NEXLNK-part
    int targetA = 1;
    int targetZ = 1;
    if (is_nucleus(targetId)) {
      targetA = get_nucleus_A(targetId);
      targetZ = get_nucleus_Z(targetId);
      CORSIKA_LOGGER_DEBUG(logger_, "target: A={}, Z={} ", targetA, targetZ);
    }
    initializeEventCoM(projectileId, beamA, beamZ, targetId, targetA, targetZ, sqrtSNN);

    // create event
    int iarg = 1;
    ::EPOS_LHC::aepos_(iarg);
    ::EPOS_LHC::afinal_();

    if (epos_listing_) { // LCOV_EXCL_START
      char nam[9] = "EPOSLHC&";
      ::EPOS_LHC::alistf_(nam, 9);
    } // LCOV_EXCL_STOP

    // NSTORE-part

    MomentumVector P_final(originalCS, {0.0_GeV, 0.0_GeV, 0.0_GeV});
    HEPEnergyType E_final = 0_GeV;

    // secondaries
    EposStack es;
    CORSIKA_LOGGER_DEBUG(logger_, "number of entries on Epos stack: {}", es.getSize());
    for (auto& psec : es) {
      if (!psec.isFinal()) continue;

      auto momentum = psec.getMomentum(csPrime);
      // transform particle output to frame defined by input 4-momenta
      auto const P4output = boost.fromCoM(FourVector{psec.getEnergy(), momentum});
      auto p3output = P4output.getSpaceLikeComponents();
      p3output.rebase(originalCS); // transform back into standard lab frame

      EposCode const eposId = psec.getPID();
      Code const pid = epos::convertFromEpos(eposId);
      HEPEnergyType const mass = get_mass(pid);
      HEPEnergyType const Ekin = sqrt(p3output.getSquaredNorm() + mass * mass) - mass;
      CORSIKA_LOGGER_TRACE(logger_,
                           " id= {}"
                           " p= {}",
                           pid, p3output.getComponents() / 1_GeV);

      auto pnew = view.addSecondary(std::make_tuple(pid, Ekin, p3output.normalized()));
      P_final += pnew.getMomentum();
      E_final += pnew.getEnergy();
    }
    CORSIKA_LOGGER_DEBUG(
        logger_,
        "conservation (all GeV): Ecm_final= n/a" /* << Ecm_final / 1_GeV*/
        ", E_final={} GeV"
        ", P_final={} GeV"
        ", no. of particles={}",
        E_final / 1_GeV, (P_final / 1_GeV).getComponents(), view.getSize());
  }
} // namespace corsika::epos
