/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/modules/Random.hpp>
#include <corsika/modules/qgsjetII/InteractionModel.hpp>
#include <corsika/modules/qgsjetII/ParticleConversion.hpp>
#include <corsika/modules/qgsjetII/QGSJetIIFragmentsStack.hpp>
#include <corsika/modules/qgsjetII/QGSJetIIStack.hpp>

#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/utility/COMBoost.hpp>

#include <sstream>
#include <tuple>

#include <qgsjet-II-04.hpp>

namespace corsika::qgsjetII {

  inline InteractionModel::InteractionModel(boost::filesystem::path const dataPath) {
    // initialize QgsjetII
    corsika::connect_random_stream(rng_, ::qgsjetII::set_rng_function);

    static bool initialized = false;
    if (!initialized) {
      CORSIKA_LOG_DEBUG("Reading QGSJetII data tables from {}", dataPath);
      qgset_();
      datadir DIR(dataPath.string() + "/");
      qgaini_(DIR.data);
      initialized = true;
    }
  }

  inline InteractionModel::~InteractionModel() {
    CORSIKA_LOG_DEBUG("QgsjetII::InteractionModel n= {}", count_);
  }

  inline bool InteractionModel::isValid(Code const projectileId, Code const targetId,
                                        HEPEnergyType const sqrtS) const {

    if (sqrtS < sqrtSmin_) { return false; }
    if (is_nucleus(targetId)) {
      size_t iTarget = get_nucleus_A(targetId);
      if (iTarget > int(maxMassNumber_) || iTarget <= 0) { return false; }
    } else if (targetId != Proton::code) {
      return false;
    }

    if (is_nucleus(projectileId)) {
      size_t iProjectile = get_nucleus_A(projectileId);
      if (iProjectile > int(maxMassNumber_) || iProjectile <= 0) { return false; }
    } else if (!is_hadron(projectileId)) {
      return false;
    }
    return true;
  }

  inline CrossSectionType InteractionModel::getCrossSection(
      Code const projectileId, Code const targetId, FourMomentum const& projectileP4,
      FourMomentum const& targetP4) const {

    if (!corsika::qgsjetII::canInteract(projectileId)) {
      return CrossSectionType::zero();
    }

    auto const AfactorProjectile =
        is_nucleus(projectileId) ? get_nucleus_A(projectileId) : 1;
    auto const AfactorTarget = is_nucleus(targetId) ? get_nucleus_A(targetId) : 1;

    // define projectile, in lab frame
    auto const S = (projectileP4 + targetP4).getNormSqr();
    auto const SNN =
        (projectileP4 / AfactorProjectile + targetP4 / AfactorTarget).getNormSqr();
    auto const sqrtSNN = sqrt(SNN);
    if (!isValid(projectileId, targetId, sqrtSNN)) { return CrossSectionType::zero(); }

    auto const projMass = get_mass(projectileId);
    auto const targetMass = get_mass(targetId);

    // lab-frame energy per projectile nucleon as required by qgsect()
    HEPEnergyType const ElabN =
        calculate_lab_energy(S, projMass, targetMass) / AfactorProjectile;

    int const iBeam = static_cast<QgsjetIIXSClassIntType>(
        corsika::qgsjetII::getQgsjetIIXSCode(projectileId));

    CORSIKA_LOG_DEBUG(
        "QgsjetII::getCrossSection Elab= {} GeV iBeam= {}"
        " iProjectile= {} iTarget= {}",
        ElabN / 1_GeV, iBeam, AfactorProjectile, AfactorTarget);
    double const ElabNGeV{ElabN * (1 / 1_GeV)};
    double const sigProd = qgsect_(ElabNGeV, iBeam, AfactorProjectile, AfactorTarget);
    CORSIKA_LOG_DEBUG("QgsjetII::getCrossSection sigProd= {} mb", sigProd);
    return sigProd * 1_mb;
  }

  inline std::tuple<CrossSectionType, CrossSectionType>
  InteractionModel::getCrossSectionInelEla(Code projCode, Code targetCode,
                                           FourMomentum const& proj4mom,
                                           FourMomentum const& target4mom) const {
    return {getCrossSection(projCode, targetCode, proj4mom, target4mom),
            CrossSectionType::zero()};
  }

  template <typename TSecondaries>
  inline void InteractionModel::doInteraction(TSecondaries& view, Code const projectileId,
                                              Code const targetId,
                                              FourMomentum const& projectileP4,
                                              FourMomentum const& targetP4) {

    CORSIKA_LOG_DEBUG(
        "ProcessQgsjetII: "
        "doInteraction: {} interaction possible? {}",
        projectileId, corsika::qgsjetII::canInteract(projectileId));

    // define projectile, in lab frame
    auto const AfactorProjectile =
        is_nucleus(projectileId) ? get_nucleus_A(projectileId) : 1;
    auto const AfactorTarget = is_nucleus(targetId) ? get_nucleus_A(targetId) : 1;

    auto const S = (projectileP4 + targetP4).getNormSqr();
    auto const SNN =
        (projectileP4 / AfactorProjectile + targetP4 / AfactorTarget).getNormSqr();
    auto const sqrtSNN = sqrt(SNN);

    if (!corsika::qgsjetII::canInteract(projectileId) ||
        !isValid(projectileId, targetId, sqrtSNN)) {
      throw std::runtime_error(fmt::format(
          "invalid target [{}]/ projectile [{}] /energy [{} GeV] combination.",
          get_name(targetId, full_name{}), get_name(projectileId, full_name{}),
          sqrtSNN / 1_GeV));
    }

    auto const projMass = get_mass(projectileId);
    auto const targetMass = get_mass(targetId);

    // lab-frame energy per projectile nucleon
    HEPEnergyType const Elab = calculate_lab_energy(S, projMass, targetMass);
    auto const ElabN = Elab / AfactorProjectile;

    CORSIKA_LOG_DEBUG("ebeam lab: {} GeV per projectile nucleon", ElabN / 1_GeV);

    int const targetMassNumber = AfactorTarget;
    CORSIKA_LOG_DEBUG("target: {}, qgsjetII code/A: {}", targetId, targetMassNumber);

    // select QGSJetII internal projectile type
    QgsjetIIHadronType qgsjet_hadron_type = qgsjetII::getQgsjetIIHadronType(projectileId);
    if (qgsjet_hadron_type == QgsjetIIHadronType::NucleusType) {
      qgsjet_hadron_type = bernoulli_(rng_) ? QgsjetIIHadronType::ProtonType
                                            : QgsjetIIHadronType::NeutronType;
    } else if (qgsjet_hadron_type == QgsjetIIHadronType::NeutralLightMesonType) {
      // from conex: replace pi0 or rho0 with pi+/pi- in alternating sequence
      qgsjet_hadron_type = alternate_;
      alternate_ =
          (alternate_ == QgsjetIIHadronType::PiPlusType ? QgsjetIIHadronType::PiMinusType
                                                        : QgsjetIIHadronType::PiPlusType);
    }

    count_++;
    int const qgsjet_hadron_type_int =
        static_cast<QgsjetIICodeIntType>(qgsjet_hadron_type);
    CORSIKA_LOG_DEBUG(
        "qgsjet_hadron_type_int={} projectileMassNumber={} targetMassNumber={}",
        qgsjet_hadron_type_int, AfactorProjectile, AfactorTarget);
    qgini_(ElabN / 1_GeV, qgsjet_hadron_type_int, AfactorProjectile, AfactorTarget);
    qgconf_();

    CoordinateSystemPtr const& rootCS = get_root_CoordinateSystem();

    // bookkeeping
    MomentumVector Plab_final(rootCS, {0.0_GeV, 0.0_GeV, 0.0_GeV});
    HEPEnergyType Elab_final = 0_GeV;

    // to read the secondaries
    // define rotation to and from CoM frame
    // CoM frame definition in QgsjetII projectile: +z

    // QGSJetII, both, in input and output only considers the lab frame with a target at
    // rest.

    // system of initial-state
    COMBoost boost(projectileP4 / AfactorProjectile, targetP4 / AfactorTarget);

    auto const& originalCS = boost.getOriginalCS();
    auto const& csPrime =
        boost.getRotatedCS(); // z is along the CM motion (projectile, in Cascade)

    HEPMomentumType const pLabMag = calculate_momentum(Elab, get_mass(projectileId));
    MomentumVector const pLab{csPrime, {0_eV, 0_eV, pLabMag}};

    // internal QGSJetII system: hadron-nucleon lab. frame!
    COMBoost const boostInternal({Elab / AfactorProjectile, pLab / AfactorProjectile},
                                 targetMass / AfactorTarget); // felix

    // fragments
    QGSJetIIFragmentsStack qfs;
    for (auto& fragm : qfs) {
      int const A = fragm.getFragmentSize();
      if (A == 1) { // nucleon
        Code const idFragm = bernoulli_(rng_) ? Code::Proton : Code::Neutron;

        HEPMassType const nucleonMass = get_mass(idFragm);
        // no pT, fragments just go forward
        MomentumVector const momentum{
            csPrime, {0_eV, 0_eV, calculate_momentum(ElabN, nucleonMass)}};

        // this is not "CoM" here, but rather the system defined by projectile+target,
        // which in Cascade-mode is already lab
        auto const P4com = boostInternal.toCoM(FourVector{ElabN, momentum});
        auto const P4output = boost.fromCoM(P4com);
        auto p3output = P4output.getSpaceLikeComponents();
        p3output.rebase(originalCS); // transform back into standard lab frame

        HEPEnergyType const Ekin =
            calculate_kinetic_energy(p3output.getNorm(), nucleonMass);

        CORSIKA_LOG_DEBUG(
            "secondary fragment> id= {}"
            " p={}",
            idFragm, p3output.getComponents());

        auto pnew =
            view.addSecondary(std::make_tuple(idFragm, Ekin, p3output.normalized()));
        Plab_final += pnew.getMomentum();
        Elab_final += pnew.getEnergy();

      } else { // nucleus, A>1

        int Z = 0;
        switch (A) {
          case 2: // deuterium
            Z = 1;
            break;
          case 3: // tritium
            Z = 1;
            break;
          case 4: // helium
            Z = 2;
            break;
          default: // nucleus
          {
            Z = int(A / 2.15 + 0.7);
          }
        }

        Code const idFragm = get_nucleus_code(A, Z);
        HEPEnergyType const mass = get_mass(idFragm);
        // no pT, frgments just go forward
        MomentumVector momentum{csPrime,
                                {0.0_GeV, 0.0_GeV, calculate_momentum(ElabN * A, mass)}};

        // this is not "CoM" here, but rather the system defined by projectile+target,
        // which in Cascade-mode is already lab
        auto const P4com = boostInternal.toCoM(FourVector{ElabN * A, momentum});
        auto const P4output = boost.fromCoM(P4com);
        auto p3output = P4output.getSpaceLikeComponents();
        p3output.rebase(originalCS); // transform back into standard lab frame

        HEPEnergyType const Ekin = calculate_kinetic_energy(p3output.getNorm(), mass);

        CORSIKA_LOG_DEBUG(
            "secondary fragment> id={}"
            " p={}"
            " A={}"
            " Z={}",
            idFragm, p3output.getComponents(), A, Z);

        auto pnew =
            view.addSecondary(std::make_tuple(idFragm, Ekin, p3output.normalized()));
        Plab_final += pnew.getMomentum();
        Elab_final += pnew.getEnergy();
      }
    }

    // secondaries
    QGSJetIIStack qs;
    for (auto& psec : qs) {
      auto momentum = psec.getMomentum(csPrime);
      // this is not "CoM" here, but rather the system defined by projectile+target,
      // which in Cascade-mode is already lab
      auto const P4com = boostInternal.toCoM(FourVector{psec.getEnergy(), momentum});
      auto const P4output = boost.fromCoM(P4com);
      auto p3output = P4output.getSpaceLikeComponents();
      p3output.rebase(originalCS); // transform back into standard lab frame

      Code const pid = corsika::qgsjetII::convertFromQgsjetII(psec.getPID());
      HEPEnergyType const mass = get_mass(pid);
      HEPEnergyType const Ekin = calculate_kinetic_energy(p3output.getNorm(), mass);

      CORSIKA_LOG_DEBUG("secondary> id= {}, p= {}", pid, p3output.getComponents());
      auto pnew = view.addSecondary(std::make_tuple(pid, Ekin, p3output.normalized()));
      Plab_final += pnew.getMomentum();
      Elab_final += pnew.getEnergy();
    }
    CORSIKA_LOG_DEBUG(
        "conservation (all GeV): Ecm_final= n/a " /* << Ecm_final / 1_GeV*/
        ", Elab_final={} "
        ", Plab_final={}"
        ", N_wounded,targ={}"
        ", N_wounded,proj={}"
        ", N_fragm,proj={}",
        Elab_final / 1_GeV, (Plab_final / 1_GeV).getComponents(),
        QGSJetIIFragmentsStackData::getWoundedNucleonsTarget(),
        QGSJetIIFragmentsStackData::getWoundedNucleonsProjectile(), qfs.getSize());
  }
} // namespace corsika::qgsjetII
