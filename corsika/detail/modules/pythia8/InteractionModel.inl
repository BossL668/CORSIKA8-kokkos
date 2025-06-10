/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <utility>
#include <stdexcept>

#include <boost/filesystem/path.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <fmt/core.h>
#include <cnpy.hpp>

#include <corsika/modules/pythia8/InteractionModel.hpp>

#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/utility/COMBoost.hpp>
#include <corsika/framework/utility/CrossSectionTable.hpp>
#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/NuclearComposition.hpp>

namespace corsika::pythia8 {

  inline InteractionModel::~InteractionModel() {}

  inline InteractionModel::InteractionModel(std::set<Code> const& stableParticles,
                                            boost::filesystem::path const& dataPath,
                                            bool const printListing)
      : crossSectionPPElastic_{loadPPTable(dataPath, "sig_el")}
      , crossSectionPPInelastic_{loadPPTable(dataPath, "sig_inel")}
      , print_listing_(printListing) {
    auto rndm = std::make_shared<corsika::pythia8::Random>();
    pythia_.setRndmEnginePtr(rndm);

    CORSIKA_LOG_INFO("Pythia8 reuse files: {}", dataPath.native());

    // projectile and target init to p Nitrogen to initialize pythia with angantyr
    pythia_.readString("Beams:allowIDASwitch = on");
    pythia_.settings.mode("Beams:idA",
                          static_cast<PDGCodeIntType>(get_PDG(Code::Proton)));
    pythia_.settings.mode("Beams:idB",
                          static_cast<PDGCodeIntType>(get_PDG(Code::Oxygen)));

    pythia_.readString("Beams:allowVariableEnergy = on");
    pythia_.readString("Beams:frameType = 1"); // CoM
    // initialize a maximum energy event
    pythia_.settings.parm("Beams:eCM", 400_TeV / 1_GeV);

    // needed when regular Pythia (not Angantyr) takes over for pp
    pythia_.readString("SoftQCD:all = on");

    /* reuse file settings
    reuseInit = 2: initialization data is reuse file, but if the file is not found,
    initialization fails reuseInit = 3: same, but if the file is not found, it will be
    generated and saved after normal initialization */
    pythia_.readString("MultipartonInteractions:reuseInit = 2");
    pythia_.readFile(dataPath.native() + "/pythia8313.cmnd");
    // switch on decays for all hadrons except the ones defined as tracked by C8
    if (!stableParticles.empty()) {
      pythia_.readString("HadronLevel:Decay = on");
      for (auto pCode : stableParticles)
        pythia_.particleData.mayDecay(static_cast<int>(get_PDG(pCode)), false);
    } else {
      // all hadrons stable
      pythia_.readString("HadronLevel:Decay = on");
    }
    // Reduce printout and relax energy-momentum conservation.
    pythia_.readString("Print:quiet = off");
    pythia_.readString("Check:epTolErr = 0.1");
    pythia_.readString("Check:epTolWarn = 0.0001");
    pythia_.readString("Check:mTolErr = 0.01");

    // Reduce statistics printout to relevant ones.
    pythia_.readString("Stat:showProcessLevel = off");
    pythia_.readString("Stat:showPartonLevel = off");

    // initialize
    // we can't test this block, LCOV_EXCL_START
    if (!pythia_.init())
      throw std::runtime_error("Pythia::InteractionModel: Initialization failed!");
    // LCOV_EXCL_STOP

    // create/load tables of cross sections
    for (Code const projectile : validProjectiles_) {
      for (Code const target : validTargets_) {
        if (xs_map_.find(projectile) == xs_map_.end()) {
          // projectile not mapped, load table directly
          auto const tablePath =
              dataPath / "pythia8_xs_npz" /
              fmt::format("xs_{}_{}.npz",
                          static_cast<PDGCodeIntType>(get_PDG(projectile)),
                          static_cast<PDGCodeIntType>(get_PDG(target)));
          auto const tables = cnpy::npz_load(tablePath.native());
          // NOTE: tables are calculated for plab. In C8 we we assume elab. starting at
          // plab=100GeV the difference is at most (1+m**2/p**2)=1.000088035
          auto const energies = tables.at("plab").as_vec<float>();
          auto const total_xs = tables.at("sig_tot").as_vec<float>();

          if (auto const e_size = energies.size(), xs_size = total_xs.size();
              xs_size != e_size) {
            throw std::runtime_error{
                fmt::format("Pythia8 table corrupt (plab size = {}; sig_tot size = {})",
                            e_size, xs_size)};
          }

          auto xs_it = boost::make_transform_iterator(total_xs.cbegin(), millibarn_mult);
          auto e_it_beg = boost::make_transform_iterator(energies.cbegin(), GeV_mult);
          decltype(e_it_beg) e_it_end =
              boost::make_transform_iterator(energies.cend(), GeV_mult);
          crossSectionTables_.insert(
              {std::pair{projectile, target},
               CrossSectionTable<InterpolationTransforms::Log>{
                   std::move(e_it_beg), std::move(e_it_end), std::move(xs_it)}});
        }
      }
    }
    // add nuclear projectiles
    for (int nuclA = 2; nuclA < 57; ++nuclA) {
      if (nuclA == 5 || nuclA == 8) continue; // skip missing projectiles fix me
      int const nuclZ = xs_nuc_map_.find(nuclA)->second;
      Code nucProj = get_nucleus_code(nuclA, nuclZ);
      for (Code const target : validTargets_) {
        if (xs_map_.find(nucProj) == xs_map_.end()) {
          // projectile not mapped, load table directly
          auto const tablePath =
              dataPath / "pythia8_xs_npz" /
              fmt::format("xs_{}_{}.npz", static_cast<PDGCodeIntType>(get_PDG(nucProj)),
                          static_cast<PDGCodeIntType>(get_PDG(target)));
          auto const tables = cnpy::npz_load(tablePath.native());
          // NOTE: tables are calculated for plab. In C8 we we assume elab. starting at
          // plab=100GeV the difference is at most (1+m**2/p**2)=1.000088035
          auto const energies = tables.at("plab").as_vec<float>();
          auto const total_xs = tables.at("sig_tot").as_vec<float>();

          if (auto const e_size = energies.size(), xs_size = total_xs.size();
              xs_size != e_size) {
            throw std::runtime_error{
                fmt::format("Pythia8 table corrupt (plab size = {}; sig_tot size = {})",
                            e_size, xs_size)};
          }

          auto xs_it = boost::make_transform_iterator(total_xs.cbegin(), millibarn_mult);
          auto e_it_beg = boost::make_transform_iterator(energies.cbegin(), GeV_mult);
          decltype(e_it_beg) e_it_end =
              boost::make_transform_iterator(energies.cend(), GeV_mult);
          crossSectionTables_.insert(
              {std::pair{nucProj, target},
               CrossSectionTable<InterpolationTransforms::Log>{
                   std::move(e_it_beg), std::move(e_it_end), std::move(xs_it)}});
        }
      }
    }
  }

  inline CrossSectionTable<InterpolationTransforms::Log> InteractionModel::loadPPTable(
      boost::filesystem::path const& dataPath, char const* key) {
    auto const ppTablePath =
        dataPath / "pythia8_xs_npz" /
        fmt::format("xs_{}_{}.npz", static_cast<PDGCodeIntType>(get_PDG(Code::Proton)),
                    static_cast<PDGCodeIntType>(get_PDG(Code::Proton)));
    auto const tables = cnpy::npz_load(ppTablePath.native());
    // NOTE: tables are calculated for plab. In C8 we we assume elab. starting at
    // plab=100GeV the difference is at most (1+m**2/p**2)=1.000088035
    auto const energies = tables.at("plab").as_vec<float>();
    auto const xs = tables.at(key).as_vec<float>();

    if (auto e_size = energies.size(), xs_size = xs.size(); xs_size != e_size) {
      throw std::runtime_error{fmt::format(
          "Pythia8 table corrupt (plab size = {}; xs size = {})", e_size, xs_size)};
    }

    auto xs_it = boost::make_transform_iterator(xs.cbegin(), millibarn_mult);
    auto e_it_beg = boost::make_transform_iterator(energies.cbegin(), GeV_mult);
    decltype(e_it_beg) e_it_end =
        boost::make_transform_iterator(energies.cend(), GeV_mult);
    return CrossSectionTable<InterpolationTransforms::Log>{
        std::move(e_it_beg), std::move(e_it_end), std::move(xs_it)};
  }

  inline bool InteractionModel::isValid(Code const projectileId, Code const targetId,
                                        HEPEnergyType const sqrtSNN) const {

    CORSIKA_LOG_DEBUG("pythia isValid: {} + {} at sqrtSNN = {} GeV", projectileId,
                      targetId, sqrtSNN / 1_GeV);
    if (is_nucleus(targetId)) CORSIKA_LOG_DEBUG("target: {}", get_nucleus_name(targetId));
    if (is_nucleus(projectileId))
      CORSIKA_LOG_DEBUG("projectile: {}", get_nucleus_name(projectileId));
    auto const Aprojectile = (is_nucleus(projectileId) ? get_nucleus_A(projectileId) : 1.);    
    if (Aprojectile == 5 || Aprojectile == 8) return false; // these nuclei do not exist
    
    if (!canInteract(projectileId)) return false;

    auto const mass_target =
        get_mass(targetId) / (is_nucleus(targetId) ? get_nucleus_A(targetId) : 1.);
    auto const mass_projectile =
        get_mass(projectileId) / Aprojectile;
    HEPEnergyType const labE =
        calculate_lab_energy(static_pow<2>(sqrtSNN), mass_projectile, mass_target);
    if (labE * Aprojectile < eKinMinLab_) return false;

    bool const validProjectile =
        (is_nucleus(projectileId)
             ? true
             : std::find(validProjectiles_.begin(), validProjectiles_.end(),
                         projectileId) != validProjectiles_.end());
    bool const validTarget = std::find(validTargets_.begin(), validTargets_.end(),
                                       targetId) != validTargets_.end();
    return validProjectile && validTarget;
  }

  inline bool InteractionModel::canInteract(Code const pCode) const {
    return is_hadron(pCode) || is_nucleus(pCode);
  }

  inline std::tuple<CrossSectionType, CrossSectionType>
  InteractionModel::getCrossSectionInelEla(Code const projectileId, Code const targetId,
                                           FourMomentum const& projectileP4,
                                           FourMomentum const& targetP4) const {
    // elastic cross sections only available for proton
    if (projectileId != Code::Proton || targetId != Code::Proton) {
      return std::tuple{CrossSectionType::zero(), CrossSectionType::zero()};
    }

    // NOTE: here we calculate SNN (per nucleon energy) AND S (per particle energy)
    // because isValid expects sqrtSNN as input but the tables need Elab
    auto const Aprojectile =
        (is_nucleus(projectileId) ? get_nucleus_A(projectileId) : 1.);
    auto const Atarget = (is_nucleus(targetId) ? get_nucleus_A(targetId) : 1.);
    auto const SNN = (projectileP4 / Aprojectile + targetP4 / Atarget).getNormSqr();
    HEPEnergyType const sqrtSNN = sqrt(SNN);
    if (!isValid(projectileId, targetId, sqrtSNN))
      return std::tuple{CrossSectionType::zero(), CrossSectionType::zero()};

    // read cross section from table
    // define system (the tables were created for lab. energies) since we cannot assume
    // the input is in the lab. frame we calculate the lab. energy from SNN =
    // (projectileP4 / Aprojectile + targetP4 / Atarget)**2 and Elab = S/(2. * mTarget /
    // Atarget) where A* is the number of nucleons in a nucleus. A* is 1 if projectile or
    // target are simple hadrons.
    HEPEnergyType const Elab = calculate_lab_energy(
        SNN, get_mass(projectileId) / Aprojectile, get_mass(targetId) / Atarget);

    CrossSectionType const xsEl = crossSectionPPElastic_.interpolate(Elab);
    CrossSectionType const xsInel = crossSectionPPInelastic_.interpolate(Elab);
    return std::pair{xsInel, xsEl};
  }

  inline CrossSectionType InteractionModel::getCrossSection(
      Code const projectileId, Code const targetId, FourMomentum const& projectileP4,
      FourMomentum const& targetP4) const {

    auto const Aprojectile =
        (is_nucleus(projectileId) ? get_nucleus_A(projectileId) : 1.);
    auto const Atarget = (is_nucleus(targetId) ? get_nucleus_A(targetId) : 1.);

    auto const SNN = (projectileP4 / Aprojectile + targetP4 / Atarget).getNormSqr();
    HEPEnergyType const sqrtSNN = sqrt(SNN);
    if (!isValid(projectileId, targetId, sqrtSNN)) return CrossSectionType::zero();

    // read cross section from table
    // define system (the tables were created for lab. energies) since we cannot assume
    // the input is in the lab. frame we calculate the lab. energy from SNN =
    // (projectileP4 / Aprojectile + targetP4 / Atarget)**2 and Elab = S/(2. * mTarget /
    // Atarget) where A* is the number of nucleons in a nucleus. A* is 1 if projectile or
    // target are simple hadrons.    
    auto const it = xs_map_.find(projectileId);
    auto const mapped_projectile = (it == xs_map_.end()) ? projectileId : it->second; // map antiparticles to their particle partners
    auto const mapped2_projectile = ( is_nucleus(mapped_projectile) ? get_nucleus_code(get_nucleus_A(mapped_projectile), xs_nuc_map_.find(get_nucleus_A(mapped_projectile))->second): mapped_projectile); // map isotopes to defined z
    auto const& table = crossSectionTables_.at(std::pair{mapped2_projectile, targetId});
    HEPEnergyType const Elab = calculate_lab_energy(
        SNN, get_mass(projectileId) / Aprojectile, get_mass(targetId) / Atarget) * Aprojectile;
    CORSIKA_LOG_DEBUG("pythia getCrossSection: {}+{} at Elab= {} GeV, sqrtSNN = {} GeV",
                      projectileId, get_nucleus_name(targetId), Elab / 1_GeV,
                      sqrtSNN / 1_GeV);
    return table.interpolate(Elab);
  }

  template <class TView>
  inline void InteractionModel::doInteraction(TView& view, Code const projectileId,
                                              Code const targetId,
                                              FourMomentum const& projectileP4,
                                              FourMomentum const& targetP4) {

    CORSIKA_LOG_DEBUG(
        "Pythia::InteractionModel: "
        "doInteraction: {} interaction? {} ",
        projectileId, corsika::pythia8::InteractionModel::canInteract(projectileId));

    // define system nucleon-nucleon
    auto const projectileP4NN =
        projectileP4 / (is_nucleus(projectileId) ? get_nucleus_A(projectileId) : 1);
    auto const targetP4NN =
        targetP4 / (is_nucleus(targetId) ? get_nucleus_A(targetId) : 1);
    auto const SNN = (projectileP4NN + targetP4NN).getNormSqr();
    auto const sqrtSNN = sqrt(SNN);
    HEPEnergyType const eProjectileLab = calculate_lab_energy(
        SNN, get_mass(projectileId), get_mass(targetId) / get_nucleus_A(targetId));

    CORSIKA_LOG_DEBUG("InteractionModel: ebeam lab: {} GeV", eProjectileLab / 1_GeV);
    CORSIKA_LOG_DEBUG(
        "InteractionModel: "
        " doInteraction: E(GeV): {}"
        " Ecm_NN(GeV): {}",
        eProjectileLab / 1_GeV, sqrtSNN / 1_GeV);

    if (!isValid(projectileId, targetId, sqrtSNN)) {
      throw std::runtime_error("invalid target,projectile,energy combination.");
    }

    COMBoost const boost{projectileP4NN, targetP4NN};
    auto const& rotCS = boost.getRotatedCS();

    // variables to talk to Pythia
    double const eCM_pythia = sqrtSNN / 1_GeV; // CoM energy hadron-Nucleon
    auto const py_projectileId = ( is_nucleus(projectileId) ? get_nucleus_code(get_nucleus_A(projectileId), xs_nuc_map_.find(get_nucleus_A(projectileId))->second): projectileId); // map isotopes to defined z
    double const idA_pythia = static_cast<double>(get_PDG(py_projectileId));
    double const idB_pythia = static_cast<double>(get_PDG(targetId));
    CORSIKA_LOG_DEBUG("re-configuring pythia idA={}, idB={}, ecm={} GeV", idA_pythia,
                      idB_pythia, eCM_pythia);
    // configure event
    pythia_.setBeamIDs(idA_pythia, idB_pythia);
    pythia_.setKinematics(eCM_pythia);

    // generate event (one chance only!)
    if (!pythia_.next()) {
      throw std::runtime_error("Pythia::InteractionModel: interaction failed!");
    }

    // LCOV_EXCL_START, we don't validate pythia8 internals
    if (print_listing_) {
      // print final state
      pythia_.event.list();
    }
    // LCOV_EXCL_STOP

    MomentumVector Plab_final{boost.getOriginalCS()};
    auto Elab_final = HEPEnergyType::zero();

    for (Pythia8::Particle const& p8p : pythia_.event) {
      // skip particles that have decayed and initial particles
      if (!p8p.isFinal()) continue;
      try {
        auto const volatile id = static_cast<PDGCode>(p8p.id());

        auto particleId = convert_from_PDG(id);
        // switch fragment nuclei to known isotopes        
        if(is_nucleus(particleId)){
          auto const Anew = get_nucleus_A(particleId);
          auto const Znew = xs_nuc_map_.find(Anew)->second;
          particleId = get_nucleus_code(Anew, Znew);
        }

        
        MomentumVector const pyPcom(
            rotCS, {p8p.px() * 1_GeV, p8p.py() * 1_GeV, p8p.pz() * 1_GeV});
        auto const pyP = boost.fromCoM(FourVector{p8p.e() * 1_GeV, pyPcom});

        HEPEnergyType const mass = get_mass(particleId);
        HEPEnergyType const Ekin =
            sqrt(pyP.getSpaceLikeComponents().getSquaredNorm() + mass * mass) - mass;

        // add to corsika stack
        auto pnew = view.addSecondary(
            std::make_tuple(particleId, Ekin, pyP.getSpaceLikeComponents().normalized()));

        Plab_final += pnew.getMomentum();
        Elab_final += pnew.getEnergy();
      }
      // irreproducible in tests, LCOV_EXCL_START
      catch (std::out_of_range const& ex) {
        CORSIKA_LOG_CRITICAL("Pythia ID {} unknown in C8", p8p.id());
        throw ex;
      }
      // LCOV_EXCL_STOP
    }

    pythia_.event.clear();

    CORSIKA_LOG_DEBUG(
        "conservation (all GeV): "
        "Elab_final= {}"
        ", Plab_final= {}",
        Elab_final / 1_GeV, (Plab_final / 1_GeV).getComponents());
  }

} // namespace corsika::pythia8
