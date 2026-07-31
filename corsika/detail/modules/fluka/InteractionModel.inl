/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cstdlib>
#include <vector>
#include <iterator>
#include <set>
#include <utility>
#include <string_view>

#include <boost/iterator/zip_iterator.hpp>
#include <Eigen/Dense>

#include <corsika/media/Environment.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/modules/Random.hpp>
#include <corsika/modules/fluka/ParticleConversion.hpp>

#include <FLUKA.hpp>

namespace corsika::fluka {
  inline InteractionModel::InteractionModel(std::set<Code> const& nuccomp)
      : materials_{genFlukaMaterials(nuccomp)}
      , cumsgx_{std::make_unique<double[]>(materials_.size() * 3)} {
    CORSIKA_LOGGER_INFO(logger_, "FLUKA version {}", ::fluka::get_version());
    for (auto const& [code, matno] : materials_) {
      CORSIKA_LOGGER_DEBUG(logger_, "FLUKA material initialization: {} -> {}",
                           get_name(code, full_name{}), matno);
    }

    if (int const ndmhep = ::fluka::ndmhep_(); ::fluka::nmxhep != ndmhep) {
      CORSIKA_LOGGER_CRITICAL(logger_, "HEPEVT dimension mismatch. FLUKA reports %d",
                              ndmhep);
      throw std::runtime_error{"FLUKA HEPEVT dimension mismatch"};
    }

    corsika::connect_random_stream(RNG_, ::fluka::set_rng_function);
  }

  inline bool InteractionModel::isValid(Code projectileID, int material,
                                        HEPEnergyType /*sqrtS*/) const {
    if (!fluka::canInteract(projectileID)) {
      // invalid projectile
      return false;
    }

    if (material < 0) {
      // invalid/uninitialized target
      return false;
    }

    // TODO: check validity range of sqrtS
    return true;
  }

  inline bool InteractionModel::isValid(Code projectileID, Code targetID,
                                        HEPEnergyType sqrtS) const {
    return isValid(projectileID, getMaterialIndex(targetID), sqrtS);
  }

  inline int InteractionModel::getMaterialIndex(Code targetID) const {
    auto compare = [=](std::pair<Code, int> const& p) { return p.first == targetID; };
    if (auto it = std::find_if(materials_.cbegin(), materials_.cend(), compare);
        it == materials_.cend()) {
      return -1;
    } else {
      return it->second;
    }
  }

  inline CrossSectionType InteractionModel::getCrossSection(
      Code const projectileId, Code const targetId, FourMomentum const& projectileP4,
      FourMomentum const& targetP4) const {
    auto const flukaMaterial = getMaterialIndex(targetId);

    HEPEnergyType const sqrtS = (projectileP4 + targetP4).getNorm();
    if (!isValid(projectileId, flukaMaterial, sqrtS)) { return CrossSectionType::zero(); }

    COMBoost const targetRestBoost{targetP4.getSpaceLikeComponents(), get_mass(targetId)};
    FourMomentum const projectileLab4mom = targetRestBoost.toCoM(projectileP4);
    HEPEnergyType const Elab = projectileLab4mom.getTimeLikeComponent();
    auto constexpr invGeV = 1 / 1_GeV;
    double const labMomentum =
        projectileLab4mom.getSpaceLikeComponents().getNorm() * invGeV;

    auto const plab = projectileLab4mom.getSpaceLikeComponents();

    CORSIKA_LOGGER_DEBUG(logger_, fmt::format("Elab = {} GeV", Elab * invGeV));
    auto const flukaCodeProj =
        static_cast<FLUKACodeIntType>(convertToFluka(projectileId));

    double const dummyEkin = 0;
    CrossSectionType const xs = ::fluka::sgmxyz_(&flukaCodeProj, &flukaMaterial,
                                                 &dummyEkin, &labMomentum, &iflxyz_) *
                                1_mb;
    return xs;
  }

  template <typename TSecondaryView>
  inline void InteractionModel::doInteraction(TSecondaryView& view,
                                              Code const projectileId,
                                              Code const targetId,
                                              FourMomentum const& projectileP4,
                                              FourMomentum const& targetP4) {
    auto finalState =
        generateFinalState(projectileId, targetId, projectileP4, targetP4);
    for (auto& secondary : finalState) {
      view.addSecondary(std::move(secondary));
    }
  }

  inline InteractionModel::FinalState InteractionModel::generateFinalState(
      Code const projectileId, Code const targetId,
      FourMomentum const& projectileP4,
      FourMomentum const& targetP4) {
    auto const flukaCodeProj =
        static_cast<FLUKACodeIntType>(convertToFluka(projectileId));
    auto const flukaMaterial = getMaterialIndex(targetId);

    HEPEnergyType const sqrtS = (projectileP4 + targetP4).getNorm();
    if (!isValid(projectileId, flukaMaterial, sqrtS)) {
      std::string const errmsg = fmt::format(
          "Event generation with invalid configuration requested: proj: {}, target: {}",
          get_name(projectileId, full_name{}), get_name(targetId, full_name{}));
      CORSIKA_LOGGER_CRITICAL(logger_, errmsg);
      throw std::runtime_error{errmsg.c_str()};
    }

    COMBoost const targetRestBoost{targetP4.getSpaceLikeComponents(), get_mass(targetId)};
    FourMomentum const projectileLab4mom = targetRestBoost.toCoM(projectileP4);
    auto constexpr invGeV = 1 / 1_GeV;

    auto const plab = projectileLab4mom.getSpaceLikeComponents();
    auto const& cs = plab.getCoordinateSystem();
    auto const labMomentum = plab.getNorm();
    double const labMomentumGeV = labMomentum * invGeV;

    auto const direction = (plab / labMomentum).getComponents().getEigenVector();

    double const dummyEkin = 0;
    ::fluka::evtxyz_(&flukaCodeProj, &flukaMaterial, &dummyEkin, &labMomentumGeV,
                     &direction[0], &direction[1], &direction[2], &iflxyz_, cumsgx_.get(),
                     cumsgx_.get() + materials_.size(),
                     cumsgx_.get() + materials_.size() * 2);

    FinalState finalState;
    finalState.reserve(static_cast<std::size_t>(::fluka::hepevt_.nhep));
    for (int i = 0; i < ::fluka::hepevt_.nhep; ++i) {
      int const status = ::fluka::hepevt_.isthep[i];
      if (status != 1) // skip non-final-state particles
        continue;

      auto const pdg = static_cast<corsika::PDGCode>(::fluka::hepevt_.idhep[i]);
      auto const c8code = corsika::convert_from_PDG(pdg);
      auto const mom = QuantityVector<hepenergy_d>{
          Eigen::Map<Eigen::Vector3d>(&::fluka::hepevt_.phep[i][0]) *
          (1_GeV).magnitude()};
      auto const pPrime = mom.getNorm();
      auto const c8mass = corsika::get_mass(c8code);

      auto const fourMomCollisionFrame =
          FourVector{calculate_total_energy(pPrime, c8mass), MomentumVector{cs, mom}};
      auto const fourMomOrigFrame = targetRestBoost.fromCoM(fourMomCollisionFrame);
      auto const momOrigFrame = fourMomOrigFrame.getSpaceLikeComponents();
      auto const p = momOrigFrame.getNorm();

      finalState.emplace_back(
          c8code, corsika::calculate_kinetic_energy(p, c8mass),
          momOrigFrame / p);
    }
    return finalState;
  }

  inline std::vector<std::pair<Code, int>> InteractionModel::genFlukaMaterials(
      std::set<Code> const& allElementsInUniverse) {

    /*
     * We define one material per element/isotope we have in C8. Cross-section averaging
     * and target isotope selection happen in C8.
     */

    int const nElements = allElementsInUniverse.size();
    auto nelmfl = std::make_unique<int[]>(nElements);
    std::vector<int> izelfl;
    izelfl.reserve(nElements);
    auto wfelml = std::make_unique<double[]>(nElements);
    auto const mxelfl = nElements;
    double const pptmax = 1e11; // GeV
    double const ef2dp3 = 0;    // GeV, 0 means default is used
    double const df2dp3 = -1;   // default
    bool const lprint = true;
    auto mtflka = std::make_unique<int[]>(mxelfl);
    // magic number that FLUKA uses to see if it's the right version
    char const crvrck[] = "76466879";
    int const size = 8;

    std::fill(&nelmfl[0], &nelmfl[nElements], 1);
    std::fill(&wfelml[0], &wfelml[nElements], 1.);
    std::transform(allElementsInUniverse.cbegin(), allElementsInUniverse.cend(),
                   std::back_inserter(izelfl), get_nucleus_Z);

    // check if FLUPRO is set
    if (std::getenv("FLUPRO") == nullptr) {
      throw std::runtime_error{"FLUPRO not set; required to initialize FLUKA"};
    }

    // call FLUKA
    ::fluka::stpxyz_(&nElements, nelmfl.get(), izelfl.data(), wfelml.get(), &mxelfl,
                     &pptmax, &ef2dp3, &df2dp3, &iflxyz_, &lprint, mtflka.get(), crvrck,
                     &size);

    // now create & fill vector of (C8 Code, FLUKA mat. no.) pairs
    std::vector<std::pair<Code, int>> mapping;
    mapping.reserve(nElements);

    auto it = boost::make_zip_iterator(
        boost::make_tuple(allElementsInUniverse.begin(), &mtflka[0]));
    auto const end = boost::make_zip_iterator(
        boost::make_tuple(allElementsInUniverse.end(), &mtflka[nElements]));
    for (; it != end; ++it) {
      boost::tuple<Code const&, int&> tup = *it;
      mapping.emplace_back(tup.get<0>(), tup.get<1>());
    }

    return mapping;
  }
} // namespace corsika::fluka
