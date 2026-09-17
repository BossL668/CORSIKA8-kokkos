#pragma once
#include <map>
#include <corsika/modules/FLUKA.hpp>
// Mountain-only FLUKA target setup, adapted from the CORSIKA BSD-3-Clause wrapper.
// Original air FLUKA headers and the external FLUKA library are unchanged.
/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */


#include <vector>
#include <utility>
#include <memory>
#include <tuple>

#include <corsika/media/Environment.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/core/HadronicBatchProtocol.hpp>
#include <corsika/framework/utility/COMBoost.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/random/RNGManager.hpp>

namespace corsika::applications::terrain::fluka_detail {
  using corsika::fluka::convertToFluka;
  using corsika::fluka::FLUKACodeIntType;
  extern "C" void c8_material_fluka_setup(int const*,int const*,int*,int*);
  /**
   * This class exposes the (hadronic) interactions of FLUKA. FLUKA needs to be
   * initialized with a predefined set of target materials and a flag describing the type
   * of interactions (elastic, inelastic, electromagnetic dissociation). Currently, only
   * inelastic events are supported.
   */
  class ExpandedInteractionModel {
  public:
    static constexpr HadronicWorkerModel
        hadronic_worker_model =
            HadronicWorkerModel::Fluka;

    /**
     * One FLUKA final-state particle in the original CORSIKA frame.
     *
     * Keeping this value independent of SecondaryView is the serialization boundary
     * needed by a process-isolated FLUKA worker.  The tuple deliberately matches the
     * existing addSecondary() input, so the legacy scalar path does not need a second
     * kinematic conversion.
     */
    using FinalStateParticle =
        std::tuple<Code, HEPEnergyType, DirectionVector>;
    using FinalState = std::vector<FinalStateParticle>;

    /**
     * Create a new ExpandedInteractionModel. The FLUKA materials are collected from the elements
     * present in the environment. Each element is its own FLUKA material, no FLUKA
     * compounds are used.
     */
    ExpandedInteractionModel(std::set<Code> const&);

    //! Return the cross-section of a given combination of projectile/target.
    CrossSectionType getCrossSection(Code projectileId, Code targetId,
                                     FourMomentum const& projectileP4,
                                     FourMomentum const& targetP4) const;

    bool isValid(Code projectileID, Code targetID, HEPEnergyType sqrtS) const;
    bool isValid(Code projectileID, int material, HEPEnergyType sqrtS) const;

    //! convert target Code to FLUKA material number
    int getMaterialIndex(Code targetID) const;

    /**
     * Generate a FLUKA final state without modifying a CORSIKA stack.
     *
     * FLUKA itself remains process-global and must not be called concurrently inside
     * one process.  This method separates that non-thread-safe call from the subsequent
     * deterministic SecondaryView commit, allowing future workers to return a complete
     * final state to the owner of the main stack.
     */
    FinalState generateFinalState(
        Code projectileId, Code targetId,
        FourMomentum const& projectileP4,
        FourMomentum const& targetP4);

    /**
     * Perform an interaction. Since FLUKA expects a fixed-target configuration, we
     * perform a Lorentz transform into the rest frame of the target.
     */
    template <typename TSecondaryView>
    void doInteraction(TSecondaryView& view, Code const projectileId, Code const targetId,
                       FourMomentum const& projectileP4, FourMomentum const& targetP4);

  private:
    default_prng_type& RNG_ = RNGManager<>::getInstance().getRandomStream("fluka");
    std::vector<std::pair<Code, int>> const
        materials_; //!< map target Code to FLUKA material no.
    std::shared_ptr<spdlog::logger> logger_ = get_logger("corsika_FLUKA_Interaction");
    std::unique_ptr<double[]> cumsgx_; //!< dump for evtxyz cumsg*, never read again

    static std::vector<std::pair<Code, int>> genFlukaMaterials(std::set<Code> const&);
  };

  inline static int const iflxyz_ = 1;
} // namespace corsika::fluka


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

namespace corsika::applications::terrain::fluka_detail {
  inline ExpandedInteractionModel::ExpandedInteractionModel(std::set<Code> const& nuccomp)
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

  inline bool ExpandedInteractionModel::isValid(Code projectileID, int material,
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

  inline bool ExpandedInteractionModel::isValid(Code projectileID, Code targetID,
                                        HEPEnergyType sqrtS) const {
    return isValid(projectileID, getMaterialIndex(targetID), sqrtS);
  }

  inline int ExpandedInteractionModel::getMaterialIndex(Code targetID) const {
    auto compare = [=](std::pair<Code, int> const& p) { return p.first == targetID; };
    if (auto it = std::find_if(materials_.cbegin(), materials_.cend(), compare);
        it == materials_.cend()) {
      return -1;
    } else {
      return it->second;
    }
  }

  inline CrossSectionType ExpandedInteractionModel::getCrossSection(
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
  inline void ExpandedInteractionModel::doInteraction(TSecondaryView& view,
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

  inline ExpandedInteractionModel::FinalState ExpandedInteractionModel::generateFinalState(
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

  inline std::vector<std::pair<Code,int>> ExpandedInteractionModel::genFlukaMaterials(std::set<Code> const& targets) {
    if(!std::getenv("FLUPRO")) throw std::runtime_error("FLUPRO is required for mountain FLUKA materials");
    std::map<int,int> elemental;
    for(auto code:targets) elemental.emplace(get_nucleus_Z(code),0);
    int count=elemental.size(),status=0;
    std::vector<int> Z,indices(count); for(auto x:elemental) Z.push_back(x.first);
    c8_material_fluka_setup(&count,Z.data(),indices.data(),&status);
    if(status) throw std::runtime_error("FLUKA material bootstrap/ABI verification failed: "+std::to_string(status));
    std::size_t i=0; for(auto& x:elemental) x.second=indices[i++];
    std::vector<std::pair<Code,int>> mapping;
    for(auto code:targets) mapping.emplace_back(code,elemental.at(get_nucleus_Z(code)));
    return mapping;
  }
} // namespace corsika::applications::terrain::fluka_detail


namespace corsika::applications::terrain {
// Keep the stock implementation for <=10 targets. The expanded entry changes
// only setup; its collision/kinematic implementation is copied from the same
// CORSIKA FLUKA wrapper above. FLUKA targets remain elemental (natural isotopes).
class MaterialFlukaInteraction : public InteractionProcess<MaterialFlukaInteraction> {
  std::unique_ptr<fluka::InteractionModel> stock_;
  std::unique_ptr<fluka_detail::ExpandedInteractionModel> expanded_;
public:
  explicit MaterialFlukaInteraction(std::set<Code> const& targets) {
    if(targets.size()<=10) stock_=std::make_unique<fluka::InteractionModel>(targets);
    else expanded_=std::make_unique<fluka_detail::ExpandedInteractionModel>(targets);
  }
  bool expandedSetup() const { return bool(expanded_); }
  bool isValid(Code p,Code t,HEPEnergyType s) const {
    return stock_ ? stock_->isValid(p,t,s) : expanded_->isValid(p,t,s);
  }
  CrossSectionType getCrossSection(Code p,Code t,FourMomentum const& a,FourMomentum const& b) const {
    return stock_ ? stock_->getCrossSection(p,t,a,b) : expanded_->getCrossSection(p,t,a,b);
  }
  template<class View> void doInteraction(View& v,Code p,Code t,FourMomentum const& a,FourMomentum const& b) {
    if(stock_) stock_->doInteraction(v,p,t,a,b); else expanded_->doInteraction(v,p,t,a,b);
  }
};
} // namespace corsika::applications::terrain
