/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

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

namespace corsika::fluka {
  /**
   * This class exposes the (hadronic) interactions of FLUKA. FLUKA needs to be
   * initialized with a predefined set of target materials and a flag describing the type
   * of interactions (elastic, inelastic, electromagnetic dissociation). Currently, only
   * inelastic events are supported.
   */
  class InteractionModel {
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
     * Create a new InteractionModel. The FLUKA materials are collected from the elements
     * present in the environment. Each element is its own FLUKA material, no FLUKA
     * compounds are used.
     */
    InteractionModel(std::set<Code> const&);

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

#include <corsika/detail/modules/fluka/InteractionModel.inl>
