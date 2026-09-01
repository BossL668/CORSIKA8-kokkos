/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/media/IMediumModel.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/framework/utility/COMBoost.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/validation/CudaReplayTrace.hpp>

#include <limits>
#include <memory>
#include <random>
#include <tuple>
#include <PROPOSAL/particle/Particle.h>
#include <corsika/modules/proposal/InteractionModel.hpp>

namespace corsika::proposal {

  template <typename THadronicLEModel, typename THadronicHEModel>
  inline std::vector<NativeInteractionCalculatorView>
  InteractionModel<THadronicLEModel, THadronicHEModel>::
      nativeCalculatorViews() const {
    std::vector<NativeInteractionCalculatorView> result;
    result.reserve(calc_.size());
    for (auto const& entry : calc_) {
      auto const medium_hash = entry.first.first;
      auto const projectile = entry.first.second;
      auto const& interaction = std::get<eINTERACTION>(entry.second);
      auto const& lpm = std::get<eLPM_SUPPRESSION>(entry.second);
      if (!interaction) {
        throw std::logic_error(
            "PROPOSAL interaction calculator view contains a null calculator");
      }
      result.push_back(
          {projectile,
           medium_hash,
           &media.at(medium_hash),
           interaction.get(),
           lpm ? lpm->photo_pair_lpm_.get() : nullptr,
           lpm ? lpm->brems_lpm_.get() : nullptr,
           proposal_energycutsettings.at(projectile),
           particle.at(projectile).mass});
    }
    return result;
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  template <typename TEnvironment>
  inline InteractionModel<THadronicLEModel, THadronicHEModel>::InteractionModel(
      TEnvironment const& _env, THadronicLEModel& _hadintLE, THadronicHEModel& _hadintHE,
      HEPEnergyType const& _enthreshold)
      : ProposalProcessBase(_env)
      , HadronicPhotonModel<THadronicLEModel, THadronicHEModel>(_hadintLE, _hadintHE,
                                                                _enthreshold) {
    //! Initialize PROPOSAL tables for all media and all particles
    for (auto const& medium : media) {
      for (auto const particle_code : tracked) {
        buildCalculator(particle_code, medium.first);
      }
    }
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  inline auto InteractionModel<THadronicLEModel, THadronicHEModel>::makeCalculator(
      Code code, size_t const& component_hash,
      HEPEnergyType energy_cut) -> calculator_t {
    // search crosssection builder for given particle
    auto p_cross = cross.find(code);
    if (p_cross == cross.end())
      throw std::runtime_error("PROPOSAL could not find corresponding builder");

    // interpolate the crosssection for given media and energy cut. These may
    // take some minutes if you have to build the tables and cannot read the tables
    // from disk
    auto c = p_cross->second(
        media.at(component_hash), energy_cut);

    // Look which interactions take place and build the corresponding
    // interaction and secondary builder. The interaction integral will
    // interpolated too and saved in the calc map by a key build out of a hash
    // of composed of the component and particle code.
    auto inter_types = PROPOSAL::CrossSectionVector::GetInteractionTypes(c);
    return std::make_tuple(
        PROPOSAL::make_secondaries(inter_types, particle[code], media.at(component_hash)),
        PROPOSAL::make_interaction(c, true, true),
        std::make_unique<LPM_calculator>(media.at(component_hash), code, inter_types),
        std::move(inter_types));
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  inline void InteractionModel<THadronicLEModel, THadronicHEModel>::buildCalculator(
      Code code, size_t const& component_hash) {
    calc_[std::make_pair(component_hash, code)] =
        makeCalculator(
            code, component_hash,
            proposal_energycutsettings[code]);
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  inline void InteractionModel<THadronicLEModel, THadronicHEModel>::
      prepareSpecifiedInteractionCalculator(
          ProposalInteractionRecord const& record,
          HEPEnergyType energy_cut) {
    if (!(energy_cut > HEPEnergyType::zero())) {
      throw std::invalid_argument(
          "specified PROPOSAL calculator requires a positive energy cut");
    }
    auto const key = specified_calc_key_t{
        record.context.medium_hash,
        record.context.projectile_id,
        record.interaction_hash};
    if (specified_calc_.find(key) != specified_calc_.end()) {
      return;
    }
    auto calculator = makeCalculator(
        record.context.projectile_id,
        record.context.medium_hash, energy_cut);
    if (std::get<eINTERACTION>(calculator)->GetHash() !=
        record.interaction_hash) {
      throw std::invalid_argument(
          "specified PROPOSAL calculator cut does not reproduce the GPU "
          "interaction hash");
    }
    specified_calc_.emplace(key, std::move(calculator));
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  inline auto
  InteractionModel<THadronicLEModel, THadronicHEModel>::getCalculatorForRecord(
      ProposalInteractionRecord const& record) -> calculator_t& {
    auto const key =
        std::make_pair(record.context.medium_hash, record.context.projectile_id);
    auto calculator = calc_.find(key);
    if (calculator == calc_.end()) {
      buildCalculator(record.context.projectile_id, record.context.medium_hash);
      calculator = calc_.find(key);
    }
    if (calculator == calc_.end()) {
      throw std::runtime_error(
          "PROPOSAL could not resolve the calculator for an interaction record");
    }
    calculator_t* resolved = &calculator->second;
    if (std::get<eINTERACTION>(*resolved)->GetHash() !=
        record.interaction_hash) {
      auto specified = specified_calc_.find(
          specified_calc_key_t{
              record.context.medium_hash,
              record.context.projectile_id,
              record.interaction_hash});
      if (specified == specified_calc_.end()) {
        throw std::invalid_argument(
            "PROPOSAL interaction record uses a different calculator hash");
      }
      resolved = &specified->second;
    }
    auto const& interaction_types =
        std::get<eINTERACTION_TYPES>(*resolved);
    if (std::find(interaction_types.begin(), interaction_types.end(),
                  record.type) == interaction_types.end()) {
      throw std::invalid_argument(
          "PROPOSAL interaction record contains an unavailable process type");
    }
    return *resolved;
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  template <typename TParticle>
  inline ProposalInteractionContext
  InteractionModel<THadronicLEModel, THadronicHEModel>::makeInteractionContext(
      TParticle const& projectile, Code const projectileId) const {
    Point const& place = projectile.getPosition();
    CoordinateSystemPtr const& labCS = place.getCoordinateSystem();
    auto const direction = projectile.getDirection().getComponents(labCS);
    auto const& composition =
        projectile.getNode()->getModelProperties().getNuclearComposition();

    return ProposalInteractionContext{
        projectileId,
        composition.getHash(),
        projectile.getEnergy() / 1_MeV,
        {place.getX(labCS) / 1_cm, place.getY(labCS) / 1_cm,
         place.getZ(labCS) / 1_cm},
        {direction.getX().magnitude(), direction.getY().magnitude(),
         direction.getZ().magnitude()},
        projectile.getTime() / 1_s};
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  template <typename TParticle>
  inline ProposalRateTable
  InteractionModel<THadronicLEModel, THadronicHEModel>::getRateTable(
      TParticle& projectile, Code const projectileId) {
    if (!canInteract(projectileId)) {
      throw std::invalid_argument(
          "Cannot evaluate PROPOSAL rates for an unsupported projectile");
    }
    auto calculator = getCalculator(projectile, calc_);
    return rate_provider_.rates(
        *std::get<eINTERACTION>(calculator->second),
        projectile.getEnergy() / 1_MeV);
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  template <typename TParticle>
  inline ProposalInteractionRecord
  InteractionModel<THadronicLEModel, THadronicHEModel>::sampleInteraction(
      TParticle& projectile, Code const projectileId,
      ProposalRateTable const& rate_table, double const selection_uniform,
      std::optional<ProposalRandomKey> random_key) {
    if (!canInteract(projectileId)) {
      throw std::invalid_argument(
          "Cannot sample a PROPOSAL interaction for an unsupported projectile");
    }
    auto calculator = getCalculator(projectile, calc_);
    return rate_provider_.sample(
        *std::get<eINTERACTION>(calculator->second), rate_table,
        makeInteractionContext(projectile, projectileId), selection_uniform,
        std::move(random_key));
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  inline std::size_t InteractionModel<
      THadronicLEModel,
      THadronicHEModel>::requiredFinalStateRandomNumbers(
      ProposalInteractionRecord const& record) {
    auto& calculator = getCalculatorForRecord(record);
    return final_state_generator_.requiredRandomNumbers(
        *std::get<eSECONDARIES>(calculator), record);
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  inline ProposalFinalState
  InteractionModel<THadronicLEModel, THadronicHEModel>::generateFinalState(
      ProposalInteractionRecord const& record,
      std::vector<double> random_numbers) {
    auto& calculator = getCalculatorForRecord(record);
    return final_state_generator_.generate(
        *std::get<eSECONDARIES>(calculator), record,
        std::move(random_numbers));
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  inline void InteractionModel<
      THadronicLEModel,
      THadronicHEModel>::completeSelectedLoss(
      ProposalInteractionRecord& record,
      double loss_quantile) {
    auto& calculator = getCalculatorForRecord(record);
    auto const rates = rate_provider_.rates(
        *std::get<eINTERACTION>(calculator),
        record.context.projectile_energy_MeV);
    if (rates.interactionHash() !=
        record.interaction_hash) {
      throw std::logic_error(
          "PROPOSAL selected-loss calculator hash changed");
    }
    record.v_loss = rate_provider_.sampleSelectedLoss(
        rates, record.type, record.component_hash,
        loss_quantile);
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  inline void InteractionModel<
      THadronicLEModel,
      THadronicHEModel>::completeNativeSelectionReplay(
      ProposalInteractionRecord& record) {
    auto& calculator = getCalculatorForRecord(record);
    auto& interaction = *std::get<eINTERACTION>(calculator);
    auto const rates = rate_provider_.rates(
        interaction, record.context.projectile_energy_MeV);
    if (rates.interactionHash() != record.interaction_hash) {
      throw std::logic_error(
          "PROPOSAL native selection-replay calculator hash changed");
    }
    auto const replayed = rate_provider_.sample(
        interaction, rates, record.context, record.selection_uniform);
    if (replayed.type != record.type ||
        replayed.component_hash != record.component_hash) {
      throw std::runtime_error(
          "PROPOSAL native selection replay changed process or component");
    }
    record.v_loss = replayed.v_loss;
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  template <typename TSecondaryView>
  inline ProcessReturn
  InteractionModel<THadronicLEModel, THadronicHEModel>::
      doSpecifiedInteraction(
          TSecondaryView& view,
          ProposalInteractionRecord const& record,
          std::vector<double> random_numbers) {
    auto& calculator = getCalculatorForRecord(record);
    auto const projectileId =
        record.context.projectile_id;
    auto projectile = view.getProjectile();
    if (projectile.getPID() != projectileId) {
      throw std::invalid_argument(
          "specified PROPOSAL interaction projectile does not match the stack");
    }

    Point const& place = projectile.getPosition();
    CoordinateSystemPtr const& labCS =
        place.getCoordinateSystem();
    auto final_state = final_state_generator_.generate(
        *std::get<eSECONDARIES>(calculator), record,
        std::move(random_numbers));
    auto& target = final_state.target;
    auto& sec = final_state.secondaries;

    if (std::get<eLPM_SUPPRESSION>(calculator)) {
      auto const lpm_suppression = CheckForLPM(
          *std::get<eLPM_SUPPRESSION>(calculator),
          projectile.getEnergy(), record.type, sec,
          projectile.getNode()
              ->getModelProperties()
              .getMassDensity(place),
          target, record.v_loss);
      if (lpm_suppression) {
        CORSIKA_LOG_DEBUG(
            "LPM suppression detected for specified PROPOSAL interaction");
        view.addSecondary(std::make_tuple(
            projectileId,
            projectile.getEnergy() - get_mass(projectileId),
            projectile.getDirection()));
        return ProcessReturn::Ok;
      }
    }

    // PROPOSAL's final-state energies for interactions on an atomic electron
    // include that initially stationary target electron.  Its rest energy is
    // therefore a genuine medium source term in a whole-shower total-energy
    // ledger.  Count it here so both the ordinary scalar path and explicitly
    // completed CUDA fallbacks follow the same convention.  Do this only
    // after a possible LPM rejection, since a rejected vertex does not consume
    // a target electron.
    if (record.type == PROPOSAL::InteractionType::Ioniz ||
        record.type == PROPOSAL::InteractionType::Compton ||
        record.type == PROPOSAL::InteractionType::Annihilation ||
        record.type == PROPOSAL::InteractionType::Photoeffect) {
      ++atomic_electron_target_interactions_;
      weighted_atomic_electron_rest_mass_input_GeV_ +=
          projectile.getWeight() * PROPOSAL::ME / 1000.;
    }

    for (auto& secondary : sec) {
      auto const energy = secondary.energy * 1_MeV;
      auto const proposal_direction =
          secondary.direction;
      auto const direction = DirectionVector(
          labCS,
          {proposal_direction.GetX(),
           proposal_direction.GetY(),
           proposal_direction.GetZ()});

      if (static_cast<PROPOSAL::ParticleType>(
              secondary.type) ==
          PROPOSAL::ParticleType::Hadron) {
        FourMomentum const photonP4(
            energy, energy * direction);
        auto const A =
            static_cast<int>(target.GetAtomicNum());
        auto const Z =
            static_cast<int>(target.GetNucCharge());
        Code const targetId = get_nucleus_code(A, Z);
        CORSIKA_LOGGER_DEBUG(
            logger_,
            "specified photo-hadronic interaction of projectile={} "
            "with target={}! Energy={} GeV",
            projectileId, targetId, energy / 1_GeV);
        this->doHadronicPhotonInteraction(
            view, labCS, photonP4, targetId);
      } else {
        auto const secondary_code = convert_from_PDG(
            static_cast<PDGCode>(secondary.type));
        auto const proposal_mass =
            PROPOSAL::ParticleDef::GetParticleDefForType(
                secondary.type)
                .mass *
            1_MeV;
        view.addSecondary(std::make_tuple(
            secondary_code, energy - proposal_mass,
            direction));
      }
    }
    return ProcessReturn::Ok;
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  template <typename TStackView>
  inline ProcessReturn
  InteractionModel<THadronicLEModel, THadronicHEModel>::doInteraction(
      TStackView& view, Code const projectileId,
      [[maybe_unused]] FourMomentum const& projectileP4) {

    auto const projectile = view.getProjectile();

    if (canInteract(projectileId)) {

      std::uniform_real_distribution<double> distr(0., 1.);

      // Evaluate rates and select type/component/v exactly once. The legacy CPU path
      // still draws from the same sequential stream and in the same order.
      auto const rates = getRateTable(projectile, projectileId);
      auto const selection_uniform = distr(RNG_);
      auto const record = sampleInteraction(
          projectile, projectileId, rates, selection_uniform);

      if (validation::CudaReplayTrace::instance().enabled()) {
        auto const& place = projectile.getPosition();
        auto const labCS = place.getCoordinateSystem();
        auto const direction =
            projectile.getDirection().getComponents(labCS);
        validation::CudaReplayTrace::instance().recordProposalSelection(
            static_cast<std::int32_t>(get_PDG(projectileId)),
            static_cast<std::int32_t>(record.type),
            static_cast<std::uint64_t>(record.component_hash),
            projectile.getEnergy() / 1_GeV, record.v_loss,
            place.getX(labCS) / 1_m, place.getY(labCS) / 1_m,
            place.getZ(labCS) / 1_m,
            direction.getX().magnitude(), direction.getY().magnitude(),
            direction.getZ().magnitude(), projectile.getTime() / 1_s);
      }

      // TODO: This should become obsolete as soon #482 is fixed
      if (record.type == PROPOSAL::InteractionType::Undefined) {
        CORSIKA_LOG_WARN(
            "PROPOSAL: No particle interaction possible. "
            "Put initial particle back on stack.");
        view.addSecondary(std::make_tuple(projectileId,
                                          projectile.getEnergy() - get_mass(projectileId),
                                          projectile.getDirection()));
        return ProcessReturn::Ok;
      }

      // Read how much random numbers are required to calculate the secondaries.
      // Calculate the secondaries and deploy them on the corsika stack.
      auto rnd =
          std::vector<double>(requiredFinalStateRandomNumbers(record));
      for (auto& it : rnd) it = distr(RNG_);
      return doSpecifiedInteraction(
          view, record, std::move(rnd));
    }
    return ProcessReturn::Ok;
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  template <typename TParticle>
  inline CrossSectionType
  InteractionModel<THadronicLEModel, THadronicHEModel>::getCrossSection(
      TParticle const& projectile, Code const projectileId,
      FourMomentum const& projectileP4) {

    // ==============================================
    // this block better diappears. RU 26.10.2021
    //
    // determine the volume where the particle is (last) known to be
    auto const* currentLogicalNode = projectile.getNode();
    NuclearComposition const& composition =
        currentLogicalNode->getModelProperties().getNuclearComposition();
    auto const meanMass = composition.getAverageMassNumber() * constants::u;
    // ==============================================

    if (canInteract(projectileId)) {
      auto c = getCalculator(projectile, calc_);
      return meanMass / (std::get<eINTERACTION>(c->second)->MeanFreePath(
                             projectileP4.getTimeLikeComponent() / 1_MeV) *
                         1_g / (1_cm * 1_cm));
    }

    return CrossSectionType::zero();
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  bool InteractionModel<THadronicLEModel, THadronicHEModel>::CheckForLPM(
      const LPM_calculator& lpm_calculator, const HEPEnergyType projectile_energy,
      const PROPOSAL::InteractionType type,
      const std::vector<PROPOSAL::ParticleState>& sec, const MassDensityType mass_density,
      const PROPOSAL::Component& target, const double v) {

    double suppression_factor;
    double current_density = mass_density * ((1_cm * 1_cm * 1_cm) / 1_g);

    if (type == PROPOSAL::InteractionType::Photopair && lpm_calculator.photo_pair_lpm_) {
      if (sec.size() != 2)
        throw std::runtime_error(
            "Invalid number of secondaries for Photopair in CheckForLPM");
      auto x =
          sec[0].energy / (sec[0].energy + sec[1].energy); // particle energy asymmetry
      double density_correction = current_density / lpm_calculator.mass_density_baseline_;
      suppression_factor = lpm_calculator.photo_pair_lpm_->suppression_factor(
          projectile_energy / 1_MeV, x, target, density_correction);
    } else if (type == PROPOSAL::InteractionType::Brems && lpm_calculator.brems_lpm_) {
      double density_correction = current_density / lpm_calculator.mass_density_baseline_;
      suppression_factor = lpm_calculator.brems_lpm_->suppression_factor(
          projectile_energy / 1_MeV, v, target, density_correction);
    } else if (type == PROPOSAL::InteractionType::Epair && lpm_calculator.epair_lpm_) {
      if (sec.size() != 3)
        throw std::runtime_error(
            "Invalid number of secondaries for EPair in CheckForLPM");
      double rho = (sec[1].energy - sec[2].energy) /
                   (sec[1].energy + sec[2].energy); // todo: check
      // it would be better if these variables are calculated within PROPOSAL
      double beta = (v * v) / (2 * (1 - v));
      double xi = (lpm_calculator.particle_mass_ * lpm_calculator.particle_mass_) /
                  (PROPOSAL::ME * PROPOSAL::ME) * v * v / 4 * (1 - rho * rho) / (1 - v);
      double density_correction = current_density / lpm_calculator.mass_density_baseline_;
      suppression_factor = lpm_calculator.epair_lpm_->suppression_factor(
          projectile_energy / 1_MeV, v, rho * rho, beta, xi, density_correction);
    } else {
      return false;
    }

    // rejection sampling
    std::uniform_real_distribution<double> distr(0., 1.);
    auto rnd_num = distr(RNG_);

    CORSIKA_LOGGER_TRACE(logger_,
                         "Checking for LPM suppression! energy: {}, v: {}, type: {}, "
                         "target: {}, mass_density: {}, mass_density_baseline: {}, "
                         "suppression_factor: {}, rnd: {}, result: {}, pmass: {} ",
                         projectile_energy, v, type, target.GetName(),
                         mass_density / (1_g / (1_cm * 1_cm * 1_cm)),
                         lpm_calculator.mass_density_baseline_, suppression_factor,
                         rnd_num, (rnd_num > suppression_factor),
                         lpm_calculator.particle_mass_);

    if (rnd_num > suppression_factor)
      return true;
    else
      return false;
  }
} // namespace corsika::proposal
