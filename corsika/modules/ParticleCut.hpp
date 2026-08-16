/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <map>
#include <unordered_map>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/process/SecondariesProcess.hpp>
#include <corsika/framework/process/ContinuousProcess.hpp>
#include <corsika/framework/core/Step.hpp>

#include <corsika/modules/writers/WriterOff.hpp>

namespace corsika {

  struct ParticleCutSpeciesStatistics {
    std::uint64_t particles{};
    double weighted_kinetic_energy_GeV{};
    double weighted_rest_mass_energy_GeV{};
  };

  struct ParticleCutStatistics {
    std::uint64_t particles{};
    std::uint64_t invisible_particles{};
    double weighted_kinetic_energy_GeV{};
    double weighted_rest_mass_energy_GeV{};
    double weighted_invisible_kinetic_energy_GeV{};
    std::map<std::int32_t, ParticleCutSpeciesStatistics> by_pdg{};
  };
  /**
   * ParticleCut process to kill particles.
   *
   * Goes through the secondaries of an interaction and
   * removes particles according to their kinetic energy. Particles with a time delay of
   * more than 10ms are removed as well. Invisible particles (neutrinos) can be removed if
   * selected. The threshold value is set to 0 by default but in principle can be
   * configured for each particle. Special constructors for cuts by the following groups
   * are implemented: (electrons,positrons), photons, hadrons and muons.
   */
  template <typename TOutput = WriterOff>
  class ParticleCut : public SecondariesProcess<ParticleCut<TOutput>>,
                      public ContinuousProcess<ParticleCut<TOutput>>,
                      public TOutput {

  public:
    /**
     * particle cut with kinetic energy thresholds for electrons, photons,
     * hadrons (including nuclei with energy per nucleon) and muons
     * invisible particles (neutrinos) can be cut or not.
     *
     * @param outputArgs - optional arguments of TOutput writer
     */
    template <typename... TArgs>
    ParticleCut(HEPEnergyType const eEleCut, HEPEnergyType const ePhoCut,
                HEPEnergyType const eHadCut, HEPEnergyType const eMuCut,
                HEPEnergyType const eTauCut, bool const inv, TArgs&&... args);

    /**
     * particle cut with kinetic energy thresholds for all particles.
     *
     * @param outputArgs - optional arguments of TOutput writer
     */
    template <typename... TArgs>
    ParticleCut(HEPEnergyType const eCut, bool const inv, TArgs&&... OutputArgs);

    /**
     * Threshold for specific particles redefined. EM and invisible particles can be set
     * to be discarded altogether.
     *
     * @param outputArgs - optional arguments of TOutput writer
     */
    template <typename... TArgs>
    ParticleCut(std::unordered_map<Code const, HEPEnergyType const> const& eCuts,
                bool const inv, TArgs&&... outputArgs);

    /**
     * Cut particles which are secondaries from discrete processes.
     *
     * @tparam TStackView
     */
    template <typename TStackView>
    void doSecondaries(TStackView&);

    /**
     * Cut particles during continuous processes (energy losses etc).
     *
     * @tparam TParticle
     * @param step
     * @param limitFlag
     * @return ProcessReturn
     */
    template <typename TParticle>
    ProcessReturn doContinuous(
        Step<TParticle>&,
        const bool limitFlag = false); // this is not used for ParticleCut

    /**
     * Limit on continuous step length imposed by ParticleCut: none.
     *
     * @tparam TParticle
     * @tparam TTrajectory
     * @return LengthType
     */
    template <typename TParticle, typename TTrajectory>
    LengthType getMaxStepLength(TParticle const&, TTrajectory const&) {
      return meter * std::numeric_limits<double>::infinity();
    }

    void printThresholds() const;

    HEPEnergyType getElectronKineticECut() const { return cut_electrons_; }
    HEPEnergyType getPhotonKineticECut() const { return cut_photons_; }
    HEPEnergyType getMuonKineticECut() const { return cut_muons_; }
    HEPEnergyType getTauKineticECut() const { return cut_tau_; }
    HEPEnergyType getHadronKineticECut() const { return cut_hadrons_; }

    ParticleCutStatistics const& statistics() const noexcept {
      return statistics_;
    }

    //! get configuration of this node, for output
    YAML::Node getConfig() const override;

  private:
    bool checkCutParticle(Code const, HEPEnergyType const, TimeType const) const;

    bool isBelowEnergyCut(Code const, HEPEnergyType const) const;

    void recordCut(Code, HEPEnergyType, double);

  private:
    HEPEnergyType cut_electrons_;
    HEPEnergyType cut_photons_;
    HEPEnergyType cut_hadrons_;
    HEPEnergyType cut_muons_;
    HEPEnergyType cut_tau_;
    bool doCutInv_;
    std::unordered_map<Code const, HEPEnergyType const> cuts_;
    ParticleCutStatistics statistics_{};
  }; // namespace corsika

} // namespace corsika

#include <corsika/detail/modules/ParticleCut.inl>
