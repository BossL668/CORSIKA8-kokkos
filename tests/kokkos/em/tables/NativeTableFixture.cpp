/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "NativeTableFixture.hpp"
#include "ProposalNativeExportReference.hpp"

#include <corsika/kokkos/em/proposal/NativeTableExporter.hpp>
#include <corsika/modules/proposal/NativeCalculatorView.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>

#include <PROPOSAL/Constants.h>
#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/propagation_utility/DisplacementBuilder.h>
#include <PROPOSAL/propagation_utility/InteractionBuilder.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace corsika::kokkos::test {
  namespace {
    using namespace corsika::units::si;

    // Each process owns its PROPOSAL globals; RUN_SERIAL prevents shared-cache writers.
    class SmallTableSettings {
    public:
      explicit SmallTableSettings(std::string const& directory)
          : path_(PROPOSAL::InterpolationSettings::TABLES_PATH)
          , upper_(PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM)
          , dndxEnergy_(PROPOSAL::InterpolationSettings::NODES_DNDX_E)
          , dndxLoss_(PROPOSAL::InterpolationSettings::NODES_DNDX_V)
          , dedx_(PROPOSAL::InterpolationSettings::NODES_DEDX)
          , utility_(PROPOSAL::InterpolationSettings::NODES_UTILITY)
          , rate_(PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT) {
        if (directory.empty())
          throw std::invalid_argument("Native table test cache is empty");
        std::filesystem::create_directories(directory);
        PROPOSAL::InterpolationSettings::TABLES_PATH = directory;
        PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM = 1.e8;
        PROPOSAL::InterpolationSettings::NODES_DNDX_E = 32;
        // This dependency's derivative construction requires a square grid;
        // PROPOSAL's production defaults are likewise 100 x 100.
        PROPOSAL::InterpolationSettings::NODES_DNDX_V = 32;
        PROPOSAL::InterpolationSettings::NODES_DEDX = 64;
        PROPOSAL::InterpolationSettings::NODES_UTILITY = 64;
        PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT = 64;
      }

      ~SmallTableSettings() {
        PROPOSAL::InterpolationSettings::TABLES_PATH = path_;
        PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM = upper_;
        PROPOSAL::InterpolationSettings::NODES_DNDX_E = dndxEnergy_;
        PROPOSAL::InterpolationSettings::NODES_DNDX_V = dndxLoss_;
        PROPOSAL::InterpolationSettings::NODES_DEDX = dedx_;
        PROPOSAL::InterpolationSettings::NODES_UTILITY = utility_;
        PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT = rate_;
      }

    private:
      std::string path_;
      double upper_;
      unsigned dndxEnergy_, dndxLoss_, dedx_, utility_, rate_;
    };

    class Calculators {
    public:
      Calculators() {
        for (Code const code : {Code::Photon, Code::Electron, Code::Positron,
                                Code::MuMinus, Code::MuPlus}) {
          // Use the official factories unchanged, including their process order,
          // parametrizations, v-cut exceptions and interpolation settings.
          auto crosses = corsika::proposal::cross.at(code)(air_, .4_MeV);
          processCount_ += crosses.size();
          interactions_.push_back(PROPOSAL::make_interaction(crosses, true, true));
          interactionViews_.emplace_back(
              code, air_.GetHash(), std::cref(air_), std::cref(*interactions_.back()),
              std::nullopt, std::nullopt, code == Code::Photon ? 0_MeV : .4_MeV,
              corsika::proposal::particle.at(code).mass);
          if (code != Code::Photon) {
            displacements_.push_back(PROPOSAL::make_displacement(crosses, true));
            continuousViews_.emplace_back(code, air_.GetHash(), std::cref(air_),
                                          std::cref(*displacements_.back()), .4_MeV,
                                          corsika::proposal::particle.at(code).mass);
          }
        }
      }

      std::vector<double> ordinary() {
        std::vector<double> result;
        for (auto const& interaction : interactions_) {
          for (double energy : {1.e3, 1.e5, 1.e7}) {
            for (auto const& cross : interaction->GetCrossSections()) {
              result.push_back(cross->CalculatedNdx(energy));
              result.push_back(cross->CalculatedEdx(energy));
            }
            auto const rates = interaction->Rates(energy);
            for (auto const& rate : rates) result.push_back(rate.rate);
          }
        }
        for (auto const& displacement : displacements_) {
          for (double energy : {1.e3, 1.e5, 1.e7}) {
            result.push_back(displacement->FunctionToIntegral(energy));
            result.push_back(displacement->SolveTrackIntegral(energy, .9 * energy));
          }
        }
        return result;
      }

      std::vector<corsika::proposal::NativeInteractionCalculatorView> const&
      getInteractions() const {
        return interactionViews_;
      }
      std::vector<corsika::proposal::NativeContinuousCalculatorView> const&
      getContinuous() const {
        return continuousViews_;
      }
      std::size_t getProcessCount() const { return processCount_; }

    private:
      PROPOSAL::Air air_;
      std::vector<std::unique_ptr<PROPOSAL::Interaction>> interactions_;
      std::vector<std::unique_ptr<PROPOSAL::Displacement>> displacements_;
      std::vector<corsika::proposal::NativeInteractionCalculatorView> interactionViews_;
      std::vector<corsika::proposal::NativeContinuousCalculatorView> continuousViews_;
      std::size_t processCount_{};
    };

    bool same(NativeAxisDescriptor const&,
              native_table_reference::NativeAxisDescriptor const&);
    bool same(NativeBicubicDescriptor const&,
              native_table_reference::NativeBicubicDescriptor const&);
    bool same(NativeCubicDescriptor const&,
              native_table_reference::NativeCubicDescriptor const&);
    bool same(NativeDndxColumn const&, native_table_reference::NativeDndxColumn const&);
    bool same(NativeTotalRateColumn const&,
              native_table_reference::NativeTotalRateColumn const&);
    bool same(NativeDedxColumn const&, native_table_reference::NativeDedxColumn const&);
    bool same(NativeUtilityColumn const&,
              native_table_reference::NativeUtilityColumn const&);
    bool same(NativePhysicalConstants const&,
              native_table_reference::NativePhysicalConstants const&);
    bool same(NativeInteractionIdentity const&,
              native_table_reference::NativeInteractionIdentity const&);
    bool same(ProposalNativeTableSet const&,
              native_table_reference::ProposalNativeTableSet const&);

    template <typename TFirst, typename TSecond>
    bool same(TFirst first, TSecond second) {
      if constexpr (std::is_enum_v<TFirst>) {
        return static_cast<std::uint32_t>(first) == static_cast<std::uint32_t>(second);
      } else {
        return first == second;
      }
    }

    bool same(double first, double second) {
      std::uint64_t firstBits{}, secondBits{};
      std::memcpy(&firstBits, &first, sizeof(first));
      std::memcpy(&secondBits, &second, sizeof(second));
      return firstBits == secondBits;
    }

    template <typename TFirst, typename TSecond>
    bool same(std::vector<TFirst> const& first, std::vector<TSecond> const& second) {
      if (first.size() != second.size()) return false;
      for (std::size_t index = 0; index < first.size(); ++index) {
        if (!same(first[index], second[index])) return false;
      }
      return true;
    }
    bool same(NativeAxisDescriptor const& current,
              native_table_reference::NativeAxisDescriptor const& original) {
      return same(current.type_, original.type) && same(current.nodes_, original.nodes) &&
             same(current.low_, original.low) && same(current.high_, original.high) &&
             same(current.step_, original.step);
    }
    bool same(NativeBicubicDescriptor const& current,
              native_table_reference::NativeBicubicDescriptor const& original) {
      return same(current.energyAxis_, original.energy_axis) &&
             same(current.lossAxis_, original.loss_axis) &&
             same(current.rows_, original.rows) &&
             same(current.columns_, original.columns) &&
             same(current.coefficientOffset_, original.coefficient_offset) &&
             same(current.polynomialCoefficientOffset_,
                  original.polynomial_coefficient_offset);
    }
    bool same(NativeCubicDescriptor const& current,
              native_table_reference::NativeCubicDescriptor const& original) {
      return same(current.axis_, original.axis) &&
             same(current.valueTransform_, original.value_transform) &&
             same(current.coefficientOffset_, original.coefficient_offset) &&
             same(current.coefficientCount_, original.coefficient_count) &&
             same(current.hasValueTransform_, original.has_value_transform);
    }
    bool same(NativeDndxColumn const& current,
              native_table_reference::NativeDndxColumn const& original) {
      return same(current.pdgId_, original.pdg_id) &&
             same(current.processId_, original.process_id) &&
             same(current.mediumHash_, original.medium_hash) &&
             same(current.componentHash_, original.component_hash) &&
             same(current.crossSectionHash_, original.cross_section_hash) &&
             same(current.tableHash_, original.table_hash) &&
             same(current.lossTransform_, original.loss_transform) &&
             same(current.kinematicModel_, original.kinematic_model) &&
             same(current.rateModel_, original.rate_model) &&
             same(current.spline_, original.spline) &&
             same(current.componentWeight_, original.component_weight) &&
             same(current.particleMassMeV_, original.particle_mass_MeV) &&
             same(current.componentNuclearCharge_, original.component_nuclear_charge) &&
             same(current.componentAtomicMass_, original.component_atomic_mass) &&
             same(current.componentAverageNucleonMassMeV_,
                  original.component_average_nucleon_mass_MeV) &&
             same(current.directRateFactor_, original.direct_rate_factor) &&
             same(current.lowerEnergyLimitMeV_, original.lower_energy_limit_MeV) &&
             same(current.absoluteCutMeV_, original.absolute_cut_MeV) &&
             same(current.relativeCut_, original.relative_cut) &&
             same(current.mediumMeanExcitationEnergyMeV_,
                  original.medium_mean_excitation_energy_MeV) &&
             same(current.selectionOrdinal_, original.selection_ordinal) &&
             same(current.reserved_, original.reserved);
    }
    bool same(NativeTotalRateColumn const& current,
              native_table_reference::NativeTotalRateColumn const& original) {
      return same(current.pdgId_, original.pdg_id) &&
             same(current.reserved_, original.reserved) &&
             same(current.mediumHash_, original.medium_hash) &&
             same(current.tableHash_, original.table_hash) &&
             same(current.spline_, original.spline) &&
             same(current.lowerEnergyLimitMeV_, original.lower_energy_limit_MeV);
    }
    bool same(NativeDedxColumn const& current,
              native_table_reference::NativeDedxColumn const& original) {
      return same(current.pdgId_, original.pdg_id) &&
             same(current.processId_, original.process_id) &&
             same(current.mediumHash_, original.medium_hash) &&
             same(current.crossSectionHash_, original.cross_section_hash) &&
             same(current.componentIndex_, original.component_index) &&
             same(current.tableHash_, original.table_hash) &&
             same(current.spline_, original.spline) &&
             same(current.componentWeight_, original.component_weight) &&
             same(current.lowerEnergyLimitMeV_, original.lower_energy_limit_MeV) &&
             same(current.accumulationOrdinal_, original.accumulation_ordinal) &&
             same(current.reserved_, original.reserved);
    }
    bool same(NativeUtilityColumn const& current,
              native_table_reference::NativeUtilityColumn const& original) {
      return same(current.pdgId_, original.pdg_id) &&
             same(current.mediumHash_, original.medium_hash) &&
             same(current.tableHash_, original.table_hash) &&
             same(current.spline_, original.spline) &&
             same(current.particleMassMeV_, original.particle_mass_MeV) &&
             same(current.lowerEnergyLimitMeV_, original.lower_energy_limit_MeV) &&
             same(current.reverse_, original.reverse);
    }
    bool same(NativePhysicalConstants const& current,
              native_table_reference::NativePhysicalConstants const& original) {
      return same(current.electronMassMeV_, original.electron_mass_MeV) &&
             same(current.chargedPionMassMeV_, original.charged_pion_mass_MeV) &&
             same(current.fineStructureConstant_, original.fine_structure_constant) &&
             same(current.classicalElectronRadiusCm_,
                  original.classical_electron_radius_cm) &&
             same(current.pi_, original.pi) && same(current.sqrtE_, original.sqrt_e) &&
             same(current.muonMassMeV_, original.muon_mass_MeV);
    }
    bool same(NativeInteractionIdentity const& current,
              native_table_reference::NativeInteractionIdentity const& original) {
      return same(current.pdgId_, original.pdg_id) &&
             same(current.reserved_, original.reserved) &&
             same(current.mediumHash_, original.medium_hash) &&
             same(current.interactionHash_, original.interaction_hash);
    }
    bool same(ProposalNativeTableSet const& current,
              native_table_reference::ProposalNativeTableSet const& original) {
      return same(current.formatVersion_, original.format_version) &&
             same(current.proposalVersion_, original.proposal_version) &&
             same(current.cubicInterpolationVersion_,
                  original.cubic_interpolation_version) &&
             same(current.constants_, original.constants) &&
             same(current.stochasticCutMeV_, original.stochastic_cut_MeV) &&
             same(current.contentHash_, original.content_hash) &&
             same(current.proposalCacheTableCount_,
                  original.proposal_cache_table_count) &&
             same(current.proposalCacheHitCount_, original.proposal_cache_hit_count) &&
             same(current.proposalCacheAllHit_, original.proposal_cache_all_hit) &&
             same(current.interactionIdentities_, original.interaction_identities) &&
             same(current.dndxColumns_, original.dndx_columns) &&
             same(current.totalRateColumns_, original.total_rate_columns) &&
             same(current.selectionColumnIndices_, original.selection_column_indices) &&
             same(current.dedxColumns_, original.dedx_columns) &&
             same(current.dedxAccumulationIndices_, original.dedx_accumulation_indices) &&
             same(current.utilityColumns_, original.utility_columns) &&
             same(current.bicubicValues_, original.bicubic_values) &&
             same(current.bicubicDerivativeEnergy_, original.bicubic_derivative_energy) &&
             same(current.bicubicDerivativeLoss_, original.bicubic_derivative_loss) &&
             same(current.bicubicMixedDerivative_, original.bicubic_mixed_derivative) &&
             same(current.bicubicPolynomialCoefficients_,
                  original.bicubic_polynomial_coefficients) &&
             same(current.cubicValues_, original.cubic_values) &&
             same(current.cubicNodeDerivatives_, original.cubic_node_derivatives);
    }

  } // namespace

  std::string native_table_cache_directory() {
    auto const* directory = std::getenv("CORSIKA_NATIVE_TABLE_CACHE");
    if (directory == nullptr || *directory == '\0') {
      throw std::runtime_error(
          "Set CORSIKA_NATIVE_TABLE_CACHE to a dedicated test cache");
    }
    return directory;
  }

  ProposalNativeTableSet make_native_tables(std::string const& cacheDirectory) {
    SmallTableSettings const settings(cacheDirectory);
    Calculators const calculators;
    return export_proposal_native_tables(calculators.getInteractions(),
                                         calculators.getContinuous());
  }

  ProposalNativeTableSet make_native_tables() {
    return make_native_tables(native_table_cache_directory());
  }

  NativeTableExportAudit audit_native_tables(std::string const& cacheDirectory) {
    SmallTableSettings const settings(cacheDirectory);
    Calculators calculators;
    auto const before = calculators.ordinary();
    NativeTableExportAudit result;
    result.table_ = export_proposal_native_tables(calculators.getInteractions(),
                                                  calculators.getContinuous());
    auto const middle = calculators.ordinary();
    std::vector<native_table_reference::proposal::NativeInteractionCalculatorView>
        interactions;
    for (auto const& view : calculators.getInteractions()) {
      interactions.push_back({view.getProjectile(), view.getMediumHash(),
                              &view.getMedium()->get(), &view.getInteraction()->get(),
                              nullptr, nullptr, view.getStochasticEnergyCut(),
                              view.getParticleMassMeV()});
      result.projectileIds_.push_back(
          static_cast<std::int32_t>(get_PDG(view.getProjectile())));
    }
    std::vector<native_table_reference::proposal::NativeContinuousCalculatorView>
        continuous;
    for (auto const& view : calculators.getContinuous()) {
      continuous.push_back({view.getProjectile(), view.getMediumHash(),
                            &view.getMedium()->get(), &view.getDisplacement()->get(),
                            view.getStochasticEnergyCut(), view.getParticleMassMeV()});
    }
    auto const original =
        native_table_reference::exportProposalNativeTables(interactions, continuous);
    auto const after = calculators.ordinary();
    result.referenceHash_ = original.content_hash;
    result.allRecordsEqual_ = same(result.table_, original);
    result.ordinaryStateUnchanged_ = same(before, middle) && same(before, after);
    result.processCount_ = calculators.getProcessCount();
    corsika::proposal::NativeInteractionCalculatorView const emptyInteraction;
    corsika::proposal::NativeContinuousCalculatorView const emptyContinuous;
    result.defaultViewsEmpty_ =
        !emptyInteraction.getMedium() && !emptyInteraction.getInteraction() &&
        !emptyInteraction.getPhotonPairLpm() && !emptyInteraction.getBremsLpm() &&
        !emptyContinuous.getMedium() && !emptyContinuous.getDisplacement();
    bool rejectedCurrent = false, rejectedOriginal = false;
    corsika::proposal::NativeInteractionCalculatorView const missingCalculator(
        Code::Photon, 0, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 0_MeV,
        0.);
    try {
      (void)export_proposal_native_tables({missingCalculator}, {});
    } catch (std::invalid_argument const&) { rejectedCurrent = true; }
    try {
      (void)native_table_reference::exportProposalNativeTables(
          {{Code::Photon, 0, nullptr, nullptr, nullptr, nullptr, 0_MeV, 0.}}, {});
    } catch (std::invalid_argument const&) { rejectedOriginal = true; }
    result.emptyViewsRejected_ = rejectedCurrent && rejectedOriginal;
    return result;
  }
} // namespace corsika::kokkos::test
