/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <CLI/CLI.hpp>

#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/Constants.h>
#include <PROPOSAL/crosssection/parametrization/PhotoPairProduction.h>
#include <PROPOSAL/math/Integral.h>
#include <PROPOSAL/particle/Particle.h>
#include <PROPOSAL/propagation_utility/InteractionBuilder.h>
#include <PROPOSAL/scattering/ScatteringFactory.h>
#include <PROPOSAL/scattering/multiple_scattering/Coefficients.h>
#include <PROPOSAL/secondaries/parametrization/epairproduction/KelnerKokoulinPetrukhinEpairProduction.h>
#include <PROPOSAL/version.h>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/gpu/em/BremsLpm.hpp>
#include <corsika/gpu/em/EpairLpm.hpp>
#include <corsika/gpu/em/EpairFinalState.hpp>
#include <corsika/gpu/em/MoliereScattering.hpp>
#include <corsika/gpu/em/PhotonPairKinematics.hpp>
#include <corsika/gpu/em/PhotonPairLpm.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>
#include <corsika/gpu/em/tables/MediumConfig.hpp>
#include <corsika/gpu/em/tables/ProposalMedium.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>
#include <corsika/modules/proposal/ProposalRateProvider.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

  using namespace corsika;
  using namespace corsika::gpu::em;
  using namespace corsika::gpu::em::tables;
  using namespace corsika::units::si;

  struct Options {
    std::filesystem::path output;
    std::filesystem::path proposal_cache;
    std::filesystem::path medium_yaml;
    std::filesystem::path epair_rho_source;
    std::filesystem::path merge_em_source;
    std::filesystem::path merge_muon_source;
    // Photons have no rest mass, so the stochastic-rate domain must begin at
    // or below the configured kinetic-energy cut.  Charged particles born
    // below their continuous-table anchor are terminated by the device-side
    // ParticleCut path before a rate lookup.
    double energy_min_MeV{0.5};
    double energy_max_MeV{1.e12};
    double energy_cut_MeV{0.5};
    double transport_cut_MeV{0.5};
    double muon_transport_cut_MeV{300.};
    double photon_pair_final_state_min_MeV{
        PhotonPairFinalStateMinimumEnergyMeV};
    double tolerance{1.e-3};
    double loss_tolerance{1.e-3};
    std::size_t initial_intervals{16};
    std::size_t max_points{20000};
    std::size_t loss_initial_energy_intervals{8};
    std::size_t loss_initial_quantile_intervals{8};
    std::size_t loss_max_energy_points{4096};
    // A column rebuilt from direct numerical integration/root finding can
    // require a denser high-quantile energy grid than the ordinary cached
    // PROPOSAL interpolation.
    std::size_t direct_loss_max_energy_points{65536};
    std::string nonmonotonic_loss_policy{"proposal-monotone"};
    std::size_t loss_max_quantile_points{2048};
    std::size_t loss_validation_samples{64};
    double epair_rho_min_energy_MeV{
        EpairLossMinimumEnergyMeV};
    std::size_t epair_rho_energy_points{65};
    std::size_t epair_rho_v_points{65};
    std::size_t epair_rho_quantile_points{129};
    std::size_t epair_rho_validation_samples{4096};
    double epair_rho_tolerance{1.e-3};
    bool enable_epair_rho_table{false};
    bool include_muons{false};
    bool muons_only{false};
    bool overwrite{false};
  };

  struct CliExit {
    int code{};
  };

  struct RateKey {
    std::int32_t process_id{};
    std::uint64_t component_hash{};

    auto tie() const { return std::tie(process_id, component_hash); }
    bool operator<(RateKey const& other) const { return tie() < other.tie(); }
    bool operator==(RateKey const& other) const {
      return tie() == other.tie();
    }
  };

  struct ColumnInfo {
    std::string process_name;
    std::string parameterization;
    std::string target_name;
  };

  using Evaluation = std::map<RateKey, double>;

  std::string interactionName(PROPOSAL::InteractionType type) {
    auto const found = PROPOSAL::Type_Interaction_Name_Map.find(type);
    if (found == PROPOSAL::Type_Interaction_Name_Map.end()) {
      return "InteractionType(" +
             std::to_string(static_cast<std::int32_t>(type)) + ")";
    }
    return found->second;
  }

  bool sameMediumComponents(
      std::vector<MediumComponent> const& left,
      std::vector<MediumComponent> const& right) {
    if (left.size() != right.size()) {
      return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
      if (left[index].corsika_pid != right[index].corsika_pid ||
          left[index].proposal_hash != right[index].proposal_hash ||
          left[index].name != right[index].name ||
          left[index].number_fraction != right[index].number_fraction) {
        return false;
      }
    }
    return true;
  }

  class ProposalReferenceEvaluator {
  public:
    ProposalReferenceEvaluator(Code code, std::int32_t pdg_id,
                               std::string particle_name,
                               PROPOSAL::Medium medium,
                               double energy_cut_MeV)
        : code_(code)
        , pdg_id_(pdg_id)
        , particle_name_(std::move(particle_name))
        , medium_(std::move(medium)) {
      cross_sections_ = proposal::make_cross_sections(
          code_, medium_, energy_cut_MeV * 1_MeV, true);
      // Keep a non-interpolated oracle available for diagnosing narrow
      // numerical features in PROPOSAL's production interpolation.  It is
      // queried only when the interpolated inverse violates monotonicity, so
      // ordinary table generation retains its cached/interpolated cost.
      direct_cross_sections_ = proposal::make_cross_sections(
          code_, medium_, energy_cut_MeV * 1_MeV, false);
      interaction_ =
          PROPOSAL::make_interaction(cross_sections_, true, true);
      if (!interaction_) {
        throw std::runtime_error("PROPOSAL did not create an interaction calculator");
      }
      medium_hash_ = static_cast<std::uint64_t>(medium_.GetHash());
      for (auto const& component : medium_.GetComponents()) {
        auto const hash =
            static_cast<std::uint64_t>(component.GetHash());
        target_names_.emplace(hash, component.GetName());
        components_.emplace(hash, component);
      }
      if (code_ == Code::Photon) {
        photon_pair_final_state_ = std::make_unique<
            PROPOSAL::secondaries::
                PhotoPairProductionKochMotzSauter>(
            PROPOSAL::GammaDef(), medium_);
      } else if (
          code_ == Code::Electron || code_ == Code::Positron ||
          code_ == Code::MuMinus || code_ == Code::MuPlus) {
        displacement_ =
            PROPOSAL::make_displacement(cross_sections_, true);
        particle_mass_MeV_ = proposal::particle.at(code_).mass;
        if (code_ == Code::Electron) {
          epair_final_state_ = std::make_unique<
              PROPOSAL::secondaries::
                  KelnerKokoulinPetrukhinEpairProduction>(
              PROPOSAL::EMinusDef(), medium_);
        } else {
          epair_final_state_ = std::make_unique<
              PROPOSAL::secondaries::
                  KelnerKokoulinPetrukhinEpairProduction>(
              PROPOSAL::EPlusDef(), medium_);
        }
      }
      target_names_.emplace(medium_hash_, medium_.GetName());
    }

    Evaluation evaluate(double energy_MeV) {
      auto const table = provider_.rates(*interaction_, energy_MeV);
      auto const& entries = table.entries();
      auto const& native = table.nativeRates();
      if (entries.size() != native.size()) {
        throw std::runtime_error("PROPOSAL native/public rate lists differ");
      }

      Evaluation evaluation;
      std::map<RateKey, ColumnInfo> current_info;
      for (std::size_t i = 0; i < entries.size(); ++i) {
        auto const key =
            RateKey{static_cast<std::int32_t>(entries[i].type),
                    static_cast<std::uint64_t>(entries[i].component_hash)};
        auto const target = target_names_.find(key.component_hash);
        auto const target_name =
            target == target_names_.end()
                ? "component_hash_" + std::to_string(key.component_hash)
                : target->second;
        ColumnInfo info{interactionName(entries[i].type),
                        native[i].crosssection->GetParametrizationName(),
                        target_name};
        if (!evaluation.emplace(key, entries[i].rate).second) {
          throw std::runtime_error(
              "PROPOSAL returned duplicate process/component rates");
        }
        auto const cross = cross_sections_by_key_.find(key);
        if (cross == cross_sections_by_key_.end()) {
          cross_sections_by_key_.emplace(key, native[i].crosssection);
        } else if (cross->second->GetHash() !=
                   native[i].crosssection->GetHash()) {
          throw std::runtime_error(
              "PROPOSAL cross section changed for a rate column");
        }
        current_info.emplace(key, std::move(info));
      }
      if (column_info_.empty()) {
        column_info_ = std::move(current_info);
      } else if (current_info.size() != column_info_.size()) {
        throw std::runtime_error(
            "PROPOSAL rate column set changed with projectile energy");
      } else {
        for (auto const& [key, info] : current_info) {
          auto const existing = column_info_.find(key);
          if (existing == column_info_.end() ||
              existing->second.parameterization != info.parameterization ||
              existing->second.process_name != info.process_name) {
            throw std::runtime_error(
                "PROPOSAL rate column identity changed with energy");
          }
        }
      }
      return evaluation;
    }

    double rate(RateKey const& key, double energy_MeV) const {
      auto const cross = crossSection(key);
      auto const rate = cross->CalculatedNdx(
          energy_MeV, static_cast<std::size_t>(key.component_hash));
      if (!std::isfinite(rate) || rate < 0.) {
        throw std::runtime_error(
            "PROPOSAL returned an invalid reference column rate");
      }
      return rate;
    }

    double sampleLoss(RateKey const& key, double energy_MeV,
                      double quantile) const {
      if (!std::isfinite(quantile) ||
          quantile < LossQuantileMinimum ||
          quantile > LossQuantileMaximum) {
        throw std::invalid_argument(
            "loss quantile is outside the device Philox support");
      }
      auto const cross = crossSection(key);
      auto const column_rate = cross->CalculatedNdx(
          energy_MeV, static_cast<std::size_t>(key.component_hash));
      if (!(column_rate > 0.)) {
        throw std::domain_error(
            "cannot sample a stochastic loss from a zero-rate column");
      }
      double loss = 0.;
      try {
        loss = cross->CalculateStochasticLoss(
            static_cast<std::size_t>(key.component_hash), energy_MeV,
            quantile * column_rate);
      } catch (std::exception const& error) {
        std::ostringstream message;
        message << "PROPOSAL reference stochastic-loss sampling failed for "
                << particle_name_ << ", process=" << key.process_id
                << ", component=" << key.component_hash
                << ", E=" << std::setprecision(17) << energy_MeV
                << " MeV, u=" << quantile << ": " << error.what();
        throw std::runtime_error(message.str());
      }
      if (!std::isfinite(loss) || loss < 0. || loss > 1.) {
        throw std::runtime_error(
            "PROPOSAL returned an invalid stochastic loss fraction");
      }
      return loss;
    }

    double sampleDirectLoss(RateKey const& key, double energy_MeV,
                            double quantile) const {
      if (!std::isfinite(quantile) ||
          quantile < LossQuantileMinimum ||
          quantile > LossQuantileMaximum) {
        throw std::invalid_argument(
            "direct-loss quantile is outside the device Philox support");
      }
      auto const cross = directCrossSection(key);
      auto const column_rate = cross->CalculatedNdx(
          energy_MeV, static_cast<std::size_t>(key.component_hash));
      if (!(column_rate > 0.)) {
        throw std::domain_error(
            "cannot sample a direct stochastic loss from a zero-rate column");
      }
      auto const loss = cross->CalculateStochasticLoss(
          static_cast<std::size_t>(key.component_hash), energy_MeV,
          quantile * column_rate);
      if (!std::isfinite(loss) || loss < 0. || loss > 1.) {
        throw std::runtime_error(
            "PROPOSAL returned an invalid direct stochastic loss fraction");
      }
      return loss;
    }

    double samplePhotonPairFinalState(
        RateKey const& key, double energy_MeV, double quantile) {
      if (key.process_id != PhotonPairProcessId ||
          !photon_pair_final_state_) {
        throw std::invalid_argument(
            "photon-pair final-state sampler received a wrong process");
      }
      auto const component = components_.find(key.component_hash);
      if (component == components_.end()) {
        throw std::out_of_range(
            "photon-pair final-state component is unavailable");
      }
      auto const split = photon_pair_final_state_->CalculateRho(
          energy_MeV, quantile, component->second);
      if (!std::isfinite(split) || split < 0. || split > 1.) {
        throw std::runtime_error(
            "PROPOSAL returned an invalid photon-pair energy split");
      }
      double normalized_split = 0.;
      if (!encodePhotonPairNormalizedSplit(
              energy_MeV / 1000., split,
              normalized_split)) {
        std::ostringstream message;
        message
            << "PROPOSAL photon-pair split is outside its "
               "kinematic interval at E="
            << std::setprecision(17) << energy_MeV
            << " MeV, rho=" << split;
        throw std::runtime_error(message.str());
      }
      return normalized_split;
    }

    double sampleEpairRho(
        std::uint64_t component_hash, double energy_MeV,
        double v, double quantile) {
      if (!epair_final_state_) {
        throw std::logic_error(
            "electron-pair final state requested for a photon");
      }
      auto const component = components_.find(component_hash);
      if (component == components_.end()) {
        throw std::out_of_range(
            "electron-pair final-state component is unavailable");
      }
      auto const rho = epair_final_state_->CalculateRho(
          energy_MeV, v, component->second, quantile, 0.75);
      if (!std::isfinite(rho) || rho < 0. || rho >= 1.) {
        throw std::runtime_error(
            "PROPOSAL returned an invalid electron-pair rho");
      }
      return rho;
    }

    std::int32_t pdgId() const { return pdg_id_; }
    std::string const& particleName() const { return particle_name_; }
    std::uint64_t interactionHash() const {
      return static_cast<std::uint64_t>(
          interaction_->GetHash());
    }
    std::map<RateKey, ColumnInfo> const& columnInfo() const {
      return column_info_;
    }

    bool hasContinuousLoss() const {
      return displacement_ != nullptr;
    }

    double particleMassMeV() const {
      if (!hasContinuousLoss()) {
        throw std::logic_error(
            "photon has no continuous-loss particle mass");
      }
      return particle_mass_MeV_;
    }

    double continuousDedx(double energy_MeV) const {
      if (!hasContinuousLoss()) {
        throw std::logic_error(
            "continuous dE/dX requested for a photon");
      }
      double result = 0.;
      for (auto const& cross : cross_sections_) {
        result += cross->CalculatedEdx(energy_MeV);
      }
      if (!std::isfinite(result) || !(result > 0.)) {
        throw std::runtime_error(
            "PROPOSAL returned an invalid continuous dE/dX");
      }
      return result;
    }

    double continuousRange(
        double energy_MeV, double minimum_energy_MeV) {
      if (!hasContinuousLoss()) {
        throw std::logic_error(
            "continuous range requested for a photon");
      }
      if (energy_MeV == minimum_energy_MeV) {
        return 0.;
      }
      auto const result = displacement_->SolveTrackIntegral(
          energy_MeV, minimum_energy_MeV);
      if (!std::isfinite(result) || !(result > 0.)) {
        std::ostringstream message;
        message
            << "PROPOSAL returned an invalid continuous range for "
            << particle_name_ << ": E=" << std::setprecision(17)
            << energy_MeV << " MeV, E_min="
            << minimum_energy_MeV << " MeV, range="
            << result << " g/cm^2";
        throw std::runtime_error(message.str());
      }
      return result;
    }

    double energyAfterContinuousGrammage(
        double initial_energy_MeV,
        double grammage_g_per_cm2) {
      if (!hasContinuousLoss() ||
          !std::isfinite(grammage_g_per_cm2) ||
          grammage_g_per_cm2 < 0.) {
        throw std::invalid_argument(
            "invalid PROPOSAL continuous inverse-range query");
      }
      auto const result =
          displacement_->UpperLimitTrackIntegral(
              initial_energy_MeV, grammage_g_per_cm2);
      if (!std::isfinite(result) ||
          !(result > particle_mass_MeV_) ||
          result > initial_energy_MeV) {
        throw std::runtime_error(
            "PROPOSAL returned an invalid final continuous energy");
      }
      return result;
    }

  private:
    std::shared_ptr<PROPOSAL::CrossSectionBase> crossSection(
        RateKey const& key) const {
      auto const cross = cross_sections_by_key_.find(key);
      if (cross == cross_sections_by_key_.end()) {
        throw std::out_of_range(
            "PROPOSAL cross section is unavailable for rate column");
      }
      return cross->second;
    }

    std::shared_ptr<PROPOSAL::CrossSectionBase> directCrossSection(
        RateKey const& key) const {
      auto const type = static_cast<PROPOSAL::InteractionType>(
          key.process_id);
      auto const cross = std::find_if(
          direct_cross_sections_.begin(), direct_cross_sections_.end(),
          [type](auto const& candidate) {
            return candidate->GetInteractionType() == type;
          });
      if (cross == direct_cross_sections_.end()) {
        throw std::out_of_range(
            "PROPOSAL direct cross section is unavailable for rate column");
      }
      return *cross;
    }

    Code code_;
    std::int32_t pdg_id_;
    std::string particle_name_;
    PROPOSAL::Medium medium_;
    std::uint64_t medium_hash_{};
    PROPOSAL::crosssection_list_t cross_sections_;
    PROPOSAL::crosssection_list_t direct_cross_sections_;
    std::unique_ptr<PROPOSAL::Interaction> interaction_;
    std::unique_ptr<PROPOSAL::Displacement> displacement_;
    double particle_mass_MeV_{};
    proposal::ProposalRateProvider provider_;
    std::map<std::uint64_t, std::string> target_names_;
    std::map<std::uint64_t, PROPOSAL::Component> components_;
    std::map<RateKey, ColumnInfo> column_info_;
    std::map<RateKey, std::shared_ptr<PROPOSAL::CrossSectionBase>>
        cross_sections_by_key_;
    std::unique_ptr<
        PROPOSAL::secondaries::PhotoPairProductionKochMotzSauter>
        photon_pair_final_state_;
    std::unique_ptr<
        PROPOSAL::secondaries::
            KelnerKokoulinPetrukhinEpairProduction>
        epair_final_state_;
  };

  double interpolate(double lower_energy, double upper_energy,
                     double lower_rate, double upper_rate,
                     double energy) {
    auto const fraction =
        (std::log(energy) - std::log(lower_energy)) /
        (std::log(upper_energy) - std::log(lower_energy));
    if (lower_rate > 0. && upper_rate > 0.) {
      return std::exp(std::log(lower_rate) +
                      fraction * (std::log(upper_rate) -
                                  std::log(lower_rate)));
    }
    return lower_rate + fraction * (upper_rate - lower_rate);
  }

  double relativeError(double expected, double actual) {
    auto const scale = std::max({expected, actual, 1.e-300});
    return std::abs(expected - actual) / scale;
  }

  double lossRelativeError(double expected, double actual) {
    auto const scale =
        std::max({expected, actual, LossFractionRelativeScaleFloor});
    return std::abs(expected - actual) / scale;
  }

  double quantileCoordinate(double quantile) {
    return std::log(quantile / (1. - quantile));
  }

  double quantileFromCoordinate(double coordinate) {
    if (coordinate >= 0.) {
      auto const exponential = std::exp(-coordinate);
      return 1. / (1. + exponential);
    }
    auto const exponential = std::exp(coordinate);
    return exponential / (1. + exponential);
  }

  double snapToPhiloxQuantile(double quantile) {
    auto const scaled =
        quantile * static_cast<double>(LossQuantileCount) - 0.5;
    auto const bounded =
        std::max(0., std::min(scaled,
                             static_cast<double>(
                                 std::numeric_limits<std::uint32_t>::max())));
    return lossQuantileFromPhiloxWord(
        static_cast<std::uint32_t>(std::llround(bounded)));
  }

  double maximumError(Evaluation const& lower, Evaluation const& upper,
                      Evaluation const& direct, double lower_energy,
                      double upper_energy, double energy) {
    if (lower.size() != upper.size() || lower.size() != direct.size()) {
      throw std::runtime_error("rate column set changed during interpolation test");
    }
    double maximum = 0.;
    for (auto const& [key, direct_rate] : direct) {
      auto const lower_rate = lower.find(key);
      auto const upper_rate = upper.find(key);
      if (lower_rate == lower.end() || upper_rate == upper.end()) {
        throw std::runtime_error(
            "rate column disappeared during interpolation test");
      }
      auto const interpolated =
          interpolate(lower_energy, upper_energy, lower_rate->second,
                      upper_rate->second, energy);
      maximum = std::max(maximum, relativeError(direct_rate, interpolated));
    }
    return maximum;
  }

  struct AdaptiveResult {
    ParticleRateTable table;
    double maximum_error{};
    double maximum_loss_error{};
  };

  double interpolateLoss(double lower, double upper, double fraction) {
    if (lower > 0. && upper > 0.) {
      return std::exp(std::log(lower) +
                      fraction * (std::log(upper) - std::log(lower)));
    }
    return lower + fraction * (upper - lower);
  }

  struct InverseCdfResult {
    InverseCdfTable table;
    double maximum_error{};
    std::size_t evaluations{};
  };

  enum class LossReference {
    ProposalInterpolated,
    ProposalDirect,
  };

  inline constexpr std::string_view ProposalMonotonePolicy =
      "proposal-monotone";
  inline constexpr std::string_view ProposalDirectPolicy =
      "proposal-direct";

  class NonMonotonicProposalInverseCdf final
      : public std::runtime_error {
  public:
    explicit NonMonotonicProposalInverseCdf(std::string message)
        : std::runtime_error(std::move(message)) {}
  };

  InverseCdfResult buildInverseCdf(ProposalReferenceEvaluator& evaluator,
                                   RateKey const& key,
                                   RateColumn const& rate_column,
                                   ParticleRateTable const& particle,
                                   Options const& options,
                                   bool final_state_split = false,
                                   LossReference reference =
                                       LossReference::ProposalInterpolated) {
    if (final_state_split &&
        reference == LossReference::ProposalDirect) {
      throw std::invalid_argument(
          "photon-pair final-state tables do not support the direct-loss "
          "reference mode");
    }
    // Adaptive probes are finite. Keep headroom for the independent
    // low-discrepancy validation points that are deliberately not part of the
    // refinement sequence.
    auto const refinement_tolerance =
        0.75 * options.loss_tolerance;
    auto const first_positive =
        std::find_if(rate_column.rates_cm2_per_g.begin(),
                     rate_column.rates_cm2_per_g.end(),
                     [](double rate) { return rate > 0.; });
    if (first_positive == rate_column.rates_cm2_per_g.end()) {
      return {};
    }
    auto const last_positive =
        std::find_if(rate_column.rates_cm2_per_g.rbegin(),
                     rate_column.rates_cm2_per_g.rend(),
                     [](double rate) { return rate > 0.; });
    auto const first_index = static_cast<std::size_t>(
        first_positive - rate_column.rates_cm2_per_g.begin());
    auto const last_index = rate_column.rates_cm2_per_g.size() - 1 -
                            static_cast<std::size_t>(
                                last_positive -
                                rate_column.rates_cm2_per_g.rbegin());
    auto active_min = particle.energies_MeV[first_index];
    auto active_max = particle.energies_MeV[last_index];
    auto const is_epair =
        key.process_id ==
        static_cast<std::int32_t>(PROPOSAL::InteractionType::Epair);
    auto const is_brems =
        key.process_id ==
        static_cast<std::int32_t>(PROPOSAL::InteractionType::Brems);
    auto const is_ionization =
        key.process_id ==
        static_cast<std::int32_t>(PROPOSAL::InteractionType::Ioniz);
    if (is_epair) {
      active_min = std::max(active_min, EpairLossMinimumEnergyMeV);
    } else if (is_brems) {
      active_min = std::max(active_min, BremsLossMinimumEnergyMeV);
    }
    if (final_state_split) {
      active_min = std::max(
          active_min,
          options.photon_pair_final_state_min_MeV);
    }
    if (!(active_max > active_min)) {
      throw std::runtime_error(
          "positive rate column has fewer than two active energy points");
    }

    using LossSampleKey = std::pair<double, double>;
    std::map<LossSampleKey, double> samples;
    std::size_t sample_evaluations = 0;
    auto erase_samples_for_energy = [&](double energy) {
      auto const first = samples.lower_bound(
          {energy, -std::numeric_limits<double>::infinity()});
      auto const last = samples.upper_bound(
          {energy, std::numeric_limits<double>::infinity()});
      samples.erase(first, last);
    };
    auto sample = [&](double energy, double quantile) {
      auto const sample_key = std::make_pair(energy, quantile);
      auto found = samples.find(sample_key);
      if (found == samples.end()) {
        auto const value =
            final_state_split
                ? evaluator.samplePhotonPairFinalState(
                      key, energy, quantile)
                : (reference == LossReference::ProposalDirect
                       ? evaluator.sampleDirectLoss(
                             key, energy, quantile)
                       : evaluator.sampleLoss(
                             key, energy, quantile));
        found = samples.emplace(sample_key, value).first;
        ++sample_evaluations;
      }
      return found->second;
    };

    constexpr std::array<double, 3> ProbeFractions{0.25, 0.5, 0.75};
    auto const quantile_minimum =
        final_state_split
            ? PhotonPairFinalStateQuantileMinimum
            : (is_epair ? EpairLossQuantileMinimum
                        : LossQuantileMinimum);
    auto const quantile_maximum =
        final_state_split
            ? PhotonPairFinalStateQuantileMaximum
            : (is_epair
                   ? EpairLossQuantileMaximum
                   : (is_ionization
                          ? IonizationLossQuantileMaximum
                          : LossQuantileMaximum));
    std::cout << "    "
              << (final_state_split ? "final-state " : "")
              << "inverse-CDF domain "
              << rate_column.process_name << "/"
              << rate_column.target_name << ": E=["
              << active_min << ", " << active_max << "] MeV, u=["
              << quantile_minimum << ", " << quantile_maximum
              << "], reference="
              << (reference == LossReference::ProposalDirect
                      ? "proposal_direct"
                      : "proposal_interpolated")
              << '\n';
    auto const quantile_coordinate_min =
        quantileCoordinate(quantile_minimum);
    auto const quantile_coordinate_max =
        quantileCoordinate(quantile_maximum);

    struct LossRow {
      std::vector<double> quantiles;
      std::vector<double> losses;
      double maximum_error{};
      double repaired_quantile_min{
          std::numeric_limits<double>::infinity()};
    };

    auto const monotone_projection =
        reference == LossReference::ProposalInterpolated &&
        options.nonmonotonic_loss_policy == ProposalMonotonePolicy;
    std::size_t monotone_repair_count = 0;
    double maximum_monotone_repair = 0.;
    double minimum_monotone_repair_quantile =
        std::numeric_limits<double>::infinity();
    std::size_t ignored_branch_probe_count = 0;
    double maximum_ignored_branch_deviation = 0.;

    auto build_row = [&](double energy) {
      std::set<double> grid{quantile_minimum, quantile_maximum};
      for (std::size_t i = 1;
           i < options.loss_initial_quantile_intervals; ++i) {
        auto const fraction =
            static_cast<double>(i) /
            options.loss_initial_quantile_intervals;
        grid.insert(snapToPhiloxQuantile(quantileFromCoordinate(
            quantile_coordinate_min +
            fraction *
                (quantile_coordinate_max - quantile_coordinate_min))));
      }
      double accepted_error = 0.;
      while (true) {
        std::set<double> insertions;
        double maximum_error = 0.;
        double maximum_quantile = 0.;
        double maximum_direct = 0.;
        double maximum_interpolated = 0.;
        double maximum_lower_quantile = 0.;
        double maximum_upper_quantile = 0.;
        double maximum_lower_loss = 0.;
        double maximum_upper_loss = 0.;
        for (auto lower = grid.begin(); lower != grid.end();) {
          auto upper = std::next(lower);
          if (upper == grid.end()) {
            break;
          }
          auto const lower_loss = sample(energy, *lower);
          auto const upper_loss = sample(energy, *upper);
          auto const lower_coordinate = quantileCoordinate(*lower);
          auto const upper_coordinate = quantileCoordinate(*upper);
          for (auto const fraction : ProbeFractions) {
            auto const quantile = snapToPhiloxQuantile(quantileFromCoordinate(
                lower_coordinate +
                fraction * (upper_coordinate - lower_coordinate)));
            if (!(quantile > *lower && quantile < *upper)) {
              continue;
            }
            auto const direct = sample(energy, quantile);
            auto const interpolated =
                interpolateLoss(lower_loss, upper_loss, fraction);
            auto const error =
                lossRelativeError(direct, interpolated);
            if (error > maximum_error) {
              maximum_error = error;
              maximum_quantile = quantile;
              maximum_direct = direct;
              maximum_interpolated = interpolated;
              maximum_lower_quantile = *lower;
              maximum_upper_quantile = *upper;
              maximum_lower_loss = lower_loss;
              maximum_upper_loss = upper_loss;
            }
            if (error > refinement_tolerance) {
              insertions.insert(quantile);
            }
          }
          lower = upper;
        }
        for (auto iterator = insertions.begin();
             iterator != insertions.end();) {
          if (grid.find(*iterator) != grid.end()) {
            iterator = insertions.erase(iterator);
          } else {
            ++iterator;
          }
        }
        if (insertions.empty()) {
          if (maximum_error > refinement_tolerance) {
            std::ostringstream message;
            message << evaluator.particleName() << "/"
                    << rate_column.process_name << "/"
                    << rate_column.target_name
                    << " inverse-CDF quantile row reached floating-point "
                       "resolution; E="
                    << std::setprecision(17) << energy
                    << ", u=" << maximum_quantile
                    << ", error=" << maximum_error
                    << ", direct v=" << maximum_direct
                    << ", interpolated v=" << maximum_interpolated
                    << ", bracket u=[" << maximum_lower_quantile << ", "
                    << maximum_upper_quantile << "]"
                    << ", bracket v=[" << maximum_lower_loss << ", "
                    << maximum_upper_loss << "]";
            throw std::runtime_error(message.str());
          }
          accepted_error = maximum_error;
          break;
        }
        if (grid.size() + insertions.size() >
            options.loss_max_quantile_points) {
          std::ostringstream message;
          message << evaluator.particleName() << "/"
                  << rate_column.process_name << "/"
                  << rate_column.target_name
                  << " inverse-CDF row exceeds --loss-max-quantile-points; "
                  << "E=" << std::setprecision(17) << energy
                  << ", u points=" << grid.size() + insertions.size()
                  << ", error=" << maximum_error
                  << " at u=" << maximum_quantile
                  << ", direct v=" << maximum_direct
                  << ", interpolated v=" << maximum_interpolated;
          throw std::runtime_error(message.str());
        }
        grid.insert(insertions.begin(), insertions.end());
      }
      LossRow row;
      row.quantiles.assign(grid.begin(), grid.end());
      row.losses.reserve(row.quantiles.size());
      row.maximum_error = accepted_error;
      double previous = -1.;
      double previous_quantile = 0.;
      for (auto const quantile : row.quantiles) {
        auto loss = sample(energy, quantile);
        if (loss + 1.e-14 < previous) {
          auto const error = lossRelativeError(previous, loss);
          if (error > refinement_tolerance &&
              !monotone_projection) {
            double direct_previous =
                std::numeric_limits<double>::quiet_NaN();
            double direct_current =
                std::numeric_limits<double>::quiet_NaN();
            std::string direct_diagnostic;
            try {
              direct_previous = evaluator.sampleDirectLoss(
                  key, energy, previous_quantile);
              direct_current = evaluator.sampleDirectLoss(
                  key, energy, quantile);
            } catch (std::exception const& direct_error) {
              direct_diagnostic = direct_error.what();
            }
            std::ostringstream message;
            message << evaluator.particleName() << "/"
                    << rate_column.process_name << "/"
                    << rate_column.target_name
                    << " PROPOSAL inverse CDF is not monotonic; E="
                    << std::setprecision(17) << energy
                    << ", u changed from " << previous_quantile
                    << " to " << quantile << ", v changed from "
                    << previous << " to " << loss
                    << ", relative difference=" << error
                    << ", refinement tolerance="
                    << refinement_tolerance;
            if (direct_diagnostic.empty()) {
              message << ", proposal_direct v changed from "
                      << direct_previous << " to " << direct_current
                      << ", proposal_direct_monotonic="
                      << (direct_current + 1.e-14 >= direct_previous
                              ? "true"
                              : "false");
            } else {
              message << ", proposal_direct diagnostic failed: "
                      << direct_diagnostic;
            }
            throw NonMonotonicProposalInverseCdf(message.str());
          }
          if (error > refinement_tolerance) {
            ++monotone_repair_count;
            maximum_monotone_repair =
                std::max(maximum_monotone_repair, error);
            row.repaired_quantile_min =
                std::min(row.repaired_quantile_min, quantile);
            minimum_monotone_repair_quantile = std::min(
                minimum_monotone_repair_quantile, quantile);
          }
          // Project solver-scale fluctuations that remain inside adaptive
          // refinement headroom. In proposal-monotone mode, larger local
          // reversals are also projected and explicitly tagged in the table.
          loss = previous;
        }
        row.losses.push_back(loss);
        previous = loss;
        previous_quantile = quantile;
      }
      // Once a row is materialized, later energy refinement uses the compact
      // LossRow and never queries its direct samples again. Releasing these
      // map nodes bounds host memory for dense proposal_direct repair grids.
      erase_samples_for_energy(energy);
      return row;
    };

    auto interpolate_row = [&](LossRow const& row, double quantile) {
      auto const upper = std::lower_bound(
          row.quantiles.begin(), row.quantiles.end(), quantile);
      if (upper == row.quantiles.begin()) {
        return row.losses.front();
      }
      if (upper == row.quantiles.end()) {
        return row.losses.back();
      }
      auto const upper_index =
          static_cast<std::size_t>(upper - row.quantiles.begin());
      if (*upper == quantile) {
        return row.losses[upper_index];
      }
      auto const lower_index = upper_index - 1;
      auto const lower_coordinate =
          quantileCoordinate(row.quantiles[lower_index]);
      auto const upper_coordinate =
          quantileCoordinate(row.quantiles[upper_index]);
      auto const fraction =
          (quantileCoordinate(quantile) - lower_coordinate) /
          (upper_coordinate - lower_coordinate);
      return interpolateLoss(row.losses[lower_index],
                             row.losses[upper_index], fraction);
    };

    auto const energyCoordinate =
        [final_state_split](double energy) {
          return std::log(
              final_state_split
                  ? energy - PhotonPairThresholdMeV
                  : energy);
        };
    auto const energyFromCoordinate =
        [final_state_split](double coordinate) {
          auto const positive = std::exp(coordinate);
          return final_state_split
                     ? PhotonPairThresholdMeV + positive
                     : positive;
        };
    std::set<double> energy_grid{active_min, active_max};
    auto const log_min = energyCoordinate(active_min);
    auto const log_max = energyCoordinate(active_max);
    for (std::size_t i = 1; i < options.loss_initial_energy_intervals; ++i) {
      auto const fraction =
          static_cast<double>(i) / options.loss_initial_energy_intervals;
      auto const energy = energyFromCoordinate(
          log_min + fraction * (log_max - log_min));
      if (evaluator.rate(key, energy) > 0.) {
        energy_grid.insert(energy);
      }
    }

    std::map<double, LossRow> rows;
    auto row = [&](double energy) -> LossRow const& {
      auto found = rows.find(energy);
      if (found == rows.end()) {
        found = rows.emplace(energy, build_row(energy)).first;
      }
      return found->second;
    };
    for (auto const energy : energy_grid) {
      row(energy);
    }

    if (monotone_projection) {
      // Discover a moving reversal before the adaptive energy pass reaches
      // the first affected interval. This makes the upper-tail exclusion a
      // column policy rather than an order-dependent row accident.
      std::vector<double> diagnostic_energies;
      for (auto lower = energy_grid.begin(); lower != energy_grid.end();) {
        auto const upper = std::next(lower);
        if (upper == energy_grid.end()) {
          break;
        }
        auto const lower_log = energyCoordinate(*lower);
        auto const upper_log = energyCoordinate(*upper);
        for (auto const fraction : ProbeFractions) {
          auto const energy = energyFromCoordinate(
              lower_log + fraction * (upper_log - lower_log));
          if (evaluator.rate(key, energy) > 0.) {
            row(energy);
            diagnostic_energies.push_back(energy);
          }
        }
        lower = upper;
      }
      for (auto const energy : diagnostic_energies) {
        if (energy_grid.find(energy) == energy_grid.end()) {
          rows.erase(energy);
        }
      }
    }

    double accepted_maximum_error = 0.;
    std::size_t energy_refinement_iteration = 0;
    while (true) {
      ++energy_refinement_iteration;
      std::set<double> energy_insertions;
      double iteration_maximum = 0.;
      double maximum_energy = 0.;
      double maximum_quantile = 0.;
      double maximum_direct = 0.;
      double maximum_interpolated = 0.;
      for (auto lower = energy_grid.begin(); lower != energy_grid.end();) {
        auto upper = std::next(lower);
        if (upper == energy_grid.end()) {
          break;
        }
        auto const& lower_row = row(*lower);
        auto const& upper_row = row(*upper);
        std::set<double> validation_quantiles(
            lower_row.quantiles.begin(), lower_row.quantiles.end());
        validation_quantiles.insert(upper_row.quantiles.begin(),
                                    upper_row.quantiles.end());
        auto const existing_quantiles = std::vector<double>(
            validation_quantiles.begin(), validation_quantiles.end());
        for (std::size_t i = 1; i < existing_quantiles.size(); ++i) {
          auto const midpoint =
              snapToPhiloxQuantile(quantileFromCoordinate(
              0.5 * (quantileCoordinate(existing_quantiles[i - 1]) +
                     quantileCoordinate(existing_quantiles[i]))));
          if (midpoint > existing_quantiles[i - 1] &&
              midpoint < existing_quantiles[i]) {
            validation_quantiles.insert(midpoint);
          }
        }
        auto const lower_log = energyCoordinate(*lower);
        auto const upper_log = energyCoordinate(*upper);
        for (auto const fraction : ProbeFractions) {
          auto const energy = energyFromCoordinate(
              lower_log +
              fraction * (upper_log - lower_log));
          if (!(evaluator.rate(key, energy) > 0.)) {
            continue;
          }
          LossRow const* monotone_row = nullptr;
          if (monotone_projection) {
            monotone_row = &row(energy);
          }
          bool refine_energy = false;
          for (auto const quantile : validation_quantiles) {
            auto const direct =
                monotone_row != nullptr
                    ? interpolate_row(*monotone_row, quantile)
                    : sample(energy, quantile);
            auto const interpolated = interpolateLoss(
                interpolate_row(lower_row, quantile),
                interpolate_row(upper_row, quantile), fraction);
            auto const error =
                lossRelativeError(direct, interpolated);
            bool moving_reversal = false;
            if (error > refinement_tolerance) {
              auto const repaired_at =
                  [quantile](LossRow const& candidate) {
                    // A cumulative-maximum projection can remain active
                    // beyond the sampled reversal until the raw branch catches
                    // up. Treat the complete upper tail as ambiguous.
                    return quantile >=
                           candidate.repaired_quantile_min;
                  };
              moving_reversal =
                  monotone_projection &&
                  (quantile >=
                       minimum_monotone_repair_quantile ||
                   repaired_at(lower_row) ||
                   repaired_at(upper_row) ||
                   (monotone_row != nullptr &&
                    repaired_at(*monotone_row)));
              if (moving_reversal) {
                ++ignored_branch_probe_count;
                maximum_ignored_branch_deviation = std::max(
                    maximum_ignored_branch_deviation, error);
              } else {
                energy_insertions.insert(energy);
                refine_energy = true;
              }
            }
            if (!moving_reversal &&
                error > iteration_maximum) {
              iteration_maximum = error;
              maximum_energy = energy;
              maximum_quantile = quantile;
              maximum_direct = direct;
              maximum_interpolated = interpolated;
            }
          }
          // Values for an accepted probe energy are never needed again.
          // Samples for a candidate insertion remain cached until build_row
          // materializes that row and then releases them.
          if (!refine_energy) {
            erase_samples_for_energy(energy);
            if (monotone_row != nullptr &&
                energy_grid.find(energy) == energy_grid.end()) {
              rows.erase(energy);
            }
          }
        }
        lower = upper;
      }
      for (auto iterator = energy_insertions.begin();
           iterator != energy_insertions.end();) {
        if (energy_grid.find(*iterator) != energy_grid.end()) {
          iterator = energy_insertions.erase(iterator);
        } else {
          ++iterator;
        }
      }
      if (reference == LossReference::ProposalDirect) {
        std::cout
            << "    proposal_direct energy refinement "
            << energy_refinement_iteration
            << ": current points=" << energy_grid.size()
            << ", candidate points="
            << energy_grid.size() + energy_insertions.size()
            << ", probe max error=" << iteration_maximum
            << '\n'
            << std::flush;
      }
      if (energy_insertions.empty()) {
        if (iteration_maximum > refinement_tolerance) {
          // The requested probe lies between adjacent representable energy
          // coordinates.  As with the bounded-grid case below, exhausting
          // the 0.75 refinement headroom is not itself a failure when the
          // complete adaptive probe set still satisfies the user-facing
          // tolerance.  Retain the grid and let the independent validation
          // (which uses a different low-discrepancy sequence) make the final
          // acceptance decision.
          if (iteration_maximum <= options.loss_tolerance) {
            accepted_maximum_error = iteration_maximum;
            for (auto const& [energy, stored_row] : rows) {
              (void)energy;
              accepted_maximum_error =
                  std::max(accepted_maximum_error,
                           stored_row.maximum_error);
            }
            std::cout
                << "    inverse-CDF energy refinement reached "
                   "floating-point resolution with probe error="
                << iteration_maximum
                << " inside requested tolerance="
                << options.loss_tolerance
                << "; deferring final acceptance to independent validation"
                << '\n';
            break;
          }
          std::ostringstream message;
          message << evaluator.particleName() << "/"
                  << rate_column.process_name << "/"
                  << rate_column.target_name
                  << " inverse-CDF energy refinement reached floating-point "
                     "resolution with error="
                  << iteration_maximum << " at E="
                  << std::setprecision(17) << maximum_energy
                  << ", u=" << maximum_quantile
                  << ", direct v=" << maximum_direct
                  << ", interpolated v=" << maximum_interpolated
                  << ", tolerance=" << options.loss_tolerance;
          throw std::runtime_error(message.str());
        }
        accepted_maximum_error = iteration_maximum;
        for (auto const& [energy, stored_row] : rows) {
          (void)energy;
          accepted_maximum_error =
              std::max(accepted_maximum_error,
                       stored_row.maximum_error);
        }
        break;
      }
      if (energy_grid.size() + energy_insertions.size() >
          options.loss_max_energy_points) {
        // The 0.75 factor used by refinement_tolerance is headroom for the
        // independent low-discrepancy validation, not the user-facing
        // acceptance criterion. PROPOSAL's interpolated inverse CDF contains
        // a few small root-branch features (notably high-quantile positron
        // ionization) for which repeated bisection no longer reduces the
        // error. If the complete adaptive probe set is already inside the
        // requested tolerance, retain the bounded grid and let the mandatory
        // independent validation below decide acceptance. An error above the
        // requested tolerance remains fatal.
        if (iteration_maximum <= options.loss_tolerance) {
          accepted_maximum_error = iteration_maximum;
          for (auto const& [energy, stored_row] : rows) {
            (void)energy;
            accepted_maximum_error =
                std::max(accepted_maximum_error,
                         stored_row.maximum_error);
          }
          std::cout
              << "    inverse-CDF energy refinement reached "
              << options.loss_max_energy_points
              << " point limit with probe error="
              << iteration_maximum
              << " inside requested tolerance="
              << options.loss_tolerance
              << "; deferring final acceptance to independent validation"
              << '\n';
          break;
        }
        std::ostringstream message;
        message << evaluator.particleName() << "/"
                << rate_column.process_name << "/"
                << rate_column.target_name
                << " inverse-CDF energy grid exceeds "
                << (reference == LossReference::ProposalDirect
                        ? "--direct-loss-max-energy-points"
                        : "--loss-max-energy-points")
                << "; E points="
                << energy_grid.size() + energy_insertions.size()
                << ", error=" << iteration_maximum
                << " at E=" << std::setprecision(17) << maximum_energy
                << ", u=" << maximum_quantile
                << ", direct v=" << maximum_direct
                << ", interpolated v=" << maximum_interpolated;
        throw std::runtime_error(message.str());
      }
      energy_grid.insert(energy_insertions.begin(),
                         energy_insertions.end());
      for (auto const energy : energy_insertions) {
        row(energy);
      }
    }

    InverseCdfTable table;
    table.reference_mode =
        reference == LossReference::ProposalDirect
            ? "proposal_direct"
            : (monotone_repair_count != 0
                   ? "proposal_interpolated_monotone"
                   : "proposal_interpolated");
    table.quantile_offsets.push_back(0);
    for (auto const energy : energy_grid) {
      auto const& stored_row = row(energy);
      table.energies_MeV.push_back(energy);
      table.quantiles.insert(table.quantiles.end(),
                             stored_row.quantiles.begin(),
                             stored_row.quantiles.end());
      table.v_loss.insert(table.v_loss.end(), stored_row.losses.begin(),
                          stored_row.losses.end());
      table.quantile_offsets.push_back(table.quantiles.size());
    }
    if (monotone_repair_count != 0) {
      std::cout
          << "    proposal-monotone repair "
          << rate_column.process_name << "/"
          << rate_column.target_name
          << ": projected reversals=" << monotone_repair_count
          << ", maximum relative projection="
          << maximum_monotone_repair
          << ", ignored moving-branch probes="
          << ignored_branch_probe_count
          << ", maximum raw branch deviation="
          << maximum_ignored_branch_deviation << '\n';
    }
    return {std::move(table), accepted_maximum_error,
            sample_evaluations};
  }

  AdaptiveResult buildAdaptiveTable(ProposalReferenceEvaluator& evaluator,
                                    Options const& options) {
    std::map<double, Evaluation> samples;
    auto evaluate = [&](double energy) -> Evaluation const& {
      auto found = samples.find(energy);
      if (found == samples.end()) {
        found = samples.emplace(energy, evaluator.evaluate(energy)).first;
      }
      return found->second;
    };

    auto const log_min = std::log(options.energy_min_MeV);
    auto const log_max = std::log(options.energy_max_MeV);
    for (std::size_t i = 0; i <= options.initial_intervals; ++i) {
      auto const fraction =
          static_cast<double>(i) / options.initial_intervals;
      evaluate(std::exp(log_min + fraction * (log_max - log_min)));
    }
    // Preserve exact user-requested endpoints despite floating-point exp/log.
    if (samples.begin()->first != options.energy_min_MeV) {
      samples.erase(samples.begin());
      evaluate(options.energy_min_MeV);
    }
    if (samples.rbegin()->first != options.energy_max_MeV) {
      samples.erase(std::prev(samples.end()));
      evaluate(options.energy_max_MeV);
    }

    // A process can be exactly zero below its kinematic threshold and positive
    // immediately above it. Pure geometric refinement can approach such a threshold
    // forever without sampling the representable zero/positive boundary, leaving a
    // nominal 100% relative interpolation error. Locate every transition explicitly
    // in log-energy and insert the last-zero/first-positive pair. The subsequent
    // adaptive pass then sees only smooth regions, while the serialized table retains
    // the exact PROPOSAL threshold branch.
    struct RateThresholdBracket {
      RateKey key{};
      double lower_energy{};
      double upper_energy{};
      bool lower_positive{};
    };
    std::vector<RateThresholdBracket> threshold_brackets;
    for (auto lower = samples.begin(); lower != samples.end();) {
      auto const upper = std::next(lower);
      if (upper == samples.end()) {
        break;
      }
      for (auto const& [key, lower_rate] : lower->second) {
        auto const upper_rate = upper->second.find(key);
        if (upper_rate == upper->second.end()) {
          throw std::runtime_error(
              "rate column disappeared while locating process thresholds");
        }
        auto const lower_positive = lower_rate > 0.;
        auto const upper_positive = upper_rate->second > 0.;
        if (lower_positive != upper_positive) {
          threshold_brackets.push_back(
              {key, lower->first, upper->first,
               lower_positive});
        }
      }
      lower = upper;
    }
    std::size_t threshold_anchor_count = 0;
    for (auto const& bracket : threshold_brackets) {
      auto lower_energy = bracket.lower_energy;
      auto upper_energy = bracket.upper_energy;
      auto lower_positive = bracket.lower_positive;
      for (std::size_t iteration = 0; iteration < 128; ++iteration) {
        auto const midpoint = std::exp(
            0.5 * (std::log(lower_energy) +
                   std::log(upper_energy)));
        if (!(midpoint > lower_energy) ||
            !(midpoint < upper_energy)) {
          break;
        }
        // Do not retain every bisection probe in the production grid. Only the
        // final adjacent zero/positive endpoints are interpolation anchors.
        auto const midpoint_values =
            evaluator.evaluate(midpoint);
        auto const midpoint_rate =
            midpoint_values.find(bracket.key);
        if (midpoint_rate == midpoint_values.end()) {
          throw std::runtime_error(
              "rate column disappeared during process-threshold bisection");
        }
        auto const midpoint_positive =
            midpoint_rate->second > 0.;
        if (midpoint_positive == lower_positive) {
          lower_energy = midpoint;
        } else {
          upper_energy = midpoint;
        }
      }
      evaluate(lower_energy);
      evaluate(upper_energy);
      threshold_anchor_count +=
          lower_energy == upper_energy ? 1 : 2;
    }
    if (threshold_anchor_count != 0) {
      std::cout << "  " << evaluator.particleName()
                << ": inserted " << threshold_anchor_count
                << " exact process-threshold anchors\n";
    }

    constexpr std::array<double, 3> ProbeFractions{0.25, 0.5, 0.75};
    double accepted_maximum_error = 0.;
    std::size_t iteration = 0;
    while (true) {
      ++iteration;
      std::vector<double> insertions;
      double iteration_maximum = 0.;
      for (auto lower = samples.begin(); lower != samples.end();) {
        auto upper = std::next(lower);
        if (upper == samples.end()) {
          break;
        }
        auto const lower_log = std::log(lower->first);
        auto const upper_log = std::log(upper->first);
        for (auto const fraction : ProbeFractions) {
          auto const energy =
              std::exp(lower_log + fraction * (upper_log - lower_log));
          auto const direct = evaluator.evaluate(energy);
          auto const error =
              maximumError(lower->second, upper->second, direct,
                           lower->first, upper->first, energy);
          iteration_maximum = std::max(iteration_maximum, error);
          if (error > options.tolerance) {
            insertions.push_back(energy);
          }
        }
        lower = upper;
      }
      if (insertions.empty()) {
        accepted_maximum_error = iteration_maximum;
        break;
      }
      std::sort(insertions.begin(), insertions.end());
      insertions.erase(std::unique(insertions.begin(), insertions.end()),
                       insertions.end());
      if (samples.size() + insertions.size() > options.max_points) {
        std::ostringstream message;
        message << evaluator.particleName()
                << " rate grid needs more than --max-points="
                << options.max_points << "; current probe error is "
                << iteration_maximum;
        throw std::runtime_error(message.str());
      }
      for (auto const energy : insertions) {
        evaluate(energy);
      }
      std::cout << "  " << evaluator.particleName() << ": refinement "
                << iteration << ", points=" << samples.size()
                << ", probe max error=" << std::scientific
                << iteration_maximum << '\n';
    }

    ParticleRateTable particle;
    particle.pdg_id = evaluator.pdgId();
    particle.particle_name = evaluator.particleName();
    particle.interaction_hash =
        evaluator.interactionHash();
    particle.energies_MeV.reserve(samples.size());
    for (auto const& [energy, values] : samples) {
      (void)values;
      particle.energies_MeV.push_back(energy);
    }
    for (auto const& [key, info] : evaluator.columnInfo()) {
      RateColumn column;
      column.process_id = key.process_id;
      column.component_hash = key.component_hash;
      column.process_name = info.process_name;
      column.parameterization = info.parameterization;
      column.target_name = info.target_name;
      column.rates_cm2_per_g.reserve(samples.size());
      for (auto const& [energy, values] : samples) {
        (void)energy;
        auto const value = values.find(key);
        if (value == values.end()) {
          throw std::runtime_error("missing rate value while assembling table");
        }
        column.rates_cm2_per_g.push_back(value->second);
      }
      particle.columns.push_back(std::move(column));
    }

    double maximum_loss_error = 0.;
    std::vector<RateColumn> auxiliary_columns;
    for (auto& column : particle.columns) {
      auto const key =
          RateKey{column.process_id, column.component_hash};
      if (isMuonPid(evaluator.pdgId()) &&
          key.process_id != IonizationProcessId) {
        // The first CUDA muon backend accelerates transport and the dominant
        // discrete ionization branch. Rare radiative/photonuclear branches retain
        // exact GPU rate/process selection, but intentionally have no device inverse
        // CDF: the existing specified-process CPU fallback resolves v and generates
        // the final state without resampling the process. Near-threshold muon
        // brems/epair inverse CDFs are too stiff for the current log(total-energy)
        // coordinate and would otherwise inflate a production table by orders of
        // magnitude.
        std::cout << "    inverse-CDF " << column.process_name
                  << "/" << column.target_name
                  << ": CPU selected-process fallback (rate-only muon column)\n";
        continue;
      }
      InverseCdfResult result;
      try {
        result = buildInverseCdf(
            evaluator, key, column, particle, options);
      } catch (NonMonotonicProposalInverseCdf const& error) {
        if (options.nonmonotonic_loss_policy !=
            ProposalDirectPolicy) {
          throw;
        }
        // The scalar PROPOSAL 7.6.2 interpolant has a narrow non-monotonic
        // root branch for the air/argon bremsstrahlung column.  Resolve the
        // ambiguity once, while preparing the table: rebuild the complete
        // affected column against interpolate=false numerical integration
        // and root finding.  Runtime CUDA transport then remains entirely on
        // device and never pays a selected-loss CPU fallback for this issue.
        std::cout
            << "    inverse-CDF " << column.process_name << "/"
            << column.target_name
            << ": interpolated PROPOSAL reference is non-monotonic ("
            << error.what()
            << "); rebuilding this complete column from proposal_direct\n";
        auto direct_options = options;
        direct_options.loss_max_energy_points =
            options.direct_loss_max_energy_points;
        result = buildInverseCdf(
            evaluator, key, column, particle, direct_options, false,
            LossReference::ProposalDirect);
      }
      maximum_loss_error =
          std::max(maximum_loss_error, result.maximum_error);
      column.inverse_cdf = std::move(result.table);
      std::cout << "    inverse-CDF " << column.process_name << "/"
                << column.target_name << ": E="
                << column.inverse_cdf.energies_MeV.size()
                << ", u=" << column.inverse_cdf.quantiles.size()
                << ", samples=" << result.evaluations
                << ", max error=" << result.maximum_error << '\n';
      if (column.process_id == PhotonPairProcessId) {
        auto final_state = buildInverseCdf(
            evaluator, key, column, particle, options, true);
        maximum_loss_error = std::max(
            maximum_loss_error, final_state.maximum_error);
        RateColumn auxiliary;
        auxiliary.process_id = PhotonPairFinalStateProcessId;
        auxiliary.component_hash = column.component_hash;
        auxiliary.process_name = "PhotopairFinalState";
        auxiliary.parameterization =
            "PhotoPairProductionKochMotzSauterNormalizedKinematicSplit";
        auxiliary.target_name = column.target_name;
        auxiliary.rates_cm2_per_g.assign(
            particle.energies_MeV.size(), 0.);
        auxiliary.inverse_cdf = std::move(final_state.table);
        std::cout << "    final-state inverse-CDF "
                  << column.process_name << "/"
                  << column.target_name << ": E="
                  << auxiliary.inverse_cdf.energies_MeV.size()
                  << ", u="
                  << auxiliary.inverse_cdf.quantiles.size()
                  << ", samples=" << final_state.evaluations
                  << ", max error="
                  << final_state.maximum_error << '\n';
        auxiliary_columns.push_back(std::move(auxiliary));
      }
    }
    particle.columns.insert(
        particle.columns.end(),
        std::make_move_iterator(auxiliary_columns.begin()),
        std::make_move_iterator(auxiliary_columns.end()));
    return {std::move(particle), accepted_maximum_error,
            maximum_loss_error};
  }

  double validateInverseCdfAgainstReference(
      ProposalReferenceEvaluator& evaluator,
      ParticleRateTable const& particle,
      Options const& options) {
    constexpr double EnergySequence = 0.6180339887498948482;
    constexpr double QuantileSequence = 0.4142135623730950488;
    double maximum_error = 0.;
    std::size_t checked = 0;
    for (auto const& column : particle.columns) {
      auto const& inverse = column.inverse_cdf;
      if (inverse.energies_MeV.empty()) {
        continue;
      }
      if (inverse.reference_mode ==
          "proposal_interpolated_monotone") {
        // This reference is the cumulative-monotone projection assembled and
        // adaptively probed inside buildInverseCdf. A pointwise raw PROPOSAL
        // call would compare against the deliberately rejected reversal, not
        // against the selected table policy.
        std::cout
            << "    independent raw-PROPOSAL validation skipped for "
            << column.process_name << "/" << column.target_name
            << " (proposal-monotone policy; adaptive projected-reference "
               "validation already passed)\n";
        continue;
      }
      auto const row_end =
          static_cast<std::size_t>(inverse.quantile_offsets[1]);
      auto const quantile_minimum = inverse.quantiles.front();
      auto const quantile_maximum = inverse.quantiles[row_end - 1];
      auto const minimum_coordinate =
          quantileCoordinate(quantile_minimum);
      auto const maximum_coordinate =
          quantileCoordinate(quantile_maximum);
      auto const minimum_log_energy =
          std::log(inverse.energies_MeV.front());
      auto const maximum_log_energy =
          std::log(inverse.energies_MeV.back());
      auto const final_state_split =
          column.process_id == PhotonPairFinalStateProcessId;
      auto const key = RateKey{
          final_state_split ? PhotonPairProcessId
                            : column.process_id,
          column.component_hash};

      for (std::size_t sample_index = 0;
           sample_index < options.loss_validation_samples;
           ++sample_index) {
        auto const energy_fraction = std::fmod(
            (static_cast<double>(sample_index) + 0.5) *
                EnergySequence,
            1.);
        auto const quantile_fraction = std::fmod(
            (static_cast<double>(sample_index) + 0.5) *
                QuantileSequence,
            1.);
        auto const energy = std::exp(
            minimum_log_energy +
            energy_fraction *
                (maximum_log_energy - minimum_log_energy));
        auto quantile = snapToPhiloxQuantile(
            quantileFromCoordinate(
                minimum_coordinate +
                quantile_fraction *
                    (maximum_coordinate - minimum_coordinate)));
        quantile =
            std::max(quantile_minimum,
                     std::min(quantile, quantile_maximum));
        auto const direct =
            final_state_split
                ? evaluator.samplePhotonPairFinalState(
                      key, energy, quantile)
                : (inverse.reference_mode == "proposal_direct"
                       ? evaluator.sampleDirectLoss(
                             key, energy, quantile)
                       : evaluator.sampleLoss(
                             key, energy, quantile));
        auto const interpolated = interpolateLossFraction(
            particle, column.process_id, column.component_hash, energy,
            quantile);
        auto const error =
            lossRelativeError(direct, interpolated);
        maximum_error = std::max(maximum_error, error);
        ++checked;
        if (error > options.loss_tolerance) {
          std::ostringstream message;
          message << evaluator.particleName() << "/"
                  << column.process_name << "/" << column.target_name
                  << " failed independent inverse-CDF validation at E="
                  << std::setprecision(17) << energy
                  << " MeV, u=" << quantile
                  << ": direct v=" << direct
                  << ", table v=" << interpolated
                  << ", relative error=" << error
                  << ", tolerance=" << options.loss_tolerance;
          throw std::runtime_error(message.str());
        }
      }
    }
    std::cout << "  " << evaluator.particleName()
              << ": independent inverse-CDF validation samples="
              << checked << ", max error=" << maximum_error << '\n';
    return maximum_error;
  }

  std::vector<double> transformedGrid(
      double minimum, double maximum, std::size_t count,
      bool logarithmic, bool logit_axis) {
    if (count < 2 || !std::isfinite(minimum) ||
        !std::isfinite(maximum) || !(maximum > minimum)) {
      throw std::invalid_argument(
          "invalid transformed Epair rho-table axis");
    }
    auto transform =
        [&](double value) {
          if (logit_axis) {
            return std::log(value / (1. - value));
          }
          return logarithmic ? std::log(value) : value;
        };
    auto inverse =
        [&](double coordinate) {
          if (logit_axis) {
            if (coordinate >= 0.) {
              auto const tail = std::exp(-coordinate);
              return 1. / (1. + tail);
            }
            auto const head = std::exp(coordinate);
            return head / (1. + head);
          }
          return logarithmic ? std::exp(coordinate)
                             : coordinate;
        };
    auto const first = transform(minimum);
    auto const last = transform(maximum);
    std::vector<double> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      auto const fraction =
          static_cast<double>(index) /
          static_cast<double>(count - 1);
      result.push_back(
          inverse(first + fraction * (last - first)));
    }
    result.front() = minimum;
    result.back() = maximum;
    return result;
  }

  EpairRhoInverseCdfTable buildEpairRhoTable(
      ParticleRateTable const& particle,
      BremsLpmSnapshot const& lpm_snapshot,
      std::vector<MediumComponent> const& medium_components,
      Options const& options) {
    if (particle.pdg_id != 11) {
      throw std::invalid_argument(
          "Epair rho table requires the electron rate table");
    }
    EpairRhoInverseCdfTable table;
    table.enabled = true;
    table.reference_mode = "proposal_compatible_kkp";
    table.requested_max_normalized_error =
        options.epair_rho_tolerance;
    table.component_hashes.reserve(
        medium_components.size());
    std::vector<RateColumn const*> epair_columns;
    epair_columns.reserve(medium_components.size());

    double active_minimum =
        std::max(
            options.epair_rho_min_energy_MeV,
            options.energy_min_MeV);
    double active_maximum = options.energy_max_MeV;
    for (auto const& medium_component : medium_components) {
      auto const column = std::find_if(
          particle.columns.begin(), particle.columns.end(),
          [&](RateColumn const& candidate) {
            return candidate.process_id ==
                       ElectronPairProcessId &&
                   candidate.component_hash ==
                       medium_component.proposal_hash;
          });
      if (column == particle.columns.end() ||
          column->inverse_cdf.energies_MeV.size() < 2) {
        throw std::runtime_error(
            "electron rate table lacks an active Epair column");
      }
      active_minimum = std::max(
          active_minimum,
          column->inverse_cdf.energies_MeV.front());
      active_maximum = std::min(
          active_maximum,
          column->inverse_cdf.energies_MeV.back());
      table.component_hashes.push_back(
          medium_component.proposal_hash);
      epair_columns.push_back(&*column);
    }
    if (!(active_maximum > active_minimum)) {
      throw std::runtime_error(
          "electron-pair rho table has no common energy domain");
    }

    table.energies_MeV = transformedGrid(
        active_minimum, active_maximum,
        options.epair_rho_energy_points, true, false);
    auto thresholdCoordinate =
        [](double energy, double v) {
          auto const coordinate =
              energy * v / (4. * PROPOSAL::ME) - 1.;
          if (!std::isfinite(coordinate) ||
              !(coordinate > 0.)) {
            throw std::runtime_error(
                "Epair loss cannot be mapped to threshold excess");
          }
          return coordinate;
        };
    double v_coordinate_minimum = 1.;
    double v_coordinate_maximum = 0.;
    for (std::size_t component_index = 0;
         component_index < table.component_hashes.size();
         ++component_index) {
      auto const component_hash =
          table.component_hashes[component_index];
      auto accumulateDomain =
          [&](double energy) {
            if (energy < active_minimum ||
                energy > active_maximum) {
              return;
            }
            for (auto const quantile :
                 {EpairLossQuantileMinimum,
                  EpairLossQuantileMaximum}) {
              auto const v = interpolateLossFraction(
                  particle, ElectronPairProcessId,
                  component_hash, energy, quantile);
              auto const coordinate =
                  thresholdCoordinate(energy, v);
              v_coordinate_minimum = std::min(
                  v_coordinate_minimum, coordinate);
              v_coordinate_maximum = std::max(
                  v_coordinate_maximum, coordinate);
            }
          };
      for (auto const energy : table.energies_MeV) {
        accumulateDomain(energy);
      }
      for (auto const energy :
           epair_columns[component_index]
               ->inverse_cdf.energies_MeV) {
        accumulateDomain(energy);
      }
    }
    if (!(v_coordinate_maximum >
          v_coordinate_minimum)) {
      throw std::runtime_error(
          "Epair rho table has an empty normalized-v domain");
    }
    auto const domain_padding = 0.5;
    v_coordinate_minimum = std::exp(
        std::log(v_coordinate_minimum) -
        domain_padding);
    v_coordinate_maximum = std::exp(
        std::log(v_coordinate_maximum) +
        domain_padding);
    table.v_coordinates = transformedGrid(
        v_coordinate_minimum, v_coordinate_maximum,
        options.epair_rho_v_points, true, false);
    table.rho_quantiles = transformedGrid(
        EpairRhoQuantileMinimum,
        EpairRhoQuantileMaximum,
        options.epair_rho_quantile_points, false, true);

    auto normalizedRho =
        [&](std::uint64_t component_hash, double energy,
            double v_coordinate, double rho_quantile) {
          auto const lepton_mass = PROPOSAL::EMinusDef().mass;
          auto const v_maximum =
              1. -
              6. * lepton_mass * lepton_mass /
                  (energy * energy);
          auto v =
              4. * PROPOSAL::ME *
              (1. + v_coordinate) / energy;
          // The global rectangular s_v grid extends beyond the recoil
          // boundary at its lowest energies. Continue the row with the last
          // physical value; runtime Epair losses never query this padding.
          if (v >= v_maximum) {
            v = std::nextafter(v_maximum, 0.);
          }
          auto const threshold =
              1. - 4. * PROPOSAL::ME / (energy * v);
          auto const recoil =
              1. -
              6. * PROPOSAL::EMinusDef().mass *
                  PROPOSAL::EMinusDef().mass /
                  (energy * energy * (1. - v));
          if (!(threshold > 0.) || !(recoil > 0.)) {
            throw std::runtime_error(
                "tabulated Epair loss is outside rho kinematics");
          }
          auto const rho_max =
              std::sqrt(threshold) * recoil;
          auto const sampled = sampleEpairRho(
              lpm_snapshot, component_hash, energy, v,
              rho_quantile, 0.75);
          if (sampled.status !=
              EpairFinalStateStatus::Success) {
            throw std::runtime_error(
                "PROPOSAL-compatible KKP rho sampling failed");
          }
          auto const rho = sampled.rho;
          auto const normalized = rho / rho_max;
          if (!std::isfinite(normalized) ||
              normalized < 0. || normalized > 1.) {
            throw std::runtime_error(
                "PROPOSAL Epair rho cannot be normalized");
          }
          return normalized;
        };

    auto const value_count =
        table.component_hashes.size() *
        table.energies_MeV.size() *
        table.v_coordinates.size() *
        table.rho_quantiles.size();
    table.normalized_rho.reserve(value_count);
    double maximum_monotonic_projection = 0.;
    std::uint64_t projection_component = 0;
    double projection_energy = 0.;
    double projection_v_coordinate = 0.;
    double projection_rho_quantile = 0.;
    double projection_direct = 0.;
    double projection_previous = 0.;
    for (auto const component_hash :
         table.component_hashes) {
      std::cout
          << "    Epair rho inverse-CDF component="
          << component_hash << ": E="
          << table.energies_MeV.size() << ", s_v="
          << table.v_coordinates.size() << ", rho-u="
          << table.rho_quantiles.size() << '\n';
      for (auto const energy : table.energies_MeV) {
        for (auto const v_coordinate :
             table.v_coordinates) {
          double previous = 0.;
          for (auto const rho_quantile :
               table.rho_quantiles) {
            auto value = normalizedRho(
                component_hash, energy, v_coordinate,
                rho_quantile);
            if (value < previous) {
              auto const projection = previous - value;
              if (projection >
                  maximum_monotonic_projection) {
                maximum_monotonic_projection = projection;
                projection_component = component_hash;
                projection_energy = energy;
                projection_v_coordinate = v_coordinate;
                projection_rho_quantile = rho_quantile;
                projection_direct = value;
                projection_previous = previous;
              }
              value = previous;
            }
            table.normalized_rho.push_back(value);
            previous = value;
          }
        }
      }
    }

    FlatRateTableView view{};
    view.epair_rho_component_hashes =
        table.component_hashes.data();
    view.epair_rho_energies_MeV =
        table.energies_MeV.data();
    view.epair_rho_v_coordinates =
        table.v_coordinates.data();
    view.epair_rho_quantiles =
        table.rho_quantiles.data();
    view.epair_rho_values =
        table.normalized_rho.data();
    view.epair_rho_component_count =
        static_cast<std::uint32_t>(
            table.component_hashes.size());
    view.epair_rho_energy_count =
        static_cast<std::uint32_t>(
            table.energies_MeV.size());
    view.epair_rho_v_coordinate_count =
        static_cast<std::uint32_t>(
            table.v_coordinates.size());
    view.epair_rho_quantile_count =
        static_cast<std::uint32_t>(
            table.rho_quantiles.size());
    view.epair_rho_value_count =
        static_cast<std::uint32_t>(
            table.normalized_rho.size());

    constexpr double EnergySequence =
        0.6180339887498948482;
    constexpr double VSequence =
        0.4142135623730950488;
    constexpr double RhoSequence =
        0.7320508075688772935;
    auto const log_energy_minimum =
        std::log(table.energies_MeV.front());
    auto const log_energy_maximum =
        std::log(table.energies_MeV.back());
    auto const loss_coordinate_minimum =
        quantileCoordinate(EpairLossQuantileMinimum);
    auto const loss_coordinate_maximum =
        quantileCoordinate(EpairLossQuantileMaximum);
    auto const rho_coordinate_minimum =
        quantileCoordinate(table.rho_quantiles.front());
    auto const rho_coordinate_maximum =
        quantileCoordinate(table.rho_quantiles.back());
    double maximum_error = 0.;
    double maximum_validation_error = 0.;
    std::uint64_t validation_component = 0;
    double validation_energy = 0.;
    double validation_v_coordinate = 0.;
    double validation_rho_quantile = 0.;
    double validation_direct = 0.;
    double validation_interpolated = 0.;
    for (std::size_t component_index = 0;
         component_index < table.component_hashes.size();
         ++component_index) {
      auto const component_hash =
          table.component_hashes[component_index];
      for (std::size_t sample_index = 0;
           sample_index <
               options.epair_rho_validation_samples;
           ++sample_index) {
        auto const index =
            static_cast<double>(sample_index) +
            0.5 +
            static_cast<double>(component_index) * 0.137;
        auto const energy_fraction =
            std::fmod(index * EnergySequence, 1.);
        auto const loss_fraction =
            std::fmod(index * VSequence, 1.);
        auto const rho_fraction =
            std::fmod(index * RhoSequence, 1.);
        auto const energy = std::exp(
            log_energy_minimum +
            energy_fraction *
                (log_energy_maximum -
                 log_energy_minimum));
        auto const loss_quantile = quantileFromCoordinate(
            loss_coordinate_minimum +
            loss_fraction *
                (loss_coordinate_maximum -
                 loss_coordinate_minimum));
        auto const v = interpolateLossFraction(
            particle, ElectronPairProcessId,
            component_hash, energy, loss_quantile);
        auto const v_coordinate =
            thresholdCoordinate(energy, v);
        if (v_coordinate <
                table.v_coordinates.front() ||
            v_coordinate >
                table.v_coordinates.back()) {
          std::ostringstream message;
          message
              << "independent Epair loss escaped the rho-table v domain"
              << ": component=" << component_hash
              << ", E=" << energy
              << ", loss_u=" << loss_quantile
              << ", v=" << v
              << ", s_v=" << v_coordinate
              << ", domain=["
              << table.v_coordinates.front() << ", "
              << table.v_coordinates.back() << "]";
          throw std::runtime_error(message.str());
        }
        auto const rho_quantile = quantileFromCoordinate(
            rho_coordinate_minimum +
            rho_fraction *
                (rho_coordinate_maximum -
                 rho_coordinate_minimum));
        auto const direct = normalizedRho(
            component_hash, energy, v_coordinate,
            rho_quantile);
        auto const interpolated = interpolateEpairRho(
            view, component_hash, energy, v_coordinate,
            rho_quantile);
        if (interpolated.status !=
            TableLookupStatus::Success) {
          throw std::runtime_error(
              "generated Epair rho table is not queryable");
        }
        auto const error = std::abs(
            direct - interpolated.normalized_rho);
        if (error > maximum_validation_error) {
          maximum_validation_error = error;
          validation_component = component_hash;
          validation_energy = energy;
          validation_v_coordinate = v_coordinate;
          validation_rho_quantile = rho_quantile;
          validation_direct = direct;
          validation_interpolated =
              interpolated.normalized_rho;
        }
        maximum_error = std::max(maximum_error, error);
      }
    }
    table.measured_max_normalized_error =
        maximum_error;
    if (maximum_error >
        table.requested_max_normalized_error) {
      auto lowerIndex =
          [](std::vector<double> const& axis,
             double value) {
            auto upper = std::lower_bound(
                axis.begin(), axis.end(), value);
            if (upper == axis.begin()) {
              return std::size_t{0};
            }
            if (upper == axis.end()) {
              return axis.size() - 2;
            }
            return static_cast<std::size_t>(
                       upper - axis.begin()) -
                   1;
          };
      auto const diagnostic_component =
          static_cast<std::size_t>(
              std::find(
                  table.component_hashes.begin(),
                  table.component_hashes.end(),
                  validation_component) -
              table.component_hashes.begin());
      auto const diagnostic_energy = lowerIndex(
          table.energies_MeV, validation_energy);
      auto const diagnostic_v = lowerIndex(
          table.v_coordinates,
          validation_v_coordinate);
      auto const diagnostic_rho = lowerIndex(
          table.rho_quantiles,
          validation_rho_quantile);
      std::ostringstream message;
      message
          << "Epair rho inverse-CDF validation failed: "
          << "max normalized error=" << maximum_error
          << ", tolerance="
          << table.requested_max_normalized_error
          << ", monotonic projection="
          << maximum_monotonic_projection
          << " at component=" << projection_component
          << ", E=" << projection_energy
          << ", s_v=" << projection_v_coordinate
          << ", rho_u=" << projection_rho_quantile
          << ", direct=" << projection_direct
          << ", previous=" << projection_previous
          << ", validation error="
          << maximum_validation_error
          << " at component=" << validation_component
          << ", E=" << validation_energy
          << ", s_v=" << validation_v_coordinate
          << ", rho_u=" << validation_rho_quantile
          << ", direct=" << validation_direct
          << ", interpolated="
          << validation_interpolated
          << ", energy bracket=["
          << table.energies_MeV[diagnostic_energy]
          << ", "
          << table.energies_MeV[diagnostic_energy + 1]
          << "], s_v bracket=["
          << table.v_coordinates[diagnostic_v]
          << ", "
          << table.v_coordinates[diagnostic_v + 1]
          << "], rho_u bracket=["
          << table.rho_quantiles[diagnostic_rho]
          << ", "
          << table.rho_quantiles[diagnostic_rho + 1]
          << "], corners=";
      for (std::size_t energy_side = 0;
           energy_side < 2; ++energy_side) {
        for (std::size_t v_side = 0;
             v_side < 2; ++v_side) {
          for (std::size_t rho_side = 0;
               rho_side < 2; ++rho_side) {
            auto const value_index =
                (((diagnostic_component *
                       table.energies_MeV.size() +
                   diagnostic_energy + energy_side) *
                      table.v_coordinates.size() +
                  diagnostic_v + v_side) *
                     table.rho_quantiles.size() +
                 diagnostic_rho + rho_side);
            message << table.normalized_rho[value_index]
                    << ",";
          }
        }
      }
      message
          << "; increase --epair-rho-*-points";
      throw std::runtime_error(message.str());
    }
    std::cout
        << "  Epair rho inverse-CDF: values="
        << table.normalized_rho.size()
        << ", bytes="
        << table.normalized_rho.size() * sizeof(double)
        << ", max normalized error=" << maximum_error
        << ", max monotonic projection="
        << maximum_monotonic_projection << '\n';
    return table;
  }

  ContinuousEnergyTable buildContinuousEnergyTable(
      ProposalReferenceEvaluator& evaluator,
      Options const& options) {
    if (!evaluator.hasContinuousLoss()) {
      throw std::invalid_argument(
          "continuous table requested for a particle without continuous loss");
    }
    auto const minimum_energy =
        evaluator.particleMassMeV() +
        ContinuousCutSafetyFactor *
            options.transport_cut_MeV;
    if (!(options.energy_max_MeV > minimum_energy)) {
      throw std::invalid_argument(
          "transport cut leaves no continuous energy domain");
    }

    struct Sample {
      double dEdX{};
      double range{};
    };
    std::map<double, Sample> samples;
    auto sample = [&](double energy) -> Sample const& {
      auto found = samples.find(energy);
      if (found == samples.end()) {
        found = samples
                    .emplace(
                        energy,
                        Sample{
                            evaluator.continuousDedx(energy),
                            evaluator.continuousRange(
                                energy, minimum_energy)})
                    .first;
      }
      return found->second;
    };

    auto const log_minimum = std::log(minimum_energy);
    auto const log_maximum =
        std::log(options.energy_max_MeV);
    for (std::size_t index = 0;
         index <= options.initial_intervals; ++index) {
      auto const fraction =
          static_cast<double>(index) /
          options.initial_intervals;
      auto const energy =
          index == 0
              ? minimum_energy
              : (index == options.initial_intervals
                     ? options.energy_max_MeV
                     : std::exp(
                           log_minimum +
                           fraction *
                               (log_maximum -
                                log_minimum)));
      sample(energy);
    }
    if (samples.begin()->first != minimum_energy) {
      samples.erase(samples.begin());
      sample(minimum_energy);
    }
    if (samples.rbegin()->first !=
        options.energy_max_MeV) {
      samples.erase(std::prev(samples.end()));
      sample(options.energy_max_MeV);
    }

    auto intervalError = [&](double lower_energy,
                             Sample const& lower,
                             double upper_energy,
                             Sample const& upper,
                             double fraction) {
      auto const log_lower = std::log(lower_energy);
      auto const log_upper = std::log(upper_energy);
      auto const energy = std::exp(
          log_lower + fraction * (log_upper - log_lower));
      auto const direct_dEdX =
          evaluator.continuousDedx(energy);
      auto const interpolated_dEdX = interpolate(
          lower_energy, upper_energy, lower.dEdX, upper.dEdX,
          energy);
      auto maximum =
          relativeError(direct_dEdX, interpolated_dEdX);

      auto const direct_range =
          evaluator.continuousRange(energy, minimum_energy);
      auto const interpolated_range =
          lower.range +
          fraction * (upper.range - lower.range);
      auto const range_scale =
          std::max(
              {std::abs(direct_range),
               std::abs(upper.range - lower.range), 1.e-12});
      maximum = std::max(
          maximum,
          std::abs(
              direct_range - interpolated_range) /
              range_scale);

      auto const target_range =
          lower.range +
          fraction * (upper.range - lower.range);
      auto const direct_inverse =
          evaluator.energyAfterContinuousGrammage(
              upper_energy, upper.range - target_range);
      auto const interpolated_inverse = std::exp(
          log_lower +
          fraction * (log_upper - log_lower));
      maximum = std::max(
          maximum,
          relativeError(
              direct_inverse, interpolated_inverse));
      return maximum;
    };

    constexpr std::array<double, 5> ProbeFractions{
        0.17320508075688773, 0.337, 0.5, 0.663,
        0.8267949192431123};
    auto const refinement_tolerance =
        0.6 * options.tolerance;
    std::size_t iteration = 0;
    while (true) {
      ++iteration;
      std::vector<double> insertions;
      double iteration_maximum = 0.;
      for (auto lower = samples.begin();
           lower != samples.end();) {
        auto const upper = std::next(lower);
        if (upper == samples.end()) {
          break;
        }
        for (auto const fraction : ProbeFractions) {
          auto const error = intervalError(
              lower->first, lower->second, upper->first,
              upper->second, fraction);
          iteration_maximum =
              std::max(iteration_maximum, error);
          if (error > refinement_tolerance) {
            insertions.push_back(std::exp(
                0.5 *
                (std::log(lower->first) +
                 std::log(upper->first))));
            break;
          }
        }
        lower = upper;
      }
      if (insertions.empty()) {
        break;
      }
      std::sort(insertions.begin(), insertions.end());
      insertions.erase(
          std::unique(insertions.begin(), insertions.end()),
          insertions.end());
      if (samples.size() + insertions.size() >
          options.max_points) {
        std::ostringstream message;
        message
            << evaluator.particleName()
            << " continuous grid needs more than --max-points="
            << options.max_points
            << "; current error is " << iteration_maximum;
        throw std::runtime_error(message.str());
      }
      for (auto const energy : insertions) {
        sample(energy);
      }
      std::cout
          << "  " << evaluator.particleName()
          << " continuous: refinement " << iteration
          << ", points=" << samples.size()
          << ", probe max error=" << std::scientific
          << iteration_maximum << '\n';
    }

    ContinuousEnergyTable table;
    table.pdg_id = evaluator.pdgId();
    table.particle_name = evaluator.particleName();
    table.reference_mode = "proposal_interpolated";
    table.mass_MeV = evaluator.particleMassMeV();
    table.minimum_total_energy_MeV = minimum_energy;
    table.energies_MeV.reserve(samples.size());
    table.dEdX_MeV_cm2_per_g.reserve(samples.size());
    table.range_g_per_cm2.reserve(samples.size());
    for (auto const& [energy, value] : samples) {
      table.energies_MeV.push_back(energy);
      table.dEdX_MeV_cm2_per_g.push_back(value.dEdX);
      table.range_g_per_cm2.push_back(value.range);
    }
    // Force the mathematical reference point to canonical +0.0.
    table.range_g_per_cm2.front() = 0.;

    constexpr std::array<double, 2> ValidationFractions{
        0.2718281828459045, 0.6180339887498948};
    double maximum_error = 0.;
    for (std::size_t index = 0;
         index + 1 < table.energies_MeV.size(); ++index) {
      auto const lower_energy = table.energies_MeV[index];
      auto const upper_energy =
          table.energies_MeV[index + 1];
      Sample const lower{
          table.dEdX_MeV_cm2_per_g[index],
          table.range_g_per_cm2[index]};
      Sample const upper{
          table.dEdX_MeV_cm2_per_g[index + 1],
          table.range_g_per_cm2[index + 1]};
      for (auto const fraction : ValidationFractions) {
        maximum_error = std::max(
            maximum_error,
            intervalError(
                lower_energy, lower, upper_energy, upper,
                fraction));
      }
    }
    table.measured_max_relative_error = maximum_error;
    if (maximum_error > options.tolerance) {
      std::ostringstream message;
      message
          << evaluator.particleName()
          << " continuous table failed independent validation: error="
          << maximum_error
          << ", tolerance=" << options.tolerance;
      throw std::runtime_error(message.str());
    }
    std::cout
        << "  " << evaluator.particleName()
        << " continuous: accepted points="
        << table.energies_MeV.size()
        << ", independent max error=" << maximum_error
        << ", E=[" << table.energies_MeV.front() << ", "
        << table.energies_MeV.back() << "] MeV\n";
    return table;
  }

  PhotonPairLpmMetadata photonPairLpmMetadata(
      PROPOSAL::Medium const& medium) {
    // This is the exact construction performed by PROPOSAL 7.6.2
    // PhotoPairLPM. eLpm_ has no public accessor, therefore tablegen freezes
    // the derived value into the versioned physics cache.
    PROPOSAL::crosssection::PhotoPairKochMotz parameterization;
    PROPOSAL::GammaDef const photon;
    constexpr double UpperEnergyMeV = 1.e14;
    PROPOSAL::Integral integral(
        PROPOSAL::IROMB, PROPOSAL::IMAXS, PROPOSAL::IPREC);

    double sum = 0.;
    auto const components = medium.GetComponents();
    for (auto const& component : components) {
      auto const limits = parameterization.GetKinematicLimits(
          photon, component, UpperEnergyMeV);
      auto const contribution = integral.Integrate(
          limits.v_min, limits.v_max,
          [&](double value) {
            return parameterization.DifferentialCrossSection(
                photon, component, UpperEnergyMeV, value);
          },
          2.);
      auto const weight_for_loss_in_medium =
          medium.GetSumNucleons() /
          (component.GetAtomInMolecule() *
           component.GetAtomicNum());
      sum += contribution / weight_for_loss_in_medium;
    }
    sum = 9. / 7. * sum * medium.GetMassDensity();
    auto e_lpm = PROPOSAL::ALPHA * PROPOSAL::ME;
    e_lpm *=
        2. * e_lpm /
        (PROPOSAL::PI * PROPOSAL::ME * PROPOSAL::RE * sum);

    PhotonPairLpmMetadata result;
    result.baseline_mass_density_g_per_cm3 =
        medium.GetMassDensity();
    result.molecular_density_per_cm3 =
        medium.GetMolDensity();
    result.sum_charge = medium.GetSumCharge();
    result.e_lpm_MeV = e_lpm;
    result.classical_electron_radius_cm = PROPOSAL::RE;
    result.fine_structure_constant = PROPOSAL::ALPHA;
    result.components.reserve(components.size());
    for (auto const& component : components) {
      result.components.push_back(
          {static_cast<std::uint64_t>(component.GetHash()),
           component.GetNucCharge(),
           component.GetLogConstant()});
    }
    return result;
  }

  double validatePhotonPairLpmMetadata(
      PROPOSAL::Medium const& medium,
      PhotonPairLpmMetadata const& metadata) {
    auto const snapshot = makePhotonPairLpmSnapshot(metadata);
    PROPOSAL::crosssection::PhotoPairKochMotz parametrization;
    PROPOSAL::crosssection::PhotoPairLPM reference(
        PROPOSAL::GammaDef(), medium, parametrization);
    double maximum_relative_error = 0.;
    for (auto const& component : medium.GetComponents()) {
      for (double energy :
           {1.e3, 1.e6, 1.e9, 1.e12, 1.e14}) {
        for (double x : {1.e-3, 1.e-2, 0.1, 0.5, 0.9}) {
          for (double density_ratio :
               {1.e-7, 1.e-4, 1.e-2, 1., 10.}) {
            auto const expected = reference.suppression_factor(
                energy, x, component, density_ratio);
            auto const actual = photonPairLpmSuppressionFactor(
                snapshot,
                static_cast<std::uint64_t>(
                    component.GetHash()),
                energy, x,
                medium.GetMassDensity() * density_ratio);
            if (actual.status != PhotonPairLpmStatus::Success) {
              throw std::runtime_error(
                  "generated photon-pair LPM metadata is not queryable");
            }
            auto const relative_error =
                std::abs(actual.survival_probability - expected) /
                std::max(1.e-300, std::abs(expected));
            maximum_relative_error =
                std::max(maximum_relative_error, relative_error);
            if (relative_error > 2.e-13) {
              throw std::runtime_error(
                  "generated photon-pair LPM metadata differs from "
                  "PROPOSAL");
            }
          }
        }
      }
    }
    return maximum_relative_error;
  }

  MoliereMetadata moliereMetadata(
      PROPOSAL::Medium const& medium) {
    MoliereMetadata result;
    result.enabled = true;
    result.reference_mode = "proposal_analytic";
    result.particle_mass_MeV = PROPOSAL::EMinusDef().mass;
    result.electron_mass_MeV = PROPOSAL::ME;
    result.fine_structure_constant = PROPOSAL::ALPHA;
    result.avogadro_per_mol = PROPOSAL::NA;
    result.hbar_MeV_s = PROPOSAL::HBAR;
    result.speed_of_light_cm_per_s = PROPOSAL::SPEED;
    result.euler_mascheroni =
        PROPOSAL::EULER_MASCHERONI;
    for (auto const& component : medium.GetComponents()) {
      result.components.push_back(
          {static_cast<std::uint64_t>(component.GetHash()),
           component.GetNucCharge(),
           component.GetAtomicNum(),
           component.GetAtomInMolecule()});
    }
    result.c1.assign(
        PROPOSAL::c1,
        PROPOSAL::c1 + MoliereSeriesCoefficientCount);
    result.c2.assign(
        PROPOSAL::c2,
        PROPOSAL::c2 + MoliereSeriesCoefficientCount);
    result.c2_large.assign(
        PROPOSAL::c2large,
        PROPOSAL::c2large +
            MoliereLargeSeriesCoefficientCount);
    result.s2_large.assign(
        PROPOSAL::s2large,
        PROPOSAL::s2large +
            MoliereLargeSeriesCoefficientCount);
    result.C1_large.assign(
        PROPOSAL::C1large,
        PROPOSAL::C1large +
            MoliereLargeIntegralCoefficientCount);
    return result;
  }

  double validateMoliereMetadata(
      PROPOSAL::Medium const& medium,
      MoliereMetadata const& metadata) {
    auto const snapshot = makeMoliereSnapshot(metadata);
    auto reference = PROPOSAL::make_multiple_scattering(
        PROPOSAL::MultipleScatteringType::Moliere,
        PROPOSAL::EMinusDef(), medium);
    double maximum_relative_error = 0.;
    for (double energy :
         {1.1, 10., 1.e3, 1.e6, 1.e9, 1.e12}) {
      for (double grammage :
           {1.e-8, 1.e-5, 1.e-2, 1., 100.}) {
        for (double uniform :
             {0.01, 0.1, 0.3, 0.7, 0.9, 0.99}) {
          auto const second = 1. - 0.73 * uniform;
          auto const expected =
              reference->CalculateScatteringAngle2D(
                  grammage, energy,
                  std::max(
                      PROPOSAL::EMinusDef().mass,
                      0.9 * energy),
                  uniform, second);
          auto const actual =
              sampleMoliereScatteringAngle2D(
                  snapshot, grammage, energy,
                  std::max(
                      PROPOSAL::EMinusDef().mass,
                      0.9 * energy),
                  uniform, second);
          if (expected == 0.) {
            if (actual.status !=
                    MoliereStatus::NoDeflection ||
                actual.angle_rad != 0.) {
              throw std::runtime_error(
                  "Moliere metadata changed a zero-deflection branch");
            }
            continue;
          }
          if (actual.status != MoliereStatus::Success) {
            throw std::runtime_error(
                "Moliere metadata validation did not converge");
          }
          maximum_relative_error = std::max(
              maximum_relative_error,
              std::abs(actual.angle_rad - expected) /
                  std::abs(expected));
        }
      }
    }
    return maximum_relative_error;
  }

  BremsLpmMetadata bremsLpmMetadata(
      PROPOSAL::Medium const& medium) {
    // Reproduce the private BremsLPM::eLpm_ construction used by
    // CORSIKA's LPM_calculator.  PROPOSAL 7.6.2 provides no accessor for
    // this numerical-integral result, so it is part of the versioned cache.
    PROPOSAL::crosssection::BremsElectronScreening parameterization;
    PROPOSAL::EMinusDef const electron;
    constexpr double UpperEnergyMeV = 1.e14;
    PROPOSAL::Integral integral(
        PROPOSAL::IROMB, PROPOSAL::IMAXS, PROPOSAL::IPREC);

    double sum = 0.;
    auto const components = medium.GetComponents();
    for (auto const& component : components) {
      auto const limits = parameterization.GetKinematicLimits(
          electron, component, UpperEnergyMeV);
      auto const contribution = integral.Integrate(
          limits.v_min, limits.v_max,
          [&](double value) {
            return parameterization.FunctionToDEdxIntegral(
                electron, component, UpperEnergyMeV, value);
          },
          2.);
      auto const weight_for_loss_in_medium =
          medium.GetSumNucleons() /
          (component.GetAtomInMolecule() *
           component.GetAtomicNum());
      sum += contribution / weight_for_loss_in_medium;
    }
    sum *= medium.GetMassDensity();
    auto e_lpm = PROPOSAL::ALPHA * electron.mass;
    e_lpm *=
        2. * e_lpm /
        (PROPOSAL::PI * PROPOSAL::ME * PROPOSAL::RE * sum);

    BremsLpmMetadata result;
    result.baseline_mass_density_g_per_cm3 =
        medium.GetMassDensity();
    result.molecular_density_per_cm3 =
        medium.GetMolDensity();
    result.sum_charge = medium.GetSumCharge();
    result.e_lpm_MeV = e_lpm;
    result.lepton_mass_MeV = electron.mass;
    result.electron_mass_MeV = PROPOSAL::ME;
    result.muon_mass_MeV = PROPOSAL::MMU;
    result.classical_electron_radius_cm = PROPOSAL::RE;
    result.fine_structure_constant = PROPOSAL::ALPHA;
    result.components.reserve(components.size());
    for (auto const& component : components) {
      result.components.push_back(
          {static_cast<std::uint64_t>(component.GetHash()),
           component.GetNucCharge(), component.GetAtomicNum(),
           component.GetLogConstant()});
    }
    return result;
  }

  double validateBremsLpmMetadata(
      PROPOSAL::Medium const& medium,
      BremsLpmMetadata const& metadata) {
    auto const snapshot = makeBremsLpmSnapshot(metadata);
    PROPOSAL::crosssection::BremsElectronScreening parametrization;
    PROPOSAL::crosssection::BremsLPM reference(
        PROPOSAL::EMinusDef(), medium, parametrization);
    double maximum_relative_error = 0.;
    for (auto const& component : medium.GetComponents()) {
      for (double energy :
           {1., 1.e3, 1.e6, 1.e9, 1.e12, 1.e14}) {
        for (double v :
             {1.e-6, 1.e-4, 1.e-2, 0.1, 0.5, 0.9}) {
          for (double density_ratio :
               {1.e-7, 1.e-4, 1.e-2, 1., 10.}) {
            auto const expected = reference.suppression_factor(
                energy, v, component, density_ratio);
            auto const actual = bremsLpmSuppressionFactor(
                snapshot,
                static_cast<std::uint64_t>(
                    component.GetHash()),
                energy, v,
                medium.GetMassDensity() * density_ratio);
            if (actual.status != BremsLpmStatus::Success) {
              throw std::runtime_error(
                  "generated bremsstrahlung LPM metadata is not queryable");
            }
            auto const relative_error =
                std::abs(actual.survival_probability - expected) /
                std::max(1.e-300, std::abs(expected));
            maximum_relative_error =
                std::max(maximum_relative_error, relative_error);
            if (relative_error > 2.e-13) {
              throw std::runtime_error(
                  "generated bremsstrahlung LPM metadata differs from "
                  "PROPOSAL");
            }
          }
        }
      }
    }
    return maximum_relative_error;
  }

  double validateEpairLpmMetadata(
      PROPOSAL::Medium const& medium,
      BremsLpmMetadata const& metadata) {
    auto const snapshot = makeBremsLpmSnapshot(metadata);
    PROPOSAL::crosssection::EpairLPM reference(
        PROPOSAL::EMinusDef(), medium);
    double maximum_relative_error = 0.;
    for (double energy :
         {10., 1.e3, 1.e6, 1.e9, 1.e12, 1.e14}) {
      for (double v :
           {1.e-4, 1.e-2, 0.1, 0.5, 0.9}) {
        for (double rho_squared :
             {0., 1.e-4, 0.01, 0.25, 0.81}) {
          for (double density_ratio :
               {1.e-7, 1.e-4, 1.e-2, 1., 10.}) {
            auto const beta =
                v * v / (2. * (1. - v));
            auto const xi =
                PROPOSAL::EMinusDef().mass *
                PROPOSAL::EMinusDef().mass /
                (PROPOSAL::ME * PROPOSAL::ME) *
                v * v / 4. * (1. - rho_squared) /
                (1. - v);
            auto expected = reference.suppression_factor(
                energy, v, rho_squared, beta, xi,
                density_ratio);
            // PROPOSAL's closed-form expression can lose a few ulps near
            // unity. Both implementations accept an open-interval random
            // number with probability one in this branch.
            if (expected > 1. - 1.e-6) {
              expected = 1.;
            }
            auto const actual = epairLpmSuppressionFactor(
                snapshot, energy, v, rho_squared,
                medium.GetMassDensity() * density_ratio);
            if (actual.status != EpairLpmStatus::Success) {
              throw std::runtime_error(
                  "generated Epair LPM metadata is not queryable");
            }
            auto const relative_error =
                std::abs(actual.survival_probability - expected) /
                std::max(1.e-300, std::abs(expected));
            maximum_relative_error =
                std::max(maximum_relative_error, relative_error);
            if (relative_error > 3.e-13) {
              throw std::runtime_error(
                  "generated Epair LPM metadata differs from "
                  "PROPOSAL");
            }
          }
        }
      }
    }
    return maximum_relative_error;
  }

  Options parseOptions(int argc, char** argv) {
    Options options;
    CLI::App app{
        "Generate versioned GPU electromagnetic dN/dX tables from the same "
        "PROPOSAL process configuration used by CORSIKA 8."};
    app.add_option("output", options.output, "Output .c8emrt table file")
        ->required();
    app.add_option(
        "--medium-yaml", options.medium_yaml,
        "Schema-v1 material definition; omitted for the legacy AirDry1Atm contract");
    app.add_option("--proposal-cache", options.proposal_cache,
                   "Directory used for PROPOSAL's own interpolation cache");
    app.add_option(
        "--epair-rho-source", options.epair_rho_source,
        "Reuse an existing v9/v10 rate table and only add the Epair rho table");
    app.add_option(
        "--merge-em-source", options.merge_em_source,
        "Reuse an already validated EM production table when constructing a "
        "combined EM+muon table");
    app.add_option(
        "--merge-muon-source", options.merge_muon_source,
        "Append an already validated muon-only table to --merge-em-source");
    app.add_option("--energy-min-MeV", options.energy_min_MeV,
                   "Minimum total particle energy in MeV");
    app.add_option("--energy-max-MeV", options.energy_max_MeV,
                   "Maximum total particle energy in MeV");
    app.add_option("--cut-MeV", options.energy_cut_MeV,
                   "Absolute stochastic energy cut in MeV");
    app.add_option(
        "--transport-cut-MeV", options.transport_cut_MeV,
        "Kinetic e-/e+ transport cut used as the zero-range reference");
    app.add_option(
        "--muon-transport-cut-MeV",
        options.muon_transport_cut_MeV,
        "Kinetic mu-/mu+ transport cut used as the zero-range reference");
    app.add_option(
        "--photon-pair-final-state-min-MeV",
        options.photon_pair_final_state_min_MeV,
        "Minimum photon energy for the normalized GPU pair final-state table");
    app.add_option("--tolerance", options.tolerance,
                   "Maximum relative dN/dX interpolation error");
    app.add_option("--loss-tolerance", options.loss_tolerance,
                   "Maximum relative inverse-CDF v(E,u) interpolation error");
    app.add_option("--initial-intervals", options.initial_intervals,
                   "Initial logarithmic energy intervals");
    app.add_option("--max-points", options.max_points,
                   "Maximum energy-grid points per particle");
    app.add_option("--loss-initial-energy-intervals",
                   options.loss_initial_energy_intervals,
                   "Initial log-energy intervals per inverse-CDF column");
    app.add_option("--loss-initial-quantile-intervals",
                   options.loss_initial_quantile_intervals,
                   "Initial quantile intervals per inverse-CDF column");
    app.add_option("--loss-max-energy-points",
                   options.loss_max_energy_points,
                   "Maximum energy points per inverse-CDF column");
    app.add_option(
        "--direct-loss-max-energy-points",
        options.direct_loss_max_energy_points,
        "Maximum energy points for a column rebuilt from PROPOSAL direct "
        "integration/root finding after an interpolated-reference reversal");
    app.add_option(
           "--nonmonotonic-loss-policy",
           options.nonmonotonic_loss_policy,
           "Repair policy: proposal-monotone or proposal-direct")
        ->check(CLI::IsMember(
            {std::string(ProposalMonotonePolicy),
             std::string(ProposalDirectPolicy)}));
    app.add_option("--loss-max-quantile-points",
                   options.loss_max_quantile_points,
                   "Maximum quantile points per inverse-CDF column");
    app.add_option("--loss-validation-samples",
                   options.loss_validation_samples,
                   "Independent direct-PROPOSAL samples per inverse-CDF column");
    app.add_option(
        "--epair-rho-min-energy-MeV",
        options.epair_rho_min_energy_MeV,
        "Minimum parent energy for the Epair rho inverse-CDF table");
    app.add_option(
        "--epair-rho-energy-points",
        options.epair_rho_energy_points,
        "Log-energy points in the dense Epair rho table");
    app.add_option(
        "--epair-rho-v-points",
        options.epair_rho_v_points,
        "Log threshold-excess points in the dense Epair rho table");
    app.add_option(
        "--epair-rho-quantile-points",
        options.epair_rho_quantile_points,
        "Logit rho-quantile points in the dense Epair rho table");
    app.add_option(
        "--epair-rho-validation-samples",
        options.epair_rho_validation_samples,
        "Independent PROPOSAL validation samples per target component");
    app.add_option(
        "--epair-rho-tolerance",
        options.epair_rho_tolerance,
        "Maximum absolute interpolation error in |rho|/rho_max");
    app.add_flag(
        "--enable-epair-rho-table",
        options.enable_epair_rho_table,
        "Experimentally generate the dense Epair rho inverse-CDF table");
    app.add_flag(
        "--include-muons", options.include_muons,
        "Also generate mu-/mu+ stochastic-rate, inverse-CDF and continuous-range "
        "tables for the experimental CUDA muon transport backend");
    app.add_flag(
        "--muons-only", options.muons_only,
        "Development mode: generate only mu-/mu+ tables for fast CUDA muon "
        "selection/transport validation");
    app.add_flag("--overwrite", options.overwrite,
                 "Replace an existing output table");
    try {
      app.parse(argc, argv);
    } catch (CLI::ParseError const& error) {
      throw CliExit{app.exit(error)};
    }

    if (!std::isfinite(options.energy_min_MeV) ||
        !std::isfinite(options.energy_max_MeV) ||
        !(options.energy_min_MeV > 0.) ||
        !(options.energy_max_MeV > options.energy_min_MeV) ||
        options.energy_max_MeV > MaximumGeneratedTableEnergyMeV) {
      throw std::invalid_argument("invalid energy range");
    }
    if (!std::isfinite(options.energy_cut_MeV) ||
        !(options.energy_cut_MeV > 0.)) {
      throw std::invalid_argument("--cut-MeV must be finite and positive");
    }
    if (options.energy_min_MeV > options.energy_cut_MeV) {
      throw std::invalid_argument(
          "--energy-min-MeV must not exceed --cut-MeV; otherwise photons "
          "above the tracking cut can fall outside the GPU rate table");
    }
    if (!std::isfinite(options.transport_cut_MeV) ||
        !(options.transport_cut_MeV > 0.)) {
      throw std::invalid_argument(
          "--transport-cut-MeV must be finite and positive");
    }
    if (!std::isfinite(
            options.muon_transport_cut_MeV) ||
        !(options.muon_transport_cut_MeV > 0.)) {
      throw std::invalid_argument(
          "--muon-transport-cut-MeV must be finite and positive");
    }
    if (!std::isfinite(
            options.photon_pair_final_state_min_MeV) ||
        !(options.photon_pair_final_state_min_MeV >
          PhotonPairThresholdMeV) ||
        !(options.photon_pair_final_state_min_MeV <
          options.energy_max_MeV)) {
      throw std::invalid_argument(
          "--photon-pair-final-state-min-MeV must be above the physical "
          "pair threshold and below --energy-max-MeV");
    }
    if (!std::isfinite(options.tolerance) || !(options.tolerance > 0.) ||
        !(options.tolerance <= 1.)) {
      throw std::invalid_argument("--tolerance must be in (0, 1]");
    }
    if (!std::isfinite(options.loss_tolerance) ||
        !(options.loss_tolerance > 0.) ||
        !(options.loss_tolerance <= 1.)) {
      throw std::invalid_argument("--loss-tolerance must be in (0, 1]");
    }
    if (options.initial_intervals < 1 || options.max_points < 2 ||
        options.initial_intervals + 1 > options.max_points) {
      throw std::invalid_argument("invalid initial interval or point limit");
    }
    if (options.loss_initial_energy_intervals < 1 ||
        options.loss_initial_quantile_intervals < 1 ||
        options.loss_max_energy_points < 2 ||
        options.direct_loss_max_energy_points < 2 ||
        options.loss_max_quantile_points < 2 ||
        options.loss_validation_samples < 1 ||
        options.loss_initial_energy_intervals + 1 >
            options.loss_max_energy_points ||
        options.loss_initial_energy_intervals + 1 >
            options.direct_loss_max_energy_points ||
        options.loss_initial_quantile_intervals + 1 >
            options.loss_max_quantile_points) {
      throw std::invalid_argument(
          "invalid inverse-CDF initial interval or point limit");
    }
    if ((options.enable_epair_rho_table ||
         !options.epair_rho_source.empty()) &&
        (!std::isfinite(
             options.epair_rho_min_energy_MeV) ||
         options.epair_rho_min_energy_MeV <
             EpairLossMinimumEnergyMeV ||
         !(options.epair_rho_min_energy_MeV <
           options.energy_max_MeV) ||
         options.epair_rho_energy_points < 2 ||
         options.epair_rho_v_points < 2 ||
         options.epair_rho_quantile_points < 2 ||
         options.epair_rho_validation_samples < 1 ||
         !std::isfinite(options.epair_rho_tolerance) ||
         !(options.epair_rho_tolerance > 0.) ||
         options.epair_rho_tolerance >
             options.loss_tolerance)) {
      throw std::invalid_argument(
          "invalid Epair rho inverse-CDF configuration");
    }
    if (std::filesystem::exists(options.output) && !options.overwrite) {
      throw std::runtime_error("output exists; pass --overwrite to replace it");
    }
    if (!options.epair_rho_source.empty() &&
        !std::filesystem::is_regular_file(
            options.epair_rho_source)) {
      throw std::invalid_argument(
          "--epair-rho-source is not a readable table file");
    }
    auto const merge_mode =
        !options.merge_em_source.empty() ||
        !options.merge_muon_source.empty();
    if (merge_mode &&
        (options.merge_em_source.empty() ||
         options.merge_muon_source.empty())) {
      throw std::invalid_argument(
          "--merge-em-source and --merge-muon-source must be supplied together");
    }
    if (merge_mode &&
        (!options.epair_rho_source.empty() ||
         options.enable_epair_rho_table ||
         options.include_muons || options.muons_only)) {
      throw std::invalid_argument(
          "validated-table merge mode cannot be combined with table generation flags");
    }
    for (auto const& input :
         {options.merge_em_source,
          options.merge_muon_source}) {
      if (!input.empty() &&
          !std::filesystem::is_regular_file(input)) {
        throw std::invalid_argument(
            "validated-table merge input is not a readable file");
      }
    }
    if (options.proposal_cache.empty()) {
      auto parent = options.output.parent_path();
      if (parent.empty()) {
        parent = std::filesystem::current_path();
      }
      options.proposal_cache = parent / "proposal_rate_cache";
    }
    return options;
  }

} // namespace

int main(int argc, char** argv) {
  try {
    auto const options = parseOptions(argc, argv);
    std::filesystem::create_directories(options.proposal_cache);
    PROPOSAL::InterpolationSettings::TABLES_PATH =
        options.proposal_cache.string();
    PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::critical);

    auto const medium_config =
        options.medium_yaml.empty()
            ? standardDryAirMediumConfig()
            : loadMediumConfig(options.medium_yaml);
    auto const medium_hash = mediumConfigHashHex(medium_config);
    auto const medium = makeProposalMedium(medium_config);
    auto const expected_components =
        makeRateTableMediumComponents(medium_config, medium);
    if (!options.merge_em_source.empty()) {
      auto output =
          readRateTable(options.merge_em_source);
      auto muons =
          readRateTable(options.merge_muon_source);
      auto const compatible_scalar = [](double left, double right) {
        auto const scale =
            std::max({1., std::abs(left), std::abs(right)});
        return std::abs(left - right) <=
               32. * std::numeric_limits<double>::epsilon() *
                   scale;
      };
      auto const& em_metadata = output.metadata;
      auto const& muon_metadata = muons.metadata;
      if (em_metadata.medium_name != medium.GetName() ||
          em_metadata.proposal_medium_hash !=
              static_cast<std::uint64_t>(medium.GetHash()) ||
          !sameMediumComponents(
              em_metadata.components, expected_components)) {
        throw std::runtime_error(
            "EM merge source does not match --medium-yaml");
      }
      if (em_metadata.proposal_version !=
              muon_metadata.proposal_version ||
          em_metadata.medium_name !=
              muon_metadata.medium_name ||
          em_metadata.proposal_medium_hash !=
              muon_metadata.proposal_medium_hash ||
          !compatible_scalar(
              em_metadata.energy_cut_MeV,
              muon_metadata.energy_cut_MeV) ||
          !compatible_scalar(
              em_metadata.relative_v_cut,
              muon_metadata.relative_v_cut) ||
          !compatible_scalar(
              em_metadata.energy_min_MeV,
              muon_metadata.energy_min_MeV) ||
          !compatible_scalar(
              em_metadata.energy_max_MeV,
              muon_metadata.energy_max_MeV) ||
          !sameMediumComponents(
              em_metadata.components,
              muon_metadata.components)) {
        throw std::runtime_error(
            "EM and muon tables have incompatible physical metadata");
      }
      if (std::any_of(
              output.particles.begin(),
              output.particles.end(),
              [](auto const& particle) {
                return isMuonPid(particle.pdg_id);
              })) {
        throw std::runtime_error(
            "EM merge source already contains a muon table");
      }
      if (muons.particles.size() != 2 ||
          muons.continuous_energy_tables.size() != 2 ||
          !std::all_of(
              muons.particles.begin(),
              muons.particles.end(),
              [](auto const& particle) {
                return isMuonPid(particle.pdg_id);
              }) ||
          !std::all_of(
              muons.continuous_energy_tables.begin(),
              muons.continuous_energy_tables.end(),
              [](auto const& table) {
                return isMuonPid(table.pdg_id);
              })) {
        throw std::runtime_error(
            "muon merge source must contain exactly mu-/mu+ rate and continuous tables");
      }
      output.particles.insert(
          output.particles.end(),
          std::make_move_iterator(muons.particles.begin()),
          std::make_move_iterator(muons.particles.end()));
      output.continuous_energy_tables.insert(
          output.continuous_energy_tables.end(),
          std::make_move_iterator(
              muons.continuous_energy_tables.begin()),
          std::make_move_iterator(
              muons.continuous_energy_tables.end()));
      output.metadata.generator_version =
          TableGeneratorContractVersion;
      output.metadata.measured_max_relative_error =
          std::max(
              output.metadata.measured_max_relative_error,
              muon_metadata.measured_max_relative_error);
      output.metadata.measured_max_loss_relative_error =
          std::max(
              output.metadata.measured_max_loss_relative_error,
              muon_metadata.measured_max_loss_relative_error);
      output.metadata.requested_relative_tolerance =
          std::max(
              output.metadata.requested_relative_tolerance,
              muon_metadata.requested_relative_tolerance);
      output.metadata.requested_loss_relative_tolerance =
          std::max(
              output.metadata.requested_loss_relative_tolerance,
              muon_metadata.requested_loss_relative_tolerance);
      auto const digest =
          writeRateTable(options.output, output);
      auto const verified =
          readRateTable(options.output);
      if (verified.content_hash != digest ||
          verified.particles.size() !=
              output.particles.size() ||
          verified.continuous_energy_tables.size() !=
              output.continuous_energy_tables.size()) {
        throw std::runtime_error(
            "combined EM+muon table failed read-back");
      }
      std::cout
          << "Merged validated EM table "
          << options.merge_em_source
          << " and muon table "
          << options.merge_muon_source << "\n"
          << "Wrote " << options.output << "\n"
          << "  SHA-256: " << toHex(digest) << '\n';
      return 0;
    }
    if (!options.epair_rho_source.empty()) {
      auto output =
          readRateTable(options.epair_rho_source);
      if (output.metadata.proposal_version !=
              getPROPOSALVersion() ||
          output.metadata.proposal_medium_hash !=
              static_cast<std::uint64_t>(
                  medium.GetHash()) ||
          output.metadata.medium_name != medium.GetName() ||
          !sameMediumComponents(
              output.metadata.components, expected_components)) {
        throw std::runtime_error(
            "Epair rho source table is physically incompatible");
      }
      if (options.epair_rho_tolerance >
          output.metadata
              .requested_loss_relative_tolerance) {
        throw std::runtime_error(
            "Epair rho tolerance exceeds source-table loss tolerance");
      }
      auto source_options = options;
      source_options.energy_min_MeV =
          output.metadata.energy_min_MeV;
      source_options.energy_max_MeV =
          output.metadata.energy_max_MeV;
      source_options.energy_cut_MeV =
          output.metadata.energy_cut_MeV;
      output.epair_rho = buildEpairRhoTable(
          findParticle(output, 11),
          makeBremsLpmSnapshot(
              output.metadata.brems_lpm),
          output.metadata.components, source_options);
      output.metadata.generator_version = TableGeneratorContractVersion;
      output.metadata.measured_max_loss_relative_error =
          std::max(
              output.metadata
                  .measured_max_loss_relative_error,
              output.epair_rho
                  .measured_max_normalized_error);
      auto const digest =
          writeRateTable(options.output, output);
      auto const verified =
          readRateTable(options.output);
      if (verified.content_hash != digest) {
        throw std::runtime_error(
            "Epair rho augmented table failed read-back");
      }
      std::cout
          << "Augmented " << options.epair_rho_source
          << " with the Epair rho inverse CDF\n"
          << "Wrote " << options.output << "\n"
          << "  SHA-256: " << toHex(digest) << '\n';
      return 0;
    }

    RateTableSet output;
    output.metadata.proposal_version = getPROPOSALVersion();
    output.metadata.generator_version = TableGeneratorContractVersion;
    output.metadata.medium_name = medium.GetName();
    output.metadata.proposal_medium_hash =
        static_cast<std::uint64_t>(medium.GetHash());
    output.metadata.energy_cut_MeV = options.energy_cut_MeV;
    static_assert(
        proposal::v_cut == ProposalRelativeVCut,
        "GPU table contract and CPU PROPOSAL v_cut must remain identical");
    output.metadata.relative_v_cut = ProposalRelativeVCut;
    output.metadata.energy_min_MeV = options.energy_min_MeV;
    output.metadata.energy_max_MeV = options.energy_max_MeV;
    output.metadata.requested_relative_tolerance = options.tolerance;
    output.metadata.requested_loss_relative_tolerance =
        options.loss_tolerance;
    output.metadata.components = expected_components;
    output.metadata.photon_pair_lpm =
        photonPairLpmMetadata(medium);
    output.metadata.brems_lpm = bremsLpmMetadata(medium);
    output.metadata.moliere = moliereMetadata(medium);
    auto const lpm_metadata_error =
        validatePhotonPairLpmMetadata(
            medium, output.metadata.photon_pair_lpm);
    auto const brems_lpm_metadata_error =
        validateBremsLpmMetadata(
            medium, output.metadata.brems_lpm);
    auto const epair_lpm_metadata_error =
        validateEpairLpmMetadata(
            medium, output.metadata.brems_lpm);
    auto const moliere_metadata_error =
        validateMoliereMetadata(
            medium, output.metadata.moliere);

    std::vector<ProposalReferenceEvaluator> evaluators;
    evaluators.reserve(
        options.muons_only
            ? 2
            : (options.include_muons ? 5 : 3));
    if (!options.muons_only) {
      evaluators.emplace_back(
          Code::Photon, 22, "gamma", medium,
          options.energy_cut_MeV);
      evaluators.emplace_back(
          Code::Electron, 11, "electron", medium,
          options.energy_cut_MeV);
      evaluators.emplace_back(
          Code::Positron, -11, "positron", medium,
          options.energy_cut_MeV);
    }
    if (options.include_muons ||
        options.muons_only) {
      evaluators.emplace_back(
          Code::MuMinus, 13, "muon_minus", medium,
          options.energy_cut_MeV);
      evaluators.emplace_back(
          Code::MuPlus, -13, "muon_plus", medium,
          options.energy_cut_MeV);
    }

    double maximum_error = 0.;
    double maximum_loss_error = 0.;
    std::cout << "Generating PROPOSAL " << output.metadata.proposal_version
              << " rate table\n"
              << "  medium: " << medium.GetName()
              << ", canonical SHA-256: " << medium_hash << "\n"
              << "  energy: [" << std::scientific << options.energy_min_MeV
              << ", " << options.energy_max_MeV << "] MeV\n"
              << "  cut: " << options.energy_cut_MeV
              << " MeV, rate tolerance: " << options.tolerance
              << ", loss tolerance: " << options.loss_tolerance
              << ", pair final-state minimum: "
              << options.photon_pair_final_state_min_MeV
              << " MeV"
              << "\n  experimental Epair rho table: "
              << (options.enable_epair_rho_table
                      ? "enabled"
                      : "disabled")
              << "\n  photon-pair LPM metadata max relative error: "
              << lpm_metadata_error
              << "\n  bremsstrahlung LPM metadata max relative error: "
              << brems_lpm_metadata_error
              << "\n  electron-pair LPM metadata max relative error: "
              << epair_lpm_metadata_error
              << "\n  Moliere metadata max relative error: "
              << moliere_metadata_error << '\n';
    for (auto& evaluator : evaluators) {
      auto evaluator_options = options;
      if (isMuonPid(evaluator.pdgId())) {
        evaluator_options.transport_cut_MeV =
            options.muon_transport_cut_MeV;
        // PROPOSAL tables use total energy. The global lower endpoint remains
        // at the photon cut, but a massive-particle rate grid must start above
        // rest mass and the configured kinetic transport cut.
        evaluator_options.energy_min_MeV =
            std::max(
                options.energy_min_MeV,
                evaluator.particleMassMeV() +
                    ContinuousCutSafetyFactor *
                        evaluator_options
                            .transport_cut_MeV);
      }
      auto result = buildAdaptiveTable(
          evaluator, evaluator_options);
      maximum_error = std::max(maximum_error, result.maximum_error);
      maximum_loss_error =
          std::max(maximum_loss_error, result.maximum_loss_error);
      auto const independent_loss_error =
          validateInverseCdfAgainstReference(evaluator, result.table,
                                             evaluator_options);
      maximum_loss_error =
          std::max(maximum_loss_error, independent_loss_error);
      if (evaluator.hasContinuousLoss()) {
        auto continuous =
            buildContinuousEnergyTable(
                evaluator, evaluator_options);
        maximum_error = std::max(
            maximum_error,
            continuous.measured_max_relative_error);
        output.continuous_energy_tables.push_back(
            std::move(continuous));
      }
      std::cout << "  " << result.table.particle_name
                << ": accepted points=" << result.table.energies_MeV.size()
                << ", columns=" << result.table.columns.size()
                << ", max rate error=" << result.maximum_error
                << ", max loss error=" << result.maximum_loss_error << '\n';
      output.particles.push_back(std::move(result.table));
    }
    if (options.enable_epair_rho_table) {
      output.epair_rho = buildEpairRhoTable(
          findParticle(output, 11),
          makeBremsLpmSnapshot(
              output.metadata.brems_lpm),
          output.metadata.components, options);
      maximum_loss_error = std::max(
          maximum_loss_error,
          output.epair_rho.measured_max_normalized_error);
    }
    output.metadata.measured_max_relative_error = maximum_error;
    output.metadata.measured_max_loss_relative_error =
        maximum_loss_error;
    if (maximum_error > options.tolerance) {
      throw std::runtime_error(
          "internal error: accepted table exceeds requested tolerance");
    }
    if (maximum_loss_error > options.loss_tolerance) {
      throw std::runtime_error(
          "internal error: accepted inverse CDF exceeds requested tolerance");
    }

    auto const digest = writeRateTable(options.output, output);
    auto const verified = readRateTable(options.output);
    if (verified.content_hash != digest) {
      throw std::runtime_error("written table failed immediate read-back validation");
    }
    validateCompatibility(
        verified,
        RateTableRequirements{
            getPROPOSALVersion(), medium.GetName(),
            static_cast<std::uint64_t>(medium.GetHash()),
            options.energy_cut_MeV,
            proposal::v_cut, options.energy_min_MeV, options.energy_max_MeV,
            options.tolerance, options.loss_tolerance,
            expected_components});
    std::cout << "Wrote " << options.output << "\n"
              << "  SHA-256: " << toHex(digest) << "\n"
              << "  measured max rate error: " << maximum_error << "\n"
              << "  measured max inverse-CDF error: "
              << maximum_loss_error << '\n';
    return 0;
  } catch (CliExit const& exit) {
    return exit.code;
  } catch (std::exception const& error) {
    std::cerr << "gpu_em_tablegen failed: " << error.what() << '\n';
    return 1;
  }
}
