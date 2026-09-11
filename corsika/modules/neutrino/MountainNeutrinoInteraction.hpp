/*
 * Migrated from corsika8-mountain/cpp/NaturalNeutrino.hpp (2026-09-08).
 * This opt-in module does not alter stock neutrino or air-shower processes.
 *
 * Physical high-energy neutrino interaction sampling for the mountain tests.
 *
 * CORSIKA 8's Pythia NeutrinoInteraction intentionally exposes a fixed 4 nb
 * helper cross section for forceInteraction().  This wrapper keeps Pythia as
 * the selected CC/NC final-state generator but supplies the energy-dependent inclusive
 * neutrino-nucleon cross section in BOTH natural and forced-vertex modes.
 * A forced vertex changes only its position, not process or target weights.
 */

#pragma once

#include <corsika/modules/neutrino/AuditedPythiaNeutrino.hpp>
#include <corsika/modules/neutrino/NeutrinoModelDomain.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/process/InteractionProcess.hpp>

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

namespace corsika::neutrino {

using namespace corsika;
using namespace corsika::units::si;

// Connolly, Thorne & Waters, Phys. Rev. D 83, 113009 (2011), Eq. (7)
// and Table III.  The fit is valid for 10^4 <= E_nu/GeV <= 10^12 and
// reproduces their calculated nu-N cross sections to approximately 1%.
inline double ctw2011CrossSectionCm2(Code const projectile,
                                     double const energyGeV,
                                     bool const chargedCurrent = true) {
  if (!is_neutrino(projectile)) return 0.0;
  if (!inCtwEnergyDomain(energyGeV)) return 0.0;
  bool const anti = static_cast<int>(get_PDG(projectile)) < 0;
  double const epsilon = std::log10(energyGeV);
  double c0 = 0.0;
  double c1 = 0.0;
  double c2 = 0.0;
  double c3 = 0.0;
  double c4 = 0.0;
  if (!anti) {
    c0 = -1.826;
    c1 = -17.31;
    c2 = chargedCurrent ? -6.406 : -6.448;
    c3 = 1.431;
    c4 = chargedCurrent ? -17.91 : -18.61;
  } else {
    c0 = -1.033;
    c1 = -15.95;
    c2 = chargedCurrent ? -7.247 : -7.296;
    c3 = 1.569;
    c4 = chargedCurrent ? -17.72 : -18.30;
  }
  double const logArgument = std::log(epsilon - c0);
  double const log10Sigma = c1 + c2 * logArgument +
                            c3 * logArgument * logArgument + c4 / logArgument;
  return std::pow(10.0, log10Sigma);
}

inline double targetMassNumber(Code const target) {
  if (is_nucleus(target)) return static_cast<double>(get_nucleus_A(target));
  if (target == Code::Proton || target == Code::Neutron) return 1.0;
  return 0.0;
}

struct InteractionRecord {
  std::uint64_t historyId{0}, parentHistoryId{0};
  std::uint32_t generation{0};
  std::vector<std::tuple<std::uint64_t, int, double>> children;
  double ccCrossSectionCm2{0.}, ncCrossSectionCm2{0.};
  int projectilePdg{0};
  int targetPdg{0};
  double energyGeV{0.0};
  double crossSectionPerNucleonCm2{0.0};
  double targetCrossSectionCm2{0.0};
  double xM{0.0};
  double yM{0.0};
  double zM{0.0};
  double timeS{0.0};
  VertexAudit vertex;
};

/** Natural mode samples enabled CC/NC distances; forced mode is conditional.
 * Nuclear corrections and below-fit weak transport are not implemented.
 * The caller must retain neutrinos at material boundaries in natural mode and
 * account escaped particles separately from deposited energy.
 */
enum class SamplingMode { NaturalCC, ForcedVertexCC, ForcedVertexNC };
enum class InteractionChannels { CCOnly, NCOnly, CCAndNC };

// One uniform variate selects the competing channel AFTER distance/target
// selection using the SUM of the rates. Neither channel loses its daughters.
inline WeakCurrent selectWeakCurrent(double cc, double nc, double u) {
  if (!std::isfinite(cc) || !std::isfinite(nc) || cc < 0. || nc < 0. ||
      !(cc+nc > 0.) || !std::isfinite(cc+nc) || !(u >= 0. && u < 1.))
    throw std::invalid_argument("invalid neutrino channel rates or random variate");
  return u*(cc+nc) < cc ? WeakCurrent::CC : WeakCurrent::NC;
}

class MountainNeutrinoInteraction
    : public InteractionProcess<MountainNeutrinoInteraction> {
 public:
  MountainNeutrinoInteraction(std::set<Code> const& stableParticles,
                             SamplingMode const mode,
                             InteractionChannels const channels = InteractionChannels::CCOnly)
      : generator_(stableParticles),
        forcedVertex_(mode != SamplingMode::NaturalCC), mode_(mode), channels_(channels) {
    if ((mode == SamplingMode::ForcedVertexCC && channels == InteractionChannels::NCOnly) ||
        (mode == SamplingMode::ForcedVertexNC && channels == InteractionChannels::CCOnly))
      throw std::invalid_argument("forced current is disabled by neutrino channels");
  }

  /** Validate the primary before constructing calculators or creating output.
   * Secondary neutrinos below the CTW fit remain non-interacting in this
   * high-energy model. They must be transported to escape, not deposited.
   */
  static void validatePrimary(Code const projectile, double const energyGeV) {
    if (!is_neutrino(projectile)) {
      throw std::invalid_argument("mountain CC/NC primary must be a neutrino");
    }
    if (!inCtwEnergyDomain(energyGeV)) {
      throw std::invalid_argument(
          "mountain CC/NC primary energy must be within CTW2011 [1e4,1e12] GeV");
    }
  }

  /** Clear event-owned diagnostics when the process is reused for another shower. */
  void clearRecords() { std::vector<InteractionRecord>{}.swap(records_); }

  CrossSectionType getCrossSection(Code const projectile, Code const target,
                                   FourMomentum const& projectileP4,
                                   FourMomentum const& targetP4) const {
    if (!isValid(projectile, target, projectileP4, targetP4)) {
      return CrossSectionType::zero();
    }
    double const energyGeV = projectileP4.getTimeLikeComponent() / 1_GeV;
    auto const [cc,nc] = channelRates(projectile, energyGeV);
    double const sigmaN = cc+nc;
    double const a = targetMassNumber(target);
    if (!(sigmaN > 0.0 && a > 0.0)) return CrossSectionType::zero();
    // 1 nb = 1e-33 cm^2.  Scaling the per-nucleon cross section by A makes
    // lambda = <A>u/<A sigma_N> = u/sigma_N in an arbitrary composition.
    return (a * sigmaN / 1.e-33) * 1_nb;
  }

  template <typename TView>
  void doInteraction(TView& view, Code const projectile, Code const target,
                     FourMomentum const& projectileP4,
                     FourMomentum const& targetP4) {
    if (!(getCrossSection(projectile, target, projectileP4, targetP4) >
          CrossSectionType::zero())) {
      throw std::runtime_error("invalid projectile/target for audited Pythia CC/NC");
    }
    auto const primary = view.getProjectile();
    auto const coordinates = primary.getPosition().getCoordinates();
    double const energyGeV = projectileP4.getTimeLikeComponent()/1_GeV;
    auto const [cc,nc] = channelRates(projectile, energyGeV);
    WeakCurrent current = cc > 0. ? WeakCurrent::CC : WeakCurrent::NC;
    if (forcedVertex_ && records_.empty()) {
      current = mode_ == SamplingMode::ForcedVertexNC ? WeakCurrent::NC : WeakCurrent::CC;
    } else if (cc > 0. && nc > 0.) {
      std::uniform_real_distribution<double> uniform(0.,1.);
      current = selectWeakCurrent(cc,nc,
          uniform(RNGManager<>::getInstance().getRandomStream("pythia")));
    }
    auto generated = generator_.generate(projectile, target, projectileP4, current);
    InteractionRecord record;
    record.historyId = primary.getHistoryId();
    record.parentHistoryId = primary.getParentHistoryId();
    record.generation = primary.getGeneration();
    record.ccCrossSectionCm2 = cc;
    record.ncCrossSectionCm2 = nc;
    for (auto const& secondary : generated.secondaries) {
      if (!(secondary.momentum.getNorm() > 0_GeV)) {
        throw std::runtime_error("Pythia produced a zero-momentum final particle");
      }
      auto child = view.addSecondary(std::make_tuple(
          secondary.pid, secondary.kineticEnergy,
          secondary.momentum.normalized()));
      record.children.emplace_back(child.getHistoryId(),
          static_cast<int>(get_PDG(child.getPID())),child.getEnergy()/1_GeV);
    }
    record.projectilePdg = static_cast<int>(get_PDG(projectile));
    record.targetPdg = static_cast<int>(get_PDG(target));
    record.energyGeV = projectileP4.getTimeLikeComponent() / 1_GeV;
    record.crossSectionPerNucleonCm2 =
        current == WeakCurrent::CC ? cc : nc;
    record.targetCrossSectionCm2 =
        record.crossSectionPerNucleonCm2 * targetMassNumber(target);
    record.xM = coordinates.getX() / 1_m;
    record.yM = coordinates.getY() / 1_m;
    record.zM = coordinates.getZ() / 1_m;
    record.timeS = primary.getTime() / 1_s;
    record.vertex = std::move(generated.audit);
    if (records_.size() >= 10000) throw std::runtime_error("neutrino vertex audit limit exceeded");
    records_.push_back(std::move(record));
  }

  bool physicalCrossSection() const { return true; }
  bool forcedVertex() const { return forcedVertex_; }
  std::vector<InteractionRecord> const& records() const { return records_; }

  YAML::Node summary() const {
    YAML::Node output;
    output["schema_version"] = 4;
    output["module"] = "corsika.modules.neutrino.MountainNeutrinoInteraction";
    output["physics_scope"] = "selected_CC_NC_channels_free_stationary_nucleon_impulse_approximation";
    output["neutral_current_included"] = channels_ != InteractionChannels::CCOnly;
    output["charged_current_included"] = channels_ != InteractionChannels::NCOnly;
    output["regenerated_neutrinos_reenter_transport"] = true;
    output["regeneration_energy_domain_GeV"] = std::array<double,2>{1.e4,1.e12};
    output["full_neutrino_physics_validated"] = false;
    output["nuclear_shadowing_included"] = false;
    output["complete_tau_regeneration_included"] = false;
    output["earth_propagation_included"] = false;
    output["forced_vertex_is_conditional"] = forcedVertex_;
    output["event_rate_weight_provided"] = false;
    output["geometry_probability_weight_provided"] = false;
    output["forced_target_selection"] = "composition_times_A_times_enabled_CTW_rates_isoscalar";
    output["pythia_final_state_Q2_min_GeV2"] = 25.;
    output["low_Q2_final_states_validated"] = false;
    output["primary_energy_min_GeV"] = 1.e4;
    output["primary_energy_max_GeV"] = 1.e12;
    output["secondary_below_fit_treatment"] =
        "zero_CC_NC_rate_transport_to_escape_not_energy_deposition_MODEL_LIMITATION";
    output["sampling_mode"] = forcedVertex_
                                  ? "forced_vertex_physical_target_weights"
                                  : "natural_exponential_grammage";
    output["force_interaction_called"] = forcedVertex_;
    output["final_state_generator"] =
        "CORSIKA8_Pythia8_selected_CC_or_NC_project_local_audited";
    output["target_kinematics_convention"] =
        "impulse_approximation_free_stationary_p_or_n_sampled_from_Z_over_A";
    output["residual_nucleus_treatment"] =
        "not_present_in_inclusive_neutrino_nucleon_final_state";
    output["cross_section_model"] = "CTW2011_CC_NC_central_isoscalar_per_nucleon";
    output["cross_section_reference"] =
        "Connolly_Thorne_Waters_PRD83_113009_2011_Eq7_TableIII";
    output["interaction_count"] = records_.size();
    for (auto const& record : records_) {
      YAML::Node row;
      row["current"] = currentName(record.vertex.current);
      row["history_id"] = record.historyId;
      row["parent_history_id"] = record.parentHistoryId;
      row["generation"] = record.generation;
      row["cc_cross_section_cm2"] = record.ccCrossSectionCm2;
      row["nc_cross_section_cm2"] = record.ncCrossSectionCm2;
      row["total_cross_section_cm2"] = record.ccCrossSectionCm2+record.ncCrossSectionCm2;
      for (auto const& [id,pdg,energy] : record.children) {
        YAML::Node child;
        child["history_id"] = id; child["pdg"] = pdg; child["energy_GeV"] = energy;
        row["daughters"].push_back(child);
      }
      row["projectile_pdg"] = record.projectilePdg;
      row["target_pdg"] = record.targetPdg;
      row["energy_GeV"] = record.energyGeV;
      row["cross_section_per_nucleon_cm2"] =
          record.crossSectionPerNucleonCm2;
      row["target_cross_section_cm2"] = record.targetCrossSectionCm2;
      row["position_enu_m"].push_back(record.xM);
      row["position_enu_m"].push_back(record.yM);
      row["position_enu_m"].push_back(record.zM);
      row["time_s"] = record.timeS;
      auto const& vertex = record.vertex;
      YAML::Node audit;
      audit["projectile_pdg"] = vertex.projectilePdg;
      audit["target_nucleus_pdg"] = vertex.targetNucleusPdg;
      audit["selected_target_nucleon_pdg"] =
          vertex.selectedTargetNucleonPdg;
      audit["expected_outgoing_lepton_pdg"] =
          vertex.expectedOutgoingLeptonPdg;
      audit["selected_outgoing_lepton_pdg"] =
          vertex.selectedOutgoingLeptonPdg;
      audit["outgoing_lepton_candidate_count"] =
          vertex.outgoingLeptonCandidateCount;
      audit["hard_status23_outgoing_lepton_candidate_count"] =
          vertex.hardOutgoingLeptonCandidateCount;
      if (vertex.current == WeakCurrent::CC) {
        // Preserve legacy CC report consumers without calling an NC neutrino charged.
        audit["expected_outgoing_charged_lepton_pdg"] = vertex.expectedOutgoingLeptonPdg;
        audit["selected_outgoing_charged_lepton_pdg"] = vertex.selectedOutgoingLeptonPdg;
        audit["outgoing_charged_lepton_candidate_count"] = vertex.outgoingLeptonCandidateCount;
        audit["hard_status23_outgoing_charged_lepton_candidate_count"] = vertex.hardOutgoingLeptonCandidateCount;
      }
      audit["final_state_count"] = vertex.finalStateCount;
      audit["initial_charge_e"] = vertex.initialChargeE;
      audit["final_charge_e"] = vertex.finalChargeE;
      for (auto const& [pdg, count] : vertex.finalStatePdgCounts) {
        audit["final_state_pdg_counts"][pdg] = count;
      }
      auto addP4 = [&audit](char const* name,
                            std::array<double, 4> const& values) {
        for (double const value : values) {
          audit["four_momentum_GeV"][name].push_back(value);
        }
      };
      addP4("projectile", vertex.projectileP4GeV);
      addP4("selected_target_nucleon", vertex.selectedTargetP4GeV);
      addP4("initial", vertex.initialP4GeV);
      addP4("pythia_raw_final", vertex.pythiaRawFinalP4GeV);
      addP4("corsika_on_shell_final", vertex.corsikaOnShellFinalP4GeV);
      addP4("corsika_final_minus_initial",
            vertex.corsikaFinalMinusInitialP4GeV);
      addP4("outgoing_lepton", vertex.outgoingLeptonP4GeV);
      if (vertex.current == WeakCurrent::CC)
        addP4("outgoing_charged_lepton", vertex.outgoingLeptonP4GeV);
      addP4("hadronic_final_state", vertex.hadronicFinalP4GeV);
      addP4("projectile_com", vertex.projectileComP4GeV);
      addP4("selected_target_nucleon_com", vertex.selectedTargetComP4GeV);
      addP4("outgoing_lepton_com", vertex.outgoingLeptonComP4GeV);
      if (vertex.current == WeakCurrent::CC)
        addP4("outgoing_charged_lepton_com", vertex.outgoingLeptonComP4GeV);
      addP4("momentum_transfer_com", vertex.momentumTransferComP4GeV);
      audit["sqrt_s_GeV"] = vertex.sqrtSGeV;
      audit["target_mass_convention"]["selected_nucleon_mass_GeV"] =
          vertex.selectedTargetNucleonMassGeV;
      audit["target_mass_convention"]["legacy_nuclear_mass_over_A_GeV"] =
          vertex.legacyNuclearMassPerNucleonGeV;
      audit["target_mass_convention"]
           ["legacy_minus_selected_relative_difference"] =
          vertex.legacyVsSelectedTargetMassRelativeDifference;
      audit["conservation"]["pythia_com_energy_residual_GeV"] =
          vertex.pythiaComEnergyResidualGeV;
      audit["conservation"]["pythia_com_momentum_residual_GeV"] =
          vertex.pythiaComMomentumResidualGeV;
      audit["conservation"]
           ["pythia_com_max_component_relative_residual"] =
          vertex.pythiaComMaxComponentRelativeResidual;
      audit["conservation"]["corsika_energy_relative_residual"] =
          vertex.corsikaEnergyRelativeResidual;
      audit["conservation"]["corsika_momentum_relative_residual"] =
          vertex.corsikaMomentumRelativeResidual;
      audit["conservation"]
           ["corsika_max_component_relative_residual"] =
          vertex.corsikaMaxComponentRelativeResidual;
      audit["dis_kinematics"]["inelasticity_y"] = vertex.inelasticityY;
      audit["dis_kinematics"]["leptonic_lab_energy_y"] =
          vertex.leptonicLabEnergyY;
      audit["dis_kinematics"]["invariant_minus_lab_y"] =
          vertex.invariantMinusLabY;
      audit["dis_kinematics"]["hadronic_energy_transfer_y"] =
          vertex.hadronicEnergyTransferY;
      audit["dis_kinematics"]["y_energy_balance_residual"] =
          vertex.yEnergyBalanceResidual;
      audit["dis_kinematics"]["Q2_GeV2"] = vertex.q2GeV2;
      audit["dis_kinematics"]["bjorken_x"] = vertex.bjorkenX;
      audit["dis_kinematics"]["W2_GeV2"] =
          vertex.hadronicInvariantMass2GeV2;
      row["vertex_audit"] = audit;
      output["interactions"].push_back(row);
    }
    return output;
  }

 private:
  std::pair<double,double> channelRates(Code pid, double energyGeV) const {
    return {channels_ == InteractionChannels::NCOnly ? 0. : ctw2011CrossSectionCm2(pid,energyGeV,true),
            channels_ == InteractionChannels::CCOnly ? 0. : ctw2011CrossSectionCm2(pid,energyGeV,false)};
  }
  static bool isValid(Code const projectile, Code const target,
                      FourMomentum const& projectileP4,
                      FourMomentum const&) {
    if (!is_neutrino(projectile) ||
        !(is_nucleus(target) || target == Code::Proton ||
          target == Code::Neutron)) {
      return false;
    }
    // The generator uses a free stationary p/n.  Requiring even the smaller
    // proton-mass sqrt(s) to exceed the 25 GeV^2 DIS Q2 floor is conservative.
    auto const energy = projectileP4.getTimeLikeComponent();
    auto const mass = get_mass(Code::Proton);
    return (mass * mass + 2.0 * energy * mass) >= 25_GeV * 1_GeV;
  }

  AuditedPythiaNeutrinoFinalState generator_;
  bool forcedVertex_{false};
  SamplingMode mode_;
  InteractionChannels channels_;
  std::vector<InteractionRecord> records_;
};

}  // namespace corsika::neutrino
