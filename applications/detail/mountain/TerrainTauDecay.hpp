// Read-only audit adapted from corsika8-mountain/cpp/TerrainTauDiagnostics.hpp.
// Stock TAUOLA/Pythia models own lifetime and all final-state draws.
#pragma once
#include <corsika/modules/neutrino/TransportedLeptonDecay.hpp>
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <yaml-cpp/yaml.h>
#include <optional>

namespace corsika::applications::terrain {
class TauDecay : public DecayProcess<TauDecay> {
 public:
  explicit TauDecay(CoordinateSystemPtr cs,
                    neutrino::TransportedLeptonDecayConfig config = {})
      : native_(config), cs_(cs), polarization_(config.prescribedTauolaPolarization
          ? config.prescribedTauolaPolarization : config.prescribedPythiaPolarization) {}
  template<class P> TimeType getLifetime(P const& p) { return native_.getLifetime(p); }
  template<class View> void doDecay(View& view) {
    auto const p = view.getProjectile();
    bool const tau = p.getPID() == Code::TauMinus || p.getPID() == Code::TauPlus;
    YAML::Node record;
    auto const energy = p.getEnergy();
    auto const momentum = p.getMomentum();
    if (tau) {
      if (records_.size() >= 10000) throw std::runtime_error("terrain tau audit record limit exceeded");
      record["parent_pdg"] = static_cast<int>(get_PDG(p.getPID()));
      record["history_id"] = p.getHistoryId();
      record["parent_history_id"] = p.getParentHistoryId();
      record["generation"] = p.getGeneration();
      if (polarization_) record["prescribed_longitudinal_polarization"] =
          p.getPID() == Code::TauMinus ? *polarization_ : -*polarization_;
      record["time_ns"] = p.getTime()/1_ns;
      record["energy_GeV"] = energy/1_GeV;
      record["weight"] = p.getWeight();
      auto const x = p.getPosition();
      record["position_enu_m"] = std::array<double,3>{x.getX(cs_)/1_m,x.getY(cs_)/1_m,x.getZ(cs_)/1_m};
      record["medium"] = dynamic_cast<corsika::terrain::TriangularMesh const*>(
          &p.getNode()->getVolume()) ? "rock" : "air";
      record["density_kg_m3"] = p.getNode()->getModelProperties().getMassDensity(x)/(1_kg/(1_m*1_m*1_m));
    }
    native_.doDecay(view);
    if (!tau) return;
    if (auto const& choice = native_.lastTauolaPolarization()) {
      record["applied_longitudinal_polarization_at_decay"] = choice->physicalPolarization;
      record["tauola_charge_conjugate_polarization"] = choice->chargeConjugatePolarization;
      record["sampled_tauola_helicity"] = choice->sampledHelicity;
      record["polarization_mixture_draw"] = choice->mixtureDraw;
    }
    HEPEnergyType finalEnergy = 0_GeV;
    MomentumVector finalMomentum(cs_, {0_GeV,0_GeV,0_GeV});
    for (auto const& child : view) {
      YAML::Node row;
      row["pdg"] = static_cast<int>(get_PDG(child.getPID()));
      row["energy_GeV"] = child.getEnergy()/1_GeV;
      row["history_id"] = child.getHistoryId();
      row["parent_history_id"] = child.getParentHistoryId();
      record["daughters"].push_back(row);
      finalEnergy += child.getEnergy();
      finalMomentum += child.getMomentum();
    }
    record["relative_energy_residual"] = (finalEnergy-energy)/energy;
    record["relative_momentum_residual"] = (finalMomentum-momentum).getNorm()/energy;
    records_.push_back(record);
  }
  YAML::Node summary() const {
    YAML::Node n;
    n["tau_mass_GeV"] = get_mass(Code::TauMinus)/1_GeV;
    n["tau_rest_lifetime_s"] = get_lifetime(Code::TauMinus)/1_s;
    n["tau_decay_count"] = records_.size(); n["decays"] = records_;
    n["native_decay_scheduling"] = true; n["force_decay_called"] = false;
    bool const tauola = native_.config().model == neutrino::TauDecayModel::Tauola;
    n["decay_model"] = tauola
        ? "stock CORSIKA TAUOLA for transported tau; Pythia8 for other decays"
        : "stock CORSIKA Pythia8; daughters recorded before thinning/cuts";
    n["tau_decay_backend"] = tauola ? "tauola" : "pythia";
    if (tauola && !polarization_) n["upstream_tauola_helicity"] =
        native_.config().helicity == tauola::Helicity::LeftHanded ? "left" :
        (native_.config().helicity == tauola::Helicity::RightHanded ? "right" : "unpolarized");
    if (tauola) {
      n["tauola_cpp_rng"] = "CORSIKA tauola stream";
      n["tauola_fortran_rng"] = "RANMAR seeded once from CORSIKA tauola stream";
      n["tauola_fortran_seed"] = neutrino::ConditionedTauolaDecay::fortranSeed();
      n["tauola_high_energy_boost"] = "E>=1000 GeV: conditioned generation frame + known-mass COMBoost";
      n["tauola_mass_convention"] = "scoped CORSIKA stable-particle masses v1; Fortran REAL*4 precision; original globals restored";
      n["tauola_effective_nutau_mass_GeV"] = get_mass(Code::NuTau)/1_GeV;
      n["tauola_decay_energy_renormalized"] = false;
    }
    n["spin_density_matrix_transferred_from_CC"] = false;
    n["prescribed_polarization_enabled"] = polarization_.has_value();
    if (polarization_) n["prescribed_tau_minus_polarization"] = *polarization_;
    n["polarization_model"] = tauola
        ? (polarization_ ? "TAUOLA_longitudinal_density_matrix_control_NOT_event_CC_spin"
                         : "original_CORSIKA_fixed_TAUOLA_helicity_NOT_event_CC_spin")
        : (polarization_ ? "explicit_longitudinal_decay_control_NOT_CC_spin_or_depolarization"
                         : "legacy_Pythia_unknown_production");
    n["energy_loss_depolarization_included"] = false;
    n["longitudinal_polarization_at_decay_supported"] = tauola;
    n["transverse_spin_transport_included"] = false;
    if (tauola && polarization_) {
      n["polarization_reference_frame"] = "tau rest frame; longitudinal axis = current lab momentum";
      n["polarization_sampling"] = "exact diagonal density mixture of unchanged TAUOLA helicities; endpoints/zero need no extra draw";
      n["spin_transport_assumption"] = "prescribed at decay; no CC-derived or material-evolved spin";
    }
    return n;
  }
 private:
  neutrino::TransportedLeptonDecay native_;
  CoordinateSystemPtr cs_;
  std::optional<double> polarization_;
  YAML::Node records_{YAML::NodeType::Sequence};
};
} // namespace corsika::applications::terrain
