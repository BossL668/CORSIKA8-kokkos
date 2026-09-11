/* Mountain lepton decay assembly; the generators and lifetime formulas are
 * the original CORSIKA modules, not reimplementations of their physics.
 */
#pragma once

#include <corsika/modules/neutrino/ConditionedTauolaDecay.hpp>
#include <corsika/modules/neutrino/LongitudinalTauolaDecay.hpp>
#include <corsika/modules/neutrino/PrescribedTauDecay.hpp>
#include <memory>
#include <optional>
#include <stdexcept>

namespace corsika::neutrino {

enum class TauDecayModel { Tauola, Pythia };

struct TransportedLeptonDecayConfig {
  // Matches the original c8_air_shower tau/Pythia process switch. This is
  // the upstream fixed-helicity convention, NOT a calculated CC spin state.
  TauDecayModel model{TauDecayModel::Tauola};
  tauola::Helicity helicity{tauola::Helicity::LeftHanded};
  std::optional<double> prescribedPythiaPolarization;
  std::optional<double> prescribedTauolaPolarization; // tau-minus P; tau-plus uses -P
};

class TransportedLeptonDecay : public DecayProcess<TransportedLeptonDecay> {
 public:
  explicit TransportedLeptonDecay(TransportedLeptonDecayConfig config = {})
      : config_(config) {
    if (config_.prescribedTauolaPolarization) {
      tauChargeConjugatePolarization(Code::TauMinus, *config_.prescribedTauolaPolarization);
      if (config_.model != TauDecayModel::Tauola || config_.prescribedPythiaPolarization)
        throw std::invalid_argument("TAUOLA polarization requires only the TAUOLA backend");
    }
    if (config_.model == TauDecayModel::Tauola) {
      if (config_.prescribedPythiaPolarization)
        throw std::invalid_argument("Pythia polarization control requires tau-decay-model pythia");
      // Register the tauola stream in the application BEFORE setSeed/init.
      // No new seed or reset is allowed when a daughter reaches this module.
      tauola_ = std::make_unique<ConditionedTauolaDecay>(config_.helicity);
    } else if (config_.prescribedPythiaPolarization) {
      polarized_ = std::make_unique<pythia8::Decay>();
      configurePrescribedTauDecay(*polarized_, *config_.prescribedPythiaPolarization);
    }
  }

  static bool isTau(Code pid) noexcept {
    return pid == Code::TauMinus || pid == Code::TauPlus;
  }

  template<class Particle> TimeType getLifetime(Particle const& p) {
    if (isTau(p.getPID()) && tauola_) return tauola_->getLifetime(p);
    return pythia_.getLifetime(p);
  }

  template<class View> void doDecay(View& view) {
    if (isTau(view.getProjectile().getPID())) {
      if (config_.prescribedTauolaPolarization) {
        double const p = *config_.prescribedTauolaPolarization;
        doDecayWithLongitudinalPolarization(view,
            view.getProjectile().getPID() == Code::TauMinus ? p : -p);
        return;
      }
      if (tauola_) { tauola_->doDecay(view); return; }
      if (polarized_) { polarized_->doDecay(view); return; }
    }
    // Non-tau decays retain the stock Pythia model. Do not impose the
    // standalone-tau polarization on taus inside a hadron's decay record.
    pythia_.doDecay(view);
  }

  TransportedLeptonDecayConfig const& config() const noexcept { return config_; }

  // Explicit per-particle hook: caller must supply P AT DECAY, in the tau
  // rest frame along its current lab momentum. It is not inferred from CC.
  template<class View> void doDecayWithLongitudinalPolarization(View& view, double p) {
    if (config_.model != TauDecayModel::Tauola)
      throw std::invalid_argument("per-particle longitudinal spin requires TAUOLA");
    longitudinal_.doDecay(view, p);
  }
  std::optional<LongitudinalTauChoice> const& lastTauolaPolarization() const {
    return longitudinal_.lastChoice();
  }

 private:
  TransportedLeptonDecayConfig config_;
  pythia8::Decay pythia_;
  std::unique_ptr<ConditionedTauolaDecay> tauola_;
  std::unique_ptr<pythia8::Decay> polarized_;
  LongitudinalTauolaDecay longitudinal_;
};
} // namespace corsika::neutrino
