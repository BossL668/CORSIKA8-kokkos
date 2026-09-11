// Opt-in longitudinal density matrix -> unchanged CORSIKA/TAUOLA decays.
// Not a CC production-polarization or material-depolarization calculation.
#pragma once
#include <corsika/modules/neutrino/ConditionedTauolaDecay.hpp>
#include <cmath>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>

namespace corsika::neutrino {

struct LongitudinalTauChoice {
  double physicalPolarization{}; // actual tau's rest-frame spin along lab momentum
  double chargeConjugatePolarization{}; // P(tau-); the opposite of P(tau+)
  double sampledHelicity{}; // upstream charge-conjugate TAUOLA convention
  bool mixtureDraw{};
};

inline double tauChargeConjugatePolarization(Code pid, double physicalP) {
  if (pid != Code::TauMinus && pid != Code::TauPlus)
    throw std::invalid_argument("longitudinal tau spin requires tau-/tau+");
  if (!std::isfinite(physicalP) || std::abs(physicalP) > 1.)
    throw std::invalid_argument("longitudinal tau polarization must be finite in [-1,1]");
  return pid == Code::TauMinus ? physicalP : -physicalP;
}

class LongitudinalTauolaDecay {
 public:
  template<class View> void doDecay(View& view, double physicalP) {
    auto const pid = view.getProjectile().getPID();
    double const h = tauChargeConjugatePolarization(pid, physicalP);
    // A longitudinal spin-1/2 density matrix is EXACTLY a probabilistic
    // mixture of its two helicities. No reweighting, daughter retry, new
    // decay formula, global TAUOLA callback mutation, or RNG reseeding.
    // Initialize before drawing, just as the original adapter does.
    if (!left_) {
      left_ = std::make_unique<ConditionedTauolaDecay>(tauola::Helicity::LeftHanded);
      right_ = std::make_unique<ConditionedTauolaDecay>(tauola::Helicity::RightHanded);
      unpolarized_ = std::make_unique<ConditionedTauolaDecay>(tauola::Helicity::Unpolarized);
    }
    last_ = LongitudinalTauChoice{physicalP, h, h, false};
    if (h == 0.) { unpolarized_->doDecay(view); return; }
    bool right = h == 1.;
    if (h != -1. && h != 1.) {
      right = std::uniform_real_distribution<double>(0., 1.)(
          RNGManager<>::getInstance().getRandomStream("tauola")) < (1.+h)/2.;
      last_->mixtureDraw = true;
    }
    last_->sampledHelicity = right ? 1. : -1.;
    (right ? right_ : left_)->doDecay(view);
  }
  std::optional<LongitudinalTauChoice> const& lastChoice() const { return last_; }
 private:
  std::unique_ptr<ConditionedTauolaDecay> left_, right_, unpolarized_;
  std::optional<LongitudinalTauChoice> last_;
};
} // namespace corsika::neutrino
