// Explicit longitudinal-polarization decay control. This is NOT a CC production
// density-matrix calculation and does NOT model energy-loss depolarization.
#pragma once
#include <corsika/modules/pythia8/Decay.hpp>
#include <cmath>
#include <stdexcept>

namespace corsika::neutrino {
enum class TauDecayChannels { All, PionNeutrinoControl };

inline void configurePrescribedTauDecay(corsika::pythia8::Decay& decay,
    double tauMinusPolarization, TauDecayChannels channels = TauDecayChannels::All) {
  if (!std::isfinite(tauMinusPolarization) || std::abs(tauMinusPolarization) > 1.)
    throw std::invalid_argument("tau-minus polarization must be within [-1,1]");
  // This helper is ONLY for CORSIKA's decay of a standalone tau AT REST.
  // PYTHIA 8.315 HelicityParticle::wave takes the P+pz==0 branch at p=0:
  // its helicity basis is aligned with -z, not the +z lab boost axis.
  // Thus physical P along the boost axis requires the NEGATIVE setting.
  // At nonzero +pz the basis is +z and this sign conversion is not applicable.
  // TauDecays::decay additionally reverses rho for tau+ automatically.
  // Its setting is cached by init(); changing settings alone has NO effect.
  decay.settings.mode("TauDecays:mode",3);
  decay.settings.parm("TauDecays:tauPolarization",-tauMinusPolarization);
  if (channels == TauDecayChannels::PionNeutrinoControl) {
    if (!decay.readString("15:onMode = off") ||
        !decay.readString("15:onIfMatch = 16 -211"))
      throw std::runtime_error("cannot select tau pion-neutrino control channel");
    // Keep the test pion stable; this switch is never used by the application.
    decay.particleData.mayDecay(211,false);
  }
  if (!static_cast<Pythia8::Pythia&>(decay).init())
    throw std::runtime_error("prescribed polarized tau decay initialization failed");
}
} // namespace corsika::neutrino
