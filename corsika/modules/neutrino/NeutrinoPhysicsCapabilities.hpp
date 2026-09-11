// Capability admission is separate from per-particle energy-domain admission.
#pragma once
#include <stdexcept>

namespace corsika::neutrino {
struct NeutrinoPhysicsRequirements {
  bool completeWeakTransport{false};
  bool eventCCPolarization{false};
  bool materialDepolarization{false};
};
inline void validateNeutrinoPhysicsRequirements(NeutrinoPhysicsRequirements const& r) {
  if (r.completeWeakTransport)
    throw std::invalid_argument("complete weak transport unavailable: CTW is limited to [1e4,1e12] GeV; low-energy/QE/RES and nuclear corrections are not supplied by the forced-vertex Pythia interface");
  if (r.eventCCPolarization)
    throw std::invalid_argument("event-level CC spin density unavailable: prescribed TAUOLA polarization is not a calculated CC production state");
  if (r.materialDepolarization)
    throw std::invalid_argument("tau material depolarization unavailable: PROPOSAL energy/momentum loss does not supply a spin-transfer map");
}
} // namespace corsika::neutrino
