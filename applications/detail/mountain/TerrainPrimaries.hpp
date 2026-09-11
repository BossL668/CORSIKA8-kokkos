#pragma once
#include <corsika/framework/core/ParticleProperties.hpp>
#include <map>
#include <string>
namespace corsika::applications::terrain {
// Explicit admission, not an arbitrary PDG escape hatch. All six neutrinos
// share the existing CC-only fit; charged mu/tau controls use the CPU models.
inline std::map<std::string, Code> const& primaries() {
  static std::map<std::string, Code> const choices{
      {"photon",Code::Photon},{"electron",Code::Electron},{"positron",Code::Positron},
      {"mu_minus",Code::MuMinus},{"mu_plus",Code::MuPlus},
      {"tau_minus",Code::TauMinus},{"tau_plus",Code::TauPlus},
      {"nu_e",Code::NuE},{"anti_nu_e",Code::NuEBar},
      {"nu_mu",Code::NuMu},{"anti_nu_mu",Code::NuMuBar},
      {"nu_tau",Code::NuTau},{"anti_nu_tau",Code::NuTauBar},
      {"proton",Code::Proton},{"anti_proton",Code::AntiProton},
      {"neutron",Code::Neutron},{"anti_neutron",Code::AntiNeutron},
      {"pi_plus",Code::PiPlus},{"pi_minus",Code::PiMinus}};
  return choices;
}
} // namespace corsika::applications::terrain
