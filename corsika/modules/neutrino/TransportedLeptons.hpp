/* Mountain CC transport configuration, adapted from corsika8-mountain's
 * TerrainTauDiagnostics.hpp. This does not alter the global C7 particle set. */
#pragma once
#include <corsika/framework/core/ParticleProperties.hpp>
#include <set>

namespace corsika::neutrino {
inline std::set<Code> withTransportedTaus(std::set<Code> particles) {
  particles.insert(Code::TauMinus);
  particles.insert(Code::TauPlus);
  return particles;
}
} // namespace corsika::neutrino
