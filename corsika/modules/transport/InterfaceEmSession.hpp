/* Reusable two-sided shower transport API. Historical terrain names remain
 * source compatible. Implementation shares beta5 EM physics, not a new model. */
#pragma once
#include <corsika/modules/terrain/TerrainEmSession.hpp>
namespace corsika::interfaces {
using EmMaterial = terrain::TerrainEmMaterial;
using EmConfig = terrain::TerrainEmConfig;
using EmStep = terrain::TerrainEmStep;
using EmOutcome = terrain::TerrainEmOutcome;
using InterfaceEmSession = terrain::TerrainEmSession;
} // namespace corsika::interfaces
