// Legacy source-compatible names; all execution/physics ownership is in transport/.
#pragma once
#include <corsika/modules/transport/InterfaceEmSession.hpp>
namespace corsika::terrain {
namespace em = gpu::em;
using TerrainEmMaterial = interfaces::EmMaterial;
using TerrainEmConfig = interfaces::EmConfig;
using TerrainEmOutcome = interfaces::EmOutcome;
using TerrainEmStep = interfaces::EmStep;
using TerrainEmSession = interfaces::InterfaceEmSession;
} // namespace corsika::terrain
