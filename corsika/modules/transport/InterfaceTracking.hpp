// Shape/medium-independent scalar entry for one oriented closed mesh inside a
// stock environment. The tracker queries each logical node's own magnetic field.
// Old terrain names remain compatible; no second integration formula is added.
#pragma once
#include <corsika/modules/terrain/TerrainMagneticTracking.hpp>
namespace corsika::interfaces {
using MeshInterfaceTracking = terrain::MagneticTracking;
using StraightMeshInterfaceTracking = terrain::StraightTrackingView;
} // namespace corsika::interfaces
