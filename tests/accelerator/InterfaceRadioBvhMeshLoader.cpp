// Native geometry is compiled as C++, separately from the POD-only CUDA probe.
#include <corsika/geometry/terrain/FlatTerrainExport.hpp>
#include <corsika/geometry/terrain/TerrainBoundary.hpp>
corsika::terrain::FlatTerrainData loadRadioBvhProbeMesh(char const* path) {
  corsika::logging::set_level(corsika::logging::level::warn);
  auto mesh=corsika::terrain::loadValidatedTerrain(path,corsika::get_root_CoordinateSystem());
  return corsika::terrain::exportFlatTerrain(mesh);
}
