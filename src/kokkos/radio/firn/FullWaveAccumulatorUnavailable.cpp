/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/radio/firn/IFullWaveAccumulator.hpp>
#include <stdexcept>
#include <corsika/kokkos/radio/firn/IGeometricalOpticsAccumulator.hpp>
namespace corsika::kokkos::radio::firn {
  std::shared_ptr<IFullWaveAccumulator> make_full_wave_accumulator(
      std::filesystem::path const&, std::string const&, std::vector<vector_type>,
      FullWaveDeviceConfig, int, int) {
    throw std::runtime_error("firn device radio requires a Kokkos-enabled build");
  }
  std::shared_ptr<IGeometricalOpticsAccumulator> make_geometrical_optics_accumulator(
      ExponentialOpticalParameters, std::vector<vector_type>, GeometricalOpticsConfig,
      int, int) {
    throw std::runtime_error("firn device GO requires a Kokkos-enabled build");
  }
} // namespace corsika::kokkos::radio::firn
