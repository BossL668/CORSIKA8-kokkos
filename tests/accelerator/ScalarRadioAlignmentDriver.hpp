/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/radio/common/Types.hpp>
#include <corsika/accelerator/radio/detail/RadioProjectionData.hpp>

#include <array>
#include <vector>

namespace scalar_radio_test {
  // Keep the actual scalar CORSIKA classes in a normal C++ translation unit;
  // only this POD interface and the Kokkos accumulator are compiled as CUDA.
  void initialize(int& argc, char**& argv);
  void finalize();
  std::array<double, 3> doppler();
  std::vector<double> observerWindow(
      std::vector<double> const& times,
      corsika::accelerator::radio::detail::DeviceObserver observer);
  corsika::gpu::radio::GpuRadioWaveforms project(
      corsika::gpu::radio::GpuRadioConfig const& config,
      corsika::gpu::em::LeptonTransportRecord const& record, bool tiled);
} // namespace scalar_radio_test
