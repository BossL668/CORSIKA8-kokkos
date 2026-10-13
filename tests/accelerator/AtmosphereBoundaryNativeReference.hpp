/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#pragma once
#include <corsika/accelerator/em/common/Types.hpp>
#include <vector>

// Compile the native units/vector implementation with the host C++ compiler,
// independently of the CUDA/HIP translation unit exercising the device code.
corsika::gpu::em::EnvironmentSnapshot makeAtmosphereBoundaryTestEnvironment();
void compareNativeAtmosphereBoundary(
    std::vector<corsika::gpu::em::EmParticleState> const& inputs,
    std::vector<corsika::gpu::em::PhotonTransportRecord> const& records,
    corsika::gpu::em::EnvironmentSnapshot const& environment);
