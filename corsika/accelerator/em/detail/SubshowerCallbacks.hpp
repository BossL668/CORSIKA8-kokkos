/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#pragma once
#include <corsika/accelerator/em/common/Types.hpp>
#include <functional>

namespace corsika::accelerator::em::detail {
// Borrowed only for a host progress call, never captured by an endpoint job.
// The coordinator owns history allocation, scalar fallback and all writers.
struct SubshowerCallbacks {
  std::function<std::uint64_t(std::uint64_t)> reserve_histories;
  std::function<void(gpu::em::ResidentPhotonCascadeResult&&, double)> photons;
  std::function<void(gpu::em::ResidentLeptonCascadeResult&&, double)> leptons;
  std::function<bool()> yield_to_scalar;
};
} // namespace corsika::accelerator::em::detail
