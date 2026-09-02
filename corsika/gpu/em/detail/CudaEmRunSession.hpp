/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/tables/MediumConfig.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>
#include <corsika/modules/proposal/NativeCalculatorView.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace corsika::gpu::em::detail {

  /** Physics-domain requirements shared by every CORSIKA CUDA application. */
  struct GpuPhysicsRequirements {
    double maximum_primary_energy_MeV{};
    double em_transport_cut_MeV{};
    double muon_transport_cut_MeV{};
    double stochastic_cut_MeV{};
  };

  struct CudaSessionBeginResult {
    CudaEmBackend* backend{};
    bool reused{};
    bool muon_transport_available{};
  };

  /**
   * Owns the immutable physics package and CUDA backend for one simulation
   * library.  The environment and physics source are fixed by the first
   * shower; later calls only reset shower-local state through beginShower().
   *
   * This class deliberately has no dependency on CLI11, YAML, an atmosphere
   * model, radio detector types, or application output policy.
   */
  class CudaEmRunSession {
  public:
    CudaEmRunSession(GpuPhysicsSource source,
                     std::filesystem::path c8emrt_cache,
                     std::filesystem::path auxiliary_cache);

    CudaSessionBeginResult beginC8EmRt(
        EnvironmentSnapshot const&, GpuEmConfig const&,
        GpuPhysicsRequirements const&);

    CudaSessionBeginResult beginProposalNative(
        EnvironmentSnapshot const&, GpuEmConfig const&,
        GpuPhysicsRequirements const&,
        std::vector<proposal::NativeInteractionCalculatorView> const&,
        std::vector<proposal::NativeContinuousCalculatorView> const&);

    bool initialized() const noexcept { return backend_ != nullptr; }
    CudaEmBackend& backend();

    tables::RateTableSet const* loadedRateTable() const noexcept {
      return loaded_rate_table_ ? &*loaded_rate_table_ : nullptr;
    }

  private:
    void requireSource(GpuPhysicsSource) const;

    GpuPhysicsSource source_{};
    std::filesystem::path auxiliary_cache_;
    std::optional<tables::RateTableSet> loaded_rate_table_;
    std::unique_ptr<CudaEmBackend> backend_;
  };

} // namespace corsika::gpu::em::detail
