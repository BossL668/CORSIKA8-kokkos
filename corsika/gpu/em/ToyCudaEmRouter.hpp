/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/RouterParticleConversion.hpp>

namespace corsika::gpu::em {

  struct ToyCudaEmRouterStatistics {
    std::uint64_t particles_staged{};
    std::uint64_t particles_returned_to_cpu{};
    std::uint64_t wavefronts{};
    std::size_t maximum_input_batch{};
  };

  /**
   * Development adapter connecting HybridCascade to the toy CUDA backend.
   *
   * Every GPU output is deliberately returned to the CPU stack after one wavefront. This
   * is inefficient and is not the production scheduler; it exists to validate unit
   * conversion, identity round-trips and CPU/GPU ownership transfer before real EM
   * physics is introduced.
   */
  template <typename TStack>
  class ToyCudaEmRouter {
  public:
    using particle_type = typename TStack::particle_type;

    ToyCudaEmRouter(CudaEmBackend& backend,
                    CoordinateSystemPtr coordinate_system)
        : backend_(backend)
        , coordinate_system_(std::move(coordinate_system)) {
      if (!coordinate_system_) {
        throw std::invalid_argument(
            "Toy CUDA EM router requires a coordinate system");
      }
    }

    bool canRoute(
        particle_type const& particle,
        transport::StepId) const {
      return is_em(particle.getPID());
    }

    void stage(particle_type const& particle, transport::HistoryId history_id,
               transport::HistoryId parent_history_id,
               transport::Generation generation, transport::StepId step_id) {
      auto state = router_detail::toDeviceState(
          particle, coordinate_system_, history_id,
          parent_history_id, generation, step_id);
      if (!backend_.canTransport(state)) {
        throw std::logic_error(
            "Toy CUDA EM router selected a particle unsupported by the backend");
      }
      backend_.enqueue(state);
      ++statistics_.particles_staged;
    }

    bool pending() const { return !backend_.empty(); }

    std::size_t advanceOneWavefrontAndReturn(TStack& stack) {
      auto const result = backend_.advanceWavefront();
      auto particles = backend_.extractActiveParticlesForTesting();
      if (particles.size() != result.output_particles) {
        throw std::runtime_error(
            "GPU wavefront count differs from downloaded particle count");
      }

      for (auto const& particle : particles) {
        router_detail::importParticle(
            stack, particle, coordinate_system_);
      }
      ++statistics_.wavefronts;
      statistics_.particles_returned_to_cpu += particles.size();
      statistics_.maximum_input_batch =
          std::max(statistics_.maximum_input_batch, result.input_particles);
      return particles.size();
    }

    ToyCudaEmRouterStatistics const& statistics() const {
      return statistics_;
    }

  private:
    CudaEmBackend& backend_;
    CoordinateSystemPtr coordinate_system_;
    ToyCudaEmRouterStatistics statistics_{};
  };

} // namespace corsika::gpu::em
