/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentPhotonCascade.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentCapacity.hpp>

namespace {

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using PhotonWorkspace = corsika::accelerator::em::kokkos_detail::
      KokkosResidentPhotonWorkspace<ExecutionSpace>;
  using LeptonWorkspace = corsika::accelerator::em::kokkos_detail::
      KokkosResidentLeptonWorkspace<ExecutionSpace>;
  using Projection = corsika::accelerator::em::kokkos_detail::
      KokkosMemoryProjection;

  void requireProjection(Projection const& projection,
                         std::size_t const actual,
                         char const* const stage) {
    if (projection.retained_bytes != actual)
      throw std::runtime_error(
          std::string(stage) +
          ": projected retained bytes " +
          std::to_string(projection.retained_bytes) +
          " differ from actual storage " + std::to_string(actual));
    if (projection.transient_peak_bytes < projection.retained_bytes)
      throw std::runtime_error(
          std::string(stage) +
          ": projected transient peak is below retained storage");
  }

} // namespace

int main(int argc, char** argv) {
  try {
    Kokkos::ScopeGuard guard(argc, argv);
    ExecutionSpace execution{};

    PhotonWorkspace photons;
    if (photons.deviceBytes() != 0)
      throw std::runtime_error(
          "default photon workspace reports unallocated device storage");
    auto projection = photons.projectedCapacity(1024);
    photons.ensureCapacity(1024, execution);
    requireProjection(projection, photons.deviceBytes(),
                      "initial photon workspace");

    projection = photons.projectedCandidateCapacity(777);
    photons.ensureCandidateCapacity(777, execution);
    requireProjection(projection, photons.deviceBytes(),
                      "photon candidate workspace");

    projection = photons.projectedBucketingCapacity(1024, execution);
    photons.bucketing.ensureCapacity(1024, execution);
    requireProjection(projection, photons.deviceBytes(),
                      "photon bucketing workspace");

    projection = photons.projectedCapacity(1537);
    photons.ensureCapacity(1537, execution);
    requireProjection(projection, photons.deviceBytes(),
                      "grown photon workspace");

    auto const photon_bytes = photons.deviceBytes();
    projection = photons.projectedCapacity(64);
    requireProjection(projection, photon_bytes,
                      "reused photon workspace");
    if (projection.transient_peak_bytes != photon_bytes)
      throw std::runtime_error(
          "reused photon workspace predicted a transient allocation");

    LeptonWorkspace leptons;
    if (leptons.deviceBytes() != 0)
      throw std::runtime_error(
          "default lepton workspace reports unallocated device storage");
    constexpr std::size_t queue_limit = 4096;
    projection = leptons.projectedQueueCapacity(queue_limit);
    leptons.ensureQueueCapacity(queue_limit, execution);
    requireProjection(projection, leptons.deviceBytes(),
                      "initial lepton queue workspace");

    projection = leptons.projectedSourceCapacity(769, queue_limit);
    leptons.ensureSourceCapacity(769, queue_limit, execution);
    requireProjection(projection, leptons.deviceBytes(),
                      "lepton source workspace");

    constexpr std::size_t photon_limit = 8192;
    projection = leptons.projectedOutputCapacity(
        769, 513, 1025, 257, 129, 65, true, queue_limit,
        photon_limit);
    leptons.ensureOutputCapacity(
        769, 513, 1025, 257, 129, 65, true, queue_limit,
        photon_limit, execution);
    requireProjection(projection, leptons.deviceBytes(),
                      "lepton output workspace");

    projection = leptons.projectedBucketingCapacity(1024, execution);
    leptons.bucketing.ensureCapacity(1024, execution);
    requireProjection(projection, leptons.deviceBytes(),
                      "lepton bucketing workspace");

    auto const lepton_bytes = leptons.deviceBytes();
    projection = leptons.projectedSourceCapacity(128, queue_limit);
    requireProjection(projection, lepton_bytes,
                      "reused lepton source workspace");
    if (projection.transient_peak_bytes != lepton_bytes)
      throw std::runtime_error(
          "reused lepton source workspace predicted a transient allocation");

    // The automatic policy must select the largest fitting capacity, respond
    // to static/radio memory, and reject impossible/overflowing limits before
    // allocating anything. A synthetic monotone projection tests the search
    // independently of the device's currently available memory.
    namespace kd = corsika::accelerator::em::kokkos_detail;
    auto linear = [](std::size_t n) { return Projection{65536 + n * 4096,
                                                      131072 + n * 4096}; };
    auto plan = kd::selectResidentCapacity(16U << 20, 256, 128, linear);
    auto const next = linear(plan.input_particles + 1);
    auto const pending_bytes = 128 * sizeof(corsika::gpu::em::EmParticleState);
    if (plan.transient_peak_bytes > (16U << 20) ||
        next.transient_peak_bytes + 3 * pending_bytes +
            plan.allocator_reserve_bytes <= (16U << 20))
      throw std::runtime_error("automatic capacity is not the largest fitting front");
    auto const larger = kd::selectResidentCapacity(32U << 20, 256, 128, linear);
    auto const more_static = kd::selectResidentCapacity(
        16U << 20, 256, 128, [&](std::size_t n) {
          auto p = linear(n);
          p.retained_bytes += 4U << 20;
          p.transient_peak_bytes += 4U << 20;
          return p;
        });
    if (larger.input_particles <= plan.input_particles ||
        more_static.input_particles >= plan.input_particles)
      throw std::runtime_error("automatic capacity does not follow byte budget");
    bool rejected = false;
    try { (void)kd::selectResidentCapacity(1024, 256, 128, linear); }
    catch (std::runtime_error const&) { rejected = true; }
    if (!rejected) throw std::runtime_error("impossible byte budget accepted");
    rejected = false;
    try {
      (void)kd::selectResidentCapacity(
          16U << 20, std::numeric_limits<std::size_t>::max(), 0, linear);
    } catch (std::length_error const&) { rejected = true; }
    if (!rejected) throw std::runtime_error("overflowing particle count accepted");

    // Exercise both the small portable and CUDA radix sorting paths, with
    // pre-existing workspace allocations as well as empty owners. Include
    // projected profile records; production normally accumulates them directly.
    using Radio = corsika::accelerator::radio::kokkos_detail::
        KokkosRadioAccumulator<ExecutionSpace>;
    Radio radio;
    {
      PhotonWorkspace fresh_photons;
      LeptonWorkspace fresh_leptons;
      auto const actual_project = [&](std::size_t n) {
        return kd::projectResidentArenas(
            n, 0, fresh_photons, fresh_leptons, radio, false, execution);
      };
      auto const actual_plan = kd::selectResidentCapacity(
          64U << 20, 1024, 0, actual_project);
      if (actual_plan.transient_peak_bytes > (64U << 20) ||
          actual_project(actual_plan.input_particles + 1).transient_peak_bytes +
              actual_plan.allocator_reserve_bytes <= (64U << 20))
        throw std::runtime_error("actual View capacity search is not maximal");
      auto const fresh = kd::projectResidentArenas(
          1024, 0, fresh_photons, fresh_leptons, radio, false, execution);
      kd::reserveResidentArenas(
          1024, fresh_photons, fresh_leptons, radio, false, execution);
      requireProjection(fresh, fresh_photons.deviceBytes() +
                                  fresh_leptons.deviceBytes(),
                        "empty automatic resident arena pair");
    }
    for (auto const count : {std::size_t{2048}, std::size_t{32768}}) {
      auto total = photons.deviceBytes() + leptons.deviceBytes() + radio.deviceBytes();
      projection = kd::projectResidentArenas(
          count, total, photons, leptons, radio, true, execution);
      kd::reserveResidentArenas(count, photons, leptons, radio, true, execution);
      total = photons.deviceBytes() + leptons.deviceBytes() + radio.deviceBytes();
      requireProjection(projection, total, "automatic resident arena pair");
      for (auto const subcount : {std::size_t{64}, std::size_t{1024}, count}) {
        auto const reuse = kd::projectResidentArenas(
            subcount, total, photons, leptons, radio, true, execution);
        requireProjection(reuse, total, "reserved smaller wavefront");
        if (reuse.transient_peak_bytes != total)
          throw std::runtime_error("smaller front unexpectedly needs allocation");
      }
    }

    execution.fence("complete Kokkos resident-memory projection test");
  } catch (std::exception const& error) {
    std::cerr << "Kokkos resident-memory projection failed: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
