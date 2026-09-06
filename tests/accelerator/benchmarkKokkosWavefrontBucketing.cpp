/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

/*
 * Local performance diagnostic for the portable Kokkos wavefront ordering.
 *
 * This is deliberately not registered as a CTest test: it is a manual CUDA
 * benchmark used to compare Kokkos' comparison sort with the same stable
 * 50-bit CUB radix sort used by the native CUDA backend.
 */

#include <Kokkos_Core.hpp>

#if !defined(KOKKOS_ENABLE_CUDA)
#error "benchmarkKokkosWavefrontBucketing requires Kokkos CUDA"
#endif

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <corsika/accelerator/em/kokkos/KokkosWavefrontBucketing.hpp>

namespace {

  using ExecutionSpace = Kokkos::Cuda;
  using Memory = typename ExecutionSpace::memory_space;
  using Particle = corsika::gpu::em::EmParticleState;
  using ParticleSoA =
      corsika::accelerator::em::kokkos_detail::ParticleSoA<Memory>;
  namespace kd = corsika::accelerator::em::kokkos_detail;

  void checkCuda(cudaError_t const status, char const* const operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }

  std::vector<Particle> makeParticles(std::size_t const count) {
    std::mt19937_64 random(0xc8125a77U + count);
    std::uniform_int_distribution<int> pid(0, 2);
    std::uniform_int_distribution<int> medium(-2, 2);
    std::uniform_int_distribution<int> energy_bin(-128, 255);
    std::uniform_real_distribution<double> mantissa(1., 2.);
    std::vector<Particle> particles(count);
    for (std::size_t index = 0; index < count; ++index) {
      auto& particle = particles[index];
      auto const selected_pid = pid(random);
      particle.pid = selected_pid == 0
                         ? static_cast<std::int32_t>(
                               corsika::gpu::em::EmPid::Photon)
                         : (selected_pid == 1
                                ? static_cast<std::int32_t>(
                                      corsika::gpu::em::EmPid::Electron)
                                : static_cast<std::int32_t>(
                                      corsika::gpu::em::EmPid::Positron));
      particle.medium_id = medium(random);
      // Deliberately leave many equal 1/16-octave keys. This is representative
      // of one atmosphere wavefront and exercises the stable-order contract.
      particle.energy_GeV =
          std::exp2(static_cast<double>(energy_bin(random)) / 16.) *
          mantissa(random);
      particle.direction[2] = -1.;
      particle.weight = 1.;
      particle.history_id = index;
    }
    return particles;
  }

  ParticleSoA upload(std::vector<Particle> const& input,
                     ExecutionSpace const& execution) {
    ParticleSoA result(
        "bucketing_benchmark_input", input.size(), execution);
    Kokkos::View<Particle*, Memory> aos("bucketing_benchmark_aos",
                                        input.size());
    auto host = Kokkos::create_mirror_view(aos);
    for (std::size_t index = 0; index < input.size(); ++index)
      host(index) = input[index];
    Kokkos::deep_copy(execution, aos, host);
    Kokkos::parallel_for(
        "bucketing_benchmark_upload",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, input.size()),
        KOKKOS_LAMBDA(std::size_t const index) {
          result.store(index, aos(index));
        });
    execution.fence();
    return result;
  }

  class PortableComparisonWorkspace {
  public:
    using Order = kd::wavefront_bucketing_detail::BucketOrder;

    PortableComparisonWorkspace(std::size_t const capacity,
                                ExecutionSpace const& execution)
        : order_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                    "portable_comparison_order"),
                 capacity),
          keys_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   "portable_comparison_keys"),
                capacity),
          output_("portable_comparison_output", capacity, execution) {}

    void run(ParticleSoA const& input, std::size_t const count,
             ExecutionSpace const& execution) {
      auto const order = order_;
      Kokkos::parallel_for(
          "portable_comparison_make_order",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const source) {
            order(source) = {kd::kokkosWavefrontBucketKey(input.load(source)),
                             static_cast<std::uint32_t>(source), 0};
          });
      Kokkos::sort(
          execution, order,
          kd::wavefront_bucketing_detail::BucketOrderLess{});
      auto const keys = keys_;
      auto const output = output_;
      Kokkos::parallel_for(
          "portable_comparison_gather",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const destination) {
            auto const selected = order(destination);
            keys(destination) = selected.bucket_key;
            output.store(destination, input.load(selected.source));
          });
    }

    kd::KokkosWavefrontBucketBatch<ExecutionSpace> batch(
        std::size_t const count) const {
      return {output_, keys_, count, true};
    }

  private:
    Kokkos::View<Order*, Memory> order_;
    Kokkos::View<std::uint64_t*, Memory> keys_;
    ParticleSoA output_;
  };

  class CubWorkspace {
  public:
    CubWorkspace(std::size_t const capacity,
                 ExecutionSpace const& execution)
        : input_keys_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                         "cub_input_keys"),
                      capacity),
          output_keys_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                          "cub_output_keys"),
                       capacity),
          input_indices_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                            "cub_input_indices"),
                         capacity),
          output_indices_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "cub_output_indices"),
                          capacity),
          output_("cub_bucketed_output", capacity, execution) {
      std::size_t bytes{};
      checkCuda(cub::DeviceRadixSort::SortPairs(
                    nullptr, bytes,
                    static_cast<std::uint64_t const*>(nullptr),
                    static_cast<std::uint64_t*>(nullptr),
                    static_cast<std::uint32_t const*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr), capacity, 0,
                    corsika::gpu::em::detail::WavefrontBucketKeyBits),
                "query CUB temporary storage");
      temporary_ = Kokkos::View<unsigned char*, Memory>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "cub_temporary"),
          bytes);
      temporary_bytes_ = bytes;
    }

    void run(ParticleSoA const& input, std::size_t const count,
             ExecutionSpace const& execution) {
      auto const input_keys = input_keys_;
      auto const input_indices = input_indices_;
      Kokkos::parallel_for(
          "cub_make_keys",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const index) {
            input_keys(index) = kd::kokkosWavefrontBucketKey(input.load(index));
            input_indices(index) = static_cast<std::uint32_t>(index);
          });
      auto bytes = temporary_bytes_;
      checkCuda(cub::DeviceRadixSort::SortPairs(
                    temporary_.data(), bytes, input_keys_.data(),
                    output_keys_.data(), input_indices_.data(),
                    output_indices_.data(), count, 0,
                    corsika::gpu::em::detail::WavefrontBucketKeyBits,
                    execution.cuda_stream()),
                "CUB stable radix sort");
      auto const indices = output_indices_;
      auto const output = output_;
      Kokkos::parallel_for(
          "cub_gather_particles",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const index) {
            output.store(index, input.load(indices(index)));
          });
    }

    Kokkos::View<std::uint64_t*, Memory> const& keys() const {
      return output_keys_;
    }
    ParticleSoA const& output() const { return output_; }

  private:
    Kokkos::View<std::uint64_t*, Memory> input_keys_;
    Kokkos::View<std::uint64_t*, Memory> output_keys_;
    Kokkos::View<std::uint32_t*, Memory> input_indices_;
    Kokkos::View<std::uint32_t*, Memory> output_indices_;
    Kokkos::View<unsigned char*, Memory> temporary_;
    ParticleSoA output_;
    std::size_t temporary_bytes_{};
  };

  class ThrustStableWorkspace {
  public:
    ThrustStableWorkspace(std::size_t const capacity,
                          ExecutionSpace const& execution)
        : keys_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   "thrust_stable_keys"),
                capacity),
          indices_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                      "thrust_stable_indices"),
                   capacity),
          output_("thrust_stable_bucketed_output", capacity, execution) {}

    void run(ParticleSoA const& input, std::size_t const count,
             ExecutionSpace const& execution) {
      auto const keys = keys_;
      auto const indices = indices_;
      Kokkos::parallel_for(
          "thrust_stable_make_keys",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const index) {
            keys(index) = kd::kokkosWavefrontBucketKey(input.load(index));
            indices(index) = static_cast<std::uint32_t>(index);
          });
      auto policy = thrust::cuda::par.on(execution.cuda_stream());
      thrust::stable_sort_by_key(
          policy, thrust::device_pointer_cast(keys_.data()),
          thrust::device_pointer_cast(keys_.data() + count),
          thrust::device_pointer_cast(indices_.data()));
      auto const output = output_;
      Kokkos::parallel_for(
          "thrust_stable_gather_particles",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const index) {
            output.store(index, input.load(indices(index)));
          });
    }

    Kokkos::View<std::uint64_t*, Memory> const& keys() const { return keys_; }
    ParticleSoA const& output() const { return output_; }

  private:
    Kokkos::View<std::uint64_t*, Memory> keys_;
    Kokkos::View<std::uint32_t*, Memory> indices_;
    ParticleSoA output_;
  };

  template <class Function>
  double measureMilliseconds(Function&& function, ExecutionSpace const& execution,
                             int const warmup, int const repetitions) {
    for (int iteration = 0; iteration < warmup; ++iteration) function();
    execution.fence();
    cudaEvent_t start{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    checkCuda(cudaEventRecord(start, execution.cuda_stream()), "record start");
    for (int iteration = 0; iteration < repetitions; ++iteration) function();
    checkCuda(cudaEventRecord(stop, execution.cuda_stream()), "record stop");
    checkCuda(cudaEventSynchronize(stop), "synchronize stop");
    float elapsed{};
    checkCuda(cudaEventElapsedTime(&elapsed, start, stop), "measure events");
    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");
    return static_cast<double>(elapsed) / repetitions;
  }

  void verifyEqual(kd::KokkosWavefrontBucketBatch<ExecutionSpace> const& kokkos,
                   CubWorkspace const& cub, std::size_t const count,
                   ExecutionSpace const& execution) {
    Kokkos::View<int, Memory> mismatch("bucketing_mismatch");
    auto const kokkos_keys = kokkos.keys;
    auto const cub_keys = cub.keys();
    auto const kokkos_particles = kokkos.particles;
    auto const cub_particles = cub.output();
    Kokkos::parallel_for(
        "verify_bucket_order",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          if (kokkos_keys(index) != cub_keys(index) ||
              kokkos_particles.history_id(index) !=
                  cub_particles.history_id(index))
            Kokkos::atomic_store(&mismatch(), 1);
        });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mismatch);
    if (host() != 0)
      throw std::runtime_error("CUB and Kokkos stable order differ");
  }

  void verifyEqual(CubWorkspace const& cub,
                   ThrustStableWorkspace const& thrust,
                   std::size_t const count,
                   ExecutionSpace const& execution) {
    Kokkos::View<int, Memory> mismatch("thrust_bucketing_mismatch");
    auto const cub_keys = cub.keys();
    auto const thrust_keys = thrust.keys();
    auto const cub_particles = cub.output();
    auto const thrust_particles = thrust.output();
    Kokkos::parallel_for(
        "verify_thrust_bucket_order",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          if (cub_keys(index) != thrust_keys(index) ||
              cub_particles.history_id(index) !=
                  thrust_particles.history_id(index))
            Kokkos::atomic_store(&mismatch(), 1);
        });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mismatch);
    if (host() != 0)
      throw std::runtime_error("CUB and Thrust stable order differ");
  }

  int repetitionsFor(std::size_t const count) {
    if (count <= 1024) return 300;
    if (count <= 4096) return 150;
    if (count <= 16384) return 60;
    if (count <= 65536) return 25;
    return 10;
  }

} // namespace

int main(int argc, char** argv) {
  try {
    Kokkos::ScopeGuard guard(argc, argv);
    ExecutionSpace execution{};
    // Repeated RTX 4060 measurements found that the comparison path has lower
    // fixed cost for the tested fronts below 32768, while persistent CUB wins
    // from 32768 onward. Keep the selected-header column around both boundary
    // values so a different device can audit that conservative crossover.
    std::cout << "# selected hybrid crossover: portable for [256,32768), "
                 "CUB stable-50 for [32768,infinity)\n";
    std::cout << "count,portable_comparison_ms,selected_header_ms,"
                 "thrust_stable64_ms,cub_radix50_ms,portable_over_cub,"
                 "selected_over_cub,thrust_over_cub\n";
    for (auto const count : {std::size_t{256}, std::size_t{1024},
                             std::size_t{4096}, std::size_t{16384},
                             std::size_t{32767}, std::size_t{32768},
                             std::size_t{65536}, std::size_t{262144}}) {
      auto const particles = makeParticles(count);
      auto const input = upload(particles, execution);
      kd::KokkosWavefrontBucketingWorkspace<ExecutionSpace> kokkos_workspace;
      PortableComparisonWorkspace portable_workspace(count, execution);
      CubWorkspace cub_workspace(count, execution);
      ThrustStableWorkspace thrust_workspace(count, execution);
      auto const selected_function = [&] {
        (void)kd::bucketKokkosWavefront<ExecutionSpace>(
            input, count, kokkos_workspace, execution);
      };
      auto const portable_function = [&] {
        portable_workspace.run(input, count, execution);
      };
      auto const cub_function = [&] {
        cub_workspace.run(input, count, execution);
      };
      auto const thrust_function = [&] {
        thrust_workspace.run(input, count, execution);
      };
      selected_function();
      portable_function();
      cub_function();
      thrust_function();
      execution.fence();
      auto const selected_batch = kd::bucketKokkosWavefront<ExecutionSpace>(
          input, count, kokkos_workspace, execution);
      portable_function();
      cub_function();
      thrust_function();
      execution.fence();
      verifyEqual(selected_batch, cub_workspace, count, execution);
      verifyEqual(portable_workspace.batch(count), cub_workspace, count,
                  execution);
      verifyEqual(cub_workspace, thrust_workspace, count, execution);
      auto const repetitions = repetitionsFor(count);
      auto const portable_ms = measureMilliseconds(
          portable_function, execution, 3, repetitions);
      auto const selected_ms = measureMilliseconds(
          selected_function, execution, 3, repetitions);
      auto const thrust_ms = measureMilliseconds(
          thrust_function, execution, 3, repetitions);
      auto const cub_ms =
          measureMilliseconds(cub_function, execution, 3, repetitions);
      std::cout << count << ',' << std::fixed << std::setprecision(6)
                << portable_ms << ',' << selected_ms << ',' << thrust_ms << ','
                << cub_ms << ',' << portable_ms / cub_ms << ','
                << selected_ms / cub_ms << ',' << thrust_ms / cub_ms << '\n';
    }
  } catch (std::exception const& error) {
    std::cerr << "bucketing benchmark failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
