/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <corsika/accelerator/em/kokkos/KokkosWavefrontBucketing.hpp>

namespace {

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using Memory = typename ExecutionSpace::memory_space;
  using Particle = corsika::gpu::em::EmParticleState;
  using ParticleSoA =
      corsika::accelerator::em::kokkos_detail::ParticleSoA<Memory>;

  Particle particle(std::int32_t const pid, std::int32_t const medium,
                    double const energy, std::uint64_t const history) {
    Particle result{};
    result.pid = pid;
    result.medium_id = medium;
    result.energy_GeV = energy;
    result.direction[2] = -1.;
    result.weight = 1.;
    result.history_id = history;
    return result;
  }

  std::uint64_t nativeHostKeyOracle(Particle const& value) {
    using corsika::gpu::em::EmPid;
    std::uint64_t pid_bucket = 3;
    if (value.pid == static_cast<std::int32_t>(EmPid::Photon))
      pid_bucket = 0;
    else if (value.pid == static_cast<std::int32_t>(EmPid::Electron))
      pid_bucket = 1;
    else if (value.pid == static_cast<std::int32_t>(EmPid::Positron))
      pid_bucket = 2;

    constexpr auto invalid = std::numeric_limits<std::uint16_t>::max();
    constexpr auto maximum_finite =
        static_cast<std::uint16_t>(invalid - 1);
    std::uint16_t energy_bucket = invalid;
    if (value.energy_GeV > 0. && std::isfinite(value.energy_GeV)) {
      auto const coordinate =
          std::floor(std::log2(value.energy_GeV) * 16. + 32768.);
      energy_bucket =
          !(coordinate > 0.)
              ? 0
              : (coordinate >= static_cast<double>(maximum_finite)
                     ? maximum_finite
                     : static_cast<std::uint16_t>(coordinate));
    }
    auto const ordered_medium =
        static_cast<std::uint32_t>(value.medium_id) ^ 0x80000000U;
    return (pid_bucket << 48U) |
           (static_cast<std::uint64_t>(ordered_medium) << 16U) |
           energy_bucket;
  }

  ParticleSoA upload(std::vector<Particle> const& input,
                     ExecutionSpace const& execution) {
    ParticleSoA result(
        "kokkos_bucketing_input", input.size(), execution);
    Kokkos::View<Particle*, Memory> device("kokkos_bucketing_input_aos",
                                           input.size());
    auto host = Kokkos::create_mirror_view(device);
    for (std::size_t index = 0; index < input.size(); ++index)
      host(index) = input[index];
    Kokkos::deep_copy(execution, device, host);
    Kokkos::parallel_for(
        "upload_kokkos_bucketing_soa",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, input.size()),
        KOKKOS_LAMBDA(std::size_t const index) {
          result.store(index, device(index));
        });
    execution.fence("upload Kokkos wavefront-bucketing test input");
    return result;
  }

  template <class Batch>
  std::vector<Particle> download(Batch const& batch,
                                 ExecutionSpace const& execution) {
    Kokkos::View<Particle*, Memory> device("kokkos_bucketing_output_aos",
                                           batch.count);
    auto const particles = batch.particles;
    Kokkos::parallel_for(
        "download_kokkos_bucketing_soa",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, batch.count),
        KOKKOS_LAMBDA(std::size_t const index) {
          device(index) = particles.load(index);
        });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    device);
    std::vector<Particle> result(batch.count);
    for (std::size_t index = 0; index < result.size(); ++index)
      result[index] = host(index);
    return result;
  }

  template <class Batch>
  std::vector<std::uint64_t> downloadKeys(Batch const& batch) {
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    batch.keys);
    std::vector<std::uint64_t> result(batch.count);
    for (std::size_t index = 0; index < result.size(); ++index)
      result[index] = host(index);
    return result;
  }

  void require(bool const condition, char const* const message) {
    if (!condition) throw std::runtime_error(message);
  }

  std::vector<Particle> thresholdParticles(std::size_t const count) {
    using corsika::gpu::em::EmPid;
    std::vector<Particle> result;
    result.reserve(count);
    for (std::size_t source = 0; source < count; ++source) {
      auto const selector = (source * 17) % 3;
      auto const pid = static_cast<std::int32_t>(
          selector == 0 ? EmPid::Photon
                        : (selector == 1 ? EmPid::Electron
                                         : EmPid::Positron));
      result.push_back(particle(
          pid, static_cast<std::int32_t>((source / 11) % 5) - 2,
          std::ldexp(1., static_cast<int>((source / 7) % 32) - 16),
          source + 1000));
    }
    if (count >= 2) {
      result[0] = particle(static_cast<std::int32_t>(EmPid::Electron), 0, 1.,
                           1000);
      result[1] = particle(static_cast<std::int32_t>(EmPid::Electron), 0, 1.,
                           1001);
    }
    return result;
  }

  std::vector<Particle> energyBoundaryParticles() {
    using corsika::gpu::em::EmPid;
    std::vector<Particle> result;
    constexpr int first_boundary = -48;
    constexpr int last_boundary = 48;
    result.reserve(3 * (last_boundary - first_boundary + 1));
    std::uint64_t history = 100000;
    for (int sixteenth = first_boundary; sixteenth <= last_boundary;
         ++sixteenth) {
      auto const boundary =
          std::exp2(static_cast<double>(sixteenth) / 16.);
      auto const below = std::nextafter(boundary, 0.);
      auto const above = std::nextafter(
          boundary, std::numeric_limits<double>::infinity());
      result.push_back(particle(
          static_cast<std::int32_t>(EmPid::Photon), 0, below, history++));
      result.push_back(particle(
          static_cast<std::int32_t>(EmPid::Photon), 0, boundary, history++));
      result.push_back(particle(
          static_cast<std::int32_t>(EmPid::Photon), 0, above, history++));
    }
    return result;
  }

  template <class Batch>
  void requireStableOrder(std::vector<Particle> const& input,
                          Batch const& batch,
                          ExecutionSpace const& execution) {
    auto actual = download(batch, execution);
    auto actual_keys = downloadKeys(batch);
    auto expected = input;
    std::stable_sort(expected.begin(), expected.end(),
                     [](Particle const& left, Particle const& right) {
                       return nativeHostKeyOracle(left) <
                              nativeHostKeyOracle(right);
                     });
    require(actual.size() == expected.size(),
            "threshold bucketed size differs");
    for (std::size_t index = 0; index < expected.size(); ++index) {
      require(actual[index].history_id == expected[index].history_id,
              "threshold bucketed stable source order differs");
      require(actual_keys[index] == nativeHostKeyOracle(expected[index]),
              "threshold bucketed key differs");
    }
  }

} // namespace

int main(int argc, char** argv) {
  try {
    Kokkos::ScopeGuard guard(argc, argv);
    ExecutionSpace execution{};
    using corsika::gpu::em::EmPid;
    namespace kd = corsika::accelerator::em::kokkos_detail;

    auto const below_one = std::nextafter(1., 0.);
    auto const above_one =
        std::nextafter(1., std::numeric_limits<double>::infinity());
    std::vector<Particle> input{
        particle(static_cast<std::int32_t>(EmPid::Electron), 0, 1., 10),
        particle(static_cast<std::int32_t>(EmPid::Electron), 0, 1., 11),
        particle(static_cast<std::int32_t>(EmPid::Photon), -1, below_one, 12),
        particle(static_cast<std::int32_t>(EmPid::Photon), -1, 1., 13),
        particle(static_cast<std::int32_t>(EmPid::Photon), -1, above_one, 14),
        particle(static_cast<std::int32_t>(EmPid::Positron),
                 std::numeric_limits<std::int32_t>::min(), 2., 15),
        particle(static_cast<std::int32_t>(EmPid::Positron),
                 std::numeric_limits<std::int32_t>::max(), 2., 16),
        particle(999, 0, 2., 17),
        particle(static_cast<std::int32_t>(EmPid::Photon), 0, 0., 18),
        particle(static_cast<std::int32_t>(EmPid::Photon), 0,
                 std::numeric_limits<double>::infinity(), 19),
        particle(static_cast<std::int32_t>(EmPid::Photon), 0,
                 -std::numeric_limits<double>::infinity(), 20),
        particle(static_cast<std::int32_t>(EmPid::Photon), 0,
                 std::numeric_limits<double>::quiet_NaN(), 21),
        particle(static_cast<std::int32_t>(EmPid::Photon), 0,
                 std::numeric_limits<double>::denorm_min(), 22),
        particle(static_cast<std::int32_t>(EmPid::Photon), 0,
                 std::numeric_limits<double>::max(), 23)};
    while (input.size() <
           corsika::gpu::em::detail::MinimumWavefrontRadixSortSize + 37) {
      auto const source = input.size();
      input.push_back(particle(
          static_cast<std::int32_t>(source % 3 == 0
                                        ? EmPid::Photon
                                        : (source % 3 == 1
                                               ? EmPid::Electron
                                               : EmPid::Positron)),
          static_cast<std::int32_t>(source % 9) - 4,
          std::ldexp(1., static_cast<int>(source % 20) - 10), source + 100));
    }

    for (auto const& value : input)
      require(kd::kokkosWavefrontBucketKey(value) ==
                  nativeHostKeyOracle(value),
              "portable key differs from the native host oracle");

    auto device = upload(input, execution);
    kd::KokkosWavefrontBucketingWorkspace<ExecutionSpace> workspace;
    auto const initial_projection =
        workspace.projectedCapacity(input.size(), execution);
    auto batch = kd::bucketKokkosWavefront<ExecutionSpace>(
        device, input.size(), workspace, execution);
    require(initial_projection.retained_bytes == workspace.deviceBytes(),
            "bucketing allocation projection differs from retained storage");
    require(initial_projection.transient_peak_bytes >=
                initial_projection.retained_bytes,
            "bucketing transient projection is smaller than retained storage");
    auto const reused_projection =
        workspace.projectedCapacity(input.size(), execution);
    require(reused_projection.retained_bytes == workspace.deviceBytes() &&
                reused_projection.transient_peak_bytes ==
                    workspace.deviceBytes(),
            "reused bucketing projection predicted an allocation");
    require(batch.sorted, "sorted wavefront was not marked sorted");
    auto actual = download(batch, execution);
    auto actual_keys = downloadKeys(batch);
#if defined(KOKKOS_ENABLE_CUDA)
    static_assert(
        kd::KokkosWavefrontBucketingWorkspace<ExecutionSpace>::
            cudaRadixSortAvailable(),
        "the Kokkos-CUDA build must make the CUB fast path available");
    static_assert(
        kd::MinimumKokkosCudaRadixSortSize ==
            corsika::gpu::em::detail::MinimumWavefrontRadixSortSize,
        "Kokkos-CUDA and native CUDA no-sort thresholds differ");
    require(workspace.portableCapacity() == 0 &&
                workspace.order().extent(0) == 0 &&
                workspace.radixCapacity() >= input.size() &&
                workspace.radixTemporaryBytes() != 0,
            "the CUDA sorted front did not use radix-only storage");
    auto const input_keys_storage = workspace.inputKeys().data();
    auto const input_indices_storage = workspace.inputIndices().data();
    auto const output_indices_storage = workspace.outputIndices().data();
    auto const temporary_storage = workspace.radixTemporary().data();
    auto const output_storage = workspace.output().pid.data();
    auto const key_storage = workspace.keys().data();
    (void)kd::bucketKokkosWavefront<ExecutionSpace>(
        device, input.size(), workspace, execution);
    require(workspace.inputKeys().data() == input_keys_storage &&
                workspace.inputIndices().data() == input_indices_storage &&
                workspace.outputIndices().data() == output_indices_storage &&
                workspace.radixTemporary().data() == temporary_storage &&
                workspace.output().pid.data() == output_storage &&
                workspace.keys().data() == key_storage,
            "the CUDA radix path did not reuse its grow-only workspace");
#else
    require(workspace.portableCapacity() >= input.size() &&
                workspace.radixCapacity() == 0,
            "the portable backend did not allocate comparison ordering");
    auto const order_storage = workspace.order().data();
    auto const output_storage = workspace.output().pid.data();
    auto const key_storage = workspace.keys().data();
    (void)kd::bucketKokkosWavefront<ExecutionSpace>(
        device, input.size(), workspace, execution);
    require(workspace.order().data() == order_storage &&
                workspace.output().pid.data() == output_storage &&
                workspace.keys().data() == key_storage,
            "the portable path did not reuse its grow-only workspace");
#endif

    auto expected = input;
    std::stable_sort(expected.begin(), expected.end(),
                     [](Particle const& left, Particle const& right) {
                       return nativeHostKeyOracle(left) <
                              nativeHostKeyOracle(right);
                     });
    require(actual.size() == expected.size(), "bucketed size differs");
    for (std::size_t index = 0; index < expected.size(); ++index) {
      require(actual[index].history_id == expected[index].history_id,
              "bucketed particle/source order differs");
      require(actual_keys[index] == nativeHostKeyOracle(expected[index]),
              "bucketed key differs");
      if (index != 0)
        require(actual_keys[index - 1] <= actual_keys[index],
                "bucket keys are not ordered");
    }
    auto const first_equal = std::find_if(
        actual.begin(), actual.end(),
        [](Particle const& value) { return value.history_id == 10; });
    auto const second_equal = std::find_if(
        actual.begin(), actual.end(),
        [](Particle const& value) { return value.history_id == 11; });
    require(first_equal < second_equal, "equal-key source order is unstable");

    auto below_sort_input = thresholdParticles(
        corsika::gpu::em::detail::MinimumWavefrontRadixSortSize - 1);
    auto below_sort_device = upload(below_sort_input, execution);
    kd::KokkosWavefrontBucketingWorkspace<ExecutionSpace>
        below_sort_workspace;
    auto below_sort_batch = kd::bucketKokkosWavefront<ExecutionSpace>(
        below_sort_device, below_sort_input.size(), below_sort_workspace,
        execution);
    require(!below_sort_batch.sorted && below_sort_workspace.capacity() == 0 &&
                below_sort_workspace.portableCapacity() == 0 &&
                below_sort_workspace.radixCapacity() == 0,
            "the 255-particle front did not remain allocation-free");

    auto minimum_sort_input = thresholdParticles(
        corsika::gpu::em::detail::MinimumWavefrontRadixSortSize);
    auto minimum_sort_device = upload(minimum_sort_input, execution);
    kd::KokkosWavefrontBucketingWorkspace<ExecutionSpace>
        minimum_sort_workspace;
    auto minimum_sort_batch = kd::bucketKokkosWavefront<ExecutionSpace>(
        minimum_sort_device, minimum_sort_input.size(), minimum_sort_workspace,
        execution);
#if defined(KOKKOS_ENABLE_CUDA)
    require(minimum_sort_batch.sorted &&
                minimum_sort_workspace.portableCapacity() == 0 &&
                minimum_sort_workspace.radixCapacity() >=
                    minimum_sort_input.size(),
            "the 256-particle CUDA front did not use radix sorting");
#else
    require(minimum_sort_batch.sorted &&
                minimum_sort_workspace.portableCapacity() >=
                    minimum_sort_input.size(),
            "the 256-particle front did not use portable sorting");
#endif
    requireStableOrder(minimum_sort_input, minimum_sort_batch, execution);

    bool alias_rejected = false;
    try {
      (void)kd::bucketKokkosWavefront<ExecutionSpace>(
          minimum_sort_workspace.output(), minimum_sort_input.size(),
          minimum_sort_workspace, execution);
    } catch (std::invalid_argument const&) {
      alias_rejected = true;
    }
    require(alias_rejected,
            "bucketing accepted input aliased to its workspace output");

    auto malformed_input = minimum_sort_device;
    malformed_input.step_id = decltype(malformed_input.step_id)(
        "short_kokkos_bucketing_step_ids", minimum_sort_input.size() - 1);
    kd::KokkosWavefrontBucketingWorkspace<ExecutionSpace>
        malformed_workspace;
    bool short_field_rejected = false;
    try {
      (void)kd::bucketKokkosWavefront<ExecutionSpace>(
          malformed_input, minimum_sort_input.size(), malformed_workspace,
          execution);
    } catch (std::length_error const&) {
      short_field_rejected = true;
    }
    require(short_field_rejected,
            "bucketing accepted a short non-PID input field");

    auto boundary_input = energyBoundaryParticles();
    require(boundary_input.size() >=
                corsika::gpu::em::detail::MinimumWavefrontRadixSortSize,
            "energy-boundary fixture does not exercise sorting");
    auto boundary_device = upload(boundary_input, execution);
    kd::KokkosWavefrontBucketingWorkspace<ExecutionSpace> boundary_workspace;
    auto boundary_batch = kd::bucketKokkosWavefront<ExecutionSpace>(
        boundary_device, boundary_input.size(), boundary_workspace, execution);
    require(boundary_batch.sorted,
            "energy-boundary wavefront was not sorted");
    requireStableOrder(boundary_input, boundary_batch, execution);

    std::vector<Particle> small{
        particle(static_cast<std::int32_t>(EmPid::Positron), 4, 8., 31),
        particle(static_cast<std::int32_t>(EmPid::Photon), -4, 1., 32)};
    auto small_device = upload(small, execution);
    kd::KokkosWavefrontBucketingWorkspace<ExecutionSpace> small_workspace;
    auto small_batch = kd::bucketKokkosWavefront<ExecutionSpace>(
        small_device, small.size(), small_workspace, execution);
    require(!small_batch.sorted, "small-wavefront threshold differs");
    require(small_workspace.capacity() == 0 && small_batch.keys.extent(0) == 0,
            "small wavefront allocated bucketing workspace");
    auto small_actual = download(small_batch, execution);
    require(small_actual[0].history_id == 31 &&
                small_actual[1].history_id == 32,
            "small wavefront did not retain native input order");
  } catch (std::exception const& error) {
    std::cerr << "Kokkos wavefront bucketing failed: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
