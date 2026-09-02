/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>
#include <corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp>

#include <Kokkos_Core.hpp>

#include <limits>
#include <stdexcept>
#include <utility>

namespace corsika::accelerator::em {

#if defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
  using SelectedExecutionSpace = Kokkos::OpenMP;
  constexpr AcceleratorKind SelectedKind =
      AcceleratorKind::KokkosOpenMP;
  constexpr char SelectedName[] = "openmp";
  constexpr bool SelectedIsGpu = false;
#elif defined(CORSIKA8_KOKKOS_BACKEND_CUDA)
  using SelectedExecutionSpace = Kokkos::Cuda;
  constexpr AcceleratorKind SelectedKind = AcceleratorKind::KokkosCuda;
  constexpr char SelectedName[] = "cuda";
  constexpr bool SelectedIsGpu = true;
#elif defined(CORSIKA8_KOKKOS_BACKEND_HIP)
  using SelectedExecutionSpace = Kokkos::HIP;
  constexpr AcceleratorKind SelectedKind = AcceleratorKind::KokkosHip;
  constexpr char SelectedName[] = "hip";
  constexpr bool SelectedIsGpu = true;
#elif defined(CORSIKA8_KOKKOS_BACKEND_SYCL)
  using SelectedExecutionSpace = Kokkos::SYCL;
  constexpr AcceleratorKind SelectedKind = AcceleratorKind::KokkosSycl;
  constexpr char SelectedName[] = "sycl";
  constexpr bool SelectedIsGpu = true;
#else
#error "Exactly one CORSIKA8 Kokkos backend must be selected"
#endif

  // Keep device lambdas at namespace scope.  NVCC rejects extended
  // host/device lambdas defined inside the private nested PIMPL class.
  template <class Execution>
  KokkosPrimitiveProbeResult runPrimitiveProbeImpl(
      std::size_t const values) {
    using Memory = typename Execution::memory_space;
    using Policy = Kokkos::RangePolicy<Execution>;
    Kokkos::View<std::uint32_t*, Memory> flags("probe_flags", values);
    Kokkos::View<std::uint64_t*, Memory> offsets("probe_offsets", values);
    Kokkos::View<std::uint64_t*, Memory> compacted(
        "probe_compacted", values);

    Execution execution;
    Kokkos::parallel_for(
        "c8_kokkos_probe_flags", Policy(execution, 0, values),
        KOKKOS_LAMBDA(std::size_t const i) {
          flags(i) = (i % 3U) != 1U ? 1U : 0U;
        });

    std::uint64_t selected{};
    Kokkos::parallel_scan(
        "c8_kokkos_probe_scan", Policy(execution, 0, values),
        KOKKOS_LAMBDA(std::size_t const i, std::uint64_t& update,
                      bool const final) {
          if (final) { offsets(i) = update; }
          update += flags(i);
        },
        selected);

    Kokkos::parallel_for(
        "c8_kokkos_probe_compact", Policy(execution, 0, values),
        KOKKOS_LAMBDA(std::size_t const i) {
          if (flags(i) != 0U) { compacted(offsets(i)) = i + 1U; }
        });

    std::uint64_t checksum{};
    Kokkos::parallel_reduce(
        "c8_kokkos_probe_reduce", Policy(execution, 0, selected),
        KOKKOS_LAMBDA(std::size_t const i, std::uint64_t& update) {
          update += compacted(i);
        },
        checksum);
    execution.fence("c8_kokkos_probe_complete");

    auto const expected_selected =
        static_cast<std::uint64_t>(values - (values + 1U) / 3U);
    return {values, selected, checksum,
            selected == expected_selected};
  }

  template <class Execution>
  KokkosQueueProbeResult runQueueProbeImpl(std::size_t const count) {
    using Queue = kokkos_detail::KokkosWavefrontQueue<Execution>;
    using Memory = typename Execution::memory_space;
    std::vector<gpu::em::EmParticleState> input(count);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i].pid = i % 2 == 0 ? 22 : 11;
      input[i].medium_id = static_cast<std::int32_t>(i % 5);
      input[i].generation = static_cast<std::uint32_t>(i % 19);
      input[i].energy_GeV = 0.5 + static_cast<double>(i);
      input[i].position_m[0] = static_cast<double>(i);
      input[i].position_m[1] = -static_cast<double>(i);
      input[i].position_m[2] = 6371000. + static_cast<double>(i);
      input[i].direction[2] = -1.;
      input[i].time_s = static_cast<double>(i) * 1.e-9;
      input[i].weight = 1. + static_cast<double>(i % 7);
      input[i].history_id = i + 1;
      input[i].parent_history_id = i / 2;
      input[i].step_id = i % 11;
    }

    Queue queue(count);
    queue.upload(input);
    typename Queue::FlagView flags("c8_queue_probe_flags", count);
    Kokkos::View<std::uint32_t*, Memory> counts(
        "c8_queue_probe_secondary_counts", count);
    auto host_flags = Kokkos::create_mirror_view(flags);
    auto host_counts = Kokkos::create_mirror_view(counts);
    std::uint64_t expected_slots = 0;
    for (std::size_t i = 0; i < count; ++i) {
      host_flags(i) = i % 3 != 1 ? 1U : 0U;
      host_counts(i) = static_cast<std::uint32_t>(i % 4);
      expected_slots += host_counts(i);
    }
    Execution execution;
    Kokkos::deep_copy(execution, flags, host_flags);
    Kokkos::deep_copy(execution, counts, host_counts);
    execution.fence("upload Kokkos queue probe inputs");

    auto const secondary_slots = queue.secondaryOffsets(counts);
    auto const retained = queue.compact(flags);
    auto const output = queue.download();
    bool stable = output.size() == retained;
    bool exact = stable;
    std::size_t output_index = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
      if (i % 3 == 1) continue;
      stable = stable && output_index < output.size() &&
               output[output_index].history_id == input[i].history_id;
      exact = exact && output_index < output.size() &&
              output[output_index].pid == input[i].pid &&
              output[output_index].energy_GeV == input[i].energy_GeV &&
              output[output_index].position_m[2] == input[i].position_m[2] &&
              output[output_index].weight == input[i].weight &&
              output[output_index].step_id == input[i].step_id;
      ++output_index;
    }
    stable = stable && output_index == output.size();
    exact = exact && secondary_slots == expected_slots;
    return {count, retained, secondary_slots, stable, exact};
  }

  class KokkosRuntime::Impl {
  public:
    explicit Impl(KokkosRuntimeConfig const& config) {
      if (Kokkos::is_initialized()) {
        throw std::logic_error(
            "KokkosRuntime requires sole ownership of Kokkos initialization");
      }
      Kokkos::InitializationSettings settings;
      if (config.device >= 0) { settings.set_device_id(config.device); }
      if (config.threads > 0) { settings.set_num_threads(config.threads); }
      Kokkos::initialize(settings);
      owns_runtime_ = true;

      SelectedExecutionSpace execution;
      info_.kind = SelectedKind;
      info_.backend = SelectedName;
      info_.kokkos_version =
          std::to_string(KOKKOS_VERSION_MAJOR) + "." +
          std::to_string(KOKKOS_VERSION_MINOR) + "." +
          std::to_string(KOKKOS_VERSION_PATCH);
      info_.device = config.device;
      info_.concurrency = execution.concurrency();
      info_.gpu = SelectedIsGpu;
      info_.openmp = !SelectedIsGpu;
      info_.device_name = SelectedExecutionSpace::name();
    }

    ~Impl() {
      if (owns_runtime_ && Kokkos::is_initialized() &&
          !Kokkos::is_finalized()) {
        Kokkos::finalize();
      }
    }

    KokkosPrimitiveProbeResult runPrimitiveProbe(
        std::size_t const values) const {
      if (values == 0 ||
          values > static_cast<std::size_t>(
                       std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "Kokkos primitive probe size must be in [1, INT_MAX]");
      }

      return runPrimitiveProbeImpl<SelectedExecutionSpace>(values);
    }

    KokkosRuntimeInfo info_{};
    bool owns_runtime_{};
  };

  KokkosRuntime::KokkosRuntime(KokkosRuntimeConfig const& config)
      : impl_(std::make_unique<Impl>(config)) {}

  KokkosRuntime::~KokkosRuntime() = default;
  KokkosRuntime::KokkosRuntime(KokkosRuntime&&) noexcept = default;
  KokkosRuntime& KokkosRuntime::operator=(KokkosRuntime&&) noexcept = default;

  KokkosRuntimeInfo const& KokkosRuntime::info() const noexcept {
    return impl_->info_;
  }

  KokkosPrimitiveProbeResult KokkosRuntime::runPrimitiveProbe(
      std::size_t const values) const {
    return impl_->runPrimitiveProbe(values);
  }

  KokkosQueueProbeResult KokkosRuntime::runQueueProbe(
      std::size_t const particles) const {
    if (particles == 0) {
      throw std::invalid_argument(
          "Kokkos queue probe particle count must be positive");
    }
    return runQueueProbeImpl<SelectedExecutionSpace>(particles);
  }

} // namespace corsika::accelerator::em
