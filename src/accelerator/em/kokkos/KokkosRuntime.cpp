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
#include <type_traits>
#include <thread>
#include <algorithm>
#ifdef __linux__
#include <sched.h>
#endif

#if defined(CORSIKA8_KOKKOS_BACKEND_OPENMP) && !defined(KOKKOS_ENABLE_OPENMP)
#error "The Kokkos OpenMP build requires KOKKOS_ENABLE_OPENMP"
#endif

#if (defined(CORSIKA8_KOKKOS_BACKEND_CUDA) ||                         \
     defined(CORSIKA8_KOKKOS_BACKEND_HIP) ||                         \
     defined(CORSIKA8_KOKKOS_BACKEND_SYCL)) &&                       \
    defined(KOKKOS_ENABLE_OPENMP)
#error "Kokkos GPU builds must use Serial host and must not enable OpenMP"
#endif

#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA) && !defined(KOKKOS_ENABLE_CUDA)
#error "The Kokkos CUDA build requires KOKKOS_ENABLE_CUDA"
#endif

#if defined(CORSIKA8_KOKKOS_BACKEND_HIP) && !defined(KOKKOS_ENABLE_HIP)
#error "The Kokkos HIP build requires KOKKOS_ENABLE_HIP"
#endif

#if defined(CORSIKA8_KOKKOS_BACKEND_SYCL) && !defined(KOKKOS_ENABLE_SYCL)
#error "The Kokkos SYCL build requires KOKKOS_ENABLE_SYCL"
#endif

#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP) && \
    (!defined(KOKKOS_ENABLE_CUDA) || !defined(KOKKOS_ENABLE_OPENMP))
#error "The dual build requires both CUDA and OpenMP in the same Kokkos package"
#endif

namespace corsika::accelerator::em {

  class KokkosRuntimeLease {
  public:
    std::thread::id owner_thread{std::this_thread::get_id()};
    int device{};
    int threads{};
    bool cooperative{};
    bool initialized{};
    ~KokkosRuntimeLease() {
      if (initialized && Kokkos::is_initialized() && !Kokkos::is_finalized())
        Kokkos::finalize();
    }
  };

  namespace {
    int defaultCooperativeThreads() {
      int available = static_cast<int>(std::thread::hardware_concurrency());
#ifdef __linux__
      cpu_set_t affinity;
      if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
        available = CPU_COUNT(&affinity);
#endif
      return std::max(1, std::min(8, available - 2));
    }
  }

#if defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
  using SelectedExecutionSpace = Kokkos::OpenMP;
  constexpr AcceleratorKind SelectedKind =
      AcceleratorKind::KokkosOpenMP;
  constexpr char SelectedName[] = "openmp";
  constexpr bool SelectedIsGpu = false;
#elif defined(CORSIKA8_KOKKOS_BACKEND_CUDA) || defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
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
      std::size_t const values, std::size_t const chunk_size) {
    using Memory = typename Execution::memory_space;
    using Policy = Kokkos::RangePolicy<Execution>;
    Kokkos::View<std::uint32_t*, Memory> flags("probe_flags", values);
    Kokkos::View<std::uint64_t*, Memory> offsets("probe_offsets", values);
    Kokkos::View<std::uint64_t*, Memory> compacted(
        "probe_compacted", values);

    Execution execution;
    auto policy = Policy(execution, 0, values);
    if (chunk_size != 0) policy.set_chunk_size(chunk_size);
    Kokkos::parallel_for(
        "c8_kokkos_probe_flags", policy,
        KOKKOS_LAMBDA(std::size_t const i) {
          flags(i) = (i % 3U) != 1U ? 1U : 0U;
        });

    std::uint64_t selected{};
    Kokkos::parallel_scan(
        "c8_kokkos_probe_scan", policy,
        KOKKOS_LAMBDA(std::size_t const i, std::uint64_t& update,
                      bool const final) {
          if (final) { offsets(i) = update; }
          update += flags(i);
        },
        selected);

    Kokkos::parallel_for(
        "c8_kokkos_probe_compact", policy,
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
  KokkosTilingProbeResult runTilingProbeImpl(
      std::size_t const tracks, std::size_t const observers,
      std::size_t const team_size, std::size_t const track_tile_size,
      std::size_t const observer_tile_size) {
    using Memory = typename Execution::memory_space;
    auto const track_tiles =
        (tracks + track_tile_size - 1) / track_tile_size;
    auto const observer_tiles =
        (observers + observer_tile_size - 1) / observer_tile_size;
    auto const league_size = track_tiles * observer_tiles;
    Kokkos::View<std::uint64_t*, Memory> values(
        "c8_kokkos_tiling_probe_values", tracks * observers);
    Execution execution;
#if defined(KOKKOS_ENABLE_OPENMP)
    if constexpr (std::is_same_v<Execution, Kokkos::OpenMP>) {
    (void)team_size;
    Kokkos::parallel_for(
        "c8_kokkos_tiling_probe",
        Kokkos::RangePolicy<Execution>(execution, 0, league_size),
        KOKKOS_LAMBDA(std::size_t const league) {
          auto const track_tile = league / observer_tiles;
          auto const observer_tile = league % observer_tiles;
          for (std::size_t track_lane = 0; track_lane < track_tile_size;
               ++track_lane) {
            auto const track = track_tile * track_tile_size + track_lane;
            if (track >= tracks) continue;
            for (std::size_t observer_lane = 0;
                 observer_lane < observer_tile_size; ++observer_lane) {
              auto const observer =
                  observer_tile * observer_tile_size + observer_lane;
              if (observer >= observers) continue;
              values(track * observers + observer) =
                  (track + 1U) * (observer + 3U);
            }
          }
        });
    } else
#endif
    {
    using Policy = Kokkos::TeamPolicy<Execution>;
    using Member = typename Policy::member_type;
    auto const tile_lanes = track_tile_size * observer_tile_size;
    auto const active_team_size = std::min(team_size, tile_lanes);
    Kokkos::parallel_for(
        "c8_kokkos_tiling_probe",
        Policy(execution, league_size, active_team_size),
        KOKKOS_LAMBDA(Member const& member) {
          auto const league = static_cast<std::size_t>(member.league_rank());
          auto const track_tile = league / observer_tiles;
          auto const observer_tile = league % observer_tiles;
          for (std::size_t lane = member.team_rank(); lane < tile_lanes;
               lane += member.team_size()) {
            auto const track_lane = lane / observer_tile_size;
            auto const observer_lane = lane % observer_tile_size;
            auto const track = track_tile * track_tile_size + track_lane;
            auto const observer =
                observer_tile * observer_tile_size + observer_lane;
            if (track < tracks && observer < observers)
              values(track * observers + observer) =
                  (track + 1U) * (observer + 3U);
          }
        });
    }
    std::uint64_t checksum{};
    Kokkos::parallel_reduce(
        "c8_kokkos_tiling_probe_reduce",
        Kokkos::RangePolicy<Execution>(execution, 0, tracks * observers),
        KOKKOS_LAMBDA(std::size_t const index, std::uint64_t& update) {
          update += values(index);
        },
        checksum);
    execution.fence("finish Kokkos tiling probe");
    auto const track_sum = tracks * (tracks + 1U) / 2U;
    auto const observer_sum = observers * (observers - 1U) / 2U +
                              3U * observers;
    auto const expected = track_sum * observer_sum;
    return {tracks, observers, checksum, checksum == expected};
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

    Execution execution{};
    Queue queue(count, execution);
    queue.upload(input, execution);
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
      if (Kokkos::is_initialized() && !config.runtime_lease) {
        throw std::logic_error(
            "KokkosRuntime requires sole ownership of Kokkos initialization");
      }
      auto const selected = resolveKokkosExecutionBackend(config.execution_backend);
      if (selected == "cuda-openmp")
        throw std::invalid_argument("cuda-openmp requires the cooperative backend coordinator");
      auto const selected_gpu = selected != "openmp";
      if (config.cooperative_owner && config.runtime_lease)
        throw std::invalid_argument("A cooperative runtime borrower cannot own initialization");
#if !defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
      if (config.cooperative_owner || config.runtime_lease)
        throw std::invalid_argument("Cooperative runtime requires a CUDA_OPENMP build");
#endif
      if (config.cooperative_owner && selected != "cuda")
        throw std::invalid_argument("Cooperative runtime owner must select cuda");
      bool const cooperative = config.cooperative_owner || bool(config.runtime_lease);
      if (selected_gpu && config.threads > 1 && !cooperative)
        throw std::invalid_argument("GPU execution requires at most one host thread");
      if (config.runtime_lease) {
        auto const& lease = *config.runtime_lease;
        if (!lease.cooperative || !lease.initialized || !Kokkos::is_initialized() ||
            Kokkos::is_finalized() || lease.owner_thread != std::this_thread::get_id() ||
            config.device != lease.device || (config.threads > 0 && config.threads != lease.threads))
          throw std::invalid_argument("Incompatible cooperative runtime lease (thread/device/lifetime)");
      }
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
      int device_count{};
      if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        throw std::runtime_error(
            "Experimental CUDA/OpenMP binary requires a working NVIDIA device "
            "during Kokkos initialization, including in OpenMP mode; use the "
            "independent OpenMP binary on CPU-only machines");
#endif
      Kokkos::InitializationSettings settings;
      if (config.device >= 0) { settings.set_device_id(config.device); }
      if (config.threads > 0) { settings.set_num_threads(config.threads); }
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
      // Compiled availability is not simultaneous shower execution. Keep the
      // OpenMP host instance at one thread when selecting the GPU algorithm.
      if (selected_gpu && !cooperative) settings.set_num_threads(1);
      if (config.cooperative_owner)
        settings.set_num_threads(config.threads > 0 ? config.threads : defaultCooperativeThreads());
#endif
      if (config.runtime_lease) {
        lifetime_ = config.runtime_lease;
      } else {
        lifetime_ = std::make_shared<KokkosRuntimeLease>();
        lifetime_->cooperative = config.cooperative_owner;
        lifetime_->device = config.device;
        Kokkos::initialize(settings);
        lifetime_->initialized = true;
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
        lifetime_->threads = Kokkos::OpenMP().concurrency();
#endif
      }

      SelectedExecutionSpace execution;
      info_.kind = selected_gpu ? SelectedKind : AcceleratorKind::KokkosOpenMP;
      info_.backend = selected;
      info_.kokkos_version =
          std::to_string(KOKKOS_VERSION_MAJOR) + "." +
          std::to_string(KOKKOS_VERSION_MINOR) + "." +
          std::to_string(KOKKOS_VERSION_PATCH);
      info_.device = config.device;
      info_.concurrency = execution.concurrency();
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
      if (!selected_gpu) info_.concurrency = Kokkos::OpenMP().concurrency();
#endif
      info_.host_threads = selected_gpu ? 1 : info_.concurrency;
      info_.gpu = selected_gpu;
      info_.openmp = !selected_gpu;
      info_.cooperative_runtime = cooperative;
      if (cooperative) info_.host_threads = lifetime_->threads;
      info_.device_name = SelectedExecutionSpace::name();
#ifdef CORSIKA8_PROJECT_REVISION
      info_.project_revision = CORSIKA8_PROJECT_REVISION;
#else
      info_.project_revision = "unknown";
#endif
#ifdef __VERSION__
      info_.compiler_version = __VERSION__;
#else
      info_.compiler_version = "unknown";
#endif
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA) || defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
      if (selected_gpu) {
      cudaDeviceProp properties{};
      if (cudaGetDeviceProperties(&properties, config.device) != cudaSuccess)
        throw std::runtime_error("failed to query the Kokkos CUDA device");
      info_.device_name = properties.name;
      info_.architecture = "sm_" + std::to_string(properties.major) +
                           std::to_string(properties.minor);
      int driver{};
      int runtime{};
      if (cudaDriverGetVersion(&driver) != cudaSuccess ||
          cudaRuntimeGetVersion(&runtime) != cudaSuccess)
        throw std::runtime_error("failed to query CUDA driver/runtime versions");
      info_.driver_version = std::to_string(driver);
      info_.runtime_version = std::to_string(runtime);
      if (cudaMemGetInfo(
              &info_.device_free_memory_bytes_at_initialization,
              &info_.device_total_memory_bytes) != cudaSuccess)
        throw std::runtime_error("failed to query available CUDA memory");
      }
#elif defined(CORSIKA8_KOKKOS_BACKEND_HIP)
      hipDeviceProp_t properties{};
      if (hipGetDeviceProperties(&properties, config.device) != hipSuccess)
        throw std::runtime_error("failed to query the Kokkos HIP device");
      info_.device_name = properties.name;
      info_.architecture = properties.gcnArchName;
      int runtime{};
      if (hipRuntimeGetVersion(&runtime) != hipSuccess)
        throw std::runtime_error("failed to query the HIP runtime version");
      info_.runtime_version = std::to_string(runtime);
      info_.driver_version = info_.runtime_version;
      if (hipMemGetInfo(
              &info_.device_free_memory_bytes_at_initialization,
              &info_.device_total_memory_bytes) != hipSuccess)
        throw std::runtime_error("failed to query available HIP memory");
#elif defined(CORSIKA8_KOKKOS_BACKEND_SYCL)
      auto const sycl_device = execution.sycl_queue().get_device();
      info_.device_name =
          sycl_device.get_info<sycl::info::device::name>();
      info_.architecture = "sycl";
      info_.driver_version =
          sycl_device.get_info<sycl::info::device::driver_version>();
      info_.runtime_version = "kokkos-sycl";
      info_.device_total_memory_bytes =
          sycl_device.get_info<sycl::info::device::global_mem_size>();
      // Standard SYCL does not expose current free device memory. Use total
      // global memory as the stable portable basis for the requested fraction;
      // allocation failures still remain fail-closed.
      info_.device_free_memory_bytes_at_initialization =
          info_.device_total_memory_bytes;
#endif
      if (!selected_gpu) {
      info_.device_name = "OpenMP";
      info_.architecture = "host";
      info_.driver_version = "not-applicable";
      info_.runtime_version = "openmp";
      }
    }

    ~Impl() = default;

    KokkosPrimitiveProbeResult runPrimitiveProbe(
        std::size_t const values, std::size_t const chunk_size) const {
      if (values == 0 ||
          values > static_cast<std::size_t>(
                       std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "Kokkos primitive probe size must be in [1, INT_MAX]");
      }

#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
      if (info_.openmp) return runPrimitiveProbeImpl<Kokkos::OpenMP>(values, chunk_size);
#endif
      return runPrimitiveProbeImpl<SelectedExecutionSpace>(values, chunk_size);
    }

    KokkosRuntimeInfo info_{};
    std::shared_ptr<KokkosRuntimeLease> lifetime_;
  };

  KokkosRuntime::KokkosRuntime(KokkosRuntimeConfig const& config)
      : impl_(std::make_unique<Impl>(config)) {}

  KokkosRuntime::~KokkosRuntime() = default;
  KokkosRuntime::KokkosRuntime(KokkosRuntime&&) noexcept = default;
  KokkosRuntime& KokkosRuntime::operator=(KokkosRuntime&&) noexcept = default;

  KokkosRuntimeInfo const& KokkosRuntime::info() const noexcept {
    return impl_->info_;
  }

  std::shared_ptr<KokkosRuntimeLease> KokkosRuntime::shareCooperativeLifetime() const {
    if (!impl_->lifetime_->cooperative)
      throw std::logic_error("Only an explicit cooperative runtime may lend its lifetime");
    return impl_->lifetime_;
  }

  KokkosPrimitiveProbeResult KokkosRuntime::runPrimitiveProbe(
      std::size_t const values, std::size_t const chunk_size) const {
    return impl_->runPrimitiveProbe(values, chunk_size);
  }

  KokkosQueueProbeResult KokkosRuntime::runQueueProbe(
      std::size_t const particles) const {
    if (particles == 0) {
      throw std::invalid_argument(
          "Kokkos queue probe particle count must be positive");
    }
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    if (impl_->info_.openmp) return runQueueProbeImpl<Kokkos::OpenMP>(particles);
#endif
    return runQueueProbeImpl<SelectedExecutionSpace>(particles);
  }

  KokkosTilingProbeResult KokkosRuntime::runTilingProbe(
      std::size_t const tracks, std::size_t const observers,
      std::size_t const team_size, std::size_t const track_tile_size,
      std::size_t const observer_tile_size) const {
    if (tracks == 0 || observers == 0 || team_size == 0 ||
        track_tile_size == 0 || observer_tile_size == 0 || team_size > 1024)
      throw std::invalid_argument("invalid Kokkos tiling-probe dimensions");
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    if (impl_->info_.openmp)
      return runTilingProbeImpl<Kokkos::OpenMP>(
          tracks, observers, team_size, track_tile_size, observer_tile_size);
#endif
    return runTilingProbeImpl<SelectedExecutionSpace>(
        tracks, observers, team_size, track_tile_size, observer_tile_size);
  }

} // namespace corsika::accelerator::em
