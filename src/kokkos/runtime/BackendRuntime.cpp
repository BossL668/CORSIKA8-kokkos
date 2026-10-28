/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/runtime/BackendRuntime.hpp>
#include <corsika/kokkos/em/execution/NativeTableStorage.hpp>
#include <corsika/kokkos/scheduling/WavefrontQueue.hpp>

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

#if defined(CORSIKA_KOKKOS_BACKEND_OPENMP) && !defined(KOKKOS_ENABLE_OPENMP)
#error "The Kokkos OpenMP build requires KOKKOS_ENABLE_OPENMP"
#endif

#if (defined(CORSIKA_KOKKOS_BACKEND_CUDA) || defined(CORSIKA_KOKKOS_BACKEND_HIP) || \
     defined(CORSIKA_KOKKOS_BACKEND_SYCL)) &&                                       \
    defined(KOKKOS_ENABLE_OPENMP)
#error "Kokkos GPU builds must use Serial host and must not enable OpenMP"
#endif

#if defined(CORSIKA_KOKKOS_BACKEND_CUDA) && !defined(KOKKOS_ENABLE_CUDA)
#error "The Kokkos CUDA build requires KOKKOS_ENABLE_CUDA"
#endif

#if defined(CORSIKA_KOKKOS_BACKEND_HIP) && !defined(KOKKOS_ENABLE_HIP)
#error "The Kokkos HIP build requires KOKKOS_ENABLE_HIP"
#endif

#if defined(CORSIKA_KOKKOS_BACKEND_SYCL) && !defined(KOKKOS_ENABLE_SYCL)
#error "The Kokkos SYCL build requires KOKKOS_ENABLE_SYCL"
#endif

#if defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP) && \
    (!defined(KOKKOS_ENABLE_CUDA) || !defined(KOKKOS_ENABLE_OPENMP))
#error "The dual build requires both CUDA and OpenMP in the same Kokkos package"
#endif

namespace corsika::kokkos {

  class RuntimeLease {
  public:
    std::thread::id ownerThread_{std::this_thread::get_id()};
    int device_{};
    int threads_{};
    bool cooperative_{};
    bool initialized_{};
    ~RuntimeLease() {
      if (initialized_ && Kokkos::is_initialized() && !Kokkos::is_finalized())
        Kokkos::finalize();
    }
  };

  namespace {
    int default_cooperative_threads() {
      int available = static_cast<int>(std::thread::hardware_concurrency());
#ifdef __linux__
      cpu_set_t affinity;
      if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
        available = CPU_COUNT(&affinity);
#endif
      return std::max(1, std::min(8, available - 2));
    }
  } // namespace

#if defined(CORSIKA_KOKKOS_BACKEND_OPENMP)
  using selected_execution_space_type = Kokkos::OpenMP;
  constexpr BackendKind selectedKind = BackendKind::OpenMP;
  constexpr char selectedName[] = "openmp";
  constexpr bool selectedIsGpu = false;
#elif defined(CORSIKA_KOKKOS_BACKEND_CUDA) || defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
  using selected_execution_space_type = Kokkos::Cuda;
  constexpr BackendKind selectedKind = BackendKind::Cuda;
  constexpr char selectedName[] = "cuda";
  constexpr bool selectedIsGpu = true;
#elif defined(CORSIKA_KOKKOS_BACKEND_HIP)
  using selected_execution_space_type = Kokkos::HIP;
  constexpr BackendKind selectedKind = BackendKind::Hip;
  constexpr char selectedName[] = "hip";
  constexpr bool selectedIsGpu = true;
#elif defined(CORSIKA_KOKKOS_BACKEND_SYCL)
  using selected_execution_space_type = Kokkos::SYCL;
  constexpr BackendKind selectedKind = BackendKind::Sycl;
  constexpr char selectedName[] = "sycl";
  constexpr bool selectedIsGpu = true;
#else
#error "Exactly one CORSIKA8 Kokkos backend must be selected"
#endif

  // Keep device lambdas at namespace scope.  NVCC rejects extended
  // host/device lambdas defined inside the private nested PIMPL class.
  template <class TExecutionSpace>
  PrimitiveProbeResult run_primitive_probe(std::size_t const values,
                                           std::size_t const chunkSize) {
    using memory_space_type = typename TExecutionSpace::memory_space;
    using policy_type = Kokkos::RangePolicy<TExecutionSpace>;
    Kokkos::View<std::uint32_t*, memory_space_type> flags("probe_flags", values);
    Kokkos::View<std::uint64_t*, memory_space_type> offsets("probe_offsets", values);
    Kokkos::View<std::uint64_t*, memory_space_type> compacted("probe_compacted", values);

    TExecutionSpace execution;
    auto policy = policy_type(execution, 0, values);
    if (chunkSize != 0) policy.set_chunk_size(chunkSize);
    Kokkos::parallel_for(
        "c8_kokkos_probe_flags", policy,
        KOKKOS_LAMBDA(std::size_t const i) { flags(i) = (i % 3U) != 1U ? 1U : 0U; });

    std::uint64_t selected{};
    Kokkos::parallel_scan(
        "c8_kokkos_probe_scan", policy,
        KOKKOS_LAMBDA(std::size_t const i, std::uint64_t& update, bool const final) {
          if (final) { offsets(i) = update; }
          update += flags(i);
        },
        selected);

    Kokkos::parallel_for(
        "c8_kokkos_probe_compact", policy, KOKKOS_LAMBDA(std::size_t const i) {
          if (flags(i) != 0U) { compacted(offsets(i)) = i + 1U; }
        });

    std::uint64_t checksum{};
    Kokkos::parallel_reduce(
        "c8_kokkos_probe_reduce", policy_type(execution, 0, selected),
        KOKKOS_LAMBDA(std::size_t const i, std::uint64_t& update) {
          update += compacted(i);
        },
        checksum);
    execution.fence("c8_kokkos_probe_complete");

    auto const expectedSelected = static_cast<std::uint64_t>(values - (values + 1U) / 3U);
    return {values, selected, checksum, selected == expectedSelected};
  }

  template <class TExecutionSpace>
  TilingProbeResult run_tiling_probe(std::size_t const tracks,
                                     std::size_t const observers,
                                     std::size_t const teamSize,
                                     std::size_t const trackTileSize,
                                     std::size_t const observerTileSize) {
    using memory_space_type = typename TExecutionSpace::memory_space;
    auto const trackTiles = (tracks + trackTileSize - 1) / trackTileSize;
    auto const observerTiles = (observers + observerTileSize - 1) / observerTileSize;
    auto const leagueSize = trackTiles * observerTiles;
    Kokkos::View<std::uint64_t*, memory_space_type> values(
        "c8_kokkos_tiling_probe_values", tracks * observers);
    TExecutionSpace execution;
#if defined(KOKKOS_ENABLE_OPENMP)
    if constexpr (std::is_same_v<TExecutionSpace, Kokkos::OpenMP>) {
      (void)teamSize;
      Kokkos::parallel_for(
          "c8_kokkos_tiling_probe",
          Kokkos::RangePolicy<TExecutionSpace>(execution, 0, leagueSize),
          KOKKOS_LAMBDA(std::size_t const league) {
            auto const trackTile = league / observerTiles;
            auto const observerTile = league % observerTiles;
            for (std::size_t trackLane = 0; trackLane < trackTileSize; ++trackLane) {
              auto const track = trackTile * trackTileSize + trackLane;
              if (track >= tracks) continue;
              for (std::size_t observerLane = 0; observerLane < observerTileSize;
                   ++observerLane) {
                auto const observer = observerTile * observerTileSize + observerLane;
                if (observer >= observers) continue;
                values(track * observers + observer) = (track + 1U) * (observer + 3U);
              }
            }
          });
    } else
#endif
    {
      using policy_type = Kokkos::TeamPolicy<TExecutionSpace>;
      using member_type = typename policy_type::member_type;
      auto const tileLanes = trackTileSize * observerTileSize;
      auto const activeTeamSize = std::min(teamSize, tileLanes);
      Kokkos::parallel_for(
          "c8_kokkos_tiling_probe", policy_type(execution, leagueSize, activeTeamSize),
          KOKKOS_LAMBDA(member_type const& member) {
            auto const league = static_cast<std::size_t>(member.league_rank());
            auto const trackTile = league / observerTiles;
            auto const observerTile = league % observerTiles;
            for (std::size_t lane = member.team_rank(); lane < tileLanes;
                 lane += member.team_size()) {
              auto const trackLane = lane / observerTileSize;
              auto const observerLane = lane % observerTileSize;
              auto const track = trackTile * trackTileSize + trackLane;
              auto const observer = observerTile * observerTileSize + observerLane;
              if (track < tracks && observer < observers)
                values(track * observers + observer) = (track + 1U) * (observer + 3U);
            }
          });
    }
    std::uint64_t checksum{};
    Kokkos::parallel_reduce(
        "c8_kokkos_tiling_probe_reduce",
        Kokkos::RangePolicy<TExecutionSpace>(execution, 0, tracks * observers),
        KOKKOS_LAMBDA(std::size_t const index, std::uint64_t& update) {
          update += values(index);
        },
        checksum);
    execution.fence("finish Kokkos tiling probe");
    auto const trackSum = tracks * (tracks + 1U) / 2U;
    auto const observerSum = observers * (observers - 1U) / 2U + 3U * observers;
    auto const expected = trackSum * observerSum;
    return {tracks, observers, checksum, checksum == expected};
  }

  template <class TExecutionSpace>
  QueueProbeResult run_queue_probe(std::size_t const count) {
    using queue_type = WavefrontQueue<TExecutionSpace>;
    using memory_space_type = typename TExecutionSpace::memory_space;
    std::vector<ParticleState> input(count);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i].pid_ = i % 2 == 0 ? 22 : 11;
      input[i].mediumId_ = static_cast<std::int32_t>(i % 5);
      input[i].generation_ = static_cast<std::uint32_t>(i % 19);
      input[i].energyGeV_ = 0.5 + static_cast<double>(i);
      input[i].positionM_[0] = static_cast<double>(i);
      input[i].positionM_[1] = -static_cast<double>(i);
      input[i].positionM_[2] = 6371000. + static_cast<double>(i);
      input[i].direction_[2] = -1.;
      input[i].timeS_ = static_cast<double>(i) * 1.e-9;
      input[i].weight_ = 1. + static_cast<double>(i % 7);
      input[i].historyId_ = i + 1;
      input[i].parentHistoryId_ = i / 2;
      input[i].stepId_ = i % 11;
    }

    TExecutionSpace execution{};
    queue_type queue(count, execution);
    queue.upload(input, execution);
    typename queue_type::flag_view_type flags("c8_queue_probe_flags", count);
    Kokkos::View<std::uint32_t*, memory_space_type> counts(
        "c8_queue_probe_secondary_counts", count);
    auto hostFlags = Kokkos::create_mirror_view(flags);
    auto hostCounts = Kokkos::create_mirror_view(counts);
    std::uint64_t expectedSlots = 0;
    for (std::size_t i = 0; i < count; ++i) {
      hostFlags(i) = i % 3 != 1 ? 1U : 0U;
      hostCounts(i) = static_cast<std::uint32_t>(i % 4);
      expectedSlots += hostCounts(i);
    }
    Kokkos::deep_copy(execution, flags, hostFlags);
    Kokkos::deep_copy(execution, counts, hostCounts);
    execution.fence("upload Kokkos queue probe inputs");

    auto const secondarySlots = queue.secondaryOffsets(counts);
    auto const retained = queue.compact(flags);
    auto const output = queue.download();
    bool stable = output.size() == retained;
    bool exact = stable;
    std::size_t outputIndex = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
      if (i % 3 == 1) continue;
      stable = stable && outputIndex < output.size() &&
               output[outputIndex].historyId_ == input[i].historyId_;
      exact = exact && outputIndex < output.size() &&
              output[outputIndex].pid_ == input[i].pid_ &&
              output[outputIndex].energyGeV_ == input[i].energyGeV_ &&
              output[outputIndex].positionM_[2] == input[i].positionM_[2] &&
              output[outputIndex].weight_ == input[i].weight_ &&
              output[outputIndex].stepId_ == input[i].stepId_;
      ++outputIndex;
    }
    stable = stable && outputIndex == output.size();
    exact = exact && secondarySlots == expectedSlots;
    return {count, retained, secondarySlots, stable, exact};
  }

  class BackendRuntime::Impl {
  public:
    explicit Impl(RuntimeConfig const& config) {
      if (Kokkos::is_initialized() && !config.runtimeLease_) {
        throw std::logic_error(
            "KokkosRuntime requires sole ownership of Kokkos initialization");
      }
      auto const selected = resolve_execution_backend(config.executionBackend_);
      if (selected == "cuda-openmp" || selected == "openmp-cuda")
        throw std::invalid_argument(
            "cuda-openmp requires the cooperative backend coordinator");
      auto const selectedGpu = selected != "openmp";
      if (config.cooperativeOwner_ && config.runtimeLease_)
        throw std::invalid_argument(
            "A cooperative runtime borrower cannot own initialization");
#if !defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
      if (config.cooperativeOwner_ || config.runtimeLease_)
        throw std::invalid_argument("Cooperative runtime requires a CUDA_OPENMP build");
#endif
      if (config.cooperativeOwner_ && selected != "cuda")
        throw std::invalid_argument("Cooperative runtime owner must select cuda");
      bool const cooperative = config.cooperativeOwner_ || bool(config.runtimeLease_);
      if (selectedGpu && config.threads_ > 1 && !cooperative)
        throw std::invalid_argument("GPU execution requires at most one host thread");
      if (config.runtimeLease_) {
        auto const& lease = *config.runtimeLease_;
        if (!lease.cooperative_ || !lease.initialized_ || !Kokkos::is_initialized() ||
            Kokkos::is_finalized() || lease.ownerThread_ != std::this_thread::get_id() ||
            config.device_ != lease.device_ ||
            (config.threads_ > 0 && config.threads_ != lease.threads_))
          throw std::invalid_argument(
              "Incompatible cooperative runtime lease (thread/device/lifetime)");
      }
#if defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
      int deviceCount{};
      if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        throw std::runtime_error(
            "Experimental CUDA/OpenMP binary requires a working NVIDIA device "
            "during Kokkos initialization, including in OpenMP mode; use the "
            "independent OpenMP binary on CPU-only machines");
#endif
      Kokkos::InitializationSettings settings;
      if (config.device_ >= 0) { settings.set_device_id(config.device_); }
      if (config.threads_ > 0) { settings.set_num_threads(config.threads_); }
#if defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
      // Compiled availability is not simultaneous shower execution. Keep the
      // OpenMP host instance at one thread when selecting the GPU algorithm.
      if (selectedGpu && !cooperative) settings.set_num_threads(1);
      if (config.cooperativeOwner_)
        settings.set_num_threads(config.threads_ > 0 ? config.threads_
                                                     : default_cooperative_threads());
#endif
      if (config.runtimeLease_) {
        lifetime_ = config.runtimeLease_;
      } else {
        lifetime_ = std::make_shared<RuntimeLease>();
        lifetime_->cooperative_ = config.cooperativeOwner_;
        lifetime_->device_ = config.device_;
        Kokkos::initialize(settings);
        lifetime_->initialized_ = true;
#if defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
        lifetime_->threads_ = Kokkos::OpenMP().concurrency();
#endif
      }

      selected_execution_space_type execution;
      info_.kind_ = selectedGpu ? selectedKind : BackendKind::OpenMP;
      info_.backend_ = selected;
      info_.kokkosVersion_ = std::to_string(KOKKOS_VERSION_MAJOR) + "." +
                             std::to_string(KOKKOS_VERSION_MINOR) + "." +
                             std::to_string(KOKKOS_VERSION_PATCH);
      info_.device_ = config.device_;
      info_.concurrency_ = execution.concurrency();
#if defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
      if (!selectedGpu) info_.concurrency_ = Kokkos::OpenMP().concurrency();
#endif
      info_.hostThreads_ = selectedGpu ? 1 : info_.concurrency_;
      info_.gpu_ = selectedGpu;
      info_.openmp_ = !selectedGpu;
      info_.cooperativeRuntime_ = cooperative;
      if (cooperative) info_.hostThreads_ = lifetime_->threads_;
      info_.deviceName_ = selected_execution_space_type::name();
#ifdef CORSIKA_KOKKOS_PROJECT_REVISION
      info_.projectRevision_ = CORSIKA_KOKKOS_PROJECT_REVISION;
#else
      info_.projectRevision_ = "unknown";
#endif
#ifdef __VERSION__
      info_.compilerVersion_ = __VERSION__;
#else
      info_.compilerVersion_ = "unknown";
#endif
#if defined(CORSIKA_KOKKOS_BACKEND_CUDA) || defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
      if (selectedGpu) {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, config.device_) != cudaSuccess)
          throw std::runtime_error("failed to query the Kokkos CUDA device");
        info_.deviceName_ = properties.name;
        info_.architecture_ =
            "sm_" + std::to_string(properties.major) + std::to_string(properties.minor);
        int driver{};
        int runtime{};
        if (cudaDriverGetVersion(&driver) != cudaSuccess ||
            cudaRuntimeGetVersion(&runtime) != cudaSuccess)
          throw std::runtime_error("failed to query CUDA driver/runtime versions");
        info_.driverVersion_ = std::to_string(driver);
        info_.runtimeVersion_ = std::to_string(runtime);
        if (cudaMemGetInfo(&info_.deviceFreeMemoryBytesAtInitialization_,
                           &info_.deviceTotalMemoryBytes_) != cudaSuccess)
          throw std::runtime_error("failed to query available CUDA memory");
      }
#elif defined(CORSIKA_KOKKOS_BACKEND_HIP)
      hipDeviceProp_t properties{};
      if (hipGetDeviceProperties(&properties, config.device_) != hipSuccess)
        throw std::runtime_error("failed to query the Kokkos HIP device");
      info_.deviceName_ = properties.name;
      info_.architecture_ = properties.gcnArchName;
      int runtime{};
      if (hipRuntimeGetVersion(&runtime) != hipSuccess)
        throw std::runtime_error("failed to query the HIP runtime version");
      info_.runtimeVersion_ = std::to_string(runtime);
      info_.driverVersion_ = info_.runtimeVersion_;
      if (hipMemGetInfo(&info_.deviceFreeMemoryBytesAtInitialization_,
                        &info_.deviceTotalMemoryBytes_) != hipSuccess)
        throw std::runtime_error("failed to query available HIP memory");
#elif defined(CORSIKA_KOKKOS_BACKEND_SYCL)
      auto const syclDevice = execution.sycl_queue().get_device();
      info_.deviceName_ = syclDevice.get_info<sycl::info::device::name>();
      info_.architecture_ = "sycl";
      info_.driverVersion_ = syclDevice.get_info<sycl::info::device::driver_version>();
      info_.runtimeVersion_ = "kokkos-sycl";
      info_.deviceTotalMemoryBytes_ =
          syclDevice.get_info<sycl::info::device::global_mem_size>();
      // Standard SYCL does not expose current free device memory. Use total
      // global memory as the stable portable basis for the requested fraction;
      // allocation failures still remain fail-closed.
      info_.deviceFreeMemoryBytesAtInitialization_ = info_.deviceTotalMemoryBytes_;
#endif
      if (!selectedGpu) {
        info_.deviceName_ = "OpenMP";
        info_.architecture_ = "host";
        info_.driverVersion_ = "not-applicable";
        info_.runtimeVersion_ = "openmp";
      }
    }

    ~Impl() = default;

    PrimitiveProbeResult runPrimitiveProbe(std::size_t const values,
                                           std::size_t const chunkSize) const {
      if (values == 0 ||
          values > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "Kokkos primitive probe size must be in [1, INT_MAX]");
      }

#if defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
      if (info_.openmp_) return run_primitive_probe<Kokkos::OpenMP>(values, chunkSize);
#endif
      return run_primitive_probe<selected_execution_space_type>(values, chunkSize);
    }

    RuntimeInfo info_{};
    std::shared_ptr<RuntimeLease> lifetime_;
  };

  BackendRuntime::BackendRuntime(RuntimeConfig const& config)
      : impl_(std::make_unique<Impl>(config)) {}

  BackendRuntime::~BackendRuntime() = default;
  BackendRuntime::BackendRuntime(BackendRuntime&&) noexcept = default;
  BackendRuntime& BackendRuntime::operator=(BackendRuntime&&) noexcept = default;

  RuntimeInfo const& BackendRuntime::getInfo() const noexcept { return impl_->info_; }

  std::shared_ptr<RuntimeLease> BackendRuntime::shareCooperativeLifetime() const {
    if (!impl_->lifetime_->cooperative_)
      throw std::logic_error(
          "Only an explicit cooperative runtime may lend its lifetime");
    return impl_->lifetime_;
  }

  PrimitiveProbeResult BackendRuntime::runPrimitiveProbe(
      std::size_t const values, std::size_t const chunkSize) const {
    return impl_->runPrimitiveProbe(values, chunkSize);
  }

  QueueProbeResult BackendRuntime::runQueueProbe(std::size_t const particles) const {
    if (particles == 0) {
      throw std::invalid_argument("Kokkos queue probe particle count must be positive");
    }
#if defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
    if (impl_->info_.openmp_) return run_queue_probe<Kokkos::OpenMP>(particles);
#endif
    return run_queue_probe<selected_execution_space_type>(particles);
  }

  TilingProbeResult BackendRuntime::runTilingProbe(
      std::size_t const tracks, std::size_t const observers, std::size_t const teamSize,
      std::size_t const trackTileSize, std::size_t const observerTileSize) const {
    if (tracks == 0 || observers == 0 || teamSize == 0 || trackTileSize == 0 ||
        observerTileSize == 0 || teamSize > 1024)
      throw std::invalid_argument("invalid Kokkos tiling-probe dimensions");
#if defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
    if (impl_->info_.openmp_)
      return run_tiling_probe<Kokkos::OpenMP>(tracks, observers, teamSize, trackTileSize,
                                              observerTileSize);
#endif
    return run_tiling_probe<selected_execution_space_type>(
        tracks, observers, teamSize, trackTileSize, observerTileSize);
  }

} // namespace corsika::kokkos
