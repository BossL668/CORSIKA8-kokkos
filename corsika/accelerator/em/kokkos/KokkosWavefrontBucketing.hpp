/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp>
#include <corsika/accelerator/em/kokkos/KokkosRangePolicy.hpp>
#include <corsika/accelerator/em/common/detail/WavefrontBucketingConstants.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  /**
   * Keep the same no-sort threshold as native CUDA.  Every Kokkos-CUDA front
   * which actually needs bucketing uses the persistent CUB radix workspace.
   * The former medium-front Kokkos::sort path delegated to Thrust, whose
   * temporary allocation introduced one stream synchronization per front.
   */
  inline constexpr std::size_t MinimumKokkosCudaRadixSortSize =
      gpu::em::detail::MinimumWavefrontRadixSortSize;

  namespace wavefront_bucketing_detail {

    inline constexpr double EnergyBinsPerOctave = 16.;
    inline constexpr double EnergyBinOffset = 32768.;
    inline constexpr std::uint16_t InvalidEnergyBin =
        std::numeric_limits<std::uint16_t>::max();
    inline constexpr std::uint16_t MaximumFiniteEnergyBin =
        InvalidEnergyBin - 1;

    KOKKOS_INLINE_FUNCTION std::uint64_t pidBucket(std::int32_t const pid) {
      using gpu::em::EmPid;
      if (pid == static_cast<std::int32_t>(EmPid::Photon)) return 0;
      if (pid == static_cast<std::int32_t>(EmPid::Electron)) return 1;
      if (pid == static_cast<std::int32_t>(EmPid::Positron)) return 2;
      return 3;
    }

    KOKKOS_INLINE_FUNCTION std::uint16_t energyBucket(
        double const energy_GeV) {
      if (!(energy_GeV > 0.) || !std::isfinite(energy_GeV))
        return InvalidEnergyBin;
      auto const coordinate = std::floor(
          std::log2(energy_GeV) * EnergyBinsPerOctave + EnergyBinOffset);
      if (!(coordinate > 0.)) return 0;
      if (coordinate >= static_cast<double>(MaximumFiniteEnergyBin))
        return MaximumFiniteEnergyBin;
      return static_cast<std::uint16_t>(coordinate);
    }

    struct BucketOrder {
      std::uint64_t bucket_key{};
      std::uint32_t source{};
      std::uint32_t reserved{};
    };

    struct BucketOrderLess {
      KOKKOS_INLINE_FUNCTION bool operator()(
          BucketOrder const& left, BucketOrder const& right) const {
        return left.bucket_key < right.bucket_key ||
               (left.bucket_key == right.bucket_key &&
                left.source < right.source);
      }
    };

#if defined(KOKKOS_ENABLE_CUDA)
    inline void checkCuda(cudaError_t const status,
                          char const* const operation) {
      if (status == cudaSuccess) return;
      throw std::runtime_error(std::string(operation) + " failed: " +
                               cudaGetErrorString(status));
    }

    inline std::size_t radixTemporaryBytes(std::size_t const count,
                                           cudaStream_t const stream) {
      std::size_t bytes{};
      checkCuda(
          cub::DeviceRadixSort::SortPairs(
              nullptr, bytes,
              static_cast<std::uint64_t const*>(nullptr),
              static_cast<std::uint64_t*>(nullptr),
              static_cast<std::uint32_t const*>(nullptr),
              static_cast<std::uint32_t*>(nullptr), count, 0,
              gpu::em::detail::WavefrontBucketKeyBits, stream),
          "query Kokkos-CUDA wavefront radix-sort storage");
      return bytes;
    }
#endif

  } // namespace wavefront_bucketing_detail

  /** The exact 50-bit PID x signed-medium x energy key used by native CUDA. */
  KOKKOS_INLINE_FUNCTION std::uint64_t kokkosWavefrontBucketKey(
      gpu::em::EmParticleState const& particle) {
    auto const ordered_medium =
        static_cast<std::uint32_t>(particle.medium_id) ^ 0x80000000U;
    return (wavefront_bucketing_detail::pidBucket(particle.pid) << 48U) |
           (static_cast<std::uint64_t>(ordered_medium) << 16U) |
           wavefront_bucketing_detail::energyBucket(particle.energy_GeV);
  }

  /**
   * Shallow owning views of one bucketed front.
   *
   * Copying the Kokkos views retains their allocations independently of the
   * input and workspace objects.  A non-sorted batch aliases the input
   * contents.  A sorted batch aliases reusable workspace contents, so another
   * use of that workspace may overwrite its values without dangling its views.
   */
  template <class ExecutionSpace>
  struct KokkosWavefrontBucketBatch {
    using memory_space = typename ExecutionSpace::memory_space;

    ParticleSoA<memory_space> particles{};
    Kokkos::View<std::uint64_t*, memory_space> keys{};
    std::size_t count{};
    bool sorted{};
  };

  /** Grow-only storage reused by every wavefront in one resident cascade. */
  template <class ExecutionSpace>
  class KokkosWavefrontBucketingWorkspace {
  public:
    using memory_space = typename ExecutionSpace::memory_space;
    using Order = wavefront_bucketing_detail::BucketOrder;

    static constexpr bool cudaRadixSortAvailable() noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
      return std::is_same_v<ExecutionSpace, Kokkos::Cuda>;
#else
      return false;
#endif
    }

    static constexpr bool usesCudaRadixSort(
        std::size_t const count) noexcept {
      return cudaRadixSortAvailable() &&
             count >= MinimumKokkosCudaRadixSortSize;
    }

    void ensureCapacity(std::size_t const requested,
                        ExecutionSpace const& execution = {}) {
      ensureOutputCapacity(requested, execution);
#if defined(KOKKOS_ENABLE_CUDA)
      if constexpr (cudaRadixSortAvailable()) {
        if (usesCudaRadixSort(requested)) {
          ensureRadixCapacity(requested, execution);
          return;
        }
      }
#endif
      ensurePortableCapacity(requested, execution);
    }

    std::size_t capacity() const noexcept { return output_capacity_; }
    std::size_t portableCapacity() const noexcept {
      return portable_capacity_;
    }
    std::size_t radixCapacity() const noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
      return radix_capacity_;
#else
      return 0;
#endif
    }
    std::size_t deviceBytes() const noexcept {
      auto const view_bytes = [](auto const& view) {
        return view.span() *
               sizeof(typename std::decay_t<decltype(view)>::value_type);
      };
      auto bytes = output_.deviceBytes() + view_bytes(keys_) +
                   view_bytes(order_);
#if defined(KOKKOS_ENABLE_CUDA)
      bytes += view_bytes(input_keys_) + view_bytes(input_indices_) +
               view_bytes(output_indices_) + view_bytes(radix_temporary_);
#endif
      return bytes;
    }

    /**
     * Predict the retained and allocation-overlap bytes of ensureCapacity().
     *
     * The projection follows the exact allocation order below.  In
     * particular, each newly constructed Kokkos::View coexists briefly with
     * the member it replaces.  Counting only the final retained size is not
     * sufficient to prevent an allocator OOM during geometric growth.
     */
    KokkosMemoryProjection projectedCapacity(
        std::size_t const requested,
        ExecutionSpace const& execution = {}) const {
      // CUB queries temporary storage through the execution stream on CUDA;
      // the portable stable-sort path does not need the object itself.
      (void)execution;
      auto const current = deviceBytes();
      if (requested < gpu::em::detail::MinimumWavefrontRadixSortSize)
        return {current, current};

      auto const view_bytes = [](auto const& view) {
        return view.span() *
               sizeof(typename std::decay_t<decltype(view)>::value_type);
      };
      KokkosMemoryProjectionBuilder projection(current);
      if (requested > output_capacity_) {
        auto const grown = grownCapacity(output_capacity_, requested);
        projection.replace(
            output_.deviceBytes(),
            ParticleSoA<memory_space>::deviceBytesForCapacity(grown));
        projection.replace(
            view_bytes(keys_),
            checkedMemoryMultiply(grown, sizeof(std::uint64_t)));
      }

#if defined(KOKKOS_ENABLE_CUDA)
      if constexpr (cudaRadixSortAvailable()) {
        if (usesCudaRadixSort(requested)) {
          if (requested > radix_capacity_) {
            auto const grown = grownCapacity(radix_capacity_, requested);
            projection.replace(
                view_bytes(input_keys_),
                checkedMemoryMultiply(grown, sizeof(std::uint64_t)));
            projection.replace(
                view_bytes(input_indices_),
                checkedMemoryMultiply(grown, sizeof(std::uint32_t)));
            projection.replace(
                view_bytes(output_indices_),
                checkedMemoryMultiply(grown, sizeof(std::uint32_t)));
            projection.replace(
                view_bytes(radix_temporary_),
                wavefront_bucketing_detail::radixTemporaryBytes(
                    grown, execution.cuda_stream()));
          }
          return projection.result();
        }
      }
#endif

      if (requested > portable_capacity_) {
        auto const grown = grownCapacity(portable_capacity_, requested);
        projection.replace(
            view_bytes(order_),
            checkedMemoryMultiply(grown, sizeof(Order)));
      }
      return projection.result();
    }
    // A reserved arena must also cover the portable small-front path used
    // below the CUDA radix threshold, even when its largest front uses CUB.
    void ensureCountRange(std::size_t const maximum_count,
                          ExecutionSpace const& execution) {
      ensureCapacity(maximum_count, execution);
      if (usesCudaRadixSort(maximum_count))
        ensureCapacity(MinimumKokkosCudaRadixSortSize - 1, execution);
    }

    KokkosMemoryProjection projectedCountRange(
        std::size_t const maximum_count,
        ExecutionSpace const& execution) const {
      auto const main = projectedCapacity(maximum_count, execution);
      if (!usesCudaRadixSort(maximum_count)) return main;
      auto const small = MinimumKokkosCudaRadixSortSize - 1;
      if (small <= portable_capacity_) return main;
      KokkosMemoryProjectionBuilder projection(main.retained_bytes);
      projection.replace(
          order_.span() * sizeof(Order),
          checkedMemoryMultiply(grownCapacity(portable_capacity_, small),
                                sizeof(Order)));
      auto result = projection.result();
      result.transient_peak_bytes =
          std::max(result.transient_peak_bytes, main.transient_peak_bytes);
      return result;
    }

    ParticleSoA<memory_space> const& output() const noexcept { return output_; }
    Kokkos::View<std::uint64_t*, memory_space> const& keys() const noexcept {
      return keys_;
    }
    Kokkos::View<Order*, memory_space> const& order() const noexcept {
      return order_;
    }
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::View<std::uint64_t*, memory_space> const&
    inputKeys() const noexcept {
      return input_keys_;
    }
    Kokkos::View<std::uint32_t*, memory_space> const&
    inputIndices() const noexcept {
      return input_indices_;
    }
    Kokkos::View<std::uint32_t*, memory_space> const&
    outputIndices() const noexcept {
      return output_indices_;
    }
    Kokkos::View<unsigned char*, memory_space> const&
    radixTemporary() const noexcept {
      return radix_temporary_;
    }
    std::size_t radixTemporaryBytes() const noexcept {
      return radix_temporary_bytes_;
    }
#endif

  private:
    static std::size_t grownCapacity(std::size_t const current,
                                     std::size_t const requested) {
      if (current == 0) return requested;
      return std::max(
          requested,
          current > std::numeric_limits<std::size_t>::max() / 2
              ? requested
              : current * 2);
    }

    void ensureOutputCapacity(std::size_t const requested,
                              ExecutionSpace const& execution) {
      if (requested <= output_capacity_) return;
      // Every allocation is grow-only and occurs outside the steady-state
      // sorting path. No explicit execution-space fence is introduced here.
      auto const grown = grownCapacity(output_capacity_, requested);
      output_ = ParticleSoA<memory_space>("c8_kokkos_bucketed_wavefront",
                                          grown, execution);
      keys_ = decltype(keys_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_bucketed_wavefront_keys"),
          grown);
      output_capacity_ = grown;
    }

    void ensurePortableCapacity(std::size_t const requested,
                                ExecutionSpace const& execution) {
      if (requested <= portable_capacity_) return;
      auto const grown = grownCapacity(portable_capacity_, requested);
      order_ = decltype(order_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_wavefront_bucket_order"),
          grown);
      portable_capacity_ = grown;
    }

#if defined(KOKKOS_ENABLE_CUDA)
    void ensureRadixCapacity(std::size_t const requested,
                             ExecutionSpace const& execution) {
      if (requested <= radix_capacity_) return;
      auto const grown = grownCapacity(radix_capacity_, requested);
      if constexpr (cudaRadixSortAvailable()) {
        input_keys_ = decltype(input_keys_)(
            Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                               "c8_kokkos_wavefront_input_keys"),
            grown);
        input_indices_ = decltype(input_indices_)(
            Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                               "c8_kokkos_wavefront_input_indices"),
            grown);
        output_indices_ = decltype(output_indices_)(
            Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                               "c8_kokkos_wavefront_output_indices"),
            grown);
        radix_temporary_bytes_ =
            wavefront_bucketing_detail::radixTemporaryBytes(
                grown, execution.cuda_stream());
        radix_temporary_ = decltype(radix_temporary_)(
            Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                               "c8_kokkos_wavefront_radix_temporary"),
            radix_temporary_bytes_);
        radix_capacity_ = grown;
      }
    }
#endif

    ParticleSoA<memory_space> output_{};
    Kokkos::View<std::uint64_t*, memory_space> keys_{};
    Kokkos::View<Order*, memory_space> order_{};
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::View<std::uint64_t*, memory_space> input_keys_{};
    Kokkos::View<std::uint32_t*, memory_space> input_indices_{};
    Kokkos::View<std::uint32_t*, memory_space> output_indices_{};
    Kokkos::View<unsigned char*, memory_space> radix_temporary_{};
    std::size_t radix_temporary_bytes_{};
    std::size_t radix_capacity_{};
#endif
    std::size_t output_capacity_{};
    std::size_t portable_capacity_{};
  };

  /**
   * Stable-equivalent wavefront bucketing for every Kokkos execution space.
   *
   * Kokkos-CUDA uses CUB's stable 50-bit radix sort on the execution-space
   * stream for every front at or above the common no-sort threshold. Other
   * backends always use the portable unique lexicographic pair
   * (bucket key, original source), because Kokkos sorting APIs do not promise
   * stability. Both paths preserve equal-key source order. The input is never
   * modified. Below the no-sort threshold the returned particles alias the
   * input. A sorted result aliases the supplied workspace's output and key
   * storage, whose values may be overwritten when that workspace is used
   * again. The returned batch independently retains owning Kokkos view handles,
   * but the caller must keep the source input and workspace alive and unchanged
   * until all submitted work on execution has completed. After that boundary,
   * destroying either source object does not dangle the returned views.
   */
  template <class ExecutionSpace>
  KokkosWavefrontBucketBatch<ExecutionSpace> bucketKokkosWavefront(
      ParticleSoA<typename ExecutionSpace::memory_space> const& input,
      std::size_t const count,
      KokkosWavefrontBucketingWorkspace<ExecutionSpace>& workspace,
      ExecutionSpace const& execution = {},
      std::size_t const openmp_chunk_size = 0) {

    auto const input_covers_count =
        count <= input.pid.extent(0) &&
        count <= input.medium_id.extent(0) &&
        count <= input.generation.extent(0) &&
        count <= input.reserved.extent(0) &&
        count <= input.energy_GeV.extent(0) &&
        count <= input.position_m.extent(0) &&
        count <= input.direction.extent(0) &&
        count <= input.time_s.extent(0) &&
        count <= input.weight.extent(0) &&
        count <= input.history_id.extent(0) &&
        count <= input.parent_history_id.extent(0) &&
        count <= input.step_id.extent(0);
    if (!input_covers_count)
      throw std::length_error(
          "Kokkos wavefront bucketing exceeds an input field extent");
    if (count > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error(
          "Kokkos wavefront bucketing exceeds 32-bit source indices");

    auto const& workspace_output = workspace.output();
    auto const aliases_workspace_output =
        (input.pid.data() != nullptr &&
         input.pid.data() == workspace_output.pid.data()) ||
        (input.medium_id.data() != nullptr &&
         input.medium_id.data() == workspace_output.medium_id.data()) ||
        (input.generation.data() != nullptr &&
         input.generation.data() == workspace_output.generation.data()) ||
        (input.reserved.data() != nullptr &&
         input.reserved.data() == workspace_output.reserved.data()) ||
        (input.energy_GeV.data() != nullptr &&
         input.energy_GeV.data() == workspace_output.energy_GeV.data()) ||
        (input.position_m.data() != nullptr &&
         input.position_m.data() == workspace_output.position_m.data()) ||
        (input.direction.data() != nullptr &&
         input.direction.data() == workspace_output.direction.data()) ||
        (input.time_s.data() != nullptr &&
         input.time_s.data() == workspace_output.time_s.data()) ||
        (input.weight.data() != nullptr &&
         input.weight.data() == workspace_output.weight.data()) ||
        (input.history_id.data() != nullptr &&
         input.history_id.data() == workspace_output.history_id.data()) ||
        (input.parent_history_id.data() != nullptr &&
         input.parent_history_id.data() ==
             workspace_output.parent_history_id.data()) ||
        (input.step_id.data() != nullptr &&
         input.step_id.data() == workspace_output.step_id.data());
    if (aliases_workspace_output)
      throw std::invalid_argument(
          "Kokkos wavefront input aliases the bucketing workspace output");

    if (count < gpu::em::detail::MinimumWavefrontRadixSortSize)
      return {input, {}, count, false};

    workspace.ensureCapacity(count, execution);
    KokkosWavefrontBucketBatch<ExecutionSpace> result{
        workspace.output(), workspace.keys(), count, true};
    // A ParticleSoA contains twelve owning Kokkos::View objects.  Capturing
    // both input and output SoAs by value makes the CUDA functor larger than
    // Kokkos' constant-memory launch threshold.  Kokkos then serializes every
    // sorted front through a shared constant buffer and an event wait.  The
    // raw views keep exactly the same data/strides while the caller retains the
    // input and workspace owners, and make each launch self-contained and
    // stream ordered.
    auto const input_raw = input.rawDeviceView();
    auto const output_raw = result.particles.rawDeviceView();

#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (
        KokkosWavefrontBucketingWorkspace<ExecutionSpace>::
            cudaRadixSortAvailable()) {
      if (KokkosWavefrontBucketingWorkspace<ExecutionSpace>::
              usesCudaRadixSort(count)) {
        auto* const input_keys = workspace.inputKeys().data();
        auto* const input_indices = workspace.inputIndices().data();
        auto const make_radix_keys = KOKKOS_LAMBDA(
            std::size_t const source) {
          input_keys[source] =
              kokkosWavefrontBucketKey(input_raw.load(source));
          input_indices[source] = static_cast<std::uint32_t>(source);
        };
        static_assert(sizeof(make_radix_keys) < 512);
        Kokkos::parallel_for("c8_kokkos_make_wavefront_radix_keys",
                             makeKokkosRangePolicy(
                                 execution, count, openmp_chunk_size),
                             make_radix_keys);

        auto temporary_bytes = workspace.radixTemporaryBytes();
        wavefront_bucketing_detail::checkCuda(
            cub::DeviceRadixSort::SortPairs(
                workspace.radixTemporary().data(), temporary_bytes,
                workspace.inputKeys().data(), workspace.keys().data(),
                workspace.inputIndices().data(),
                workspace.outputIndices().data(), count, 0,
                gpu::em::detail::WavefrontBucketKeyBits,
                execution.cuda_stream()),
            "Kokkos-CUDA stable wavefront radix sort");

        auto const* const output_indices = workspace.outputIndices().data();
        auto const gather_radix = KOKKOS_LAMBDA(
            std::size_t const destination) {
          output_raw.store(destination,
                           input_raw.load(output_indices[destination]));
        };
        static_assert(sizeof(gather_radix) < 512);
        Kokkos::parallel_for("c8_kokkos_gather_radix_bucketed_wavefront",
                             makeKokkosRangePolicy(
                                 execution, count, openmp_chunk_size),
                             gather_radix);
        return result;
      }
    }
#endif

    auto const active_order = Kokkos::subview(
        workspace.order(), std::make_pair<std::size_t>(0, count));
    auto const make_bucket_order = KOKKOS_LAMBDA(
        std::size_t const source) {
      active_order(source) = {
          kokkosWavefrontBucketKey(input_raw.load(source)),
          static_cast<std::uint32_t>(source), 0};
    };
    static_assert(sizeof(make_bucket_order) < 512);
    Kokkos::parallel_for("c8_kokkos_make_wavefront_bucket_order",
                         makeKokkosRangePolicy(
                             execution, count, openmp_chunk_size),
                         make_bucket_order);

    Kokkos::sort(execution, active_order,
                 wavefront_bucketing_detail::BucketOrderLess{});

    auto* const keys = result.keys.data();
    auto const gather_bucketed = KOKKOS_LAMBDA(
        std::size_t const destination) {
      auto const selected = active_order(destination);
      output_raw.store(destination, input_raw.load(selected.source));
      keys[destination] = selected.bucket_key;
    };
    static_assert(sizeof(gather_bucketed) < 512);
    Kokkos::parallel_for("c8_kokkos_gather_bucketed_wavefront",
                         makeKokkosRangePolicy(
                             execution, count, openmp_chunk_size),
                         gather_bucketed);
    return result;
  }

} // namespace corsika::accelerator::em::kokkos_detail
