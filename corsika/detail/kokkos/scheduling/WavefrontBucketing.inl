/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <Kokkos_Sort.hpp>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#endif

namespace corsika::kokkos {

  namespace wavefront_bucketing_detail {

    inline constexpr double EnergyBinsPerOctave = 16.;
    inline constexpr double EnergyBinOffset = 32768.;
    inline constexpr std::uint16_t InvalidEnergyBin =
        std::numeric_limits<std::uint16_t>::max();
    inline constexpr std::uint16_t MaximumFiniteEnergyBin = InvalidEnergyBin - 1;

    KOKKOS_INLINE_FUNCTION std::uint64_t pid_bucket(std::int32_t const pid) {
      using corsika::kokkos::EmPid;
      if (pid == static_cast<std::int32_t>(EmPid::Photon)) return 0;
      if (pid == static_cast<std::int32_t>(EmPid::Electron)) return 1;
      if (pid == static_cast<std::int32_t>(EmPid::Positron)) return 2;
      return 3;
    }

    KOKKOS_INLINE_FUNCTION std::uint16_t energy_bucket(double const energyGeV) {
      if (!(energyGeV > 0.) || !std::isfinite(energyGeV)) return InvalidEnergyBin;
      auto const coordinate =
          std::floor(std::log2(energyGeV) * EnergyBinsPerOctave + EnergyBinOffset);
      if (!(coordinate > 0.)) return 0;
      if (coordinate >= static_cast<double>(MaximumFiniteEnergyBin))
        return MaximumFiniteEnergyBin;
      return static_cast<std::uint16_t>(coordinate);
    }

    KOKKOS_INLINE_FUNCTION bool BucketOrderLess::operator()(
        BucketOrder const& left, BucketOrder const& right) const {
      return left.bucketKey_ < right.bucketKey_ ||
             (left.bucketKey_ == right.bucketKey_ && left.source_ < right.source_);
    }

#if defined(KOKKOS_ENABLE_CUDA)
    inline void check_cuda(cudaError_t const status, char const* const operation) {
      if (status == cudaSuccess) return;
      throw std::runtime_error(std::string(operation) +
                               " failed: " + cudaGetErrorString(status));
    }

    inline std::size_t radix_temporary_bytes(std::size_t const count,
                                             cudaStream_t const stream) {
      std::size_t bytes{};
      check_cuda(cub::DeviceRadixSort::SortPairs(
                     nullptr, bytes, static_cast<std::uint64_t const*>(nullptr),
                     static_cast<std::uint64_t*>(nullptr),
                     static_cast<std::uint32_t const*>(nullptr),
                     static_cast<std::uint32_t*>(nullptr), count, 0,
                     WavefrontBucketKeyBits, stream),
                 "query Kokkos-CUDA wavefront radix-sort storage");
      return bytes;
    }
#endif

  } // namespace wavefront_bucketing_detail

  KOKKOS_INLINE_FUNCTION std::uint64_t wavefront_bucket_key(
      ParticleState const& particle) {
    auto const orderedMedium =
        static_cast<std::uint32_t>(particle.mediumId_) ^ 0x80000000U;
    return (wavefront_bucketing_detail::pid_bucket(particle.pid_) << 48U) |
           (static_cast<std::uint64_t>(orderedMedium) << 16U) |
           wavefront_bucketing_detail::energy_bucket(particle.energyGeV_);
  }
  template <class TExecutionSpace>
  constexpr bool
  WavefrontBucketingWorkspace<TExecutionSpace>::cudaRadixSortAvailable() noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
    return std::is_same_v<TExecutionSpace, Kokkos::Cuda>;
#else
    return false;
#endif
  }

  template <class TExecutionSpace>
  constexpr bool WavefrontBucketingWorkspace<TExecutionSpace>::usesCudaRadixSort(
      std::size_t const count) noexcept {
    return cudaRadixSortAvailable() && count >= MinimumKokkosCudaRadixSortSize;
  }

  template <class TExecutionSpace>
  void WavefrontBucketingWorkspace<TExecutionSpace>::ensureCapacity(
      std::size_t const requested, TExecutionSpace const& execution) {
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

  template <class TExecutionSpace>
  std::size_t WavefrontBucketingWorkspace<TExecutionSpace>::capacity() const noexcept {
    return outputCapacity_;
  }

  template <class TExecutionSpace>
  std::size_t WavefrontBucketingWorkspace<TExecutionSpace>::portableCapacity() const
      noexcept {
    return portableCapacity_;
  }

  template <class TExecutionSpace>
  std::size_t WavefrontBucketingWorkspace<TExecutionSpace>::radixCapacity() const
      noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
    return radixCapacity_;
#else
    return 0;
#endif
  }

  template <class TExecutionSpace>
  std::size_t WavefrontBucketingWorkspace<TExecutionSpace>::deviceBytes() const noexcept {
    auto const viewBytes = [](auto const& view) {
      return view.span() * sizeof(typename std::decay_t<decltype(view)>::value_type);
    };
    auto bytes = output_.deviceBytes() + viewBytes(keys_) + viewBytes(order_);
#if defined(KOKKOS_ENABLE_CUDA)
    bytes += viewBytes(inputKeys_) + viewBytes(inputIndices_) +
             viewBytes(outputIndices_) + viewBytes(radixTemporary_);
#endif
    return bytes;
  }

  template <class TExecutionSpace>
  MemoryProjection WavefrontBucketingWorkspace<TExecutionSpace>::projectedCapacity(
      std::size_t const requested, TExecutionSpace const& execution) const {
    // CUB queries temporary storage through the execution stream on CUDA;
    // the portable stable-sort path does not need the object itself.
    (void)execution;
    auto const current = deviceBytes();
    if (requested < MinimumWavefrontRadixSortSize) return {current, current};

    auto const viewBytes = [](auto const& view) {
      return view.span() * sizeof(typename std::decay_t<decltype(view)>::value_type);
    };
    MemoryProjectionBuilder projection(current);
    if (requested > outputCapacity_) {
      auto const grown = grownCapacity(outputCapacity_, requested);
      projection.replace(output_.deviceBytes(),
                         ParticleSoA<memory_space_type>::deviceBytesForCapacity(grown));
      projection.replace(viewBytes(keys_),
                         checked_memory_multiply(grown, sizeof(std::uint64_t)));
    }

#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (cudaRadixSortAvailable()) {
      if (usesCudaRadixSort(requested)) {
        if (requested > radixCapacity_) {
          auto const grown = grownCapacity(radixCapacity_, requested);
          projection.replace(viewBytes(inputKeys_),
                             checked_memory_multiply(grown, sizeof(std::uint64_t)));
          projection.replace(viewBytes(inputIndices_),
                             checked_memory_multiply(grown, sizeof(std::uint32_t)));
          projection.replace(viewBytes(outputIndices_),
                             checked_memory_multiply(grown, sizeof(std::uint32_t)));
          projection.replace(viewBytes(radixTemporary_),
                             wavefront_bucketing_detail::radix_temporary_bytes(
                                 grown, execution.cuda_stream()));
        }
        return projection.result();
      }
    }
#endif

    if (requested > portableCapacity_) {
      auto const grown = grownCapacity(portableCapacity_, requested);
      projection.replace(viewBytes(order_),
                         checked_memory_multiply(grown, sizeof(order_type)));
    }
    return projection.result();
  }

  template <class TExecutionSpace>
  void WavefrontBucketingWorkspace<TExecutionSpace>::ensureCountRange(
      std::size_t const maximumCount, TExecutionSpace const& execution) {
    ensureCapacity(maximumCount, execution);
    if (usesCudaRadixSort(maximumCount))
      ensureCapacity(MinimumKokkosCudaRadixSortSize - 1, execution);
  }

  template <class TExecutionSpace>
  MemoryProjection WavefrontBucketingWorkspace<TExecutionSpace>::projectedCountRange(
      std::size_t const maximumCount, TExecutionSpace const& execution) const {
    auto const main = projectedCapacity(maximumCount, execution);
    if (!usesCudaRadixSort(maximumCount)) return main;
    auto const small = MinimumKokkosCudaRadixSortSize - 1;
    if (small <= portableCapacity_) return main;
    MemoryProjectionBuilder projection(main.retainedBytes_);
    projection.replace(order_.span() * sizeof(order_type),
                       checked_memory_multiply(grownCapacity(portableCapacity_, small),
                                               sizeof(order_type)));
    auto result = projection.result();
    result.transientPeakBytes_ =
        std::max(result.transientPeakBytes_, main.transientPeakBytes_);
    return result;
  }

  template <class TExecutionSpace>
  ParticleSoA<typename TExecutionSpace::memory_space> const&
  WavefrontBucketingWorkspace<TExecutionSpace>::output() const noexcept {
    return output_;
  }

  template <class TExecutionSpace>
  Kokkos::View<std::uint64_t*, typename TExecutionSpace::memory_space> const&
  WavefrontBucketingWorkspace<TExecutionSpace>::keys() const noexcept {
    return keys_;
  }

  template <class TExecutionSpace>
  Kokkos::View<typename WavefrontBucketingWorkspace<TExecutionSpace>::order_type*,
               typename TExecutionSpace::memory_space> const&
  WavefrontBucketingWorkspace<TExecutionSpace>::order() const noexcept {
    return order_;
  }

#if defined(KOKKOS_ENABLE_CUDA)
  template <class TExecutionSpace>
  Kokkos::View<std::uint64_t*, typename TExecutionSpace::memory_space> const&
  WavefrontBucketingWorkspace<TExecutionSpace>::inputKeys() const noexcept {
    return inputKeys_;
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  template <class TExecutionSpace>
  Kokkos::View<std::uint32_t*, typename TExecutionSpace::memory_space> const&
  WavefrontBucketingWorkspace<TExecutionSpace>::inputIndices() const noexcept {
    return inputIndices_;
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  template <class TExecutionSpace>
  Kokkos::View<std::uint32_t*, typename TExecutionSpace::memory_space> const&
  WavefrontBucketingWorkspace<TExecutionSpace>::outputIndices() const noexcept {
    return outputIndices_;
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  template <class TExecutionSpace>
  Kokkos::View<unsigned char*, typename TExecutionSpace::memory_space> const&
  WavefrontBucketingWorkspace<TExecutionSpace>::radixTemporary() const noexcept {
    return radixTemporary_;
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  template <class TExecutionSpace>
  std::size_t WavefrontBucketingWorkspace<TExecutionSpace>::radixTemporaryBytes() const
      noexcept {
    return radixTemporaryBytes_;
  }
#endif

  template <class TExecutionSpace>
  std::size_t WavefrontBucketingWorkspace<TExecutionSpace>::grownCapacity(
      std::size_t const current, std::size_t const requested) {
    if (current == 0) return requested;
    return std::max(requested, current > std::numeric_limits<std::size_t>::max() / 2
                                   ? requested
                                   : current * 2);
  }

  template <class TExecutionSpace>
  void WavefrontBucketingWorkspace<TExecutionSpace>::ensureOutputCapacity(
      std::size_t const requested, TExecutionSpace const& execution) {
    if (requested <= outputCapacity_) return;
    // Every allocation is grow-only and occurs outside the steady-state
    // sorting path. No explicit execution-space fence is introduced here.
    auto const grown = grownCapacity(outputCapacity_, requested);
    output_ =
        ParticleSoA<memory_space_type>("c8_kokkos_bucketed_wavefront", grown, execution);
    keys_ = decltype(keys_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                               "c8_kokkos_bucketed_wavefront_keys"),
                            grown);
    outputCapacity_ = grown;
  }

  template <class TExecutionSpace>
  void WavefrontBucketingWorkspace<TExecutionSpace>::ensurePortableCapacity(
      std::size_t const requested, TExecutionSpace const& execution) {
    if (requested <= portableCapacity_) return;
    auto const grown = grownCapacity(portableCapacity_, requested);
    order_ = decltype(order_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                 "c8_kokkos_wavefront_bucket_order"),
                              grown);
    portableCapacity_ = grown;
  }

#if defined(KOKKOS_ENABLE_CUDA)
  template <class TExecutionSpace>
  void WavefrontBucketingWorkspace<TExecutionSpace>::ensureRadixCapacity(
      std::size_t const requested, TExecutionSpace const& execution) {
    if (requested <= radixCapacity_) return;
    auto const grown = grownCapacity(radixCapacity_, requested);
    if constexpr (cudaRadixSortAvailable()) {
      inputKeys_ =
          decltype(inputKeys_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                  "c8_kokkos_wavefront_input_keys"),
                               grown);
      inputIndices_ = decltype(inputIndices_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_wavefront_input_indices"),
          grown);
      outputIndices_ = decltype(outputIndices_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_wavefront_output_indices"),
          grown);
      radixTemporaryBytes_ = wavefront_bucketing_detail::radix_temporary_bytes(
          grown, execution.cuda_stream());
      radixTemporary_ = decltype(radixTemporary_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_wavefront_radix_temporary"),
          radixTemporaryBytes_);
      radixCapacity_ = grown;
    }
  }
#endif
  template <class TExecutionSpace>
  WavefrontBucketBatch<TExecutionSpace> bucket_wavefront(
      ParticleSoA<typename TExecutionSpace::memory_space> const& input,
      std::size_t const count, WavefrontBucketingWorkspace<TExecutionSpace>& workspace,
      TExecutionSpace const& execution, std::size_t const openmpChunkSize) {

    auto const inputCoversCount =
        count <= input.pid_.extent(0) && count <= input.mediumId_.extent(0) &&
        count <= input.generation_.extent(0) && count <= input.reserved_.extent(0) &&
        count <= input.environmentNode_.extent(0) &&
        count <= input.energyGeV_.extent(0) && count <= input.positionM_.extent(0) &&
        count <= input.direction_.extent(0) && count <= input.timeS_.extent(0) &&
        count <= input.weight_.extent(0) && count <= input.historyId_.extent(0) &&
        count <= input.parentHistoryId_.extent(0) && count <= input.stepId_.extent(0);
    if (!inputCoversCount)
      throw std::length_error("Kokkos wavefront bucketing exceeds an input field extent");
    if (count > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("Kokkos wavefront bucketing exceeds 32-bit source indices");

    auto const& workspaceOutput = workspace.output();
    auto const aliasesWorkspaceOutput =
        (input.pid_.data() != nullptr &&
         input.pid_.data() == workspaceOutput.pid_.data()) ||
        (input.mediumId_.data() != nullptr &&
         input.mediumId_.data() == workspaceOutput.mediumId_.data()) ||
        (input.generation_.data() != nullptr &&
         input.generation_.data() == workspaceOutput.generation_.data()) ||
        (input.reserved_.data() != nullptr &&
         input.reserved_.data() == workspaceOutput.reserved_.data()) ||
        (input.environmentNode_.data() != nullptr &&
         input.environmentNode_.data() == workspaceOutput.environmentNode_.data()) ||
        (input.energyGeV_.data() != nullptr &&
         input.energyGeV_.data() == workspaceOutput.energyGeV_.data()) ||
        (input.positionM_.data() != nullptr &&
         input.positionM_.data() == workspaceOutput.positionM_.data()) ||
        (input.direction_.data() != nullptr &&
         input.direction_.data() == workspaceOutput.direction_.data()) ||
        (input.timeS_.data() != nullptr &&
         input.timeS_.data() == workspaceOutput.timeS_.data()) ||
        (input.weight_.data() != nullptr &&
         input.weight_.data() == workspaceOutput.weight_.data()) ||
        (input.historyId_.data() != nullptr &&
         input.historyId_.data() == workspaceOutput.historyId_.data()) ||
        (input.parentHistoryId_.data() != nullptr &&
         input.parentHistoryId_.data() == workspaceOutput.parentHistoryId_.data()) ||
        (input.stepId_.data() != nullptr &&
         input.stepId_.data() == workspaceOutput.stepId_.data());
    if (aliasesWorkspaceOutput)
      throw std::invalid_argument(
          "Kokkos wavefront input aliases the bucketing workspace output");

    if (count < MinimumWavefrontRadixSortSize) return {input, {}, count, false};

    workspace.ensureCapacity(count, execution);
    WavefrontBucketBatch<TExecutionSpace> result{workspace.output(), workspace.keys(),
                                                 count, true};
    // A ParticleSoA contains twelve owning Kokkos::View objects.  Capturing
    // both input and output SoAs by value makes the CUDA functor larger than
    // Kokkos' constant-memory launch threshold.  Kokkos then serializes every
    // sorted front through a shared constant buffer and an event wait.  The
    // raw views keep exactly the same data/strides while the caller retains the
    // input and workspace owners, and make each launch self-contained and
    // stream ordered.
    auto const inputRaw = input.rawDeviceView();
    auto const outputRaw = result.particles_.rawDeviceView();

#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (WavefrontBucketingWorkspace<
                      TExecutionSpace>::cudaRadixSortAvailable()) {
      if (WavefrontBucketingWorkspace<TExecutionSpace>::usesCudaRadixSort(count)) {
        auto* const inputKeys = workspace.inputKeys().data();
        auto* const inputIndices = workspace.inputIndices().data();
        auto const makeRadixKeys = KOKKOS_LAMBDA(std::size_t const source) {
          inputKeys[source] = wavefront_bucket_key(inputRaw.load(source));
          inputIndices[source] = static_cast<std::uint32_t>(source);
        };
        static_assert(sizeof(makeRadixKeys) < 512);
        Kokkos::parallel_for("c8_kokkos_make_wavefront_radix_keys",
                             make_range_policy(execution, count, openmpChunkSize),
                             makeRadixKeys);

        auto temporaryBytes = workspace.radixTemporaryBytes();
        wavefront_bucketing_detail::check_cuda(
            cub::DeviceRadixSort::SortPairs(
                workspace.radixTemporary().data(), temporaryBytes,
                workspace.inputKeys().data(), workspace.keys().data(),
                workspace.inputIndices().data(), workspace.outputIndices().data(), count,
                0, WavefrontBucketKeyBits, execution.cuda_stream()),
            "Kokkos-CUDA stable wavefront radix sort");

        auto const* const outputIndices = workspace.outputIndices().data();
        auto const gatherRadix = KOKKOS_LAMBDA(std::size_t const destination) {
          outputRaw.store(destination, inputRaw.load(outputIndices[destination]));
        };
        static_assert(sizeof(gatherRadix) < 512);
        Kokkos::parallel_for("c8_kokkos_gather_radix_bucketed_wavefront",
                             make_range_policy(execution, count, openmpChunkSize),
                             gatherRadix);
        return result;
      }
    }
#endif

    auto const activeOrder =
        Kokkos::subview(workspace.order(), std::make_pair<std::size_t>(0, count));
    auto const makeBucketOrder = KOKKOS_LAMBDA(std::size_t const source) {
      activeOrder(source) = {wavefront_bucket_key(inputRaw.load(source)),
                             static_cast<std::uint32_t>(source), 0};
    };
    static_assert(sizeof(makeBucketOrder) < 512);
    Kokkos::parallel_for("c8_kokkos_make_wavefront_bucket_order",
                         make_range_policy(execution, count, openmpChunkSize),
                         makeBucketOrder);

    Kokkos::sort(execution, activeOrder, wavefront_bucketing_detail::BucketOrderLess{});

    auto* const keys = result.keys_.data();
    auto const gatherBucketed = KOKKOS_LAMBDA(std::size_t const destination) {
      auto const selected = activeOrder(destination);
      outputRaw.store(destination, inputRaw.load(selected.source_));
      keys[destination] = selected.bucketKey_;
    };
    static_assert(sizeof(gatherBucketed) < 512);
    Kokkos::parallel_for("c8_kokkos_gather_bucketed_wavefront",
                         make_range_policy(execution, count, openmpChunkSize),
                         gatherBucketed);
    return result;
  }

} // namespace corsika::kokkos
