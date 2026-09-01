/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <corsika/gpu/em/tables/CudaProposalNativeTable.hpp>

namespace corsika::gpu::em::tables {
  namespace {
    constexpr unsigned int ThreadsPerBlock = 256;

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) return;
      std::ostringstream message;
      message << operation << " failed: " << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    __device__ NativeQueryResult executeNativeQuery(
        ProposalNativeDeviceView const& view,
        ProposalNativeQuery const& query) {
      switch (query.kind) {
      case ProposalNativeQueryKind::Rate:
        return queryProposalNativeRate(
            view, query.pdg_id, query.process_id,
            query.component_hash, query.energy_MeV);
      case ProposalNativeQueryKind::TotalRate:
        return queryProposalNativeTotalRate(
            view, query.pdg_id, query.energy_MeV);
      case ProposalNativeQueryKind::CumulativeRate: {
        auto const* column = native_detail::findDndx(
            view, query.pdg_id, query.process_id,
            query.component_hash);
        if (!column)
          return {NativeQueryStatus::ColumnNotFound, 0, 0, 0.};
        return queryProposalNativeCumulativeRate(
            view, *column, query.energy_MeV, query.argument);
      }
      case ProposalNativeQueryKind::LossFraction:
        return queryProposalNativeLossFraction(
            view, query.pdg_id, query.process_id,
            query.component_hash, query.energy_MeV,
            query.argument);
      case ProposalNativeQueryKind::ContinuousDedx:
        return queryProposalNativeDedx(
            view, query.pdg_id, query.energy_MeV);
      case ProposalNativeQueryKind::ContinuousRange:
        return queryProposalNativeRange(
            view, query.pdg_id, query.energy_MeV);
      case ProposalNativeQueryKind::ContinuousEnergy:
        return queryProposalNativeEnergy(
            view, query.pdg_id, query.argument);
      case ProposalNativeQueryKind::ContinuousEnergyAfterLoss:
        return queryProposalNativeEnergyAfterContinuousLoss(
            view, query.pdg_id, query.energy_MeV, query.argument);
      }
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    }

    __global__ void queryNativeTableKernel(
        ProposalNativeDeviceView view,
        ProposalNativeQuery const* queries,
        NativeQueryResult* results, std::size_t count) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index < count)
        results[index] = executeNativeQuery(view, queries[index]);
    }

    __device__ ProposalNativeSelectionResult executeNativeSelection(
        ProposalNativeDeviceView const& view,
        ProposalNativeSelectionQuery const& query) {
      if (!native_detail::validView(view))
        return {NativeQueryStatus::InvalidView, 0, 0, 0., 0., 0., 0};
      if (!native_detail::finite(query.energy_MeV) ||
          !native_detail::finite(query.threshold) ||
          query.threshold < 0.)
        return {NativeQueryStatus::NonFiniteInput, 0, 0, 0., 0., 0., 0};
      auto const range =
          native_detail::dndxParticleRange(view, query.pdg_id);
      if (range.begin == range.end)
        return {NativeQueryStatus::ParticleNotFound, 0, 0, 0., 0., 0., 0};
      double cumulative = 0.;
      double boundary_cumulative = 0.;
      std::int32_t process = 0;
      std::uint64_t component = 0;
      double residual_quantile = 0.;
      bool selected = false;
      for (auto index = range.begin; index < range.end; ++index) {
        auto const* selected_column =
            native_detail::selectionDndx(view, index);
        if (!selected_column || selected_column->pdg_id != query.pdg_id)
          return {NativeQueryStatus::InvalidView, 0, 0, cumulative,
                  boundary_cumulative, 0., 0};
        auto const& column = *selected_column;
        NativeQueryResult rate{NativeQueryStatus::Success, 0, 0, 0.};
        if (!proposalNativeRateBelowThreshold(column, query.energy_MeV))
          rate = queryProposalNativeRateForColumn(
              view, column, query.energy_MeV);
        if (rate.status != NativeQueryStatus::Success)
          return {rate.status, column.process_id, column.component_hash,
                  cumulative, boundary_cumulative, 0., 0};
        cumulative += rate.value;
        if (column.process_id == query.boundary_process_id &&
            column.component_hash == query.boundary_component_hash)
          boundary_cumulative = cumulative;
        if (!selected && query.threshold < cumulative) {
          selected = true;
          process = column.process_id;
          component = column.component_hash;
          residual_quantile =
              rate.value > 0. ? (cumulative - query.threshold) / rate.value
                              : 0.;
        }
      }
      return {NativeQueryStatus::Success, process, component, cumulative,
              boundary_cumulative, residual_quantile,
              selected ? 1u : 0u};
    }

    __global__ void selectNativeTableKernel(
        ProposalNativeDeviceView view,
        ProposalNativeSelectionQuery const* queries,
        ProposalNativeSelectionResult* results, std::size_t count) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index < count)
        results[index] = executeNativeSelection(view, queries[index]);
    }
  } // namespace

  class CudaProposalNativeTable::Impl {
  public:
    ~Impl() { reset(); }

    void initialize(ProposalNativeTableSet const& source, int device,
                    std::size_t maximum_device_bytes) {
      if (initialized_)
        throw std::logic_error(
            "CUDA PROPOSAL native table is already initialized");
      if (device < 0)
        throw std::invalid_argument(
            "CUDA PROPOSAL native-table device must be non-negative");
      validateProposalNativeTable(source);
      auto const calculated_hash = calculateProposalNativeHash(source);
      if (calculated_hash != source.content_hash)
        throw std::invalid_argument(
            "PROPOSAL native-table content hash does not match its "
            "coefficient payload");
      auto const requested = proposalNativeTableBytes(source);
      if (requested > maximum_device_bytes)
        throw std::runtime_error(
            "PROPOSAL native table exceeds the configured device budget");

      checkCuda(cudaSetDevice(device),
                "cudaSetDevice(PROPOSAL native table)");
      device_selected_ = true;
      device_ = device;
      int count = 0;
      checkCuda(cudaGetDeviceCount(&count),
                "cudaGetDeviceCount(PROPOSAL native table)");
      if (device >= count) {
        reset();
        throw std::invalid_argument(
            "requested CUDA PROPOSAL native-table device does not exist");
      }
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes),
                "cudaMemGetInfo(PROPOSAL native table)");
      (void)total_bytes;
      if (requested > free_bytes) {
        reset();
        throw std::runtime_error(
            "insufficient free CUDA memory for PROPOSAL native table");
      }

      try {
        upload(source.dndx_columns, dndx_, "dN/dX descriptors");
        upload(source.total_rate_columns, total_rate_,
               "mean-free-path rate descriptors");
        upload(source.selection_column_indices, selection_indices_,
               "selection-order index");
        upload(source.dedx_columns, dedx_, "dE/dX descriptors");
        upload(source.dedx_accumulation_indices,
               dedx_accumulation_indices_,
               "dE/dX accumulation-order index");
        upload(source.utility_columns, utility_, "utility descriptors");
        upload(source.bicubic_values, bicubic_values_, "bicubic values");
        upload(source.bicubic_derivative_energy, bicubic_denergy_,
               "bicubic energy derivatives");
        upload(source.bicubic_derivative_loss, bicubic_dloss_,
               "bicubic loss derivatives");
        upload(source.bicubic_mixed_derivative, bicubic_mixed_,
               "bicubic mixed derivatives");
        upload(source.bicubic_polynomial_coefficients,
               bicubic_polynomial_coefficients_,
               "bicubic polynomial coefficients");
        upload(source.cubic_values, cubic_values_, "cubic values");
        upload(source.cubic_node_derivatives, cubic_derivatives_,
               "cubic node derivatives");
        view_ = {
            dndx_, total_rate_, selection_indices_, dedx_,
            dedx_accumulation_indices_, utility_,
            static_cast<std::uint32_t>(source.dndx_columns.size()),
            static_cast<std::uint32_t>(source.total_rate_columns.size()),
            static_cast<std::uint32_t>(
                source.selection_column_indices.size()),
            static_cast<std::uint32_t>(source.dedx_columns.size()),
            static_cast<std::uint32_t>(
                source.dedx_accumulation_indices.size()),
            static_cast<std::uint32_t>(source.utility_columns.size()),
            bicubic_values_, bicubic_denergy_, bicubic_dloss_,
            bicubic_mixed_, source.bicubic_values.size(),
            bicubic_polynomial_coefficients_,
            source.bicubic_polynomial_coefficients.size(),
            cubic_values_, cubic_derivatives_, source.cubic_values.size(),
            source.constants};
        hash_ = source.content_hash;
        bytes_ = requested;
        initialized_ = true;
      } catch (...) {
        reset();
        throw;
      }
    }

    void reset() noexcept {
      if (device_selected_) cudaSetDevice(device_);
      freeDevice(cubic_derivatives_);
      freeDevice(cubic_values_);
      freeDevice(bicubic_polynomial_coefficients_);
      freeDevice(bicubic_mixed_);
      freeDevice(bicubic_dloss_);
      freeDevice(bicubic_denergy_);
      freeDevice(bicubic_values_);
      freeDevice(utility_);
      freeDevice(dedx_accumulation_indices_);
      freeDevice(dedx_);
      freeDevice(selection_indices_);
      freeDevice(total_rate_);
      freeDevice(dndx_);
      view_ = {};
      hash_ = {};
      bytes_ = 0;
      initialized_ = false;
      device_selected_ = false;
      device_ = 0;
    }

    bool initialized() const noexcept { return initialized_; }
    std::size_t deviceBytes() const noexcept { return bytes_; }
    Sha256Digest const& sourceContentHash() const {
      requireInitialized();
      return hash_;
    }
    ProposalNativeDeviceView deviceView() const {
      requireInitialized();
      return view_;
    }

    std::vector<NativeQueryResult> queryForValidation(
        std::vector<ProposalNativeQuery> const& queries) const {
      requireInitialized();
      std::vector<NativeQueryResult> output(queries.size());
      if (queries.empty()) return output;
      if (queries.size() >
              std::numeric_limits<std::size_t>::max() /
                  sizeof(ProposalNativeQuery) ||
          queries.size() >
              std::numeric_limits<std::size_t>::max() /
                  sizeof(NativeQueryResult))
        throw std::length_error(
            "PROPOSAL native validation batch is too large");
      auto const blocks =
          (queries.size() + ThreadsPerBlock - 1) / ThreadsPerBlock;
      if (blocks > std::numeric_limits<unsigned int>::max())
        throw std::length_error(
            "PROPOSAL native validation launch is too large");

      checkCuda(cudaSetDevice(device_),
                "cudaSetDevice(PROPOSAL native validation)");
      ProposalNativeQuery* device_queries = nullptr;
      NativeQueryResult* device_results = nullptr;
      try {
        checkCuda(cudaMalloc(
                      reinterpret_cast<void**>(&device_queries),
                      queries.size() * sizeof(ProposalNativeQuery)),
                  "allocate PROPOSAL native validation queries");
        checkCuda(cudaMalloc(
                      reinterpret_cast<void**>(&device_results),
                      queries.size() * sizeof(NativeQueryResult)),
                  "allocate PROPOSAL native validation results");
        checkCuda(cudaMemcpy(
                      device_queries, queries.data(),
                      queries.size() * sizeof(ProposalNativeQuery),
                      cudaMemcpyHostToDevice),
                  "upload PROPOSAL native validation queries");
        queryNativeTableKernel<<<static_cast<unsigned int>(blocks),
                                 ThreadsPerBlock>>>(
            view_, device_queries, device_results, queries.size());
        checkCuda(cudaGetLastError(),
                  "launch PROPOSAL native validation kernel");
        checkCuda(cudaMemcpy(
                      output.data(), device_results,
                      output.size() * sizeof(NativeQueryResult),
                      cudaMemcpyDeviceToHost),
                  "download PROPOSAL native validation results");
      } catch (...) {
        cudaFree(device_results);
        cudaFree(device_queries);
        throw;
      }
      cudaFree(device_results);
      cudaFree(device_queries);
      return output;
    }

    std::vector<ProposalNativeSelectionResult> selectForValidation(
        std::vector<ProposalNativeSelectionQuery> const& queries) const {
      requireInitialized();
      std::vector<ProposalNativeSelectionResult> output(queries.size());
      if (queries.empty()) return output;
      if (queries.size() >
              std::numeric_limits<std::size_t>::max() /
                  sizeof(ProposalNativeSelectionQuery) ||
          queries.size() >
              std::numeric_limits<std::size_t>::max() /
                  sizeof(ProposalNativeSelectionResult))
        throw std::length_error(
            "PROPOSAL native selection-validation batch is too large");
      auto const blocks =
          (queries.size() + ThreadsPerBlock - 1) / ThreadsPerBlock;
      if (blocks > std::numeric_limits<unsigned int>::max())
        throw std::length_error(
            "PROPOSAL native selection-validation launch is too large");

      checkCuda(cudaSetDevice(device_),
                "cudaSetDevice(PROPOSAL native selection validation)");
      ProposalNativeSelectionQuery* device_queries = nullptr;
      ProposalNativeSelectionResult* device_results = nullptr;
      try {
        checkCuda(cudaMalloc(
                      reinterpret_cast<void**>(&device_queries),
                      queries.size() * sizeof(*device_queries)),
                  "allocate PROPOSAL native selection queries");
        checkCuda(cudaMalloc(
                      reinterpret_cast<void**>(&device_results),
                      queries.size() * sizeof(*device_results)),
                  "allocate PROPOSAL native selection results");
        checkCuda(cudaMemcpy(
                      device_queries, queries.data(),
                      queries.size() * sizeof(*device_queries),
                      cudaMemcpyHostToDevice),
                  "upload PROPOSAL native selection queries");
        selectNativeTableKernel<<<static_cast<unsigned int>(blocks),
                                  ThreadsPerBlock>>>(
            view_, device_queries, device_results, queries.size());
        checkCuda(cudaGetLastError(),
                  "launch PROPOSAL native selection-validation kernel");
        checkCuda(cudaMemcpy(
                      output.data(), device_results,
                      output.size() * sizeof(*device_results),
                      cudaMemcpyDeviceToHost),
                  "download PROPOSAL native selection results");
      } catch (...) {
        cudaFree(device_results);
        cudaFree(device_queries);
        throw;
      }
      cudaFree(device_results);
      cudaFree(device_queries);
      return output;
    }

  private:
    template <typename T>
    static void upload(std::vector<T> const& source, T*& destination,
                       char const* name) {
      if (source.empty()) {
        destination = nullptr;
        return;
      }
      std::ostringstream allocation;
      allocation << "allocate PROPOSAL native " << name;
      checkCuda(cudaMalloc(
                    reinterpret_cast<void**>(&destination),
                    source.size() * sizeof(T)),
                allocation.str().c_str());
      std::ostringstream copy;
      copy << "upload PROPOSAL native " << name;
      checkCuda(cudaMemcpy(
                    destination, source.data(), source.size() * sizeof(T),
                    cudaMemcpyHostToDevice),
                copy.str().c_str());
    }

    template <typename T>
    static void freeDevice(T*& pointer) noexcept {
      if (pointer) {
        cudaFree(pointer);
        pointer = nullptr;
      }
    }

    void requireInitialized() const {
      if (!initialized_)
        throw std::logic_error(
            "CUDA PROPOSAL native table is not initialized");
    }

    bool initialized_{};
    bool device_selected_{};
    int device_{};
    std::size_t bytes_{};
    Sha256Digest hash_{};
    ProposalNativeDeviceView view_{};
    NativeDndxColumn* dndx_{};
    NativeTotalRateColumn* total_rate_{};
    std::uint32_t* selection_indices_{};
    NativeDedxColumn* dedx_{};
    std::uint32_t* dedx_accumulation_indices_{};
    NativeUtilityColumn* utility_{};
    double* bicubic_values_{};
    double* bicubic_denergy_{};
    double* bicubic_dloss_{};
    double* bicubic_mixed_{};
    double* bicubic_polynomial_coefficients_{};
    double* cubic_values_{};
    double* cubic_derivatives_{};
  };

  CudaProposalNativeTable::CudaProposalNativeTable()
      : impl_(std::make_unique<Impl>()) {}
  CudaProposalNativeTable::~CudaProposalNativeTable() = default;
  CudaProposalNativeTable::CudaProposalNativeTable(
      CudaProposalNativeTable&&) noexcept = default;
  CudaProposalNativeTable& CudaProposalNativeTable::operator=(
      CudaProposalNativeTable&&) noexcept = default;

  void CudaProposalNativeTable::initialize(
      ProposalNativeTableSet const& source, int device,
      std::size_t maximum_device_bytes) {
    impl_->initialize(source, device, maximum_device_bytes);
  }
  void CudaProposalNativeTable::reset() noexcept { impl_->reset(); }
  bool CudaProposalNativeTable::initialized() const noexcept {
    return impl_->initialized();
  }
  std::size_t CudaProposalNativeTable::deviceBytes() const noexcept {
    return impl_->deviceBytes();
  }
  Sha256Digest const& CudaProposalNativeTable::sourceContentHash() const {
    return impl_->sourceContentHash();
  }
  ProposalNativeDeviceView CudaProposalNativeTable::deviceView() const {
    return impl_->deviceView();
  }
  std::vector<NativeQueryResult>
  CudaProposalNativeTable::queryForValidation(
      std::vector<ProposalNativeQuery> const& queries) const {
    return impl_->queryForValidation(queries);
  }
  std::vector<ProposalNativeSelectionResult>
  CudaProposalNativeTable::selectForValidation(
      std::vector<ProposalNativeSelectionQuery> const& queries) const {
    return impl_->selectForValidation(queries);
  }
} // namespace corsika::gpu::em::tables
