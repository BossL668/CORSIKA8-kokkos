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

#include <corsika/gpu/em/tables/CudaRateTable.hpp>

namespace corsika::gpu::em::tables {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) {
        return;
      }
      std::ostringstream message;
      message << operation << " failed: " << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    __global__ void queryRateTableKernel(
        FlatRateTableView view, TableQuery const* queries,
        TableQueryResult* results, std::size_t count) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index < count) {
        results[index] = executeTableQuery(view, queries[index]);
      }
    }

  } // namespace

  class CudaRateTable::Impl {
  public:
    ~Impl() { reset(); }

    void initialize(RateTableSet const& source, int device,
                    std::size_t maximum_device_bytes) {
      if (initialized_) {
        throw std::logic_error(
            "CUDA rate table is already initialized");
      }
      if (device < 0) {
        throw std::invalid_argument(
            "CUDA rate-table device index must be non-negative");
      }
      auto const flat = flattenRateTable(source);
      auto const requested_bytes = flatRateTableBytes(flat);
      if (requested_bytes > maximum_device_bytes) {
        throw std::runtime_error(
            "flat rate table exceeds its configured device-memory budget");
      }

      checkCuda(cudaSetDevice(device), "cudaSetDevice(rate table)");
      device_selected_ = true;
      device_ = device;
      int device_count = 0;
      checkCuda(cudaGetDeviceCount(&device_count),
                "cudaGetDeviceCount(rate table)");
      if (device >= device_count) {
        reset();
        throw std::invalid_argument(
            "requested CUDA rate-table device does not exist");
      }
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes),
                "cudaMemGetInfo(rate table)");
      (void)total_bytes;
      if (requested_bytes > free_bytes) {
        reset();
        throw std::runtime_error(
            "insufficient free CUDA memory for the rate table");
      }

      try {
        upload(flat.particle_pdg_ids, particle_pdg_ids_,
               "particle PDG IDs");
        upload(flat.particle_energy_offsets, particle_energy_offsets_,
               "particle energy offsets");
        upload(flat.particle_energy_counts, particle_energy_counts_,
               "particle energy counts");
        upload(flat.particle_column_offsets, particle_column_offsets_,
               "particle column offsets");
        upload(flat.particle_column_counts, particle_column_counts_,
               "particle column counts");
        upload(flat.column_process_ids, column_process_ids_,
               "column process IDs");
        upload(flat.column_component_hashes, column_component_hashes_,
               "column component hashes");
        upload(flat.column_rate_offsets, column_rate_offsets_,
               "column rate offsets");
        upload(flat.column_inverse_energy_offsets,
               column_inverse_energy_offsets_,
               "column inverse-energy offsets");
        upload(flat.column_inverse_energy_counts,
               column_inverse_energy_counts_,
               "column inverse-energy counts");
        upload(flat.column_inverse_row_offsets,
               column_inverse_row_offsets_,
               "column inverse-row offsets");
        upload(flat.rate_energies_MeV, rate_energies_MeV_,
               "rate energies");
        upload(flat.rates_cm2_per_g, rates_cm2_per_g_, "rates");
        upload(flat.inverse_energies_MeV, inverse_energies_MeV_,
               "inverse-CDF energies");
        upload(flat.inverse_row_offsets, inverse_row_offsets_,
               "inverse-CDF row offsets");
        upload(flat.inverse_quantiles, inverse_quantiles_,
               "inverse-CDF quantiles");
        upload(flat.inverse_v_loss, inverse_v_loss_,
               "inverse-CDF values");
        upload(flat.continuous_pdg_ids, continuous_pdg_ids_,
               "continuous particle PDG IDs");
        upload(flat.continuous_energy_offsets,
               continuous_energy_offsets_,
               "continuous energy offsets");
        upload(flat.continuous_energy_counts,
               continuous_energy_counts_,
               "continuous energy counts");
        upload(flat.continuous_masses_MeV,
               continuous_masses_MeV_,
               "continuous particle masses");
        upload(flat.continuous_minimum_energies_MeV,
               continuous_minimum_energies_MeV_,
               "continuous minimum energies");
        upload(flat.continuous_energies_MeV,
               continuous_energies_MeV_,
               "continuous energies");
        upload(flat.continuous_dEdX_MeV_cm2_per_g,
               continuous_dEdX_MeV_cm2_per_g_,
               "continuous energy losses");
        upload(flat.continuous_ranges_g_per_cm2,
               continuous_ranges_g_per_cm2_,
               "continuous ranges");
        upload(
            flat.epair_rho_component_hashes,
            epair_rho_component_hashes_,
            "Epair rho component hashes");
        upload(
            flat.epair_rho_energies_MeV,
            epair_rho_energies_MeV_,
            "Epair rho energies");
        upload(
            flat.epair_rho_v_coordinates,
            epair_rho_v_coordinates_,
            "Epair rho v coordinates");
        upload(
            flat.epair_rho_quantiles,
            epair_rho_quantiles_,
            "Epair rho quantiles");
        upload(
            flat.epair_rho_values, epair_rho_values_,
            "Epair rho inverse-CDF values");

        auto const host_view = makeFlatRateTableView(flat);
        device_view_ = {
            particle_pdg_ids_,
            particle_energy_offsets_,
            particle_energy_counts_,
            particle_column_offsets_,
            particle_column_counts_,
            host_view.particle_count,
            column_process_ids_,
            column_component_hashes_,
            column_rate_offsets_,
            column_inverse_energy_offsets_,
            column_inverse_energy_counts_,
            column_inverse_row_offsets_,
            host_view.column_count,
            rate_energies_MeV_,
            rates_cm2_per_g_,
            inverse_energies_MeV_,
            inverse_row_offsets_,
            inverse_quantiles_,
            inverse_v_loss_,
            host_view.rate_energy_count,
            host_view.rate_value_count,
            host_view.inverse_energy_count,
            host_view.inverse_row_offset_count,
            host_view.inverse_value_count,
            continuous_pdg_ids_,
            continuous_energy_offsets_,
            continuous_energy_counts_,
            continuous_masses_MeV_,
            continuous_minimum_energies_MeV_,
            host_view.continuous_particle_count,
            continuous_energies_MeV_,
            continuous_dEdX_MeV_cm2_per_g_,
            continuous_ranges_g_per_cm2_,
            host_view.continuous_value_count,
            host_view.energy_cut_MeV,
            epair_rho_component_hashes_,
            epair_rho_energies_MeV_,
            epair_rho_v_coordinates_,
            epair_rho_quantiles_,
            epair_rho_values_,
            host_view.epair_rho_component_count,
            host_view.epair_rho_energy_count,
            host_view.epair_rho_v_coordinate_count,
            host_view.epair_rho_quantile_count,
            host_view.epair_rho_value_count};
        source_hash_ = flat.content_hash;
        device_bytes_ = requested_bytes;
        initialized_ = true;
      } catch (...) {
        reset();
        throw;
      }
    }

    void reset() noexcept {
      if (device_selected_) {
        cudaSetDevice(device_);
      }
      freeDevice(epair_rho_values_);
      freeDevice(epair_rho_quantiles_);
      freeDevice(epair_rho_v_coordinates_);
      freeDevice(epair_rho_energies_MeV_);
      freeDevice(epair_rho_component_hashes_);
      freeDevice(continuous_ranges_g_per_cm2_);
      freeDevice(continuous_dEdX_MeV_cm2_per_g_);
      freeDevice(continuous_energies_MeV_);
      freeDevice(continuous_minimum_energies_MeV_);
      freeDevice(continuous_masses_MeV_);
      freeDevice(continuous_energy_counts_);
      freeDevice(continuous_energy_offsets_);
      freeDevice(continuous_pdg_ids_);
      freeDevice(inverse_v_loss_);
      freeDevice(inverse_quantiles_);
      freeDevice(inverse_row_offsets_);
      freeDevice(inverse_energies_MeV_);
      freeDevice(rates_cm2_per_g_);
      freeDevice(rate_energies_MeV_);
      freeDevice(column_inverse_row_offsets_);
      freeDevice(column_inverse_energy_counts_);
      freeDevice(column_inverse_energy_offsets_);
      freeDevice(column_rate_offsets_);
      freeDevice(column_component_hashes_);
      freeDevice(column_process_ids_);
      freeDevice(particle_column_counts_);
      freeDevice(particle_column_offsets_);
      freeDevice(particle_energy_counts_);
      freeDevice(particle_energy_offsets_);
      freeDevice(particle_pdg_ids_);
      device_view_ = {};
      source_hash_ = {};
      device_bytes_ = 0;
      initialized_ = false;
      device_selected_ = false;
      device_ = 0;
    }

    bool initialized() const noexcept { return initialized_; }
    std::size_t deviceBytes() const noexcept { return device_bytes_; }

    Sha256Digest const& sourceContentHash() const {
      requireInitialized();
      return source_hash_;
    }

    FlatRateTableView deviceView() const {
      requireInitialized();
      return device_view_;
    }

    std::vector<TableQueryResult> queryForValidation(
        std::vector<TableQuery> const& queries) const {
      requireInitialized();
      std::vector<TableQueryResult> results(queries.size());
      if (queries.empty()) {
        return results;
      }
      if (queries.size() >
              std::numeric_limits<std::size_t>::max() /
                  sizeof(TableQuery) ||
          queries.size() >
              std::numeric_limits<std::size_t>::max() /
                  sizeof(TableQueryResult)) {
        throw std::length_error(
            "CUDA rate-table validation batch is too large");
      }
      auto const block_count =
          (queries.size() + ThreadsPerBlock - 1) /
          ThreadsPerBlock;
      if (block_count >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "CUDA rate-table validation launch is too large");
      }
      checkCuda(cudaSetDevice(device_),
                "cudaSetDevice(rate-table validation)");
      TableQuery* device_queries = nullptr;
      TableQueryResult* device_results = nullptr;
      try {
        checkCuda(cudaMalloc(
                      reinterpret_cast<void**>(&device_queries),
                      queries.size() * sizeof(TableQuery)),
                  "allocate rate-table validation queries");
        checkCuda(cudaMalloc(
                      reinterpret_cast<void**>(&device_results),
                      queries.size() * sizeof(TableQueryResult)),
                  "allocate rate-table validation results");
        checkCuda(cudaMemcpy(
                      device_queries, queries.data(),
                      queries.size() * sizeof(TableQuery),
                      cudaMemcpyHostToDevice),
                  "upload rate-table validation queries");
        queryRateTableKernel<<<static_cast<unsigned int>(block_count),
                               ThreadsPerBlock>>>(
            device_view_, device_queries, device_results,
            queries.size());
        checkCuda(cudaGetLastError(),
                  "rate-table validation kernel launch");
        checkCuda(cudaMemcpy(
                      results.data(), device_results,
                      results.size() * sizeof(TableQueryResult),
                      cudaMemcpyDeviceToHost),
                  "download rate-table validation results");
      } catch (...) {
        cudaFree(device_results);
        cudaFree(device_queries);
        throw;
      }
      cudaFree(device_results);
      cudaFree(device_queries);
      return results;
    }

  private:
    template <typename T>
    static void upload(std::vector<T> const& source, T*& destination,
                       char const* name) {
      if (source.empty()) {
        destination = nullptr;
        return;
      }
      std::ostringstream allocate_operation;
      allocate_operation << "allocate CUDA rate-table " << name;
      checkCuda(cudaMalloc(
                    reinterpret_cast<void**>(&destination),
                    source.size() * sizeof(T)),
                allocate_operation.str().c_str());
      std::ostringstream copy_operation;
      copy_operation << "upload CUDA rate-table " << name;
      checkCuda(cudaMemcpy(
                    destination, source.data(),
                    source.size() * sizeof(T), cudaMemcpyHostToDevice),
                copy_operation.str().c_str());
    }

    template <typename T>
    static void freeDevice(T*& pointer) noexcept {
      if (pointer != nullptr) {
        cudaFree(pointer);
        pointer = nullptr;
      }
    }

    void requireInitialized() const {
      if (!initialized_) {
        throw std::logic_error(
            "CUDA rate table is not initialized");
      }
    }

    bool initialized_{};
    bool device_selected_{};
    int device_{};
    std::size_t device_bytes_{};
    Sha256Digest source_hash_{};
    FlatRateTableView device_view_{};

    std::int32_t* particle_pdg_ids_{};
    std::uint32_t* particle_energy_offsets_{};
    std::uint32_t* particle_energy_counts_{};
    std::uint32_t* particle_column_offsets_{};
    std::uint32_t* particle_column_counts_{};
    std::int32_t* column_process_ids_{};
    std::uint64_t* column_component_hashes_{};
    std::uint32_t* column_rate_offsets_{};
    std::uint32_t* column_inverse_energy_offsets_{};
    std::uint32_t* column_inverse_energy_counts_{};
    std::uint32_t* column_inverse_row_offsets_{};
    double* rate_energies_MeV_{};
    double* rates_cm2_per_g_{};
    double* inverse_energies_MeV_{};
    std::uint32_t* inverse_row_offsets_{};
    double* inverse_quantiles_{};
    double* inverse_v_loss_{};
    std::int32_t* continuous_pdg_ids_{};
    std::uint32_t* continuous_energy_offsets_{};
    std::uint32_t* continuous_energy_counts_{};
    double* continuous_masses_MeV_{};
    double* continuous_minimum_energies_MeV_{};
    double* continuous_energies_MeV_{};
    double* continuous_dEdX_MeV_cm2_per_g_{};
    double* continuous_ranges_g_per_cm2_{};
    std::uint64_t* epair_rho_component_hashes_{};
    double* epair_rho_energies_MeV_{};
    double* epair_rho_v_coordinates_{};
    double* epair_rho_quantiles_{};
    double* epair_rho_values_{};
  };

  CudaRateTable::CudaRateTable() : impl_(std::make_unique<Impl>()) {}
  CudaRateTable::~CudaRateTable() = default;
  CudaRateTable::CudaRateTable(CudaRateTable&&) noexcept = default;
  CudaRateTable& CudaRateTable::operator=(
      CudaRateTable&&) noexcept = default;

  void CudaRateTable::initialize(
      RateTableSet const& source, int device,
      std::size_t maximum_device_bytes) {
    impl_->initialize(source, device, maximum_device_bytes);
  }

  void CudaRateTable::reset() noexcept { impl_->reset(); }

  bool CudaRateTable::initialized() const noexcept {
    return impl_->initialized();
  }

  std::size_t CudaRateTable::deviceBytes() const noexcept {
    return impl_->deviceBytes();
  }

  Sha256Digest const& CudaRateTable::sourceContentHash() const {
    return impl_->sourceContentHash();
  }

  FlatRateTableView CudaRateTable::deviceView() const {
    return impl_->deviceView();
  }

  std::vector<TableQueryResult> CudaRateTable::queryForValidation(
      std::vector<TableQuery> const& queries) const {
    return impl_->queryForValidation(queries);
  }

} // namespace corsika::gpu::em::tables
