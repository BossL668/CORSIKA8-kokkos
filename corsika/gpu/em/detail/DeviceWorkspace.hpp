/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace corsika::gpu::em::detail {

  inline std::size_t checkedWorkspaceAdd(
      std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
      throw std::overflow_error("CUDA workspace size overflow");
    }
    return left + right;
  }

  inline std::size_t alignedWorkspaceOffset(
      std::size_t offset, std::size_t alignment) {
    auto const remainder = offset % alignment;
    return remainder == 0
               ? offset
               : checkedWorkspaceAdd(offset, alignment - remainder);
  }

  class WorkspaceSize {
  public:
    template <typename T>
    void add(std::size_t count) {
      if (count > std::numeric_limits<std::size_t>::max() /
                      sizeof(T)) {
        throw std::overflow_error("CUDA workspace array size overflow");
      }
      addBytes(count * sizeof(T), alignof(T));
    }

    void addBytes(std::size_t bytes,
                  std::size_t alignment = 256) {
      bytes_ = alignedWorkspaceOffset(bytes_, alignment);
      bytes_ = checkedWorkspaceAdd(bytes_, bytes);
    }

    std::size_t bytes() const { return bytes_; }

  private:
    std::size_t bytes_{};
  };

  /**
   * One reusable device allocation shared by sequential physical batch stages.
   *
   * prepare() may grow the arena but acquire() never allocates. A backend can
   * therefore pre-size it before shower transport and reset it between
   * selection, tracking and final-state kernels.
   */
  class DeviceWorkspace {
  public:
    DeviceWorkspace() = default;
    ~DeviceWorkspace() { release(); }

    DeviceWorkspace(DeviceWorkspace const&) = delete;
    DeviceWorkspace& operator=(DeviceWorkspace const&) = delete;

    void configure(int device, std::size_t byte_limit) {
      if (device < 0 || byte_limit == 0) {
        throw std::invalid_argument(
            "invalid CUDA workspace configuration");
      }
      device_ = device;
      byte_limit_ = byte_limit;
      configured_ = true;
    }

    void prepare(std::size_t required_bytes) {
      requireConfigured();
      if (required_bytes > byte_limit_) {
        throw std::runtime_error(
            "CUDA physical workspace exceeds configured memory budget");
      }
      if (required_bytes > capacity_bytes_) {
        auto target = std::max<std::size_t>(capacity_bytes_, 4096);
        while (target < required_bytes) {
          if (target > byte_limit_ / 2) {
            target = byte_limit_;
            break;
          }
          target *= 2;
        }
        void* replacement = nullptr;
        auto const status = cudaMalloc(&replacement, target);
        if (status != cudaSuccess) {
          std::ostringstream message;
          message << "allocate " << target
                  << "-byte CUDA physical workspace failed: "
                  << cudaGetErrorString(status);
          throw std::runtime_error(message.str());
        }
        if (data_ != nullptr) {
          cudaFree(data_);
        }
        data_ = replacement;
        capacity_bytes_ = target;
      }
      offset_bytes_ = 0;
      high_water_bytes_ =
          std::max(high_water_bytes_, required_bytes);
    }

    template <typename T>
    T* acquire(std::size_t count) {
      if (count >
          std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("CUDA workspace array overflow");
      }
      auto const bytes = count * sizeof(T);
      offset_bytes_ =
          alignedWorkspaceOffset(offset_bytes_, alignof(T));
      auto const end = checkedWorkspaceAdd(offset_bytes_, bytes);
      if (end > capacity_bytes_) {
        throw std::logic_error(
            "CUDA workspace sizing and allocation disagree");
      }
      auto* result = static_cast<unsigned char*>(data_) +
                     offset_bytes_;
      offset_bytes_ = end;
      return reinterpret_cast<T*>(result);
    }

    void* acquireBytes(std::size_t bytes,
                       std::size_t alignment = 256) {
      offset_bytes_ =
          alignedWorkspaceOffset(offset_bytes_, alignment);
      auto const end = checkedWorkspaceAdd(offset_bytes_, bytes);
      if (end > capacity_bytes_) {
        throw std::logic_error(
            "CUDA workspace byte allocation exceeds arena");
      }
      auto* result = static_cast<unsigned char*>(data_) +
                     offset_bytes_;
      offset_bytes_ = end;
      return result;
    }

    std::size_t capacityBytes() const { return capacity_bytes_; }
    std::size_t highWaterBytes() const { return high_water_bytes_; }
    std::size_t byteLimit() const {
      requireConfigured();
      return byte_limit_;
    }

    void release() noexcept {
      if (data_ != nullptr) {
        cudaSetDevice(device_);
        cudaFree(data_);
      }
      data_ = nullptr;
      capacity_bytes_ = 0;
      offset_bytes_ = 0;
      high_water_bytes_ = 0;
      byte_limit_ = 0;
      configured_ = false;
    }

  private:
    void requireConfigured() const {
      if (!configured_) {
        throw std::logic_error(
            "CUDA physical workspace is not configured");
      }
    }

    int device_{};
    bool configured_{};
    void* data_{};
    std::size_t byte_limit_{};
    std::size_t capacity_bytes_{};
    std::size_t offset_bytes_{};
    std::size_t high_water_bytes_{};
  };

} // namespace corsika::gpu::em::detail
