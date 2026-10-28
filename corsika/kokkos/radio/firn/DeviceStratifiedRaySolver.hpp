/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/ExponentialOpticalParameters.hpp>
#include <corsika/kokkos/radio/firn/ExponentialEndpointData.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <algorithm>

namespace corsika::kokkos::radio::firn::detail::go {
  /// Fixed numerical records used only by the device geometry kernels.
  struct Vector {
    double values_[3]{};
    KOKKOS_INLINE_FUNCTION double& operator[](int i) { return values_[i]; }
    KOKKOS_INLINE_FUNCTION double operator[](int i) const { return values_[i]; }
  };
  enum class Status {
    Success,
    OutsideProfile,
    NoPath,
    FoldGuard,
    BranchOverflow,
    InvalidGeometry
  };
  struct Path {
    bool valid_{false};
    int topology_{6};
    double invariantP_{0.0};
    double sourceRefractiveIndex_{1.0};
    double observerRefractiveIndex_{1.0};
    double geometricLengthM_{0.0};
    double opticalPathM_{0.0};
    double travelTimeS_{0.0};
    double attenuationExponent_{0.0};
    double phaseOffsetRad_{0.0};
    double horizontalResidualM_{0.0};
    double fermatRelativeResidual_{0.0};
    double snellResidual_{0.0};
    double geometricSpreadingPerM_{0.0};
    Vector emitDirection_{};
    Vector receiveDirection_{};
  };
  template <typename T>
  struct BranchList {
    static constexpr int capacity_ = 8;
    T values_[capacity_]{};
    int size_{0};
    bool overflow_{false};
    bool invalid_{false};
    KOKKOS_INLINE_FUNCTION int size() const { return size_; }
    KOKKOS_INLINE_FUNCTION bool empty() const { return size_ == 0; }
    KOKKOS_INLINE_FUNCTION T& back() { return values_[size_ - 1]; }
    KOKKOS_INLINE_FUNCTION T& operator[](int i) { return values_[i]; }
    KOKKOS_INLINE_FUNCTION T const& operator[](int i) const { return values_[i]; }
    KOKKOS_INLINE_FUNCTION void push_back(T const& value) {
      if (size_ == capacity_) {
        overflow_ = true;
        return;
      }
      values_[size_++] = value;
    }
  };
  struct Query {
    BranchList<Path> paths_{};
    Status status_{Status::Success};
  };
  struct Profile {
    ExponentialOpticalParameters data_;
    KOKKOS_INLINE_FUNCTION double minimumSupportedZ() const { return data_.minimumZM_; }
    KOKKOS_INLINE_FUNCTION double maximumSupportedZ() const { return data_.surfaceZM_; }
    KOKKOS_INLINE_FUNCTION double refractiveIndexAtZ(double z) const {
      return data_.deepIndex_ -
             data_.deltaIndex_ * std::exp((z - data_.surfaceZM_) / data_.scaleDepthM_);
    }
    KOKKOS_INLINE_FUNCTION double attenuationAtZ(double, double) const {
      return data_.attenuationPerM_;
    }
    KOKKOS_INLINE_FUNCTION double turningLegHorizontal(double p, double endpointZM) const;
    KOKKOS_INLINE_FUNCTION double turningLegHorizontalDerivative(double p,
                                                                 double endpointZM) const;
    KOKKOS_INLINE_FUNCTION ExponentialEndpointData
    endpointRayData(double q, double deepZM, double shallowZM) const;
  };
  /** Numerical port of StratifiedRaySolver: same endpoint regularization,
   * quadrature counts, bracketing scans and root iterations. The fixed branch
   * storage reports overflow instead of silently discarding solutions.
   */
  class RaySolver {
  public:
    KOKKOS_INLINE_FUNCTION explicit RaySolver(ExponentialOpticalParameters profile)
        : profile_{profile} {}
    KOKKOS_INLINE_FUNCTION Path direct(Vector const&, Vector const&, double = 0.) const;
    KOKKOS_INLINE_FUNCTION BranchList<Path> turningBranches(Vector const&, Vector const&,
                                                            double = 0.) const;
    KOKKOS_INLINE_FUNCTION Path surfaceReflected(Vector const&, Vector const&,
                                                 double = 0.) const;
    KOKKOS_INLINE_FUNCTION Query evaluate(Vector const&, Vector const&, bool, bool,
                                          double) const;

  private:
    KOKKOS_INLINE_FUNCTION Path nearEndpoint(Vector const&, Vector const&, bool) const;
    Profile profile_;
  };
} // namespace corsika::kokkos::radio::firn::detail::go
#include <corsika/detail/kokkos/radio/firn/DeviceStratifiedRaySolver.inl>
