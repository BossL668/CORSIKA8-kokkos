/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <filesystem>
#include <stdexcept>

#include <corsika/gpu/em/MoliereScattering.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  /** Immutable execution-space copy of the shared Moliere helper data. */
  template <class ExecutionSpace>
  class KokkosMoliereInterpolation {
  public:
    using memory_space = typename ExecutionSpace::memory_space;

    void initialize(
        gpu::em::MoliereSnapshot const& snapshot,
        std::filesystem::path const& cache_path = {}) {
      if (initialized_) {
        throw std::logic_error(
            "Kokkos Moliere interpolation is already initialized");
      }
      auto const host_table = gpu::em::loadOrMakeMoliereInterpolationTable(
          snapshot, cache_path);
      polynomials_ = PolynomialView(
          "c8_kokkos_moliere_polynomials", host_table.polynomials.size());
      initial_guess_ = DoubleView(
          "c8_kokkos_moliere_initial_guess",
          host_table.initial_guess_delta.size());
      auto host_polynomials = Kokkos::create_mirror_view(polynomials_);
      auto host_initial = Kokkos::create_mirror_view(initial_guess_);
      for (std::size_t i = 0; i < host_table.polynomials.size(); ++i)
        host_polynomials(i) = host_table.polynomials[i];
      for (std::size_t i = 0; i < host_table.initial_guess_delta.size(); ++i)
        host_initial(i) = host_table.initial_guess_delta[i];
      Kokkos::deep_copy(execution_, polynomials_, host_polynomials);
      Kokkos::deep_copy(execution_, initial_guess_, host_initial);
      execution_.fence("upload Kokkos Moliere interpolation");
      view_ = gpu::em::makeMoliereInterpolationView(
          polynomials_.data(), initial_guess_.data());
      initialized_ = true;
    }

    gpu::em::MoliereInterpolationView deviceView() const {
      if (!initialized_)
        throw std::logic_error("Kokkos Moliere interpolation is unavailable");
      return view_;
    }

    std::size_t deviceBytes() const noexcept {
      return polynomials_.extent(0) * sizeof(gpu::em::MoliereCubicPolynomial) +
             initial_guess_.extent(0) * sizeof(double);
    }

  private:
    using PolynomialView = Kokkos::View<
        gpu::em::MoliereCubicPolynomial*, memory_space>;
    using DoubleView = Kokkos::View<double*, memory_space>;
    ExecutionSpace execution_{};
    PolynomialView polynomials_{};
    DoubleView initial_guess_{};
    gpu::em::MoliereInterpolationView view_{};
    bool initialized_{};
  };

} // namespace corsika::accelerator::em::kokkos_detail
