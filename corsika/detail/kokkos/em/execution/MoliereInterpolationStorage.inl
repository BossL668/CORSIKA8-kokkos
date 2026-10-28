/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <stdexcept>

namespace corsika::kokkos {

  template <typename TExecutionSpace>
  constexpr std::size_t
  MoliereInterpolationStorage<TExecutionSpace>::projectedDeviceBytes() noexcept {
    return MoliereInterpolationFunctionCount * MoliereInterpolationIntervalCount *
               sizeof(MoliereCubicPolynomial) +
           MoliereInitialGuessBNodeCount * MoliereInitialGuessBetaSquaredNodeCount *
               MoliereInitialGuessCoordinateNodeCount * sizeof(double);
  }

  template <typename TExecutionSpace>
  void MoliereInterpolationStorage<TExecutionSpace>::initialize(
      MoliereSnapshot const& snapshot, std::filesystem::path const& cachePath) {
    if (initialized_) {
      throw std::logic_error("Kokkos Moliere interpolation is already initialized");
    }
    auto const host_table = load_or_make_moliere_interpolation_table(snapshot, cachePath);
    polynomials_ = polynomial_view_type("c8_kokkos_moliere_polynomials",
                                        host_table.polynomials_.size());
    initialGuess_ = double_view_type("c8_kokkos_moliere_initial_guess",
                                     host_table.initialGuessDelta_.size());
    auto host_polynomials = Kokkos::create_mirror_view(polynomials_);
    auto host_initial = Kokkos::create_mirror_view(initialGuess_);
    for (std::size_t index = 0; index < host_table.polynomials_.size(); ++index)
      host_polynomials(index) = host_table.polynomials_[index];
    for (std::size_t index = 0; index < host_table.initialGuessDelta_.size(); ++index)
      host_initial(index) = host_table.initialGuessDelta_[index];
    Kokkos::deep_copy(execution_, polynomials_, host_polynomials);
    Kokkos::deep_copy(execution_, initialGuess_, host_initial);
    execution_.fence("upload Kokkos Moliere interpolation");
    view_ = make_moliere_interpolation_view(
        NativeArrayView<MoliereCubicPolynomial>{polynomials_},
        NativeArrayView<double>{initialGuess_});
    initialized_ = true;
  }

  template <typename TExecutionSpace>
  MoliereInterpolationView MoliereInterpolationStorage<TExecutionSpace>::getDeviceView()
      const {
    if (!initialized_)
      throw std::logic_error("Kokkos Moliere interpolation is unavailable");
    return view_;
  }

  template <typename TExecutionSpace>
  std::size_t MoliereInterpolationStorage<TExecutionSpace>::getDeviceBytes() const
      noexcept {
    return polynomials_.extent(0) * sizeof(MoliereCubicPolynomial) +
           initialGuess_.extent(0) * sizeof(double);
  }

  template <typename TExecutionSpace>
  bool MoliereInterpolationStorage<TExecutionSpace>::isInitialized() const noexcept {
    return initialized_;
  }

} // namespace corsika::kokkos
