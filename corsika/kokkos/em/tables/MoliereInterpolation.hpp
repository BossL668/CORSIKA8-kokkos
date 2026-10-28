/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/kokkos/em/models/MoliereSnapshot.hpp>
#include <corsika/kokkos/em/tables/ProposalNativeData.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <type_traits>

namespace corsika::kokkos {
  inline constexpr std::size_t MoliereInterpolationNodeCount = 100;
  inline constexpr std::size_t MoliereInterpolationIntervalCount =
      MoliereInterpolationNodeCount - 1;
  inline constexpr std::size_t MoliereInterpolationFunctionCount = 4;
  inline constexpr double MoliereInterpolationLower = 0.;
  inline constexpr double MoliereInterpolationUpper = 20.;
  inline constexpr std::size_t MoliereInitialGuessBNodeCount = 64;
  inline constexpr std::size_t MoliereInitialGuessBetaSquaredNodeCount = 8;
  inline constexpr std::size_t MoliereInitialGuessCoordinateNodeCount = 128;
  inline constexpr double MoliereInitialGuessBLower = 4.5;
  inline constexpr double MoliereInitialGuessBUpper = 64.;
  inline constexpr double MoliereInitialGuessBetaSquaredLower = 0.25;
  inline constexpr double MoliereInitialGuessBetaSquaredUpper = 1.;
  inline constexpr double MoliereInitialGuessCoordinateLower = 0.;
  inline constexpr double MoliereInitialGuessCoordinateUpper = 5.;

  /** Cubic in the normalized interval coordinate, in ascending power order. */
  struct MoliereCubicPolynomial {
    double c_[4]{};
  };

  /** Host-owned production polynomials and Newton initial-guess corrections.
   * The member order and zero initialization preserve the v1 raw cache bytes.
   * Initial guesses do not replace the final multi-component Newton solve. */
  struct MoliereInterpolationTable {
    std::array<MoliereCubicPolynomial,
               MoliereInterpolationFunctionCount * MoliereInterpolationIntervalCount>
        polynomials_{};
    /**
     * Delta from the Gaussian coordinate to the snapshot-specific
     * multi-component Moliere inverse, tabulated in reference B, beta^2 and
     * quantile coordinate. It is only a Newton initial guess; the full
     * multi-component PDF/CDF still determines the returned angle.
     */
    std::array<double, MoliereInitialGuessBNodeCount *
                           MoliereInitialGuessBetaSquaredNodeCount *
                           MoliereInitialGuessCoordinateNodeCount>
        initialGuessDelta_{};
  };

  /** Original non-pointer interpolation metadata; no new bounds or repair. */
  struct MoliereInterpolationParameters {
    std::uint32_t intervalCount_{};
    std::uint32_t initialGuessBNodeCount_{};
    std::uint32_t initialGuessBetaSquaredNodeCount_{};
    std::uint32_t initialGuessCoordinateNodeCount_{};
    std::uint32_t reserved_{};
    double lower_{};
    double upper_{};
    double inverseStep_{};
    double initialGuessBLower_{};
    double initialGuessBUpper_{};
    double initialGuessInverseBStep_{};
    double initialGuessBetaSquaredLower_{};
    double initialGuessBetaSquaredUpper_{};
    double initialGuessInverseBetaSquaredStep_{};
    double initialGuessCoordinateLower_{};
    double initialGuessCoordinateUpper_{};
    double initialGuessInverseCoordinateStep_{};
  };

  /** Non-owning view retaining the production private pointer/count layout.
   * The supplied storage must remain accessible and alive until kernels finish.
   * This address-bearing view is not serialized; only the table is cached. */
  class MoliereInterpolationView {
  public:
    CORSIKA_HOST_DEVICE MoliereInterpolationView();
    /** Bind the four consecutive polynomial segments and optional initial guesses.
     * Storage must cover the production table dimensions and outlive all queries.
     * A missing polynomial allocation returns the default empty view. */
    CORSIKA_HOST_DEVICE explicit MoliereInterpolationView(
        NativeArrayView<MoliereCubicPolynomial> const& polynomials,
        NativeArrayView<double> const& initialGuess = {});
    /** Bind independent arrays and the original metadata without validation.
     * This preserves incomplete/malformed-view states for the existing validator;
     * non-null arrays must cover the indices reached by a valid query. */
    CORSIKA_HOST_DEVICE MoliereInterpolationView(
        NativeArrayView<MoliereCubicPolynomial> const& f1,
        NativeArrayView<MoliereCubicPolynomial> const& f2,
        NativeArrayView<MoliereCubicPolynomial> const& integralF1,
        NativeArrayView<MoliereCubicPolynomial> const& integralF2,
        NativeArrayView<double> const& initialGuess,
        MoliereInterpolationParameters const& parameters);
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<MoliereCubicPolynomial> getF1()
        const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<MoliereCubicPolynomial> getF2()
        const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<MoliereCubicPolynomial>
    getIntegralF1() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<MoliereCubicPolynomial>
    getIntegralF2() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<double> getInitialGuessDelta()
        const;
    [[nodiscard]] CORSIKA_HOST_DEVICE std::uint32_t getIntervalCount() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE std::uint32_t getInitialGuessBNodeCount() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE std::uint32_t getInitialGuessBetaSquaredNodeCount()
        const;
    [[nodiscard]] CORSIKA_HOST_DEVICE std::uint32_t getInitialGuessCoordinateNodeCount()
        const;
    [[nodiscard]] CORSIKA_HOST_DEVICE std::uint32_t getReserved() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getLower() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getUpper() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getInverseStep() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getInitialGuessBLower() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getInitialGuessBUpper() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getInitialGuessInverseBStep() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getInitialGuessBetaSquaredLower() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getInitialGuessBetaSquaredUpper() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getInitialGuessInverseBetaSquaredStep()
        const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getInitialGuessCoordinateLower() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getInitialGuessCoordinateUpper() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getInitialGuessInverseCoordinateStep() const;

  private:
    MoliereCubicPolynomial const* f1_{};
    MoliereCubicPolynomial const* f2_{};
    MoliereCubicPolynomial const* integralF1_{};
    MoliereCubicPolynomial const* integralF2_{};
    double const* initialGuessDelta_{};
    std::uint32_t intervalCount_{};
    std::uint32_t initialGuessBNodeCount_{};
    std::uint32_t initialGuessBetaSquaredNodeCount_{};
    std::uint32_t initialGuessCoordinateNodeCount_{};
    std::uint32_t reserved_{};
    double lower_{};
    double upper_{};
    double inverseStep_{};
    double initialGuessBLower_{};
    double initialGuessBUpper_{};
    double initialGuessInverseBStep_{};
    double initialGuessBetaSquaredLower_{};
    double initialGuessBetaSquaredUpper_{};
    double initialGuessInverseBetaSquaredStep_{};
    double initialGuessCoordinateLower_{};
    double initialGuessCoordinateUpper_{};
    double initialGuessInverseCoordinateStep_{};
  };

  /** Build a non-owning view; arguments do not transfer storage ownership. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline MoliereInterpolationView
  make_moliere_interpolation_view(
      NativeArrayView<MoliereCubicPolynomial> const& polynomials,
      NativeArrayView<double> const& initialGuess = {});

  /** Generate the production cardinal-spline polynomials and inverse seeds.
   * CubicInterpolation implementation is compiled host-only, not included here. */
  [[nodiscard]] MoliereInterpolationTable make_moliere_interpolation_table(
      MoliereSnapshot const& snapshot);
  /** Load the validated v1 cache or generate a usable in-memory table.
   * Lock/write failure retains the original fallback-to-memory semantics. */
  [[nodiscard]] MoliereInterpolationTable load_or_make_moliere_interpolation_table(
      MoliereSnapshot const& snapshot, std::filesystem::path const& cachePath);

  static_assert(std::is_standard_layout_v<MoliereCubicPolynomial>);
  static_assert(std::is_trivially_copyable_v<MoliereCubicPolynomial>);
  static_assert(std::is_standard_layout_v<MoliereInterpolationTable>);
  static_assert(std::is_trivially_copyable_v<MoliereInterpolationTable>);
  static_assert(offsetof(MoliereInterpolationTable, polynomials_) == 0);
  static_assert(std::is_standard_layout_v<MoliereInterpolationView>);
  static_assert(std::is_trivially_copyable_v<MoliereInterpolationView>);
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/tables/MoliereInterpolation.inl>
