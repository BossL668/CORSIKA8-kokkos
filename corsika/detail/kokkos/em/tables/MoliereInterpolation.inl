/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::kokkos {
  CORSIKA_HOST_DEVICE inline MoliereInterpolationView::MoliereInterpolationView() {}
  CORSIKA_HOST_DEVICE inline MoliereInterpolationView::MoliereInterpolationView(
      NativeArrayView<MoliereCubicPolynomial> const& polynomials,
      NativeArrayView<double> const& initialGuess) {
    auto const* device_or_host_polynomials = polynomials.data_;
    auto const* device_or_host_initial_guess = initialGuess.data_;
    if (device_or_host_polynomials == nullptr) return;
    constexpr auto intervals = MoliereInterpolationIntervalCount;
    *this = MoliereInterpolationView{
        NativeArrayView<MoliereCubicPolynomial>{device_or_host_polynomials, intervals},
        NativeArrayView<MoliereCubicPolynomial>{device_or_host_polynomials + intervals,
                                                intervals},
        NativeArrayView<MoliereCubicPolynomial>{
            device_or_host_polynomials + 2 * intervals, intervals},
        NativeArrayView<MoliereCubicPolynomial>{
            device_or_host_polynomials + 3 * intervals, intervals},
        NativeArrayView<double>{device_or_host_initial_guess, initialGuess.size_},
        MoliereInterpolationParameters{
            static_cast<std::uint32_t>(intervals),
            device_or_host_initial_guess == nullptr
                ? 0U
                : static_cast<std::uint32_t>(MoliereInitialGuessBNodeCount),
            device_or_host_initial_guess == nullptr
                ? 0U
                : static_cast<std::uint32_t>(MoliereInitialGuessBetaSquaredNodeCount),
            device_or_host_initial_guess == nullptr
                ? 0U
                : static_cast<std::uint32_t>(MoliereInitialGuessCoordinateNodeCount),
            0, MoliereInterpolationLower, MoliereInterpolationUpper,
            static_cast<double>(intervals) /
                (MoliereInterpolationUpper - MoliereInterpolationLower),
            MoliereInitialGuessBLower, MoliereInitialGuessBUpper,
            static_cast<double>(MoliereInitialGuessBNodeCount - 1) /
                (MoliereInitialGuessBUpper - MoliereInitialGuessBLower),
            MoliereInitialGuessBetaSquaredLower, MoliereInitialGuessBetaSquaredUpper,
            static_cast<double>(MoliereInitialGuessBetaSquaredNodeCount - 1) /
                (MoliereInitialGuessBetaSquaredUpper -
                 MoliereInitialGuessBetaSquaredLower),
            MoliereInitialGuessCoordinateLower, MoliereInitialGuessCoordinateUpper,
            static_cast<double>(MoliereInitialGuessCoordinateNodeCount - 1) /
                (MoliereInitialGuessCoordinateUpper -
                 MoliereInitialGuessCoordinateLower)}};
  }
  CORSIKA_HOST_DEVICE inline MoliereInterpolationView::MoliereInterpolationView(
      NativeArrayView<MoliereCubicPolynomial> const& f1,
      NativeArrayView<MoliereCubicPolynomial> const& f2,
      NativeArrayView<MoliereCubicPolynomial> const& integralF1,
      NativeArrayView<MoliereCubicPolynomial> const& integralF2,
      NativeArrayView<double> const& initialGuess,
      MoliereInterpolationParameters const& parameters)
      : f1_(f1.data_)
      , f2_(f2.data_)
      , integralF1_(integralF1.data_)
      , integralF2_(integralF2.data_)
      , initialGuessDelta_(initialGuess.data_)
      , intervalCount_(parameters.intervalCount_)
      , initialGuessBNodeCount_(parameters.initialGuessBNodeCount_)
      , initialGuessBetaSquaredNodeCount_(parameters.initialGuessBetaSquaredNodeCount_)
      , initialGuessCoordinateNodeCount_(parameters.initialGuessCoordinateNodeCount_)
      , reserved_(parameters.reserved_)
      , lower_(parameters.lower_)
      , upper_(parameters.upper_)
      , inverseStep_(parameters.inverseStep_)
      , initialGuessBLower_(parameters.initialGuessBLower_)
      , initialGuessBUpper_(parameters.initialGuessBUpper_)
      , initialGuessInverseBStep_(parameters.initialGuessInverseBStep_)
      , initialGuessBetaSquaredLower_(parameters.initialGuessBetaSquaredLower_)
      , initialGuessBetaSquaredUpper_(parameters.initialGuessBetaSquaredUpper_)
      , initialGuessInverseBetaSquaredStep_(
            parameters.initialGuessInverseBetaSquaredStep_)
      , initialGuessCoordinateLower_(parameters.initialGuessCoordinateLower_)
      , initialGuessCoordinateUpper_(parameters.initialGuessCoordinateUpper_)
      , initialGuessInverseCoordinateStep_(
            parameters.initialGuessInverseCoordinateStep_) {}

  CORSIKA_HOST_DEVICE inline NativeArrayView<MoliereCubicPolynomial>
  MoliereInterpolationView::getF1() const {
    return {f1_, intervalCount_};
  }
  CORSIKA_HOST_DEVICE inline NativeArrayView<MoliereCubicPolynomial>
  MoliereInterpolationView::getF2() const {
    return {f2_, intervalCount_};
  }
  CORSIKA_HOST_DEVICE inline NativeArrayView<MoliereCubicPolynomial>
  MoliereInterpolationView::getIntegralF1() const {
    return {integralF1_, intervalCount_};
  }
  CORSIKA_HOST_DEVICE inline NativeArrayView<MoliereCubicPolynomial>
  MoliereInterpolationView::getIntegralF2() const {
    return {integralF2_, intervalCount_};
  }
  CORSIKA_HOST_DEVICE inline NativeArrayView<double>
  MoliereInterpolationView::getInitialGuessDelta() const {
    return {initialGuessDelta_, static_cast<std::uint64_t>(initialGuessBNodeCount_) *
                                    initialGuessBetaSquaredNodeCount_ *
                                    initialGuessCoordinateNodeCount_};
  }
  CORSIKA_HOST_DEVICE inline std::uint32_t MoliereInterpolationView::getIntervalCount()
      const {
    return intervalCount_;
  }
  CORSIKA_HOST_DEVICE inline std::uint32_t
  MoliereInterpolationView::getInitialGuessBNodeCount() const {
    return initialGuessBNodeCount_;
  }
  CORSIKA_HOST_DEVICE inline std::uint32_t
  MoliereInterpolationView::getInitialGuessBetaSquaredNodeCount() const {
    return initialGuessBetaSquaredNodeCount_;
  }
  CORSIKA_HOST_DEVICE inline std::uint32_t
  MoliereInterpolationView::getInitialGuessCoordinateNodeCount() const {
    return initialGuessCoordinateNodeCount_;
  }
  CORSIKA_HOST_DEVICE inline std::uint32_t MoliereInterpolationView::getReserved() const {
    return reserved_;
  }
  CORSIKA_HOST_DEVICE inline double MoliereInterpolationView::getLower() const {
    return lower_;
  }
  CORSIKA_HOST_DEVICE inline double MoliereInterpolationView::getUpper() const {
    return upper_;
  }
  CORSIKA_HOST_DEVICE inline double MoliereInterpolationView::getInverseStep() const {
    return inverseStep_;
  }
  CORSIKA_HOST_DEVICE inline double MoliereInterpolationView::getInitialGuessBLower()
      const {
    return initialGuessBLower_;
  }
  CORSIKA_HOST_DEVICE inline double MoliereInterpolationView::getInitialGuessBUpper()
      const {
    return initialGuessBUpper_;
  }
  CORSIKA_HOST_DEVICE inline double
  MoliereInterpolationView::getInitialGuessInverseBStep() const {
    return initialGuessInverseBStep_;
  }
  CORSIKA_HOST_DEVICE inline double
  MoliereInterpolationView::getInitialGuessBetaSquaredLower() const {
    return initialGuessBetaSquaredLower_;
  }
  CORSIKA_HOST_DEVICE inline double
  MoliereInterpolationView::getInitialGuessBetaSquaredUpper() const {
    return initialGuessBetaSquaredUpper_;
  }
  CORSIKA_HOST_DEVICE inline double
  MoliereInterpolationView::getInitialGuessInverseBetaSquaredStep() const {
    return initialGuessInverseBetaSquaredStep_;
  }
  CORSIKA_HOST_DEVICE inline double
  MoliereInterpolationView::getInitialGuessCoordinateLower() const {
    return initialGuessCoordinateLower_;
  }
  CORSIKA_HOST_DEVICE inline double
  MoliereInterpolationView::getInitialGuessCoordinateUpper() const {
    return initialGuessCoordinateUpper_;
  }
  CORSIKA_HOST_DEVICE inline double
  MoliereInterpolationView::getInitialGuessInverseCoordinateStep() const {
    return initialGuessInverseCoordinateStep_;
  }
  CORSIKA_HOST_DEVICE inline MoliereInterpolationView make_moliere_interpolation_view(
      NativeArrayView<MoliereCubicPolynomial> const& polynomials,
      NativeArrayView<double> const& initialGuess) {
    return MoliereInterpolationView(polynomials, initialGuess);
  }
} // namespace corsika::kokkos
