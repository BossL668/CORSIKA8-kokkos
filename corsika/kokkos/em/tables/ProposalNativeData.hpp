/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

/**
 * \file
 * Format-v7 numerical records exported from PROPOSAL 7.6.2 calculators.
 *
 * Energies and masses are in MeV. Grammage is in g/cm^2, rates in cm^2/g,
 * and stopping power in MeV cm^2/g. Hashes identify the original calculator,
 * cross section and component; canonical column order is distinct from the
 * stored selection and stopping-power accumulation orders. Record field order
 * and enum values are retained. Serialization uses the canonical host encoder,
 * never structure padding or the address-bearing query view.
 */

namespace corsika::kokkos {

  // v6 additionally preserves Displacement::FunctionToIntegral()'s exact
  // cross-list/component dE/dX accumulation order.  Descriptors remain in
  // canonical identity order; a compact index permutation restores the
  // scalar floating-point summation order on the device.
  // v7 adds PROPOSAL's muon mass for the BKK secondary-split limits.
  inline constexpr std::uint32_t ProposalNativeTableFormatVersion = 7;
  inline constexpr char ProposalNativePhysicsSource[] = "proposal-native";
  inline constexpr std::uint64_t NoNativePolynomialCoefficients =
      std::numeric_limits<std::uint64_t>::max();

  enum class NativeAxisType : std::uint32_t {
    Linear = 0,
    Exponential = 1,
    ExponentialMinusOne = 2,
  };

  enum class NativeLossTransform : std::uint32_t {
    RelativeLog = 0,
    RelativeLogPowerOnePointFive = 1,
    OneMinusLog = 2,
  };

  enum class NativeKinematicModel : std::uint32_t {
    Unsupported = 0,
    OnlyStochastic = 1,
    // PhotoPairKochMotz::CalculateStochasticLoss() in PROPOSAL 7.6.2
    // returns one: the stochastic photon is consumed completely.  The
    // exported cumulative dN/dX spline is still used for the interaction
    // rate, but it is not an inverse CDF for an externally observable loss
    // fraction.
    PhotonPairFullLoss = 2,
    Compton = 3,
    BremsElectronScreening = 4,
    ElectronPair = 5,
    IonizationMoller = 6,
    IonizationBhabha = 7,
    IonizationBetheBlochRossi = 8,
    BremsKelnerKokoulinPetrukhin = 9,
    PhotonuclearAllm97 = 10,
  };

  enum class NativeRateModel : std::uint32_t {
    BicubicSpline = 0,
    AnnihilationHeitler = 1,
    PhotoeffectSauter = 2,
    PhotoproductionHeckC7Shadowing = 3,
  };

  enum class NativeQueryStatus : std::uint32_t {
    Success = 0,
    InvalidView = 1,
    NonFiniteInput = 2,
    ParticleNotFound = 3,
    ColumnNotFound = 4,
    EnergyOutOfRange = 5,
    QuantileOutOfRange = 6,
    UnsupportedKinematics = 7,
    RootNotConverged = 8,
  };

  /** Original status, solver counts and result; inspect the complete outcome. */
  struct NativeQueryResult {
    NativeQueryStatus status_{NativeQueryStatus::InvalidView};
    std::uint32_t newtonIterations_{};
    std::uint32_t bisectionIterations_{};
    double value_{};
  };

  /** Native interpolation-axis transformation and equidistant transformed nodes. */
  struct NativeAxisDescriptor {
    NativeAxisType type_{NativeAxisType::Linear};
    std::uint32_t nodes_{};
    double low_{};
    double high_{};
    double step_{};
  };

  /** Offsets into four column-major spline arrays and optional cell polynomials. */
  struct NativeBicubicDescriptor {
    NativeAxisDescriptor energyAxis_{};
    NativeAxisDescriptor lossAxis_{};
    std::uint32_t rows_{};
    std::uint32_t columns_{};
    std::uint64_t coefficientOffset_{};
    // Optional per-cell polynomial coefficients, derived once on the host
    // with CubicInterpolation 0.1.5's Eigen expression.  The original spline
    // arrays above remain authoritative and are always retained.
    std::uint64_t polynomialCoefficientOffset_{NoNativePolynomialCoefficients};
  };

  /** Cubic node/value-derivative slice and its optional output transformation. */
  struct NativeCubicDescriptor {
    NativeAxisDescriptor axis_{};
    NativeAxisDescriptor valueTransform_{};
    std::uint64_t coefficientOffset_{};
    std::uint32_t coefficientCount_{};
    std::uint32_t hasValueTransform_{};
  };

  /** Canonical stochastic-rate/CDF identity, kinematics, cut and component data. */
  struct NativeDndxColumn {
    std::int32_t pdgId_{};
    std::int32_t processId_{};
    std::uint64_t mediumHash_{};
    std::uint64_t componentHash_{};
    std::uint64_t crossSectionHash_{};
    std::uint64_t tableHash_{};
    NativeLossTransform lossTransform_{NativeLossTransform::RelativeLog};
    NativeKinematicModel kinematicModel_{NativeKinematicModel::Unsupported};
    NativeRateModel rateModel_{NativeRateModel::BicubicSpline};
    NativeBicubicDescriptor spline_{};
    double componentWeight_{1.};
    double particleMassMeV_{};
    double componentNuclearCharge_{};
    double componentAtomicMass_{};
    double componentAverageNucleonMassMeV_{};
    double directRateFactor_{};
    double lowerEnergyLimitMeV_{};
    double absoluteCutMeV_{};
    double relativeCut_{};
    // PROPOSAL Medium::GetI() converted from eV to MeV.  The native
    // Bethe--Bloch--Rossi ionization boundary is medium dependent through
    // v_min = I / E, so it cannot be reconstructed from component data.
    double mediumMeanExcitationEnergyMeV_{};
    // Stable ordinal in Interaction::Rates(): cross_list insertion order,
    // followed by the exact per-target order returned by each cross section.
    // Identity lookup remains canonical and uses process/component instead.
    std::uint32_t selectionOrdinal_{};
    std::uint32_t reserved_{};
  };

  /** Independent Interaction mean-free-path rate spline, not a sum of columns. */
  struct NativeTotalRateColumn {
    std::int32_t pdgId_{};
    std::uint32_t reserved_{};
    std::uint64_t mediumHash_{};
    std::uint64_t tableHash_{};
    NativeCubicDescriptor spline_{};
    double lowerEnergyLimitMeV_{};
  };

  /** Continuous-loss component, retaining its original accumulation identity. */
  struct NativeDedxColumn {
    std::int32_t pdgId_{};
    std::int32_t processId_{};
    std::uint64_t mediumHash_{};
    std::uint64_t crossSectionHash_{};
    // Stable ordinal in PROPOSAL's underlying per-component dE/dX vector.
    // It is assigned before non-interpolant entries are skipped, so it is an
    // identity rather than merely an index in the exported subset.
    std::uint64_t componentIndex_{};
    std::uint64_t tableHash_{};
    NativeCubicDescriptor spline_{};
    double componentWeight_{1.};
    double lowerEnergyLimitMeV_{};
    std::uint32_t accumulationOrdinal_{};
    std::uint32_t reserved_{};
  };

  /** Signed displacement integral and the particle's original lower-energy limit. */
  struct NativeUtilityColumn {
    std::int32_t pdgId_{};
    std::uint64_t mediumHash_{};
    std::uint64_t tableHash_{};
    NativeCubicDescriptor spline_{};
    double particleMassMeV_{};
    double lowerEnergyLimitMeV_{};
    std::uint32_t reverse_{};
  };

  /** Constants copied from the actual version-locked PROPOSAL calculator. */
  struct NativePhysicalConstants {
    double electronMassMeV_{};
    double chargedPionMassMeV_{};
    double fineStructureConstant_{};
    double classicalElectronRadiusCm_{};
    double pi_{};
    double sqrtE_{};
    double muonMassMeV_{};
  };

  /** Whole Interaction calculator identity, independent of individual columns. */
  struct NativeInteractionIdentity {
    std::int32_t pdgId_{};
    std::uint32_t reserved_{};
    std::uint64_t mediumHash_{};
    std::uint64_t interactionHash_{};
  };

  class ProposalNativeDeviceView;

  /**
   * Read-only, non-owning contiguous input for the C++17 numerical table interface.
   *
   * A container must supply data() and size(); the element and count types must
   * match the exported record. Construction performs no validation or allocation.
   * The owner must retain storage in the accessible memory space until queries
   * finish. Indices must be in range. A non-null zero-length allocation remains
   * distinguishable from missing storage, preserving the production contract.
   */
  template <typename TValue, typename TSize = std::uint64_t>
  class NativeArrayView {
  public:
    using value_type = TValue;
    using size_type = TSize;

    CORSIKA_HOST_DEVICE NativeArrayView();
    template <typename TContainer>
    explicit NativeArrayView(TContainer const& values);
    [[nodiscard]] CORSIKA_HOST_DEVICE size_type getSize() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE bool hasStorage() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE value_type const& operator[](size_type index) const;

  private:
    friend class ProposalNativeDeviceView;
    friend class MoliereInterpolationView;
    CORSIKA_HOST_DEVICE NativeArrayView(value_type const* data, size_type size);
    value_type const* data_{};
    size_type size_{};
  };

  /**
   * Non-owning numerical table view with the production pointer/count layout.
   *
   * Bind all arrays from one validated format-v7 export. Column/order counts
   * remain separate; four bicubic arrays share the values count and both cubic
   * arrays share the cubic-values count, exactly as in the production view.
   * No copying, new validation, normalization or ownership is introduced here.
   * The host or Kokkos owner must outlive every query and asynchronous kernel.
   * This address-bearing view is not the portable serialized table format.
   */
  class ProposalNativeDeviceView {
  public:
    CORSIKA_HOST_DEVICE ProposalNativeDeviceView();
    CORSIKA_HOST_DEVICE ProposalNativeDeviceView(
        NativeArrayView<NativeDndxColumn, std::uint32_t> const& dndxColumns,
        NativeArrayView<NativeTotalRateColumn, std::uint32_t> const& totalRateColumns,
        NativeArrayView<std::uint32_t, std::uint32_t> const& selectionColumnIndices,
        NativeArrayView<NativeDedxColumn, std::uint32_t> const& dedxColumns,
        NativeArrayView<std::uint32_t, std::uint32_t> const& dedxAccumulationIndices,
        NativeArrayView<NativeUtilityColumn, std::uint32_t> const& utilityColumns,
        NativeArrayView<double, std::uint64_t> const& bicubicValues,
        NativeArrayView<double, std::uint64_t> const& bicubicDerivativeEnergy,
        NativeArrayView<double, std::uint64_t> const& bicubicDerivativeLoss,
        NativeArrayView<double, std::uint64_t> const& bicubicMixedDerivative,
        NativeArrayView<double, std::uint64_t> const& bicubicPolynomialCoefficients,
        NativeArrayView<double, std::uint64_t> const& cubicValues,
        NativeArrayView<double, std::uint64_t> const& cubicNodeDerivatives,
        NativePhysicalConstants const& constants);
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<NativeDndxColumn, std::uint32_t>
    getDndxColumns() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE
        NativeArrayView<NativeTotalRateColumn, std::uint32_t>
        getTotalRateColumns() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<std::uint32_t, std::uint32_t>
    getSelectionColumnIndices() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<NativeDedxColumn, std::uint32_t>
    getDedxColumns() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<std::uint32_t, std::uint32_t>
    getDedxAccumulationIndices() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<NativeUtilityColumn, std::uint32_t>
    getUtilityColumns() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<double, std::uint64_t>
    getBicubicValues() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<double, std::uint64_t>
    getBicubicDerivativeEnergy() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<double, std::uint64_t>
    getBicubicDerivativeLoss() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<double, std::uint64_t>
    getBicubicMixedDerivative() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<double, std::uint64_t>
    getBicubicPolynomialCoefficients() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<double, std::uint64_t>
    getCubicValues() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativeArrayView<double, std::uint64_t>
    getCubicNodeDerivatives() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE NativePhysicalConstants const& getConstants() const;

  private:
    NativeDndxColumn const* dndxColumns_{};
    NativeTotalRateColumn const* totalRateColumns_{};
    std::uint32_t const* selectionColumnIndices_{};
    NativeDedxColumn const* dedxColumns_{};
    std::uint32_t const* dedxAccumulationIndices_{};
    NativeUtilityColumn const* utilityColumns_{};
    std::uint32_t dndxColumnCount_{};
    std::uint32_t totalRateColumnCount_{};
    std::uint32_t selectionColumnIndexCount_{};
    std::uint32_t dedxColumnCount_{};
    std::uint32_t dedxAccumulationIndexCount_{};
    std::uint32_t utilityColumnCount_{};
    double const* bicubicValues_{};
    double const* bicubicDerivativeEnergy_{};
    double const* bicubicDerivativeLoss_{};
    double const* bicubicMixedDerivative_{};
    std::uint64_t bicubicCoefficientCount_{};
    double const* bicubicPolynomialCoefficients_{};
    std::uint64_t bicubicPolynomialCoefficientCount_{};
    double const* cubicValues_{};
    double const* cubicNodeDerivatives_{};
    std::uint64_t cubicCoefficientCount_{};
    NativePhysicalConstants constants_{};
  };

  static_assert(std::is_standard_layout_v<NativeAxisDescriptor>);
  static_assert(std::is_trivially_copyable_v<NativeAxisDescriptor>);
  static_assert(std::is_standard_layout_v<NativeDndxColumn>);
  static_assert(std::is_trivially_copyable_v<NativeDndxColumn>);
  static_assert(std::is_standard_layout_v<ProposalNativeDeviceView>);
  static_assert(std::is_trivially_copyable_v<ProposalNativeDeviceView>);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/tables/ProposalNativeData.inl>
