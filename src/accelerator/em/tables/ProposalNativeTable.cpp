/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTableExporter.hpp>

#include <PROPOSAL/Constants.h>
#include <PROPOSAL/crosssection/CrossSection.h>
#include <PROPOSAL/interpolation/InterpolationTableExport.h>
#include <PROPOSAL/medium/Components.h>
#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/propagation_utility/Displacement.h>
#include <PROPOSAL/propagation_utility/Interaction.h>
#include <PROPOSAL/version.h>
#include <CubicInterpolation/version.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace corsika::gpu::em::tables {

  using namespace corsika::units::si;

  namespace {

    class CanonicalEncoder {
    public:
      void unsigned32(std::uint32_t value) {
        for (unsigned int byte = 0; byte < 4; ++byte)
          data_.push_back(static_cast<std::uint8_t>(value >> (8 * byte)));
      }

      void signed32(std::int32_t value) {
        unsigned32(static_cast<std::uint32_t>(value));
      }

      void unsigned64(std::uint64_t value) {
        for (unsigned int byte = 0; byte < 8; ++byte)
          data_.push_back(static_cast<std::uint8_t>(value >> (8 * byte)));
      }

      void floating(double value) {
        static_assert(sizeof(double) == sizeof(std::uint64_t));
        std::uint64_t bits{};
        std::memcpy(&bits, &value, sizeof(bits));
        unsigned64(bits);
      }

      void text(std::string const& value) {
        unsigned64(value.size());
        data_.insert(data_.end(), value.begin(), value.end());
      }

      template <typename T, typename Function>
      void sequence(std::vector<T> const& values, Function encode) {
        unsigned64(values.size());
        for (auto const& value : values) encode(value);
      }

      Sha256Digest digest() const { return sha256(data_); }

    private:
      std::vector<std::uint8_t> data_;
    };

    NativeAxisDescriptor exportAxis(
        cubic_splines::AxisDescriptor<double> const& source) {
      NativeAxisType type{};
      switch (source.type) {
      case cubic_splines::AxisType::Linear:
        type = NativeAxisType::Linear;
        break;
      case cubic_splines::AxisType::Exponential:
        type = NativeAxisType::Exponential;
        break;
      case cubic_splines::AxisType::ExponentialMinusOne:
        type = NativeAxisType::ExponentialMinusOne;
        break;
      default:
        throw std::invalid_argument(
            "unsupported CubicInterpolation coordinate axis");
      }
      if (source.nodes > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("native interpolation axis is too large");
      return {type, static_cast<std::uint32_t>(source.nodes),
              source.low, source.high, source.stepsize};
    }

    using CompositionIdentity =
        std::vector<std::pair<std::uint64_t, std::uint64_t>>;

    std::uint64_t doubleBits(double value) {
      std::uint64_t result{};
      static_assert(sizeof(result) == sizeof(value));
      std::memcpy(&result, &value, sizeof(result));
      return result;
    }

    CompositionIdentity compositionIdentity(PROPOSAL::Medium const& medium) {
      CompositionIdentity result;
      result.reserve(medium.GetComponents().size() + 1);
      result.emplace_back(0u, doubleBits(medium.GetSumNucleons()));
      for (auto const& component : medium.GetComponents()) {
        result.emplace_back(
            static_cast<std::uint64_t>(component.GetHash()),
            doubleBits(component.GetAtomInMolecule()));
      }
      std::sort(result.begin() + 1, result.end());
      return result;
    }

    bool supportedNativeProjectile(Code projectile) {
      switch (projectile) {
      case Code::Photon:
      case Code::Electron:
      case Code::Positron:
      case Code::MuMinus:
      case Code::MuPlus:
        return true;
      default:
        return false;
      }
    }

    void requireSingleComposition(
        std::vector<proposal::NativeInteractionCalculatorView> const& interactions,
        std::vector<proposal::NativeContinuousCalculatorView> const& continuous) {
      CompositionIdentity expected;
      std::array<std::uint64_t, 7> expected_ionization_parameters{};
      bool have_expected = false;
      auto inspect = [&](PROPOSAL::Medium const* medium) {
        if (!medium)
          throw std::invalid_argument(
              "null medium in PROPOSAL native calculator view");
        auto const identity = compositionIdentity(*medium);
        auto const ionization_parameters =
            std::array<std::uint64_t, 7>{
                doubleBits(medium->GetI()), doubleBits(medium->GetC()),
                doubleBits(medium->GetA()), doubleBits(medium->GetM()),
                doubleBits(medium->GetX0()), doubleBits(medium->GetX1()),
                doubleBits(medium->GetD0())};
        if (!have_expected) {
          expected = identity;
          expected_ionization_parameters = ionization_parameters;
          have_expected = true;
        } else if (identity != expected ||
                   ionization_parameters !=
                       expected_ionization_parameters) {
          throw std::invalid_argument(
              "proposal-native supports one medium physics definition; "
              "layers may differ only in mass density, not composition, "
              "mean excitation energy, or density-effect parameters");
        }
      };
      for (auto const& view : interactions)
        if (supportedNativeProjectile(view.projectile)) inspect(view.medium);
      for (auto const& view : continuous)
        if (supportedNativeProjectile(view.projectile)) inspect(view.medium);
      if (!have_expected)
        throw std::invalid_argument(
            "proposal-native export contains no supported projectile calculators");
    }

    NativeLossTransform exportLossTransform(
        PROPOSAL::RelativeLossTransform transform) {
      switch (transform) {
      case PROPOSAL::RelativeLossTransform::RelativeLog:
        return NativeLossTransform::RelativeLog;
      case PROPOSAL::RelativeLossTransform::RelativeLogPowerOnePointFive:
        return NativeLossTransform::RelativeLogPowerOnePointFive;
      case PROPOSAL::RelativeLossTransform::OneMinusLog:
        return NativeLossTransform::OneMinusLog;
      }
      throw std::invalid_argument("unsupported PROPOSAL relative-loss transform");
    }

    NativeKinematicModel classifyKinematicModel(
        std::int32_t process_id, std::string const& parametrization,
        std::int32_t pdg_id) {
      auto const electron = pdg_id == 11 || pdg_id == -11;
      auto const muon = pdg_id == 13 || pdg_id == -13;
      if (process_id == 1000000013 && pdg_id == 22 &&
          parametrization == "kochmotz")
        return NativeKinematicModel::PhotonPairFullLoss;
      if (process_id == 1000000010 && pdg_id == 22 &&
          parametrization == "KleinNishina")
        return NativeKinematicModel::Compton;
      if (process_id == 1000000016 && pdg_id == 22 &&
          parametrization == "Sauter")
        return NativeKinematicModel::OnlyStochastic;
      if (process_id == 1000000012 && pdg_id == -11 &&
          parametrization == "Heitler")
        return NativeKinematicModel::OnlyStochastic;
      if (process_id == 1000000014 && pdg_id == 22 &&
          parametrization == "HeckC7Shadowing")
        return NativeKinematicModel::OnlyStochastic;
      if (process_id == 1000000002 && electron &&
          parametrization == "ElectronScreening")
        return NativeKinematicModel::BremsElectronScreening;
      if (process_id == 1000000004 && electron &&
          parametrization == "ForElectronPositron")
        return NativeKinematicModel::ElectronPair;
      if (process_id == 1000000003 && pdg_id == 11 &&
          parametrization == "BergerSeltzerMoller")
        return NativeKinematicModel::IonizationMoller;
      if (process_id == 1000000003 && pdg_id == -11 &&
          parametrization == "BergerSeltzerBhabha")
        return NativeKinematicModel::IonizationBhabha;
      if (process_id == 1000000003 && muon &&
          parametrization == "BetheBlochRossi")
        return NativeKinematicModel::IonizationBetheBlochRossi;

      if (process_id == 1000000002 && muon &&
          parametrization == "KelnerKokoulinPetrukhin")
        return NativeKinematicModel::BremsKelnerKokoulinPetrukhin;
      if (process_id == 1000000004 && muon &&
          parametrization == "KelnerKokoulinPetrukhin")
        return NativeKinematicModel::ElectronPair;
      if (process_id == 1000000005 && (electron || muon) &&
          parametrization == "AbramowiczLevinLevyMaor97")
        return NativeKinematicModel::PhotonuclearAllm97;
      if (process_id == 1000000015 && pdg_id == 22 &&
          parametrization == "BurkhardtKelnerKokoulin") {
        // This process' 0.5 +/- sqrt(...) interval is the *internal* x used
        // by the CPU secondary generator.  Parametrization.hpp specializes
        // is_only_stochastic<PhotoMuPairBurkhardtKelnerKokoulin> to true, so
        // CrossSection::CalculateStochasticLoss returns the external loss
        // fraction v=1 for every residual rate.  Keep that scalar contract;
        // the selected CPU final-state completion samples x independently.
        return NativeKinematicModel::OnlyStochastic;
      }

      std::ostringstream message;
      message << "unsupported proposal-native process/PID/parametrization: "
              << "process=" << process_id << ", pdg=" << pdg_id
              << ", parametrization=" << parametrization;
      throw std::invalid_argument(message.str());
    }

    PROPOSAL::Component componentForHash(
        PROPOSAL::Medium const& medium, std::uint64_t hash) {
      for (auto const& component : medium.GetComponents()) {
        if (static_cast<std::uint64_t>(component.GetHash()) == hash)
          return component;
      }
      if (static_cast<std::uint64_t>(medium.GetHash()) == hash)
        return PROPOSAL::Component{};
      throw std::invalid_argument(
          "PROPOSAL native table references a component outside its medium");
    }

    NativeBicubicDescriptor appendBicubic(
        ProposalNativeTableSet& result,
        PROPOSAL::DndxInterpolationExport const& source,
        bool precompute_polynomial) {
      auto const rows = source.spline.dimensions[0];
      auto const columns = source.spline.dimensions[1];
      if (rows < 2 || columns < 2 ||
          rows > std::numeric_limits<std::uint32_t>::max() ||
          columns > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("invalid native bicubic dimensions");
      auto const count = static_cast<std::uint64_t>(rows) * columns;
      if (source.spline.values.size() != count ||
          source.spline.derivative_axis0.size() != count ||
          source.spline.derivative_axis1.size() != count ||
          source.spline.mixed_derivative.size() != count)
        throw std::invalid_argument("incomplete native bicubic coefficient arrays");
      if (result.bicubic_values.size() >
          std::numeric_limits<std::uint64_t>::max() - count)
        throw std::overflow_error("native bicubic coefficient offset overflow");
      auto const offset =
          static_cast<std::uint64_t>(result.bicubic_values.size());
      result.bicubic_values.insert(result.bicubic_values.end(),
                                   source.spline.values.begin(),
                                   source.spline.values.end());
      result.bicubic_derivative_energy.insert(
          result.bicubic_derivative_energy.end(),
          source.spline.derivative_axis0.begin(),
          source.spline.derivative_axis0.end());
      result.bicubic_derivative_loss.insert(
          result.bicubic_derivative_loss.end(),
          source.spline.derivative_axis1.begin(),
          source.spline.derivative_axis1.end());
      result.bicubic_mixed_derivative.insert(
          result.bicubic_mixed_derivative.end(),
          source.spline.mixed_derivative.begin(),
          source.spline.mixed_derivative.end());
      auto polynomial_offset = NoNativePolynomialCoefficients;
      if (precompute_polynomial) {
        polynomial_offset = static_cast<std::uint64_t>(
            result.bicubic_polynomial_coefficients.size());
        auto const cells = static_cast<std::uint64_t>(rows - 1) *
                           static_cast<std::uint64_t>(columns - 1);
        if (cells > (std::numeric_limits<std::size_t>::max() -
                     result.bicubic_polynomial_coefficients.size()) /
                        16u)
          throw std::overflow_error(
              "native bicubic polynomial coefficient overflow");
        result.bicubic_polynomial_coefficients.reserve(
            result.bicubic_polynomial_coefficients.size() + 16u * cells);
        Eigen::Matrix4d m;
        m << 1., 0., -3., 2., 0., 0., 3., -2., 0., 1., -2., 1.,
            0., 0., -1., 1.;
        auto const element = [&](std::vector<double> const& values,
                                 std::size_t row, std::size_t column) {
          return values[row + rows * column];
        };
        // Match BicubicSplines::evaluate(): form C=M^T*(T*M) with Eigen on
        // the host.  This is a lossless layout conversion of the exported
        // spline, not a second physics sampling table.
        auto const row_count = static_cast<std::size_t>(rows);
        auto const column_count = static_cast<std::size_t>(columns);
        for (std::size_t loss = 0; loss + 1 < column_count; ++loss) {
          for (std::size_t energy = 0; energy + 1 < row_count; ++energy) {
            Eigen::Matrix4d temp;
            for (std::size_t row = 0; row < 4; ++row) {
              for (std::size_t column = 0; column < 4; ++column) {
                auto const source_row = energy + (row & 1u);
                auto const source_column = loss + (column & 1u);
                auto const* values = &source.spline.values;
                if (row >= 2 && column < 2)
                  values = &source.spline.derivative_axis0;
                else if (row < 2 && column >= 2)
                  values = &source.spline.derivative_axis1;
                else if (row >= 2 && column >= 2)
                  values = &source.spline.mixed_derivative;
                temp(row, column) =
                    element(*values, source_row, source_column);
              }
            }
            Eigen::Matrix4d const coefficients =
                m.transpose() * (temp * m);
            for (std::size_t row = 0; row < 4; ++row)
              for (std::size_t column = 0; column < 4; ++column)
                result.bicubic_polynomial_coefficients.push_back(
                    coefficients(row, column));
          }
        }
      }
      return {exportAxis(source.axes[0]), exportAxis(source.axes[1]),
              static_cast<std::uint32_t>(rows),
              static_cast<std::uint32_t>(columns), offset,
              polynomial_offset};
    }

    NativeCubicDescriptor appendCubic(
        ProposalNativeTableSet& result,
        cubic_splines::AxisDescriptor<double> const& axis,
        bool has_transform,
        cubic_splines::AxisDescriptor<double> const& value_transform,
        cubic_splines::CubicSplines<double>::ExportData const& spline) {
      if (spline.values.size() < 2 ||
          spline.values.size() != spline.node_derivatives.size() ||
          spline.values.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("invalid native cubic coefficient arrays");
      auto const offset =
          static_cast<std::uint64_t>(result.cubic_values.size());
      result.cubic_values.insert(result.cubic_values.end(),
                                 spline.values.begin(), spline.values.end());
      result.cubic_node_derivatives.insert(
          result.cubic_node_derivatives.end(),
          spline.node_derivatives.begin(), spline.node_derivatives.end());
      return {exportAxis(axis),
              has_transform ? exportAxis(value_transform)
                            : NativeAxisDescriptor{},
              offset, static_cast<std::uint32_t>(spline.values.size()),
              has_transform ? 1u : 0u};
    }

    void validateAxis(NativeAxisDescriptor const& axis,
                      std::uint32_t expected_nodes,
                      char const* description) {
      switch (axis.type) {
      case NativeAxisType::Linear:
      case NativeAxisType::Exponential:
      case NativeAxisType::ExponentialMinusOne:
        break;
      default:
        throw std::invalid_argument(
            std::string{"invalid PROPOSAL native axis type in "} +
            description);
      }
      if (axis.nodes < 2 ||
          (expected_nodes != 0 && axis.nodes != expected_nodes))
        throw std::invalid_argument(
            std::string{"invalid PROPOSAL native axis node count in "} +
            description);
      if (!std::isfinite(axis.low) || !std::isfinite(axis.high) ||
          !std::isfinite(axis.step) || !(axis.high > axis.low) ||
          !(axis.step > 0.))
        throw std::invalid_argument(
            std::string{"invalid PROPOSAL native axis domain in "} +
            description);
      if ((axis.type == NativeAxisType::Exponential ||
           axis.type == NativeAxisType::ExponentialMinusOne) &&
          !(axis.low > 0.))
        throw std::invalid_argument(
            std::string{"non-positive PROPOSAL native logarithmic axis in "} +
            description);

      // Check the exact maps used on the device as well as their inputs.
      // This catches finite but internally inconsistent descriptors before a
      // kernel can turn them into NaN/Inf coordinates.
      auto const lower = native_detail::axisTransform(axis, axis.low);
      auto const upper = native_detail::axisTransform(axis, axis.high);
      auto const last = native_detail::axisBackTransform(
          axis, static_cast<double>(axis.nodes - 1u));
      if (!std::isfinite(lower) || !std::isfinite(upper) ||
          !std::isfinite(last))
        throw std::invalid_argument(
            std::string{"non-finite PROPOSAL native axis mapping in "} +
            description);
      auto const endpoint_scale =
          std::max({1., std::abs(axis.high), std::abs(last)});
      auto const coordinate_scale =
          std::max(1., static_cast<double>(axis.nodes - 1u));
      auto const tolerance =
          128. * std::numeric_limits<double>::epsilon();
      if (std::abs(lower) > tolerance ||
          std::abs(upper - static_cast<double>(axis.nodes - 1u)) >
              tolerance * coordinate_scale ||
          std::abs(last - axis.high) > tolerance * endpoint_scale)
        throw std::invalid_argument(
            std::string{"inconsistent PROPOSAL native axis step/domain in "} +
            description);
    }

    void validateCubicDescriptor(NativeCubicDescriptor const& spline,
                                 std::size_t coefficient_count,
                                 char const* description) {
      if (spline.coefficient_count < 2 ||
          spline.coefficient_offset > coefficient_count ||
          spline.coefficient_count >
              coefficient_count - spline.coefficient_offset)
        throw std::invalid_argument(
            std::string{"native cubic spline is out of bounds in "} +
            description);
      validateAxis(spline.axis, spline.coefficient_count, description);
      if (spline.has_value_transform > 1u)
        throw std::invalid_argument(
            std::string{"invalid native cubic value-transform flag in "} +
            description);
      if (spline.has_value_transform != 0u) {
        // A function-value transform is an invertible coordinate map, not an
        // interpolation grid.  CubicInterpolation may therefore export only
        // one descriptive node; nodes/high are not used by axisBackTransform.
        auto const& transform = spline.value_transform;
        switch (transform.type) {
        case NativeAxisType::Linear:
        case NativeAxisType::Exponential:
        case NativeAxisType::ExponentialMinusOne:
          break;
        default:
          throw std::invalid_argument(
              std::string{"invalid native cubic value-transform type in "} +
              description);
        }
        if (!std::isfinite(transform.low) ||
            !std::isfinite(transform.high) ||
            !std::isfinite(transform.step) || !(transform.step > 0.) ||
            ((transform.type == NativeAxisType::Exponential ||
              transform.type == NativeAxisType::ExponentialMinusOne) &&
             !(transform.low > 0.)))
          throw std::invalid_argument(
              std::string{"invalid native cubic value transform in "} +
              description);
      }
    }

    bool supportedNativePdg(std::int32_t pdg_id) {
      return pdg_id == 22 || pdg_id == 11 || pdg_id == -11 ||
             pdg_id == 13 || pdg_id == -13;
    }

    bool supportedNativeChargedPdg(std::int32_t pdg_id) {
      return pdg_id == 11 || pdg_id == -11 || pdg_id == 13 ||
             pdg_id == -13;
    }

    template <typename Values>
    void requireFiniteCoefficients(Values const& values,
                                   char const* description) {
      if (!std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument(
            std::string{"non-finite PROPOSAL native coefficients in "} +
            description);
    }

    using CoefficientSpan = std::pair<std::uint64_t, std::uint64_t>;

    void validateCompleteCoefficientPartition(
        std::vector<CoefficientSpan> spans, std::size_t coefficient_count,
        char const* description) {
      std::sort(spans.begin(), spans.end());
      std::uint64_t expected_offset = 0;
      for (auto const& span : spans) {
        if (span.first != expected_offset || span.second <= span.first)
          throw std::invalid_argument(
              std::string{"overlapping or incomplete coefficient ranges in "} +
              description);
        expected_offset = span.second;
      }
      if (expected_offset != coefficient_count)
        throw std::invalid_argument(
            std::string{"unreferenced PROPOSAL native coefficients in "} +
            description);
    }

    void encodeAxis(CanonicalEncoder& encoder, NativeAxisDescriptor const& axis) {
      encoder.unsigned32(static_cast<std::uint32_t>(axis.type));
      encoder.unsigned32(axis.nodes);
      encoder.floating(axis.low);
      encoder.floating(axis.high);
      encoder.floating(axis.step);
    }

    void encodeCubic(CanonicalEncoder& encoder, NativeCubicDescriptor const& spline) {
      encodeAxis(encoder, spline.axis);
      encodeAxis(encoder, spline.value_transform);
      encoder.unsigned64(spline.coefficient_offset);
      encoder.unsigned32(spline.coefficient_count);
      encoder.unsigned32(spline.has_value_transform);
    }

  } // namespace

  ProposalNativeDeviceView makeProposalNativeHostView(
      ProposalNativeTableSet const& table) {
    return {table.dndx_columns.data(), table.total_rate_columns.data(),
            table.selection_column_indices.data(),
            table.dedx_columns.data(),
            table.dedx_accumulation_indices.data(),
            table.utility_columns.data(),
            static_cast<std::uint32_t>(table.dndx_columns.size()),
            static_cast<std::uint32_t>(table.total_rate_columns.size()),
            static_cast<std::uint32_t>(
                table.selection_column_indices.size()),
            static_cast<std::uint32_t>(table.dedx_columns.size()),
            static_cast<std::uint32_t>(
                table.dedx_accumulation_indices.size()),
            static_cast<std::uint32_t>(table.utility_columns.size()),
            table.bicubic_values.data(),
            table.bicubic_derivative_energy.data(),
            table.bicubic_derivative_loss.data(),
            table.bicubic_mixed_derivative.data(),
            table.bicubic_values.size(),
            table.bicubic_polynomial_coefficients.data(),
            table.bicubic_polynomial_coefficients.size(),
            table.cubic_values.data(),
            table.cubic_node_derivatives.data(), table.cubic_values.size(),
            table.constants};
  }

  NativeKinematicModel classifyProposalNativeKinematicModel(
      std::int32_t process_id, std::string const& parametrization,
      std::int32_t pdg_id) {
    return classifyKinematicModel(process_id, parametrization, pdg_id);
  }

  std::size_t proposalNativeTableBytes(ProposalNativeTableSet const& table) {
    return table.dndx_columns.size() * sizeof(NativeDndxColumn) +
           table.total_rate_columns.size() * sizeof(NativeTotalRateColumn) +
           table.selection_column_indices.size() * sizeof(std::uint32_t) +
           table.dedx_columns.size() * sizeof(NativeDedxColumn) +
           table.dedx_accumulation_indices.size() * sizeof(std::uint32_t) +
           table.utility_columns.size() * sizeof(NativeUtilityColumn) +
           (table.bicubic_values.size() +
            table.bicubic_derivative_energy.size() +
            table.bicubic_derivative_loss.size() +
            table.bicubic_mixed_derivative.size() +
            table.bicubic_polynomial_coefficients.size() +
            table.cubic_values.size() +
            table.cubic_node_derivatives.size()) *
               sizeof(double);
  }

  void validateProposalNativeTable(ProposalNativeTableSet const& table) {
    if (table.format_version != ProposalNativeTableFormatVersion)
      throw std::invalid_argument("unsupported PROPOSAL native table format");
    if (table.proposal_version != "7.6.2" ||
        table.cubic_interpolation_version != "0.1.5")
      throw std::invalid_argument("PROPOSAL native dependency version mismatch");
    if (!(table.constants.electron_mass_MeV > 0.) ||
        !(table.constants.charged_pion_mass_MeV > 0.) ||
        !(table.constants.fine_structure_constant > 0.) ||
        !(table.constants.classical_electron_radius_cm > 0.) ||
        !(table.constants.pi > 3. && table.constants.pi < 4.) ||
        !(table.constants.sqrt_e > 1. && table.constants.sqrt_e < 2.))
      throw std::invalid_argument("invalid PROPOSAL native physical constants");
    if (table.dndx_columns.empty())
      throw std::invalid_argument("PROPOSAL native table has no rate columns");
    if (table.total_rate_columns.empty())
      throw std::invalid_argument(
          "PROPOSAL native table has no mean-free-path rate columns");
    if (table.selection_column_indices.size() !=
        table.dndx_columns.size())
      throw std::invalid_argument(
          "PROPOSAL native selection index does not cover dN/dX columns");
    if (table.dedx_accumulation_indices.size() !=
        table.dedx_columns.size())
      throw std::invalid_argument(
          "PROPOSAL native dE/dX accumulation index does not cover all "
          "continuous-loss columns");
    if (!(table.stochastic_cut_MeV > 0.) ||
        !std::isfinite(table.stochastic_cut_MeV))
      throw std::invalid_argument("invalid PROPOSAL native stochastic cut");
    auto const coefficient_count = table.bicubic_values.size();
    if (table.bicubic_derivative_energy.size() != coefficient_count ||
        table.bicubic_derivative_loss.size() != coefficient_count ||
        table.bicubic_mixed_derivative.size() != coefficient_count)
      throw std::invalid_argument("PROPOSAL native bicubic arrays differ in size");
    if (table.cubic_values.size() != table.cubic_node_derivatives.size())
      throw std::invalid_argument("PROPOSAL native cubic arrays differ in size");
    requireFiniteCoefficients(table.bicubic_values, "bicubic values");
    requireFiniteCoefficients(table.bicubic_derivative_energy,
                              "bicubic energy derivatives");
    requireFiniteCoefficients(table.bicubic_derivative_loss,
                              "bicubic loss derivatives");
    requireFiniteCoefficients(table.bicubic_mixed_derivative,
                              "bicubic mixed derivatives");
    requireFiniteCoefficients(table.bicubic_polynomial_coefficients,
                              "bicubic polynomial coefficients");
    requireFiniteCoefficients(table.cubic_values, "cubic values");
    requireFiniteCoefficients(table.cubic_node_derivatives,
                              "cubic node derivatives");

    // findDndx() performs a binary search on exactly this identity.  Enforce
    // both its canonical order and its uniqueness before uploading the table.
    // Medium is deliberately absent from the device lookup key, so it must be
    // constant throughout each PID range.
    using DndxIdentity =
        std::tuple<std::int32_t, std::int32_t, std::uint64_t>;
    std::set<DndxIdentity> dndx_identities;
    std::map<std::int32_t, std::uint64_t> medium_by_pid;
    std::set<std::int32_t> interaction_pids;
    std::set<std::int32_t> dndx_pids;
    std::vector<CoefficientSpan> bicubic_spans;
    std::vector<CoefficientSpan> polynomial_spans;
    auto registerMedium = [&](std::int32_t pdg_id, std::uint64_t medium_hash,
                              char const* description) {
      auto const [found, inserted] =
          medium_by_pid.emplace(pdg_id, medium_hash);
      if (!inserted && found->second != medium_hash) {
        std::ostringstream message;
        message << "PROPOSAL native table has multiple media for pdg="
                << pdg_id << " (detected in " << description << ")";
        throw std::invalid_argument(message.str());
      }
    };
    for (auto const& identity : table.interaction_identities) {
      if (!supportedNativePdg(identity.pdg_id) || identity.reserved != 0u)
        throw std::invalid_argument(
            "invalid PROPOSAL native interaction identity PID/reserved field");
      if (!interaction_pids.emplace(identity.pdg_id).second)
        throw std::invalid_argument(
            "duplicate PROPOSAL native interaction identity PID");
      registerMedium(identity.pdg_id, identity.medium_hash,
                     "interaction identities");
    }

    DndxIdentity previous_dndx{};
    bool have_previous_dndx = false;
    std::set<std::pair<std::int32_t, std::uint64_t>> muon_minus_columns;
    std::set<std::pair<std::int32_t, std::uint64_t>> muon_plus_columns;
    std::map<std::int32_t, std::set<std::uint32_t>> selection_ordinals;
    for (auto const& column : table.dndx_columns) {
      if (!supportedNativePdg(column.pdg_id))
        throw std::invalid_argument("unsupported PROPOSAL native dN/dX PID");
      dndx_pids.emplace(column.pdg_id);
      registerMedium(column.pdg_id, column.medium_hash, "dN/dX columns");
      DndxIdentity const identity{column.pdg_id, column.process_id,
                                  column.component_hash};
      if (have_previous_dndx && identity < previous_dndx)
        throw std::invalid_argument(
            "PROPOSAL native dN/dX columns are not in canonical "
            "(pid, process, component) order");
      previous_dndx = identity;
      have_previous_dndx = true;
      if (!dndx_identities.emplace(identity).second)
        throw std::invalid_argument(
            "duplicate PROPOSAL native (pid, process, component) column");
      if (column.reserved != 0u ||
          !selection_ordinals[column.pdg_id]
               .emplace(column.selection_ordinal).second)
        throw std::invalid_argument(
            "invalid or duplicate PROPOSAL native selection ordinal");
      if (column.pdg_id == 13)
        muon_minus_columns.emplace(column.process_id, column.component_hash);
      else if (column.pdg_id == -13)
        muon_plus_columns.emplace(column.process_id, column.component_hash);

      if (!(column.component_weight > 0.) ||
          !std::isfinite(column.component_weight))
        throw std::invalid_argument("invalid PROPOSAL native component weight");
      if (!(column.medium_mean_excitation_energy_MeV > 0.) ||
          !std::isfinite(column.medium_mean_excitation_energy_MeV))
        throw std::invalid_argument(
            "invalid PROPOSAL native medium mean excitation energy");
      if (column.rate_model == NativeRateModel::BicubicSpline) {
        if (column.spline.rows < 2 || column.spline.columns < 2)
          throw std::invalid_argument("invalid native bicubic dimensions");
        if (column.spline.loss_axis.type != NativeAxisType::Linear)
          throw std::invalid_argument(
              "unsupported PROPOSAL native loss-axis transform");
        validateAxis(column.spline.energy_axis, column.spline.rows,
                     "dN/dX energy axis");
        validateAxis(column.spline.loss_axis, column.spline.columns,
                     "dN/dX loss axis");
        auto const count = static_cast<std::uint64_t>(column.spline.rows) *
                           column.spline.columns;
        if (count == 0 || column.spline.coefficient_offset > coefficient_count ||
            count > coefficient_count - column.spline.coefficient_offset)
          throw std::invalid_argument("native bicubic descriptor is out of bounds");
        bicubic_spans.emplace_back(
            column.spline.coefficient_offset,
            column.spline.coefficient_offset + count);
        if (column.spline.polynomial_coefficient_offset ==
            NoNativePolynomialCoefficients)
          throw std::invalid_argument(
              "native bicubic descriptor lacks polynomial coefficients");
        auto const cells =
            static_cast<std::uint64_t>(column.spline.rows - 1u) *
            (column.spline.columns - 1u);
        auto const offset = column.spline.polynomial_coefficient_offset;
        auto const available =
            table.bicubic_polynomial_coefficients.size();
        if (offset > available || cells > (available - offset) / 16u)
          throw std::invalid_argument(
              "native bicubic polynomial descriptor is out of bounds");
        polynomial_spans.emplace_back(offset, offset + 16u * cells);
      }
    }
    if (interaction_pids != dndx_pids)
      throw std::invalid_argument(
          "PROPOSAL native interaction and dN/dX PID sets differ");
    validateCompleteCoefficientPartition(
        std::move(bicubic_spans), table.bicubic_values.size(),
        "dN/dX bicubic splines");
    validateCompleteCoefficientPartition(
        std::move(polynomial_spans),
        table.bicubic_polynomial_coefficients.size(),
        "dN/dX bicubic polynomial splines");
    if (muon_minus_columns != muon_plus_columns)
      throw std::invalid_argument(
          "PROPOSAL native mu-/mu+ process-component sets are asymmetric");

    // The indirection must be a complete permutation, grouped by the same PID
    // ranges as the canonical dN/dX array, with ordinals exactly 0..N-1.
    std::vector<bool> selected_once(table.dndx_columns.size(), false);
    std::size_t selection_position = 0;
    for (auto const& [pdg, ordinals] : selection_ordinals) {
      std::uint32_t expected_ordinal = 0;
      for (auto const ordinal : ordinals) {
        if (ordinal != expected_ordinal++)
          throw std::invalid_argument(
              "PROPOSAL native selection ordinals are not contiguous");
        if (selection_position >= table.selection_column_indices.size())
          throw std::invalid_argument(
              "PROPOSAL native selection index is truncated");
        auto const column_index =
            table.selection_column_indices[selection_position++];
        if (column_index >= table.dndx_columns.size() ||
            selected_once[column_index])
          throw std::invalid_argument(
              "PROPOSAL native selection index is not a permutation");
        selected_once[column_index] = true;
        auto const& column = table.dndx_columns[column_index];
        if (column.pdg_id != pdg ||
            column.selection_ordinal != ordinal)
          throw std::invalid_argument(
              "PROPOSAL native selection index/order identity mismatch");
      }
    }
    if (selection_position != table.selection_column_indices.size() ||
        !std::all_of(selected_once.begin(), selected_once.end(),
                     [](bool value) { return value; }))
      throw std::invalid_argument(
          "PROPOSAL native selection index has incomplete coverage");

    // One process legitimately exports one dE/dX spline per material
    // component.  The stable source-vector ordinal is the component identity:
    // table_hash is only provenance and can legally be shared by two
    // components with identical interpolation definitions.  Repeating this
    // identity would make queryProposalNativeDedx() count the same physical
    // contribution twice.
    using DedxIdentity =
        std::tuple<std::int32_t, std::int32_t, std::uint64_t,
                   std::uint64_t>;
    std::set<DedxIdentity> dedx_identities;
    DedxIdentity previous_dedx{};
    bool have_previous_dedx = false;
    std::map<std::int32_t, std::set<std::int32_t>> dedx_processes;
    std::vector<CoefficientSpan> cubic_spans;
    std::set<std::int32_t> total_rate_pids;
    std::int32_t previous_total_rate_pdg =
        std::numeric_limits<std::int32_t>::min();
    for (auto const& column : table.total_rate_columns) {
      if (!supportedNativePdg(column.pdg_id) || column.reserved != 0u ||
          !total_rate_pids.emplace(column.pdg_id).second ||
          column.pdg_id < previous_total_rate_pdg)
        throw std::invalid_argument(
            "invalid, duplicate, or unordered native total-rate column");
      previous_total_rate_pdg = column.pdg_id;
      registerMedium(column.pdg_id, column.medium_hash,
                     "mean-free-path rate columns");
      validateCubicDescriptor(column.spline, table.cubic_values.size(),
                              "mean-free-path rate column");
      if (!(column.lower_energy_limit_MeV > 0.) ||
          !std::isfinite(column.lower_energy_limit_MeV) ||
          column.lower_energy_limit_MeV < column.spline.axis.low ||
          column.lower_energy_limit_MeV > column.spline.axis.high)
        throw std::invalid_argument(
            "invalid native mean-free-path rate energy domain");
      cubic_spans.emplace_back(
          column.spline.coefficient_offset,
          column.spline.coefficient_offset + column.spline.coefficient_count);
    }
    if (total_rate_pids != dndx_pids)
      throw std::invalid_argument(
          "PROPOSAL native total-rate and dN/dX PID sets differ");
    for (auto const& column : table.dedx_columns) {
      if (!supportedNativeChargedPdg(column.pdg_id) ||
          dndx_pids.count(column.pdg_id) == 0 || column.reserved != 0u)
        throw std::invalid_argument("unsupported PROPOSAL native dE/dX PID");
      DedxIdentity const identity{column.pdg_id, column.process_id,
                                  column.cross_section_hash,
                                  column.component_index};
      if (have_previous_dedx && identity < previous_dedx)
        throw std::invalid_argument(
            "PROPOSAL native dE/dX columns are not in canonical "
            "(pid, process, cross section, component index) order");
      previous_dedx = identity;
      have_previous_dedx = true;
      if (!dedx_identities.emplace(identity).second)
        throw std::invalid_argument(
            "duplicate PROPOSAL native (pid, process, cross section, "
            "component index) dE/dX column");
      dedx_processes[column.pdg_id].emplace(column.process_id);
      registerMedium(column.pdg_id, column.medium_hash, "dE/dX columns");
      validateCubicDescriptor(column.spline, table.cubic_values.size(),
                              "dE/dX column");
      if (!(column.component_weight > 0.) ||
          !std::isfinite(column.component_weight) ||
          !(column.lower_energy_limit_MeV > 0.) ||
          !std::isfinite(column.lower_energy_limit_MeV) ||
          column.lower_energy_limit_MeV > column.spline.axis.high)
        throw std::invalid_argument(
            "invalid PROPOSAL native dE/dX column metadata");
      cubic_spans.emplace_back(
          column.spline.coefficient_offset,
          column.spline.coefficient_offset + column.spline.coefficient_count);
    }

    std::map<std::int32_t, std::uint32_t> expected_dedx_ordinal;
    std::vector<bool> accumulated_once(table.dedx_columns.size(), false);
    std::int32_t previous_accumulation_pdg =
        std::numeric_limits<std::int32_t>::min();
    for (auto const index : table.dedx_accumulation_indices) {
      if (index >= table.dedx_columns.size() || accumulated_once[index])
        throw std::invalid_argument(
            "PROPOSAL native dE/dX accumulation index is duplicate or "
            "outside the descriptor array");
      accumulated_once[index] = true;
      auto const& column = table.dedx_columns[index];
      if (column.pdg_id < previous_accumulation_pdg ||
          column.accumulation_ordinal !=
              expected_dedx_ordinal[column.pdg_id]++)
        throw std::invalid_argument(
            "PROPOSAL native dE/dX accumulation order does not match its "
            "scalar source ordinal");
      previous_accumulation_pdg = column.pdg_id;
    }
    if (!std::all_of(accumulated_once.begin(), accumulated_once.end(),
                     [](bool value) { return value; }))
      throw std::invalid_argument(
          "PROPOSAL native dE/dX accumulation index has incomplete "
          "coverage");

    auto requireChargeSymmetricProcesses = [&](std::int32_t negative,
                                               std::int32_t positive,
                                               char const* description) {
      auto const left = dedx_processes.find(negative);
      auto const right = dedx_processes.find(positive);
      if ((left == dedx_processes.end()) !=
              (right == dedx_processes.end()) ||
          (left != dedx_processes.end() && left->second != right->second))
        throw std::invalid_argument(
            std::string{"asymmetric PROPOSAL native dE/dX processes for "} +
            description);
    };
    requireChargeSymmetricProcesses(11, -11, "electron/positron");
    requireChargeSymmetricProcesses(13, -13, "mu-/mu+");

    std::map<std::int32_t, std::uint32_t> utility_reverse_by_pid;
    for (auto const& column : table.utility_columns) {
      if (!supportedNativeChargedPdg(column.pdg_id) ||
          dndx_pids.count(column.pdg_id) == 0)
        throw std::invalid_argument("unsupported PROPOSAL native utility PID");
      if (!utility_reverse_by_pid.emplace(column.pdg_id, column.reverse).second)
        throw std::invalid_argument(
            "duplicate PROPOSAL native utility PID column");
      registerMedium(column.pdg_id, column.medium_hash, "utility columns");
      validateCubicDescriptor(column.spline, table.cubic_values.size(),
                              "utility column");
      if (!std::isfinite(column.particle_mass_MeV) ||
          !(column.particle_mass_MeV > 0.) ||
          !std::isfinite(column.lower_energy_limit_MeV) ||
          column.lower_energy_limit_MeV < column.particle_mass_MeV) {
        std::ostringstream message;
        message << "invalid PROPOSAL native utility energy domain for pdg="
                << column.pdg_id << ": mass="
                << column.particle_mass_MeV << " MeV, lower="
                << column.lower_energy_limit_MeV << " MeV";
        throw std::invalid_argument(message.str());
      }
      if (column.reverse > 1u ||
          column.lower_energy_limit_MeV < column.spline.axis.low ||
          column.lower_energy_limit_MeV > column.spline.axis.high)
        throw std::invalid_argument(
            "invalid PROPOSAL native utility direction/domain");
      cubic_spans.emplace_back(
          column.spline.coefficient_offset,
          column.spline.coefficient_offset + column.spline.coefficient_count);
    }

    for (auto const pdg_id : dndx_pids) {
      if (pdg_id == 22) continue;
      if (dedx_processes.count(pdg_id) == 0)
        throw std::invalid_argument(
            "PROPOSAL native charged PID has no dE/dX columns");
      if (utility_reverse_by_pid.count(pdg_id) == 0)
        throw std::invalid_argument(
            "PROPOSAL native charged PID has no utility column");
    }
    auto requireChargeSymmetricUtility = [&](std::int32_t negative,
                                             std::int32_t positive,
                                             char const* description) {
      auto const left = utility_reverse_by_pid.find(negative);
      auto const right = utility_reverse_by_pid.find(positive);
      if ((left == utility_reverse_by_pid.end()) !=
              (right == utility_reverse_by_pid.end()) ||
          (left != utility_reverse_by_pid.end() &&
           left->second != right->second))
        throw std::invalid_argument(
            std::string{"asymmetric PROPOSAL native utility direction for "} +
            description);
    };
    requireChargeSymmetricUtility(11, -11, "electron/positron");
    requireChargeSymmetricUtility(13, -13, "mu-/mu+");
    validateCompleteCoefficientPartition(
        std::move(cubic_spans), table.cubic_values.size(),
        "dE/dX and utility cubic splines");
  }

  Sha256Digest calculateProposalNativeHash(
      ProposalNativeTableSet const& table) {
    CanonicalEncoder encoder;
    encoder.unsigned32(table.format_version);
    encoder.text(table.proposal_version);
    encoder.text(table.cubic_interpolation_version);
    encoder.floating(table.constants.electron_mass_MeV);
    encoder.floating(table.constants.charged_pion_mass_MeV);
    encoder.floating(table.constants.fine_structure_constant);
    encoder.floating(table.constants.classical_electron_radius_cm);
    encoder.floating(table.constants.pi);
    encoder.floating(table.constants.sqrt_e);
    encoder.floating(table.stochastic_cut_MeV);
    encoder.sequence(
        table.interaction_identities,
        [&](NativeInteractionIdentity const& identity) {
          encoder.signed32(identity.pdg_id);
          encoder.unsigned32(identity.reserved);
          encoder.unsigned64(identity.medium_hash);
          encoder.unsigned64(identity.interaction_hash);
        });
    encoder.sequence(table.dndx_columns, [&](NativeDndxColumn const& column) {
      encoder.signed32(column.pdg_id);
      encoder.signed32(column.process_id);
      encoder.unsigned64(column.medium_hash);
      encoder.unsigned64(column.component_hash);
      encoder.unsigned64(column.cross_section_hash);
      encoder.unsigned64(column.table_hash);
      encoder.unsigned32(static_cast<std::uint32_t>(column.loss_transform));
      encoder.unsigned32(static_cast<std::uint32_t>(column.kinematic_model));
      encoder.unsigned32(static_cast<std::uint32_t>(column.rate_model));
      encodeAxis(encoder, column.spline.energy_axis);
      encodeAxis(encoder, column.spline.loss_axis);
      encoder.unsigned32(column.spline.rows);
      encoder.unsigned32(column.spline.columns);
      encoder.unsigned64(column.spline.coefficient_offset);
      encoder.unsigned64(column.spline.polynomial_coefficient_offset);
      encoder.floating(column.component_weight);
      encoder.floating(column.particle_mass_MeV);
      encoder.floating(column.component_nuclear_charge);
      encoder.floating(column.component_atomic_mass);
      encoder.floating(column.component_average_nucleon_mass_MeV);
      encoder.floating(column.direct_rate_factor);
      encoder.floating(column.lower_energy_limit_MeV);
      encoder.floating(column.absolute_cut_MeV);
      encoder.floating(column.relative_cut);
      encoder.floating(column.medium_mean_excitation_energy_MeV);
      encoder.unsigned32(column.selection_ordinal);
      encoder.unsigned32(column.reserved);
    });
    encoder.sequence(
        table.total_rate_columns,
        [&](NativeTotalRateColumn const& column) {
          encoder.signed32(column.pdg_id);
          encoder.unsigned32(column.reserved);
          encoder.unsigned64(column.medium_hash);
          encoder.unsigned64(column.table_hash);
          encodeCubic(encoder, column.spline);
          encoder.floating(column.lower_energy_limit_MeV);
        });
    encoder.sequence(
        table.selection_column_indices,
        [&](std::uint32_t index) { encoder.unsigned32(index); });
    encoder.sequence(table.dedx_columns, [&](NativeDedxColumn const& column) {
      encoder.signed32(column.pdg_id);
      encoder.signed32(column.process_id);
      encoder.unsigned64(column.medium_hash);
      encoder.unsigned64(column.cross_section_hash);
      encoder.unsigned64(column.component_index);
      encoder.unsigned64(column.table_hash);
      encodeCubic(encoder, column.spline);
      encoder.floating(column.component_weight);
      encoder.floating(column.lower_energy_limit_MeV);
      encoder.unsigned32(column.accumulation_ordinal);
      encoder.unsigned32(column.reserved);
    });
    encoder.sequence(
        table.dedx_accumulation_indices,
        [&](std::uint32_t index) { encoder.unsigned32(index); });
    encoder.sequence(table.utility_columns, [&](NativeUtilityColumn const& column) {
      encoder.signed32(column.pdg_id);
      encoder.unsigned64(column.medium_hash);
      encoder.unsigned64(column.table_hash);
      encodeCubic(encoder, column.spline);
      encoder.floating(column.particle_mass_MeV);
      encoder.floating(column.lower_energy_limit_MeV);
      encoder.unsigned32(column.reverse);
    });
    auto encodeDoubleVector = [&](std::vector<double> const& values) {
      encoder.sequence(values, [&](double value) { encoder.floating(value); });
    };
    encodeDoubleVector(table.bicubic_values);
    encodeDoubleVector(table.bicubic_derivative_energy);
    encodeDoubleVector(table.bicubic_derivative_loss);
    encodeDoubleVector(table.bicubic_mixed_derivative);
    encodeDoubleVector(table.bicubic_polynomial_coefficients);
    encodeDoubleVector(table.cubic_values);
    encodeDoubleVector(table.cubic_node_derivatives);
    return encoder.digest();
  }

  ProposalNativeTableSet exportProposalNativeTables(
      std::vector<proposal::NativeInteractionCalculatorView> const& interactions,
      std::vector<proposal::NativeContinuousCalculatorView> const& continuous) {
    ProposalNativeTableSet result;
    result.proposal_version = getPROPOSALVersion();
    result.cubic_interpolation_version = getCubicInterpolationVersion();
    result.constants = {PROPOSAL::ME, PROPOSAL::MPI, PROPOSAL::ALPHA,
                        PROPOSAL::RE, PROPOSAL::PI, PROPOSAL::SQRTE};
    requireSingleComposition(interactions, continuous);
    for (auto const& view : interactions) {
      if (!supportedNativeProjectile(view.projectile)) continue;
      auto const cut_MeV =
          static_cast<double>(view.stochastic_energy_cut / 1_MeV);
      if (!(cut_MeV > 0.)) continue;
      if (result.stochastic_cut_MeV == 0.)
        result.stochastic_cut_MeV = cut_MeV;
      else if (std::abs(result.stochastic_cut_MeV - cut_MeV) >
               1.e-12 * result.stochastic_cut_MeV)
        throw std::invalid_argument(
            "proposal-native calculators use inconsistent stochastic cuts");
    }

    // CORSIKA's five atmosphere nodes construct separate PROPOSAL Medium
    // objects whose hashes include their reference density. Cross sections
    // are rates per grammage and therefore depend on composition, not that
    // reference density. Export one deterministic representative calculator
    // per projectile and leave the local density to GPU tracking/LPM.
    std::map<Code, proposal::NativeInteractionCalculatorView const*>
        unique_interactions;
    for (auto const& view : interactions) {
      if (!supportedNativeProjectile(view.projectile)) continue;
      if (!view.medium || !view.interaction)
        throw std::invalid_argument("null PROPOSAL native interaction view");
      auto const found = unique_interactions.find(view.projectile);
      if (found == unique_interactions.end() ||
          view.medium_hash < found->second->medium_hash)
        unique_interactions[view.projectile] = &view;
    }

    for (auto const& entry : unique_interactions) {
      auto const& view = *entry.second;
      auto const pdg_id = static_cast<std::int32_t>(get_PDG(view.projectile));
      auto const medium_hash = static_cast<std::uint64_t>(view.medium_hash);
      result.interaction_identities.push_back(
          {pdg_id, 0u, medium_hash,
           static_cast<std::uint64_t>(view.interaction->GetHash())});
      std::uint32_t selection_ordinal = 0;
      std::uint32_t dedx_accumulation_ordinal = 0;
      for (auto const& cross_section : view.interaction->GetCrossSections()) {
        if (!cross_section)
          throw std::logic_error("null PROPOSAL cross section during native export");
        auto const process_id = static_cast<std::int32_t>(
            cross_section->GetInteractionType());
        auto const parametrization =
            cross_section->GetParametrizationName();
        auto const process_kinematic_model =
            classifyProposalNativeKinematicModel(
                process_id, parametrization, pdg_id);
        auto const first_process_column = result.dndx_columns.size();
        auto dndx = cross_section->ExportDndxInterpolation();
        for (auto const& source : dndx) {
          ++result.proposal_cache_table_count;
          if (source.cache_hit) ++result.proposal_cache_hit_count;
          auto const component = componentForHash(
              *view.medium, static_cast<std::uint64_t>(source.component_hash));
          if (source.parametrization != parametrization)
            throw std::invalid_argument(
                "PROPOSAL native dN/dX export changed its parametrization identity");
          auto const kinematic_model =
              classifyProposalNativeKinematicModel(
                  process_id, source.parametrization, pdg_id);
          if (kinematic_model != process_kinematic_model)
            throw std::logic_error(
                "PROPOSAL native kinematic classification is inconsistent");
          result.dndx_columns.push_back(
              {pdg_id,
               process_id,
               medium_hash,
               static_cast<std::uint64_t>(source.component_hash),
               static_cast<std::uint64_t>(source.cross_section_hash),
               static_cast<std::uint64_t>(source.table_hash),
               exportLossTransform(source.loss_transform),
               kinematic_model,
               NativeRateModel::BicubicSpline,
               // Pre-expand every native Hermite cell using the same Eigen
               // matrix products as CubicInterpolation 0.1.5.  Earlier
               // revisions did this only for the very small muon rates, but
               // the generic basis form can also perturb a tiny residual
               // quantile for photon/electron selection.  This remains a
               // deterministic layout conversion, not physics resampling.
               appendBicubic(result, source, true),
               source.component_weight,
               view.particle_mass_MeV,
               source.component_hash == view.medium_hash
                   ? 0.
                   : component.GetNucCharge(),
               source.component_hash == view.medium_hash
                   ? 0.
                   : component.GetAtomicNum(),
               source.component_hash == view.medium_hash
                   ? 0.
                   : component.GetAverageNucleonWeight(),
               0.,
               source.lower_energy_limit_MeV,
               source.absolute_cut_MeV,
               source.relative_cut});
        }

        // ParametrizationDirect processes have no interpolation object to
        // export.  The production CORSIKA configuration uses three closed
        // formulae; export only their immutable component parameters and
        // evaluate the same formula on the device.  This is not resampling.
        if (dndx.empty() && process_id == 1000000012 &&
            cross_section->GetParametrizationName() == "Heitler") {
          for (auto const& component : view.medium->GetComponents()) {
            auto const weight =
                view.medium->GetSumNucleons() /
                (component.GetAtomInMolecule() * component.GetAtomicNum());
            auto const factor =
                PROPOSAL::NA * component.GetNucCharge() /
                component.GetAtomicNum() * PROPOSAL::PI * PROPOSAL::RE *
                PROPOSAL::RE / weight;
            result.dndx_columns.push_back(
                {pdg_id,
                 process_id,
                 medium_hash,
                 static_cast<std::uint64_t>(component.GetHash()),
                 static_cast<std::uint64_t>(cross_section->GetHash()),
                 0,
                 NativeLossTransform::RelativeLog,
                 NativeKinematicModel::OnlyStochastic,
                 NativeRateModel::AnnihilationHeitler,
                 {},
                 1.,
                 view.particle_mass_MeV,
                 component.GetNucCharge(),
                 component.GetAtomicNum(),
                 component.GetAverageNucleonWeight(),
                 factor,
                 cross_section->GetLowerEnergyLim(),
                 0.,
                 0.});
          }
        } else if (dndx.empty() && process_id == 1000000016 &&
                   cross_section->GetParametrizationName() == "Sauter") {
          for (auto const& component : view.medium->GetComponents()) {
            auto const weight =
                view.medium->GetSumNucleons() /
                (component.GetAtomInMolecule() * component.GetAtomicNum());
            auto const factor =
                PROPOSAL::NA / component.GetAtomicNum() / weight;
            auto const z_alpha =
                component.GetNucCharge() * PROPOSAL::ALPHA;
            auto const threshold =
                z_alpha * z_alpha * PROPOSAL::ME / 2.;
            result.dndx_columns.push_back(
                {pdg_id,
                 process_id,
                 medium_hash,
                 static_cast<std::uint64_t>(component.GetHash()),
                 static_cast<std::uint64_t>(cross_section->GetHash()),
                 0,
                 NativeLossTransform::RelativeLog,
                 NativeKinematicModel::OnlyStochastic,
                 NativeRateModel::PhotoeffectSauter,
                 {},
                 1.,
                 view.particle_mass_MeV,
                 component.GetNucCharge(),
                 component.GetAtomicNum(),
                 component.GetAverageNucleonWeight(),
                 factor,
                 threshold,
                 0.,
                 0.});
          }
        } else if (
            dndx.empty() && process_id == 1000000014 &&
            cross_section->GetParametrizationName() ==
                "HeckC7Shadowing") {
          for (auto const& component : view.medium->GetComponents()) {
            auto const weight =
                view.medium->GetSumNucleons() /
                (component.GetAtomInMolecule() * component.GetAtomicNum());
            auto const factor =
                PROPOSAL::NA / component.GetAtomicNum() * 1.e-30 / weight;
            auto const threshold =
                PROPOSAL::MPI + PROPOSAL::MPI * PROPOSAL::MPI /
                                    (2. * component.GetAverageNucleonWeight());
            result.dndx_columns.push_back(
                {pdg_id,
                 process_id,
                 medium_hash,
                 static_cast<std::uint64_t>(component.GetHash()),
                 static_cast<std::uint64_t>(cross_section->GetHash()),
                 0,
                 NativeLossTransform::RelativeLog,
                 NativeKinematicModel::OnlyStochastic,
                 NativeRateModel::PhotoproductionHeckC7Shadowing,
                 {},
                 1.,
                 view.particle_mass_MeV,
                 component.GetNucCharge(),
                 component.GetAtomicNum(),
                 component.GetAverageNucleonWeight(),
                 factor,
                 threshold,
                 0.,
                 0.});
          }
        } else if (dndx.empty()) {
          std::ostringstream message;
          message << "proposal-native has no device representation for direct process "
                  << cross_section->GetParametrizationName() << " (" << process_id
                  << ")";
          throw std::invalid_argument(message.str());
        }

        auto const medium_mean_excitation_energy_MeV =
            1.e-6 * view.medium->GetI();
        if (!(medium_mean_excitation_energy_MeV > 0.) ||
            !std::isfinite(medium_mean_excitation_energy_MeV))
          throw std::invalid_argument(
              "invalid PROPOSAL medium mean excitation energy");
        for (auto index = first_process_column;
             index < result.dndx_columns.size(); ++index) {
          result.dndx_columns[index].medium_mean_excitation_energy_MeV =
              medium_mean_excitation_energy_MeV;
          result.dndx_columns[index].selection_ordinal =
              selection_ordinal++;
        }

        // PROPOSAL represents a process with no interpolated continuous-loss
        // contribution by returning an empty vector.  That is a valid state
        // (notably for direct or stochastic-only processes).  A non-empty
        // export, however, must contain a complete cubic spline; accepting a
        // default/degenerate entry here would silently remove dE/dX on device.
        auto const dedx_sources =
            cross_section->ExportDedxInterpolation();
        for (auto const& source : dedx_sources) {
          if (source.axis.nodes < 2 || source.spline.values.size() < 2 ||
              source.spline.values.size() !=
                  source.spline.node_derivatives.size()) {
            std::ostringstream message;
            message << "invalid non-empty PROPOSAL native dE/dX export: pid="
                    << pdg_id << ", process=" << process_id
                    << ", parametrization="
                    << cross_section->GetParametrizationName()
                    << ", axis_nodes=" << source.axis.nodes
                    << ", values=" << source.spline.values.size()
                    << ", derivatives="
                    << source.spline.node_derivatives.size();
            throw std::invalid_argument(message.str());
          }
          ++result.proposal_cache_table_count;
          if (source.cache_hit) ++result.proposal_cache_hit_count;
          result.dedx_columns.push_back(
              {pdg_id,
               process_id,
               medium_hash,
               static_cast<std::uint64_t>(source.cross_section_hash),
               static_cast<std::uint64_t>(source.component_index),
               static_cast<std::uint64_t>(source.table_hash),
               appendCubic(result, source.axis, source.has_value_transform,
                           source.value_transform, source.spline),
               source.component_weight,
               source.lower_energy_limit_MeV,
               dedx_accumulation_ordinal++,
               0u});
        }
      }

      auto const total_rate =
          view.interaction->ExportTotalRateInterpolation();
      if (!total_rate.available || total_rate.axis.nodes < 2 ||
          total_rate.spline.values.size() < 2 ||
          total_rate.spline.values.size() !=
              total_rate.spline.node_derivatives.size())
        throw std::invalid_argument(
            "proposal-native requires the interpolated Interaction "
            "mean-free-path rate table");
      ++result.proposal_cache_table_count;
      if (total_rate.cache_hit) ++result.proposal_cache_hit_count;
      result.total_rate_columns.push_back(
          {pdg_id,
           0u,
           medium_hash,
           static_cast<std::uint64_t>(total_rate.table_hash),
           appendCubic(result, total_rate.axis,
                       total_rate.has_value_transform,
                       total_rate.value_transform,
                       total_rate.spline),
           total_rate.lower_energy_limit_MeV});
    }

    std::map<Code, proposal::NativeContinuousCalculatorView const*>
        unique_continuous;
    for (auto const& view : continuous) {
      if (!supportedNativeProjectile(view.projectile)) continue;
      if (!view.medium || !view.displacement)
        throw std::invalid_argument("null PROPOSAL native continuous view");
      auto const found = unique_continuous.find(view.projectile);
      if (found == unique_continuous.end() ||
          view.medium_hash < found->second->medium_hash)
        unique_continuous[view.projectile] = &view;
    }
    for (auto const& entry : unique_continuous) {
      auto const& view = *entry.second;
      auto const source = view.displacement->ExportInterpolation();
      if (!source.available)
        throw std::invalid_argument(
            "proposal-native requires interpolated displacement calculators");
      ++result.proposal_cache_table_count;
      if (source.cache_hit) ++result.proposal_cache_hit_count;
      result.utility_columns.push_back(
          {static_cast<std::int32_t>(get_PDG(view.projectile)),
           static_cast<std::uint64_t>(view.medium_hash),
           static_cast<std::uint64_t>(source.table_hash),
           appendCubic(result, source.axis, source.has_value_transform,
                       source.value_transform, source.spline),
           view.particle_mass_MeV,
           source.lower_energy_limit_MeV,
           source.reverse ? 1u : 0u});
    }

    std::sort(result.dndx_columns.begin(), result.dndx_columns.end(),
              [](auto const& left, auto const& right) {
                return std::tie(left.pdg_id, left.medium_hash, left.process_id,
                                left.component_hash) <
                       std::tie(right.pdg_id, right.medium_hash, right.process_id,
                                right.component_hash);
              });
    result.selection_column_indices.resize(result.dndx_columns.size());
    std::iota(result.selection_column_indices.begin(),
              result.selection_column_indices.end(), 0u);
    std::sort(
        result.selection_column_indices.begin(),
        result.selection_column_indices.end(),
        [&](std::uint32_t left_index, std::uint32_t right_index) {
          auto const& left = result.dndx_columns[left_index];
          auto const& right = result.dndx_columns[right_index];
          return std::tie(left.pdg_id, left.selection_ordinal) <
                 std::tie(right.pdg_id, right.selection_ordinal);
        });
    std::sort(result.total_rate_columns.begin(),
              result.total_rate_columns.end(),
              [](auto const& left, auto const& right) {
                return left.pdg_id < right.pdg_id;
              });
    std::sort(
        result.interaction_identities.begin(),
        result.interaction_identities.end(),
        [](auto const& left, auto const& right) {
          return std::tie(left.pdg_id, left.medium_hash,
                          left.interaction_hash) <
                 std::tie(right.pdg_id, right.medium_hash,
                          right.interaction_hash);
        });
    std::sort(result.dedx_columns.begin(), result.dedx_columns.end(),
              [](auto const& left, auto const& right) {
                return std::tie(left.pdg_id, left.medium_hash, left.process_id,
                                left.cross_section_hash,
                                left.component_index, left.table_hash) <
                       std::tie(right.pdg_id, right.medium_hash, right.process_id,
                                right.cross_section_hash,
                                right.component_index, right.table_hash);
              });
    result.dedx_accumulation_indices.resize(result.dedx_columns.size());
    std::iota(result.dedx_accumulation_indices.begin(),
              result.dedx_accumulation_indices.end(), 0u);
    std::sort(
        result.dedx_accumulation_indices.begin(),
        result.dedx_accumulation_indices.end(),
        [&](std::uint32_t left_index, std::uint32_t right_index) {
          auto const& left = result.dedx_columns[left_index];
          auto const& right = result.dedx_columns[right_index];
          return std::tie(left.pdg_id, left.accumulation_ordinal) <
                 std::tie(right.pdg_id, right.accumulation_ordinal);
        });
    std::sort(result.utility_columns.begin(), result.utility_columns.end(),
              [](auto const& left, auto const& right) {
                return std::tie(left.pdg_id, left.medium_hash, left.table_hash) <
                       std::tie(right.pdg_id, right.medium_hash, right.table_hash);
              });
    result.proposal_cache_all_hit =
        result.proposal_cache_table_count != 0 &&
        result.proposal_cache_hit_count == result.proposal_cache_table_count;
    validateProposalNativeTable(result);
    result.content_hash = calculateProposalNativeHash(result);
    return result;
  }

} // namespace corsika::gpu::em::tables
