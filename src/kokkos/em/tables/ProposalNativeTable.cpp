/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/tables/ProposalNativeTable.hpp>
#include <corsika/kokkos/em/tables/ProposalNativeQueries.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace corsika::kokkos {

  namespace {

    class CanonicalEncoder {
    public:
      void unsigned32(std::uint32_t value) {
        for (unsigned int byte = 0; byte < 4; ++byte)
          data_.push_back(static_cast<std::uint8_t>(value >> (8 * byte)));
      }

      void signed32(std::int32_t value) { unsigned32(static_cast<std::uint32_t>(value)); }

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

      template <typename TValue, typename TFunction>
      void sequence(std::vector<TValue> const& values, TFunction encode) {
        unsigned64(values.size());
        for (auto const& value : values) encode(value);
      }

      sha256_digest_type digest() const { return sha256(data_); }

    private:
      std::vector<std::uint8_t> data_;
    };

    void validate_axis(NativeAxisDescriptor const& axis, std::uint32_t expected_nodes,
                       char const* description) {
      switch (axis.type_) {
        case NativeAxisType::Linear:
        case NativeAxisType::Exponential:
        case NativeAxisType::ExponentialMinusOne:
          break;
        default:
          throw std::invalid_argument(
              std::string{"invalid PROPOSAL native axis type in "} + description);
      }
      if (axis.nodes_ < 2 || (expected_nodes != 0 && axis.nodes_ != expected_nodes))
        throw std::invalid_argument(
            std::string{"invalid PROPOSAL native axis node count in "} + description);
      if (!std::isfinite(axis.low_) || !std::isfinite(axis.high_) ||
          !std::isfinite(axis.step_) || !(axis.high_ > axis.low_) || !(axis.step_ > 0.))
        throw std::invalid_argument(
            std::string{"invalid PROPOSAL native axis domain in "} + description);
      if ((axis.type_ == NativeAxisType::Exponential ||
           axis.type_ == NativeAxisType::ExponentialMinusOne) &&
          !(axis.low_ > 0.))
        throw std::invalid_argument(
            std::string{"non-positive PROPOSAL native logarithmic axis in "} +
            description);

      // Check the exact maps used on the device as well as their inputs.
      // This catches finite but internally inconsistent descriptors before a
      // kernel can turn them into NaN/Inf coordinates.
      auto const lower = proposal_native_detail::axis_transform(axis, axis.low_);
      auto const upper = proposal_native_detail::axis_transform(axis, axis.high_);
      auto const last = proposal_native_detail::axis_back_transform(
          axis, static_cast<double>(axis.nodes_ - 1u));
      if (!std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(last))
        throw std::invalid_argument(
            std::string{"non-finite PROPOSAL native axis mapping in "} + description);
      auto const endpoint_scale = std::max({1., std::abs(axis.high_), std::abs(last)});
      auto const coordinate_scale = std::max(1., static_cast<double>(axis.nodes_ - 1u));
      auto const tolerance = 128. * std::numeric_limits<double>::epsilon();
      if (std::abs(lower) > tolerance ||
          std::abs(upper - static_cast<double>(axis.nodes_ - 1u)) >
              tolerance * coordinate_scale ||
          std::abs(last - axis.high_) > tolerance * endpoint_scale)
        throw std::invalid_argument(
            std::string{"inconsistent PROPOSAL native axis step/domain in "} +
            description);
    }

    void validate_cubic_descriptor(NativeCubicDescriptor const& spline,
                                   std::size_t coefficient_count,
                                   char const* description) {
      if (spline.coefficientCount_ < 2 || spline.coefficientOffset_ > coefficient_count ||
          spline.coefficientCount_ > coefficient_count - spline.coefficientOffset_)
        throw std::invalid_argument(
            std::string{"native cubic spline is out of bounds in "} + description);
      validate_axis(spline.axis_, spline.coefficientCount_, description);
      if (spline.hasValueTransform_ > 1u)
        throw std::invalid_argument(
            std::string{"invalid native cubic value-transform flag in "} + description);
      if (spline.hasValueTransform_ != 0u) {
        // A function-value transform is an invertible coordinate map, not an
        // interpolation grid.  CubicInterpolation may therefore export only
        // one descriptive node; nodes/high are not used by axis_back_transform.
        auto const& transform = spline.valueTransform_;
        switch (transform.type_) {
          case NativeAxisType::Linear:
          case NativeAxisType::Exponential:
          case NativeAxisType::ExponentialMinusOne:
            break;
          default:
            throw std::invalid_argument(
                std::string{"invalid native cubic value-transform type in "} +
                description);
        }
        if (!std::isfinite(transform.low_) || !std::isfinite(transform.high_) ||
            !std::isfinite(transform.step_) || !(transform.step_ > 0.) ||
            ((transform.type_ == NativeAxisType::Exponential ||
              transform.type_ == NativeAxisType::ExponentialMinusOne) &&
             !(transform.low_ > 0.)))
          throw std::invalid_argument(
              std::string{"invalid native cubic value transform in "} + description);
      }
    }

    bool supported_native_pdg(std::int32_t pdg_id) {
      return pdg_id == 22 || pdg_id == 11 || pdg_id == -11 || pdg_id == 13 ||
             pdg_id == -13;
    }

    bool supported_native_charged_pdg(std::int32_t pdg_id) {
      return pdg_id == 11 || pdg_id == -11 || pdg_id == 13 || pdg_id == -13;
    }

    template <typename TValues>
    void require_finite_coefficients(TValues const& values, char const* description) {
      if (!std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument(
            std::string{"non-finite PROPOSAL native coefficients in "} + description);
    }

    using coefficient_span_type = std::pair<std::uint64_t, std::uint64_t>;

    void validate_complete_coefficient_partition(std::vector<coefficient_span_type> spans,
                                                 std::size_t coefficient_count,
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
            std::string{"unreferenced PROPOSAL native coefficients in "} + description);
    }

    void encode_axis(CanonicalEncoder& encoder, NativeAxisDescriptor const& axis) {
      encoder.unsigned32(static_cast<std::uint32_t>(axis.type_));
      encoder.unsigned32(axis.nodes_);
      encoder.floating(axis.low_);
      encoder.floating(axis.high_);
      encoder.floating(axis.step_);
    }

    void encode_cubic(CanonicalEncoder& encoder, NativeCubicDescriptor const& spline) {
      encode_axis(encoder, spline.axis_);
      encode_axis(encoder, spline.valueTransform_);
      encoder.unsigned64(spline.coefficientOffset_);
      encoder.unsigned32(spline.coefficientCount_);
      encoder.unsigned32(spline.hasValueTransform_);
    }

  } // namespace

  ProposalNativeDeviceView make_proposal_native_host_view(
      ProposalNativeTableSet const& table) {
    return {
        NativeArrayView<NativeDndxColumn, std::uint32_t>{table.dndxColumns_},
        NativeArrayView<NativeTotalRateColumn, std::uint32_t>{table.totalRateColumns_},
        NativeArrayView<std::uint32_t, std::uint32_t>{table.selectionColumnIndices_},
        NativeArrayView<NativeDedxColumn, std::uint32_t>{table.dedxColumns_},
        NativeArrayView<std::uint32_t, std::uint32_t>{table.dedxAccumulationIndices_},
        NativeArrayView<NativeUtilityColumn, std::uint32_t>{table.utilityColumns_},
        NativeArrayView<double, std::uint64_t>{table.bicubicValues_},
        NativeArrayView<double, std::uint64_t>{table.bicubicDerivativeEnergy_},
        NativeArrayView<double, std::uint64_t>{table.bicubicDerivativeLoss_},
        NativeArrayView<double, std::uint64_t>{table.bicubicMixedDerivative_},
        NativeArrayView<double, std::uint64_t>{table.bicubicPolynomialCoefficients_},
        NativeArrayView<double, std::uint64_t>{table.cubicValues_},
        NativeArrayView<double, std::uint64_t>{table.cubicNodeDerivatives_},
        table.constants_};
  }

  std::size_t proposal_native_table_bytes(ProposalNativeTableSet const& table) {
    return table.dndxColumns_.size() * sizeof(NativeDndxColumn) +
           table.totalRateColumns_.size() * sizeof(NativeTotalRateColumn) +
           table.selectionColumnIndices_.size() * sizeof(std::uint32_t) +
           table.dedxColumns_.size() * sizeof(NativeDedxColumn) +
           table.dedxAccumulationIndices_.size() * sizeof(std::uint32_t) +
           table.utilityColumns_.size() * sizeof(NativeUtilityColumn) +
           (table.bicubicValues_.size() + table.bicubicDerivativeEnergy_.size() +
            table.bicubicDerivativeLoss_.size() + table.bicubicMixedDerivative_.size() +
            table.bicubicPolynomialCoefficients_.size() + table.cubicValues_.size() +
            table.cubicNodeDerivatives_.size()) *
               sizeof(double);
  }

  void validate_proposal_native_table(ProposalNativeTableSet const& table) {
    if (table.formatVersion_ != ProposalNativeTableFormatVersion)
      throw std::invalid_argument("unsupported PROPOSAL native table format");
    if (table.proposalVersion_ != "7.6.2" || table.cubicInterpolationVersion_ != "0.1.5")
      throw std::invalid_argument("PROPOSAL native dependency version mismatch");
    if (!(table.constants_.electronMassMeV_ > 0.) ||
        !(table.constants_.muonMassMeV_ > 0.) ||
        !(table.constants_.chargedPionMassMeV_ > 0.) ||
        !(table.constants_.fineStructureConstant_ > 0.) ||
        !(table.constants_.classicalElectronRadiusCm_ > 0.) ||
        !(table.constants_.pi_ > 3. && table.constants_.pi_ < 4.) ||
        !(table.constants_.sqrtE_ > 1. && table.constants_.sqrtE_ < 2.))
      throw std::invalid_argument("invalid PROPOSAL native physical constants");
    if (table.dndxColumns_.empty())
      throw std::invalid_argument("PROPOSAL native table has no rate columns");
    if (table.totalRateColumns_.empty())
      throw std::invalid_argument(
          "PROPOSAL native table has no mean-free-path rate columns");
    if (table.selectionColumnIndices_.size() != table.dndxColumns_.size())
      throw std::invalid_argument(
          "PROPOSAL native selection index does not cover dN/dX columns");
    if (table.dedxAccumulationIndices_.size() != table.dedxColumns_.size())
      throw std::invalid_argument(
          "PROPOSAL native dE/dX accumulation index does not cover all "
          "continuous-loss columns");
    if (!(table.stochasticCutMeV_ > 0.) || !std::isfinite(table.stochasticCutMeV_))
      throw std::invalid_argument("invalid PROPOSAL native stochastic cut");
    auto const coefficient_count = table.bicubicValues_.size();
    if (table.bicubicDerivativeEnergy_.size() != coefficient_count ||
        table.bicubicDerivativeLoss_.size() != coefficient_count ||
        table.bicubicMixedDerivative_.size() != coefficient_count)
      throw std::invalid_argument("PROPOSAL native bicubic arrays differ in size");
    if (table.cubicValues_.size() != table.cubicNodeDerivatives_.size())
      throw std::invalid_argument("PROPOSAL native cubic arrays differ in size");
    require_finite_coefficients(table.bicubicValues_, "bicubic values");
    require_finite_coefficients(table.bicubicDerivativeEnergy_,
                                "bicubic energy derivatives");
    require_finite_coefficients(table.bicubicDerivativeLoss_, "bicubic loss derivatives");
    require_finite_coefficients(table.bicubicMixedDerivative_,
                                "bicubic mixed derivatives");
    require_finite_coefficients(table.bicubicPolynomialCoefficients_,
                                "bicubic polynomial coefficients");
    require_finite_coefficients(table.cubicValues_, "cubic values");
    require_finite_coefficients(table.cubicNodeDerivatives_, "cubic node derivatives");

    // find_dndx() performs a binary search on exactly this identity.  Enforce
    // both its canonical order and its uniqueness before uploading the table.
    // Medium is deliberately absent from the device lookup key, so it must be
    // constant throughout each PID range.
    using dndx_identity_type = std::tuple<std::int32_t, std::int32_t, std::uint64_t>;
    std::set<dndx_identity_type> dndx_identities;
    std::map<std::int32_t, std::uint64_t> medium_by_pid;
    std::set<std::int32_t> interaction_pids;
    std::set<std::int32_t> dndx_pids;
    std::vector<coefficient_span_type> bicubic_spans;
    std::vector<coefficient_span_type> polynomial_spans;
    auto registerMedium = [&](std::int32_t pdg_id, std::uint64_t medium_hash,
                              char const* description) {
      auto const [found, inserted] = medium_by_pid.emplace(pdg_id, medium_hash);
      if (!inserted && found->second != medium_hash) {
        std::ostringstream message;
        message << "PROPOSAL native table has multiple media for pdg=" << pdg_id
                << " (detected in " << description << ")";
        throw std::invalid_argument(message.str());
      }
    };
    for (auto const& identity : table.interactionIdentities_) {
      if (!supported_native_pdg(identity.pdgId_) || identity.reserved_ != 0u)
        throw std::invalid_argument(
            "invalid PROPOSAL native interaction identity PID/reserved field");
      if (!interaction_pids.emplace(identity.pdgId_).second)
        throw std::invalid_argument("duplicate PROPOSAL native interaction identity PID");
      registerMedium(identity.pdgId_, identity.mediumHash_, "interaction identities");
    }

    dndx_identity_type previous_dndx{};
    bool have_previous_dndx = false;
    std::set<std::pair<std::int32_t, std::uint64_t>> muon_minus_columns;
    std::set<std::pair<std::int32_t, std::uint64_t>> muon_plus_columns;
    std::map<std::int32_t, std::set<std::uint32_t>> selection_ordinals;
    for (auto const& column : table.dndxColumns_) {
      if (!supported_native_pdg(column.pdgId_))
        throw std::invalid_argument("unsupported PROPOSAL native dN/dX PID");
      dndx_pids.emplace(column.pdgId_);
      registerMedium(column.pdgId_, column.mediumHash_, "dN/dX columns");
      dndx_identity_type const identity{column.pdgId_, column.processId_,
                                        column.componentHash_};
      if (have_previous_dndx && identity < previous_dndx)
        throw std::invalid_argument(
            "PROPOSAL native dN/dX columns are not in canonical "
            "(pid, process, component) order");
      previous_dndx = identity;
      have_previous_dndx = true;
      if (!dndx_identities.emplace(identity).second)
        throw std::invalid_argument(
            "duplicate PROPOSAL native (pid, process, component) column");
      if (column.reserved_ != 0u ||
          !selection_ordinals[column.pdgId_].emplace(column.selectionOrdinal_).second)
        throw std::invalid_argument(
            "invalid or duplicate PROPOSAL native selection ordinal");
      if (column.pdgId_ == 13)
        muon_minus_columns.emplace(column.processId_, column.componentHash_);
      else if (column.pdgId_ == -13)
        muon_plus_columns.emplace(column.processId_, column.componentHash_);

      if (!(column.componentWeight_ > 0.) || !std::isfinite(column.componentWeight_))
        throw std::invalid_argument("invalid PROPOSAL native component weight");
      if (!(column.mediumMeanExcitationEnergyMeV_ > 0.) ||
          !std::isfinite(column.mediumMeanExcitationEnergyMeV_))
        throw std::invalid_argument(
            "invalid PROPOSAL native medium mean excitation energy");
      if (column.rateModel_ == NativeRateModel::BicubicSpline) {
        if (column.spline_.rows_ < 2 || column.spline_.columns_ < 2)
          throw std::invalid_argument("invalid native bicubic dimensions");
        if (column.spline_.lossAxis_.type_ != NativeAxisType::Linear)
          throw std::invalid_argument("unsupported PROPOSAL native loss-axis transform");
        validate_axis(column.spline_.energyAxis_, column.spline_.rows_,
                      "dN/dX energy axis");
        validate_axis(column.spline_.lossAxis_, column.spline_.columns_,
                      "dN/dX loss axis");
        auto const count =
            static_cast<std::uint64_t>(column.spline_.rows_) * column.spline_.columns_;
        if (count == 0 || column.spline_.coefficientOffset_ > coefficient_count ||
            count > coefficient_count - column.spline_.coefficientOffset_)
          throw std::invalid_argument("native bicubic descriptor is out of bounds");
        bicubic_spans.emplace_back(column.spline_.coefficientOffset_,
                                   column.spline_.coefficientOffset_ + count);
        if (column.spline_.polynomialCoefficientOffset_ == NoNativePolynomialCoefficients)
          throw std::invalid_argument(
              "native bicubic descriptor lacks polynomial coefficients");
        auto const cells = static_cast<std::uint64_t>(column.spline_.rows_ - 1u) *
                           (column.spline_.columns_ - 1u);
        auto const offset = column.spline_.polynomialCoefficientOffset_;
        auto const available = table.bicubicPolynomialCoefficients_.size();
        if (offset > available || cells > (available - offset) / 16u)
          throw std::invalid_argument(
              "native bicubic polynomial descriptor is out of bounds");
        polynomial_spans.emplace_back(offset, offset + 16u * cells);
      }
    }
    if (interaction_pids != dndx_pids)
      throw std::invalid_argument(
          "PROPOSAL native interaction and dN/dX PID sets differ");
    validate_complete_coefficient_partition(
        std::move(bicubic_spans), table.bicubicValues_.size(), "dN/dX bicubic splines");
    validate_complete_coefficient_partition(std::move(polynomial_spans),
                                            table.bicubicPolynomialCoefficients_.size(),
                                            "dN/dX bicubic polynomial splines");
    if (muon_minus_columns != muon_plus_columns)
      throw std::invalid_argument(
          "PROPOSAL native mu-/mu+ process-component sets are asymmetric");

    // The indirection must be a complete permutation, grouped by the same PID
    // ranges as the canonical dN/dX array, with ordinals exactly 0..N-1.
    std::vector<bool> selected_once(table.dndxColumns_.size(), false);
    std::size_t selection_position = 0;
    for (auto const& [pdg, ordinals] : selection_ordinals) {
      std::uint32_t expected_ordinal = 0;
      for (auto const ordinal : ordinals) {
        if (ordinal != expected_ordinal++)
          throw std::invalid_argument(
              "PROPOSAL native selection ordinals are not contiguous");
        if (selection_position >= table.selectionColumnIndices_.size())
          throw std::invalid_argument("PROPOSAL native selection index is truncated");
        auto const column_index = table.selectionColumnIndices_[selection_position++];
        if (column_index >= table.dndxColumns_.size() || selected_once[column_index])
          throw std::invalid_argument(
              "PROPOSAL native selection index is not a permutation");
        selected_once[column_index] = true;
        auto const& column = table.dndxColumns_[column_index];
        if (column.pdgId_ != pdg || column.selectionOrdinal_ != ordinal)
          throw std::invalid_argument(
              "PROPOSAL native selection index/order identity mismatch");
      }
    }
    if (selection_position != table.selectionColumnIndices_.size() ||
        !std::all_of(selected_once.begin(), selected_once.end(),
                     [](bool value) { return value; }))
      throw std::invalid_argument(
          "PROPOSAL native selection index has incomplete coverage");

    // One process legitimately exports one dE/dX spline per material
    // component.  The stable source-vector ordinal is the component identity:
    // table_hash is only provenance and can legally be shared by two
    // components with identical interpolation definitions.  Repeating this
    // identity would make query_proposal_native_dedx() count the same physical
    // contribution twice.
    using dedx_identity_type =
        std::tuple<std::int32_t, std::int32_t, std::uint64_t, std::uint64_t>;
    std::set<dedx_identity_type> dedx_identities;
    dedx_identity_type previous_dedx{};
    bool have_previous_dedx = false;
    std::map<std::int32_t, std::set<std::int32_t>> dedx_processes;
    std::vector<coefficient_span_type> cubic_spans;
    std::set<std::int32_t> total_rate_pids;
    std::int32_t previous_total_rate_pdg = std::numeric_limits<std::int32_t>::min();
    for (auto const& column : table.totalRateColumns_) {
      if (!supported_native_pdg(column.pdgId_) || column.reserved_ != 0u ||
          !total_rate_pids.emplace(column.pdgId_).second ||
          column.pdgId_ < previous_total_rate_pdg)
        throw std::invalid_argument(
            "invalid, duplicate, or unordered native total-rate column");
      previous_total_rate_pdg = column.pdgId_;
      registerMedium(column.pdgId_, column.mediumHash_, "mean-free-path rate columns");
      validate_cubic_descriptor(column.spline_, table.cubicValues_.size(),
                                "mean-free-path rate column");
      if (!(column.lowerEnergyLimitMeV_ > 0.) ||
          !std::isfinite(column.lowerEnergyLimitMeV_) ||
          column.lowerEnergyLimitMeV_ < column.spline_.axis_.low_ ||
          column.lowerEnergyLimitMeV_ > column.spline_.axis_.high_)
        throw std::invalid_argument("invalid native mean-free-path rate energy domain");
      cubic_spans.emplace_back(
          column.spline_.coefficientOffset_,
          column.spline_.coefficientOffset_ + column.spline_.coefficientCount_);
    }
    if (total_rate_pids != dndx_pids)
      throw std::invalid_argument("PROPOSAL native total-rate and dN/dX PID sets differ");
    for (auto const& column : table.dedxColumns_) {
      if (!supported_native_charged_pdg(column.pdgId_) ||
          dndx_pids.count(column.pdgId_) == 0 || column.reserved_ != 0u)
        throw std::invalid_argument("unsupported PROPOSAL native dE/dX PID");
      dedx_identity_type const identity{column.pdgId_, column.processId_,
                                        column.crossSectionHash_, column.componentIndex_};
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
      dedx_processes[column.pdgId_].emplace(column.processId_);
      registerMedium(column.pdgId_, column.mediumHash_, "dE/dX columns");
      validate_cubic_descriptor(column.spline_, table.cubicValues_.size(),
                                "dE/dX column");
      if (!(column.componentWeight_ > 0.) || !std::isfinite(column.componentWeight_) ||
          !(column.lowerEnergyLimitMeV_ > 0.) ||
          !std::isfinite(column.lowerEnergyLimitMeV_) ||
          column.lowerEnergyLimitMeV_ > column.spline_.axis_.high_)
        throw std::invalid_argument("invalid PROPOSAL native dE/dX column metadata");
      cubic_spans.emplace_back(
          column.spline_.coefficientOffset_,
          column.spline_.coefficientOffset_ + column.spline_.coefficientCount_);
    }

    std::map<std::int32_t, std::uint32_t> expected_dedx_ordinal;
    std::vector<bool> accumulated_once(table.dedxColumns_.size(), false);
    std::int32_t previous_accumulation_pdg = std::numeric_limits<std::int32_t>::min();
    for (auto const index : table.dedxAccumulationIndices_) {
      if (index >= table.dedxColumns_.size() || accumulated_once[index])
        throw std::invalid_argument(
            "PROPOSAL native dE/dX accumulation index is duplicate or "
            "outside the descriptor array");
      accumulated_once[index] = true;
      auto const& column = table.dedxColumns_[index];
      if (column.pdgId_ < previous_accumulation_pdg ||
          column.accumulationOrdinal_ != expected_dedx_ordinal[column.pdgId_]++)
        throw std::invalid_argument(
            "PROPOSAL native dE/dX accumulation order does not match its "
            "scalar source ordinal");
      previous_accumulation_pdg = column.pdgId_;
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
      if ((left == dedx_processes.end()) != (right == dedx_processes.end()) ||
          (left != dedx_processes.end() && left->second != right->second))
        throw std::invalid_argument(
            std::string{"asymmetric PROPOSAL native dE/dX processes for "} + description);
    };
    requireChargeSymmetricProcesses(11, -11, "electron/positron");
    requireChargeSymmetricProcesses(13, -13, "mu-/mu+");

    std::map<std::int32_t, std::uint32_t> utility_reverse_by_pid;
    for (auto const& column : table.utilityColumns_) {
      if (!supported_native_charged_pdg(column.pdgId_) ||
          dndx_pids.count(column.pdgId_) == 0)
        throw std::invalid_argument("unsupported PROPOSAL native utility PID");
      if (!utility_reverse_by_pid.emplace(column.pdgId_, column.reverse_).second)
        throw std::invalid_argument("duplicate PROPOSAL native utility PID column");
      registerMedium(column.pdgId_, column.mediumHash_, "utility columns");
      validate_cubic_descriptor(column.spline_, table.cubicValues_.size(),
                                "utility column");
      if (!std::isfinite(column.particleMassMeV_) || !(column.particleMassMeV_ > 0.) ||
          !std::isfinite(column.lowerEnergyLimitMeV_) ||
          column.lowerEnergyLimitMeV_ < column.particleMassMeV_) {
        std::ostringstream message;
        message << "invalid PROPOSAL native utility energy domain for pdg="
                << column.pdgId_ << ": mass=" << column.particleMassMeV_
                << " MeV, lower=" << column.lowerEnergyLimitMeV_ << " MeV";
        throw std::invalid_argument(message.str());
      }
      if (column.reverse_ > 1u ||
          column.lowerEnergyLimitMeV_ < column.spline_.axis_.low_ ||
          column.lowerEnergyLimitMeV_ > column.spline_.axis_.high_)
        throw std::invalid_argument("invalid PROPOSAL native utility direction/domain");
      cubic_spans.emplace_back(
          column.spline_.coefficientOffset_,
          column.spline_.coefficientOffset_ + column.spline_.coefficientCount_);
    }

    for (auto const pdg_id : dndx_pids) {
      if (pdg_id == 22) continue;
      if (dedx_processes.count(pdg_id) == 0)
        throw std::invalid_argument("PROPOSAL native charged PID has no dE/dX columns");
      if (utility_reverse_by_pid.count(pdg_id) == 0)
        throw std::invalid_argument("PROPOSAL native charged PID has no utility column");
    }
    auto requireChargeSymmetricUtility = [&](std::int32_t negative, std::int32_t positive,
                                             char const* description) {
      auto const left = utility_reverse_by_pid.find(negative);
      auto const right = utility_reverse_by_pid.find(positive);
      if ((left == utility_reverse_by_pid.end()) !=
              (right == utility_reverse_by_pid.end()) ||
          (left != utility_reverse_by_pid.end() && left->second != right->second))
        throw std::invalid_argument(
            std::string{"asymmetric PROPOSAL native utility direction for "} +
            description);
    };
    requireChargeSymmetricUtility(11, -11, "electron/positron");
    requireChargeSymmetricUtility(13, -13, "mu-/mu+");
    validate_complete_coefficient_partition(std::move(cubic_spans),
                                            table.cubicValues_.size(),
                                            "dE/dX and utility cubic splines");
  }

  sha256_digest_type calculate_proposal_native_hash(ProposalNativeTableSet const& table) {
    CanonicalEncoder encoder;
    encoder.unsigned32(table.formatVersion_);
    encoder.text(table.proposalVersion_);
    encoder.text(table.cubicInterpolationVersion_);
    encoder.floating(table.constants_.electronMassMeV_);
    encoder.floating(table.constants_.chargedPionMassMeV_);
    encoder.floating(table.constants_.fineStructureConstant_);
    encoder.floating(table.constants_.classicalElectronRadiusCm_);
    encoder.floating(table.constants_.pi_);
    encoder.floating(table.constants_.sqrtE_);
    encoder.floating(table.constants_.muonMassMeV_);
    encoder.floating(table.stochasticCutMeV_);
    encoder.sequence(table.interactionIdentities_,
                     [&](NativeInteractionIdentity const& identity) {
                       encoder.signed32(identity.pdgId_);
                       encoder.unsigned32(identity.reserved_);
                       encoder.unsigned64(identity.mediumHash_);
                       encoder.unsigned64(identity.interactionHash_);
                     });
    encoder.sequence(table.dndxColumns_, [&](NativeDndxColumn const& column) {
      encoder.signed32(column.pdgId_);
      encoder.signed32(column.processId_);
      encoder.unsigned64(column.mediumHash_);
      encoder.unsigned64(column.componentHash_);
      encoder.unsigned64(column.crossSectionHash_);
      encoder.unsigned64(column.tableHash_);
      encoder.unsigned32(static_cast<std::uint32_t>(column.lossTransform_));
      encoder.unsigned32(static_cast<std::uint32_t>(column.kinematicModel_));
      encoder.unsigned32(static_cast<std::uint32_t>(column.rateModel_));
      encode_axis(encoder, column.spline_.energyAxis_);
      encode_axis(encoder, column.spline_.lossAxis_);
      encoder.unsigned32(column.spline_.rows_);
      encoder.unsigned32(column.spline_.columns_);
      encoder.unsigned64(column.spline_.coefficientOffset_);
      encoder.unsigned64(column.spline_.polynomialCoefficientOffset_);
      encoder.floating(column.componentWeight_);
      encoder.floating(column.particleMassMeV_);
      encoder.floating(column.componentNuclearCharge_);
      encoder.floating(column.componentAtomicMass_);
      encoder.floating(column.componentAverageNucleonMassMeV_);
      encoder.floating(column.directRateFactor_);
      encoder.floating(column.lowerEnergyLimitMeV_);
      encoder.floating(column.absoluteCutMeV_);
      encoder.floating(column.relativeCut_);
      encoder.floating(column.mediumMeanExcitationEnergyMeV_);
      encoder.unsigned32(column.selectionOrdinal_);
      encoder.unsigned32(column.reserved_);
    });
    encoder.sequence(table.totalRateColumns_, [&](NativeTotalRateColumn const& column) {
      encoder.signed32(column.pdgId_);
      encoder.unsigned32(column.reserved_);
      encoder.unsigned64(column.mediumHash_);
      encoder.unsigned64(column.tableHash_);
      encode_cubic(encoder, column.spline_);
      encoder.floating(column.lowerEnergyLimitMeV_);
    });
    encoder.sequence(table.selectionColumnIndices_,
                     [&](std::uint32_t index) { encoder.unsigned32(index); });
    encoder.sequence(table.dedxColumns_, [&](NativeDedxColumn const& column) {
      encoder.signed32(column.pdgId_);
      encoder.signed32(column.processId_);
      encoder.unsigned64(column.mediumHash_);
      encoder.unsigned64(column.crossSectionHash_);
      encoder.unsigned64(column.componentIndex_);
      encoder.unsigned64(column.tableHash_);
      encode_cubic(encoder, column.spline_);
      encoder.floating(column.componentWeight_);
      encoder.floating(column.lowerEnergyLimitMeV_);
      encoder.unsigned32(column.accumulationOrdinal_);
      encoder.unsigned32(column.reserved_);
    });
    encoder.sequence(table.dedxAccumulationIndices_,
                     [&](std::uint32_t index) { encoder.unsigned32(index); });
    encoder.sequence(table.utilityColumns_, [&](NativeUtilityColumn const& column) {
      encoder.signed32(column.pdgId_);
      encoder.unsigned64(column.mediumHash_);
      encoder.unsigned64(column.tableHash_);
      encode_cubic(encoder, column.spline_);
      encoder.floating(column.particleMassMeV_);
      encoder.floating(column.lowerEnergyLimitMeV_);
      encoder.unsigned32(column.reverse_);
    });
    auto encodeDoubleVector = [&](std::vector<double> const& values) {
      encoder.sequence(values, [&](double value) { encoder.floating(value); });
    };
    encodeDoubleVector(table.bicubicValues_);
    encodeDoubleVector(table.bicubicDerivativeEnergy_);
    encodeDoubleVector(table.bicubicDerivativeLoss_);
    encodeDoubleVector(table.bicubicMixedDerivative_);
    encodeDoubleVector(table.bicubicPolynomialCoefficients_);
    encodeDoubleVector(table.cubicValues_);
    encodeDoubleVector(table.cubicNodeDerivatives_);
    return encoder.digest();
  }

} // namespace corsika::kokkos
