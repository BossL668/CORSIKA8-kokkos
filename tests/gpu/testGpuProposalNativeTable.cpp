/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <Eigen/Dense>
#include <PROPOSAL/PROPOSAL.h>

#include <array>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/EpairFinalState.hpp>
#include <corsika/gpu/em/CudaBremsFinalState.hpp>
#include <corsika/gpu/em/tables/CudaProposalNativeTable.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>
#include <corsika/gpu/em/tables/ProposalNativeAux.hpp>
#include <corsika/gpu/em/tables/ProposalNativeTableExporter.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>

namespace {
  using namespace corsika;
  using namespace corsika::gpu::em::tables;

  std::size_t checks{};

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
  }

  template <typename Mutation>
  void requireStructuralRejection(ProposalNativeTableSet const& source,
                                  Mutation mutation,
                                  std::string const& message) {
    auto damaged = source;
    mutation(damaged);
    bool rejected = false;
    try {
      validateProposalNativeTable(damaged);
    } catch (std::invalid_argument const&) {
      rejected = true;
    }
    require(rejected, message);
  }

  void requireClose(double actual, double expected, double relative,
                    std::string const& message) {
    ++checks;
    auto const scale = std::max(std::abs(expected), 1.e-300);
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > relative * scale) {
      std::ostringstream details;
      details << std::setprecision(17) << message << ": actual=" << actual
              << ", expected=" << expected
              << ", relative_difference="
              << std::abs(actual - expected) / scale;
      throw std::runtime_error(details.str());
    }
  }

  std::uint64_t positiveUlpDistance(double left, double right) {
    if (!(left >= 0.) || !(right >= 0.) || !std::isfinite(left) ||
        !std::isfinite(right))
      return std::numeric_limits<std::uint64_t>::max();
    std::uint64_t left_bits{};
    std::uint64_t right_bits{};
    std::memcpy(&left_bits, &left, sizeof(left));
    std::memcpy(&right_bits, &right, sizeof(right));
    return left_bits > right_bits ? left_bits - right_bits
                                  : right_bits - left_bits;
  }

  double relativeDifference(double left, double right) {
    return std::abs(left - right) /
           std::max({std::abs(left), std::abs(right), 1.e-300});
  }

  std::string preciseDouble(double value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
  }

  template <typename ByteContainer>
  std::string hexadecimalBytes(ByteContainer const& bytes) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (auto const byte : bytes)
      stream << std::setw(2) << static_cast<unsigned int>(byte);
    return stream.str();
  }

  template <typename T>
  bool byteIdentical(std::vector<T> const& left,
                     std::vector<T> const& right) {
    static_assert(std::is_trivially_copyable_v<T>);
    return left.size() == right.size() &&
           (left.empty() ||
            std::memcmp(left.data(), right.data(),
                        left.size() * sizeof(T)) == 0);
  }

  bool exportedArraysIdentical(ProposalNativeTableSet const& left,
                               ProposalNativeTableSet const& right) {
    // Descriptor structs are hashed field by field by the canonical encoder;
    // comparing their raw storage would also compare unspecified C++ padding.
    // Coefficient arrays contain no padding and must be bit-identical.
    return left.content_hash == right.content_hash &&
           byteIdentical(left.bicubic_values, right.bicubic_values) &&
           byteIdentical(left.bicubic_derivative_energy,
                         right.bicubic_derivative_energy) &&
           byteIdentical(left.bicubic_derivative_loss,
                         right.bicubic_derivative_loss) &&
           byteIdentical(left.bicubic_mixed_derivative,
                         right.bicubic_mixed_derivative) &&
           byteIdentical(left.bicubic_polynomial_coefficients,
                         right.bicubic_polynomial_coefficients) &&
           byteIdentical(left.cubic_values, right.cubic_values) &&
           byteIdentical(left.cubic_node_derivatives,
                         right.cubic_node_derivatives);
  }

  double evaluateEigenReference(
      ProposalNativeDeviceView const& view,
      NativeBicubicDescriptor const& spline,
      double energy_MeV, double transformed_loss) {
    auto const x0 = native_detail::axisTransform(
        spline.energy_axis, energy_MeV);
    auto const x1 = native_detail::axisTransform(
        spline.loss_axis, transformed_loss);
    auto const n0 = native_detail::intervalNode(x0, spline.rows);
    auto const n1 = native_detail::intervalNode(x1, spline.columns);
    Eigen::Matrix4d temp;
    for (std::uint32_t row = 0; row < 4; ++row) {
      for (std::uint32_t column = 0; column < 4; ++column) {
        auto const source_row = n0 + (row & 1u);
        auto const source_column = n1 + (column & 1u);
        double const* coefficients = view.bicubic_values;
        if (row >= 2 && column < 2)
          coefficients = view.bicubic_derivative_energy;
        else if (row < 2 && column >= 2)
          coefficients = view.bicubic_derivative_loss;
        else if (row >= 2 && column >= 2)
          coefficients = view.bicubic_mixed_derivative;
        temp(row, column) = native_detail::bicubicElement(
            view, spline, coefficients, source_row, source_column);
      }
    }
    Eigen::Matrix4d m;
    m << 1., 0., -3., 2., 0., 0., 3., -2., 0., 1., -2., 1.,
        0., 0., -1., 1.;
    Eigen::Vector4d v0;
    Eigen::Vector4d v1;
    auto const f0 = x0 - n0;
    auto const f1 = x1 - n1;
    v0 << 1., f0, f0 * f0, f0 * f0 * f0;
    v1 << 1., f1, f1 * f1, f1 * f1 * f1;
    return v0.dot((m.transpose() * (temp * m)) * v1);
  }

  double evaluatePrecomputedCoefficientReference(
      ProposalNativeDeviceView const& view,
      NativeBicubicDescriptor const& spline,
      double energy_MeV, double transformed_loss,
      bool inner_pairwise = true, bool outer_pairwise = true) {
    auto const x0 = native_detail::axisTransform(
        spline.energy_axis, energy_MeV);
    auto const x1 = native_detail::axisTransform(
        spline.loss_axis, transformed_loss);
    auto const n0 = native_detail::intervalNode(x0, spline.rows);
    auto const n1 = native_detail::intervalNode(x1, spline.columns);
    Eigen::Matrix4d temp;
    for (std::uint32_t row = 0; row < 4; ++row) {
      for (std::uint32_t column = 0; column < 4; ++column) {
        auto const source_row = n0 + (row & 1u);
        auto const source_column = n1 + (column & 1u);
        double const* source = view.bicubic_values;
        if (row >= 2 && column < 2)
          source = view.bicubic_derivative_energy;
        else if (row < 2 && column >= 2)
          source = view.bicubic_derivative_loss;
        else if (row >= 2 && column >= 2)
          source = view.bicubic_mixed_derivative;
        temp(row, column) = native_detail::bicubicElement(
            view, spline, source, source_row, source_column);
      }
    }
    Eigen::Matrix4d m;
    m << 1., 0., -3., 2., 0., 0., 3., -2., 0., 1., -2., 1.,
        0., 0., -1., 1.;
    auto const coefficients = m.transpose() * (temp * m);
    auto const f0 = x0 - n0;
    auto const f1 = x1 - n1;
    double const v0[4]{1., f0, f0 * f0, f0 * f0 * f0};
    double const v1[4]{1., f1, f1 * f1, f1 * f1 * f1};
    auto dot4 = [](double a0, double a1, double a2, double a3,
                   double b0, double b1, double b2, double b3,
                   bool pairwise) {
      auto const p0 = a0 * b0;
      auto const p1 = a1 * b1;
      auto const p2 = a2 * b2;
      auto const p3 = a3 * b3;
      return pairwise ? (p0 + p1) + (p2 + p3)
                      : ((p0 + p1) + p2) + p3;
    };
    double projected[4]{};
    for (int row = 0; row < 4; ++row)
      projected[row] = dot4(
          coefficients(row, 0), coefficients(row, 1),
          coefficients(row, 2), coefficients(row, 3),
          v1[0], v1[1], v1[2], v1[3], inner_pairwise);
    return dot4(v0[0], v0[1], v0[2], v0[3], projected[0],
                projected[1], projected[2], projected[3],
                outer_pairwise);
  }

  double evaluateStagedReference(
      ProposalNativeDeviceView const& view,
      NativeBicubicDescriptor const& spline,
      double energy_MeV, double transformed_loss, unsigned int mask) {
    auto const x0 = native_detail::axisTransform(
        spline.energy_axis, energy_MeV);
    auto const x1 = native_detail::axisTransform(
        spline.loss_axis, transformed_loss);
    auto const n0 = native_detail::intervalNode(x0, spline.rows);
    auto const n1 = native_detail::intervalNode(x1, spline.columns);
    double temp[4][4]{};
    for (std::uint32_t row = 0; row < 4; ++row) {
      for (std::uint32_t column = 0; column < 4; ++column) {
        auto const source_row = n0 + (row & 1u);
        auto const source_column = n1 + (column & 1u);
        double const* coefficients = view.bicubic_values;
        if (row >= 2 && column < 2)
          coefficients = view.bicubic_derivative_energy;
        else if (row < 2 && column >= 2)
          coefficients = view.bicubic_derivative_loss;
        else if (row >= 2 && column >= 2)
          coefficients = view.bicubic_mixed_derivative;
        temp[row][column] = native_detail::bicubicElement(
            view, spline, coefficients, source_row, source_column);
      }
    }
    constexpr double m[4][4]{
        {1., 0., -3., 2.}, {0., 0., 3., -2.},
        {0., 1., -2., 1.}, {0., 0., -1., 1.}};
    auto dot4 = [](double a0, double a1, double a2, double a3,
                   double b0, double b1, double b2, double b3,
                   bool pairwise) {
      auto const p0 = a0 * b0;
      auto const p1 = a1 * b1;
      auto const p2 = a2 * b2;
      auto const p3 = a3 * b3;
      return pairwise ? (p0 + p1) + (p2 + p3)
                      : ((p0 + p1) + p2) + p3;
    };
    double first[4][4]{};
    double second[4][4]{};
    for (int row = 0; row < 4; ++row)
      for (int column = 0; column < 4; ++column)
        first[row][column] = dot4(
            temp[row][0], temp[row][1], temp[row][2], temp[row][3],
            m[0][column], m[1][column], m[2][column], m[3][column],
            (mask & 1u) != 0u);
    for (int row = 0; row < 4; ++row)
      for (int column = 0; column < 4; ++column)
        second[row][column] = dot4(
            m[0][row], m[1][row], m[2][row], m[3][row],
            first[0][column], first[1][column],
            first[2][column], first[3][column], (mask & 2u) != 0u);
    auto const f0 = x0 - n0;
    auto const f1 = x1 - n1;
    double const v0[4]{1., f0, f0 * f0, f0 * f0 * f0};
    double const v1[4]{1., f1, f1 * f1, f1 * f1 * f1};
    double projected[4]{};
    for (int row = 0; row < 4; ++row)
      projected[row] = dot4(
          second[row][0], second[row][1], second[row][2],
          second[row][3], v1[0], v1[1], v1[2], v1[3],
          (mask & 4u) != 0u);
    return dot4(v0[0], v0[1], v0[2], v0[3], projected[0],
                projected[1], projected[2], projected[3],
                (mask & 8u) != 0u);
  }

  double evaluateHermitePairwiseReference(
      ProposalNativeDeviceView const& view,
      NativeBicubicDescriptor const& spline,
      double energy_MeV, double transformed_loss) {
    auto const x0 = native_detail::axisTransform(
        spline.energy_axis, energy_MeV);
    auto const x1 = native_detail::axisTransform(
        spline.loss_axis, transformed_loss);
    auto const n0 = native_detail::intervalNode(x0, spline.rows);
    auto const n1 = native_detail::intervalNode(x1, spline.columns);
    double h0[4]{};
    double h1[4]{};
    native_detail::hermiteBasis(x0 - n0, h0);
    native_detail::hermiteBasis(x1 - n1, h1);
    auto pairwiseDot = [](double const (&left)[4],
                          double const (&right)[4]) {
      return (left[0] * right[0] + left[1] * right[1]) +
             (left[2] * right[2] + left[3] * right[3]);
    };
    double projected[4]{};
    for (std::uint32_t row = 0; row < 4; ++row) {
      double coefficients_at_row[4]{};
      for (std::uint32_t column = 0; column < 4; ++column) {
        auto const source_row = n0 + (row & 1u);
        auto const source_column = n1 + (column & 1u);
        double const* coefficients = view.bicubic_values;
        if (row >= 2 && column < 2)
          coefficients = view.bicubic_derivative_energy;
        else if (row < 2 && column >= 2)
          coefficients = view.bicubic_derivative_loss;
        else if (row >= 2 && column >= 2)
          coefficients = view.bicubic_mixed_derivative;
        coefficients_at_row[column] = native_detail::bicubicElement(
            view, spline, coefficients, source_row, source_column);
      }
      projected[row] = pairwiseDot(coefficients_at_row, h1);
    }
    return pairwiseDot(h0, projected);
  }

  double evaluateLongDoubleReference(
      ProposalNativeDeviceView const& view,
      NativeBicubicDescriptor const& spline,
      double energy_MeV, double transformed_loss) {
    auto const x0_double = native_detail::axisTransform(
        spline.energy_axis, energy_MeV);
    auto const x1_double = native_detail::axisTransform(
        spline.loss_axis, transformed_loss);
    auto const n0 = native_detail::intervalNode(x0_double, spline.rows);
    auto const n1 = native_detail::intervalNode(x1_double, spline.columns);
    auto const x0 = static_cast<long double>(x0_double - n0);
    auto const x1 = static_cast<long double>(x1_double - n1);
    long double h0[4]{1.L - 3.L * x0 * x0 + 2.L * x0 * x0 * x0,
                      3.L * x0 * x0 - 2.L * x0 * x0 * x0,
                      x0 - 2.L * x0 * x0 + x0 * x0 * x0,
                      -x0 * x0 + x0 * x0 * x0};
    long double h1[4]{1.L - 3.L * x1 * x1 + 2.L * x1 * x1 * x1,
                      3.L * x1 * x1 - 2.L * x1 * x1 * x1,
                      x1 - 2.L * x1 * x1 + x1 * x1 * x1,
                      -x1 * x1 + x1 * x1 * x1};
    long double result = 0.L;
    for (std::uint32_t row = 0; row < 4; ++row) {
      for (std::uint32_t column = 0; column < 4; ++column) {
        auto const source_row = n0 + (row & 1u);
        auto const source_column = n1 + (column & 1u);
        double const* coefficients = view.bicubic_values;
        if (row >= 2 && column < 2)
          coefficients = view.bicubic_derivative_energy;
        else if (row < 2 && column >= 2)
          coefficients = view.bicubic_derivative_loss;
        else if (row >= 2 && column >= 2)
          coefficients = view.bicubic_mixed_derivative;
        result += h0[row] * static_cast<long double>(
                    native_detail::bicubicElement(
                        view, spline, coefficients, source_row,
                        source_column)) *
                  h1[column];
      }
    }
    return static_cast<double>(result);
  }

  void requireCloseOrUlps(double actual, double expected, double relative,
                          std::uint64_t ulps,
                          std::string const& message) {
    ++checks;
    auto const scale = std::max(std::abs(expected), 1.e-300);
    auto const relative_difference = std::abs(actual - expected) / scale;
    auto const ulp_difference = positiveUlpDistance(actual, expected);
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        (relative_difference > relative && ulp_difference > ulps)) {
      std::ostringstream details;
      details << std::setprecision(17) << message << ": actual=" << actual
              << ", expected=" << expected
              << ", relative_difference=" << relative_difference
              << ", ulp_difference=" << ulp_difference;
      throw std::runtime_error(details.str());
    }
  }

  void requireRateCloseOrUlps(double actual, double expected,
                              double tolerance, std::uint64_t ulps,
                              std::string const& message) {
    ++checks;
    auto const difference = std::abs(actual - expected);
    auto const relative = difference /
                          std::max(std::abs(expected), 1.e-300);
    auto const ulp_difference = positiveUlpDistance(actual, expected);
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        (difference > tolerance && relative > tolerance &&
         ulp_difference > ulps)) {
      std::ostringstream details;
      details << std::setprecision(17) << message << ": actual=" << actual
              << ", expected=" << expected
              << ", absolute_difference=" << difference
              << ", relative_difference=" << relative
              << ", ulp_difference=" << ulp_difference;
      throw std::runtime_error(details.str());
    }
  }

  struct CalculatorOwner {
    PROPOSAL::Air medium;
    Code code{Code::Unknown};
    PROPOSAL::crosssection_list_t cross_sections;
    std::unique_ptr<PROPOSAL::Interaction> interaction;
    std::unique_ptr<PROPOSAL::Displacement> displacement;
    std::unique_ptr<PROPOSAL::crosssection::PhotoPairLPM> photon_pair_lpm;
    std::unique_ptr<PROPOSAL::crosssection::BremsLPM> brems_lpm;
    double mass_MeV{};
    HEPEnergyType cut{};

    CalculatorOwner(Code requested, HEPEnergyType requested_cut)
        : code(requested), mass_MeV(proposal::particle.at(requested).mass),
          cut(requested_cut) {
      cross_sections = proposal::make_cross_sections(
          code, medium, cut, true);
      interaction = PROPOSAL::make_interaction(
          cross_sections, true, true);
      displacement = PROPOSAL::make_displacement(
          cross_sections, true);
      if (code == Code::Photon)
        photon_pair_lpm =
            std::make_unique<PROPOSAL::crosssection::PhotoPairLPM>(
                proposal::particle.at(code), medium,
                PROPOSAL::crosssection::PhotoPairKochMotz());
      if (code == Code::Electron || code == Code::Positron)
        brems_lpm = std::make_unique<PROPOSAL::crosssection::BremsLPM>(
            proposal::particle.at(code), medium,
            PROPOSAL::crosssection::BremsElectronScreening());
    }

    proposal::NativeInteractionCalculatorView interactionView() const {
      return {code, medium.GetHash(), &medium, interaction.get(),
              photon_pair_lpm.get(), brems_lpm.get(), cut,
              mass_MeV};
    }
    proposal::NativeContinuousCalculatorView continuousView() const {
      return {code, medium.GetHash(), &medium, displacement.get(), cut,
              mass_MeV};
    }
  };
}
int main(int argc, char** argv) {
  try {
    std::size_t randomized_samples_per_column = 0;
    std::size_t randomized_chunk_size = 65536;
    if (argc >= 2)
      randomized_samples_per_column = std::stoull(argv[1]);
    if (argc >= 3)
      randomized_chunk_size = std::stoull(argv[2]);
    if (argc > 3 || randomized_chunk_size == 0)
      throw std::invalid_argument(
          "usage: testGpuProposalNativeTable [SAMPLES_PER_COLUMN] [CHUNK_SIZE]");
    PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::critical);
    auto const* cache_roundtrip_environment =
        std::getenv("C8_PROPOSAL_NATIVE_CACHE_ROUNDTRIP_DIR");
    auto const cache_roundtrip =
        cache_roundtrip_environment != nullptr &&
        *cache_roundtrip_environment != '\0';
    std::filesystem::path cache_roundtrip_directory;
    if (cache_roundtrip) {
      cache_roundtrip_directory = cache_roundtrip_environment;
      if (cache_roundtrip_directory.empty() ||
          cache_roundtrip_directory == cache_roundtrip_directory.root_path())
        throw std::invalid_argument(
            "refusing an unsafe PROPOSAL cache round-trip directory");
      std::filesystem::remove_all(cache_roundtrip_directory);
      std::filesystem::create_directories(cache_roundtrip_directory);
      PROPOSAL::InterpolationSettings::TABLES_PATH =
          cache_roundtrip_directory.string();
    }
    int devices = 0;
    auto const status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) return 77;

    CalculatorOwner photon{Code::Photon, 0.4_MeV};
    CalculatorOwner electron{Code::Electron, 0.4_MeV};
    CalculatorOwner positron{Code::Positron, 0.4_MeV};
    CalculatorOwner muon_minus{Code::MuMinus, 0.4_MeV};
    CalculatorOwner muon_plus{Code::MuPlus, 0.4_MeV};
    std::vector<proposal::NativeInteractionCalculatorView> interactions{
        photon.interactionView(), electron.interactionView(),
        positron.interactionView(), muon_minus.interactionView(),
        muon_plus.interactionView()};
    std::vector<proposal::NativeContinuousCalculatorView> continuous{
        electron.continuousView(), positron.continuousView(),
        muon_minus.continuousView(), muon_plus.continuousView()};
    std::array<CalculatorOwner const*, 5> owners{
        &photon, &electron, &positron, &muon_minus, &muon_plus};
    std::size_t processes_without_dedx_table = 0;
    for (auto const* owner : owners) {
      for (auto const& cross_section : owner->cross_sections) {
        auto const sources = cross_section->ExportDedxInterpolation();
        if (sources.empty()) {
          // Empty is PROPOSAL's explicit representation of a process without
          // an interpolated continuous-loss contribution.  It must remain
          // legal; only a present but malformed spline is an export error.
          ++processes_without_dedx_table;
          continue;
        }
        for (std::size_t source_index = 0; source_index < sources.size();
             ++source_index) {
          auto const& source = sources[source_index];
          require(source.component_index == source_index,
                  "PROPOSAL dE/dX export lost its stable source ordinal");
          require(source.axis.nodes >= 2,
                  "non-empty PROPOSAL dE/dX export has a degenerate axis");
          require(source.spline.values.size() >= 2,
                  "non-empty PROPOSAL dE/dX export has no spline values");
          require(source.spline.values.size() ==
                      source.spline.node_derivatives.size(),
                  "non-empty PROPOSAL dE/dX export has mismatched coefficients");
        }
      }
    }
    require(processes_without_dedx_table > 0,
            "native fixture does not exercise legal absent dE/dX tables");
    auto const ownerForPdg = [&](std::int32_t pdg) {
      auto const found = std::find_if(
          owners.begin(), owners.end(), [&](auto const* owner) {
            return static_cast<std::int32_t>(get_PDG(owner->code)) == pdg;
          });
      if (found == owners.end())
        throw std::runtime_error("no calculator owner for exported PDG");
      return *found;
    };

    auto const first = exportProposalNativeTables(interactions, continuous);
    auto second = exportProposalNativeTables(interactions, continuous);
    auto interactions_with_unsupported = interactions;
    auto ignored_tau_interaction = muon_minus.interactionView();
    ignored_tau_interaction.projectile = Code::TauMinus;
    ignored_tau_interaction.stochastic_energy_cut = 300_MeV;
    interactions_with_unsupported.push_back(ignored_tau_interaction);
    auto continuous_with_unsupported = continuous;
    auto ignored_tau_continuous = muon_minus.continuousView();
    ignored_tau_continuous.projectile = Code::TauMinus;
    ignored_tau_continuous.stochastic_energy_cut = 300_MeV;
    continuous_with_unsupported.push_back(ignored_tau_continuous);
    auto const filtered = exportProposalNativeTables(
        interactions_with_unsupported, continuous_with_unsupported);
    require(filtered.content_hash == first.content_hash &&
                exportedArraysIdentical(filtered, first),
            "unsupported tau calculators changed the native photon/e/muon export");
    if (cache_roundtrip) {
      CalculatorOwner reloaded_photon{Code::Photon, 0.4_MeV};
      CalculatorOwner reloaded_electron{Code::Electron, 0.4_MeV};
      CalculatorOwner reloaded_positron{Code::Positron, 0.4_MeV};
      CalculatorOwner reloaded_muon_minus{Code::MuMinus, 0.4_MeV};
      CalculatorOwner reloaded_muon_plus{Code::MuPlus, 0.4_MeV};
      std::vector<proposal::NativeInteractionCalculatorView>
          reloaded_interactions{
              reloaded_photon.interactionView(),
              reloaded_electron.interactionView(),
              reloaded_positron.interactionView(),
              reloaded_muon_minus.interactionView(),
              reloaded_muon_plus.interactionView()};
      std::vector<proposal::NativeContinuousCalculatorView>
          reloaded_continuous{
              reloaded_electron.continuousView(),
              reloaded_positron.continuousView(),
              reloaded_muon_minus.continuousView(),
              reloaded_muon_plus.continuousView()};
      second = exportProposalNativeTables(
          reloaded_interactions, reloaded_continuous);
      require(!first.proposal_cache_all_hit,
              "fresh PROPOSAL cache unexpectedly reported all hits");
      require(second.proposal_cache_all_hit,
              "reloaded PROPOSAL calculators did not report all cache hits");
    }
    require(first.content_hash == second.content_hash,
            "repeated native export changed its canonical hash");
    require(exportedArraysIdentical(first, second),
            "repeated native export changed coefficient arrays");
    require(first.proposal_version == "7.6.2" &&
                first.cubic_interpolation_version == "0.1.5",
            "native export dependency versions are not locked");
    require(first.format_version == 6u &&
                first.format_version == ProposalNativeTableFormatVersion,
            "native selection/total-rate/dE/dX-order semantics did not bump "
            "the table format");
    require(first.proposal_cache_table_count > 0 &&
                first.proposal_cache_hit_count <=
                    first.proposal_cache_table_count,
            "native export did not report PROPOSAL cache state");

    // The exporter must classify only the exact PROPOSAL 7.6.2 tuples used by
    // the production builders.  A plausible process id with a different
    // parametrization must fail closed instead of becoming a generic CPU
    // selected-loss column.
    require(classifyProposalNativeKinematicModel(
                gpu::em::PhotonPairProcessId, "kochmotz", 22) ==
                NativeKinematicModel::PhotonPairFullLoss,
            "native parametrization whitelist rejected photon pair production");
    require(classifyProposalNativeKinematicModel(
                gpu::em::BremsProcessId, "KelnerKokoulinPetrukhin", 13) ==
                NativeKinematicModel::BremsKelnerKokoulinPetrukhin,
            "native parametrization whitelist rejected muon KKP brems");
    require(classifyProposalNativeKinematicModel(
                gpu::em::ElectronPairProcessId,
                "KelnerKokoulinPetrukhin", -13) ==
                NativeKinematicModel::ElectronPair,
            "native parametrization whitelist rejected muon KKP epair");
    require(classifyProposalNativeKinematicModel(
                gpu::em::PhotonuclearProcessId,
                "AbramowiczLevinLevyMaor97", 11) ==
                NativeKinematicModel::PhotonuclearAllm97,
            "native parametrization whitelist rejected electron ALLM97");
    require(classifyProposalNativeKinematicModel(
                gpu::em::PhotonuclearProcessId,
                "AbramowiczLevinLevyMaor97", -13) ==
                NativeKinematicModel::PhotonuclearAllm97,
            "native parametrization whitelist rejected muon ALLM97");
    require(classifyProposalNativeKinematicModel(
                gpu::em::PhotonMuonPairProcessId,
                "BurkhardtKelnerKokoulin", 22) ==
                NativeKinematicModel::OnlyStochastic,
            "native parametrization whitelist rejected photon BKK mu-pair");
    bool unknown_parametrization_rejected = false;
    try {
      static_cast<void>(classifyProposalNativeKinematicModel(
          gpu::em::PhotonPairProcessId, "Tsai", 22));
    } catch (std::invalid_argument const&) {
      unknown_parametrization_rejected = true;
    }
    require(unknown_parametrization_rejected,
            "native exporter accepted an unknown photon-pair parametrization");
    bool wrong_pid_rejected = false;
    try {
      static_cast<void>(classifyProposalNativeKinematicModel(
          gpu::em::ComptonProcessId, "KleinNishina", 11));
    } catch (std::invalid_argument const&) {
      wrong_pid_rejected = true;
    }
    require(wrong_pid_rejected,
            "native exporter accepted a parametrization for the wrong PID");

    // Structural gates protect the assumptions made by device-side binary
    // lookup and spline evaluation.  Exercise each gate with a deliberately
    // damaged copy; the production export above must remain untouched.
    validateProposalNativeTable(first);
    require(first.dndx_columns.size() >= 2,
            "native export has too few dN/dX columns for ordering tests");
    requireStructuralRejection(
        first,
        [](auto& table) {
          std::swap(table.dndx_columns[0], table.dndx_columns[1]);
        },
        "native validation accepted non-canonical dN/dX ordering");
    requireStructuralRejection(
        first,
        [](auto& table) {
          table.dndx_columns.front().medium_hash ^=
              0x9e3779b97f4a7c15ull;
        },
        "native validation accepted multiple media for one PID");
    requireStructuralRejection(
        first,
        [](auto& table) {
          table.dndx_columns.push_back(table.dndx_columns.front());
          std::sort(table.dndx_columns.begin(), table.dndx_columns.end(),
                    [](auto const& left, auto const& right) {
                      return std::tie(left.pdg_id, left.process_id,
                                      left.component_hash) <
                             std::tie(right.pdg_id, right.process_id,
                                      right.component_hash);
                    });
        },
        "native validation accepted a duplicate dN/dX identity");
    requireStructuralRejection(
        first,
        [](auto& table) { table.selection_column_indices.pop_back(); },
        "native validation accepted a truncated selection permutation");
    requireStructuralRejection(
        first,
        [](auto& table) {
          table.selection_column_indices.back() =
              table.selection_column_indices.front();
        },
        "native validation accepted a duplicate selection index");
    requireStructuralRejection(
        first,
        [](auto& table) {
          auto const pdg = table.dndx_columns.front().pdg_id;
          auto duplicate = std::find_if(
              std::next(table.dndx_columns.begin()),
              table.dndx_columns.end(),
              [pdg](auto const& column) { return column.pdg_id == pdg; });
          if (duplicate != table.dndx_columns.end())
            duplicate->selection_ordinal =
                table.dndx_columns.front().selection_ordinal;
        },
        "native validation accepted an invalid selection ordinal");
    requireStructuralRejection(
        first,
        [](auto& table) { table.total_rate_columns.pop_back(); },
        "native validation accepted a missing mean-free-path rate spline");

    auto const bicubic = std::find_if(
        first.dndx_columns.begin(), first.dndx_columns.end(),
        [](auto const& column) {
          return column.rate_model == NativeRateModel::BicubicSpline;
        });
    require(bicubic != first.dndx_columns.end(),
            "native export has no bicubic column for structure tests");
    auto const bicubic_index = static_cast<std::size_t>(
        std::distance(first.dndx_columns.begin(), bicubic));
    requireStructuralRejection(
        first,
        [bicubic_index](auto& table) {
          table.dndx_columns[bicubic_index].spline.energy_axis.step =
              std::numeric_limits<double>::quiet_NaN();
        },
        "native validation accepted a non-finite axis step");
    requireStructuralRejection(
        first,
        [bicubic_index](auto& table) {
          ++table.dndx_columns[bicubic_index].spline.energy_axis.nodes;
        },
        "native validation accepted an axis/node dimension mismatch");
    require(!first.bicubic_derivative_energy.empty(),
            "native export has no bicubic derivative coefficients");
    requireStructuralRejection(
        first,
        [](auto& table) { table.bicubic_derivative_energy.pop_back(); },
        "native validation accepted unequal bicubic array sizes");

    require(!first.dedx_columns.empty(),
            "native export has no dE/dX column for cubic structure tests");
    requireStructuralRejection(
        first,
        [](auto& table) { table.dedx_columns.front().spline.axis.step = 0.; },
        "native validation accepted an invalid cubic axis");
    requireStructuralRejection(
        first,
        [](auto& table) {
          table.dndx_columns.front().medium_mean_excitation_energy_MeV = 0.;
        },
        "native validation accepted an invalid mean excitation energy");

    require(!first.dedx_columns.empty() && !first.utility_columns.empty() &&
                !first.cubic_values.empty(),
            "native export lacks continuous columns for negative structure tests");
    requireStructuralRejection(
        first,
        [](auto& table) {
          auto duplicate = table.dedx_columns.back();
          duplicate.table_hash ^= 0x9e3779b97f4a7c15ull;
          table.dedx_columns.push_back(duplicate);
        },
        "native validation used provenance table hash as the dE/dX component identity");
    requireStructuralRejection(
        first,
        [](auto& table) {
          std::swap(table.dedx_columns.front(), table.dedx_columns.back());
        },
        "native validation accepted non-canonical dE/dX component ordering");
    requireStructuralRejection(
        first,
        [](auto& table) { table.dedx_accumulation_indices.pop_back(); },
        "native validation accepted a truncated dE/dX accumulation order");
    requireStructuralRejection(
        first,
        [](auto& table) {
          table.dedx_accumulation_indices.back() =
              table.dedx_accumulation_indices.front();
        },
        "native validation accepted a duplicate dE/dX accumulation index");
    {
      // Equal table hashes are legal: two medium components can have
      // identical interpolation definitions.  Their stable source ordinals,
      // rather than table_hash, keep both physical weighted contributions.
      auto shared_table_hash = first;
      auto const source = shared_table_hash.dedx_columns.back();
      auto distinct_component = source;
      for (auto const& column : shared_table_hash.dedx_columns) {
        if (column.pdg_id == source.pdg_id)
          distinct_component.accumulation_ordinal =
              std::max(distinct_component.accumulation_ordinal,
                       column.accumulation_ordinal + 1u);
        if (column.pdg_id == source.pdg_id &&
            column.process_id == source.process_id &&
            column.cross_section_hash == source.cross_section_hash)
          distinct_component.component_index =
              std::max(distinct_component.component_index,
                       column.component_index + 1u);
      }
      auto const old_offset =
          static_cast<std::size_t>(source.spline.coefficient_offset);
      auto const count =
          static_cast<std::size_t>(source.spline.coefficient_count);
      std::vector<double> const copied_values(
          shared_table_hash.cubic_values.begin() + old_offset,
          shared_table_hash.cubic_values.begin() + old_offset + count);
      std::vector<double> const copied_derivatives(
          shared_table_hash.cubic_node_derivatives.begin() + old_offset,
          shared_table_hash.cubic_node_derivatives.begin() + old_offset +
              count);
      distinct_component.spline.coefficient_offset =
          shared_table_hash.cubic_values.size();
      shared_table_hash.cubic_values.insert(
          shared_table_hash.cubic_values.end(),
          copied_values.begin(), copied_values.end());
      shared_table_hash.cubic_node_derivatives.insert(
          shared_table_hash.cubic_node_derivatives.end(),
          copied_derivatives.begin(), copied_derivatives.end());
      shared_table_hash.dedx_columns.push_back(distinct_component);
      std::sort(
          shared_table_hash.dedx_columns.begin(),
          shared_table_hash.dedx_columns.end(),
          [](auto const& left, auto const& right) {
            return std::tie(left.pdg_id, left.medium_hash, left.process_id,
                            left.cross_section_hash, left.component_index,
                            left.table_hash) <
                   std::tie(right.pdg_id, right.medium_hash,
                            right.process_id, right.cross_section_hash,
                            right.component_index, right.table_hash);
          });
      shared_table_hash.dedx_accumulation_indices.resize(
          shared_table_hash.dedx_columns.size());
      std::iota(shared_table_hash.dedx_accumulation_indices.begin(),
                shared_table_hash.dedx_accumulation_indices.end(), 0u);
      std::sort(
          shared_table_hash.dedx_accumulation_indices.begin(),
          shared_table_hash.dedx_accumulation_indices.end(),
          [&](std::uint32_t left_index, std::uint32_t right_index) {
            auto const& left =
                shared_table_hash.dedx_columns[left_index];
            auto const& right =
                shared_table_hash.dedx_columns[right_index];
            return std::tie(left.pdg_id, left.accumulation_ordinal) <
                   std::tie(right.pdg_id, right.accumulation_ordinal);
          });
      validateProposalNativeTable(shared_table_hash);
      require(true,
              "native validation rejected distinct components sharing a table hash");
    }
    requireStructuralRejection(
        first,
        [](auto& table) {
          table.utility_columns.push_back(table.utility_columns.front());
        },
        "native validation accepted a duplicate utility PID column");
    requireStructuralRejection(
        first,
        [](auto& table) {
          table.dedx_columns.front().pdg_id = 22;
        },
        "native validation accepted a photon dE/dX column");
    requireStructuralRejection(
        first,
        [](auto& table) {
          table.dedx_columns.front().component_weight =
              std::numeric_limits<double>::infinity();
        },
        "native validation accepted non-finite dE/dX metadata");
    requireStructuralRejection(
        first,
        [](auto& table) {
          auto const offset = static_cast<std::size_t>(
              table.dedx_columns.front().spline.coefficient_offset);
          table.cubic_values[offset] =
              std::numeric_limits<double>::quiet_NaN();
        },
        "native validation accepted a non-finite cubic coefficient");
    requireStructuralRejection(
        first,
        [](auto& table) {
          table.utility_columns.front().reverse = 2u;
        },
        "native validation accepted an invalid utility direction flag");
    requireStructuralRejection(
        first,
        [](auto& table) {
          table.utility_columns.erase(table.utility_columns.begin());
        },
        "native validation accepted a missing charged-particle utility column");
    requireStructuralRejection(
        first,
        [](auto& table) {
          table.cubic_values.push_back(0.);
          table.cubic_node_derivatives.push_back(0.);
        },
        "native validation accepted unreferenced cubic coefficients");

    auto const muon_plus_column = std::find_if(
        first.dndx_columns.begin(), first.dndx_columns.end(),
        [](auto const& column) { return column.pdg_id == -13; });
    auto const muon_minus_column = std::find_if(
        first.dndx_columns.begin(), first.dndx_columns.end(),
        [](auto const& column) { return column.pdg_id == 13; });
    require(muon_plus_column != first.dndx_columns.end() &&
                muon_minus_column != first.dndx_columns.end(),
            "native export lacks a muon charge for symmetry tests");
    auto const muon_plus_index = static_cast<std::size_t>(
        std::distance(first.dndx_columns.begin(), muon_plus_column));
    requireStructuralRejection(
        first,
        [muon_plus_index](auto& table) {
          table.dndx_columns.erase(table.dndx_columns.begin() +
                                   muon_plus_index);
        },
        "native validation accepted asymmetric mu-/mu+ column sets");

    // Keep the core-export and structural contract independently testable
    // while auxiliary-cache algorithms are being developed.  The default
    // invocation still executes the complete aux and CUDA query suite.
    auto const* table_only =
        std::getenv("C8_PROPOSAL_NATIVE_TABLE_ONLY");
    if (table_only != nullptr && std::string_view(table_only) == "1") {
      std::cout << "PROPOSAL native core-table checks passed: " << checks
                << ", format=" << first.format_version
                << ", dndx_columns=" << first.dndx_columns.size()
                << ", dedx_columns=" << first.dedx_columns.size()
                << ", utility_columns=" << first.utility_columns.size()
                << ", hash=" << hexadecimalBytes(first.content_hash) << '\n';
      return 0;
    }

    auto const aux_directory =
        std::filesystem::temp_directory_path() /
        "c8-proposal-native-aux-test";
    std::filesystem::remove_all(aux_directory);
    auto const first_aux =
        loadOrCreateProposalNativeAux(interactions, aux_directory);
    auto const second_aux =
        loadOrCreateProposalNativeAux(interactions, aux_directory);
    auto const filtered_aux = loadOrCreateProposalNativeAux(
        interactions_with_unsupported, aux_directory);
    require(!first_aux.cache_hit && second_aux.cache_hit &&
                filtered_aux.cache_hit &&
                first_aux.content_hash == second_aux.content_hash &&
                filtered_aux.content_hash == first_aux.content_hash &&
                filtered_aux.key_hash == first_aux.key_hash,
            "c8emaux generation/cache reload is not deterministic");
    require(first_aux.format_version == ProposalNativeAuxFormatVersion &&
                std::string_view(ProposalNativeAuxAlgorithmVersion).find(
                    "epair-device-rejection-v1") != std::string_view::npos,
            "c8emaux does not declare the beta4 Epair device-rejection sampler");
    require(std::filesystem::file_size(first_aux.cache_file) < 2u * 1024u * 1024u,
            "c8emaux unexpectedly contains a large Epair inverse-CDF payload");
    {
      std::fstream damaged(
          first_aux.cache_file,
          std::ios::binary | std::ios::in | std::ios::out);
      require(static_cast<bool>(damaged),
              "cannot open c8emaux cache for corruption test");
      damaged.seekg(-1, std::ios::end);
      char byte{};
      damaged.read(&byte, 1);
      require(static_cast<bool>(damaged),
              "cannot read c8emaux corruption-test byte");
      byte = static_cast<char>(
          static_cast<unsigned char>(byte) ^ 0x5au);
      damaged.seekp(-1, std::ios::end);
      damaged.write(&byte, 1);
      damaged.flush();
      require(static_cast<bool>(damaged),
              "cannot write c8emaux corruption-test byte");
    }
    auto const repaired_aux =
        loadOrCreateProposalNativeAux(interactions, aux_directory);
    auto const repaired_aux_reload =
        loadOrCreateProposalNativeAux(interactions, aux_directory);
    require(!repaired_aux.cache_hit && repaired_aux_reload.cache_hit &&
                repaired_aux.content_hash == first_aux.content_hash &&
                repaired_aux_reload.content_hash == first_aux.content_hash,
            "c8emaux corruption was not rebuilt and revalidated deterministically");

    auto const host = makeProposalNativeHostView(first);
    {
      // The PhotoPairKochMotz cumulative spline supplies the stochastic
      // interaction rate, but PROPOSAL's public stochastic-loss contract for
      // this process is a full photon loss (v=1).  Re-inverting the internal
      // pair-energy integral changes the transported primary energy and was
      // the first strict-oracle divergence in decision-tape replay.
      std::size_t photon_pair_columns = 0;
      for (auto const& column : first.dndx_columns) {
        if (column.pdg_id != 22 ||
            column.process_id != gpu::em::PhotonPairProcessId)
          continue;
        ++photon_pair_columns;
        require(column.kinematic_model ==
                    NativeKinematicModel::PhotonPairFullLoss,
                "photon-pair column does not use full-loss semantics");
        auto const energy = std::sqrt(
            column.spline.energy_axis.low *
            std::min(column.spline.energy_axis.high, 1.e8));
        auto const rates = photon.interaction->Rates(energy);
        auto const oracle = std::find_if(
            rates.begin(), rates.end(), [&](auto const& rate) {
              return static_cast<std::int32_t>(
                         rate.crosssection->GetInteractionType()) ==
                         column.process_id &&
                     static_cast<std::uint64_t>(rate.comp_hash) ==
                         column.component_hash;
            });
        require(oracle != rates.end() && oracle->rate > 0.,
                "cannot locate live PROPOSAL photon-pair rate column");
        auto const native_rate = queryProposalNativeRate(
            host, column.pdg_id, column.process_id, column.component_hash,
            energy);
        require(native_rate.status == NativeQueryStatus::Success,
                "native photon-pair rate query failed");
        requireRateCloseOrUlps(
            native_rate.value, oracle->rate, 2.e-12, 16,
            "native photon-pair rate differs from PROPOSAL");
        for (double const quantile :
             {0., 1.e-6, 0.1, 0.5, 0.9, 1. - 1.e-6, 1.}) {
          auto const expected =
              oracle->crosssection->CalculateStochasticLoss(
                  oracle->comp_hash, energy, quantile * oracle->rate);
          require(expected == 1.,
                  "live PROPOSAL photon-pair loss is not exactly one");
          auto const loss = queryProposalNativeLossFraction(
              host, column.pdg_id, column.process_id,
              column.component_hash, energy, quantile);
          require(loss.status == NativeQueryStatus::Success &&
                      loss.value == 1.,
                  "native photon-pair query did not return full loss");
          require(loss.newton_iterations == 0 &&
                      loss.bisection_iterations == 0,
                  "native photon-pair full loss unnecessarily used a root solver");
        }
      }
      require(photon_pair_columns > 0,
              "native fixture contains no photon-pair rate columns");
    }
    {
      // PhotoMuPairBurkhardtKelnerKokoulin has an internal x interval in its
      // differential cross section, but PROPOSAL 7.6.2 declares the process
      // only-stochastic.  Therefore the public SampleLoss/transport variable
      // is exactly v=1; x is sampled later by the CPU secondary generator.
      std::size_t photon_mu_pair_columns = 0;
      for (auto const& column : first.dndx_columns) {
        if (column.pdg_id != 22 ||
            column.process_id != gpu::em::PhotonMuonPairProcessId)
          continue;
        ++photon_mu_pair_columns;
        require(column.kinematic_model ==
                    NativeKinematicModel::OnlyStochastic,
                "photon BKK mu-pair column is not only-stochastic");
        auto const energy = std::sqrt(
            column.spline.energy_axis.low *
            std::min(column.spline.energy_axis.high, 1.e10));
        auto const rates = photon.interaction->Rates(energy);
        auto const oracle = std::find_if(
            rates.begin(), rates.end(), [&](auto const& rate) {
              return static_cast<std::int32_t>(
                         rate.crosssection->GetInteractionType()) ==
                         column.process_id &&
                     static_cast<std::uint64_t>(rate.comp_hash) ==
                         column.component_hash;
            });
        require(oracle != rates.end() && oracle->rate > 0.,
                "cannot locate live PROPOSAL photon BKK mu-pair column");
        for (double const quantile : {0., 1.e-6, 0.25, 0.5, 0.9, 1.}) {
          auto const expected =
              oracle->crosssection->CalculateStochasticLoss(
                  oracle->comp_hash, energy, quantile * oracle->rate);
          require(expected == 1.,
                  "live PROPOSAL photon BKK mu-pair loss is not exactly one");
          auto const loss = queryProposalNativeLossFraction(
              host, column.pdg_id, column.process_id,
              column.component_hash, energy, quantile);
          require(loss.status == NativeQueryStatus::Success &&
                      loss.value == 1.,
                  "native photon BKK mu-pair loss is not exactly one");
        }
      }
      require(photon_mu_pair_columns > 0,
              "native fixture contains no photon BKK mu-pair columns");
    }
    {
      constexpr double observed_energy_MeV = 633.1475437997642;
      constexpr double observed_process_uniform = 0.2480323432246223;
      FlatRateTableView selector_view{};
      selector_view.proposal_native = host;
      selector_view.physics_source = 1u;
      selector_view.em_transport_cut_MeV = 0.5;
      selector_view.muon_transport_cut_MeV = 300.;
      for (auto const pdg : {11, -11, 13, -13}) {
        auto const mass = queryContinuousMass(selector_view, pdg);
        auto const minimum =
            queryContinuousMinimumEnergy(selector_view, pdg);
        require(mass.status == TableLookupStatus::Success &&
                    minimum.status == TableLookupStatus::Success,
                "native continuous minimum-energy query failed");
        auto const expected_cut =
            (pdg == 13 || pdg == -13) ? 300. : 0.5;
        auto const expected =
            mass.value + ContinuousCutSafetyFactor * expected_cut;
        require(std::abs(minimum.value - expected) <=
                    16. * std::numeric_limits<double>::epsilon() *
                        std::max(1., std::abs(expected)),
                "native continuous minimum energy used the wrong transport cut");
      }
      auto const selection = selectRateColumnByUniform(
          selector_view, static_cast<std::int32_t>(get_PDG(Code::MuPlus)),
          observed_energy_MeV, observed_process_uniform);
      require(selection.status == TableLookupStatus::Success,
              "host native mu+ vertex selection failed at the observed energy");
      require(selection.total_rate > 0. && selection.selected != 0u &&
                  selection.residual_quantile >= 0. &&
                  selection.residual_quantile <= 1.,
              "host native mu+ vertex selection returned an invalid result");
      auto const live_rates =
          muon_plus.interaction->Rates(observed_energy_MeV);
      auto const live_loss = muon_plus.interaction->SampleLoss(
          observed_energy_MeV, live_rates, observed_process_uniform);
      require(selection.process_id ==
                  static_cast<std::int32_t>(live_loss.type) &&
                  selection.component_hash ==
                      static_cast<std::uint64_t>(live_loss.comp_hash),
              "native original-order selection differs from live SampleLoss");
      auto const native_loss = queryProposalNativeLossFraction(
          host, static_cast<std::int32_t>(get_PDG(Code::MuPlus)),
          selection.process_id, selection.component_hash,
          observed_energy_MeV, selection.residual_quantile);
      require(native_loss.status == NativeQueryStatus::Success,
              "native original-order selected-loss query failed");
      requireClose(native_loss.value, live_loss.v_loss, 1.e-3,
                   "native original-order selected v differs from live SampleLoss");
    }
    std::vector<ProposalNativeQuery> queries;
    std::vector<double> host_values;
    std::vector<ProposalNativeQuery> expected_selected_loss_fallbacks;

    // Independently validate the muon ionization boundary against the
    // authoritative PROPOSAL parametrization.  This deliberately does not
    // transform the CPU answer through the native implementation: doing so
    // would hide a wrong v_min/v_max model behind a failed oracle residual.
    PROPOSAL::EnergyCutSettings const muon_ionization_cut(0.4, 1.);
    PROPOSAL::crosssection::IonizBetheBlochRossi const
        muon_ionization(muon_ionization_cut);
    for (auto const* owner : {&muon_minus, &muon_plus}) {
      auto const pdg = static_cast<std::int32_t>(get_PDG(owner->code));
      auto const column = std::find_if(
          first.dndx_columns.begin(), first.dndx_columns.end(),
          [pdg](auto const& candidate) {
            return candidate.pdg_id == pdg &&
                   candidate.process_id == 1000000003;
          });
      require(column != first.dndx_columns.end() &&
                  column->kinematic_model ==
                      NativeKinematicModel::IonizationBetheBlochRossi,
              "muon ionization column does not use BetheBlochRossi limits");
      auto const first_energy = std::nextafter(
          std::max(column->particle_mass_MeV,
                   column->spline.energy_axis.low),
          std::numeric_limits<double>::infinity());
      for (double const energy :
           {first_energy, 633.1475437997642, 1.e3, 1.e6}) {
        if (energy > column->spline.energy_axis.high ||
            energy <= column->particle_mass_MeV)
          continue;
        auto const raw = muon_ionization.GetKinematicLimits(
            proposal::particle.at(owner->code), owner->medium, energy);
        auto const expected_minimum =
            muon_ionization_cut.GetCut(raw, energy);
        auto const native_limits =
            native_detail::kinematicLimits(host, *column, energy);
        require(native_limits.valid,
                "native BetheBlochRossi limits are invalid");
        requireRateCloseOrUlps(
            native_limits.minimum, expected_minimum, 2.e-12, 16,
            "native BetheBlochRossi lower limit differs from PROPOSAL");
        requireRateCloseOrUlps(
            native_limits.maximum, raw.v_max, 2.e-12, 16,
            "native BetheBlochRossi upper limit differs from PROPOSAL");
        for (auto const endpoint : {0., 1.}) {
          auto const expected = endpoint == 0. ? expected_minimum : raw.v_max;
          auto const loss = queryProposalNativeLossFraction(
              host, pdg, column->process_id, column->component_hash,
              energy, endpoint);
          require(loss.status == NativeQueryStatus::Success,
                  "native BetheBlochRossi endpoint query failed");
          requireRateCloseOrUlps(
              loss.value, expected, 2.e-12, 16,
              "native BetheBlochRossi endpoint differs from PROPOSAL");
          queries.push_back(
              {ProposalNativeQueryKind::LossFraction, pdg,
               column->process_id, column->component_hash, energy,
               endpoint});
          host_values.push_back(expected);
        }
      }
    }

    for (auto const& column : first.dndx_columns) {
      if ((column.pdg_id == 11 || column.pdg_id == -11) &&
          column.process_id == gpu::em::BremsProcessId) {
        require(column.absolute_cut_MeV == 0.4 &&
                    column.relative_cut == proposal::v_cut,
                "native electron brems column lost its PROPOSAL cut");
        constexpr double near_cut_energy_MeV = 1.0365222668750245;
        auto const loss = queryProposalNativeLossFraction(
            host, column.pdg_id, column.process_id,
            column.component_hash, near_cut_energy_MeV, 0.5);
        require(loss.status == NativeQueryStatus::Success &&
                    loss.value > 0. && loss.value < 1.,
                "native near-cut brems inverse returned an invalid loss");
      }
    }
    for (auto const* owner : owners) {
      auto const pdg = static_cast<std::int32_t>(get_PDG(owner->code));
      // Include a low-energy atmospheric muon observed in an end-to-end
      // proton shower.  It exposed a finite-rate hole that the decade-only
      // probes did not exercise.
      for (double const energy :
           {10., 100., 633.1475437997642, 1.e3, 1.e5, 1.e8}) {
        auto rates = owner->interaction->Rates(energy);
        for (auto const& rate : rates) {
          auto const process = static_cast<std::int32_t>(
              rate.crosssection->GetInteractionType());
          auto const native = queryProposalNativeRate(
              host, pdg, process,
              static_cast<std::uint64_t>(rate.comp_hash), energy);
          require(native.status == NativeQueryStatus::Success,
                  "host native rate query failed: pdg=" +
                      std::to_string(pdg) + ", process=" +
                      std::to_string(process) + ", component=" +
                      std::to_string(rate.comp_hash) + ", energy=" +
                      std::to_string(energy) + ", status=" +
                      std::to_string(static_cast<unsigned>(native.status)));
          requireRateCloseOrUlps(native.value, rate.rate, 2.e-12, 16,
                       "native rate differs from PROPOSAL: pdg=" +
                           std::to_string(pdg) + ", process=" +
                           std::to_string(process) + ", energy=" +
                           std::to_string(energy));
          queries.push_back(
              {ProposalNativeQueryKind::Rate, pdg, process,
               static_cast<std::uint64_t>(rate.comp_hash), energy, 0.});
          host_values.push_back(native.value);
        }
      }
      // Exercise the aggregate query at a non-grid energy observed in the
      // end-to-end photon smoke test.  This catches a bad direct-rate column
      // even when every sampled interpolated column is individually valid.
      constexpr double aggregate_energy_MeV = 3603.59344235154;
      auto const aggregate = queryProposalNativeTotalRate(
          host, pdg, aggregate_energy_MeV);
      require(aggregate.status == NativeQueryStatus::Success &&
                  std::isfinite(aggregate.value) &&
                  aggregate.value >= 0.,
              "host native total-rate query failed");
      auto const mean_free_path =
          owner->interaction->MeanFreePath(aggregate_energy_MeV);
      auto const expected_distance_rate =
          std::isfinite(mean_free_path) ? 1. / mean_free_path : 0.;
      requireRateCloseOrUlps(
          aggregate.value, expected_distance_rate, 2.e-12, 16,
          "native mean-free-path spline differs from Interaction::MeanFreePath");
      auto const live_selection_rates =
          owner->interaction->Rates(aggregate_energy_MeV);
      auto const expected_selection_total = std::accumulate(
          live_selection_rates.begin(), live_selection_rates.end(), 0.,
          [](double sum, PROPOSAL::Interaction::Rate const& rate) {
            return sum + rate.rate;
          });
      auto const selection_total = queryProposalNativeSelectionTotalRate(
          host, pdg, aggregate_energy_MeV);
      require(selection_total.status == NativeQueryStatus::Success,
              "host native flat selection-total query failed");
      requireRateCloseOrUlps(
          selection_total.value, expected_selection_total, 2.e-12, 16,
          "native flat selection total differs from Interaction::Rates");
      queries.push_back(
          {ProposalNativeQueryKind::TotalRate, pdg, 0, 0,
           aggregate_energy_MeV, 0.});
      host_values.push_back(aggregate.value);
    }

    // Verify inverse cumulative-rate evaluation independently of final-state
    // generation. Unsupported CPU-only loss models are intentionally skipped.
    double maximum_cpu_inverse_relative_difference = 0.;
    std::size_t cpu_inverse_oracle_failures = 0;
    for (auto const& column : first.dndx_columns) {
      if (column.rate_model != NativeRateModel::BicubicSpline ||
          column.kinematic_model == NativeKinematicModel::Unsupported)
        continue;
      auto const energy = std::sqrt(
          column.spline.energy_axis.low *
          std::min(column.spline.energy_axis.high, 1.e8));
      for (double const quantile : {0., 1.e-6, 0.1, 0.5, 0.9, 1. - 1.e-6}) {
        auto const loss = queryProposalNativeLossFraction(
            host, column.pdg_id, column.process_id,
            column.component_hash, energy, quantile);
        if (loss.status == NativeQueryStatus::RootNotConverged) {
          require(quantile < 2.e-6 || quantile > 1. - 2.e-6,
                  "native inverse requested CPU completion away from a CDF endpoint");
          expected_selected_loss_fallbacks.push_back(
              {ProposalNativeQueryKind::LossFraction, column.pdg_id,
               column.process_id, column.component_hash, energy, quantile});
          continue;
        }
        require(loss.status == NativeQueryStatus::Success,
                "host native inverse-CDF query failed: pdg=" +
                    std::to_string(column.pdg_id) + ", process=" +
                    std::to_string(column.process_id) + ", component=" +
                    std::to_string(column.component_hash) + ", energy=" +
                    std::to_string(energy) + ", quantile=" +
                    std::to_string(quantile) + ", status=" +
                    std::to_string(static_cast<unsigned>(loss.status)));
        require(loss.value >= 0. && loss.value <= 1.,
                "host native inverse-CDF returned a non-fractional loss");
        auto const* owner = ownerForPdg(column.pdg_id);
        auto const rates = owner->interaction->Rates(energy);
        auto const oracle = std::find_if(
            rates.begin(), rates.end(), [&](auto const& rate) {
              return static_cast<std::int32_t>(
                         rate.crosssection->GetInteractionType()) ==
                         column.process_id &&
                     static_cast<std::uint64_t>(rate.comp_hash) ==
                         column.component_hash;
            });
        require(oracle != rates.end() && oracle->rate > 0.,
                "cannot locate PROPOSAL inverse-CDF oracle column");
        double expected = std::numeric_limits<double>::quiet_NaN();
        bool cpu_inverse_available = true;
        try {
          expected = oracle->crosssection->CalculateStochasticLoss(
              oracle->comp_hash, energy, quantile * oracle->rate);
        } catch (std::exception const&) {
          // PROPOSAL 7.6.2 can itself fail to bracket a root for a small
          // subset of tabulated points.  Record that separately: it is not a
          // native-table failure and must not suppress the host/device check.
          cpu_inverse_available = false;
          ++cpu_inverse_oracle_failures;
        }
        if (cpu_inverse_available) {
          auto const expected_transformed = native_detail::transformedLoss(
              host, column, energy, expected);
          if (!std::isfinite(expected_transformed)) {
            cpu_inverse_available = false;
          } else {
            auto const expected_cumulative =
                oracle->crosssection->EvaluateDndxInterpolation(
                    oracle->comp_hash, energy, expected_transformed);
            auto const expected_target = quantile * oracle->rate;
            auto const residual_scale = std::max(oracle->rate, 1.e-300);
            if (!std::isfinite(expected_cumulative) ||
                std::abs(expected_cumulative - expected_target) >
                    1.e-4 * residual_scale)
              cpu_inverse_available = false;
          }
          if (!cpu_inverse_available) ++cpu_inverse_oracle_failures;
        }
        auto const transformed = native_detail::transformedLoss(
            host, column, energy, loss.value);
        require(std::isfinite(transformed),
                "native physical loss cannot be transformed back");
        auto const native_cumulative = queryProposalNativeCumulativeRate(
            host, column, energy, transformed);
        require(native_cumulative.status == NativeQueryStatus::Success,
                "native cumulative-rate evaluation failed");
        auto const cpu_cumulative =
            oracle->crosssection->EvaluateDndxInterpolation(
                oracle->comp_hash, energy, transformed);
        requireCloseOrUlps(
            native_cumulative.value, cpu_cumulative, 2.e-12, 16,
            "exported cumulative spline differs from PROPOSAL's live interpolant");
        // PROPOSAL 7.6.2 intentionally stops its Boost Newton solver at 20
        // binary digits. The native device solver uses the same initial
        // bracket and reports its iterations, but the algebraically
        // equivalent Hermite evaluation can end on the neighbouring final
        // Newton iterate. Keep this oracle check explicit and bounded; the
        // stricter host/device and cumulative-residual checks below determine
        // whether the transported distribution changed.
        if (cpu_inverse_available) {
          auto const inverse_scale = std::max(std::abs(expected), 1.e-300);
          maximum_cpu_inverse_relative_difference = std::max(
              maximum_cpu_inverse_relative_difference,
              std::abs(loss.value - expected) / inverse_scale);
          requireClose(
              loss.value, expected, 1.e-3,
              "native inverse-CDF differs from PROPOSAL oracle: pdg=" +
                  std::to_string(column.pdg_id) + ", process=" +
                  std::to_string(column.process_id) + ", component=" +
                  std::to_string(column.component_hash) + ", energy=" +
                  std::to_string(energy) + ", quantile=" +
                  std::to_string(quantile) + ", cpu_rate=" +
                  std::to_string(oracle->rate));
        }
        queries.push_back(
            {ProposalNativeQueryKind::LossFraction, column.pdg_id,
             column.process_id, column.component_hash, energy, quantile});
        host_values.push_back(loss.value);
      }
    }

    // Continuous transport must use the same one-dimensional dE/dX and
    // displacement interpolants as the scalar calculator. Validate their
    // values and append identical queries to the CUDA comparison below.
    for (auto const* owner :
         {&electron, &positron, &muon_minus, &muon_plus}) {
      auto const pdg = static_cast<std::int32_t>(get_PDG(owner->code));
      auto const utility = std::find_if(
          first.utility_columns.begin(), first.utility_columns.end(),
          [pdg](auto const& column) { return column.pdg_id == pdg; });
      require(utility != first.utility_columns.end(),
              "native export has no continuous utility column");
      auto const range_minimum = utility->lower_energy_limit_MeV;
      auto const low_energy = std::nextafter(
          range_minimum, std::numeric_limits<double>::infinity());
      auto const low_dedx =
          queryProposalNativeDedx(host, pdg, low_energy);
      require(low_dedx.status == NativeQueryStatus::Success,
              "host native low-energy dE/dX query failed");
      double expected_low_dedx = 0.;
      for (auto const& cross_section : owner->cross_sections)
        expected_low_dedx += cross_section->CalculatedEdx(low_energy);
      requireCloseOrUlps(
          low_dedx.value, expected_low_dedx, 2.e-12, 16,
          "native low-energy dE/dX differs from scalar PROPOSAL");
      queries.push_back(
          {ProposalNativeQueryKind::ContinuousDedx, pdg, 0, 0,
           low_energy, 0.});
      host_values.push_back(low_dedx.value);
      auto minimum = range_minimum;
      auto maximum = std::min(utility->spline.axis.high, 1.e8);
      for (auto const& dedx_column : first.dedx_columns) {
        if (dedx_column.pdg_id != pdg) continue;
        minimum = std::max(
            {minimum, dedx_column.lower_energy_limit_MeV,
             dedx_column.spline.axis.low});
        maximum = std::min(maximum, dedx_column.spline.axis.high);
      }
      require(maximum > minimum,
              "continuous native columns have no common energy domain");
      for (double const fraction : {0.01, 0.1, 0.5, 0.9}) {
        auto const energy = std::exp(
            std::log(minimum) +
            fraction * (std::log(maximum) - std::log(minimum)));

        auto const dedx = queryProposalNativeDedx(host, pdg, energy);
        require(dedx.status == NativeQueryStatus::Success,
                "host native dE/dX query failed");
        double expected_dedx = 0.;
        for (auto const& cross_section : owner->cross_sections)
          expected_dedx += cross_section->CalculatedEdx(energy);
        requireCloseOrUlps(
            dedx.value, expected_dedx, 2.e-12, 16,
            "native dE/dX differs from scalar PROPOSAL");
        queries.push_back(
            {ProposalNativeQueryKind::ContinuousDedx, pdg, 0, 0,
             energy, 0.});
        host_values.push_back(dedx.value);

        auto const range = queryProposalNativeRange(host, pdg, energy);
        require(range.status == NativeQueryStatus::Success,
                "host native range query failed");
        auto const expected_range = owner->displacement->SolveTrackIntegral(
            energy, range_minimum);
        requireCloseOrUlps(
            range.value, expected_range, 1.e-11, 16,
            "native range differs from scalar PROPOSAL");
        queries.push_back(
            {ProposalNativeQueryKind::ContinuousRange, pdg, 0, 0,
             energy, 0.});
        host_values.push_back(range.value);

        auto const inverse = queryProposalNativeEnergy(host, pdg, range.value);
        require(inverse.status == NativeQueryStatus::Success,
                "host native inverse-range query failed");
        requireClose(inverse.value, energy, 1.e-10,
                     "native range/inverse-range round trip changed energy");
        queries.push_back(
            {ProposalNativeQueryKind::ContinuousEnergy, pdg, 0, 0,
             0., range.value});
        host_values.push_back(inverse.value);
      }
    }

    CudaProposalNativeTable device;
    device.initialize(first, 0);
    require(device.deviceBytes() == proposalNativeTableBytes(first),
            "native CUDA byte accounting differs from host export");
    require(device.sourceContentHash() == first.content_hash,
            "native CUDA table lost its source hash");
    std::vector<NativeDndxColumn> uploaded_dndx(
        first.dndx_columns.size());
    require(
        cudaMemcpy(uploaded_dndx.data(), device.deviceView().dndx_columns,
                   uploaded_dndx.size() * sizeof(NativeDndxColumn),
                   cudaMemcpyDeviceToHost) == cudaSuccess,
        "cannot download native dN/dX descriptors for upload audit");
    require(
        std::memcmp(uploaded_dndx.data(), first.dndx_columns.data(),
                    uploaded_dndx.size() * sizeof(NativeDndxColumn)) == 0,
        "native CUDA dN/dX descriptor upload changed bytes");

    // The upload boundary must authenticate the complete canonical payload,
    // not merely trust the caller-provided digest.  Change one finite spline
    // coefficient by one representable double while deliberately retaining
    // the old hash; initialize() must fail before any such table reaches a
    // kernel.
    require(!first.bicubic_values.empty(),
            "native export has no bicubic coefficient for hash guard");
    auto corrupted = first;
    corrupted.bicubic_values.front() = std::nextafter(
        corrupted.bicubic_values.front(),
        std::numeric_limits<double>::infinity());
    require(std::isfinite(corrupted.bicubic_values.front()),
            "hash-guard mutation produced a non-finite coefficient");
    bool rejected_corrupted_hash = false;
    try {
      CudaProposalNativeTable corrupted_device;
      corrupted_device.initialize(corrupted, 0);
    } catch (std::exception const&) {
      rejected_corrupted_hash = true;
    }
    require(rejected_corrupted_hash,
            "native CUDA upload accepted a payload with a stale hash");

    if (std::getenv("C8_PROPOSAL_NATIVE_EIGEN_PROBE") != nullptr) {
      constexpr std::int32_t probe_pdg = -13;
      constexpr std::int32_t probe_process =
          gpu::em::ElectronPairProcessId;
      constexpr std::uint64_t probe_component =
          7183588842005576818ull;
      constexpr double probe_energy = 338.95457894315058;
      auto const* probe_column = native_detail::findDndx(
          host, probe_pdg, probe_process, probe_component);
      require(probe_column != nullptr, "Eigen probe column is absent");
      auto const native = queryProposalNativeRate(
          host, probe_pdg, probe_process, probe_component, probe_energy);
      auto const eigen = evaluateEigenReference(
                             host, probe_column->spline,
                             probe_energy, 1.) /
                         probe_column->component_weight;
      auto const staged_sequential = evaluateStagedReference(
                                         host, probe_column->spline,
                                         probe_energy, 1., 0u) /
                                     probe_column->component_weight;
      auto const staged_pairwise = evaluateStagedReference(
                                       host, probe_column->spline,
                                       probe_energy, 1., 15u) /
                                   probe_column->component_weight;
      auto const hermite_pairwise = evaluateHermitePairwiseReference(
                                        host, probe_column->spline,
                                        probe_energy, 1.) /
                                    probe_column->component_weight;
      auto const precomputed = evaluatePrecomputedCoefficientReference(
                                   host, probe_column->spline,
                               probe_energy, 1.) /
                               probe_column->component_weight;
      auto const precomputed_ss = evaluatePrecomputedCoefficientReference(
                                      host, probe_column->spline,
                                      probe_energy, 1., false, false) /
                                  probe_column->component_weight;
      auto const precomputed_sp = evaluatePrecomputedCoefficientReference(
                                      host, probe_column->spline,
                                      probe_energy, 1., false, true) /
                                  probe_column->component_weight;
      auto const precomputed_ps = evaluatePrecomputedCoefficientReference(
                                      host, probe_column->spline,
                                      probe_energy, 1., true, false) /
                                  probe_column->component_weight;
      auto const long_double = evaluateLongDoubleReference(
                                   host, probe_column->spline,
                                   probe_energy, 1.) /
                               probe_column->component_weight;
      auto const* probe_owner = ownerForPdg(probe_pdg);
      auto const live_rates = probe_owner->interaction->Rates(probe_energy);
      auto const live = std::find_if(
          live_rates.begin(), live_rates.end(), [](auto const& rate) {
            return rate.crosssection &&
                   static_cast<std::int32_t>(
                       rate.crosssection->GetInteractionType()) ==
                       gpu::em::ElectronPairProcessId &&
                   static_cast<std::uint64_t>(rate.comp_hash) ==
                       probe_component;
          });
      require(live != live_rates.end(), "Eigen probe live rate is absent");
      unsigned int best_staged_mask = 0;
      double best_staged = 0.;
      double best_staged_relative = std::numeric_limits<double>::infinity();
      for (unsigned int mask = 0; mask < 16; ++mask) {
        auto const candidate = evaluateStagedReference(
                                   host, probe_column->spline,
                                   probe_energy, 1., mask) /
                               probe_column->component_weight;
        auto const relative = relativeDifference(candidate, live->rate);
        if (relative < best_staged_relative) {
          best_staged_mask = mask;
          best_staged = candidate;
          best_staged_relative = relative;
        }
      }
      std::cout << std::setprecision(17)
                << "eigen_rate_probe: native=" << native.value
                << ", eigen=" << eigen << ", live=" << live->rate
                << ", staged_sequential=" << staged_sequential
                << ", staged_pairwise=" << staged_pairwise
                << ", hermite_pairwise=" << hermite_pairwise
                << ", precomputed=" << precomputed
                << ", precomputed_ss=" << precomputed_ss
                << ", precomputed_sp=" << precomputed_sp
                << ", precomputed_ps=" << precomputed_ps
                << ", long_double=" << long_double
                << ", best_staged_mask=" << best_staged_mask
                << ", best_staged=" << best_staged
                << ", native_relative="
                << relativeDifference(native.value, live->rate)
                << ", eigen_relative="
                << relativeDifference(eigen, live->rate)
                << ", sequential_relative="
                << relativeDifference(staged_sequential, live->rate)
                << ", pairwise_relative="
                << relativeDifference(staged_pairwise, live->rate)
                << ", hermite_pairwise_relative="
                << relativeDifference(hermite_pairwise, live->rate)
                << ", precomputed_relative="
                << relativeDifference(precomputed, live->rate)
                << ", precomputed_ss_relative="
                << relativeDifference(precomputed_ss, live->rate)
                << ", precomputed_sp_relative="
                << relativeDifference(precomputed_sp, live->rate)
                << ", precomputed_ps_relative="
                << relativeDifference(precomputed_ps, live->rate)
                << ", long_double_relative="
                << relativeDifference(long_double, live->rate)
                << ", best_staged_relative=" << best_staged_relative
                << '\n';
    }
    auto const gpu = device.queryForValidation(queries);
    require(gpu.size() == host_values.size(),
            "native CUDA validation result count changed");
    for (std::size_t i = 0; i < gpu.size(); ++i) {
      auto const& query = queries[i];
      require(
          gpu[i].status == NativeQueryStatus::Success,
          "native CUDA validation query failed: index=" +
              std::to_string(i) + ", kind=" +
              std::to_string(static_cast<unsigned>(query.kind)) +
              ", pdg=" + std::to_string(query.pdg_id) +
              ", process=" + std::to_string(query.process_id) +
              ", component=" + std::to_string(query.component_hash) +
              ", energy=" + std::to_string(query.energy_MeV) +
              ", argument=" + std::to_string(query.argument) +
              ", status=" +
              std::to_string(static_cast<unsigned>(gpu[i].status)));
      auto relative_tolerance = 2.e-12;
      if (query.kind == ProposalNativeQueryKind::LossFraction)
        relative_tolerance = 1.e-10;
      else if (query.kind == ProposalNativeQueryKind::ContinuousDedx ||
               query.kind == ProposalNativeQueryKind::ContinuousRange ||
               query.kind == ProposalNativeQueryKind::ContinuousEnergy)
        relative_tolerance = 1.e-11;
      requireCloseOrUlps(
          gpu[i].value, host_values[i], relative_tolerance, 16,
          "host/device native table evaluation differs: index=" +
              std::to_string(i) + ", kind=" +
              std::to_string(static_cast<unsigned>(query.kind)) +
              ", pdg=" + std::to_string(query.pdg_id) +
              ", process=" + std::to_string(query.process_id) +
              ", energy=" + std::to_string(query.energy_MeV));
    }
    auto const fallback_gpu =
        device.queryForValidation(expected_selected_loss_fallbacks);
    require(fallback_gpu.size() == expected_selected_loss_fallbacks.size(),
            "fixed selected-loss fallback count changed");
    for (auto const& result : fallback_gpu)
      require(result.status == NativeQueryStatus::RootNotConverged,
              "host/device selected-loss fallback decision differs");

    // Opt-in numerical probe for reproducing a specific tail inverse without
    // paying for the high-statistics campaign.  This is deliberately a test
    // diagnostic and does not affect the production solver or default ctest
    // output.
    if (std::getenv("C8_PROPOSAL_NATIVE_PROBE") != nullptr) {
      constexpr std::int32_t probe_pdg = -11;
      constexpr std::int32_t probe_process =
          gpu::em::IonizationProcessId;
      constexpr double probe_energy = 748935993076.25818;
      constexpr double probe_quantile = 0.99999989486566343;
      auto const rates = positron.interaction->Rates(probe_energy);
      for (auto const& probe_column : first.dndx_columns) {
        if (probe_column.pdg_id != probe_pdg ||
            probe_column.process_id != probe_process)
          continue;
        auto const probe_component = probe_column.component_hash;
        auto const host_probe = queryProposalNativeLossFraction(
            host, probe_pdg, probe_process, probe_component,
            probe_energy, probe_quantile);
        auto const gpu_probe = device.queryForValidation({{
            ProposalNativeQueryKind::LossFraction, probe_pdg,
            probe_process, probe_component, probe_energy,
            probe_quantile}}).front();
        auto const total_probe = queryProposalNativeCumulativeRate(
            host, probe_column, probe_energy, 1.);
        auto probeResidual = [&](double physical_loss) {
          auto const transformed = native_detail::transformedLoss(
              host, probe_column, probe_energy, physical_loss);
          auto const cumulative = queryProposalNativeCumulativeRate(
              host, probe_column, probe_energy, transformed);
          return (cumulative.value - probe_quantile * total_probe.value) /
                 std::max(total_probe.value, 1.e-300);
        };
        auto const live = std::find_if(
            rates.begin(), rates.end(), [&](auto const& rate) {
              return rate.crosssection &&
                     static_cast<std::int32_t>(
                         rate.crosssection->GetInteractionType()) ==
                         probe_process &&
                     static_cast<std::uint64_t>(rate.comp_hash) ==
                         probe_component;
            });
        require(live != rates.end(), "tail inverse live oracle is absent");
        auto const live_loss = live->crosssection->CalculateStochasticLoss(
            live->comp_hash, probe_energy, probe_quantile * live->rate);
        std::cout << std::setprecision(17)
                  << "tail_inverse_probe: component=" << probe_component
                  << ", host=" << host_probe.value
                  << ", gpu=" << gpu_probe.value
                  << ", live=" << live_loss
                  << ", host_relative_residual="
                  << probeResidual(host_probe.value)
                  << ", gpu_relative_residual="
                  << probeResidual(gpu_probe.value)
                  << ", live_relative_residual="
                  << probeResidual(live_loss)
                  << ", host_newton=" << host_probe.newton_iterations
                  << ", gpu_newton=" << gpu_probe.newton_iterations
                  << ", host_bisection=" << host_probe.bisection_iterations
                  << ", gpu_bisection=" << gpu_probe.bisection_iterations
                  << '\n';
      }
    }

    // Optional high-statistics acceptance mode. The ordinary ctest remains
    // quick; invoking this executable with e.g. `1000000 65536` evaluates at
    // least one million deterministic (E,u) points for every exported
    // PID/process/component column without allocating the full campaign at
    // once. This compares the GPU against both the exported host evaluator
    // and the live scalar PROPOSAL interpolant used to create the view.
    std::uint64_t randomized_rate_queries = 0;
    std::uint64_t randomized_cumulative_queries = 0;
    std::uint64_t randomized_loss_queries = 0;
    std::uint64_t randomized_rate_outside_acceptance = 0;
    std::uint64_t randomized_cumulative_outside_acceptance = 0;
    std::uint64_t randomized_loss_outside_acceptance = 0;
    std::uint64_t randomized_selected_loss_fallbacks = 0;
    std::uint64_t randomized_fallback_status_mismatches = 0;
    double randomized_max_rate_relative = 0.;
    double randomized_max_rate_absolute = 0.;
    double randomized_max_cumulative_relative = 0.;
    double randomized_max_cumulative_rate_normalized = 0.;
    double randomized_max_loss_relative = 0.;
    std::string randomized_max_loss_record;
    std::uint64_t randomized_cumulative_floor_samples = 0;
    auto const unitUniform = [](std::uint64_t value) {
      value += 0x9e3779b97f4a7c15ull;
      value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
      value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
      value ^= value >> 31;
      return (static_cast<double>(value >> 11) + 0.5) *
             0x1.0p-53;
    };
    if (randomized_samples_per_column != 0) {
      for (std::size_t column_index = 0;
           column_index < first.dndx_columns.size(); ++column_index) {
        auto const& column = first.dndx_columns[column_index];
        auto const* owner = ownerForPdg(column.pdg_id);
        auto const cross = std::find_if(
            owner->cross_sections.begin(), owner->cross_sections.end(),
            [&](auto const& candidate) {
              return candidate &&
                     static_cast<std::int32_t>(
                         candidate->GetInteractionType()) ==
                         column.process_id;
            });
        require(cross != owner->cross_sections.end(),
                "randomized oracle cannot locate PROPOSAL cross section");

        auto energy_low = std::max(
            column.lower_energy_limit_MeV,
            column.rate_model == NativeRateModel::BicubicSpline
                ? column.spline.energy_axis.low
                : column.lower_energy_limit_MeV);
        if (!(energy_low > 0.)) energy_low = 0.4;
        energy_low *= 1. + 64. * std::numeric_limits<double>::epsilon();
        auto energy_high =
            column.rate_model == NativeRateModel::BicubicSpline
                ? column.spline.energy_axis.high
                : 1.e14;
        energy_high *= 1. - 64. * std::numeric_limits<double>::epsilon();
        require(energy_high > energy_low,
                "randomized oracle column has no positive energy domain");
        auto const log_low = std::log(energy_low);
        auto const log_high = std::log(energy_high);

        for (std::size_t begin = 0;
             begin < randomized_samples_per_column;
             begin += randomized_chunk_size) {
          auto const count = std::min(
              randomized_chunk_size,
              randomized_samples_per_column - begin);
          std::vector<ProposalNativeQuery> randomized_queries;
          std::vector<double> randomized_host_values;
          std::vector<NativeQueryStatus> randomized_host_status;
          randomized_queries.reserve(
              count * (column.rate_model ==
                               NativeRateModel::BicubicSpline &&
                           column.kinematic_model !=
                               NativeKinematicModel::Unsupported
                       ? 3
                       : 1));
          randomized_host_values.reserve(randomized_queries.capacity());
          randomized_host_status.reserve(randomized_queries.capacity());
          for (std::size_t local = 0; local < count; ++local) {
            auto const sample = begin + local;
            auto const key =
                (static_cast<std::uint64_t>(column_index) << 48) ^ sample;
            auto const energy_coordinate = unitUniform(key);
            auto const u = unitUniform(key ^ 0xd1b54a32d192ed03ull);
            auto const energy = std::exp(
                log_low + energy_coordinate * (log_high - log_low));

            auto const native_rate = queryProposalNativeRate(
                host, column.pdg_id, column.process_id,
                column.component_hash, energy);
            require(native_rate.status == NativeQueryStatus::Success,
                    "randomized host rate query failed");
            auto const proposal_rate = (*cross)->CalculatedNdx(
                energy, static_cast<std::size_t>(column.component_hash));
            requireRateCloseOrUlps(
                native_rate.value, proposal_rate, 2.e-12, 16,
                "randomized native rate differs from live PROPOSAL: pdg=" +
                    std::to_string(column.pdg_id) + ", process=" +
                    std::to_string(column.process_id) + ", component=" +
                    std::to_string(column.component_hash) + ", energy=" +
                    preciseDouble(energy));
            randomized_max_rate_relative = std::max(
                randomized_max_rate_relative,
                relativeDifference(native_rate.value, proposal_rate));
            randomized_max_rate_absolute = std::max(
                randomized_max_rate_absolute,
                std::abs(native_rate.value - proposal_rate));
            randomized_queries.push_back(
                {ProposalNativeQueryKind::Rate, column.pdg_id,
                 column.process_id, column.component_hash, energy, 0.});
            randomized_host_values.push_back(native_rate.value);
            randomized_host_status.push_back(native_rate.status);
            ++randomized_rate_queries;

            if (column.rate_model != NativeRateModel::BicubicSpline)
              continue;
            auto const native_cumulative =
                queryProposalNativeCumulativeRate(host, column, energy, u);
            require(native_cumulative.status == NativeQueryStatus::Success,
                    "randomized host cumulative-rate query failed");
            auto const proposal_cumulative =
                (*cross)->EvaluateDndxInterpolation(
                    static_cast<std::size_t>(column.component_hash), energy,
                    u);
            auto const cumulative_absolute = std::abs(
                native_cumulative.value - proposal_cumulative);
            auto const cumulative_floor =
                std::abs(proposal_rate) * 1.e-12;
            auto const cumulative_denominator = std::max(
                {std::abs(proposal_cumulative), cumulative_floor, 1.e-300});
            auto const cumulative_normalized =
                cumulative_absolute / cumulative_denominator;
            if (std::abs(proposal_cumulative) < cumulative_floor)
              ++randomized_cumulative_floor_samples;
            randomized_max_cumulative_rate_normalized = std::max(
                randomized_max_cumulative_rate_normalized,
                cumulative_normalized);
            if (cumulative_normalized > 1.e-10 &&
                positiveUlpDistance(
                    native_cumulative.value, proposal_cumulative) > 16) {
              std::ostringstream message;
              message << std::setprecision(17)
                      << "randomized cumulative rate differs from live "
                         "PROPOSAL: pdg="
                      << column.pdg_id << ", process=" << column.process_id
                      << ", component=" << column.component_hash
                      << ", energy=" << energy
                      << ", transformed_loss=" << u
                      << ", actual=" << native_cumulative.value
                      << ", expected=" << proposal_cumulative
                      << ", raw_relative="
                      << relativeDifference(
                             native_cumulative.value, proposal_cumulative)
                      << ", column_rate_normalized="
                      << cumulative_normalized;
              throw std::runtime_error(message.str());
            }
            randomized_max_cumulative_relative = std::max(
                randomized_max_cumulative_relative,
                relativeDifference(
                    native_cumulative.value, proposal_cumulative));
            randomized_queries.push_back(
                {ProposalNativeQueryKind::CumulativeRate, column.pdg_id,
                 column.process_id, column.component_hash, energy, u});
            randomized_host_values.push_back(native_cumulative.value);
            randomized_host_status.push_back(native_cumulative.status);
            ++randomized_cumulative_queries;

            if (column.kinematic_model ==
                NativeKinematicModel::Unsupported)
              continue;
            auto const loss = queryProposalNativeLossFraction(
                host, column.pdg_id, column.process_id,
                column.component_hash, energy, u);
            require(
                loss.status == NativeQueryStatus::Success ||
                    loss.status == NativeQueryStatus::RootNotConverged,
                "randomized host inverse-CDF query failed: pdg=" +
                    std::to_string(column.pdg_id) + ", process=" +
                    std::to_string(column.process_id) + ", component=" +
                    std::to_string(column.component_hash) + ", energy=" +
                    std::to_string(energy) + ", quantile=" +
                    std::to_string(u) + ", status=" +
                    std::to_string(static_cast<unsigned>(loss.status)));
            randomized_queries.push_back(
                {ProposalNativeQueryKind::LossFraction, column.pdg_id,
                 column.process_id, column.component_hash, energy, u});
            randomized_host_values.push_back(loss.value);
            randomized_host_status.push_back(loss.status);
            ++randomized_loss_queries;
          }

          auto const randomized_gpu =
              device.queryForValidation(randomized_queries);
          require(randomized_gpu.size() == randomized_host_values.size(),
                  "randomized CUDA query count changed");
          require(randomized_gpu.size() == randomized_host_status.size(),
                  "randomized native status count changed");
          for (std::size_t index = 0; index < randomized_gpu.size(); ++index) {
            auto const kind = randomized_queries[index].kind;
            if (randomized_host_status[index] ==
                NativeQueryStatus::RootNotConverged) {
              ++randomized_selected_loss_fallbacks;
              if (kind != ProposalNativeQueryKind::LossFraction ||
                  randomized_gpu[index].status !=
                      NativeQueryStatus::RootNotConverged)
                ++randomized_fallback_status_mismatches;
              continue;
            }
            require(randomized_gpu[index].status ==
                        NativeQueryStatus::Success,
                    "randomized CUDA query failed");
            auto const relative = relativeDifference(
                randomized_gpu[index].value,
                randomized_host_values[index]);
            auto const tolerance =
                kind == ProposalNativeQueryKind::LossFraction
                    ? 1.e-10
                    : (kind == ProposalNativeQueryKind::CumulativeRate
                           ? 1.e-10
                           : 2.e-12);
            auto const absolute = std::abs(
                randomized_gpu[index].value -
                randomized_host_values[index]);
            auto const accepted =
                std::isfinite(randomized_gpu[index].value) &&
                ((kind == ProposalNativeQueryKind::Rate
                      ? (absolute <= 2.e-12 || relative <= 2.e-12)
                      : relative <= tolerance) ||
                 positiveUlpDistance(
                     randomized_gpu[index].value,
                     randomized_host_values[index]) <= 16);
            if (!accepted) {
              if (kind == ProposalNativeQueryKind::LossFraction)
                ++randomized_loss_outside_acceptance;
              else if (kind == ProposalNativeQueryKind::CumulativeRate)
                ++randomized_cumulative_outside_acceptance;
              else
                ++randomized_rate_outside_acceptance;
            }
            if (kind == ProposalNativeQueryKind::LossFraction)
              {
                if (relative > randomized_max_loss_relative) {
                  randomized_max_loss_relative = relative;
                  std::ostringstream record;
                  record << std::setprecision(17)
                         << "pdg=" << randomized_queries[index].pdg_id
                         << ", process="
                         << randomized_queries[index].process_id
                         << ", component="
                         << randomized_queries[index].component_hash
                         << ", energy="
                         << randomized_queries[index].energy_MeV
                         << ", quantile="
                         << randomized_queries[index].argument
                         << ", host=" << randomized_host_values[index]
                         << ", gpu=" << randomized_gpu[index].value
                         << ", gpu_newton="
                         << randomized_gpu[index].newton_iterations
                         << ", gpu_bisection="
                         << randomized_gpu[index].bisection_iterations;
                  randomized_max_loss_record = record.str();
                }
              }
            else if (kind == ProposalNativeQueryKind::CumulativeRate)
              randomized_max_cumulative_relative = std::max(
                  randomized_max_cumulative_relative, relative);
            else if (kind == ProposalNativeQueryKind::Rate)
              randomized_max_rate_relative = std::max(
                  randomized_max_rate_relative, relative);
          }
        }
      }
    }

    std::cout << "PROPOSAL native table checks passed: " << checks
              << ", columns=" << first.dndx_columns.size()
              << ", bytes=" << device.deviceBytes()
              << ", maximum_cpu_inverse_relative_difference="
              << std::setprecision(8)
              << maximum_cpu_inverse_relative_difference
              << ", cpu_inverse_oracle_failures="
              << cpu_inverse_oracle_failures
              << ", randomized_samples_per_column="
              << randomized_samples_per_column
              << ", randomized_rate_queries="
              << randomized_rate_queries
              << ", randomized_cumulative_queries="
              << randomized_cumulative_queries
              << ", randomized_loss_queries="
              << randomized_loss_queries
              << ", randomized_rate_outside_acceptance="
              << randomized_rate_outside_acceptance
              << ", randomized_cumulative_outside_acceptance="
              << randomized_cumulative_outside_acceptance
              << ", randomized_loss_outside_acceptance="
              << randomized_loss_outside_acceptance
              << ", randomized_selected_loss_fallbacks="
              << randomized_selected_loss_fallbacks
              << ", randomized_fallback_status_mismatches="
              << randomized_fallback_status_mismatches
              << ", randomized_max_rate_relative="
              << randomized_max_rate_relative
              << ", randomized_max_rate_absolute="
              << randomized_max_rate_absolute
              << ", randomized_max_cumulative_relative="
              << randomized_max_cumulative_relative
              << ", randomized_max_cumulative_rate_normalized="
              << randomized_max_cumulative_rate_normalized
              << ", randomized_cumulative_floor_samples="
              << randomized_cumulative_floor_samples
              << ", randomized_max_loss_relative="
              << randomized_max_loss_relative
              << ", randomized_max_loss_record=\""
              << randomized_max_loss_record << "\"\n";
    if (randomized_rate_outside_acceptance != 0 ||
        randomized_cumulative_outside_acceptance != 0 ||
        randomized_loss_outside_acceptance != 0 ||
        randomized_fallback_status_mismatches != 0) {
      throw std::runtime_error(
          "randomized proposal-native acceptance limits were exceeded; "
          "see the preceding counters");
    }
    std::filesystem::remove_all(aux_directory);
  } catch (std::exception const& error) {
    std::cerr << "testGpuProposalNativeTable failed: " << error.what()
              << '\n';
    return 1;
  }
  return 0;
}
