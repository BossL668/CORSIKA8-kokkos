/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

// Validation-only first-divergence oracle for the proposal-native backend.
// This executable is excluded from normal builds and never changes cascade
// state. It feeds the uniforms emitted by the real CUDA selectors into the
// exported host evaluator and the live scalar PROPOSAL rate calculators.

#include <cuda_runtime_api.h>
#include <PROPOSAL/PROPOSAL.h>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/CudaInteractionSelector.hpp>
#include <corsika/gpu/em/tables/CudaProposalNativeTable.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>
#include <corsika/gpu/em/tables/ProposalNativeAux.hpp>
#include <corsika/gpu/em/tables/ProposalNativeTableExporter.hpp>
#include <corsika/media/MediumProperties.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  using namespace corsika;
  using namespace corsika::gpu::em;
  using namespace corsika::gpu::em::tables;
  using namespace corsika::units::si;

  constexpr double InteractionOracleTolerance = 1.e-10;
  constexpr std::uint64_t InteractionOracleUlpTolerance = 16;

  PROPOSAL::Medium makeProductionCorsikaAir() {
    // Match ProposalProcessBase exactly for the five-layer atmosphere used by
    // c8_air_shower. PROPOSAL::Air is close to, but not byte-identical with,
    // CORSIKA's AirDry1Atm composition and reference density. The oracle must
    // exercise the same calculator tables as the production exporter.
    auto const& properties = mediumData(Medium::AirDry1Atm);
    std::vector<PROPOSAL::Component> components;
    for (auto const& [code, fraction] :
         std::array<std::pair<Code, double>, 3>{
             std::pair{Code::Nitrogen, 0.78479},
             std::pair{Code::Oxygen, 0.21052},
             std::pair{Code::Argon, 0.00469}}) {
      components.emplace_back(std::string(get_name(code)),
                              get_nucleus_Z(code), get_nucleus_A(code),
                              fraction);
    }
    return PROPOSAL::Medium(
        properties.getName(), properties.getIeff(), -properties.getCbar(),
        properties.getAA(), properties.getSK(), properties.getX0(),
        properties.getX1(), properties.getDlt0(),
        properties.getCorrectedDensity(), std::move(components));
  }

  std::size_t productionCorsikaAirCompositionHash() {
    return NuclearComposition(
               {Code::Nitrogen, Code::Oxygen, Code::Argon},
               {0.78479, 0.21052, 0.00469})
        .getHash();
  }

  struct CalculatorOwner {
    PROPOSAL::Medium medium{makeProductionCorsikaAir()};
    std::size_t medium_hash{productionCorsikaAirCompositionHash()};
    Code code{Code::Unknown};
    PROPOSAL::crosssection_list_t cross_sections;
    std::unique_ptr<PROPOSAL::Interaction> interaction;
    std::unique_ptr<PROPOSAL::Displacement> displacement;
    std::unique_ptr<PROPOSAL::crosssection::PhotoPairLPM> photon_pair_lpm;
    std::unique_ptr<PROPOSAL::crosssection::BremsLPM> brems_lpm;
    double mass_MeV{};
    HEPEnergyType stochastic_cut{};

    explicit CalculatorOwner(Code requested,
                             HEPEnergyType requested_stochastic_cut)
        : code(requested), mass_MeV(proposal::particle.at(requested).mass),
          stochastic_cut(requested_stochastic_cut) {
      cross_sections =
          proposal::make_cross_sections(code, medium, stochastic_cut, true);
      interaction = PROPOSAL::make_interaction(cross_sections, true, true);
      if (code != Code::Photon)
        displacement = PROPOSAL::make_displacement(cross_sections, true);
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
      return {code, medium_hash, &medium, interaction.get(),
              photon_pair_lpm.get(), brems_lpm.get(), stochastic_cut,
              mass_MeV};
    }
    proposal::NativeContinuousCalculatorView continuousView() const {
      return {code, medium_hash, &medium, displacement.get(),
              stochastic_cut, mass_MeV};
    }
    double stochasticCutMeV() const {
      return stochastic_cut / 1_MeV;
    }

  };

  struct FirstDivergence {
    std::string kind;
    std::int32_t pid{};
    std::size_t sample{};
    double energy_MeV{};
    double process_uniform{};
    double loss_uniform{};
    std::int32_t expected_process{};
    std::int32_t gpu_process{};
    std::uint64_t expected_component{};
    std::uint64_t gpu_component{};
    double expected_v{};
    double gpu_v{};
    double relative_v{};
    double live_loss_uniform{};
  };

  struct Summary {
    std::int32_t pid{};
    std::size_t requested{};
    std::size_t compared{};
    std::size_t fallbacks{};
    std::size_t native_process_mismatches{};
    std::size_t live_proposal_sample_loss_mismatches{};
    std::size_t residual_quantile_outside_tolerance{};
    std::size_t supported_loss_compared{};
    std::size_t loss_outside_tolerance{};
    double maximum_loss_relative{};
    std::optional<FirstDivergence> first;
    std::optional<FirstDivergence> first_loss_divergence;
    std::vector<FirstDivergence> loss_divergences;
  };

  struct ScalarDivergence {
    std::string kind;
    std::int32_t pid{};
    std::int32_t process{};
    std::uint64_t component{};
    std::size_t sample{};
    double energy_MeV{};
    double argument{};
    double live_value{};
    double exported_host_value{};
    double gpu_value{};
    double relative_difference{};
    double scale_normalized_difference{};
    double live_equation_residual{};
    double exported_host_equation_residual{};
    NativeQueryStatus gpu_status{NativeQueryStatus::InvalidView};
    NativeQueryStatus exported_host_status{NativeQueryStatus::InvalidView};
  };

  struct LiveOracleException {
    std::int32_t pid{};
    std::int32_t process{};
    std::uint64_t component{};
    std::size_t sample{};
    double energy_MeV{};
    double argument{};
    std::string message;
  };

  struct MetricSummary {
    std::size_t live_oracle_attempts{};
    std::size_t live_calculate_stochastic_loss_calls{};
    std::size_t compared{};
    std::size_t device_fallbacks{};
    std::size_t live_oracle_failures{};
    // PROPOSAL's Newton/bisection implementation reports an unbracketed root
    // as MathException.  Keep this separate from all other live-oracle
    // failures: a scalar-root failure is a real validation outcome and must
    // never be hidden as a successful CUDA comparison or a device fallback.
    std::size_t live_root_exceptions{};
    std::size_t status_failures{};
    std::size_t outside_1e10{};
    std::size_t exported_host_outside_1e10{};
    std::size_t outside_1e11{};
    std::size_t exported_host_outside_1e11{};
    double maximum_relative{};
    double maximum_scale_normalized{};
    double maximum_exported_host_scale_normalized{};
    std::optional<ScalarDivergence> first;
    std::optional<LiveOracleException> first_live_root_exception;
  };

  struct ColumnOracleSummary {
    std::int32_t pid{};
    std::int32_t process{};
    std::uint64_t component{};
    MetricSummary rate;
    MetricSummary cumulative_rate;
    MetricSummary inverse_loss;
  };

  struct ContinuousOracleSummary {
    std::int32_t pid{};
    MetricSummary dedx;
    MetricSummary range;
    MetricSummary inverse_range;
  };

  struct Selection {
    std::int32_t process{};
    std::uint64_t component{};
    double residual_quantile{};
    double v_loss{};
    double total_rate{};
  };

  struct BoundaryDivergence {
    std::int32_t pid{};
    std::int32_t boundary_process{};
    std::uint64_t boundary_component{};
    std::string relation;
    double energy_MeV{};
    double threshold{};
    double live_boundary_cumulative{};
    double gpu_boundary_cumulative{};
    std::uint64_t boundary_ulp_distance{};
    double live_total{};
    double gpu_total{};
    std::int32_t expected_process{};
    std::uint64_t expected_component{};
    std::int32_t gpu_process{};
    std::uint64_t gpu_component{};
    NativeQueryStatus gpu_status{NativeQueryStatus::InvalidView};
  };

  struct BoundaryProbeSummary {
    std::size_t columns_considered{};
    std::size_t positive_columns{};
    std::size_t boundary_cases{};
    std::size_t mismatches{};
    std::size_t nextafter_minus_mismatches{};
    std::size_t exact_mismatches{};
    std::size_t nextafter_plus_mismatches{};
    std::size_t mismatches_over_16_ulp{};
    std::size_t status_failures{};
    double maximum_total_relative{};
    std::uint64_t maximum_boundary_ulp_distance{};
    std::optional<BoundaryDivergence> first;
    std::optional<BoundaryDivergence> first_over_16_ulp;
  };

  struct BoundaryExpected {
    ProposalNativeSelectionQuery query;
    std::int32_t boundary_process{};
    std::uint64_t boundary_component{};
    std::string relation;
    Selection selection;
    double live_total{};
    double live_boundary_cumulative{};
  };

  struct ExpectedScalarQuery {
    ProposalNativeQuery query;
    std::string kind;
    std::size_t sample{};
    double live_value{};
    // Cumulative rates can be arbitrarily close to zero.  Their physically
    // meaningful error is normalized to the complete rate of the same
    // process/component column, not to a vanishing partial integral.
    double comparison_scale{};
    NativeQueryResult exported_host{};
    double live_equation_residual{};
    double exported_host_equation_residual{};
  };

  double relativeDifference(double left, double right) {
    return std::abs(left - right) /
           std::max({std::abs(left), std::abs(right), 1.e-300});
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

  double unitUniform(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    value ^= value >> 31;
    return (static_cast<double>(value >> 11) + .5) * 0x1.0p-53;
  }

  double productionEnergyFloorMeV(std::int32_t pid, double mass_MeV) {
    if (pid == 22) return .5;
    if (pid == 13 || pid == -13) return mass_MeV + 300.;
    if (pid == 11 || pid == -11) return mass_MeV + .5;
    throw std::invalid_argument("unsupported oracle PID");
  }

  void recordLiveRootException(MetricSummary& summary,
                               NativeDndxColumn const& column,
                               std::size_t sample, double energy_MeV,
                               double argument, MathException const& error) {
    ++summary.live_root_exceptions;
    if (!summary.first_live_root_exception)
      summary.first_live_root_exception = LiveOracleException{
          column.pdg_id, column.process_id, column.component_hash, sample,
          energy_MeV, argument, error.what()};
  }

  void observeScalar(MetricSummary& summary,
                     ExpectedScalarQuery const& expected,
                     NativeQueryResult const& gpu) {
    if (expected.exported_host.status == NativeQueryStatus::Success &&
        std::isfinite(expected.exported_host.value)) {
      auto const normalized =
          std::abs(expected.exported_host.value - expected.live_value) /
          std::max(std::abs(expected.comparison_scale), 1.e-300);
      summary.maximum_exported_host_scale_normalized = std::max(
          summary.maximum_exported_host_scale_normalized, normalized);
      if (normalized > 1.e-10 &&
          positiveUlpDistance(expected.exported_host.value,
                              expected.live_value) > 16)
        ++summary.exported_host_outside_1e10;
      if (normalized > 1.e-11 &&
          positiveUlpDistance(expected.exported_host.value,
                              expected.live_value) > 16)
        ++summary.exported_host_outside_1e11;
    }
    if ((gpu.status == NativeQueryStatus::RootNotConverged ||
         gpu.status == NativeQueryStatus::UnsupportedKinematics) &&
        expected.query.kind == ProposalNativeQueryKind::LossFraction) {
      // This is the specified selected-loss CPU continuation.  It keeps the
      // already selected process/component/quantile and therefore is not a
      // physics mismatch, but it is reported because no device v exists to
      // compare at this point.
      ++summary.device_fallbacks;
      return;
    }
    if (gpu.status != NativeQueryStatus::Success ||
        !std::isfinite(gpu.value)) {
      ++summary.status_failures;
      if (!summary.first)
        summary.first = ScalarDivergence{
            expected.kind, expected.query.pdg_id,
            expected.query.process_id, expected.query.component_hash,
            expected.sample, expected.query.energy_MeV,
            expected.query.argument, expected.live_value,
            expected.exported_host.value, gpu.value,
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            expected.live_equation_residual,
            expected.exported_host_equation_residual, gpu.status,
            expected.exported_host.status};
      return;
    }
    ++summary.compared;
    auto const relative = relativeDifference(gpu.value, expected.live_value);
    auto const normalized =
        std::abs(gpu.value - expected.live_value) /
        std::max(std::abs(expected.comparison_scale), 1.e-300);
    summary.maximum_relative = std::max(summary.maximum_relative, relative);
    summary.maximum_scale_normalized =
        std::max(summary.maximum_scale_normalized, normalized);
    // 16 ULP is retained as an exact-arithmetic escape hatch at values where
    // a relative error is not meaningful.  Otherwise this oracle deliberately
    // applies the requested 1e-10 threshold to the live scalar value (or the
    // full column rate for a cumulative-rate query).
    auto const accepted = normalized <= 1.e-10 ||
                          positiveUlpDistance(gpu.value,
                                              expected.live_value) <= 16;
    if (!accepted) ++summary.outside_1e10;
    auto const accepted_1e11 =
        normalized <= 1.e-11 ||
        positiveUlpDistance(gpu.value, expected.live_value) <= 16;
    if (!accepted_1e11) {
      ++summary.outside_1e11;
      if (!summary.first)
        summary.first = ScalarDivergence{
            expected.kind, expected.query.pdg_id,
            expected.query.process_id, expected.query.component_hash,
            expected.sample, expected.query.energy_MeV,
            expected.query.argument, expected.live_value,
            expected.exported_host.value, gpu.value, relative, normalized,
            expected.live_equation_residual,
            expected.exported_host_equation_residual, gpu.status,
            expected.exported_host.status};
    }
  }

  std::vector<EmParticleState> makeParticles(
      std::int32_t pid, double mass_MeV, std::size_t count,
      std::uint64_t first_history, std::size_t first_sample) {
    auto const low = std::max(
        productionEnergyFloorMeV(pid, mass_MeV),
        pid == 22 ? 2. : (pid == 11 || pid == -11 ? 1.1 : 0.));
    auto const log_low = std::log(low);
    auto const log_high = std::log(1.e8);
    std::vector<EmParticleState> result;
    result.reserve(count);
    for (std::size_t local = 0; local < count; ++local) {
      auto const index = first_sample + local;
      auto const coordinate = std::fmod(
          (static_cast<double>(index) + .5) * .6180339887498948482, 1.);
      EmParticleState particle{};
      particle.pid = pid;
      particle.energy_GeV =
          std::exp(log_low + coordinate * (log_high - log_low)) * 1.e-3;
      particle.direction[2] = -1.;
      particle.weight = 1.;
      particle.history_id = first_history + local;
      particle.step_id = index % 19;
      result.push_back(particle);
    }
    return result;
  }

  std::vector<EmInteractionRecord> selectOnDevice(
      CudaEmBackend& backend, std::int32_t pid,
      std::vector<EmParticleState> const& particles,
      std::size_t& fallback_count) {
    auto selected = backend.selectInteractionsForValidation(particles);
    fallback_count = selected.fallback_events.size();
    auto const appendSelectedFallbacks = [](
        std::vector<EmInteractionRecord>& interactions,
        std::vector<ProposalFallbackEvent> const& fallbacks) {
      for (auto const& event : fallbacks) {
        if (event.process_id == 0 || event.component_hash == 0 ||
            !std::isfinite(event.selection_uniform))
          continue;
        EmInteractionRecord record{};
        record.particle = event.particle;
        record.input_index = event.input_index;
        record.process_id = event.process_id;
        record.status = EmInteractionStatus::Selected;
        record.component_hash = event.component_hash;
        record.process_uniform = event.outer_acceptance_uniform;
        record.process_random_process_id =
            event.outer_acceptance_random_process_id;
        record.process_draw_id = event.outer_acceptance_draw_id;
        record.proposal_selection_uniform = event.selection_uniform;
        record.proposal_selection_random_process_id =
            event.random_process_id;
        record.proposal_selection_draw_id = event.random_draw_id;
        record.loss_quantile =
            std::numeric_limits<double>::quiet_NaN();
        record.energy_fraction =
            std::numeric_limits<double>::quiet_NaN();
        interactions.push_back(record);
      }
    };
    if (pid == 22) {
      auto interactions = std::move(selected.interactions);
      appendSelectedFallbacks(interactions, selected.fallback_events);
      return interactions;
    }
    std::vector<EmInteractionRecord> vertices;
    for (auto record : selected.interactions) {
      if (record.status != EmInteractionStatus::DistanceSampled) continue;
      record.status = EmInteractionStatus::RequiresReselection;
      record.vertex_total_rate_cm2_per_g = record.total_rate_cm2_per_g;
      vertices.push_back(record);
    }
    auto at_vertex =
        backend.reselectLeptonInteractionsAtVertexForValidation(vertices);
    fallback_count += at_vertex.fallback_events.size();
    auto interactions = std::move(at_vertex.interactions);
    // A selected-loss fallback occurs after the device selector has already
    // fixed process/component and its uniform.  Preserve that valid selector
    // decision in this selector-only oracle, while marking loss_quantile NaN
    // so it is not mistaken for a completed device inverse-CDF result.
    appendSelectedFallbacks(interactions, at_vertex.fallback_events);
    return interactions;
  }

  Selection selectHostNative(ProposalNativeDeviceView const& host,
                             std::int32_t pid, double energy_MeV,
                             double uniform) {
    FlatRateTableView view{};
    view.proposal_native = host;
    view.physics_source = 1u;
    auto const selected = selectRateColumnByUniform(
        view, pid, energy_MeV, uniform);
    if (selected.status != TableLookupStatus::Success || !selected.selected)
      return {};
    auto const loss = queryProposalNativeLossFraction(
        host, pid, selected.process_id, selected.component_hash,
        energy_MeV, selected.residual_quantile);
    return {selected.process_id, selected.component_hash,
            selected.residual_quantile,
            loss.status == NativeQueryStatus::Success
                ? loss.value
                : std::numeric_limits<double>::quiet_NaN(),
            selected.total_rate};
  }

  Selection selectLiveProposal(CalculatorOwner const& owner,
                               double energy_MeV, double uniform) {
    // This is intentionally the unmodified live Rates vector.  Reordering it
    // to match an exported table would mask precisely the scalar insertion-
    // order defect this oracle is meant to detect.
    auto const rates = owner.interaction->Rates(energy_MeV);
    auto const total = std::accumulate(
        rates.begin(), rates.end(), 0.,
        [](double sum, PROPOSAL::Interaction::Rate const& rate) {
          return sum + rate.rate;
        });
    if (!(total > 0.)) return {};
    auto sampled_rate = uniform * total;
    double residual = 0.;
    for (auto const& rate : rates) {
      sampled_rate -= rate.rate;
      if (sampled_rate < 0.) {
        residual = (-sampled_rate) / rate.rate;
        break;
      }
    }
    auto const loss = owner.interaction->SampleLoss(
        energy_MeV, rates, uniform);
    return {static_cast<std::int32_t>(loss.type),
            static_cast<std::uint64_t>(loss.comp_hash), residual,
            loss.v_loss, total};
  }

  Selection selectLiveProposalByThreshold(
      std::vector<PROPOSAL::Interaction::Rate> const& rates,
      double threshold) {
    auto sampled_rate = threshold;
    auto const total = std::accumulate(
        rates.begin(), rates.end(), 0.,
        [](double sum, PROPOSAL::Interaction::Rate const& rate) {
          return sum + rate.rate;
        });
    for (auto const& rate : rates) {
      sampled_rate -= rate.rate;
      if (sampled_rate < 0.)
        return {static_cast<std::int32_t>(
                    rate.crosssection->GetInteractionType()),
                static_cast<std::uint64_t>(rate.comp_hash),
                (-sampled_rate) / rate.rate, 0., total};
    }
    return {};
  }

  Summary compare(CudaEmBackend& backend,
                  ProposalNativeDeviceView const& host,
                  CalculatorOwner const& owner, std::size_t samples,
                  std::size_t chunk_size, std::uint64_t first_history) {
    Summary summary;
    summary.pid = static_cast<std::int32_t>(get_PDG(owner.code));
    summary.requested = samples;
    for (std::size_t begin = 0; begin < samples; begin += chunk_size) {
      auto const count = std::min(chunk_size, samples - begin);
      auto const particles = makeParticles(
          summary.pid, owner.mass_MeV, count, first_history + begin, begin);
      std::size_t chunk_fallbacks{};
      auto records = selectOnDevice(
          backend, summary.pid, particles, chunk_fallbacks);
      summary.fallbacks += chunk_fallbacks;
      std::sort(records.begin(), records.end(), [](auto const& left,
                                                    auto const& right) {
        return left.input_index < right.input_index;
      });
      for (auto const& gpu : records) {
      if (gpu.status != EmInteractionStatus::Selected) continue;
      ++summary.compared;
      auto const sample = begin + gpu.input_index;
      auto const energy_MeV = gpu.particle.energy_GeV * 1000.;
      auto const expected = selectHostNative(
          host, summary.pid, energy_MeV,
          gpu.proposal_selection_uniform);
      auto const live = selectLiveProposal(
          owner, energy_MeV, gpu.proposal_selection_uniform);
      auto divergence = FirstDivergence{
          {}, summary.pid, sample, energy_MeV,
          gpu.proposal_selection_uniform, gpu.loss_quantile,
          expected.process, gpu.process_id, expected.component,
          gpu.component_hash, 0., gpu.energy_fraction, 0.};
      divergence.live_loss_uniform = live.residual_quantile;
      if (gpu.process_random_process_id !=
              InteractionColumnRandomProcessId ||
          gpu.proposal_selection_random_process_id !=
              ProposalSelectionRandomProcessId ||
          gpu.process_uniform == gpu.proposal_selection_uniform) {
        divergence.kind = "outer_inner_rng_provenance";
      }
      if (expected.process != gpu.process_id ||
          expected.component != gpu.component_hash) {
        ++summary.native_process_mismatches;
        divergence.kind = "native_process_or_component";
      }
      if (live.process != gpu.process_id ||
          live.component != gpu.component_hash)
        ++summary.live_proposal_sample_loss_mismatches;
      auto const* selected_column = native_detail::findDndx(
          host, summary.pid, gpu.process_id, gpu.component_hash);
      // A residual quantile is physically observable only for a cumulative
      // dN/dX column that must be inverted.  PROPOSAL's fixed-loss processes
      // (photoelectric/annihilation and full-loss photon pair semantics) do
      // not consume that residual to obtain v, so comparing it would reject
      // two identical v=1 decisions for an irrelevant bookkeeping value.
      if (selected_column &&
          selected_column->rate_model == NativeRateModel::BicubicSpline &&
          std::isfinite(gpu.loss_quantile) &&
          relativeDifference(gpu.loss_quantile,
                             live.residual_quantile) > 2.e-12) {
        ++summary.residual_quantile_outside_tolerance;
        if (divergence.kind.empty())
          divergence.kind = "live_proposal_residual_quantile";
      }

      auto const loss = std::isfinite(gpu.loss_quantile)
          ? queryProposalNativeLossFraction(
                host, summary.pid, gpu.process_id, gpu.component_hash,
                energy_MeV, gpu.loss_quantile)
          : NativeQueryResult{NativeQueryStatus::UnsupportedKinematics,
                              0, 0, 0.};
      if (std::isfinite(gpu.loss_quantile) &&
          loss.status == NativeQueryStatus::Success) {
        ++summary.supported_loss_compared;
        divergence.expected_v = live.v_loss;
        divergence.relative_v = relativeDifference(
            live.v_loss, gpu.energy_fraction);
        summary.maximum_loss_relative = std::max(
            summary.maximum_loss_relative, divergence.relative_v);
        auto const v_ulp = positiveUlpDistance(
            live.v_loss, gpu.energy_fraction);
        if (divergence.relative_v > InteractionOracleTolerance &&
            v_ulp > InteractionOracleUlpTolerance) {
          ++summary.loss_outside_tolerance;
          auto loss_divergence = divergence;
          loss_divergence.kind = "live_proposal_sample_loss_v";
          if (!summary.first_loss_divergence)
            summary.first_loss_divergence = loss_divergence;
          if (summary.loss_divergences.size() < 64)
            summary.loss_divergences.push_back(loss_divergence);
          if (divergence.kind.empty())
            divergence.kind = "live_proposal_sample_loss_v";
        }
      }
      if (!divergence.kind.empty() && !summary.first)
        summary.first = divergence;
      }
    }
    return summary;
  }

  BoundaryProbeSummary compareRateBoundariesToLiveProposal(
      CudaProposalNativeTable const& device,
      ProposalNativeDeviceView const& host,
      ProposalNativeTableSet const& table,
      std::vector<CalculatorOwner const*> const& owners) {
    (void)host;
    BoundaryProbeSummary summary;
    std::vector<BoundaryExpected> expected;
    expected.reserve(table.dndx_columns.size() * 3);
    auto const ownerForPdg = [&](std::int32_t pdg)
        -> CalculatorOwner const& {
      auto const found = std::find_if(
          owners.begin(), owners.end(), [&](auto const* owner) {
            return static_cast<std::int32_t>(get_PDG(owner->code)) == pdg;
          });
      if (found == owners.end())
        throw std::runtime_error(
            "boundary oracle has no calculator for native PID");
      return **found;
    };
    constexpr double coordinates[]{.5, .25, .75, .125, .875,
                                   .01, .99, .001, .999};
    for (std::size_t column_index = 0;
         column_index < table.dndx_columns.size(); ++column_index) {
      ++summary.columns_considered;
      auto const& column = table.dndx_columns[column_index];
      auto const& owner = ownerForPdg(column.pdg_id);
      auto energy_low = std::max(
          column.lower_energy_limit_MeV,
          productionEnergyFloorMeV(column.pdg_id,
                                   column.particle_mass_MeV));
      auto energy_high = 1.e14;
      if (column.rate_model == NativeRateModel::BicubicSpline) {
        energy_low = std::max(energy_low, column.spline.energy_axis.low);
        energy_high = column.spline.energy_axis.high;
      }
      energy_low = std::nextafter(
          std::max(energy_low, 1.e-12),
          std::numeric_limits<double>::infinity());
      energy_high = std::nextafter(energy_high, 0.);
      if (!(energy_high > energy_low)) continue;
      auto const log_low = std::log(energy_low);
      auto const log_high = std::log(energy_high);
      bool found_positive = false;
      for (auto const coordinate : coordinates) {
        auto const energy =
            std::exp(log_low + coordinate * (log_high - log_low));
        auto const rates = owner.interaction->Rates(energy);
        auto const target = std::find_if(
            rates.begin(), rates.end(), [&](auto const& rate) {
              return rate.crosssection &&
                     static_cast<std::int32_t>(
                         rate.crosssection->GetInteractionType()) ==
                         column.process_id &&
                     static_cast<std::uint64_t>(rate.comp_hash) ==
                         column.component_hash;
            });
        if (target == rates.end() || !(target->rate > 0.) ||
            !std::isfinite(target->rate))
          continue;
        auto const target_offset = static_cast<std::size_t>(
            std::distance(rates.begin(), target));
        auto const add_rate =
            [](double sum, PROPOSAL::Interaction::Rate const& rate) {
              return sum + rate.rate;
            };
        auto const total = std::accumulate(
            rates.begin(), rates.end(), 0., add_rate);
        auto const boundary = std::accumulate(
            rates.begin(), rates.begin() + target_offset + 1, 0., add_rate);
        if (!(total > 0.) || !std::isfinite(total) ||
            !(boundary > 0.) || !std::isfinite(boundary))
          continue;
        found_positive = true;
        ++summary.positive_columns;
        struct Case {
          char const* relation;
          double threshold;
        };
        Case const cases[]{
            {"nextafter_minus",
             std::nextafter(boundary,
                            -std::numeric_limits<double>::infinity())},
            {"exact", boundary},
            {"nextafter_plus",
             std::nextafter(boundary,
                            std::numeric_limits<double>::infinity())}};
        for (auto const& value : cases) {
          expected.push_back({
              {column.pdg_id, energy, value.threshold,
               column.process_id, column.component_hash},
              column.process_id, column.component_hash, value.relation,
              selectLiveProposalByThreshold(rates, value.threshold),
              total, boundary});
        }
        break;
      }
      (void)found_positive;
    }
    std::vector<ProposalNativeSelectionQuery> queries;
    queries.reserve(expected.size());
    for (auto const& value : expected) queries.push_back(value.query);
    auto const gpu = device.selectForValidation(queries);
    if (gpu.size() != expected.size())
      throw std::runtime_error(
          "native CUDA boundary result count changed");
    summary.boundary_cases = expected.size();
    for (std::size_t index = 0; index < expected.size(); ++index) {
      auto const& reference = expected[index];
      auto const& result = gpu[index];
      auto const total_relative = relativeDifference(
          result.total_rate, reference.live_total);
      summary.maximum_total_relative =
          std::max(summary.maximum_total_relative, total_relative);
      auto const status_failure =
          result.status != NativeQueryStatus::Success;
      if (status_failure) ++summary.status_failures;
      auto const gpu_process = result.selected ? result.process_id : 0;
      auto const gpu_component = result.selected ? result.component_hash : 0;
      auto const mismatch = status_failure ||
          gpu_process != reference.selection.process ||
          gpu_component != reference.selection.component;
      auto const boundary_ulp = positiveUlpDistance(
          result.boundary_cumulative_rate,
          reference.live_boundary_cumulative);
      summary.maximum_boundary_ulp_distance = std::max(
          summary.maximum_boundary_ulp_distance, boundary_ulp);
      if (!mismatch) continue;
      ++summary.mismatches;
      if (reference.relation == "nextafter_minus")
        ++summary.nextafter_minus_mismatches;
      else if (reference.relation == "exact")
        ++summary.exact_mismatches;
      else if (reference.relation == "nextafter_plus")
        ++summary.nextafter_plus_mismatches;
      auto const divergence = BoundaryDivergence{
            reference.query.pdg_id, reference.boundary_process,
            reference.boundary_component, reference.relation,
            reference.query.energy_MeV, reference.query.threshold,
            reference.live_boundary_cumulative,
            result.boundary_cumulative_rate, boundary_ulp,
            reference.live_total, result.total_rate,
            reference.selection.process, reference.selection.component,
            gpu_process, gpu_component, result.status};
      if (!summary.first) summary.first = divergence;
      if (boundary_ulp > 16) {
        ++summary.mismatches_over_16_ulp;
        if (!summary.first_over_16_ulp)
          summary.first_over_16_ulp = divergence;
      }
    }
    return summary;
  }

  PROPOSAL::CrossSectionBase* findLiveCrossSection(
      CalculatorOwner const& owner, NativeDndxColumn const& column) {
    auto const exact = std::find_if(
        owner.cross_sections.begin(), owner.cross_sections.end(),
        [&](auto const& cross) {
          return cross &&
                 static_cast<std::int32_t>(cross->GetInteractionType()) ==
                     column.process_id &&
                 static_cast<std::uint64_t>(cross->GetHash()) ==
                     column.cross_section_hash;
        });
    if (exact != owner.cross_sections.end()) return exact->get();
    // A missing hash match would make the oracle ambiguous.  Do not silently
    // select a calculator by process name alone: that could compare a native
    // column against a different parametrization of the same interaction.
    throw std::runtime_error(
        "cannot match native column to live PROPOSAL cross section");
  }

  void writeProgressMetricCsv(std::ostream& output,
                              MetricSummary const& metric) {
    output << '"' << metric.live_oracle_attempts << ';'
           << metric.live_calculate_stochastic_loss_calls << ';'
           << metric.compared << ';' << metric.device_fallbacks << ';'
           << metric.live_oracle_failures << ';'
           << metric.live_root_exceptions << ';' << metric.status_failures
           << ';' << metric.outside_1e10 << ';'
           << metric.exported_host_outside_1e10 << ';'
           << metric.outside_1e11 << ';'
           << metric.exported_host_outside_1e11 << ';'
           << metric.maximum_scale_normalized << ';'
           << metric.maximum_exported_host_scale_normalized << ';';
    if (metric.first)
      output << metric.first->sample;
    output << ';';
    if (metric.first_live_root_exception)
      output << metric.first_live_root_exception->sample;
    output << '"';
  }

  ColumnOracleSummary compareColumnToLiveProposal(
      CudaProposalNativeTable const& device,
      ProposalNativeDeviceView const& host,
      NativeDndxColumn const& column, CalculatorOwner const& owner,
      std::size_t samples, std::size_t chunk_size,
      std::uint64_t sequence_key, std::ostream* progress) {
    ColumnOracleSummary summary;
    summary.pid = column.pdg_id;
    summary.process = column.process_id;
    summary.component = column.component_hash;
    auto* cross = findLiveCrossSection(owner, column);

    auto energy_low = std::max(
        column.lower_energy_limit_MeV,
        productionEnergyFloorMeV(column.pdg_id,
                                 column.particle_mass_MeV));
    auto energy_high = 1.e14;
    if (column.rate_model == NativeRateModel::BicubicSpline) {
      energy_low = std::max(energy_low, column.spline.energy_axis.low);
      energy_high = column.spline.energy_axis.high;
    }
    if (!(energy_low > 0.)) energy_low = .4;
    energy_low = std::nextafter(
        energy_low, std::numeric_limits<double>::infinity());
    energy_high = std::nextafter(energy_high, 0.);
    if (!(energy_high > energy_low))
      throw std::runtime_error(
          "native column has no positive oracle energy interval");
    auto const log_low = std::log(energy_low);
    auto const log_high = std::log(energy_high);

    for (std::size_t begin = 0; begin < samples; begin += chunk_size) {
      auto const count = std::min(chunk_size, samples - begin);
      std::vector<ExpectedScalarQuery> expected;
      expected.reserve(count * 3);
      for (std::size_t local = 0; local < count; ++local) {
      auto const sample = begin + local;
      auto const key = sequence_key ^ static_cast<std::uint64_t>(sample);
      auto const energy = std::exp(
          log_low + unitUniform(key) * (log_high - log_low));
      double live_rate{};
      ++summary.rate.live_oracle_attempts;
      try {
        live_rate = cross->CalculatedNdx(
            energy, static_cast<std::size_t>(column.component_hash));
      } catch (std::exception const&) {
        ++summary.rate.live_oracle_failures;
        continue;
      }
      if (!std::isfinite(live_rate) || live_rate < 0.) {
        ++summary.rate.live_oracle_failures;
        continue;
      }
      expected.push_back({
          {ProposalNativeQueryKind::Rate, column.pdg_id,
           column.process_id, column.component_hash, energy, 0.},
          "rate", sample, live_rate, live_rate,
          queryProposalNativeRate(
              host, column.pdg_id, column.process_id,
              column.component_hash, energy)});

      if (column.rate_model == NativeRateModel::BicubicSpline) {
        auto const transformed_loss =
            unitUniform(key ^ 0xd1b54a32d192ed03ull);
        ++summary.cumulative_rate.live_oracle_attempts;
        try {
          auto const live_cumulative = cross->EvaluateDndxInterpolation(
              static_cast<std::size_t>(column.component_hash), energy,
              transformed_loss);
          if (!std::isfinite(live_cumulative))
            throw std::runtime_error("non-finite live cumulative rate");
          expected.push_back({
              {ProposalNativeQueryKind::CumulativeRate, column.pdg_id,
               column.process_id, column.component_hash, energy,
               transformed_loss},
            "cumulative_rate", sample, live_cumulative,
            std::max(std::abs(live_rate),
                     std::abs(live_cumulative)),
            queryProposalNativeCumulativeRate(
                host, column, energy, transformed_loss)});
        } catch (std::exception const&) {
          ++summary.cumulative_rate.live_oracle_failures;
        }
      }

      if (!(live_rate > 0.)) continue;
      // Exercise the complete production device-inverse domain.  The two
      // narrower endpoint bands are deliberately replayed by the scalar
      // SampleLoss path, so they are covered by the selector/fallback oracle
      // above rather than being misclassified as device inverse failures.
      auto const endpoint = NativeSelectionReplayEndpointWidth;
      auto const quantile =
          endpoint + (1. - 2. * endpoint) *
                         unitUniform(key ^ 0x94d049bb133111ebull);
      ++summary.inverse_loss.live_oracle_attempts;
      if (column.kinematic_model != NativeKinematicModel::OnlyStochastic)
        ++summary.inverse_loss.live_calculate_stochastic_loss_calls;
      try {
        // Direct only-stochastic processes have no dN/dX interpolant object
        // on which CalculateStochasticLoss() can be called.  Their scalar
        // CrossSection implementation nevertheless defines the stochastic
        // loss exactly as one; use that explicit PROPOSAL semantic here.
        auto const live_loss =
            column.kinematic_model == NativeKinematicModel::OnlyStochastic
                ? 1.
                : cross->CalculateStochasticLoss(
                      static_cast<std::size_t>(column.component_hash),
                      energy, quantile * live_rate);
        if (!std::isfinite(live_loss))
          throw std::runtime_error("non-finite live stochastic loss");
        auto const exported_host_loss = queryProposalNativeLossFraction(
            host, column.pdg_id, column.process_id,
            column.component_hash, energy, quantile);
        double live_residual = 0.;
        double exported_host_residual = 0.;
        if (column.rate_model == NativeRateModel::BicubicSpline &&
            exported_host_loss.status == NativeQueryStatus::Success) {
          auto residual = [&](double physical_loss) {
            auto const transformed = native_detail::transformedLoss(
                host, column, energy, physical_loss);
            if (!std::isfinite(transformed))
              return std::numeric_limits<double>::quiet_NaN();
            auto const cumulative = cross->EvaluateDndxInterpolation(
                static_cast<std::size_t>(column.component_hash), energy,
                transformed);
            return (cumulative - quantile * live_rate) /
                   std::max(live_rate, 1.e-300);
          };
          live_residual = residual(live_loss);
          exported_host_residual = residual(exported_host_loss.value);
        }
        expected.push_back({
            {ProposalNativeQueryKind::LossFraction, column.pdg_id,
             column.process_id, column.component_hash, energy, quantile},
            "inverse_loss", sample, live_loss, live_loss,
            exported_host_loss, live_residual, exported_host_residual});
      } catch (MathException const& error) {
        recordLiveRootException(summary.inverse_loss, column, sample,
                                energy, quantile, error);
      } catch (std::exception const&) {
        ++summary.inverse_loss.live_oracle_failures;
      }
      }

      std::vector<ProposalNativeQuery> queries;
      queries.reserve(expected.size());
      for (auto const& value : expected) queries.push_back(value.query);
      auto const gpu = device.queryForValidation(queries);
      if (gpu.size() != expected.size())
        throw std::runtime_error("native CUDA oracle result count changed");
      for (std::size_t index = 0; index < expected.size(); ++index) {
        switch (expected[index].query.kind) {
        case ProposalNativeQueryKind::Rate:
          observeScalar(summary.rate, expected[index], gpu[index]);
          break;
        case ProposalNativeQueryKind::CumulativeRate:
          observeScalar(summary.cumulative_rate, expected[index], gpu[index]);
          break;
        case ProposalNativeQueryKind::LossFraction:
          observeScalar(summary.inverse_loss, expected[index], gpu[index]);
          break;
        default:
          throw std::logic_error("unexpected column oracle query kind");
        }
      }
      if (progress) {
        *progress << "column," << column.pdg_id << ','
                  << column.process_id << ',' << column.component_hash
                  << ',' << begin + count << ',';
        writeProgressMetricCsv(*progress, summary.rate);
        *progress << ',';
        writeProgressMetricCsv(*progress, summary.cumulative_rate);
        *progress << ',';
        writeProgressMetricCsv(*progress, summary.inverse_loss);
        *progress << '\n';
        progress->flush();
      }
    }
    return summary;
  }

  ContinuousOracleSummary compareContinuousToLiveProposal(
      CudaProposalNativeTable const& device,
      ProposalNativeDeviceView const& host,
      ProposalNativeTableSet const& table, CalculatorOwner const& owner,
      std::size_t samples, std::size_t chunk_size,
      std::uint64_t sequence_key, std::ostream* progress) {
    ContinuousOracleSummary summary;
    summary.pid = static_cast<std::int32_t>(get_PDG(owner.code));
    auto const utility = std::find_if(
        table.utility_columns.begin(), table.utility_columns.end(),
        [&](auto const& column) { return column.pdg_id == summary.pid; });
    if (utility == table.utility_columns.end() || !owner.displacement)
      throw std::runtime_error(
          "continuous oracle has no live displacement calculator");

    auto energy_low = std::max(
        {utility->lower_energy_limit_MeV, utility->spline.axis.low,
         productionEnergyFloorMeV(summary.pid,
                                  utility->particle_mass_MeV)});
    auto energy_high = utility->spline.axis.high;
    for (auto const& column : table.dedx_columns) {
      if (column.pdg_id != summary.pid) continue;
      energy_high = std::min(energy_high, column.spline.axis.high);
    }
    energy_low = std::nextafter(
        energy_low, std::numeric_limits<double>::infinity());
    energy_high = std::nextafter(energy_high, 0.);
    if (!(energy_high > energy_low))
      throw std::runtime_error(
          "continuous oracle has no common positive energy interval");
    auto const log_low = std::log(energy_low);
    auto const log_high = std::log(energy_high);

    for (std::size_t begin = 0; begin < samples; begin += chunk_size) {
      auto const count = std::min(chunk_size, samples - begin);
      std::vector<ExpectedScalarQuery> expected;
      expected.reserve(count * 3);
      for (std::size_t local = 0; local < count; ++local) {
      auto const sample = begin + local;
      auto const key = sequence_key ^ static_cast<std::uint64_t>(sample);
      auto const energy = std::exp(
          log_low + unitUniform(key) * (log_high - log_low));
      ++summary.dedx.live_oracle_attempts;
      try {
        double live_dedx = 0.;
        for (auto const& cross : owner.cross_sections)
          live_dedx += cross->CalculatedEdx(energy);
        if (!std::isfinite(live_dedx) || !(live_dedx > 0.))
          throw std::runtime_error("invalid live dE/dX");
        expected.push_back({
            {ProposalNativeQueryKind::ContinuousDedx, summary.pid, 0, 0,
             energy, 0.},
            "continuous_dedx", sample, live_dedx, live_dedx,
            queryProposalNativeDedx(host, summary.pid, energy)});
      } catch (std::exception const&) {
        ++summary.dedx.live_oracle_failures;
      }

      double live_range{};
      ++summary.range.live_oracle_attempts;
      try {
        live_range = owner.displacement->SolveTrackIntegral(
            energy, utility->lower_energy_limit_MeV);
        if (!std::isfinite(live_range) || !(live_range > 0.))
          throw std::runtime_error("invalid live range");
        expected.push_back({
            {ProposalNativeQueryKind::ContinuousRange, summary.pid, 0, 0,
             energy, 0.},
            "continuous_range", sample, live_range, live_range,
            queryProposalNativeRange(host, summary.pid, energy)});
      } catch (std::exception const&) {
        ++summary.range.live_oracle_failures;
        continue;
      }

      ++summary.inverse_range.live_oracle_attempts;
      try {
        auto const fraction =
            .05 + .9 * unitUniform(key ^ 0xbf58476d1ce4e5b9ull);
        auto const target_range = fraction * live_range;
        auto const grammage = live_range - target_range;
        auto const live_energy =
            owner.displacement->UpperLimitTrackIntegral(
                energy, grammage);
        if (!std::isfinite(live_energy) || !(live_energy > 0.))
          throw std::runtime_error("invalid live inverse range");
        expected.push_back({
            {ProposalNativeQueryKind::ContinuousEnergyAfterLoss,
             summary.pid, 0, 0, energy, grammage},
            "energy_after_continuous_loss", sample, live_energy,
            live_energy,
            queryProposalNativeEnergyAfterContinuousLoss(
                host, summary.pid, energy, grammage)});
      } catch (std::exception const&) {
        ++summary.inverse_range.live_oracle_failures;
      }
      }

      std::vector<ProposalNativeQuery> queries;
      queries.reserve(expected.size());
      for (auto const& value : expected) queries.push_back(value.query);
      auto const gpu = device.queryForValidation(queries);
      if (gpu.size() != expected.size())
        throw std::runtime_error(
            "continuous CUDA oracle result count changed");
      for (std::size_t index = 0; index < expected.size(); ++index) {
        switch (expected[index].query.kind) {
        case ProposalNativeQueryKind::ContinuousDedx:
          observeScalar(summary.dedx, expected[index], gpu[index]);
          break;
        case ProposalNativeQueryKind::ContinuousRange:
          observeScalar(summary.range, expected[index], gpu[index]);
          break;
        case ProposalNativeQueryKind::ContinuousEnergy:
        case ProposalNativeQueryKind::ContinuousEnergyAfterLoss:
          observeScalar(summary.inverse_range, expected[index], gpu[index]);
          break;
        default:
          throw std::logic_error("unexpected continuous oracle query kind");
        }
      }
      if (progress) {
        *progress << "continuous," << summary.pid << ",0,0,"
                  << begin + count << ',';
        writeProgressMetricCsv(*progress, summary.dedx);
        *progress << ',';
        writeProgressMetricCsv(*progress, summary.range);
        *progress << ',';
        writeProgressMetricCsv(*progress, summary.inverse_range);
        *progress << '\n';
        progress->flush();
      }
    }
    return summary;
  }

  void writeJsonString(std::ostream& output, std::string const& value) {
    output << '"';
    for (auto const character : value) {
      switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default: output << character; break;
      }
    }
    output << '"';
  }

  void writeLiveOracleException(
      std::ostream& output,
      std::optional<LiveOracleException> const& first) {
    if (!first) {
      output << "null";
      return;
    }
    auto const& error = *first;
    output << "{\"pid\":" << error.pid << ",\"process\":"
           << error.process << ",\"component\":" << error.component
           << ",\"sample\":" << error.sample << ",\"energy_MeV\":"
           << error.energy_MeV << ",\"argument\":" << error.argument
           << ",\"message\":";
    writeJsonString(output, error.message);
    output << "}";
  }

  void writeDivergence(std::ostream& output,
                       std::optional<FirstDivergence> const& first) {
    if (!first) {
      output << "null";
      return;
    }
    auto const& d = *first;
    output << "{\"kind\":\"" << d.kind << "\",\"pid\":" << d.pid
           << ",\"sample\":" << d.sample << ",\"energy_MeV\":"
           << d.energy_MeV << ",\"process_uniform\":"
           << d.process_uniform << ",\"loss_uniform\":"
           << d.loss_uniform << ",\"live_loss_uniform\":"
           << d.live_loss_uniform << ",\"expected_process\":"
           << d.expected_process << ",\"gpu_process\":" << d.gpu_process
           << ",\"expected_component\":" << d.expected_component
           << ",\"gpu_component\":" << d.gpu_component
           << ",\"expected_v\":" << d.expected_v << ",\"gpu_v\":"
           << d.gpu_v << ",\"relative_v\":" << d.relative_v << "}";
  }

  void writeBoundaryDivergence(
      std::ostream& output,
      std::optional<BoundaryDivergence> const& first) {
    if (!first) {
      output << "null";
      return;
    }
    auto const& d = *first;
    output << "{\"pid\":" << d.pid
           << ",\"boundary_process\":" << d.boundary_process
           << ",\"boundary_component\":" << d.boundary_component
           << ",\"relation\":";
    writeJsonString(output, d.relation);
    output << ",\"energy_MeV\":" << d.energy_MeV
           << ",\"threshold\":" << d.threshold
           << ",\"live_boundary_cumulative\":"
           << d.live_boundary_cumulative
           << ",\"gpu_boundary_cumulative\":"
           << d.gpu_boundary_cumulative
           << ",\"boundary_ulp_distance\":"
           << d.boundary_ulp_distance
           << ",\"live_total\":" << d.live_total
           << ",\"gpu_total\":" << d.gpu_total
           << ",\"expected_process\":" << d.expected_process
           << ",\"expected_component\":" << d.expected_component
           << ",\"gpu_process\":" << d.gpu_process
           << ",\"gpu_component\":" << d.gpu_component
           << ",\"gpu_status\":"
           << static_cast<std::uint32_t>(d.gpu_status) << "}";
  }

  void writeScalarDivergence(
      std::ostream& output,
      std::optional<ScalarDivergence> const& first) {
    if (!first) {
      output << "null";
      return;
    }
    auto const& d = *first;
    auto const writeNumber = [&](double value) {
      if (std::isfinite(value))
        output << value;
      else
        output << "null";
    };
    output << "{\"kind\":\"" << d.kind << "\",\"pid\":" << d.pid
           << ",\"process\":" << d.process << ",\"component\":"
           << d.component << ",\"sample\":" << d.sample
           << ",\"energy_MeV\":";
    writeNumber(d.energy_MeV);
    output << ",\"argument\":";
    writeNumber(d.argument);
    output << ",\"live_value\":";
    writeNumber(d.live_value);
    output << ",\"exported_host_value\":";
    writeNumber(d.exported_host_value);
    output << ",\"gpu_value\":";
    writeNumber(d.gpu_value);
    output << ",\"relative_difference\":";
    writeNumber(d.relative_difference);
    output << ",\"scale_normalized_difference\":";
    writeNumber(d.scale_normalized_difference);
    output << ",\"live_equation_residual\":";
    writeNumber(d.live_equation_residual);
    output << ",\"exported_host_equation_residual\":";
    writeNumber(d.exported_host_equation_residual);
    output << ",\"gpu_status\":"
           << static_cast<std::uint32_t>(d.gpu_status)
           << ",\"exported_host_status\":"
           << static_cast<std::uint32_t>(d.exported_host_status) << "}";
  }

  void writeMetric(std::ostream& output, MetricSummary const& metric) {
    output << "{\"live_oracle_attempts\":"
           << metric.live_oracle_attempts
           << ",\"live_calculate_stochastic_loss_calls\":"
           << metric.live_calculate_stochastic_loss_calls
           << ",\"compared\":" << metric.compared
           << ",\"device_fallbacks\":" << metric.device_fallbacks
           << ",\"live_oracle_failures\":"
           << metric.live_oracle_failures
           << ",\"live_root_exceptions\":"
           << metric.live_root_exceptions << ",\"status_failures\":"
           << metric.status_failures << ",\"outside_1e10\":"
           << metric.outside_1e10
           << ",\"exported_host_outside_1e10\":"
           << metric.exported_host_outside_1e10
           << ",\"outside_1e11\":" << metric.outside_1e11
           << ",\"exported_host_outside_1e11\":"
           << metric.exported_host_outside_1e11
           << ",\"maximum_relative\":"
           << metric.maximum_relative
           << ",\"maximum_scale_normalized\":"
           << metric.maximum_scale_normalized
           << ",\"maximum_exported_host_scale_normalized\":"
           << metric.maximum_exported_host_scale_normalized
           << ",\"first_divergence\":";
    writeScalarDivergence(output, metric.first);
    output << ",\"first_live_root_exception\":";
    writeLiveOracleException(output, metric.first_live_root_exception);
    output << "}";
  }

  bool metricAccepted(MetricSummary const& metric,
                      double tolerance = 1.e-10) {
    auto const gpu_outside = tolerance <= 1.e-11
                                 ? metric.outside_1e11
                                 : metric.outside_1e10;
    auto const host_outside = tolerance <= 1.e-11
                                  ? metric.exported_host_outside_1e11
                                  : metric.exported_host_outside_1e10;
    return metric.live_oracle_failures == 0 &&
           metric.live_root_exceptions == 0 &&
           metric.status_failures == 0 && gpu_outside == 0 &&
           host_outside == 0;
  }
}

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 7)
      throw std::invalid_argument(
          "usage: gpu_em_proposal_native_decision_oracle OUTPUT_JSON "
          "[ORACLE_SAMPLES_PER_COLUMN] [SEED] [DEVICE] [CHUNK_SIZE] "
          "[DECISION_SAMPLES_PER_PID]");
    auto const output_path = std::filesystem::path(argv[1]);
    auto const samples = argc >= 3 ? std::stoull(argv[2]) : 4096;
    auto const seed = argc >= 4 ? std::stoull(argv[3]) : 2026083001ull;
    auto const device_index = argc >= 5 ? std::stoi(argv[4]) : 0;
    auto const chunk_size = argc >= 6 ? std::stoull(argv[5]) : 65536;
    auto const decision_samples = argc >= 7 ? std::stoull(argv[6]) : samples;
    if (samples == 0 || chunk_size == 0 || decision_samples == 0)
      throw std::invalid_argument(
          "oracle, decision, and chunk counts must be positive");
    PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::critical);

    // The scalar production path resolves the common 0.5 MeV production
    // threshold to the nearest cached PROPOSAL stochastic cut below it.  This
    // is 0.4 MeV for every tracked projectile, including muons.  The 300 MeV
    // muon cut below is a transport termination threshold and must never be
    // used to construct a PROPOSAL rate calculator.
    auto const proposal_stochastic_cut = .4_MeV;
    auto const em_transport_cut = .5_MeV;
    auto const muon_transport_cut = 300_MeV;
    CalculatorOwner photon{Code::Photon, proposal_stochastic_cut};
    CalculatorOwner electron{Code::Electron, proposal_stochastic_cut};
    CalculatorOwner positron{Code::Positron, proposal_stochastic_cut};
    CalculatorOwner muon_minus{Code::MuMinus, proposal_stochastic_cut};
    CalculatorOwner muon_plus{Code::MuPlus, proposal_stochastic_cut};
    std::vector<CalculatorOwner const*> calculator_owners{
        &photon, &electron, &positron, &muon_minus, &muon_plus};
    for (auto const* owner : calculator_owners) {
      if (owner->stochasticCutMeV() !=
          proposal_stochastic_cut / 1_MeV) {
        throw std::logic_error(
            "decision oracle calculators do not share the production "
            "PROPOSAL stochastic cut");
      }
    }
    std::vector<proposal::NativeInteractionCalculatorView> interactions{
        photon.interactionView(), electron.interactionView(),
        positron.interactionView(), muon_minus.interactionView(),
        muon_plus.interactionView()};
    std::vector<proposal::NativeContinuousCalculatorView> continuous{
        electron.continuousView(), positron.continuousView(),
        muon_minus.continuousView(), muon_plus.continuousView()};
    auto const table = exportProposalNativeTables(interactions, continuous);
    auto const aux_directory = std::filesystem::temp_directory_path() /
                               "c8-native-decision-oracle-aux";
    auto const aux = loadOrCreateProposalNativeAux(
        interactions, aux_directory);

    GpuEmConfig config{};
    config.device = device_index;
    config.min_batch_size = 1;
    config.memory_fraction = .05;
    config.table_tolerance = 1.e-3;
    config.em_transport_cut_MeV = em_transport_cut / 1_MeV;
    config.muon_transport_cut_MeV = muon_transport_cut / 1_MeV;
    config.random_seed = seed;
    config.shower_id = 1;
    config.physics_source = GpuPhysicsSource::ProposalNative;
    config.auxiliary_cache_directory = aux_directory;
    CudaEmBackend backend;
    backend.initialize(EnvironmentSnapshot{}, table, aux, config);
    auto const host = makeProposalNativeHostView(table);

    // Use a separate immutable device-table handle so the numerical oracle
    // exercises the real CUDA query kernels directly rather than treating the
    // exported host POD evaluator as the expected GPU result.
    CudaProposalNativeTable native_device;
    native_device.initialize(table, device_index);

    if (!output_path.parent_path().empty())
      std::filesystem::create_directories(output_path.parent_path());
    auto progress_path = output_path;
    progress_path += ".progress.csv";
    std::ofstream progress(progress_path, std::ios::trunc);
    if (!progress)
      throw std::runtime_error("cannot open oracle progress report");
    progress << std::setprecision(17)
             << "# column metrics: rate,cumulative_rate,inverse_loss; "
                "continuous metrics: dedx,range,inverse_range\n"
             << "# each metric fields: live_oracle_attempts,"
                "live_calculate_stochastic_loss_calls,"
                "compared,device_fallbacks,live_oracle_failures,"
                "live_root_exceptions,"
                "status_failures,outside_1e10,host_outside_1e10,"
                "outside_1e11,host_outside_1e11,"
                "max_normalized,host_max_normalized,first_divergence_sample,"
                "first_root_exception_sample\n"
             << "kind,pid,process,component,completed_samples,metric_1,"
                "metric_2,metric_3\n";
    progress.flush();

    std::vector<Summary> summaries;
    std::uint64_t first_history = 1;
    for (auto const* owner : calculator_owners) {
      summaries.push_back(compare(
          backend, host, *owner, decision_samples, chunk_size,
          first_history));
      first_history += decision_samples + 1;
    }

    auto const boundary_summary = compareRateBoundariesToLiveProposal(
        native_device, host, table, calculator_owners);

    auto const ownerForPdg = [&](std::int32_t pdg) -> CalculatorOwner const& {
      auto const found = std::find_if(
          calculator_owners.begin(), calculator_owners.end(),
          [&](auto const* owner) {
            return static_cast<std::int32_t>(get_PDG(owner->code)) == pdg;
          });
      if (found == calculator_owners.end())
        throw std::runtime_error("native oracle has no calculator for PID");
      return **found;
    };
    std::vector<ColumnOracleSummary> column_oracles;
    column_oracles.reserve(table.dndx_columns.size());
    for (std::size_t index = 0; index < table.dndx_columns.size(); ++index) {
      auto const& column = table.dndx_columns[index];
      column_oracles.push_back(compareColumnToLiveProposal(
          native_device, host, column, ownerForPdg(column.pdg_id), samples,
          chunk_size,
          seed ^ (static_cast<std::uint64_t>(index + 1) << 32),
          &progress));
    }
    std::vector<ContinuousOracleSummary> continuous_oracles;
    for (auto const* owner :
         {&electron, &positron, &muon_minus, &muon_plus}) {
      continuous_oracles.push_back(compareContinuousToLiveProposal(
          native_device, host, table, *owner, samples,
          chunk_size,
          seed ^ (static_cast<std::uint64_t>(continuous_oracles.size() + 1)
                  << 56),
          &progress));
    }

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("cannot open decision report");
    output << std::setprecision(17)
           << "{\n  \"mode\":\"proposal-native-first-divergence\",\n"
           << "  \"production_path_modified\":false,\n"
           << "  \"decision_samples_per_particle\":"
           << decision_samples << ",\n"
           << "  \"decision_loss_comparison_tolerance\":"
           << InteractionOracleTolerance << ",\n"
           << "  \"decision_loss_ulp_tolerance\":"
           << InteractionOracleUlpTolerance << ",\n"
           << "  \"oracle_samples_per_column\":" << samples << ",\n"
           << "  \"oracle_chunk_size\":" << chunk_size << ",\n"
           << "  \"progress_sidecar\":";
    writeJsonString(output, progress_path.filename().string());
    output << ",\n"
           << "  \"seed\":" << seed << ",\n"
           << "  \"table_sha256\":\"" << toHex(table.content_hash)
           << "\",\n  \"provenance\":{\n"
           << "    \"proposal_stochastic_cut_MeV\":"
           << proposal_stochastic_cut / 1_MeV << ",\n"
           << "    \"em_transport_cut_MeV\":"
           << config.em_transport_cut_MeV << ",\n"
           << "    \"muon_transport_cut_MeV\":"
           << config.muon_transport_cut_MeV << ",\n"
           << "    \"muon_transport_cut_used_as_stochastic_cut\":false,\n"
           << "    \"calculator_stochastic_cuts\":[\n";
    for (std::size_t index = 0; index < calculator_owners.size(); ++index) {
      auto const* owner = calculator_owners[index];
      output << "      {\"pid\":"
             << static_cast<std::int32_t>(get_PDG(owner->code))
             << ",\"stochastic_cut_MeV\":"
             << owner->stochasticCutMeV() << "}"
             << (index + 1 == calculator_owners.size() ? "\n" : ",\n");
    }
    output << "    ]\n  },\n  \"particles\":[\n";
    bool accepted = true;
    for (std::size_t index = 0; index < summaries.size(); ++index) {
      auto const& s = summaries[index];
      output << "    {\"pid\":" << s.pid << ",\"requested\":"
             << s.requested << ",\"compared\":" << s.compared
             << ",\"fallbacks\":" << s.fallbacks
             << ",\"native_process_mismatches\":"
             << s.native_process_mismatches
             << ",\"live_proposal_sample_loss_mismatches\":"
             << s.live_proposal_sample_loss_mismatches
             << ",\"residual_quantile_outside_tolerance\":"
             << s.residual_quantile_outside_tolerance
             << ",\"supported_loss_compared\":"
             << s.supported_loss_compared
             << ",\"loss_outside_tolerance\":"
             << s.loss_outside_tolerance
             << ",\"maximum_loss_relative\":"
             << s.maximum_loss_relative << ",\"first_divergence\":";
      writeDivergence(output, s.first);
      output << ",\"first_loss_divergence\":";
      writeDivergence(output, s.first_loss_divergence);
      output << ",\"loss_divergences\":[";
      for (std::size_t divergence_index = 0;
           divergence_index < s.loss_divergences.size();
           ++divergence_index) {
        if (divergence_index != 0) output << ',';
        writeDivergence(
            output,
            std::optional<FirstDivergence>{
                s.loss_divergences[divergence_index]});
      }
      output << ']';
      output << "}" << (index + 1 == summaries.size() ? "\n" : ",\n");
      accepted = accepted && s.native_process_mismatches == 0 &&
                 s.live_proposal_sample_loss_mismatches == 0 &&
                 s.loss_outside_tolerance == 0;
      std::cout << "pid=" << s.pid << " compared=" << s.compared
                << " fallbacks=" << s.fallbacks
                << " native_process_mismatch="
                << s.native_process_mismatches
                << " live_canonical_mismatch="
                << s.live_proposal_sample_loss_mismatches
                << " residual_quantile_outside="
                << s.residual_quantile_outside_tolerance
                << " loss_outside=" << s.loss_outside_tolerance
                << " max_loss_relative=" << s.maximum_loss_relative
                << '\n';
    }
    output << "  ],\n  \"rate_cumulative_boundary_probes\":{\n"
           << "    \"gates_random_selector_acceptance\":false,\n"
           << "    \"selection_consistent\":"
           << (boundary_summary.mismatches == 0 ? "true" : "false")
           << ",\n"
           << "    \"columns_considered\":"
           << boundary_summary.columns_considered << ",\n"
           << "    \"positive_columns\":"
           << boundary_summary.positive_columns << ",\n"
           << "    \"boundary_cases\":"
           << boundary_summary.boundary_cases << ",\n"
           << "    \"mismatches\":" << boundary_summary.mismatches
           << ",\n    \"nextafter_minus_mismatches\":"
           << boundary_summary.nextafter_minus_mismatches
           << ",\n    \"exact_mismatches\":"
           << boundary_summary.exact_mismatches
           << ",\n    \"nextafter_plus_mismatches\":"
           << boundary_summary.nextafter_plus_mismatches
           << ",\n    \"mismatches_over_16_ulp\":"
           << boundary_summary.mismatches_over_16_ulp
           << ",\n    \"status_failures\":"
           << boundary_summary.status_failures
           << ",\n    \"maximum_total_relative\":"
           << boundary_summary.maximum_total_relative
           << ",\n    \"maximum_boundary_ulp_distance\":"
           << boundary_summary.maximum_boundary_ulp_distance
           << ",\n    \"first_divergence\":";
    writeBoundaryDivergence(output, boundary_summary.first);
    output << ",\n    \"first_divergence_over_16_ulp\":";
    writeBoundaryDivergence(output, boundary_summary.first_over_16_ulp);
    output << "\n  },\n  \"live_scalar_oracle\":{\n"
           << "    \"interaction_comparison_tolerance\":1e-10,\n"
           << "    \"continuous_comparison_tolerance\":1e-11,\n"
           << "    \"device_table_bytes\":"
           << native_device.deviceBytes() << ",\n"
           << "    \"columns\":[\n";
    // Exact and adjacent-ULP boundary probes are deliberately adversarial:
    // a few ULP of otherwise accepted device arithmetic can move the branch
    // at a set of measure zero.  Keep their consistency outcome explicit,
    // but do not mix it into the random-selector/statistical acceptance gate.
    for (std::size_t index = 0; index < column_oracles.size(); ++index) {
      auto const& column = column_oracles[index];
      output << "      {\"pid\":" << column.pid
             << ",\"process\":" << column.process
             << ",\"component\":" << column.component << ",\"rate\":";
      writeMetric(output, column.rate);
      output << ",\"cumulative_rate\":";
      writeMetric(output, column.cumulative_rate);
      output << ",\"inverse_loss\":";
      writeMetric(output, column.inverse_loss);
      output << "}"
             << (index + 1 == column_oracles.size() ? "\n" : ",\n");
      accepted = accepted && metricAccepted(column.rate) &&
                 metricAccepted(column.cumulative_rate) &&
                 metricAccepted(column.inverse_loss);
    }
    output << "    ],\n    \"continuous\":[\n";
    for (std::size_t index = 0; index < continuous_oracles.size(); ++index) {
      auto const& continuous = continuous_oracles[index];
      output << "      {\"pid\":" << continuous.pid << ",\"dedx\":";
      writeMetric(output, continuous.dedx);
      output << ",\"range\":";
      writeMetric(output, continuous.range);
      output << ",\"inverse_range\":";
      writeMetric(output, continuous.inverse_range);
      output << "}"
             << (index + 1 == continuous_oracles.size() ? "\n" : ",\n");
      accepted = accepted && metricAccepted(continuous.dedx, 1.e-11) &&
                 metricAccepted(continuous.range, 1.e-11) &&
                 metricAccepted(continuous.inverse_range, 1.e-11);
    }
    output << "    ]\n  },\n  \"accepted\":"
           << (accepted ? "true" : "false") << "\n}\n";
    return accepted ? 0 : 1;
  } catch (std::exception const& error) {
    std::cerr << "proposal-native decision oracle failed: "
              << error.what() << '\n';
    return 1;
  }
}
