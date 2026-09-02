/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/gpu/em/detail/CudaEmRunSession.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/gpu/em/tables/ProposalNativeAux.hpp>
#include <corsika/gpu/em/tables/ProposalNativeTableExporter.hpp>
#include <corsika/gpu/em/tables/Sha256.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace corsika::gpu::em::detail {

  namespace {

    using namespace corsika::units::si;
    using namespace corsika::gpu::em::tables;

    bool tableHasPid(RateTableSet const& table, std::int32_t const pdg) {
      return std::any_of(
          table.particles.begin(), table.particles.end(),
          [pdg](auto const& particle) { return particle.pdg_id == pdg; });
    }

    bool validateC8EmRtRequirements(
        RateTableSet const& table,
        GpuPhysicsRequirements const& requirements) {
      auto const cut_scale = std::max(
          {1., std::abs(requirements.stochastic_cut_MeV),
           std::abs(table.metadata.energy_cut_MeV)});
      if (std::abs(table.metadata.energy_cut_MeV -
                   requirements.stochastic_cut_MeV) >
          32. * std::numeric_limits<double>::epsilon() * cut_scale) {
        throw std::runtime_error(
            "CUDA EM table stochastic cut does not match the scalar "
            "PROPOSAL cut resolved from the configured production threshold");
      }
      if (table.metadata.energy_min_MeV >
          std::min(requirements.em_transport_cut_MeV,
                   requirements.stochastic_cut_MeV)) {
        throw std::runtime_error(
            "CUDA EM table energy domain does not cover the configured "
            "transport and stochastic cuts");
      }
      for (auto const pdg : {11, -11}) {
        auto const& continuous = findContinuousEnergyTable(table, pdg);
        auto const code = convert_from_PDG(static_cast<PDGCode>(pdg));
        auto const table_transport_cut_MeV =
            continuous.minimum_total_energy_MeV - get_mass(code) / 1_MeV;
        auto const expected_transport_cut_MeV =
            ContinuousCutSafetyFactor * requirements.em_transport_cut_MeV;
        if (std::abs(table_transport_cut_MeV -
                     expected_transport_cut_MeV) >
            2.e-6 * cut_scale) {
          throw std::runtime_error(
              "CUDA EM table transport cut does not match --emcut");
        }
      }

      auto const positive_muon = tableHasPid(table, 13);
      auto const negative_muon = tableHasPid(table, -13);
      if (positive_muon != negative_muon) {
        throw std::runtime_error(
            "CUDA EM table contains only one muon charge state");
      }
      auto const gpu_muon_transport_available =
          positive_muon && negative_muon;
      if (gpu_muon_transport_available) {
        auto const expected_muon_transport_cut_MeV =
            ContinuousCutSafetyFactor * requirements.muon_transport_cut_MeV;
        for (auto const pdg : {13, -13}) {
          auto const& continuous = findContinuousEnergyTable(table, pdg);
          auto const code = convert_from_PDG(static_cast<PDGCode>(pdg));
          auto const table_transport_cut_MeV =
              continuous.minimum_total_energy_MeV - get_mass(code) / 1_MeV;
          auto const muon_cut_scale = std::max(
              {1., std::abs(table_transport_cut_MeV),
               std::abs(expected_muon_transport_cut_MeV)});
          if (std::abs(table_transport_cut_MeV -
                       expected_muon_transport_cut_MeV) >
              2.e-6 * muon_cut_scale) {
            throw std::runtime_error(
                "CUDA muon table transport cut does not match --mucut");
          }
        }
      }
      if (requirements.maximum_primary_energy_MeV >
          table.metadata.energy_max_MeV) {
        throw std::runtime_error(
            "primary energy exceeds the CUDA EM table energy domain");
      }
      return gpu_muon_transport_available;
    }

    void validateProposalNativeRequirements(
        ProposalNativeTableSet const& native_tables,
        GpuPhysicsRequirements const& requirements) {
      auto const native_domain_epsilon =
          64. * std::numeric_limits<double>::epsilon();
      for (auto const pdg : {22, 11, -11, 13, -13}) {
        auto const total = std::find_if(
            native_tables.total_rate_columns.begin(),
            native_tables.total_rate_columns.end(),
            [pdg](auto const& column) { return column.pdg_id == pdg; });
        if (total == native_tables.total_rate_columns.end()) {
          throw std::runtime_error(
              "PROPOSAL native table has no total-rate spline for a "
              "routed particle");
        }
        double upper_energy_MeV = total->spline.axis.high;
        double lower_energy_MeV = total->spline.axis.low;
        for (auto const& column : native_tables.dndx_columns) {
          if (column.pdg_id == pdg &&
              column.rate_model == NativeRateModel::BicubicSpline) {
            upper_energy_MeV =
                std::min(upper_energy_MeV,
                         column.spline.energy_axis.high);
          }
        }
        if (pdg != 22) {
          auto const utility = std::find_if(
              native_tables.utility_columns.begin(),
              native_tables.utility_columns.end(),
              [pdg](auto const& column) { return column.pdg_id == pdg; });
          if (utility == native_tables.utility_columns.end()) {
            throw std::runtime_error(
                "PROPOSAL native table has no continuous range spline "
                "for a routed charged lepton");
          }
          lower_energy_MeV = std::max(
              lower_energy_MeV,
              std::max(utility->lower_energy_limit_MeV,
                       utility->spline.axis.low));
          upper_energy_MeV =
              std::min(upper_energy_MeV, utility->spline.axis.high);
          for (auto const& column : native_tables.dedx_columns) {
            if (column.pdg_id == pdg) {
              upper_energy_MeV =
                  std::min(upper_energy_MeV, column.spline.axis.high);
            }
          }
        }
        auto const code = convert_from_PDG(static_cast<PDGCode>(pdg));
        auto const minimum_transport_energy_MeV =
            pdg == 22
                ? requirements.em_transport_cut_MeV
                : get_mass(code) / 1_MeV + ContinuousCutSafetyFactor *
                      (std::abs(pdg) == 13
                           ? requirements.muon_transport_cut_MeV
                           : requirements.em_transport_cut_MeV);
        auto const lower_scale = std::max(
            {1., std::abs(lower_energy_MeV),
             std::abs(minimum_transport_energy_MeV)});
        if (minimum_transport_energy_MeV +
                native_domain_epsilon * lower_scale <
            lower_energy_MeV) {
          throw std::runtime_error(
              "PROPOSAL native interpolation domain does not cover the "
              "configured transport cut");
        }
        auto const upper_scale = std::max(
            {1., std::abs(upper_energy_MeV),
             std::abs(requirements.maximum_primary_energy_MeV)});
        if (requirements.maximum_primary_energy_MeV >
            upper_energy_MeV + native_domain_epsilon * upper_scale) {
          throw std::runtime_error(
              "configured maximum primary energy exceeds the common "
              "PROPOSAL native interpolation domain");
        }
      }
    }

  } // namespace

  CudaEmRunSession::CudaEmRunSession(
      GpuPhysicsSource const source,
      std::filesystem::path c8emrt_cache,
      std::filesystem::path auxiliary_cache)
      : source_{source}, auxiliary_cache_{std::move(auxiliary_cache)} {
    if (source_ != GpuPhysicsSource::C8EmRt) { return; }
    if (c8emrt_cache.empty()) {
      throw std::runtime_error(
          "c8emrt CUDA physics source requires a table cache");
    }
    loaded_rate_table_.emplace(readRateTable(c8emrt_cache));
    if (loaded_rate_table_->metadata.generator_version !=
        TableGeneratorContractVersion) {
      throw std::runtime_error(
          "CUDA EM table generator contract mismatch: table=" +
          loaded_rate_table_->metadata.generator_version +
          ", required=" + TableGeneratorContractVersion +
          "; regenerate the table with gpu_em_table_prepare");
    }
    for (auto const& particle : loaded_rate_table_->particles) {
      for (auto const& column : particle.columns) {
        if (column.inverse_cdf.reference_mode ==
            SelectedLossCpuFallbackReferenceMode) {
          throw std::runtime_error(
              "CUDA EM table contains a runtime selected-loss CPU fallback; "
              "regenerate it with table contract " +
              std::string(TableGeneratorContractVersion));
        }
      }
    }
  }

  void CudaEmRunSession::requireSource(
      GpuPhysicsSource const requested) const {
    if (source_ != requested) {
      throw std::logic_error(
          "CUDA EM run session was used with a different physics source");
    }
  }

  CudaSessionBeginResult CudaEmRunSession::beginC8EmRt(
      EnvironmentSnapshot const& environment, GpuEmConfig const& config,
      GpuPhysicsRequirements const& requirements) {
    requireSource(GpuPhysicsSource::C8EmRt);
    if (!loaded_rate_table_) {
      throw std::logic_error("c8emrt table was not loaded by the run session");
    }
    auto const muon_available =
        validateC8EmRtRequirements(*loaded_rate_table_, requirements);
    auto const reused = initialized();
    if (!backend_) {
      ProposalTableSet descriptor{};
      descriptor.process_count = rateTableProcessCount(*loaded_rate_table_);
      descriptor.content_hash = loaded_rate_table_->content_hash;
      backend_ = std::make_unique<CudaEmBackend>();
      backend_->initialize(environment, descriptor, config);
    } else {
      backend_->beginShower(makeGpuEmShowerConfig(config));
    }
    return {backend_.get(), reused, muon_available};
  }

  CudaSessionBeginResult CudaEmRunSession::beginProposalNative(
      EnvironmentSnapshot const& environment, GpuEmConfig const& config,
      GpuPhysicsRequirements const& requirements,
      std::vector<proposal::NativeInteractionCalculatorView> const& interactions,
      std::vector<proposal::NativeContinuousCalculatorView> const& continuous) {
    requireSource(GpuPhysicsSource::ProposalNative);
    auto const reused = initialized();
    if (!backend_) {
      auto const native_tables =
          exportProposalNativeTables(interactions, continuous);
      validateProposalNativeRequirements(native_tables, requirements);
      auto const native_aux =
          loadOrCreateProposalNativeAux(interactions, auxiliary_cache_);
      backend_ = std::make_unique<CudaEmBackend>();
      backend_->initialize(environment, native_tables, native_aux, config);
    } else {
      backend_->beginShower(makeGpuEmShowerConfig(config));
    }
    return {backend_.get(), reused, true};
  }

  CudaEmBackend& CudaEmRunSession::backend() {
    if (!backend_) {
      throw std::logic_error("CUDA EM backend has not been initialized");
    }
    return *backend_;
  }

} // namespace corsika::gpu::em::detail
