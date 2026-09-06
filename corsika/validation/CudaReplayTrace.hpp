/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

/*
 * Diagnostic trace used to compare the scalar PROPOSAL and CUDA EM paths.
 *
 * This is deliberately independent of OutputManager: an incomplete CUDA
 * shower must still leave a readable prefix of its process trace.
 */
#pragma once

#include <corsika/accelerator/em/common/Types.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

namespace corsika::validation {

  class CudaReplayTrace {
  public:
    static CudaReplayTrace& instance() {
      static CudaReplayTrace recorder;
      return recorder;
    }

    void open(std::filesystem::path const& path, std::string backend) {
      if (path.empty()) { return; }
      if (output_.is_open()) {
        throw std::logic_error("CUDA replay trace is already open");
      }
      if (std::filesystem::exists(path)) {
        throw std::runtime_error(
            "refusing to overwrite CUDA replay trace: " + path.string());
      }
      if (!path.parent_path().empty() &&
          !std::filesystem::is_directory(path.parent_path())) {
        throw std::runtime_error(
            "CUDA replay trace parent directory does not exist: " +
            path.parent_path().string());
      }
      output_.open(path);
      if (!output_) {
        throw std::runtime_error(
            "cannot open CUDA replay trace: " + path.string());
      }
      backend_ = std::move(backend);
      output_ << "backend,shower,ordinal,record_kind,history_id,step_id,pdg,"
                 "process_id,component_hash,start_energy_GeV,end_energy_GeV,"
                 "loss_fraction,deposited_energy_GeV,weight,x_m,y_m,z_m,"
                 "nx,ny,nz,time_s,total_rate_cm2_per_g,"
                 "vertex_total_rate_cm2_per_g,interaction_grammage_g_per_cm2,"
                 "distance_uniform,process_uniform,proposal_selection_uniform,"
                 "loss_quantile,final_state_uniform,azimuth_uniform,"
                 "auxiliary_uniform,lpm_survival_probability,lpm_uniform,"
                 "distance_draw_id,process_draw_id,proposal_selection_draw_id,"
                 "loss_draw_id,final_state_draw_id,azimuth_draw_id,"
                 "auxiliary_draw_id,lpm_draw_id,secondary_count\n";
      output_ << std::setprecision(17) << std::scientific;
    }

    bool enabled() const noexcept { return output_.is_open(); }

    void beginShower(unsigned int shower) noexcept {
      shower_ = shower;
      ordinal_ = 0;
    }

    void recordProposalSelection(
        std::int32_t pdg, std::int32_t process_id,
        std::uint64_t component_hash, double start_energy_GeV,
        double loss_fraction, double x_m, double y_m, double z_m,
        double nx, double ny, double nz, double time_s) {
      if (!enabled()) { return; }
      auto const nan = std::numeric_limits<double>::quiet_NaN();
      writeRow(
          "proposal_selection", 0, 0, pdg, process_id, component_hash,
          start_energy_GeV, start_energy_GeV * (1. - loss_fraction),
          loss_fraction, nan, 1., x_m, y_m, z_m, nx, ny, nz, time_s);
    }

    void recordGpuStep(
        std::uint64_t history_id, std::uint64_t step_id, std::int32_t pdg,
        std::int32_t process_id, double start_energy_GeV,
        double end_energy_GeV, double deposited_energy_GeV, double weight,
        double x_m, double y_m, double z_m, double time_s) {
      if (!enabled()) { return; }
      auto const loss_fraction =
          start_energy_GeV > 0.
              ? (start_energy_GeV - end_energy_GeV) / start_energy_GeV
              : 0.;
      auto const nan = std::numeric_limits<double>::quiet_NaN();
      writeRow(
          "gpu_step", history_id, step_id, pdg, process_id, 0,
          start_energy_GeV, end_energy_GeV, loss_fraction,
          deposited_energy_GeV, weight, x_m, y_m, z_m,
          nan, nan, nan, time_s);
    }

    void recordGpuInteraction(gpu::em::EmInteractionRecord const& record) {
      if (!enabled()) { return; }
      auto const& particle = record.particle;
      ExtraFields extra{};
      extra.total_rate_cm2_per_g = record.total_rate_cm2_per_g;
      extra.vertex_total_rate_cm2_per_g =
          record.vertex_total_rate_cm2_per_g;
      extra.interaction_grammage_g_per_cm2 =
          record.interaction_grammage_g_per_cm2;
      extra.distance_uniform = record.distance_uniform;
      extra.process_uniform = record.process_uniform;
      extra.proposal_selection_uniform =
          record.proposal_selection_uniform;
      extra.loss_quantile = record.loss_quantile;
      extra.distance_draw_id = record.distance_draw_id;
      extra.process_draw_id = record.process_draw_id;
      extra.proposal_selection_draw_id =
          record.proposal_selection_draw_id;
      extra.loss_draw_id = record.loss_draw_id;
      writeRow(
          "gpu_interaction", particle.history_id, particle.step_id,
          particle.pid, record.process_id, record.component_hash,
          particle.energy_GeV,
          particle.energy_GeV * (1. - record.energy_fraction),
          record.energy_fraction,
          std::numeric_limits<double>::quiet_NaN(), particle.weight,
          particle.position_m[0], particle.position_m[1],
          particle.position_m[2], particle.direction[0],
          particle.direction[1], particle.direction[2], particle.time_s,
          extra);
    }

    void recordGpuPhotonFinalState(
        gpu::em::PhotonFinalStateRecord const& record,
        gpu::em::EmInteractionRecord const* interaction) {
      if (!enabled()) { return; }
      auto const nan = std::numeric_limits<double>::quiet_NaN();
      ExtraFields extra{};
      extra.final_state_uniform = record.split_uniform;
      extra.azimuth_uniform = record.azimuth_uniform;
      extra.auxiliary_uniform = record.electron_polar_uniform;
      extra.lpm_survival_probability = record.lpm_survival_probability;
      extra.lpm_uniform = record.lpm_uniform;
      extra.final_state_draw_id = record.split_draw_id;
      extra.azimuth_draw_id = record.azimuth_draw_id;
      extra.auxiliary_draw_id = record.electron_polar_draw_id;
      extra.lpm_draw_id = record.lpm_draw_id;
      extra.secondary_count = record.secondary_count;
      auto const history = record.parent_history_id;
      auto const step = interaction == nullptr ? 0 : interaction->particle.step_id;
      auto const pid = interaction == nullptr ? 0 : interaction->particle.pid;
      auto const component = interaction == nullptr ? 0 : interaction->component_hash;
      auto const energy = interaction == nullptr ? nan : interaction->particle.energy_GeV;
      auto const weight = interaction == nullptr ? nan : interaction->particle.weight;
      writeRow(
          "gpu_photon_final_state", history, step, pid, record.process_id,
          component, energy, energy, record.energy_split_fraction, nan,
          weight, nan, nan, nan, nan, nan, nan, nan, extra);
    }

    void recordGpuLeptonFinalState(
        gpu::em::BremsFinalStateRecord const& record,
        gpu::em::EmInteractionRecord const* interaction) {
      if (!enabled()) { return; }
      auto const nan = std::numeric_limits<double>::quiet_NaN();
      ExtraFields extra{};
      extra.final_state_uniform = record.final_state_uniform;
      extra.azimuth_uniform = record.azimuth_uniform;
      extra.auxiliary_uniform = record.auxiliary_uniform;
      extra.lpm_survival_probability = record.lpm_survival_probability;
      extra.lpm_uniform = record.lpm_uniform;
      extra.final_state_draw_id = record.final_state_draw_id;
      extra.azimuth_draw_id = record.azimuth_draw_id;
      extra.auxiliary_draw_id = record.auxiliary_draw_id;
      extra.lpm_draw_id = record.lpm_draw_id;
      extra.secondary_count = record.secondary_count;
      auto const history = record.parent_history_id;
      auto const step = interaction == nullptr ? 0 : interaction->particle.step_id;
      auto const pid = interaction == nullptr ? 0 : interaction->particle.pid;
      auto const component = interaction == nullptr ? 0 : interaction->component_hash;
      auto const energy = interaction == nullptr ? nan : interaction->particle.energy_GeV;
      auto const weight = interaction == nullptr ? nan : interaction->particle.weight;
      writeRow(
          "gpu_lepton_final_state", history, step, pid, record.process_id,
          component, energy, energy, record.photon_energy_fraction, nan,
          weight, nan, nan, nan, nan, nan, nan, nan, extra);
    }

    void close() {
      if (output_.is_open()) {
        output_.flush();
        output_.close();
      }
    }

  private:
    CudaReplayTrace() = default;

    struct ExtraFields {
      double total_rate_cm2_per_g{std::numeric_limits<double>::quiet_NaN()};
      double vertex_total_rate_cm2_per_g{
          std::numeric_limits<double>::quiet_NaN()};
      double interaction_grammage_g_per_cm2{
          std::numeric_limits<double>::quiet_NaN()};
      double distance_uniform{std::numeric_limits<double>::quiet_NaN()};
      double process_uniform{std::numeric_limits<double>::quiet_NaN()};
      double proposal_selection_uniform{
          std::numeric_limits<double>::quiet_NaN()};
      double loss_quantile{std::numeric_limits<double>::quiet_NaN()};
      double final_state_uniform{std::numeric_limits<double>::quiet_NaN()};
      double azimuth_uniform{std::numeric_limits<double>::quiet_NaN()};
      double auxiliary_uniform{std::numeric_limits<double>::quiet_NaN()};
      double lpm_survival_probability{
          std::numeric_limits<double>::quiet_NaN()};
      double lpm_uniform{std::numeric_limits<double>::quiet_NaN()};
      std::uint64_t distance_draw_id{};
      std::uint64_t process_draw_id{};
      std::uint64_t proposal_selection_draw_id{};
      std::uint64_t loss_draw_id{};
      std::uint64_t final_state_draw_id{};
      std::uint64_t azimuth_draw_id{};
      std::uint64_t auxiliary_draw_id{};
      std::uint64_t lpm_draw_id{};
      std::uint32_t secondary_count{};
    };

    void writeRow(
        char const* kind, std::uint64_t history_id, std::uint64_t step_id,
        std::int32_t pdg, std::int32_t process_id,
        std::uint64_t component_hash, double start_energy_GeV,
        double end_energy_GeV, double loss_fraction,
        double deposited_energy_GeV, double weight, double x_m, double y_m,
        double z_m, double nx, double ny, double nz, double time_s) {
      writeRow(
          kind, history_id, step_id, pdg, process_id, component_hash,
          start_energy_GeV, end_energy_GeV, loss_fraction,
          deposited_energy_GeV, weight, x_m, y_m, z_m, nx, ny, nz, time_s,
          ExtraFields{});
    }

    void writeRow(
        char const* kind, std::uint64_t history_id, std::uint64_t step_id,
        std::int32_t pdg, std::int32_t process_id,
        std::uint64_t component_hash, double start_energy_GeV,
        double end_energy_GeV, double loss_fraction,
        double deposited_energy_GeV, double weight, double x_m, double y_m,
        double z_m, double nx, double ny, double nz, double time_s,
        ExtraFields const& extra) {
      output_ << backend_ << ',' << shower_ << ',' << ordinal_++ << ','
              << kind << ',' << history_id << ',' << step_id << ',' << pdg
              << ',' << process_id << ',' << component_hash << ','
              << start_energy_GeV << ',' << end_energy_GeV << ','
              << loss_fraction << ',' << deposited_energy_GeV << ',' << weight
              << ',' << x_m << ',' << y_m << ',' << z_m << ',' << nx << ','
              << ny << ',' << nz << ',' << time_s << ','
              << extra.total_rate_cm2_per_g << ','
              << extra.vertex_total_rate_cm2_per_g << ','
              << extra.interaction_grammage_g_per_cm2 << ','
              << extra.distance_uniform << ',' << extra.process_uniform << ','
              << extra.proposal_selection_uniform << ','
              << extra.loss_quantile << ',' << extra.final_state_uniform << ','
              << extra.azimuth_uniform << ',' << extra.auxiliary_uniform << ','
              << extra.lpm_survival_probability << ',' << extra.lpm_uniform
              << ',' << extra.distance_draw_id << ',' << extra.process_draw_id
              << ',' << extra.proposal_selection_draw_id << ','
              << extra.loss_draw_id << ',' << extra.final_state_draw_id << ','
              << extra.azimuth_draw_id << ',' << extra.auxiliary_draw_id << ','
              << extra.lpm_draw_id << ',' << extra.secondary_count << '\n';
    }

    std::ofstream output_{};
    std::string backend_{};
    unsigned int shower_{};
    std::uint64_t ordinal_{};
  };

} // namespace corsika::validation
