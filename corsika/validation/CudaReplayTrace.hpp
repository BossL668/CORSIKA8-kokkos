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
                 "nx,ny,nz,time_s\n";
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

    void close() {
      if (output_.is_open()) {
        output_.flush();
        output_.close();
      }
    }

  private:
    CudaReplayTrace() = default;

    void writeRow(
        char const* kind, std::uint64_t history_id, std::uint64_t step_id,
        std::int32_t pdg, std::int32_t process_id,
        std::uint64_t component_hash, double start_energy_GeV,
        double end_energy_GeV, double loss_fraction,
        double deposited_energy_GeV, double weight, double x_m, double y_m,
        double z_m, double nx, double ny, double nz, double time_s) {
      output_ << backend_ << ',' << shower_ << ',' << ordinal_++ << ','
              << kind << ',' << history_id << ',' << step_id << ',' << pdg
              << ',' << process_id << ',' << component_hash << ','
              << start_energy_GeV << ',' << end_energy_GeV << ','
              << loss_fraction << ',' << deposited_energy_GeV << ',' << weight
              << ',' << x_m << ',' << y_m << ',' << z_m << ',' << nx << ','
              << ny << ',' << nz << ',' << time_s << '\n';
    }

    std::ofstream output_{};
    std::string backend_{};
    unsigned int shower_{};
    std::uint64_t ordinal_{};
  };

} // namespace corsika::validation
