/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/output/ReplayTrace.hpp>
#include <corsika/kokkos/em/InteractionRecord.hpp>
#include <corsika/kokkos/em/PhotonFinalStateRecord.hpp>
#include <corsika/kokkos/em/LeptonFinalStateRecord.hpp>
#include <iomanip>
#include <stdexcept>
#include <utility>
namespace corsika {
  ReplayTrace& ReplayTrace::instance() {
    static ReplayTrace recorder;
    return recorder;
  }

  void ReplayTrace::open(std::filesystem::path const& path, std::string backend) {
    if (path.empty()) { return; }
    if (output_.is_open()) {
      throw std::logic_error("CUDA replay trace is already open");
    }
    if (std::filesystem::exists(path)) {
      throw std::runtime_error("refusing to overwrite CUDA replay trace: " +
                               path.string());
    }
    if (!path.parent_path().empty() &&
        !std::filesystem::is_directory(path.parent_path())) {
      throw std::runtime_error("CUDA replay trace parent directory does not exist: " +
                               path.parent_path().string());
    }
    output_.open(path);
    if (!output_) {
      throw std::runtime_error("cannot open CUDA replay trace: " + path.string());
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

  bool ReplayTrace::enabled() const noexcept { return output_.is_open(); }

  void ReplayTrace::beginShower(unsigned int shower) noexcept {
    shower_ = shower;
    ordinal_ = 0;
  }

  void ReplayTrace::recordProposalSelection(std::int32_t pdg, std::int32_t processId,
                                            std::uint64_t componentHash,
                                            double startEnergyGeV, double lossFraction,
                                            double xM, double yM, double zM, double nx,
                                            double ny, double nz, double timeS) {
    if (!enabled()) { return; }
    auto const nan = std::numeric_limits<double>::quiet_NaN();
    writeRow("proposal_selection", 0, 0, pdg, processId, componentHash, startEnergyGeV,
             startEnergyGeV * (1. - lossFraction), lossFraction, nan, 1., xM, yM, zM, nx,
             ny, nz, timeS);
  }

  void ReplayTrace::recordGpuStep(std::uint64_t historyId, std::uint64_t stepId,
                                  std::int32_t pdg, std::int32_t processId,
                                  double startEnergyGeV, double endEnergyGeV,
                                  double depositedEnergyGeV, double weight, double xM,
                                  double yM, double zM, double timeS) {
    if (!enabled()) { return; }
    auto const lossFraction =
        startEnergyGeV > 0. ? (startEnergyGeV - endEnergyGeV) / startEnergyGeV : 0.;
    auto const nan = std::numeric_limits<double>::quiet_NaN();
    writeRow("gpu_step", historyId, stepId, pdg, processId, 0, startEnergyGeV,
             endEnergyGeV, lossFraction, depositedEnergyGeV, weight, xM, yM, zM, nan, nan,
             nan, timeS);
  }

  void ReplayTrace::recordGpuInteraction(kokkos::InteractionRecord const& record) {
    if (!enabled()) { return; }
    auto const& particle = record.particle_;
    ExtraFields extra{};
    extra.totalRateCm2PerG_ = record.totalRateCm2PerG_;
    extra.vertexTotalRateCm2PerG_ = record.vertexTotalRateCm2PerG_;
    extra.interactionGrammageGPerCm2_ = record.interactionGrammageGPerCm2_;
    extra.distanceUniform_ = record.distanceUniform_;
    extra.processUniform_ = record.processUniform_;
    extra.proposalSelectionUniform_ = record.proposalSelectionUniform_;
    extra.lossQuantile_ = record.lossQuantile_;
    extra.distanceDrawId_ = record.distanceDrawId_;
    extra.processDrawId_ = record.processDrawId_;
    extra.proposalSelectionDrawId_ = record.proposalSelectionDrawId_;
    extra.lossDrawId_ = record.lossDrawId_;
    writeRow("gpu_interaction", particle.historyId_, particle.stepId_, particle.pid_,
             record.processId_, record.componentHash_, particle.energyGeV_,
             particle.energyGeV_ * (1. - record.energyFraction_), record.energyFraction_,
             std::numeric_limits<double>::quiet_NaN(), particle.weight_,
             particle.positionM_[0], particle.positionM_[1], particle.positionM_[2],
             particle.direction_[0], particle.direction_[1], particle.direction_[2],
             particle.timeS_, extra);
  }

  void ReplayTrace::recordGpuPhotonFinalState(
      kokkos::PhotonFinalStateRecord const& record,
      kokkos::InteractionRecord const* interaction) {
    if (!enabled()) { return; }
    auto const nan = std::numeric_limits<double>::quiet_NaN();
    ExtraFields extra{};
    extra.finalStateUniform_ = record.splitUniform_;
    extra.azimuthUniform_ = record.azimuthUniform_;
    extra.auxiliaryUniform_ = record.electronPolarUniform_;
    extra.lpmSurvivalProbability_ = record.lpmSurvivalProbability_;
    extra.lpmUniform_ = record.lpmUniform_;
    extra.finalStateDrawId_ = record.splitDrawId_;
    extra.azimuthDrawId_ = record.azimuthDrawId_;
    extra.auxiliaryDrawId_ = record.electronPolarDrawId_;
    extra.lpmDrawId_ = record.lpmDrawId_;
    extra.secondaryCount_ = record.secondaryCount_;
    auto const history = record.parentHistoryId_;
    auto const step = interaction == nullptr ? 0 : interaction->particle_.stepId_;
    auto const pid = interaction == nullptr ? 0 : interaction->particle_.pid_;
    auto const component = interaction == nullptr ? 0 : interaction->componentHash_;
    auto const energy = interaction == nullptr ? nan : interaction->particle_.energyGeV_;
    auto const weight = interaction == nullptr ? nan : interaction->particle_.weight_;
    writeRow("gpu_photon_final_state", history, step, pid, record.processId_, component,
             energy, energy, record.energySplitFraction_, nan, weight, nan, nan, nan, nan,
             nan, nan, nan, extra);
  }

  void ReplayTrace::recordGpuLeptonFinalState(
      kokkos::LeptonFinalStateRecord const& record,
      kokkos::InteractionRecord const* interaction) {
    if (!enabled()) { return; }
    auto const nan = std::numeric_limits<double>::quiet_NaN();
    ExtraFields extra{};
    extra.finalStateUniform_ = record.finalStateUniform_;
    extra.azimuthUniform_ = record.azimuthUniform_;
    extra.auxiliaryUniform_ = record.auxiliaryUniform_;
    extra.lpmSurvivalProbability_ = record.lpmSurvivalProbability_;
    extra.lpmUniform_ = record.lpmUniform_;
    extra.finalStateDrawId_ = record.finalStateDrawId_;
    extra.azimuthDrawId_ = record.azimuthDrawId_;
    extra.auxiliaryDrawId_ = record.auxiliaryDrawId_;
    extra.lpmDrawId_ = record.lpmDrawId_;
    extra.secondaryCount_ = record.secondaryCount_;
    auto const history = record.parentHistoryId_;
    auto const step = interaction == nullptr ? 0 : interaction->particle_.stepId_;
    auto const pid = interaction == nullptr ? 0 : interaction->particle_.pid_;
    auto const component = interaction == nullptr ? 0 : interaction->componentHash_;
    auto const energy = interaction == nullptr ? nan : interaction->particle_.energyGeV_;
    auto const weight = interaction == nullptr ? nan : interaction->particle_.weight_;
    writeRow("gpu_lepton_final_state", history, step, pid, record.processId_, component,
             energy, energy, record.photonEnergyFraction_, nan, weight, nan, nan, nan,
             nan, nan, nan, nan, extra);
  }

  void ReplayTrace::close() {
    if (output_.is_open()) {
      output_.flush();
      output_.close();
    }
  }

  void ReplayTrace::writeRow(char const* kind, std::uint64_t historyId,
                             std::uint64_t stepId, std::int32_t pdg,
                             std::int32_t processId, std::uint64_t componentHash,
                             double startEnergyGeV, double endEnergyGeV,
                             double lossFraction, double depositedEnergyGeV,
                             double weight, double xM, double yM, double zM, double nx,
                             double ny, double nz, double timeS) {
    writeRow(kind, historyId, stepId, pdg, processId, componentHash, startEnergyGeV,
             endEnergyGeV, lossFraction, depositedEnergyGeV, weight, xM, yM, zM, nx, ny,
             nz, timeS, ExtraFields{});
  }

  void ReplayTrace::writeRow(char const* kind, std::uint64_t historyId,
                             std::uint64_t stepId, std::int32_t pdg,
                             std::int32_t processId, std::uint64_t componentHash,
                             double startEnergyGeV, double endEnergyGeV,
                             double lossFraction, double depositedEnergyGeV,
                             double weight, double xM, double yM, double zM, double nx,
                             double ny, double nz, double timeS,
                             ExtraFields const& extra) {
    output_ << backend_ << ',' << shower_ << ',' << ordinal_++ << ',' << kind << ','
            << historyId << ',' << stepId << ',' << pdg << ',' << processId << ','
            << componentHash << ',' << startEnergyGeV << ',' << endEnergyGeV << ','
            << lossFraction << ',' << depositedEnergyGeV << ',' << weight << ',' << xM
            << ',' << yM << ',' << zM << ',' << nx << ',' << ny << ',' << nz << ','
            << timeS << ',' << extra.totalRateCm2PerG_ << ','
            << extra.vertexTotalRateCm2PerG_ << ',' << extra.interactionGrammageGPerCm2_
            << ',' << extra.distanceUniform_ << ',' << extra.processUniform_ << ','
            << extra.proposalSelectionUniform_ << ',' << extra.lossQuantile_ << ','
            << extra.finalStateUniform_ << ',' << extra.azimuthUniform_ << ','
            << extra.auxiliaryUniform_ << ',' << extra.lpmSurvivalProbability_ << ','
            << extra.lpmUniform_ << ',' << extra.distanceDrawId_ << ','
            << extra.processDrawId_ << ',' << extra.proposalSelectionDrawId_ << ','
            << extra.lossDrawId_ << ',' << extra.finalStateDrawId_ << ','
            << extra.azimuthDrawId_ << ',' << extra.auxiliaryDrawId_ << ','
            << extra.lpmDrawId_ << ',' << extra.secondaryCount_ << '\n';
  }

} // namespace corsika
