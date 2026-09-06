/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/geometry/Line.hpp>
#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <corsika/framework/process/ProcessReturn.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/radio/common/Types.hpp>
#include <corsika/modules/writers/FirstInteractionSnapshot.hpp>

namespace corsika::gpu::em {

  /**
   * Marker used by validation routers that retain records but do not stream
   * them into production CORSIKA writers.
   */
  struct NullCorsikaOutputSink {};

  struct CorsikaOutputSinkStatistics {
    std::uint64_t steps{};
    std::uint64_t deposited_steps{};
    std::uint64_t radio_tracks{};
    std::uint64_t observations{};
    std::uint64_t first_interactions{};
    double weighted_deposited_energy_GeV{};
    double weighted_muon_parent_productions{};
  };

  namespace output_detail {

    inline Code codeFromDevicePid(std::int32_t pid) {
      return convert_from_PDG(static_cast<PDGCode>(pid));
    }

    inline Point pointFromArray(CoordinateSystemPtr const& coordinate_system,
                                double const values_m[3]) {
      return Point{coordinate_system, values_m[0] * meter,
                   values_m[1] * meter, values_m[2] * meter};
    }

    inline DirectionVector directionFromArray(
        CoordinateSystemPtr const& coordinate_system, double const values[3]) {
      DirectionVector direction{coordinate_system,
                                {values[0], values[1], values[2]}};
      auto const norm = direction.getNorm();
      if (!std::isfinite(norm) || !(norm > 0.)) {
        throw std::runtime_error(
            "GPU EM output contains an invalid particle direction");
      }
      return direction.normalized();
    }

    /**
     * Minimal particle facade required by Step and the current CoREAS/ZHS
     * implementations.  It intentionally does not emulate a stack handle.
     */
    class RadioTrackParticle {
    public:
      RadioTrackParticle(Code pid, HEPEnergyType kinetic_energy,
                         DirectionVector direction, Point position, TimeType time,
                         double weight)
          : pid_(pid)
          , kinetic_energy_(kinetic_energy)
          , direction_(std::move(direction))
          , position_(std::move(position))
          , time_(time)
          , weight_(weight) {}

      Code getPID() const { return pid_; }
      HEPEnergyType getKineticEnergy() const { return kinetic_energy_; }
      DirectionVector const& getDirection() const { return direction_; }
      Point const& getPosition() const { return position_; }
      TimeType getTime() const { return time_; }
      double getWeight() const { return weight_; }

    private:
      Code pid_;
      HEPEnergyType kinetic_energy_;
      DirectionVector direction_;
      Point position_;
      TimeType time_;
      double weight_;
    };

  } // namespace output_detail

  /**
   * Stream records produced by PhysicalCudaEmRouter into the same writers and
   * CPU radio algorithms used by the scalar cascade.
   *
   * The device schema stores total energy in GeV.  CORSIKA stack particles
   * and the observation writer use kinetic energy, so the rest mass is
   * subtracted exactly at this boundary.  Energy deposited by a weighted
   * particle is also weighted here, matching ContinuousProcess and
   * ParticleCut on the scalar path.
   */
  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  class CorsikaOutputSink {
  public:
    CorsikaOutputSink(CoordinateSystemPtr coordinate_system,
                      TEnergyLossWriter& energy_loss,
                      TLongitudinalWriter& longitudinal,
                      TProductionWriter& production,
                      TObservationPlane& observation,
                      TInteractionWriter& interaction,
                      TCoreas& coreas, TZhs& zhs,
                      bool radio_enabled, bool gpu_radio_enabled = false)
        : coordinate_system_(std::move(coordinate_system))
        , energy_loss_(energy_loss)
        , longitudinal_(longitudinal)
        , production_(production)
        , observation_(observation)
        , interaction_(interaction)
        , coreas_(coreas)
        , zhs_(zhs)
        , radio_enabled_(radio_enabled)
        , gpu_radio_enabled_(gpu_radio_enabled) {
      if (!coordinate_system_) {
        throw std::invalid_argument(
            "CUDA EM CORSIKA output sink requires a coordinate system");
      }
    }

    void onStep(EmStepRecord const& record) {
      validateStep(record);
      auto const pid = output_detail::codeFromDevicePid(record.pid);
      auto const start =
          output_detail::pointFromArray(coordinate_system_, record.start_position_m);
      auto const end =
          output_detail::pointFromArray(coordinate_system_, record.end_position_m);

      longitudinal_.write(start, end, pid, record.weight);
      auto const continuous_deposit =
          record.deposited_energy_GeV - record.cut_deposited_energy_GeV;
      if (continuous_deposit > 0.) {
        auto const weighted_deposit =
            continuous_deposit * record.weight;
        energy_loss_.write(start, end, pid, weighted_deposit * 1_GeV);
        statistics_.weighted_deposited_energy_GeV += weighted_deposit;
        ++statistics_.deposited_steps;
      }
      if (record.cut_deposited_energy_GeV > 0.) {
        auto const weighted_deposit = record.cut_deposited_energy_GeV * record.weight;
        energy_loss_.write(end, pid, weighted_deposit * 1_GeV);
        statistics_.weighted_deposited_energy_GeV += weighted_deposit;
        ++statistics_.deposited_steps;
      }
      ++statistics_.steps;
    }

    void onProjectedStep(ProjectedEmStepRecord const& record) {
      validateProjectedStep(record);
      auto const pid = output_detail::codeFromDevicePid(record.pid);
      auto const start =
          record.start_grammage_g_per_cm2 * 1_g / square(1_cm);
      auto const end =
          record.end_grammage_g_per_cm2 * 1_g / square(1_cm);

      longitudinal_.writeProjected(start, end, pid, record.weight);
      auto const continuous_deposit =
          record.deposited_energy_GeV - record.cut_deposited_energy_GeV;
      if (continuous_deposit > 0.) {
        auto const weighted_deposit =
            continuous_deposit * record.weight;
        energy_loss_.writeProjected(
            start, end, pid, weighted_deposit * 1_GeV);
        statistics_.weighted_deposited_energy_GeV +=
            weighted_deposit;
        ++statistics_.deposited_steps;
      }
      if (record.cut_deposited_energy_GeV > 0.) {
        auto const weighted_deposit = record.cut_deposited_energy_GeV * record.weight;
        // Use the point-writer bin convention, independent of the continuous
        // dX threshold (which may be zero). No original CPU writer is changed.
        auto const bin_width = energy_loss_.getConfig()["bin-size"].template as<double>();
        auto const bins = energy_loss_.GetNBins();
        if (!std::isfinite(bin_width) || !(bin_width > 0.) || bins == 0)
          throw std::runtime_error("CUDA projected point deposit has invalid binning");
        auto const candidate = static_cast<std::size_t>(
            record.end_grammage_g_per_cm2 / bin_width);
        auto const bin = candidate < bins ? candidate : bins - 1;
        if (is_em(pid))
          energy_loss_.addElectromagneticBin(bin, weighted_deposit * 1_GeV);
        else
          energy_loss_.addBin(bin, weighted_deposit * 1_GeV);
        statistics_.weighted_deposited_energy_GeV += weighted_deposit;
        ++statistics_.deposited_steps;
      }
      ++statistics_.steps;
    }

    void onGpuProfile(GpuProfileResult const& result) {
      auto const bins = result.photons.size();
      if (result.electrons.size() != bins ||
          result.positrons.size() != bins ||
          result.muons_minus.size() != bins ||
          result.muons_plus.size() != bins ||
          result.muon_parent_productions.size() != bins ||
          result.energy_loss_GeV.size() != bins ||
          result.muon_energy_loss_GeV.size() != bins ||
          bins != longitudinal_.getNBins() ||
          bins != production_.getNBins() ||
          bins != energy_loss_.GetNBins()) {
        throw std::runtime_error(
            "CUDA resident profile binning does not match CORSIKA writers");
      }
      if (result.fixed_point_overflows != 0 ||
          result.invalid_records != 0 ||
          !std::isfinite(
              result.weighted_deposited_energy_GeV) ||
          result.weighted_deposited_energy_GeV < 0. ||
          !std::isfinite(
              result.weighted_medium_rest_mass_input_GeV) ||
          result.weighted_medium_rest_mass_input_GeV < 0. ||
          !std::isfinite(
              result.weighted_cut_rest_mass_energy_GeV) ||
          result.weighted_cut_rest_mass_energy_GeV < 0. ||
          !std::isfinite(
              result.weighted_observed_total_energy_GeV) ||
          result.weighted_observed_total_energy_GeV < 0. ||
          !std::isfinite(
              result.weighted_escaped_total_energy_GeV) ||
          result.weighted_escaped_total_energy_GeV < 0. ||
          !std::isfinite(result.weighted_unwritten_photoelectric_binding_energy_GeV) ||
          result.weighted_unwritten_photoelectric_binding_energy_GeV < 0. ||
          !std::isfinite(result.weighted_observation_cut_overlap_energy_GeV) ||
          result.weighted_observation_cut_overlap_energy_GeV < 0. ||
          !std::isfinite(result.weighted_mass_convention_correction_GeV)) {
        throw std::runtime_error(
            "CUDA resident profile contains invalid accumulated output");
      }
      for (std::size_t bin = 0; bin < bins; ++bin) {
        auto const photon = result.photons[bin];
        auto const electron = result.electrons[bin];
        auto const positron = result.positrons[bin];
        auto const muon_minus = result.muons_minus[bin];
        auto const muon_plus = result.muons_plus[bin];
        auto const muon_parent_productions =
            result.muon_parent_productions[bin];
        auto const deposited = result.energy_loss_GeV[bin];
        auto const muon_deposited =
            result.muon_energy_loss_GeV[bin];
        if (!std::isfinite(photon) || photon < 0. ||
            !std::isfinite(electron) || electron < 0. ||
            !std::isfinite(positron) || positron < 0. ||
            !std::isfinite(muon_minus) || muon_minus < 0. ||
            !std::isfinite(muon_plus) || muon_plus < 0. ||
            !std::isfinite(muon_parent_productions) ||
            muon_parent_productions < 0. ||
            !std::isfinite(deposited) || deposited < 0. ||
            !std::isfinite(muon_deposited) ||
            muon_deposited < 0. ||
            muon_deposited > deposited) {
          throw std::runtime_error(
              "CUDA resident profile contains a non-finite or negative bin");
        }
        if (photon != 0.) {
          longitudinal_.addBin(
              bin, Code::Photon, photon);
        }
        if (electron != 0.) {
          longitudinal_.addBin(
              bin, Code::Electron, electron);
        }
        if (positron != 0.) {
          longitudinal_.addBin(
              bin, Code::Positron, positron);
        }
        if (muon_minus != 0.) {
          longitudinal_.addBin(
              bin, Code::MuMinus, muon_minus);
        }
        if (muon_plus != 0.) {
          longitudinal_.addBin(
              bin, Code::MuPlus, muon_plus);
        }
        if (muon_parent_productions != 0.) {
          // ProductionWriter classifies both muon signs into the same parent
          // column and always mirrors the contribution into "all".
          production_.addBin(
              bin, Code::MuMinus,
              muon_parent_productions);
          statistics_.weighted_muon_parent_productions +=
              muon_parent_productions;
        }
        if (deposited != 0.) {
          auto const electromagnetic_deposited =
              deposited - muon_deposited;
          if (electromagnetic_deposited != 0.) {
            energy_loss_.addElectromagneticBin(
                bin, electromagnetic_deposited * 1_GeV);
          }
          if (muon_deposited != 0.) {
            energy_loss_.addBin(
                bin, muon_deposited * 1_GeV);
          }
        }
      }
      statistics_.steps += result.steps;
      statistics_.deposited_steps +=
          result.deposited_steps;
      statistics_.weighted_deposited_energy_GeV +=
          result.weighted_deposited_energy_GeV;
    }

    void onRadioTrack(RadioTrackRecord const& record) {
      if (!radio_enabled_) {
        return;
      }
      if (gpu_radio_enabled_) {
        return;
      }
      validateStep(record.step);
      auto const pid = output_detail::codeFromDevicePid(record.step.pid);
      if (pid != Code::Electron && pid != Code::Positron) {
        return;
      }

      auto const start = output_detail::pointFromArray(
          coordinate_system_, record.step.start_position_m);
      auto const end = output_detail::pointFromArray(
          coordinate_system_, record.step.end_position_m);
      auto const start_direction = output_detail::directionFromArray(
          coordinate_system_, record.start_direction);
      auto const end_direction = output_detail::directionFromArray(
          coordinate_system_, record.end_direction);
      auto const duration =
          (record.step.end_time_s - record.step.start_time_s) * second;
      auto const displacement = end - start;
      if (!(duration > TimeType::zero()) ||
          displacement.getSquaredNorm() == static_pow<2>(0_m)) {
        return;
      }

      auto const mass = get_mass(pid);
      auto const kinetic_energy =
          record.step.start_energy_GeV * 1_GeV - mass;
      if (kinetic_energy < HEPEnergyType::zero()) {
        throw std::runtime_error(
            "CUDA EM radio record has total energy below rest mass");
      }
      output_detail::RadioTrackParticle particle{
          pid, kinetic_energy, start_direction, start,
          record.step.start_time_s * second, record.step.weight};

      auto const average_speed = displacement.getNorm() / duration;
      Line const line{start, displacement / duration};
      StraightTrajectory const trajectory{
          line, duration, duration, start_direction * average_speed,
          end_direction * average_speed};
      Step<output_detail::RadioTrackParticle> step{particle, trajectory};
      step.add_dEkin((record.step.end_energy_GeV -
                      record.step.start_energy_GeV) *
                     1_GeV);

      auto const coreas_return = coreas_.doContinuous(step, false);
      auto const zhs_return = zhs_.doContinuous(step, false);
      if (coreas_return != ProcessReturn::Ok ||
          zhs_return != ProcessReturn::Ok) {
        throw std::runtime_error(
            "CPU radio process rejected a CUDA EM track segment");
      }
      ++statistics_.radio_tracks;
    }

    void onGpuRadioWaveforms(
        radio::GpuRadioWaveforms const& waveforms,
        std::uint64_t track_count) {
      if (!radio_enabled_ || !gpu_radio_enabled_) {
        throw std::logic_error(
            "received CUDA radio waveforms while GPU radio is disabled");
      }
      auto& coreas_observers =
          coreas_.observerCollection().getObservers();
      auto& zhs_observers =
          zhs_.observerCollection().getObservers();
      if (waveforms.coreas.size() != coreas_observers.size() ||
          waveforms.zhs.size() != zhs_observers.size()) {
        throw std::runtime_error(
            "CUDA radio waveform observer count mismatch");
      }
      for (std::size_t index = 0;
           index < waveforms.coreas.size(); ++index) {
        auto const& waveform = waveforms.coreas[index];
        coreas_observers[index].addWaveform(
            waveform.x, waveform.y, waveform.z);
      }
      for (std::size_t index = 0;
           index < waveforms.zhs.size(); ++index) {
        auto const& waveform = waveforms.zhs[index];
        zhs_observers[index].addWaveform(
            waveform.x, waveform.y, waveform.z);
      }
      statistics_.radio_tracks += track_count;
    }

    void onObservation(ObservationRecord const& record) {
      if (record.status != ObservationStatus::ReachedObservationSurface) {
        return;
      }
      auto const pid = output_detail::codeFromDevicePid(record.particle.pid);
      auto const total_energy = record.particle.energy_GeV * 1_GeV;
      auto const kinetic_energy = total_energy - get_mass(pid);
      if (kinetic_energy < HEPEnergyType::zero()) {
        throw std::runtime_error(
            "CUDA EM observation has total energy below rest mass");
      }
      auto const position = output_detail::pointFromArray(
          coordinate_system_, record.particle.position_m);
      auto const direction = output_detail::directionFromArray(
          coordinate_system_, record.particle.direction);
      observation_.writeParticle(
          pid, kinetic_energy, position, direction,
          record.particle.time_s * second, record.particle.weight);
      ++statistics_.observations;
    }

    void onFirstInteraction(
        GpuFirstInteractionSnapshot const& record) {
      if (record.parent_at_vertex.generation != 0 ||
          record.secondary_count == 0 ||
          record.secondary_count > 3) {
        throw std::runtime_error(
            "CUDA EM first-interaction record is structurally invalid");
      }
      auto const parent_pid = output_detail::codeFromDevicePid(
          record.parent_at_vertex.pid);
      auto const parent_total_energy =
          record.parent_at_vertex.energy_GeV * 1_GeV;
      auto const parent_kinetic_energy =
          parent_total_energy - get_mass(parent_pid);
      if (parent_kinetic_energy < HEPEnergyType::zero()) {
        throw std::runtime_error(
            "CUDA first-interaction parent energy is below rest mass");
      }
      FirstInteractionSnapshot snapshot{
          parent_pid,
          parent_kinetic_energy,
          output_detail::pointFromArray(
              coordinate_system_,
              record.parent_at_vertex.position_m),
          output_detail::directionFromArray(
              coordinate_system_,
              record.parent_at_vertex.direction),
          record.parent_at_vertex.time_s * second,
          {}};
      snapshot.secondaries.reserve(record.secondary_count);
      for (std::uint32_t index = 0;
           index < record.secondary_count; ++index) {
        auto const& child = record.secondaries[index];
        auto const pid = output_detail::codeFromDevicePid(child.pid);
        auto const total_energy = child.energy_GeV * 1_GeV;
        if (total_energy < get_mass(pid)) {
          throw std::runtime_error(
              "CUDA first-interaction child energy is below rest mass");
        }
        snapshot.secondaries.push_back(
            InteractionSecondarySnapshot{
                pid, total_energy,
                output_detail::directionFromArray(
                    coordinate_system_, child.direction)});
      }
      if (interaction_.recordFirstInteraction(snapshot)) {
        ++statistics_.first_interactions;
      }
    }

    CorsikaOutputSinkStatistics const& statistics() const noexcept {
      return statistics_;
    }

  private:
    static void validateStep(EmStepRecord const& record) {
      if (!std::isfinite(record.start_time_s) ||
          !std::isfinite(record.end_time_s) ||
          record.end_time_s < record.start_time_s ||
          !std::isfinite(record.start_energy_GeV) ||
          !std::isfinite(record.end_energy_GeV) ||
          record.start_energy_GeV < 0. || record.end_energy_GeV < 0. ||
          !std::isfinite(record.deposited_energy_GeV) ||
          record.deposited_energy_GeV < 0. ||
          !std::isfinite(record.cut_deposited_energy_GeV) ||
          record.cut_deposited_energy_GeV < 0. ||
          record.cut_deposited_energy_GeV > record.deposited_energy_GeV ||
          !std::isfinite(record.weight) ||
          record.weight < 0.) {
        throw std::runtime_error(
            "CUDA EM output contains an invalid step record");
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(record.start_position_m[axis]) ||
            !std::isfinite(record.end_position_m[axis])) {
          throw std::runtime_error(
              "CUDA EM output contains a non-finite position");
        }
      }
    }

    static void validateProjectedStep(
        ProjectedEmStepRecord const& record) {
      if (!std::isfinite(record.start_grammage_g_per_cm2) ||
          !std::isfinite(record.end_grammage_g_per_cm2) ||
          record.start_grammage_g_per_cm2 < 0. ||
          record.end_grammage_g_per_cm2 < 0. ||
          !std::isfinite(record.end_energy_GeV) ||
          record.end_energy_GeV < 0. ||
          !std::isfinite(record.deposited_energy_GeV) ||
          record.deposited_energy_GeV < 0. ||
          !std::isfinite(record.cut_deposited_energy_GeV) ||
          record.cut_deposited_energy_GeV < 0. ||
          record.cut_deposited_energy_GeV > record.deposited_energy_GeV ||
          !std::isfinite(record.weight) ||
          record.weight < 0.) {
        throw std::runtime_error(
            "CUDA EM output contains an invalid projected step record");
      }
    }

    CoordinateSystemPtr coordinate_system_;
    TEnergyLossWriter& energy_loss_;
    TLongitudinalWriter& longitudinal_;
    TProductionWriter& production_;
    TObservationPlane& observation_;
    TInteractionWriter& interaction_;
    TCoreas& coreas_;
    TZhs& zhs_;
    bool radio_enabled_{};
    bool gpu_radio_enabled_{};
    CorsikaOutputSinkStatistics statistics_{};
  };

} // namespace corsika::gpu::em
