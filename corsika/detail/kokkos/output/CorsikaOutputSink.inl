/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  inline CorsikaOutputSink<TEnergyLossWriter, TLongitudinalWriter, TProductionWriter,
                           TObservationPlane, TInteractionWriter, TCoreas,
                           TZhs>::CorsikaOutputSink(CoordinateSystemPtr coordinate_system,
                                                    TEnergyLossWriter& energy_loss,
                                                    TLongitudinalWriter& longitudinal,
                                                    TProductionWriter& production,
                                                    TObservationPlane& observation,
                                                    TInteractionWriter& interaction,
                                                    TCoreas& coreas, TZhs& zhs,
                                                    bool radio_enabled,
                                                    bool gpu_radio_enabled)
      : coordinateSystem_(std::move(coordinate_system))
      , energyLoss_(energy_loss)
      , longitudinal_(longitudinal)
      , production_(production)
      , observation_(observation)
      , interaction_(interaction)
      , coreas_(coreas)
      , zhs_(zhs)
      , radioEnabled_(radio_enabled)
      , gpuRadioEnabled_(gpu_radio_enabled) {
    if (!coordinateSystem_) {
      throw std::invalid_argument(
          "CUDA EM CORSIKA output sink requires a coordinate system");
    }
  }

  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  inline void CorsikaOutputSink<TEnergyLossWriter, TLongitudinalWriter, TProductionWriter,
                                TObservationPlane, TInteractionWriter, TCoreas,
                                TZhs>::onStep(StepRecord const& record) {
    validateStep(record);
    auto const pid = output_detail::code_from_device_pid(record.pid_);
    auto const start =
        output_detail::point_from_array(coordinateSystem_, record.startPositionM_);
    auto const end =
        output_detail::point_from_array(coordinateSystem_, record.endPositionM_);

    longitudinal_.write(start, end, pid, record.weight_);
    auto const continuous_deposit =
        record.depositedEnergyGeV_ - record.cutDepositedEnergyGeV_;
    if (continuous_deposit > 0.) {
      auto const weighted_deposit = continuous_deposit * record.weight_;
      energyLoss_.write(start, end, pid, weighted_deposit * 1_GeV);
      statistics_.weightedDepositedEnergyGeV_ += weighted_deposit;
      ++statistics_.depositedSteps_;
    }
    if (record.cutDepositedEnergyGeV_ > 0. || record.nativeMuonPairCut_ != 0U) {
      auto const weighted_deposit = record.cutDepositedEnergyGeV_ * record.weight_;
      energyLoss_.write(end, pid, weighted_deposit * 1_GeV);
      statistics_.weightedDepositedEnergyGeV_ += weighted_deposit;
      ++statistics_.depositedSteps_;
    }
    ++statistics_.steps_;
  }

  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  inline void
  CorsikaOutputSink<TEnergyLossWriter, TLongitudinalWriter, TProductionWriter,
                    TObservationPlane, TInteractionWriter, TCoreas,
                    TZhs>::onProjectedStep(ProjectedStepRecord const& record) {
    validateProjectedStep(record);
    auto const pid = output_detail::code_from_device_pid(record.pid_);
    auto const start = record.startGrammageGPerCm2_ * 1_g / square(1_cm);
    auto const end = record.endGrammageGPerCm2_ * 1_g / square(1_cm);

    longitudinal_.writeProjected(start, end, pid, record.weight_);
    auto const continuous_deposit =
        record.depositedEnergyGeV_ - record.cutDepositedEnergyGeV_;
    if (continuous_deposit > 0.) {
      auto const weighted_deposit = continuous_deposit * record.weight_;
      energyLoss_.writeProjected(start, end, pid, weighted_deposit * 1_GeV);
      statistics_.weightedDepositedEnergyGeV_ += weighted_deposit;
      ++statistics_.depositedSteps_;
    }
    if (record.cutDepositedEnergyGeV_ > 0. || record.nativeMuonPairCut_ != 0U) {
      auto const weighted_deposit = record.cutDepositedEnergyGeV_ * record.weight_;
      // Use the point-writer bin convention, independent of the continuous
      // dX threshold (which may be zero). No original CPU writer is changed.
      auto const bin_width = energyLoss_.getConfig()["bin-size"].template as<double>();
      auto const bins = energyLoss_.GetNBins();
      if (!std::isfinite(bin_width) || !(bin_width > 0.) || bins == 0)
        throw std::runtime_error("CUDA projected point deposit has invalid binning");
      auto const candidate =
          static_cast<std::size_t>(record.endGrammageGPerCm2_ / bin_width);
      auto const bin = candidate < bins ? candidate : bins - 1;
      if (is_em(pid))
        energyLoss_.addElectromagneticBin(bin, weighted_deposit * 1_GeV);
      else
        energyLoss_.addBin(bin, weighted_deposit * 1_GeV);
      statistics_.weightedDepositedEnergyGeV_ += weighted_deposit;
      ++statistics_.depositedSteps_;
    }
    ++statistics_.steps_;
  }

  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  inline void CorsikaOutputSink<TEnergyLossWriter, TLongitudinalWriter, TProductionWriter,
                                TObservationPlane, TInteractionWriter, TCoreas,
                                TZhs>::onGpuProfile(ProfileResult const& result) {
    auto const bins = result.photons_.size();
    if (result.electrons_.size() != bins || result.positrons_.size() != bins ||
        result.muonsMinus_.size() != bins || result.muonsPlus_.size() != bins ||
        result.muonParentProductions_.size() != bins ||
        result.energyLossGeV_.size() != bins ||
        result.muonEnergyLossGeV_.size() != bins || bins != longitudinal_.getNBins() ||
        bins != production_.getNBins() || bins != energyLoss_.GetNBins()) {
      throw std::runtime_error(
          "CUDA resident profile binning does not match CORSIKA writers");
    }
    if (result.fixedPointOverflows_ != 0 || result.invalidRecords_ != 0 ||
        !std::isfinite(result.weightedDepositedEnergyGeV_) ||
        !std::isfinite(result.weightedNativeMuonPairCutEnergyGeV_) ||
        result.weightedNativeMuonPairCutEnergyGeV_ > 0. ||
        result.weightedDepositedEnergyGeV_ <
            result.weightedNativeMuonPairCutEnergyGeV_ - 1.e-12 ||
        !std::isfinite(result.weightedMediumRestMassInputGeV_) ||
        result.weightedMediumRestMassInputGeV_ < 0. ||
        !std::isfinite(result.weightedCutRestMassEnergyGeV_) ||
        result.weightedCutRestMassEnergyGeV_ < 0. ||
        !std::isfinite(result.weightedObservedTotalEnergyGeV_) ||
        result.weightedObservedTotalEnergyGeV_ < 0. ||
        !std::isfinite(result.weightedEscapedTotalEnergyGeV_) ||
        result.weightedEscapedTotalEnergyGeV_ < 0. ||
        !std::isfinite(result.weightedUnwrittenPhotoelectricBindingEnergyGeV_) ||
        result.weightedUnwrittenPhotoelectricBindingEnergyGeV_ < 0. ||
        !std::isfinite(result.weightedObservationCutOverlapEnergyGeV_) ||
        result.weightedObservationCutOverlapEnergyGeV_ < 0. ||
        !std::isfinite(result.weightedMassConventionCorrectionGeV_)) {
      throw std::runtime_error(
          "CUDA resident profile contains invalid accumulated output");
    }
    for (std::size_t bin = 0; bin < bins; ++bin) {
      auto const photon = result.photons_[bin];
      auto const electron = result.electrons_[bin];
      auto const positron = result.positrons_[bin];
      auto const muon_minus = result.muonsMinus_[bin];
      auto const muon_plus = result.muonsPlus_[bin];
      auto const muon_parent_productions = result.muonParentProductions_[bin];
      auto const deposited = result.energyLossGeV_[bin];
      auto const muon_deposited = result.muonEnergyLossGeV_[bin];
      if (!std::isfinite(photon) || photon < 0. || !std::isfinite(electron) ||
          electron < 0. || !std::isfinite(positron) || positron < 0. ||
          !std::isfinite(muon_minus) || muon_minus < 0. || !std::isfinite(muon_plus) ||
          muon_plus < 0. || !std::isfinite(muon_parent_productions) ||
          muon_parent_productions < 0. || !std::isfinite(deposited) ||
          deposited < result.weightedNativeMuonPairCutEnergyGeV_ - 1.e-12 ||
          !std::isfinite(muon_deposited) ||
          muon_deposited < result.weightedNativeMuonPairCutEnergyGeV_ - 1.e-12 ||
          muon_deposited > deposited) {
        throw std::runtime_error(
            "CUDA resident profile contains a non-finite or negative bin");
      }
      if (photon != 0.) { longitudinal_.addBin(bin, Code::Photon, photon); }
      if (electron != 0.) { longitudinal_.addBin(bin, Code::Electron, electron); }
      if (positron != 0.) { longitudinal_.addBin(bin, Code::Positron, positron); }
      if (muon_minus != 0.) { longitudinal_.addBin(bin, Code::MuMinus, muon_minus); }
      if (muon_plus != 0.) { longitudinal_.addBin(bin, Code::MuPlus, muon_plus); }
      if (muon_parent_productions != 0.) {
        // ProductionWriter classifies both muon signs into the same parent
        // column and always mirrors the contribution into "all".
        production_.addBin(bin, Code::MuMinus, muon_parent_productions);
        statistics_.weightedMuonParentProductions_ += muon_parent_productions;
      }
      if (deposited != 0. || muon_deposited != 0.) {
        auto const electromagnetic_deposited = deposited - muon_deposited;
        if (electromagnetic_deposited != 0.) {
          energyLoss_.addElectromagneticBin(bin, electromagnetic_deposited * 1_GeV);
        }
        if (muon_deposited != 0.) { energyLoss_.addBin(bin, muon_deposited * 1_GeV); }
      }
    }
    statistics_.steps_ += result.steps_;
    statistics_.depositedSteps_ += result.depositedSteps_;
    statistics_.weightedDepositedEnergyGeV_ += result.weightedDepositedEnergyGeV_;
  }

  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  inline void CorsikaOutputSink<TEnergyLossWriter, TLongitudinalWriter, TProductionWriter,
                                TObservationPlane, TInteractionWriter, TCoreas,
                                TZhs>::onRadioTrack(RadioTrackRecord const& record) {
    if (!radioEnabled_) { return; }
    if (gpuRadioEnabled_) { return; }
    validateStep(record.step_);
    auto const pid = output_detail::code_from_device_pid(record.step_.pid_);
    if (pid != Code::Electron && pid != Code::Positron) { return; }

    auto const start =
        output_detail::point_from_array(coordinateSystem_, record.step_.startPositionM_);
    auto const end =
        output_detail::point_from_array(coordinateSystem_, record.step_.endPositionM_);
    auto const start_direction =
        output_detail::direction_from_array(coordinateSystem_, record.startDirection_);
    auto const end_direction =
        output_detail::direction_from_array(coordinateSystem_, record.endDirection_);
    auto const duration = (record.step_.endTimeS_ - record.step_.startTimeS_) * second;
    auto const displacement = end - start;
    if (!(duration > TimeType::zero()) ||
        displacement.getSquaredNorm() == static_pow<2>(0_m)) {
      return;
    }

    auto const mass = get_mass(pid);
    auto const kinetic_energy = record.step_.startEnergyGeV_ * 1_GeV - mass;
    if (kinetic_energy < HEPEnergyType::zero()) {
      throw std::runtime_error("CUDA EM radio record has total energy below rest mass");
    }
    output_detail::RadioTrackParticle particle{pid,
                                               kinetic_energy,
                                               start_direction,
                                               start,
                                               record.step_.startTimeS_ * second,
                                               record.step_.weight_};

    auto const average_speed = displacement.getNorm() / duration;
    Line const line{start, displacement / duration};
    StraightTrajectory const trajectory{line, duration, duration,
                                        start_direction * average_speed,
                                        end_direction * average_speed};
    Step<output_detail::RadioTrackParticle> step{particle, trajectory};
    step.add_dEkin((record.step_.endEnergyGeV_ - record.step_.startEnergyGeV_) * 1_GeV);

    auto const coreas_return = coreas_.doContinuous(step, false);
    auto const zhs_return = zhs_.doContinuous(step, false);
    if (coreas_return != ProcessReturn::Ok || zhs_return != ProcessReturn::Ok) {
      throw std::runtime_error("CPU radio process rejected a CUDA EM track segment");
    }
    ++statistics_.radioTracks_;
  }

  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  inline void
  CorsikaOutputSink<TEnergyLossWriter, TLongitudinalWriter, TProductionWriter,
                    TObservationPlane, TInteractionWriter, TCoreas,
                    TZhs>::onGpuRadioWaveforms(radio::RadioWaveforms const& waveforms,
                                               std::uint64_t track_count) {
    if (!radioEnabled_ || !gpuRadioEnabled_) {
      throw std::logic_error("received CUDA radio waveforms while GPU radio is disabled");
    }
    auto& coreas_observers = coreas_.observerCollection().getObservers();
    auto& zhs_observers = zhs_.observerCollection().getObservers();
    if (waveforms.coreas_.size() != coreas_observers.size() ||
        waveforms.zhs_.size() != zhs_observers.size()) {
      throw std::runtime_error("CUDA radio waveform observer count mismatch");
    }
    for (std::size_t index = 0; index < waveforms.coreas_.size(); ++index) {
      auto const& waveform = waveforms.coreas_[index];
      coreas_observers[index].addWaveform(waveform.x_, waveform.y_, waveform.z_);
    }
    for (std::size_t index = 0; index < waveforms.zhs_.size(); ++index) {
      auto const& waveform = waveforms.zhs_[index];
      zhs_observers[index].addWaveform(waveform.x_, waveform.y_, waveform.z_);
    }
    statistics_.radioTracks_ += track_count;
  }

  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  inline void CorsikaOutputSink<TEnergyLossWriter, TLongitudinalWriter, TProductionWriter,
                                TObservationPlane, TInteractionWriter, TCoreas,
                                TZhs>::onObservation(ObservationRecord const& record) {
    if (record.status_ != ObservationStatus::ReachedObservationSurface) { return; }
    auto const pid = output_detail::code_from_device_pid(record.particle_.pid_);
    auto const total_energy = record.particle_.energyGeV_ * 1_GeV;
    auto const kinetic_energy = total_energy - get_mass(pid);
    if (kinetic_energy < HEPEnergyType::zero()) {
      throw std::runtime_error("CUDA EM observation has total energy below rest mass");
    }
    auto const position =
        output_detail::point_from_array(coordinateSystem_, record.particle_.positionM_);
    auto const direction = output_detail::direction_from_array(
        coordinateSystem_, record.particle_.direction_);
    observation_.writeParticle(pid, kinetic_energy, position, direction,
                               record.particle_.timeS_ * second,
                               record.particle_.weight_);
    ++statistics_.observations_;
  }

  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  inline void
  CorsikaOutputSink<TEnergyLossWriter, TLongitudinalWriter, TProductionWriter,
                    TObservationPlane, TInteractionWriter, TCoreas,
                    TZhs>::onFirstInteraction(FirstInteractionRecord const& record) {
    if (record.parentAtVertex_.generation_ != 0 || record.secondaryCount_ == 0 ||
        record.secondaryCount_ > 3) {
      throw std::runtime_error(
          "CUDA EM first-interaction record is structurally invalid");
    }
    auto const parent_pid =
        output_detail::code_from_device_pid(record.parentAtVertex_.pid_);
    auto const parent_total_energy = record.parentAtVertex_.energyGeV_ * 1_GeV;
    auto const parent_kinetic_energy = parent_total_energy - get_mass(parent_pid);
    if (parent_kinetic_energy < HEPEnergyType::zero()) {
      throw std::runtime_error("CUDA first-interaction parent energy is below rest mass");
    }
    FirstInteractionSnapshot snapshot{
        parent_pid,
        parent_kinetic_energy,
        output_detail::point_from_array(coordinateSystem_,
                                        record.parentAtVertex_.positionM_),
        output_detail::direction_from_array(coordinateSystem_,
                                            record.parentAtVertex_.direction_),
        record.parentAtVertex_.timeS_ * second,
        {}};
    snapshot.secondaries_.reserve(record.secondaryCount_);
    for (std::uint32_t index = 0; index < record.secondaryCount_; ++index) {
      auto const& child = record.secondaries_[index];
      auto const pid = output_detail::code_from_device_pid(child.pid_);
      auto const total_energy = child.energyGeV_ * 1_GeV;
      if (total_energy < get_mass(pid)) {
        throw std::runtime_error(
            "CUDA first-interaction child energy is below rest mass");
      }
      snapshot.secondaries_.push_back(InteractionSecondarySnapshot{
          pid, total_energy,
          output_detail::direction_from_array(coordinateSystem_, child.direction_)});
    }
    if (interaction_.recordFirstInteraction(snapshot)) {
      ++statistics_.firstInteractions_;
    }
  }

  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  inline CorsikaOutputSinkStatistics const& CorsikaOutputSink<
      TEnergyLossWriter, TLongitudinalWriter, TProductionWriter, TObservationPlane,
      TInteractionWriter, TCoreas, TZhs>::statistics() const noexcept {
    return statistics_;
  }

  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  inline void CorsikaOutputSink<TEnergyLossWriter, TLongitudinalWriter, TProductionWriter,
                                TObservationPlane, TInteractionWriter, TCoreas,
                                TZhs>::validateStep(StepRecord const& record) {
    auto const native_cut = is_native_muon_pair_cut_deposit(
                                record.pid_, record.nativeMuonPairCut_,
                                record.endEnergyGeV_, record.cutDepositedEnergyGeV_) &&
                            record.startTimeS_ == record.endTimeS_ &&
                            record.startEnergyGeV_ == record.endEnergyGeV_ &&
                            record.depositedEnergyGeV_ == record.cutDepositedEnergyGeV_;
    if (!std::isfinite(record.startTimeS_) || !std::isfinite(record.endTimeS_) ||
        record.endTimeS_ < record.startTimeS_ || !std::isfinite(record.startEnergyGeV_) ||
        !std::isfinite(record.endEnergyGeV_) || record.startEnergyGeV_ < 0. ||
        record.endEnergyGeV_ < 0. || !std::isfinite(record.depositedEnergyGeV_) ||
        (record.depositedEnergyGeV_ < 0. && !native_cut) ||
        !std::isfinite(record.cutDepositedEnergyGeV_) ||
        (record.cutDepositedEnergyGeV_ < 0. && !native_cut) ||
        (record.nativeMuonPairCut_ != 0U && !native_cut) ||
        record.cutDepositedEnergyGeV_ > record.depositedEnergyGeV_ ||
        !std::isfinite(record.weight_) || record.weight_ < 0.) {
      throw std::runtime_error("CUDA EM output contains an invalid step record");
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(record.startPositionM_[axis]) ||
          !std::isfinite(record.endPositionM_[axis]) ||
          (native_cut && record.startPositionM_[axis] != record.endPositionM_[axis])) {
        throw std::runtime_error("CUDA EM output contains a non-finite position");
      }
    }
  }

  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  inline void
  CorsikaOutputSink<TEnergyLossWriter, TLongitudinalWriter, TProductionWriter,
                    TObservationPlane, TInteractionWriter, TCoreas,
                    TZhs>::validateProjectedStep(ProjectedStepRecord const& record) {
    auto const native_cut =
        is_native_muon_pair_cut_deposit(record.pid_, record.nativeMuonPairCut_,
                                        record.endEnergyGeV_,
                                        record.cutDepositedEnergyGeV_) &&
        record.startGrammageGPerCm2_ == record.endGrammageGPerCm2_ &&
        record.transportLimit_ ==
            static_cast<std::int32_t>(LeptonTransportLimit::ParticleCut) &&
        record.depositedEnergyGeV_ == record.cutDepositedEnergyGeV_;
    if (!std::isfinite(record.startGrammageGPerCm2_) ||
        !std::isfinite(record.endGrammageGPerCm2_) || record.startGrammageGPerCm2_ < 0. ||
        record.endGrammageGPerCm2_ < 0. || !std::isfinite(record.endEnergyGeV_) ||
        record.endEnergyGeV_ < 0. || !std::isfinite(record.depositedEnergyGeV_) ||
        (record.depositedEnergyGeV_ < 0. && !native_cut) ||
        !std::isfinite(record.cutDepositedEnergyGeV_) ||
        (record.cutDepositedEnergyGeV_ < 0. && !native_cut) ||
        (record.nativeMuonPairCut_ != 0U && !native_cut) ||
        record.cutDepositedEnergyGeV_ > record.depositedEnergyGeV_ ||
        !std::isfinite(record.weight_) || record.weight_ < 0.) {
      throw std::runtime_error(
          "CUDA EM output contains an invalid projected step record");
    }
  }
} // namespace corsika::kokkos
