/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  using namespace corsika::units::si;
  template <typename TBackend>
  inline VariableIndexRadioProcess<TBackend>::VariableIndexRadioProcess(
      std::vector<RadioObserver> observers, TBackend backend, OnlineRadioConfig config,
      CoordinateSystemPtr radioCoordinates,
      std::shared_ptr<IFullWaveAccumulator> accumulator,
      std::shared_ptr<IGeometricalOpticsAccumulator> goAccumulator)
      : accumulator_(std::move(accumulator))
      , goAccumulator_(std::move(goAccumulator))
      , observers_(std::move(observers))
      , backend_(std::move(backend))
      , radioCoordinates_(std::move(radioCoordinates))
      , config_(std::move(config)) {
    if (!radioCoordinates_)
      throw std::invalid_argument("firn radio needs explicit local coordinates");
    config_.validate();
    if (accumulator_ && goAccumulator_)
      throw std::invalid_argument("select one firn device radio model");
    if (observers_.empty()) {
      throw std::runtime_error("variable-index radio needs at least one observer");
    }
    for (auto const& observer : observers_) {
      if (observer.name_.empty() || !finite(observer.positionM_)) {
        throw std::runtime_error("invalid radio observer");
      }
    }
    spectra_.resize(observers_.size());
    for (auto& observer : spectra_) {
      for (auto& component : observer) {
        component.assign(config_.grid_.frequencyCount(), {0.0, 0.0});
      }
    }
    if (config_.writeBranchComponents_) {
      branchSpectra_.resize(observers_.size());
      for (auto& observer : branchSpectra_) {
        for (auto& branch : observer) {
          for (auto& component : branch) {
            component.assign(config_.grid_.frequencyCount(), {0.0, 0.0});
          }
        }
      }
    }
  }

  template <typename TBackend>
  template <typename TParticle>
  inline ProcessReturn VariableIndexRadioProcess<TBackend>::doContinuous(
      Step<TParticle> const& step, bool const) {
    Code const pid = step.getParticlePre().getPID();
    if (pid != Code::Electron && pid != Code::Positron) return ProcessReturn::Ok;
    vector_type const start = pointCoordinates(step.getPositionPre());
    vector_type const end = pointCoordinates(step.getPositionPost());
    double const startTimeS = step.getTimePre() / 1_s;
    double const endTimeS = step.getTimePost() / 1_s;
    double const durationS = endTimeS - startTimeS;
    vector_type const displacement = subtract(end, start);
    double const lengthM = norm(displacement);
    double const weight = step.getParticlePre().getWeight();
    if (!finite(start) || !finite(end) || !std::isfinite(startTimeS) ||
        !std::isfinite(endTimeS) || !std::isfinite(weight) || weight < 0.0) {
      throw std::runtime_error("non-finite charged step reached radio process");
    }
    if (!(durationS > 0.0) || !(lengthM > 0.0) || weight == 0.0) {
      ++skippedDegenerateSegments_;
      return ProcessReturn::Ok;
    }
    if (config_.writeReplayTape_) {
      *(replayOutput_.getWriter())
          << showerId_ << static_cast<std::int64_t>(inputSegments_)
          << static_cast<std::int32_t>(get_PDG(pid)) << weight << start[0] << start[1]
          << start[2] << startTimeS << end[0] << end[1] << end[2] << endTimeS
          << parquet::EndRow;
    }

    if (goAccumulator_) {
      goAccumulator_->append({start, end, startTimeS, endTimeS,
                              (pid == Code::Electron ? -1. : 1.) * weight});
      ++inputSegments_;
      return ProcessReturn::Ok;
    }
    if (accumulator_) {
      return accumulateFullWave(start, end, startTimeS, endTimeS, weight, pid);
    }
    vector_type const midpoint = add(start, multiply(displacement, 0.5));
    std::vector<GreenResponseBatch> midpointResponses;
    midpointResponses.reserve(observers_.size());
    double maximumCurvatureCycles = 0.0;
    double maximumPointCurrentIndicator = 0.0;
    for (auto const& observer : observers_) {
      auto response = backend_.evaluate(midpoint, observer.positionM_, config_.grid_);
      validateResponse(response);
      ++responseQueries_;
      auto updateCurvature = [&](PathDiagnostics const& path) {
        double const effectiveDistance =
            1.0 / std::max(path.geometricSpreadingPerM_, 1.0e-30);
        // The exactly integrated finite-track phase removes the O(L) phase
        // variation.  This is the conservative spherical-wave O(L^2/R)
        // remainder of the local eikonal linearization.
        maximumCurvatureCycles =
            std::max(maximumCurvatureCycles,
                     config_.grid_.nyquistHz() * config_.maximumRefractiveIndex_ *
                         lengthM * lengthM / (8.0 * c_m_per_s * effectiveDistance));
      };
      auto updatePointCurrent = [&](PathDiagnostics const& path,
                                    FiniteTrackCoupling coupling) {
        if (coupling != FiniteTrackCoupling::PointCurrentDyadic) return;
        maximumPointCurrentIndicator =
            std::max(maximumPointCurrentIndicator,
                     config_.grid_.nyquistHz() *
                         (durationS + path.sourceRefractiveIndex_ * lengthM / c_m_per_s));
      };
      if (response.branches_.empty()) {
        updateCurvature(response.path_);
        updatePointCurrent(response.path_, response.finiteTrackCoupling_);
      } else {
        for (auto const& branch : response.branches_) {
          updateCurvature(branch.path_);
          updatePointCurrent(branch.path_, branch.finiteTrackCoupling_);
        }
      }
      midpointResponses.push_back(std::move(response));
    }
    std::size_t subdivisions = static_cast<std::size_t>(
        std::max(1.0, std::max({std::ceil(lengthM / config_.maximumLinearizationLengthM_),
                                std::ceil(std::sqrt(maximumCurvatureCycles /
                                                    config_.maximumPhaseCycles_)),
                                std::ceil(maximumPointCurrentIndicator /
                                          config_.maximumPointCurrentPhaseIndicator_)})));
    if (subdivisions > config_.maximumSubdivisions_) {
      throw std::runtime_error("radio phase subdivision exceeds configured limit");
    }
    std::size_t const additionalQueries =
        subdivisions == 1 ? 0 : subdivisions * observers_.size();
    if (responseQueries_ + additionalQueries > config_.maximumResponseQueries_) {
      throw std::runtime_error("radio response-query budget exceeded");
    }
    ++inputSegments_;
    subdividedSegments_ += subdivisions;
    maximumSubdivisionsSeen_ = std::max(maximumSubdivisionsSeen_, subdivisions);
    maximumPhaseCyclesSeen_ = std::max(
        maximumPhaseCyclesSeen_,
        maximumCurvatureCycles / (static_cast<double>(subdivisions) * subdivisions));
    maximumPointCurrentPhaseIndicatorSeen_ =
        std::max(maximumPointCurrentPhaseIndicatorSeen_,
                 maximumPointCurrentIndicator / static_cast<double>(subdivisions));
    double const charge = pid == Code::Electron ? -1.0 : 1.0;

    for (std::size_t child = 0; child < subdivisions; ++child) {
      double const fraction0 = static_cast<double>(child) / subdivisions;
      double const fraction1 = static_cast<double>(child + 1) / subdivisions;
      double const fractionMiddle = 0.5 * (fraction0 + fraction1);
      vector_type const childStart = add(start, multiply(displacement, fraction0));
      vector_type const childEnd = add(start, multiply(displacement, fraction1));
      vector_type const childDisplacement = subtract(childEnd, childStart);
      vector_type const childMidpoint =
          add(start, multiply(displacement, fractionMiddle));
      double const childDurationS = durationS / subdivisions;
      double const midpointTimeS = startTimeS + fractionMiddle * durationS;
      for (std::size_t observerIndex = 0; observerIndex < observers_.size();
           ++observerIndex) {
        GreenResponseBatch const response =
            subdivisions == 1
                ? midpointResponses[observerIndex]
                : backend_.evaluate(childMidpoint, observers_[observerIndex].positionM_,
                                    config_.grid_);
        validateResponse(response);
        if (subdivisions != 1) ++responseQueries_;
        auto accumulateBranch = [&](PathDiagnostics const& path,
                                    std::vector<complex_matrix_type> const& dyadic,
                                    FiniteTrackCoupling coupling) {
          recordBranch(path);
          double const retardedDurationS =
              childDurationS - path.sourceRefractiveIndex_ *
                                   dot(path.emitDirection_, childDisplacement) /
                                   c_m_per_s;
          for (std::size_t frequencyIndex = 1;
               frequencyIndex < config_.grid_.frequencyCount(); ++frequencyIndex) {
            double const frequency = config_.grid_.frequencyHz(frequencyIndex);
            complex_vector_type const projected =
                contract(dyadic[frequencyIndex], childDisplacement);
            double const sincArgument = pi * frequency * retardedDurationS;
            double const finiteTrackFactor =
                coupling == FiniteTrackCoupling::PointCurrentDyadic
                    ? 1.0
                    : (std::abs(sincArgument) < 1.0e-8
                           ? 1.0 - sincArgument * sincArgument / 6.0
                           : std::sin(sincArgument) / sincArgument);
            std::complex<double> const common =
                mu0_h_per_m * elementary_charge_c * charge * weight / (4.0 * pi) *
                std::complex<double>(0.0, -2.0 * pi * frequency) * finiteTrackFactor *
                std::exp(std::complex<double>(
                    0.0, -2.0 * pi * frequency * (midpointTimeS - config_.timeOriginS_)));
            for (std::size_t component = 0; component < 3; ++component) {
              std::complex<double> const value = common * projected[component];
              spectra_[observerIndex][component][frequencyIndex] += value;
              if (config_.writeBranchComponents_) {
                branchSpectra_[observerIndex][branchFamilyIndex(path.topology_)]
                              [component][frequencyIndex] += value;
              }
            }
          }
        };
        if (response.branches_.empty()) {
          accumulateBranch(response.path_, response.dyadicPerM_,
                           response.finiteTrackCoupling_);
        } else {
          for (auto const& branch : response.branches_) {
            accumulateBranch(branch.path_, branch.dyadicPerM_,
                             branch.finiteTrackCoupling_);
          }
        }
      }
    }
    return ProcessReturn::Ok;
  }

  template <typename TBackend>
  template <typename TParticle, typename TTrack>
  inline LengthType VariableIndexRadioProcess<TBackend>::getMaxStepLength(
      TParticle const&, TTrack const&) const {
    return std::numeric_limits<double>::infinity() * 1_m;
  }

  template <typename TBackend>
  inline void VariableIndexRadioProcess<TBackend>::startOfLibrary(
      boost::filesystem::path const& directory) {
    waveformOutput_.initStreamer((directory / "radio_waveforms.parquet").string());
    waveformOutput_.enableCompression();
    waveformOutput_.addField("observer", parquet::Repetition::REQUIRED,
                             parquet::Type::INT32, parquet::ConvertedType::INT_32);
    waveformOutput_.addField("time_s", parquet::Repetition::REQUIRED,
                             parquet::Type::DOUBLE, parquet::ConvertedType::NONE);
    for (auto const* name : {"E_x_Vpm", "E_y_Vpm", "E_z_Vpm"}) {
      waveformOutput_.addField(name, parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
                               parquet::ConvertedType::NONE);
    }
    waveformOutput_.buildStreamer();

    spectrumOutput_.initStreamer((directory / "radio_spectra.parquet").string());
    spectrumOutput_.enableCompression();
    spectrumOutput_.addField("observer", parquet::Repetition::REQUIRED,
                             parquet::Type::INT32, parquet::ConvertedType::INT_32);
    spectrumOutput_.addField("frequency_hz", parquet::Repetition::REQUIRED,
                             parquet::Type::DOUBLE, parquet::ConvertedType::NONE);
    for (auto const* name : {"E_x_real_Vpm_s", "E_x_imag_Vpm_s", "E_y_real_Vpm_s",
                             "E_y_imag_Vpm_s", "E_z_real_Vpm_s", "E_z_imag_Vpm_s"}) {
      spectrumOutput_.addField(name, parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
                               parquet::ConvertedType::NONE);
    }
    spectrumOutput_.buildStreamer();

    if (config_.writeBranchComponents_) {
      branchWaveformOutput_.initStreamer(
          (directory / "radio_branch_waveforms.parquet").string());
      branchWaveformOutput_.enableCompression();
      branchWaveformOutput_.addField("observer", parquet::Repetition::REQUIRED,
                                     parquet::Type::INT32,
                                     parquet::ConvertedType::INT_32);
      branchWaveformOutput_.addField("branch", parquet::Repetition::REQUIRED,
                                     parquet::Type::INT32,
                                     parquet::ConvertedType::INT_32);
      branchWaveformOutput_.addField("time_s", parquet::Repetition::REQUIRED,
                                     parquet::Type::DOUBLE, parquet::ConvertedType::NONE);
      for (auto const* name : {"E_x_Vpm", "E_y_Vpm", "E_z_Vpm"}) {
        branchWaveformOutput_.addField(name, parquet::Repetition::REQUIRED,
                                       parquet::Type::DOUBLE,
                                       parquet::ConvertedType::NONE);
      }
      branchWaveformOutput_.buildStreamer();

      branchSpectrumOutput_.initStreamer(
          (directory / "radio_branch_spectra.parquet").string());
      branchSpectrumOutput_.enableCompression();
      branchSpectrumOutput_.addField("observer", parquet::Repetition::REQUIRED,
                                     parquet::Type::INT32,
                                     parquet::ConvertedType::INT_32);
      branchSpectrumOutput_.addField("branch", parquet::Repetition::REQUIRED,
                                     parquet::Type::INT32,
                                     parquet::ConvertedType::INT_32);
      branchSpectrumOutput_.addField("frequency_hz", parquet::Repetition::REQUIRED,
                                     parquet::Type::DOUBLE, parquet::ConvertedType::NONE);
      for (auto const* name : {"E_x_real_Vpm_s", "E_x_imag_Vpm_s", "E_y_real_Vpm_s",
                               "E_y_imag_Vpm_s", "E_z_real_Vpm_s", "E_z_imag_Vpm_s"}) {
        branchSpectrumOutput_.addField(name, parquet::Repetition::REQUIRED,
                                       parquet::Type::DOUBLE,
                                       parquet::ConvertedType::NONE);
      }
      branchSpectrumOutput_.buildStreamer();
    }

    if (config_.writeReplayTape_) {
      replayOutput_.initStreamer((directory / "radio_steps.parquet").string());
      replayOutput_.enableCompression();
      replayOutput_.addField("segment", parquet::Repetition::REQUIRED,
                             parquet::Type::INT64, parquet::ConvertedType::INT_64);
      replayOutput_.addField("pdg", parquet::Repetition::REQUIRED, parquet::Type::INT32,
                             parquet::ConvertedType::INT_32);
      replayOutput_.addField("weight", parquet::Repetition::REQUIRED,
                             parquet::Type::DOUBLE, parquet::ConvertedType::NONE);
      for (auto const* name : {"start_x_m", "start_y_m", "start_z_m", "start_time_s",
                               "end_x_m", "end_y_m", "end_z_m", "end_time_s"}) {
        replayOutput_.addField(name, parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
                               parquet::ConvertedType::NONE);
      }
      replayOutput_.buildStreamer();
    }
  }

  template <typename TBackend>
  inline void VariableIndexRadioProcess<TBackend>::startOfShower(unsigned int showerId) {
    showerId_ = showerId;
    resetEvent();
  }

  template <typename TBackend>
  inline void VariableIndexRadioProcess<TBackend>::endOfShower(unsigned int showerId) {
    if (showerId != showerId_) throw std::runtime_error("radio shower id mismatch");
    materializeFullWaveSpectra();
    materializeGeometricalOpticsSpectra();
    YAML::Node event;
    if (accumulator_) {
      auto const& statistics = accumulator_->statistics();
      auto device = event["kokkos_full_wave"];
      device["execution_space"] = accumulator_->executionSpaceName();
      device["device_direct_input"] = false;
      device["input_segments"] = statistics.inputSegments_;
      device["batches"] = statistics.batches_;
      device["table_domain_failures"] = statistics.tableDomainFailures_;
      device["response_queries"] = statistics.responseQueries_;
      device["allocated_bytes"] = statistics.totalDeviceBytes_;
      device["host_to_device_mib"] = statistics.hostToDeviceMiB_;
      device["device_to_host_mib"] = statistics.deviceToHostMiB_;
      device["geometry_kernel_ms"] = statistics.geometryKernelMs_;
      device["spectrum_kernel_ms"] = statistics.spectrumKernelMs_;
      device["transfer_ms"] = statistics.transferMs_;
    }
    if (goAccumulator_) {
      auto const& statistics = goAccumulator_->statistics();
      auto device = event["kokkos_geometrical_optics"];
      device["execution_space"] = goAccumulator_->executionSpaceName();
      device["device_direct_input"] = false;
      device["input_segments"] = statistics.inputSegments_;
      device["batches"] = statistics.batches_;
      device["failed_queries"] = statistics.failedQueries_;
      device["allocated_bytes"] = statistics.allocatedBytes_;
      device["geometry_kernel_ms"] = statistics.geometryMs_;
      device["spectrum_kernel_ms"] = statistics.spectrumMs_;
      device["transfer_ms"] = statistics.transferMs_;
    }
    event["input_segments"] = inputSegments_;
    event["subdivided_segments"] = subdividedSegments_;
    event["skipped_degenerate_segments"] = skippedDegenerateSegments_;
    event["response_queries"] = responseQueries_;
    event["maximum_subdivisions"] = maximumSubdivisionsSeen_;
    event["maximum_phase_cycles"] = maximumPhaseCyclesSeen_;
    event["maximum_point_current_phase_indicator"] =
        maximumPointCurrentPhaseIndicatorSeen_;
    event["propagation_branch_contributions"] = branchContributionCount_;
    event["minimum_branch_travel_time_s"] = minimumBranchTravelTimeS_;
    event["maximum_branch_travel_time_s"] = maximumBranchTravelTimeS_;
    event["maximum_branch_fermat_relative_residual"] = maximumBranchFermatResidual_;
    event["worst_fermat_branch_topology"] = worstFermatTopology_;
    event["worst_fermat_horizontal_residual_m"] = worstFermatHorizontalResidualM_;
    event["worst_fermat_invariant_p"] = worstFermatInvariantP_;
    event["maximum_branch_snell_residual"] = maximumBranchSnellResidual_;
    for (auto const& [topology, count] : branchTopologyCounts_) {
      event["branch_topology_counts"][topology] = count;
    }

    for (std::size_t observer = 0; observer < observers_.size(); ++observer) {
      std::array<std::vector<double>, 3> waveforms;
      double maximumImaginary = 0.0;
      double maximumParseval = 0.0;
      for (std::size_t component = 0; component < 3; ++component) {
        spectra_[observer][component].front() = {0.0, 0.0};
        spectra_[observer][component].back() = {
            spectra_[observer][component].back().real(), 0.0};
        double imaginary = 0.0;
        waveforms[component] = inverse_real_spectrum(spectra_[observer][component],
                                                     config_.grid_, &imaginary);
        maximumImaginary = std::max(maximumImaginary, imaginary);
        maximumParseval =
            std::max(maximumParseval,
                     parseval_relative(waveforms[component],
                                       spectra_[observer][component], config_.grid_));
      }
      for (std::size_t index = 0; index < config_.grid_.sampleCount_; ++index) {
        *(waveformOutput_.getWriter())
            << showerId_ << static_cast<std::int32_t>(observer)
            << config_.timeOriginS_ + index * config_.grid_.sampleIntervalS_
            << waveforms[0][index] << waveforms[1][index] << waveforms[2][index]
            << parquet::EndRow;
      }
      for (std::size_t index = 0; index < config_.grid_.frequencyCount(); ++index) {
        *(spectrumOutput_.getWriter())
            << showerId_ << static_cast<std::int32_t>(observer)
            << config_.grid_.frequencyHz(index) << spectra_[observer][0][index].real()
            << spectra_[observer][0][index].imag() << spectra_[observer][1][index].real()
            << spectra_[observer][1][index].imag() << spectra_[observer][2][index].real()
            << spectra_[observer][2][index].imag() << parquet::EndRow;
      }
      auto node = event["observers"][observer];
      node["name"] = observers_[observer].name_;
      for (double coordinate : observers_[observer].positionM_) {
        node["position_local_m"].push_back(coordinate);
      }
      node["maximum_ifft_imaginary_Vpm"] = maximumImaginary;
      node["maximum_parseval_relative"] = maximumParseval;
      node["hermitian_gate_lt_1e-12"] = maximumImaginary < 1.0e-12;
      node["parseval_gate_lt_1e-12"] = maximumParseval < 1.0e-12;

      if (config_.writeBranchComponents_) {
        for (std::size_t branch = 0; branch < branchFamilyCount_; ++branch) {
          std::array<std::vector<double>, 3> branchWaveforms;
          for (std::size_t component = 0; component < 3; ++component) {
            branchSpectra_[observer][branch][component].front() = {0.0, 0.0};
            branchSpectra_[observer][branch][component].back() = {
                branchSpectra_[observer][branch][component].back().real(), 0.0};
            branchWaveforms[component] = inverse_real_spectrum(
                branchSpectra_[observer][branch][component], config_.grid_);
          }
          for (std::size_t index = 0; index < config_.grid_.sampleCount_; ++index) {
            *(branchWaveformOutput_.getWriter())
                << showerId_ << static_cast<std::int32_t>(observer)
                << static_cast<std::int32_t>(branch)
                << config_.timeOriginS_ + index * config_.grid_.sampleIntervalS_
                << branchWaveforms[0][index] << branchWaveforms[1][index]
                << branchWaveforms[2][index] << parquet::EndRow;
          }
          for (std::size_t index = 0; index < config_.grid_.frequencyCount(); ++index) {
            *(branchSpectrumOutput_.getWriter())
                << showerId_ << static_cast<std::int32_t>(observer)
                << static_cast<std::int32_t>(branch) << config_.grid_.frequencyHz(index)
                << branchSpectra_[observer][branch][0][index].real()
                << branchSpectra_[observer][branch][0][index].imag()
                << branchSpectra_[observer][branch][1][index].real()
                << branchSpectra_[observer][branch][1][index].imag()
                << branchSpectra_[observer][branch][2][index].real()
                << branchSpectra_[observer][branch][2][index].imag() << parquet::EndRow;
          }
        }
      }
    }
    diagnosticSummary_["shower_" + std::to_string(showerId_)] = event;
  }

  template <typename TBackend>
  inline void VariableIndexRadioProcess<TBackend>::endOfLibrary() {
    waveformOutput_.closeStreamer();
    spectrumOutput_.closeStreamer();
    if (config_.writeBranchComponents_) {
      branchWaveformOutput_.closeStreamer();
      branchSpectrumOutput_.closeStreamer();
    }
    if (config_.writeReplayTape_) replayOutput_.closeStreamer();
  }

  template <typename TBackend>
  inline YAML::Node VariableIndexRadioProcess<TBackend>::getConfig() const {
    YAML::Node node;
    node["type"] = "VariableIndexRadioProcess";
    node["execution"] = goAccumulator_
                            ? goAccumulator_->executionSpaceName()
                            : (accumulator_ ? accumulator_->executionSpaceName() : "CPU");
    node["source_model"] = "finite_track_point_current_phase_bounded";
    node["propagation_backend"] = backend_.name();
    node["sample_interval_ns"] = config_.grid_.sampleIntervalS_ / 1.0e-9;
    node["sample_count"] = config_.grid_.sampleCount_;
    node["nyquist_hz"] = config_.grid_.nyquistHz();
    node["time_origin_s"] = config_.timeOriginS_;
    node["maximum_phase_cycles"] = config_.maximumPhaseCycles_;
    node["maximum_point_current_phase_indicator"] =
        config_.maximumPointCurrentPhaseIndicator_;
    node["maximum_linearization_length_m"] = config_.maximumLinearizationLengthM_;
    node["analysis_filter_applied"] = false;
    node["coordinate_system"] = "caller-supplied local orthonormal axes, z up";
    node["component_order"] = "local x, y, z";
    node["write_double_precision_replay_tape"] = config_.writeReplayTape_;
    node["write_branch_components"] = config_.writeBranchComponents_;
    if (config_.writeBranchComponents_) {
      for (std::size_t index = 0; index < branchFamilyCount_; ++index) {
        node["branch_codes"][index] = branchFamilyName(index);
      }
    }
    return node;
  }

  template <typename TBackend>
  inline YAML::Node VariableIndexRadioProcess<TBackend>::getSummary() const {
    return diagnosticSummary_;
  }

  template <typename TBackend>
  inline std::size_t VariableIndexRadioProcess<TBackend>::residentSpectrumBytes() const {
    std::size_t const multiplicity =
        config_.writeBranchComponents_ ? 1 + branchFamilyCount_ : 1;
    return observers_.size() * multiplicity * 3 * config_.grid_.frequencyCount() *
           sizeof(std::complex<double>);
  }

  template <typename TBackend>
  template <typename TPoint>
  inline vector_type VariableIndexRadioProcess<TBackend>::pointCoordinates(
      TPoint const& point) const {
    auto const values = point.getCoordinates(radioCoordinates_);
    return {values.getX() / 1_m, values.getY() / 1_m, values.getZ() / 1_m};
  }

  template <typename TBackend>
  inline void VariableIndexRadioProcess<TBackend>::resetEvent() {
    for (auto& observer : spectra_) {
      for (auto& component : observer) {
        std::fill(component.begin(), component.end(), std::complex<double>{0.0, 0.0});
      }
    }
    for (auto& observer : branchSpectra_) {
      for (auto& branch : observer) {
        for (auto& component : branch) {
          std::fill(component.begin(), component.end(), std::complex<double>{0.0, 0.0});
        }
      }
    }
    inputSegments_ = 0;
    subdividedSegments_ = 0;
    skippedDegenerateSegments_ = 0;
    responseQueries_ = 0;
    maximumSubdivisionsSeen_ = 0;
    maximumPhaseCyclesSeen_ = 0.0;
    maximumPointCurrentPhaseIndicatorSeen_ = 0.0;
    branchContributionCount_ = 0;
    minimumBranchTravelTimeS_ = std::numeric_limits<double>::infinity();
    maximumBranchTravelTimeS_ = 0.0;
    maximumBranchFermatResidual_ = 0.0;
    maximumBranchSnellResidual_ = 0.0;
    worstFermatTopology_ = "none";
    worstFermatHorizontalResidualM_ = 0.0;
    worstFermatInvariantP_ = 0.0;
    branchTopologyCounts_.clear();
    if (accumulator_) accumulator_->reset();
    if (goAccumulator_) goAccumulator_->reset();
  }

  template <typename TBackend>
  inline void VariableIndexRadioProcess<TBackend>::validateResponse(
      GreenResponseBatch const& response) const {
    auto valid = [&](PathDiagnostics const& path,
                     std::vector<complex_matrix_type> const& dyadic) {
      return path.valid_ && dyadic.size() == config_.grid_.frequencyCount();
    };
    if (!valid(response.path_, response.dyadicPerM_)) {
      throw std::runtime_error("propagation backend returned invalid aggregate");
    }
    for (auto const& branch : response.branches_) {
      if (!valid(branch.path_, branch.dyadicPerM_)) {
        throw std::runtime_error("propagation backend returned invalid branch");
      }
    }
  }

  template <typename TBackend>
  inline void VariableIndexRadioProcess<TBackend>::recordBranch(
      PathDiagnostics const& path) {
    ++branchContributionCount_;
    ++branchTopologyCounts_[path.topology_];
    minimumBranchTravelTimeS_ = std::min(minimumBranchTravelTimeS_, path.travelTimeS_);
    maximumBranchTravelTimeS_ = std::max(maximumBranchTravelTimeS_, path.travelTimeS_);
    if (path.fermatRelativeResidual_ > maximumBranchFermatResidual_) {
      maximumBranchFermatResidual_ = path.fermatRelativeResidual_;
      worstFermatTopology_ = path.topology_;
      worstFermatHorizontalResidualM_ = path.horizontalResidualM_;
      worstFermatInvariantP_ = path.invariantP_;
    }
    maximumBranchSnellResidual_ =
        std::max(maximumBranchSnellResidual_, path.snellResidual_);
  }

  template <typename TBackend>
  inline std::size_t VariableIndexRadioProcess<TBackend>::branchFamilyIndex(
      std::string const& topology) {
    if (topology == "direct" || topology == "uniform_direct" ||
        topology == "straight_path_average_index")
      return 0;
    if (topology == "surface_reflected") return 1;
    if (topology == "turning_short") return 2;
    if (topology == "turning_long") return 3;
    if (topology == "fullwave_direct_matched") return 4;
    if (topology == "fullwave_air_boundary") return 5;
    return 6;
  }

  template <typename TBackend>
  inline char const* VariableIndexRadioProcess<TBackend>::branchFamilyName(
      std::size_t index) {
    constexpr std::array<char const*, branchFamilyCount_> names{"direct_or_baseline",
                                                                "surface_reflected",
                                                                "turning_short",
                                                                "turning_long",
                                                                "fullwave_direct_matched",
                                                                "fullwave_air_boundary",
                                                                "other"};
    return names.at(index);
  }
  template <typename TBackend>
  inline ProcessReturn VariableIndexRadioProcess<TBackend>::accumulateFullWave(
      vector_type const& start, vector_type const& end, double startTimeS,
      double endTimeS, double weight, Code pid) {
    vector_type const displacement = subtract(end, start);
    double const lengthM = norm(displacement);
    double const durationS = endTimeS - startTimeS;
    vector_type const midpoint = add(start, multiply(displacement, 0.5));
    double const sourceIndex = accumulator_->sourceRefractiveIndex(midpoint[2]);
    double maximumCurvatureCycles = 0.0;
    for (std::size_t observerIndex = 0; observerIndex < observers_.size();
         ++observerIndex) {
      auto const& observer = observers_[observerIndex];
      if (!accumulator_->covers(midpoint, observer.positionM_)) {
        double const horizontalRadiusM = std::hypot(observer.positionM_[0] - midpoint[0],
                                                    observer.positionM_[1] - midpoint[1]);
        throw std::runtime_error(
            "Kokkos full-wave online query is outside the certified table; "
            "observer_index=" +
            std::to_string(observerIndex) + ", observer_name=" + observer.name_ +
            ", source_local_m=[" + std::to_string(midpoint[0]) + "," +
            std::to_string(midpoint[1]) + "," + std::to_string(midpoint[2]) +
            "], horizontal_radius_m=" + std::to_string(horizontalRadiusM) +
            "; Kokkos hybrid GO fallback is intentionally not implicit");
      }
      double const distance = norm(subtract(observer.positionM_, midpoint));
      maximumCurvatureCycles =
          std::max(maximumCurvatureCycles,
                   config_.grid_.nyquistHz() * config_.maximumRefractiveIndex_ * lengthM *
                       lengthM / (8.0 * c_m_per_s * std::max(distance, 1.0e-30)));
    }
    double const maximumPointCurrentIndicator =
        config_.grid_.nyquistHz() * (durationS + sourceIndex * lengthM / c_m_per_s);
    std::size_t const subdivisions = static_cast<std::size_t>(
        std::max(1.0, std::max({std::ceil(lengthM / config_.maximumLinearizationLengthM_),
                                std::ceil(std::sqrt(maximumCurvatureCycles /
                                                    config_.maximumPhaseCycles_)),
                                std::ceil(maximumPointCurrentIndicator /
                                          config_.maximumPointCurrentPhaseIndicator_)})));
    if (subdivisions > config_.maximumSubdivisions_) {
      throw std::runtime_error(
          "Kokkos full-wave phase subdivision exceeds configured limit");
    }
    std::size_t const addedQueries = subdivisions * observers_.size();
    if (responseQueries_ + addedQueries > config_.maximumResponseQueries_) {
      throw std::runtime_error("Kokkos full-wave response-query budget exceeded");
    }
    ++inputSegments_;
    subdividedSegments_ += subdivisions;
    maximumSubdivisionsSeen_ = std::max(maximumSubdivisionsSeen_, subdivisions);
    maximumPhaseCyclesSeen_ = std::max(
        maximumPhaseCyclesSeen_,
        maximumCurvatureCycles / (static_cast<double>(subdivisions) * subdivisions));
    maximumPointCurrentPhaseIndicatorSeen_ =
        std::max(maximumPointCurrentPhaseIndicatorSeen_,
                 maximumPointCurrentIndicator / static_cast<double>(subdivisions));
    double const charge = pid == Code::Electron ? -1.0 : 1.0;
    for (std::size_t child = 0; child < subdivisions; ++child) {
      double const fraction0 = static_cast<double>(child) / subdivisions;
      double const fraction1 = static_cast<double>(child + 1) / subdivisions;
      double const fractionMiddle = 0.5 * (fraction0 + fraction1);
      vector_type const childDisplacement = multiply(displacement, 1.0 / subdivisions);
      vector_type const childMidpoint =
          add(start, multiply(displacement, fractionMiddle));
      double const midpointTimeS = startTimeS + fractionMiddle * durationS;
      double const childSourceIndex =
          accumulator_->sourceRefractiveIndex(childMidpoint[2]);
      for (auto const& observer : observers_) {
        auto const paths =
            accumulator_->opticalPathMeters(childMidpoint, observer.positionM_);
        for (std::size_t branch = 0; branch < 2; ++branch) {
          PathDiagnostics path;
          path.valid_ = true;
          path.topology_ =
              branch == 0 ? "fullwave_direct_matched" : "fullwave_air_boundary";
          path.sourceRefractiveIndex_ = childSourceIndex;
          path.opticalPathM_ = paths[branch];
          path.travelTimeS_ = paths[branch] / c_m_per_s;
          path.geometricLengthM_ = norm(subtract(observer.positionM_, childMidpoint));
          path.geometricSpreadingPerM_ = 1.0 / std::max(path.geometricLengthM_, 1.0e-12);
          recordBranch(path);
        }
      }
      accumulator_->append(
          {childMidpoint, childDisplacement, midpointTimeS, charge * weight});
    }
    responseQueries_ += addedQueries;
    return ProcessReturn::Ok;
  }

  template <typename TBackend>
  inline void VariableIndexRadioProcess<TBackend>::materializeFullWaveSpectra() {
    if (!accumulator_) return;
    auto result = accumulator_->download();
    if (result.total_.size() != spectra_.size())
      throw std::runtime_error("full-wave observer count changed");
    spectra_ = std::move(result.total_);
    if (config_.writeBranchComponents_) {
      if (result.direct_.size() != branchSpectra_.size() ||
          result.boundary_.size() != branchSpectra_.size())
        throw std::runtime_error("full-wave branch spectra are missing");
      for (std::size_t i = 0; i < observers_.size(); ++i) {
        branchSpectra_[i][4] = std::move(result.direct_[i]);
        branchSpectra_[i][5] = std::move(result.boundary_[i]);
      }
    }
  }
  template <typename TBackend>
  inline void VariableIndexRadioProcess<TBackend>::materializeGeometricalOpticsSpectra() {
    if (!goAccumulator_) return;
    auto result = goAccumulator_->download();
    if (result.total_.size() != spectra_.size())
      throw std::runtime_error("GO observer count changed");
    spectra_ = std::move(result.total_);
    if (config_.writeBranchComponents_) {
      if (result.branches_.size() != branchSpectra_.size())
        throw std::runtime_error("GO branch spectra are missing");
      branchSpectra_ = std::move(result.branches_);
    }
    auto const& s = goAccumulator_->statistics();
    if (s.inputSegments_ != inputSegments_)
      throw std::runtime_error("GO input segment count changed");
    subdividedSegments_ = s.subdividedSegments_;
    responseQueries_ = s.responseQueries_;
    maximumSubdivisionsSeen_ = s.maximumSubdivisions_;
    maximumPhaseCyclesSeen_ = s.maximumPhaseCycles_;
    branchContributionCount_ = s.branchContributions_;
    minimumBranchTravelTimeS_ = s.minimumTravelTimeS_;
    maximumBranchTravelTimeS_ = s.maximumTravelTimeS_;
    maximumBranchFermatResidual_ = s.worstFermatPath_.fermatRelativeResidual_;
    worstFermatTopology_ = s.worstFermatPath_.topology_;
    worstFermatHorizontalResidualM_ = s.worstFermatPath_.horizontalResidualM_;
    worstFermatInvariantP_ = s.worstFermatPath_.invariantP_;
    maximumBranchSnellResidual_ = s.maximumSnellResidual_;
    constexpr char const* names[]{
        "direct", "surface_reflected",       "turning_short", "turning_long", "unused",
        "unused", "horizontal_uniform_layer"};
    for (std::size_t i = 0; i < s.branchCounts_.size(); ++i)
      if (s.branchCounts_[i]) branchTopologyCounts_[names[i]] = s.branchCounts_[i];
  }
} // namespace corsika::kokkos::radio::firn
