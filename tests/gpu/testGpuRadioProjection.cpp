/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalConstants.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/geometry/Line.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/radio/CudaRadioAccumulator.hpp>
#include <corsika/modules/radio/CoREAS.hpp>
#include <corsika/modules/radio/RadioProcess.hpp>
#include <corsika/modules/radio/ZHS.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <corsika/modules/radio/propagators/SignalPath.hpp>

namespace {

  using namespace corsika;
  using namespace corsika::gpu;

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void checkCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string(operation) + ": " +
          cudaGetErrorString(status));
    }
  }

  class TestParticle {
  public:
    TestParticle(Code pid, Point position, DirectionVector direction,
                 TimeType time, double weight)
        : pid_(pid)
        , position_(std::move(position))
        , direction_(std::move(direction))
        , time_(time)
        , weight_(weight) {}

    Code getPID() const { return pid_; }
    HEPEnergyType getKineticEnergy() const { return 10_GeV; }
    Point const& getPosition() const { return position_; }
    DirectionVector const& getDirection() const { return direction_; }
    TimeType getTime() const { return time_; }
    double getWeight() const { return weight_; }

  private:
    Code pid_;
    Point position_;
    DirectionVector direction_;
    TimeType time_;
    double weight_;
  };

  class ConstantStraightPropagator {
  public:
    explicit ConstantStraightPropagator(double refractive_index)
        : refractive_index_(refractive_index) {}

    template <typename Particle>
    std::vector<SignalPath> propagate(
        Particle const&, Point const& source,
        Point const& destination) {
      auto const emit = (destination - source).normalized();
      auto const receive = -emit;
      auto const distance = (destination - source).getNorm();
      std::deque<Point> points{source, destination};
      return {SignalPath{
          refractive_index_ * distance / constants::c,
          refractive_index_, refractive_index_,
          refractive_index_, emit, receive, distance, points}};
    }

  private:
    double refractive_index_;
  };

  radio::GpuRadioConfig makeGpuConfig(
      Point const& observer_location, TimeType start_time,
      TimeType duration, InverseTimeType sample_rate,
      std::size_t number_of_bins, double refractive_index) {
    radio::GpuRadioConfig config{};
    config.enabled = true;
    config.coreas_enabled = true;
    config.zhs_enabled = true;
    auto& propagation = config.propagation;
    propagation.minimum_height_m = -1000.;
    propagation.maximum_height_m = 1000.;
    propagation.step_m = 1.;
    propagation.inverse_step_per_m = 1.;
    propagation.refractivity.assign(
        2001, refractive_index - 1.);
    propagation.integrated_refractivity.resize(2001);
    for (std::size_t index = 0;
         index < propagation.integrated_refractivity.size();
         ++index) {
      propagation.integrated_refractivity[index] =
          static_cast<double>(index + 1) *
          (refractive_index - 1.);
    }
    auto const position = observer_location.getCoordinates();
    radio::RadioObserverSnapshot observer{};
    for (int axis = 0; axis < 3; ++axis) {
      observer.position_m[axis] = position[axis] / 1_m;
    }
    observer.start_time_s = start_time / 1_s;
    observer.duration_s = duration / 1_s;
    observer.sample_rate_Hz = sample_rate / 1_Hz;
    observer.number_of_bins = number_of_bins;
    config.coreas_observers.push_back(observer);
    config.zhs_observers.push_back(observer);
    return config;
  }

  double maximumRelativeDifference(
      std::vector<double> const& actual,
      std::vector<double> const& expected) {
    require(actual.size() == expected.size(),
            "waveform sizes differ");
    auto maximum_difference = 0.;
    auto maximum_scale = 0.;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      maximum_difference =
          std::max(maximum_difference,
                   std::abs(actual[index] - expected[index]));
      maximum_scale =
          std::max({maximum_scale, std::abs(actual[index]),
                    std::abs(expected[index])});
    }
    return maximum_difference /
           std::max(maximum_scale, 1.e-300);
  }

  void compareWaveform(
      radio::RadioWaveform const& gpu,
      TimeDomainObserver const& cpu, double tolerance,
      char const* algorithm) {
    auto const dx =
        maximumRelativeDifference(gpu.x, cpu.getWaveformX());
    auto const dy =
        maximumRelativeDifference(gpu.y, cpu.getWaveformY());
    auto const dz =
        maximumRelativeDifference(gpu.z, cpu.getWaveformZ());
    std::ostringstream detail;
    detail.precision(17);
    detail << algorithm
           << " CUDA waveform differs from scalar reference: dx="
           << dx << ", dy=" << dy << ", dz=" << dz;
    require(
        std::max({dx, dy, dz}) <= tolerance,
        detail.str());
  }

} // namespace

int main() {
  int device_count = 0;
  auto const status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no CUDA device is accessible: "
              << cudaGetErrorString(status) << '\n';
    return 77;
  }

  try {
    auto const& coordinate_system =
        get_root_CoordinateSystem();
    constexpr double RefractiveIndex = 1.0003;
    auto const start = Point{
        coordinate_system, 0_m, 0_m, 100_m};
    auto const end = Point{
        coordinate_system, 0_m, 0_m, 90_m};
    auto const observer_location = Point{
        coordinate_system, 100_m, 0_m, 0_m};
    auto const direction =
        (end - start).normalized();
    auto const duration =
        (end - start).getNorm() / (0.999 * constants::c);
    auto const waveform_start = 0_s;
    auto const waveform_duration = 2_us;
    auto const sample_rate = 1_GHz;
    auto const weight = 3.25;

    ObserverCollection<TimeDomainObserver> coreas_detector;
    ObserverCollection<TimeDomainObserver> zhs_detector;
    coreas_detector.addObserver(TimeDomainObserver{
        "coreas", observer_location, coordinate_system,
        waveform_start, waveform_duration, sample_rate,
        waveform_start});
    zhs_detector.addObserver(TimeDomainObserver{
        "zhs", observer_location, coordinate_system,
        waveform_start, waveform_duration, sample_rate,
        waveform_start});

    ConstantStraightPropagator propagator{RefractiveIndex};
    RadioProcess<
        decltype(coreas_detector),
        CoREAS<decltype(coreas_detector),
               ConstantStraightPropagator>,
        ConstantStraightPropagator>
        coreas{coreas_detector, propagator};
    RadioProcess<
        decltype(zhs_detector),
        ZHS<decltype(zhs_detector),
            ConstantStraightPropagator>,
        ConstantStraightPropagator>
        zhs{zhs_detector, propagator};

    TestParticle particle{
        Code::Electron, start, direction, 0_s, weight};
    Line line{start, (end - start) / duration};
    StraightTrajectory trajectory{line, duration};
    Step cpu_step{particle, trajectory};
    require(coreas.doContinuous(cpu_step, false) ==
                ProcessReturn::Ok,
            "scalar CoREAS rejected reference track");
    require(zhs.doContinuous(cpu_step, false) ==
                ProcessReturn::Ok,
            "scalar ZHS rejected reference track");

    auto config = makeGpuConfig(
        observer_location, waveform_start, waveform_duration,
        sample_rate,
        coreas_detector.at(0).getWaveformX().size(),
        RefractiveIndex);
    constexpr std::size_t ObserverCount = 70;
    auto const coreas_observer =
        config.coreas_observers.front();
    auto const zhs_observer =
        config.zhs_observers.front();
    while (config.coreas_observers.size() < ObserverCount) {
      config.coreas_observers.push_back(coreas_observer);
      config.zhs_observers.push_back(zhs_observer);
    }
    config.track_diagnostics = true;
    radio::CudaRadioAccumulator accumulator;
    accumulator.initialize(config, 0, 64 * 1024 * 1024);
    auto rejected_small_budget = false;
    try {
      accumulator.setMemoryBudgetBytes(
          accumulator.deviceBytes() - 1);
    } catch (std::runtime_error const&) {
      rejected_small_budget = true;
    }
    require(
        rejected_small_budget,
        "CUDA radio accepted a budget below its current allocation");
    accumulator.setMemoryBudgetBytes(64 * 1024 * 1024);

    em::LeptonTransportRecord record{};
    record.start.pid =
        static_cast<std::int32_t>(em::EmPid::Electron);
    record.end.pid = record.start.pid;
    record.start.weight = weight;
    record.end.weight = weight;
    record.start.time_s = 0.;
    record.end.time_s = duration / 1_s;
    record.start.energy_GeV = 10.;
    record.end.energy_GeV = 9.9;
    auto const start_coordinates = start.getCoordinates();
    auto const end_coordinates = end.getCoordinates();
    auto const direction_components =
        direction.getComponents();
    for (int axis = 0; axis < 3; ++axis) {
      record.start.position_m[axis] =
          start_coordinates[axis] / 1_m;
      record.end.position_m[axis] =
          end_coordinates[axis] / 1_m;
      record.start.direction[axis] =
          direction_components[axis].magnitude();
      record.end.direction[axis] =
          direction_components[axis].magnitude();
    }

    std::array<em::LeptonTransportRecord, 3> records{
        record, record, record};
    // Resident transport emits a bookkeeping record when a particle is
    // consumed without moving. Scalar CorsikaOutputSink filters it before
    // invoking RadioProcess; the device path must use the same semantics.
    records[1].end.position_m[0] =
        records[1].start.position_m[0];
    records[1].end.position_m[1] =
        records[1].start.position_m[1];
    records[1].end.position_m[2] =
        records[1].start.position_m[2];
    records[1].end.time_s =
        records[1].start.time_s;
    // Muon transport records share the resident lepton queue, but neither
    // scalar RadioProcess nor the CUDA projection kernels accept them.
    records[2].start.pid =
        static_cast<std::int32_t>(em::EmPid::MuonMinus);
    records[2].end.pid = records[2].start.pid;

    em::LeptonTransportRecord* device_record = nullptr;
    checkCuda(cudaMalloc(
                  reinterpret_cast<void**>(&device_record),
                  sizeof(records)),
              "allocate radio test track");
    try {
      checkCuda(cudaMemcpy(
                    device_record, records.data(), sizeof(records),
                    cudaMemcpyHostToDevice),
                "upload radio test track");
      accumulator.accumulateLeptonTracksOnDevice(
          device_record, records.size());
      auto const waveforms = accumulator.downloadWaveforms();
      require(waveforms.coreas.size() == ObserverCount &&
                  waveforms.zhs.size() == ObserverCount,
              "CUDA radio did not return both algorithms");
      for (std::size_t observer = 0;
           observer < ObserverCount; ++observer) {
        compareWaveform(
            waveforms.coreas[observer], coreas_detector.at(0),
            1.e-9, "CoREAS");
        compareWaveform(
            waveforms.zhs[observer], zhs_detector.at(0),
            1.e-9, "ZHS");
      }
      auto const& statistics = accumulator.statistics();
      require(statistics.lepton_tracks == 1,
              "CUDA radio electron/positron track count is incorrect");
      require(
          statistics.track_observer_pairs ==
              2 * records.size() * ObserverCount,
              "CUDA radio track-observer pair count is incorrect");
      require(
          statistics.fused_track_observer_pairs ==
              records.size() * ObserverCount,
              "CUDA CoREAS/ZHS shared-geometry kernel was not exercised");
      require(
          statistics.track_precompute_enabled &&
              statistics.track_tile_size == 4 &&
              statistics.observer_tile_size == 64 &&
              statistics.track_precompute_batches == 1 &&
              statistics.track_precomputed_records == records.size() &&
              statistics.projection_tiles == 2 &&
              statistics.track_workspace_bytes != 0 &&
              statistics.maximum_track_batch == records.size() &&
              statistics.track_precompute_device_time_ms >= 0. &&
              statistics.projection_device_time_ms >= 0.,
          "CUDA radio track precompute/observer tiling was not exercised");
      auto const expected_length_m =
          (end - start).getNorm() / 1_m;
      require(
          statistics.track_diagnostics_enabled &&
              std::abs(
                  statistics.weighted_segment_count - weight) <
                  1.e-14 &&
              std::abs(
                  statistics.track_length_m -
                  expected_length_m) <
                  1.e-12 &&
              std::abs(
                  statistics.weighted_track_length_m -
                  weight * expected_length_m) <
                  1.e-12 &&
              statistics.positron_weighted_track_length_m == 0. &&
              statistics.weighted_direction_change_rad == 0. &&
              statistics
                      .weighted_direction_change_squared_rad2 ==
                  0. &&
              statistics.maximum_direction_change_rad == 0. &&
              std::isfinite(
                  statistics
                      .weighted_beta_deficit_track_length_m) &&
              std::isfinite(
                  statistics.weighted_time_residual_s),
          "CUDA radio track diagnostics do not match the accepted electron");
      require(statistics.coreas_contributions != 0,
              "CUDA CoREAS produced no contribution");
      require(statistics.zhs_contributions != 0,
              "CUDA ZHS produced no contribution");
      require(statistics.zhs_subtracks > 1,
              "CUDA ZHS Fraunhofer subdivision was not exercised");
      require(statistics.fixed_point_overflows == 0,
              "CUDA radio fixed-point accumulator overflowed");

      accumulator.reset();
      accumulator.accumulateLeptonTracksOnDevice(
          device_record, records.size());
      auto const repeated = accumulator.downloadWaveforms();
      require(
          accumulator.statistics().lepton_tracks == 1 &&
              accumulator.statistics().track_observer_pairs ==
                  2 * records.size() * ObserverCount &&
              accumulator.statistics().fused_track_observer_pairs ==
                  records.size() * ObserverCount &&
              accumulator.statistics().track_precompute_batches == 1 &&
              accumulator.statistics().projection_tiles == 2,
          "CUDA radio reset retained preceding-shower statistics");
      require(
          repeated.coreas.size() == ObserverCount &&
              repeated.zhs.size() == ObserverCount,
          "CUDA radio reset changed the observer count");
      auto coreas_repeatable =
          repeated.coreas.size() == waveforms.coreas.size();
      auto zhs_repeatable =
          repeated.zhs.size() == waveforms.zhs.size();
      for (std::size_t observer = 0;
           observer < ObserverCount; ++observer) {
        coreas_repeatable =
            coreas_repeatable &&
            repeated.coreas[observer].x ==
                waveforms.coreas[observer].x &&
            repeated.coreas[observer].y ==
                waveforms.coreas[observer].y &&
            repeated.coreas[observer].z ==
                waveforms.coreas[observer].z;
        zhs_repeatable =
            zhs_repeatable &&
            repeated.zhs[observer].x ==
                waveforms.zhs[observer].x &&
            repeated.zhs[observer].y ==
                waveforms.zhs[observer].y &&
            repeated.zhs[observer].z ==
                waveforms.zhs[observer].z;
      }
      require(
          coreas_repeatable,
          "deterministic CoREAS waveform is not bitwise repeatable");
      require(
          zhs_repeatable,
          "deterministic ZHS waveform is not bitwise repeatable");
    } catch (...) {
      cudaFree(device_record);
      throw;
    }
    cudaFree(device_record);

    std::cout << "GPU radio projection: " << checks
              << " checks passed\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "GPU radio projection test failed after "
              << checks << " checks: " << error.what() << '\n';
    return 1;
  }
}
