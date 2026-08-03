/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

/*
 * Replay one scalar CORSIKA decision tape on CUDA.
 *
 * The shower itself is not resampled: every scalar transport segment is
 * uploaded and byte-verified on the GPU, and the recorded e-/e+ segments are
 * projected with the CUDA CoREAS/ZHS kernels.  This is intentionally a
 * validation oracle rather than the production CUDA Monte Carlo path.
 */

#include <cuda_runtime_api.h>

#include <CLI/CLI.hpp>

#include <corsika/gpu/em/CudaDecisionReplayVerifier.hpp>
#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/radio/CudaRadioAccumulator.hpp>
#include <corsika/gpu/radio/Types.hpp>
#include <corsika/output/ParquetStreamer.hpp>
#include <corsika/validation/CudaDecisionReplayTape.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  using corsika::gpu::em::LeptonTransportRecord;
  using corsika::validation::ReplayRadioObserver;
  using corsika::validation::ReplayTapeData;
  using corsika::validation::ReplayTransportStep;

  struct Options {
    std::filesystem::path tape;
    std::filesystem::path output;
    int device{0};
    double memory_fraction{0.70};
    double fixed_point_field_limit_V_per_m{1.};
    bool deterministic{true};
  };

  corsika::gpu::radio::GpuRadioConfig makeGpuRadioConfig(
      ReplayTapeData const& tape, Options const& options) {
    corsika::gpu::radio::GpuRadioConfig config{};
    config.enabled =
        !tape.radio.coreas_observers.empty() ||
        !tape.radio.zhs_observers.empty();
    config.coreas_enabled =
        !tape.radio.coreas_observers.empty();
    config.zhs_enabled =
        !tape.radio.zhs_observers.empty();
    config.deterministic = options.deterministic;
    config.fixed_point_field_limit_V_per_m =
        options.fixed_point_field_limit_V_per_m;
    auto const& source = tape.radio.propagation;
    auto& destination = config.propagation;
    destination.minimum_height_m = source.minimum_height_m;
    destination.maximum_height_m = source.maximum_height_m;
    destination.step_m = source.step_m;
    destination.inverse_step_per_m =
        source.inverse_step_per_m;
    destination.slope_refractivity_lower =
        source.slope_refractivity_lower;
    destination.slope_integrated_refractivity_lower =
        source.slope_integrated_refractivity_lower;
    destination.slope_refractivity_upper =
        source.slope_refractivity_upper;
    destination.slope_integrated_refractivity_upper =
        source.slope_integrated_refractivity_upper;
    destination.refractivity = source.refractivity;
    destination.integrated_refractivity =
        source.integrated_refractivity;
    auto convert = [](ReplayRadioObserver const& observer) {
      corsika::gpu::radio::RadioObserverSnapshot result{};
      for (int axis = 0; axis < 3; ++axis) {
        result.position_m[axis] = observer.position_m[axis];
      }
      result.start_time_s = observer.start_time_s;
      result.duration_s = observer.duration_s;
      result.sample_rate_Hz = observer.sample_rate_Hz;
      result.number_of_bins = observer.number_of_bins;
      return result;
    };
    config.coreas_observers.reserve(
        tape.radio.coreas_observers.size());
    for (auto const& observer :
         tape.radio.coreas_observers) {
      config.coreas_observers.push_back(convert(observer));
    }
    config.zhs_observers.reserve(
        tape.radio.zhs_observers.size());
    for (auto const& observer :
         tape.radio.zhs_observers) {
      config.zhs_observers.push_back(convert(observer));
    }
    return config;
  }

  LeptonTransportRecord makeRadioTrack(
      ReplayTransportStep const& source) {
    LeptonTransportRecord result{};
    result.start.pid = source.pdg;
    result.end.pid = source.pdg;
    result.start.energy_GeV =
        source.start_total_energy_GeV;
    result.end.energy_GeV =
        source.end_total_energy_GeV;
    result.start.time_s = source.start_time_s;
    result.end.time_s = source.end_time_s;
    result.start.weight = source.weight;
    result.end.weight = source.weight;
    result.start.history_id = source.ordinal + 1;
    result.end.history_id = source.ordinal + 1;
    result.start.step_id = source.ordinal;
    result.end.step_id = source.ordinal + 1;
    for (int axis = 0; axis < 3; ++axis) {
      result.start.position_m[axis] =
          source.start_position_m[axis];
      result.end.position_m[axis] =
          source.end_position_m[axis];
      result.start.direction[axis] =
          source.start_direction[axis];
      result.end.direction[axis] =
          source.end_direction[axis];
    }
    auto distance2 = 0.;
    for (int axis = 0; axis < 3; ++axis) {
      auto const delta =
          source.end_position_m[axis] -
          source.start_position_m[axis];
      distance2 += delta * delta;
    }
    result.distance_m = std::sqrt(distance2);
    return result;
  }

  class ReplayWaveformWriter {
  public:
    ReplayWaveformWriter(
        std::filesystem::path const& root,
        ReplayTapeData const& tape)
        : tape_(tape) {
      initialize(
          coreas_, root / "CoREAS" / "observers.parquet");
      initialize(
          zhs_, root / "ZHS" / "observers.parquet");
      writeConfig(
          root / "CoREAS" / "config.yaml", "CoREAS",
          tape.radio.coreas_observers);
      writeConfig(
          root / "ZHS" / "config.yaml", "ZHS",
          tape.radio.zhs_observers);
    }

    ~ReplayWaveformWriter() {
      try {
        close();
      } catch (...) {
      }
    }

    void write(
        std::uint64_t shower,
        corsika::gpu::radio::GpuRadioWaveforms const& waveforms) {
      if (waveforms.coreas.size() !=
              tape_.radio.coreas_observers.size() ||
          waveforms.zhs.size() !=
              tape_.radio.zhs_observers.size()) {
        throw std::runtime_error(
            "CUDA replay waveform observer count mismatch");
      }
      for (std::size_t observer_index = 0;
           observer_index < waveforms.coreas.size();
           ++observer_index) {
        auto const& observer =
            tape_.radio.coreas_observers[observer_index];
        auto const& waveform =
            waveforms.coreas[observer_index];
        validateWaveform(observer, waveform);
        auto const dt_ns =
            1.e9 / observer.sample_rate_Hz;
        for (std::size_t bin = 0;
             bin + 1 < observer.number_of_bins; ++bin) {
          *coreas_.getWriter()
              << static_cast<std::uint32_t>(shower)
              << observer.axis_start_ns +
                     static_cast<double>(bin) * dt_ns
              << waveform.x[bin]
              << waveform.y[bin]
              << waveform.z[bin]
              << parquet::EndRow;
        }
      }
      for (std::size_t observer_index = 0;
           observer_index < waveforms.zhs.size();
           ++observer_index) {
        auto const& observer =
            tape_.radio.zhs_observers[observer_index];
        auto const& waveform =
            waveforms.zhs[observer_index];
        validateWaveform(observer, waveform);
        auto const dt_ns =
            1.e9 / observer.sample_rate_Hz;
        for (std::size_t bin = 0;
             bin + 1 < observer.number_of_bins; ++bin) {
          auto const finite_difference = [&](auto const& values) {
            return -(values[bin + 1] - values[bin]) *
                   observer.sample_rate_Hz;
          };
          *zhs_.getWriter()
              << static_cast<std::uint32_t>(shower)
              << observer.axis_start_ns +
                     (static_cast<double>(bin) + 0.5) * dt_ns
              << finite_difference(waveform.x)
              << finite_difference(waveform.y)
              << finite_difference(waveform.z)
              << parquet::EndRow;
        }
      }
    }

    void close() {
      if (closed_) { return; }
      coreas_.closeStreamer();
      zhs_.closeStreamer();
      closed_ = true;
    }

  private:
    static void initialize(
        corsika::ParquetStreamer& output,
        std::filesystem::path const& path) {
      std::filesystem::create_directories(path.parent_path());
      output.initStreamer(path.string());
      output.enableCompression();
      output.addField(
          "Time", parquet::Repetition::REQUIRED,
          parquet::Type::DOUBLE,
          parquet::ConvertedType::NONE);
      output.addField(
          "Ex", parquet::Repetition::REQUIRED,
          parquet::Type::DOUBLE,
          parquet::ConvertedType::NONE);
      output.addField(
          "Ey", parquet::Repetition::REQUIRED,
          parquet::Type::DOUBLE,
          parquet::ConvertedType::NONE);
      output.addField(
          "Ez", parquet::Repetition::REQUIRED,
          parquet::Type::DOUBLE,
          parquet::ConvertedType::NONE);
      output.buildStreamer();
    }

    static void writeConfig(
        std::filesystem::path const& path,
        char const* algorithm,
        std::vector<ReplayRadioObserver> const& observers) {
      std::ofstream output(path);
      if (!output) {
        throw std::runtime_error(
            "cannot write CUDA replay radio config: " +
            path.string());
      }
      output << std::setprecision(17);
      output << "type: RadioProcess\n";
      output << "algorithm: " << algorithm << "\n";
      output << "units:\n";
      output << "  time: ns\n";
      output << "  frequency: GHz\n";
      output << "  electric field: V/m\n";
      output << "  distance: m\n";
      output << "observers:\n";
      for (std::size_t index = 0;
           index < observers.size(); ++index) {
        auto const& observer = observers[index];
        output << "  replay_"
               << std::setw(4) << std::setfill('0') << index
               << std::setfill(' ') << ":\n";
        output << "    type: TimeDomainObserver\n";
        output << "    start time: "
               << observer.start_time_s * 1.e9 << "\n";
        output << "    duration: "
               << observer.duration_s * 1.e9 << "\n";
        output << "    number of bins: "
               << (observer.number_of_bins == 0
                       ? 0
                       : observer.number_of_bins - 1)
               << "\n";
        output << "    sampling frequency: "
               << observer.sample_rate_Hz / 1.e9 << "\n";
        output << "    location: ["
               << observer.position_m[0] << ", "
               << observer.position_m[1] << ", "
               << observer.position_m[2] << "]\n";
      }
    }

    static void validateWaveform(
        ReplayRadioObserver const& observer,
        corsika::gpu::radio::RadioWaveform const& waveform) {
      auto const size =
          static_cast<std::size_t>(observer.number_of_bins);
      if (waveform.x.size() != size ||
          waveform.y.size() != size ||
          waveform.z.size() != size) {
        throw std::runtime_error(
            "CUDA replay waveform bin count mismatch");
      }
    }

    ReplayTapeData const& tape_;
    corsika::ParquetStreamer coreas_{};
    corsika::ParquetStreamer zhs_{};
    bool closed_{};
  };

  void writeSummary(
      std::filesystem::path const& path,
      Options const& options,
      ReplayTapeData const& tape,
      corsika::gpu::em::CudaDecisionReplayVerification const& verification,
      corsika::gpu::radio::GpuRadioStatistics const& radio,
      cudaDeviceProp const& properties,
      std::map<std::uint64_t, std::uint64_t> const& tracks_by_shower) {
    std::ofstream output(path);
    if (!output) {
      throw std::runtime_error(
          "cannot write CUDA replay summary: " + path.string());
    }
    output << std::setprecision(17);
    output << "{\n";
    output << "  \"mode\": \"scalar-decision-injection-cuda-replay\",\n";
    output << "  \"production_cuda_sampling\": false,\n";
    output << "  \"tape\": \"" << options.tape.string() << "\",\n";
    output << "  \"format_version\": "
           << tape.header.format_version << ",\n";
    output << "  \"seed\": " << tape.header.random_seed << ",\n";
    output << "  \"configured_showers\": "
           << tape.header.configured_showers << ",\n";
    output << "  \"gpu\": {\n";
    output << "    \"device\": " << options.device << ",\n";
    output << "    \"name\": \"" << properties.name << "\",\n";
    output << "    \"compute_capability\": \""
           << properties.major << "." << properties.minor << "\"\n";
    output << "  },\n";
    output << "  \"transport_replay\": {\n";
    output << "    \"records\": " << verification.records << ",\n";
    output << "    \"electromagnetic_records\": "
           << verification.electromagnetic_records << ",\n";
    output << "    \"lepton_records\": "
           << verification.lepton_records << ",\n";
    output << "    \"photon_records\": "
           << verification.photon_records << ",\n";
    output << "    \"invalid_records\": "
           << verification.invalid_records << ",\n";
    output << "    \"nonfinite_records\": "
           << verification.nonfinite_records << ",\n";
    output << "    \"negative_energy_records\": "
           << verification.negative_energy_records << ",\n";
    output << "    \"negative_weight_records\": "
           << verification.negative_weight_records << ",\n";
    output << "    \"backward_time_records\": "
           << verification.backward_time_records << ",\n";
    output << "    \"invalid_direction_records\": "
           << verification.invalid_direction_records << ",\n";
    output << "    \"superluminal_records\": "
           << verification.superluminal_records << ",\n";
    output << "    \"byte_hash_mismatches\": "
           << verification.byte_hash_mismatches << ",\n";
    output << "    \"ordered_host_hash\": \""
           << std::hex << verification.ordered_host_hash
           << std::dec << "\",\n";
    output << "    \"ordered_device_hash\": \""
           << std::hex << verification.ordered_device_hash
           << std::dec << "\",\n";
    output << "    \"weighted_track_length_m\": "
           << verification.weighted_track_length_m << ",\n";
    output << "    \"weighted_lepton_track_length_m\": "
           << verification.weighted_lepton_track_length_m << ",\n";
    output << "    \"maximum_direction_norm_error\": "
           << verification.maximum_direction_norm_error << ",\n";
    output << "    \"maximum_speed_over_c\": "
           << verification.maximum_speed_over_c << "\n";
    output << "  },\n";
    output << "  \"radio_replay\": {\n";
    output << "    \"lepton_tracks\": "
           << radio.lepton_tracks << ",\n";
    output << "    \"track_observer_pairs\": "
           << radio.track_observer_pairs << ",\n";
    output << "    \"coreas_contributions\": "
           << radio.coreas_contributions << ",\n";
    output << "    \"zhs_contributions\": "
           << radio.zhs_contributions << ",\n";
    output << "    \"zhs_subtracks\": "
           << radio.zhs_subtracks << ",\n";
    output << "    \"fixed_point_overflows\": "
           << radio.fixed_point_overflows << ",\n";
    output << "    \"host_to_device_bytes\": "
           << radio.host_to_device_bytes << ",\n";
    output << "    \"device_to_host_bytes\": "
           << radio.device_to_host_bytes << ",\n";
    output << "    \"device_time_ms\": "
           << radio.device_time_ms << ",\n";
    output << "    \"tracks_by_shower\": {";
    bool first = true;
    for (auto const& [shower, count] : tracks_by_shower) {
      output << (first ? "\n" : ",\n");
      first = false;
      output << "      \"" << shower << "\": " << count;
    }
    if (!tracks_by_shower.empty()) { output << "\n    "; }
    output << "}\n";
    output << "  },\n";
    output << "  \"accepted\": "
           << (verification.byte_hash_mismatches == 0 &&
                       verification.invalid_records == 0 &&
                       radio.fixed_point_overflows == 0
                   ? "true"
                   : "false")
           << "\n";
    output << "}\n";
  }

} // namespace

int main(int argc, char** argv) {
  Options options{};
  CLI::App app{
      "Replay a scalar CORSIKA transport tape and radio signal on CUDA"};
  app.add_option("--tape", options.tape, "Scalar decision tape")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("--output", options.output, "Replay output directory")
      ->required();
  app.add_option("--device", options.device, "CUDA device index")
      ->check(CLI::NonNegativeNumber);
  app.add_option(
         "--memory-fraction", options.memory_fraction,
         "Fraction of currently free GPU memory available to radio buffers")
      ->check(CLI::Range(0.01, 1.0));
  app.add_option(
         "--fixed-point-field-limit",
         options.fixed_point_field_limit_V_per_m,
         "Deterministic fixed-point waveform range in V/m")
      ->check(CLI::PositiveNumber);
  app.add_option(
      "--deterministic", options.deterministic,
      "Use deterministic checked fixed-point accumulation");
  CLI11_PARSE(app, argc, argv);

  try {
    if (std::filesystem::exists(options.output)) {
      throw std::runtime_error(
          "refusing to overwrite CUDA replay output: " +
          options.output.string());
    }
    std::filesystem::create_directories(options.output);

    auto const tape =
        corsika::validation::CudaDecisionReplayTape::read(options.tape);
    auto const verification =
        corsika::gpu::em::verifyDecisionReplayOnCuda(
            tape.steps, options.device);
    if (verification.byte_hash_mismatches != 0) {
      throw std::runtime_error(
          "CUDA replay did not consume byte-identical scalar records");
    }
    if (verification.invalid_records != 0) {
      std::ostringstream message;
      message
          << "CUDA replay tape contains "
          << verification.invalid_records
          << " invalid transport records"
          << " (nonfinite=" << verification.nonfinite_records
          << ", negative_energy="
          << verification.negative_energy_records
          << ", negative_weight="
          << verification.negative_weight_records
          << ", backward_time="
          << verification.backward_time_records
          << ", direction="
          << verification.invalid_direction_records
          << ", superluminal="
          << verification.superluminal_records
          << ", maximum_speed_over_c="
          << std::setprecision(17)
          << verification.maximum_speed_over_c << ")";
      throw std::runtime_error(message.str());
    }

    cudaDeviceProp properties{};
    auto status =
        cudaGetDeviceProperties(&properties, options.device);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("cannot inspect CUDA device: ") +
          cudaGetErrorString(status));
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("cannot query CUDA memory: ") +
          cudaGetErrorString(status));
    }
    auto const memory_budget = static_cast<std::size_t>(
        static_cast<long double>(free_bytes) *
        options.memory_fraction);

    auto radio_config = makeGpuRadioConfig(tape, options);
    corsika::gpu::radio::CudaRadioAccumulator radio;
    radio.initialize(
        radio_config, options.device, memory_budget);
    ReplayWaveformWriter writer{options.output, tape};

    std::map<std::uint64_t, std::vector<LeptonTransportRecord>>
        tracks_by_shower;
    for (auto const& step : tape.steps) {
      if (std::abs(step.pdg) == 11) {
        tracks_by_shower[step.shower].push_back(
            makeRadioTrack(step));
      }
    }
    std::map<std::uint64_t, std::uint64_t> track_counts;
    bool first_shower = true;
    for (auto const& [shower, tracks] : tracks_by_shower) {
      if (!first_shower) { radio.reset(); }
      first_shower = false;
      radio.accumulateLeptonTracksFromHost(tracks);
      auto const waveforms = radio.downloadWaveforms();
      writer.write(shower, waveforms);
      track_counts[shower] = tracks.size();
    }
    writer.close();
    auto const statistics = radio.statistics();
    writeSummary(
        options.output / "replay_summary.json",
        options, tape, verification, statistics,
        properties, track_counts);
    std::cout << "CUDA decision replay accepted "
              << verification.records
              << " scalar transport records and projected "
              << verification.lepton_records
              << " lepton records on " << properties.name << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "CUDA decision replay failed: "
              << error.what() << '\n';
    return 1;
  }
}
