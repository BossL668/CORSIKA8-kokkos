/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/process/ProcessReturn.hpp>
#include <corsika/validation/CudaDecisionReplayTypes.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika::validation {

  /**
   * Why this tape exists
   * --------------------
   *
   * A production CUDA shower uses history-keyed Philox streams and a
   * wavefront scheduler.  A scalar CORSIKA shower uses sequential random
   * streams and a LIFO stack.  Equal seeds therefore do not identify equal
   * events across those two algorithms.
   *
   * This file format is the explicit exception: the scalar run records its
   * already selected transport segments, and a CUDA replay consumer is
   * required to use those exact segments.  It is a debug/validation oracle,
   * not an alternative source of production Monte Carlo events.
   */

  namespace replay_tape_detail {

    template <typename T, typename = void>
    struct HasGetWeight : std::false_type {};

    template <typename T>
    struct HasGetWeight<
        T, std::void_t<decltype(std::declval<T const&>().getWeight())>>
        : std::true_type {};

    template <typename T>
    double getWeightOrOne(T const& particle) {
      if constexpr (HasGetWeight<T>::value) {
        return particle.getWeight();
      }
      return 1.;
    }

    template <typename T>
    void writePod(std::ostream& output, T const& value,
                  char const* description) {
      static_assert(std::is_trivially_copyable_v<T>);
      output.write(reinterpret_cast<char const*>(&value), sizeof(value));
      if (!output) {
        throw std::runtime_error(
            std::string("cannot write CUDA replay ") + description);
      }
    }

    template <typename T>
    void writeVector(std::ostream& output, std::vector<T> const& values,
                     char const* description) {
      static_assert(std::is_trivially_copyable_v<T>);
      if (values.empty()) { return; }
      output.write(
          reinterpret_cast<char const*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(T)));
      if (!output) {
        throw std::runtime_error(
            std::string("cannot write CUDA replay ") + description);
      }
    }

    template <typename T>
    T readPod(std::istream& input, char const* description) {
      static_assert(std::is_trivially_copyable_v<T>);
      T value{};
      input.read(reinterpret_cast<char*>(&value), sizeof(value));
      if (!input) {
        throw std::runtime_error(
            std::string("cannot read CUDA replay ") + description);
      }
      return value;
    }

    template <typename T>
    std::vector<T> readVector(
        std::istream& input, std::uint64_t count,
        char const* description) {
      static_assert(std::is_trivially_copyable_v<T>);
      if (count >
          std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::runtime_error(
            std::string("CUDA replay ") + description +
            " count overflows host address space");
      }
      std::vector<T> values(static_cast<std::size_t>(count));
      if (values.empty()) { return values; }
      input.read(
          reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(T)));
      if (!input) {
        throw std::runtime_error(
            std::string("cannot read CUDA replay ") + description);
      }
      return values;
    }

    template <typename TDetector>
    std::vector<ReplayRadioObserver> makeObservers(
        TDetector const& detector,
        CoordinateSystemPtr const& coordinate_system) {
      std::vector<ReplayRadioObserver> result;
      result.reserve(static_cast<std::size_t>(detector.size()));
      for (auto const& observer : detector.getObservers()) {
        ReplayRadioObserver snapshot{};
        auto const position =
            observer.getLocation().getCoordinates(coordinate_system);
        for (int axis = 0; axis < 3; ++axis) {
          snapshot.position_m[axis] = position[axis] / 1_m;
        }
        snapshot.start_time_s = observer.getStartTime() / 1_s;
        snapshot.sample_rate_Hz = observer.getSampleRate() / 1_Hz;
        snapshot.number_of_bins =
            observer.getWaveformX().size();
        snapshot.duration_s =
            snapshot.number_of_bins == 0
                ? 0.
                : static_cast<double>(
                      snapshot.number_of_bins - 1) /
                      snapshot.sample_rate_Hz;
        auto const time_axis = observer.getAxis();
        snapshot.axis_start_ns =
            time_axis.empty() ? 0. : static_cast<double>(time_axis.front());
        result.push_back(snapshot);
      }
      return result;
    }

  } // namespace replay_tape_detail

  /**
   * Build the replay radio snapshot with the same nearest-bin flat-atmosphere
   * discretization used by TabulatedFlatAtmospherePropagator and the CUDA
   * radio backend.
   */
  template <typename TEnvironment, typename TCoreasDetector,
            typename TZhsDetector>
  ReplayRadioSnapshot makeReplayRadioSnapshot(
      TEnvironment const& environment, Point const& upper_limit,
      Point const& lower_limit, LengthType const step,
      TCoreasDetector const& coreas_detector,
      TZhsDetector const& zhs_detector) {
    if (!(step > 0_m)) {
      throw std::invalid_argument(
          "CUDA replay propagation table requires a positive step");
    }

    ReplayRadioSnapshot result{};
    auto& table = result.propagation;
    auto const coordinate_system = environment.getCoordinateSystem();
    auto const minimum_x = lower_limit.getCoordinates().getX();
    auto const minimum_y = lower_limit.getCoordinates().getY();
    auto const maximum_height = upper_limit.getCoordinates().getZ();
    auto const minimum_height =
        lower_limit.getCoordinates().getZ() - 1_km;
    auto const inverse_step = 1 / step;
    auto const number_of_bins = static_cast<std::size_t>(
        (maximum_height - minimum_height) * inverse_step + 1);
    if (number_of_bins < 11) {
      throw std::invalid_argument(
          "CUDA replay propagation table requires at least eleven points");
    }

    table.minimum_height_m = minimum_height / 1_m;
    table.maximum_height_m = maximum_height / 1_m;
    table.step_m = step / 1_m;
    table.inverse_step_per_m = inverse_step * 1_m;
    table.refractivity.reserve(number_of_bins);
    table.integrated_refractivity.reserve(number_of_bins);

    auto const* universe = environment.getUniverse().get();
    for (std::size_t index = 0; index < number_of_bins; ++index) {
      Point point{coordinate_system, minimum_x, minimum_y,
                  minimum_height + index * step};
      auto const* node = universe->getContainingNode(point);
      table.refractivity.push_back(
          node->getModelProperties().getRefractiveIndex(point) - 1.);
    }

    auto const step_over_meter = inverse_step * 1_m;
    table.integrated_refractivity.push_back(
        table.refractivity.front() * step_over_meter);
    for (std::size_t index = 1; index < number_of_bins; ++index) {
      table.integrated_refractivity.push_back(
          table.integrated_refractivity.back() +
          table.refractivity[index] * step_over_meter);
    }

    table.slope_refractivity_lower =
        (table.refractivity[10] - table.refractivity.front()) / 10.;
    table.slope_integrated_refractivity_lower =
        (table.integrated_refractivity[10] -
         table.integrated_refractivity.front()) /
        10.;
    auto const last = number_of_bins - 1;
    table.slope_refractivity_upper =
        (table.refractivity[last] - table.refractivity[last - 10]) /
        10.;
    table.slope_integrated_refractivity_upper =
        (table.integrated_refractivity[last] -
         table.integrated_refractivity[last - 10]) /
        10.;

    result.coreas_observers =
        replay_tape_detail::makeObservers(
            coreas_detector, coordinate_system);
    result.zhs_observers =
        replay_tape_detail::makeObservers(
            zhs_detector, coordinate_system);
    return result;
  }

  class CudaDecisionReplayTape {
  public:
    static CudaDecisionReplayTape& instance() {
      static CudaDecisionReplayTape tape;
      return tape;
    }

    void open(
        std::filesystem::path const& path,
        ReplayRadioSnapshot const& radio,
        std::uint64_t random_seed,
        std::uint64_t configured_showers) {
      if (path.empty()) { return; }
      if (stream_.is_open()) {
        throw std::logic_error(
            "CUDA decision replay tape is already open");
      }
      if (std::filesystem::exists(path)) {
        throw std::runtime_error(
            "refusing to overwrite CUDA decision replay tape: " +
            path.string());
      }
      if (!path.parent_path().empty() &&
          !std::filesystem::is_directory(path.parent_path())) {
        throw std::runtime_error(
            "CUDA decision replay tape parent directory does not exist: " +
            path.parent_path().string());
      }
      stream_.open(
          path, std::ios::binary | std::ios::in |
                    std::ios::out | std::ios::trunc);
      if (!stream_) {
        throw std::runtime_error(
            "cannot open CUDA decision replay tape: " + path.string());
      }

      header_ = {};
      std::copy(
          CudaReplayTapeMagic.begin(), CudaReplayTapeMagic.end(),
          header_.magic);
      header_.format_version = CudaReplayTapeVersion;
      header_.header_bytes = sizeof(ReplayTapeHeader);
      header_.endian_marker = CudaReplayEndianMarker;
      header_.record_bytes = sizeof(ReplayTransportStep);
      header_.configured_showers = configured_showers;
      header_.random_seed = random_seed;
      header_.refractivity_count =
          radio.propagation.refractivity.size();
      header_.integrated_refractivity_count =
          radio.propagation.integrated_refractivity.size();
      header_.coreas_observer_count =
          radio.coreas_observers.size();
      header_.zhs_observer_count =
          radio.zhs_observers.size();
      header_.minimum_height_m =
          radio.propagation.minimum_height_m;
      header_.maximum_height_m =
          radio.propagation.maximum_height_m;
      header_.propagation_step_m =
          radio.propagation.step_m;
      header_.inverse_propagation_step_per_m =
          radio.propagation.inverse_step_per_m;
      header_.slope_refractivity_lower =
          radio.propagation.slope_refractivity_lower;
      header_.slope_integrated_refractivity_lower =
          radio.propagation.slope_integrated_refractivity_lower;
      header_.slope_refractivity_upper =
          radio.propagation.slope_refractivity_upper;
      header_.slope_integrated_refractivity_upper =
          radio.propagation.slope_integrated_refractivity_upper;

      replay_tape_detail::writePod(
          stream_, header_, "header");
      replay_tape_detail::writeVector(
          stream_, radio.propagation.refractivity,
          "refractivity table");
      replay_tape_detail::writeVector(
          stream_, radio.propagation.integrated_refractivity,
          "integrated refractivity table");
      replay_tape_detail::writeVector(
          stream_, radio.coreas_observers, "CoREAS observers");
      replay_tape_detail::writeVector(
          stream_, radio.zhs_observers, "ZHS observers");
      stream_.flush();
      if (!stream_) {
        throw std::runtime_error(
            "cannot flush CUDA decision replay tape header");
      }
      record_count_ = 0;
      ordinal_ = 0;
      shower_ = 0;
    }

    bool enabled() const noexcept { return stream_.is_open(); }

    void beginShower(std::uint64_t shower) noexcept {
      shower_ = shower;
      ordinal_ = 0;
    }

    template <typename TParticle>
    void recordTransportStep(
        Step<TParticle> const& step,
        ReplayTransportLimit transport_limit,
        ProcessReturn process_return) {
      if (!enabled()) { return; }
      ReplayTransportStep record{};
      record.shower = shower_;
      record.ordinal = ordinal_++;
      // StackIteratorInterface keeps its physical storage index protected.
      // It is intentionally not promoted to a replay identity: ordinary
      // scalar Stack compaction can change that index in any case.
      record.stack_index = 0;
      auto const pid = step.getParticlePre().getPID();
      record.pdg = static_cast<std::int32_t>(get_PDG(pid));
      record.transport_limit = transport_limit;
      record.process_return =
          static_cast<std::int32_t>(process_return);
      auto const start_position =
          step.getPositionPre().getCoordinates();
      auto const end_position =
          step.getPositionPost().getCoordinates();
      auto const start_direction =
          step.getDirectionPre().getComponents();
      auto const end_direction =
          step.getDirectionPost().getComponents();
      for (int axis = 0; axis < 3; ++axis) {
        record.start_position_m[axis] =
            start_position[axis] / 1_m;
        record.end_position_m[axis] =
            end_position[axis] / 1_m;
        record.start_direction[axis] =
            start_direction[axis].magnitude();
        record.end_direction[axis] =
            end_direction[axis].magnitude();
      }
      record.start_time_s = step.getTimePre() / 1_s;
      record.end_time_s = step.getTimePost() / 1_s;
      auto const mass = get_mass(pid);
      record.start_total_energy_GeV =
          (step.getEkinPre() + mass) / 1_GeV;
      record.end_total_energy_GeV =
          (step.getEkinPost() + mass) / 1_GeV;
      record.weight = replay_tape_detail::getWeightOrOne(
          step.getParticlePre());
      replay_tape_detail::writePod(
          stream_, record, "transport step");
      ++record_count_;
    }

    void close() {
      if (!stream_.is_open()) { return; }
      header_.record_count = record_count_;
      stream_.seekp(0, std::ios::beg);
      replay_tape_detail::writePod(
          stream_, header_, "finalized header");
      stream_.flush();
      if (!stream_) {
        stream_.close();
        throw std::runtime_error(
            "cannot finalize CUDA decision replay tape");
      }
      stream_.close();
    }

    static ReplayTapeData read(
        std::filesystem::path const& path) {
      std::ifstream input(path, std::ios::binary);
      if (!input) {
        throw std::runtime_error(
            "cannot open CUDA decision replay tape: " + path.string());
      }
      ReplayTapeData result{};
      result.header = replay_tape_detail::readPod<
          ReplayTapeHeader>(input, "header");
      if (!std::equal(
              CudaReplayTapeMagic.begin(), CudaReplayTapeMagic.end(),
              result.header.magic)) {
        throw std::runtime_error(
            "CUDA decision replay tape has invalid magic");
      }
      if (result.header.format_version != CudaReplayTapeVersion ||
          result.header.header_bytes != sizeof(ReplayTapeHeader) ||
          result.header.record_bytes != sizeof(ReplayTransportStep) ||
          result.header.endian_marker != CudaReplayEndianMarker) {
        throw std::runtime_error(
            "CUDA decision replay tape format is incompatible");
      }
      auto& propagation = result.radio.propagation;
      propagation.minimum_height_m =
          result.header.minimum_height_m;
      propagation.maximum_height_m =
          result.header.maximum_height_m;
      propagation.step_m = result.header.propagation_step_m;
      propagation.inverse_step_per_m =
          result.header.inverse_propagation_step_per_m;
      propagation.slope_refractivity_lower =
          result.header.slope_refractivity_lower;
      propagation.slope_integrated_refractivity_lower =
          result.header.slope_integrated_refractivity_lower;
      propagation.slope_refractivity_upper =
          result.header.slope_refractivity_upper;
      propagation.slope_integrated_refractivity_upper =
          result.header.slope_integrated_refractivity_upper;
      propagation.refractivity =
          replay_tape_detail::readVector<double>(
              input, result.header.refractivity_count,
              "refractivity table");
      propagation.integrated_refractivity =
          replay_tape_detail::readVector<double>(
              input,
              result.header.integrated_refractivity_count,
              "integrated refractivity table");
      result.radio.coreas_observers =
          replay_tape_detail::readVector<ReplayRadioObserver>(
              input, result.header.coreas_observer_count,
              "CoREAS observers");
      result.radio.zhs_observers =
          replay_tape_detail::readVector<ReplayRadioObserver>(
              input, result.header.zhs_observer_count,
              "ZHS observers");
      result.steps =
          replay_tape_detail::readVector<ReplayTransportStep>(
              input, result.header.record_count,
              "transport records");
      char extra = 0;
      if (input.read(&extra, 1)) {
        throw std::runtime_error(
            "CUDA decision replay tape has trailing bytes");
      }
      if (!input.eof()) {
        throw std::runtime_error(
            "CUDA decision replay tape ended unexpectedly");
      }
      return result;
    }

  private:
    CudaDecisionReplayTape() = default;
    ~CudaDecisionReplayTape() {
      try {
        close();
      } catch (...) {
      }
    }

    std::fstream stream_{};
    ReplayTapeHeader header_{};
    std::uint64_t record_count_{};
    std::uint64_t shower_{};
    std::uint64_t ordinal_{};
  };

} // namespace corsika::validation
