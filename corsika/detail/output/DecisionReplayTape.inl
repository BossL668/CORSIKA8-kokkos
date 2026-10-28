/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::diagnostics {
  namespace replay_tape_detail {

    template <typename T, typename = void>
    struct HasGetWeight : std::false_type {};

    template <typename T>
    struct HasGetWeight<T, std::void_t<decltype(std::declval<T const&>().getWeight())>>
        : std::true_type {};

    template <typename T>
    double get_weight_or_one(T const& particle) {
      if constexpr (HasGetWeight<T>::value) { return particle.getWeight(); }
      return 1.;
    }

    template <typename T>
    void write_pod(std::ostream& output, T const& value, char const* description) {
      static_assert(std::is_trivially_copyable_v<T>);
      output.write(reinterpret_cast<char const*>(&value), sizeof(value));
      if (!output) {
        throw std::runtime_error(std::string("cannot write CUDA replay ") + description);
      }
    }

    template <typename T>
    void write_vector(std::ostream& output, std::vector<T> const& values,
                      char const* description) {
      static_assert(std::is_trivially_copyable_v<T>);
      if (values.empty()) { return; }
      output.write(reinterpret_cast<char const*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(T)));
      if (!output) {
        throw std::runtime_error(std::string("cannot write CUDA replay ") + description);
      }
    }

    template <typename T>
    T read_pod(std::istream& input, char const* description) {
      static_assert(std::is_trivially_copyable_v<T>);
      T value{};
      input.read(reinterpret_cast<char*>(&value), sizeof(value));
      if (!input) {
        throw std::runtime_error(std::string("cannot read CUDA replay ") + description);
      }
      return value;
    }

    template <typename T>
    std::vector<T> read_vector(std::istream& input, std::uint64_t count,
                               char const* description) {
      static_assert(std::is_trivially_copyable_v<T>);
      if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::runtime_error(std::string("CUDA replay ") + description +
                                 " count overflows host address space");
      }
      std::vector<T> values(static_cast<std::size_t>(count));
      if (values.empty()) { return values; }
      input.read(reinterpret_cast<char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
      if (!input) {
        throw std::runtime_error(std::string("cannot read CUDA replay ") + description);
      }
      return values;
    }

    template <typename TDetector>
    std::vector<ReplayRadioObserver> make_observers(
        TDetector const& detector, CoordinateSystemPtr const& coordinate_system) {
      std::vector<ReplayRadioObserver> result;
      result.reserve(static_cast<std::size_t>(detector.size()));
      for (auto const& observer : detector.getObservers()) {
        ReplayRadioObserver snapshot{};
        auto const position = observer.getLocation().getCoordinates(coordinate_system);
        for (int axis = 0; axis < 3; ++axis) {
          snapshot.positionM_[axis] = position[axis] / 1_m;
        }
        snapshot.startTimeS_ = observer.getStartTime() / 1_s;
        snapshot.sampleRateHz_ = observer.getSampleRate() / 1_Hz;
        snapshot.numberOfBins_ = observer.getWaveformX().size();
        snapshot.durationS_ = snapshot.numberOfBins_ == 0
                                  ? 0.
                                  : static_cast<double>(snapshot.numberOfBins_ - 1) /
                                        snapshot.sampleRateHz_;
        auto const time_axis = observer.getAxis();
        snapshot.axisStartNs_ =
            time_axis.empty() ? 0. : static_cast<double>(time_axis.front());
        result.push_back(snapshot);
      }
      return result;
    }

  } // namespace replay_tape_detail
  template <typename TEnvironment, typename TCoreasDetector, typename TZhsDetector>
  ReplayRadioSnapshot make_replay_radio_snapshot(TEnvironment const& environment,
                                                 Point const& upper_limit,
                                                 Point const& lower_limit,
                                                 LengthType const step,
                                                 TCoreasDetector const& coreas_detector,
                                                 TZhsDetector const& zhs_detector) {
    if (!(step > 0_m)) {
      throw std::invalid_argument(
          "CUDA replay propagation table requires a positive step");
    }

    ReplayRadioSnapshot result{};
    auto& table = result.propagation_;
    auto const coordinate_system = environment.getCoordinateSystem();
    auto const minimum_x = lower_limit.getCoordinates().getX();
    auto const minimum_y = lower_limit.getCoordinates().getY();
    auto const maximum_height = upper_limit.getCoordinates().getZ();
    auto const minimum_height = lower_limit.getCoordinates().getZ() - 1_km;
    auto const inverse_step = 1 / step;
    auto const number_of_bins =
        static_cast<std::size_t>((maximum_height - minimum_height) * inverse_step + 1);
    if (number_of_bins < 11) {
      throw std::invalid_argument(
          "CUDA replay propagation table requires at least eleven points");
    }

    table.minimumHeightM_ = minimum_height / 1_m;
    table.maximumHeightM_ = maximum_height / 1_m;
    table.stepM_ = step / 1_m;
    table.inverseStepPerM_ = inverse_step * 1_m;
    table.refractivity_.reserve(number_of_bins);
    table.integratedRefractivity_.reserve(number_of_bins);

    auto const* universe = environment.getUniverse().get();
    for (std::size_t index = 0; index < number_of_bins; ++index) {
      Point point{coordinate_system, minimum_x, minimum_y, minimum_height + index * step};
      auto const* node = universe->getContainingNode(point);
      table.refractivity_.push_back(node->getModelProperties().getRefractiveIndex(point) -
                                    1.);
    }

    auto const step_over_meter = inverse_step * 1_m;
    table.integratedRefractivity_.push_back(table.refractivity_.front() *
                                            step_over_meter);
    for (std::size_t index = 1; index < number_of_bins; ++index) {
      table.integratedRefractivity_.push_back(table.integratedRefractivity_.back() +
                                              table.refractivity_[index] *
                                                  step_over_meter);
    }

    table.slopeRefractivityLower_ =
        (table.refractivity_[10] - table.refractivity_.front()) / 10.;
    table.slopeIntegratedRefractivityLower_ =
        (table.integratedRefractivity_[10] - table.integratedRefractivity_.front()) / 10.;
    auto const last = number_of_bins - 1;
    table.slopeRefractivityUpper_ =
        (table.refractivity_[last] - table.refractivity_[last - 10]) / 10.;
    table.slopeIntegratedRefractivityUpper_ =
        (table.integratedRefractivity_[last] - table.integratedRefractivity_[last - 10]) /
        10.;

    result.coreasObservers_ =
        replay_tape_detail::make_observers(coreas_detector, coordinate_system);
    result.zhsObservers_ =
        replay_tape_detail::make_observers(zhs_detector, coordinate_system);
    return result;
  }
  inline DecisionReplayTape& DecisionReplayTape::instance() {
    static DecisionReplayTape tape;
    return tape;
  }
  inline void DecisionReplayTape::open(std::filesystem::path const& path,
                                       ReplayRadioSnapshot const& radio,
                                       std::uint64_t random_seed,
                                       std::uint64_t configured_showers) {
    if (path.empty()) { return; }
    if (stream_.is_open()) {
      throw std::logic_error("CUDA decision replay tape is already open");
    }
    if (std::filesystem::exists(path)) {
      throw std::runtime_error("refusing to overwrite CUDA decision replay tape: " +
                               path.string());
    }
    if (!path.parent_path().empty() &&
        !std::filesystem::is_directory(path.parent_path())) {
      throw std::runtime_error(
          "CUDA decision replay tape parent directory does not exist: " +
          path.parent_path().string());
    }
    stream_.open(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!stream_) {
      throw std::runtime_error("cannot open CUDA decision replay tape: " + path.string());
    }

    header_ = {};
    std::copy(ReplayTapeMagic.begin(), ReplayTapeMagic.end(), header_.magic_);
    header_.formatVersion_ = ReplayTapeVersion;
    header_.headerBytes_ = sizeof(ReplayTapeHeader);
    header_.endianMarker_ = ReplayEndianMarker;
    header_.recordBytes_ = sizeof(ReplayTransportStep);
    header_.configuredShowers_ = configured_showers;
    header_.randomSeed_ = random_seed;
    header_.refractivityCount_ = radio.propagation_.refractivity_.size();
    header_.integratedRefractivityCount_ =
        radio.propagation_.integratedRefractivity_.size();
    header_.coreasObserverCount_ = radio.coreasObservers_.size();
    header_.zhsObserverCount_ = radio.zhsObservers_.size();
    header_.minimumHeightM_ = radio.propagation_.minimumHeightM_;
    header_.maximumHeightM_ = radio.propagation_.maximumHeightM_;
    header_.propagationStepM_ = radio.propagation_.stepM_;
    header_.inversePropagationStepPerM_ = radio.propagation_.inverseStepPerM_;
    header_.slopeRefractivityLower_ = radio.propagation_.slopeRefractivityLower_;
    header_.slopeIntegratedRefractivityLower_ =
        radio.propagation_.slopeIntegratedRefractivityLower_;
    header_.slopeRefractivityUpper_ = radio.propagation_.slopeRefractivityUpper_;
    header_.slopeIntegratedRefractivityUpper_ =
        radio.propagation_.slopeIntegratedRefractivityUpper_;

    replay_tape_detail::write_pod(stream_, header_, "header");
    replay_tape_detail::write_vector(stream_, radio.propagation_.refractivity_,
                                     "refractivity table");
    replay_tape_detail::write_vector(stream_, radio.propagation_.integratedRefractivity_,
                                     "integrated refractivity table");
    replay_tape_detail::write_vector(stream_, radio.coreasObservers_, "CoREAS observers");
    replay_tape_detail::write_vector(stream_, radio.zhsObservers_, "ZHS observers");
    stream_.flush();
    if (!stream_) {
      throw std::runtime_error("cannot flush CUDA decision replay tape header");
    }
    recordCount_ = 0;
    ordinal_ = 0;
    shower_ = 0;
  }
  inline bool DecisionReplayTape::enabled() const noexcept { return stream_.is_open(); }
  inline void DecisionReplayTape::beginShower(std::uint64_t shower) noexcept {
    shower_ = shower;
    ordinal_ = 0;
  }
  inline void DecisionReplayTape::close() {
    if (!stream_.is_open()) { return; }
    header_.recordCount_ = recordCount_;
    stream_.seekp(0, std::ios::beg);
    replay_tape_detail::write_pod(stream_, header_, "finalized header");
    stream_.flush();
    if (!stream_) {
      stream_.close();
      throw std::runtime_error("cannot finalize CUDA decision replay tape");
    }
    stream_.close();
  }
  inline ReplayTapeData DecisionReplayTape::read(std::filesystem::path const& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("cannot open CUDA decision replay tape: " + path.string());
    }
    ReplayTapeData result{};
    result.header_ = replay_tape_detail::read_pod<ReplayTapeHeader>(input, "header");
    if (!std::equal(ReplayTapeMagic.begin(), ReplayTapeMagic.end(),
                    result.header_.magic_)) {
      throw std::runtime_error("CUDA decision replay tape has invalid magic");
    }
    if (result.header_.formatVersion_ != ReplayTapeVersion ||
        result.header_.headerBytes_ != sizeof(ReplayTapeHeader) ||
        result.header_.recordBytes_ != sizeof(ReplayTransportStep) ||
        result.header_.endianMarker_ != ReplayEndianMarker) {
      throw std::runtime_error("CUDA decision replay tape format is incompatible");
    }
    auto& propagation = result.radio_.propagation_;
    propagation.minimumHeightM_ = result.header_.minimumHeightM_;
    propagation.maximumHeightM_ = result.header_.maximumHeightM_;
    propagation.stepM_ = result.header_.propagationStepM_;
    propagation.inverseStepPerM_ = result.header_.inversePropagationStepPerM_;
    propagation.slopeRefractivityLower_ = result.header_.slopeRefractivityLower_;
    propagation.slopeIntegratedRefractivityLower_ =
        result.header_.slopeIntegratedRefractivityLower_;
    propagation.slopeRefractivityUpper_ = result.header_.slopeRefractivityUpper_;
    propagation.slopeIntegratedRefractivityUpper_ =
        result.header_.slopeIntegratedRefractivityUpper_;
    propagation.refractivity_ = replay_tape_detail::read_vector<double>(
        input, result.header_.refractivityCount_, "refractivity table");
    propagation.integratedRefractivity_ = replay_tape_detail::read_vector<double>(
        input, result.header_.integratedRefractivityCount_,
        "integrated refractivity table");
    result.radio_.coreasObservers_ = replay_tape_detail::read_vector<ReplayRadioObserver>(
        input, result.header_.coreasObserverCount_, "CoREAS observers");
    result.radio_.zhsObservers_ = replay_tape_detail::read_vector<ReplayRadioObserver>(
        input, result.header_.zhsObserverCount_, "ZHS observers");
    result.steps_ = replay_tape_detail::read_vector<ReplayTransportStep>(
        input, result.header_.recordCount_, "transport records");
    char extra = 0;
    if (input.read(&extra, 1)) {
      throw std::runtime_error("CUDA decision replay tape has trailing bytes");
    }
    if (!input.eof()) {
      throw std::runtime_error("CUDA decision replay tape ended unexpectedly");
    }
    return result;
  }
  inline DecisionReplayTape::~DecisionReplayTape() {
    try {
      close();
    } catch (...) {}
  }
  template <typename TParticle>
  void DecisionReplayTape::recordTransportStep(Step<TParticle> const& step,
                                               ReplayTransportLimit transport_limit,
                                               ProcessReturn process_return) {
    if (!enabled()) { return; }
    ReplayTransportStep record{};
    record.shower_ = shower_;
    record.ordinal_ = ordinal_++;
    // StackIteratorInterface keeps its physical storage index protected.
    // It is intentionally not promoted to a replay identity: ordinary
    // scalar Stack compaction can change that index in any case.
    record.stackIndex_ = 0;
    auto const pid = step.getParticlePre().getPID();
    record.pdg_ = static_cast<std::int32_t>(get_PDG(pid));
    record.transportLimit_ = transport_limit;
    record.processReturn_ = static_cast<std::int32_t>(process_return);
    auto const start_position = step.getPositionPre().getCoordinates();
    auto const end_position = step.getPositionPost().getCoordinates();
    auto const start_direction = step.getDirectionPre().getComponents();
    auto const end_direction = step.getDirectionPost().getComponents();
    for (int axis = 0; axis < 3; ++axis) {
      record.startPositionM_[axis] = start_position[axis] / 1_m;
      record.endPositionM_[axis] = end_position[axis] / 1_m;
      record.startDirection_[axis] = start_direction[axis].magnitude();
      record.endDirection_[axis] = end_direction[axis].magnitude();
    }
    record.startTimeS_ = step.getTimePre() / 1_s;
    record.endTimeS_ = step.getTimePost() / 1_s;
    auto const mass = get_mass(pid);
    record.startTotalEnergyGeV_ = (step.getEkinPre() + mass) / 1_GeV;
    record.endTotalEnergyGeV_ = (step.getEkinPost() + mass) / 1_GeV;
    record.weight_ = replay_tape_detail::get_weight_or_one(step.getParticlePre());
    replay_tape_detail::write_pod(stream_, record, "transport step");
    ++recordCount_;
  }
} // namespace corsika::diagnostics
