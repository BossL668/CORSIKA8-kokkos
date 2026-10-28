/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "OfficialWriterReference.hpp"
#include "ProductionOutputReference.hpp"
#include <array>
#include <boost/filesystem.hpp>
#include <cmath>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <corsika/modules/writers/SubWriter.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <parquet/stream_reader.h>
#include <string>
#include <type_traits>
#include <vector>

namespace {
  using namespace corsika;
#ifndef CORSIKA_OUTPUT_BRIDGE_NATIVE_ONLY
  namespace k = corsika::kokkos;
#endif
  namespace ref = corsika::production_output_reference;
  namespace old = corsika::official_output_reference;
  unsigned checks = 0;
  void require(bool value, std::string const& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
  }
  template <typename TFunction>
  bool throws(TFunction&& function) {
    try {
      function();
    } catch (std::exception const&) { return true; }
    return false;
  }
  struct Directory {
    boost::filesystem::path path_ =
        boost::filesystem::unique_path("output-bridge-%%%%-%%%%");
    Directory() { boost::filesystem::create_directory(path_); }
    ~Directory() {
      boost::system::error_code error;
      boost::filesystem::remove_all(path_, error);
    }
    boost::filesystem::path child(std::string const& name) const {
      auto path = path_ / name;
      boost::filesystem::create_directories(path);
      return path;
    }
  };
  struct EnvironmentFixture {
    Environment<IMediumModel> environment_;
    CoordinateSystemPtr coordinates_ = environment_.getCoordinateSystem();
    std::unique_ptr<ShowerAxis> axis_;
    EnvironmentFixture() {
      auto medium = Environment<IMediumModel>::createNode<Sphere>(
          Point(coordinates_, 0_m, 0_m, 0_m), 100_km);
      medium->setModelProperties<HomogeneousMedium<IMediumModel>>(
          1_kg / (1_m * 1_m * 1_m), NuclearComposition({Code::Nitrogen}, {1.}));
      environment_.getUniverse()->addChild(std::move(medium));
      Point injection(coordinates_, 0_m, 0_m, 1000_m);
      Point ground(coordinates_, 0_m, 0_m, 0_m);
      axis_ = std::make_unique<ShowerAxis>(injection, ground - injection, environment_,
                                           false, 1000);
    }
    Point point(double grammage) const {
      return Point(coordinates_, 0_m, 0_m, (1000. - 10. * grammage) * 1_m);
    }
  };
  using Rows = std::vector<std::vector<double>>;
  template <typename TWriter>
  struct BorrowedWriter final : SubWriter<TWriter> {
    using SubWriter<TWriter>::SubWriter;
    YAML::Node getConfig() const override { return {}; }
  };
  Rows profile_rows(boost::filesystem::path const& path, unsigned columns) {
    parquet::StreamReader reader(parquet::ParquetFileReader::OpenFile(path.string()));
    Rows rows;
    while (!reader.eof()) {
      unsigned id;
      reader >> id;
      std::vector<double> row{static_cast<double>(id)};
      for (unsigned i = 0; i < columns + 1; ++i) {
        float value;
        reader >> value;
        row.push_back(value);
      }
      reader >> parquet::EndRow;
      rows.push_back(row);
    }
    return rows;
  }
  Rows particle_rows(boost::filesystem::path const& path) {
    parquet::StreamReader reader(parquet::ParquetFileReader::OpenFile(path.string()));
    Rows rows;
    while (!reader.eof()) {
      unsigned id;
      int pid;
      float energy, x, y, nx, ny, nz, weight;
      double time;
      reader >> id >> pid >> energy >> x >> y >> nx >> ny >> nz >> time >> weight >>
          parquet::EndRow;
      rows.push_back({double(id), double(pid), energy, x, y, nx, ny, nz, time, weight});
    }
    return rows;
  }
  Rows interaction_rows(boost::filesystem::path const& path) {
    parquet::StreamReader reader(parquet::ParquetFileReader::OpenFile(path.string()));
    Rows rows;
    while (!reader.eof()) {
      unsigned id;
      int pid;
      float x, y, z;
      reader >> id >> pid >> x >> y >> z >> parquet::EndRow;
      rows.push_back({double(id), double(pid), x, y, z});
    }
    return rows;
  }
  std::array<std::array<double, 3>, 10> const cases{{{{10.1, 9.9, 1.25}},
                                                     {{9.9, 10.1, 2.5}},
                                                     {{10., 10., 3.}},
                                                     {{9., 10., 4.}},
                                                     {{10., 9., 5.}},
                                                     {{9., 9.999999, 6.}},
                                                     {{10.000001, 9., 7.}},
                                                     {{0., 30., 0.}},
                                                     {{30., 0., 0.5}},
                                                     {{20., 30., 8.}}}};

  template <typename TWriter>
  YAML::Node exercise_energy(TWriter& writer, EnvironmentFixture const& fixture,
                             boost::filesystem::path const& path) {
    writer.startOfLibrary(path);
    for (unsigned event = 0; event < cases.size(); ++event) {
      writer.startOfShower(event);
      auto const& c = cases[event];
      auto pid = event % 2 ? Code::Electron : Code::MuMinus;
      BorrowedWriter<TWriter> borrowed(writer);
      borrowed.write(fixture.point(c[0]), fixture.point(c[1]), pid, c[2] * 1_GeV);
      require(std::abs(writer.getEnergyLost() / 1_GeV - c[2]) < 1.e-12,
              "energy integral lost or weighted twice");
      writer.endOfShower(event);
    }
    auto summary = writer.getSummary();
    writer.endOfLibrary();
    return summary;
  }
  template <typename TWriter>
  Rows exercise_longitudinal(TWriter& writer, EnvironmentFixture const& fixture,
                             boost::filesystem::path const& path) {
    writer.startOfLibrary(path);
    writer.startOfShower(0);
    std::array<Code, 7> const ids{{Code::Photon, Code::Electron, Code::Positron,
                                   Code::MuMinus, Code::MuPlus, Code::Proton,
                                   Code::Neutron}};
    for (auto pid : ids)
      for (auto const& c : cases)
        writer.write(fixture.point(c[0]), fixture.point(c[1]), pid, c[2]);
    writer.endOfShower(0);
    writer.endOfLibrary();
    return profile_rows(path / "profile.parquet", number_profile::NColumns);
  }
  void writer_conventions() {
    Directory directory;
    EnvironmentFixture fixture;
    auto const bin = 10_g / square(1_cm);
    auto const threshold = 1_g / square(1_cm);
    EnergyLossWriter<> native(*fixture.axis_, 16, bin, threshold);
    old::EnergyLossWriter<> official(*fixture.axis_, 16, bin, threshold);
    EnergyLossWriter<> migrated(*fixture.axis_, 16, bin, threshold,
                                ShortTrackDepositMode::LowerGrammage);
    ref::EnergyLossWriter<> production(*fixture.axis_, 16, bin, threshold);
    auto a = directory.child("native"), b = directory.child("official");
    auto c = directory.child("migrated"), d = directory.child("production");
    auto ns = exercise_energy(native, fixture, a);
    auto os = exercise_energy(official, fixture, b);
    auto ms = exercise_energy(migrated, fixture, c);
    auto ps = exercise_energy(production, fixture, d);
    require(profile_rows(a / "dEdX.parquet", 1) == profile_rows(b / "dEdX.parquet", 1),
            "native StartPoint differs from frozen official writer");
    require(profile_rows(c / "dEdX.parquet", 1) == profile_rows(d / "dEdX.parquet", 1),
            "explicit LowerGrammage differs from frozen production writer");
    require(profile_rows(a / "dEdX.parquet", 1) != profile_rows(c / "dEdX.parquet", 1),
            "fixture must expose the historical upstream tiny-track distinction");
    require(YAML::Dump(ms) == YAML::Dump(ps), "production per-event energy summary");
    for (unsigned event = 0; event < cases.size(); ++event) {
      auto key = "shower_" + std::to_string(event);
      require(ns[key]["sum_dEdX"].as<double>() == os[key]["sum_dEdX"].as<double>(),
              "official per-event energy sum");
    }
    LongitudinalWriter<> defaultWriter(*fixture.axis_, bin);
    old::LongitudinalWriter<> officialProfile(*fixture.axis_, bin);
    require(defaultWriter.getCrossingMode() == ProfileCrossingMode::OriginalC8,
            "two-argument native constructor changed mode");
    LongitudinalWriter<> axisOnly(*fixture.axis_);
    LongitudinalWriter<> explicitBins(*fixture.axis_, std::size_t(16));
    LongitudinalWriter<> explicitBinning(*fixture.axis_, std::size_t(16), bin);
    require(axisOnly.getCrossingMode() == ProfileCrossingMode::OriginalC8 &&
                explicitBins.getCrossingMode() == ProfileCrossingMode::OriginalC8 &&
                explicitBinning.getCrossingMode() == ProfileCrossingMode::OriginalC8,
            "native constructor overload changed default mode");
    require(exercise_longitudinal(defaultWriter, fixture, directory.child("default")) ==
                exercise_longitudinal(officialProfile, fixture,
                                      directory.child("old-profile")),
            "native exact-endpoint profile differs");
    unsigned i = 0;
    for (auto mode : {ProfileCrossingMode::Both, ProfileCrossingMode::Forward,
                      ProfileCrossingMode::OriginalC8}) {
      LongitudinalWriter<> actual(*fixture.axis_, 16, bin, mode);
      ref::LongitudinalWriter<> expected(*fixture.axis_, 16, bin, mode);
      require(exercise_longitudinal(actual, fixture,
                                    directory.child("mode" + std::to_string(i))) ==
                  exercise_longitudinal(expected, fixture,
                                        directory.child("ref-mode" + std::to_string(i))),
              "explicit crossing mode changed production rows");
      ++i;
    }
    // The delegating energy constructor must carry the policy too; SubWriter
    // keeps borrowing the selected writer, as CPU fallback does in production.
    EnergyLossWriter<> nativeDelegating(*fixture.axis_, bin, threshold);
    old::EnergyLossWriter<> officialDelegating(*fixture.axis_, bin, threshold);
    EnergyLossWriter<> productionDelegating(*fixture.axis_, bin, threshold,
                                            ShortTrackDepositMode::LowerGrammage);
    ref::EnergyLossWriter<> referenceDelegating(*fixture.axis_, bin, threshold);
    auto n = directory.child("native-delegating"),
         o = directory.child("official-delegating");
    auto p = directory.child("production-delegating"),
         r = directory.child("reference-delegating");
    exercise_energy(nativeDelegating, fixture, n);
    exercise_energy(officialDelegating, fixture, o);
    exercise_energy(productionDelegating, fixture, p);
    exercise_energy(referenceDelegating, fixture, r);
    require(profile_rows(n / "dEdX.parquet", 1) == profile_rows(o / "dEdX.parquet", 1),
            "delegating StartPoint constructor");
    require(profile_rows(p / "dEdX.parquet", 1) == profile_rows(r / "dEdX.parquet", 1),
            "delegating LowerGrammage constructor");
    for (unsigned thresholdIndex = 0; thresholdIndex < 2; ++thresholdIndex) {
      auto thresholdValue = thresholdIndex * 1_g / square(1_cm);
      EnergyLossWriter<> actual(*fixture.axis_, 16, bin, thresholdValue);
      ref::EnergyLossWriter<> expected(*fixture.axis_, 16, bin, thresholdValue);
      auto ap = directory.child("projected-actual" + std::to_string(thresholdIndex));
      auto ep = directory.child("projected-reference" + std::to_string(thresholdIndex));
      actual.startOfLibrary(ap);
      expected.startOfLibrary(ep);
      actual.startOfShower(0);
      expected.startOfShower(0);
      for (auto const& pair : std::array<std::array<double, 2>, 5>{{{{-0.2, 0.2}},
                                                                    {{10.1, 9.9}},
                                                                    {{1., 80.}},
                                                                    {{80., 1.}},
                                                                    {{999., 998.}}}}) {
        actual.writeProjected(pair[0] * 1_g / square(1_cm), pair[1] * 1_g / square(1_cm),
                              Code::Electron, 2_GeV);
        expected.writeProjected(pair[0] * 1_g / square(1_cm),
                                pair[1] * 1_g / square(1_cm), Code::Electron, 2_GeV);
      }
      actual.endOfShower(0);
      expected.endOfShower(0);
      actual.endOfLibrary();
      expected.endOfLibrary();
      require(
          profile_rows(ap / "dEdX.parquet", 1) == profile_rows(ep / "dEdX.parquet", 1),
          "projected clamping/zero-threshold production contract");
    }
  }

#ifndef CORSIKA_OUTPUT_BRIDGE_NATIVE_ONLY
  // A radio-call recorder checks the adapter's real Step facade. Waveform
  // accumulation uses actual C8 TimeDomainObservers; projection is covered by
  // the previously accepted CoREAS/ZHS module tests, not reimplemented here.
  struct Radio {
    ObserverCollection<TimeDomainObserver> observers_;
    Rows steps_;
    explicit Radio(CoordinateSystemPtr coordinates) {
      observers_.addObserver(TimeDomainObserver("antenna",
                                                Point(coordinates, 0_m, 0_m, 0_m),
                                                coordinates, 0_ns, 3_ns, 1_GHz, 0_ns));
    }
    auto& observerCollection() { return observers_; }
    template <typename TParticle>
    ProcessReturn doContinuous(Step<TParticle> const& step, bool limit) {
      steps_.push_back({double(static_cast<int>(get_PDG(step.getParticlePre().getPID()))),
                        step.getEkinPre() / 1_GeV, step.getEkinPost() / 1_GeV,
                        step.getTimePre() / 1_s, step.getTimePost() / 1_s,
                        step.getParticlePre().getWeight(), double(limit),
                        step.getDisplacement().getNorm() / 1_m});
      return ProcessReturn::Ok;
    }
  };
  struct NoTracking {};
  struct Result {
    std::array<Rows, 5> rows_;
    std::array<std::string, 3> summaries_;
    std::vector<double> counters_, waveform_;
    Rows radio_;
  };
  template <bool Production>
  Result exercise_sink(EnvironmentFixture& fixture, boost::filesystem::path const& path) {
    using Energy =
        std::conditional_t<Production, ref::EnergyLossWriter<>, EnergyLossWriter<>>;
    using Long =
        std::conditional_t<Production, ref::LongitudinalWriter<>, LongitudinalWriter<>>;
    using Prod =
        std::conditional_t<Production, ref::ProductionWriter<>, ProductionWriter<>>;
    using Obs = std::conditional_t<Production, ref::ObservationPlane<NoTracking>,
                                   ObservationPlane<NoTracking>>;
    using Interaction =
        std::conditional_t<Production,
                           ref::InteractionWriter<NoTracking, ParticleWriterParquet>,
                           InteractionWriter<NoTracking, ParticleWriterParquet>>;
    auto energy = [&]() {
      if constexpr (Production)
        return Energy(*fixture.axis_, 16, 10_g / square(1_cm), 1_g / square(1_cm));
      else
        return Energy(*fixture.axis_, 16, 10_g / square(1_cm), 1_g / square(1_cm),
                      ShortTrackDepositMode::LowerGrammage);
    }();
    Long longitudinal(*fixture.axis_, 16, 10_g / square(1_cm), ProfileCrossingMode::Both);
    Prod production(*fixture.axis_, 16, 10_g / square(1_cm));
    auto cs = fixture.coordinates_;
    Obs observation(Plane(Point(cs, 0_m, 0_m, 0_m), DirectionVector(cs, {0., 0., 1.})),
                    DirectionVector(cs, {1., 0., 0.}), true, false);
    Interaction interaction(*fixture.axis_, observation);
    Radio coreas(cs), zhs(cs);
    using Sink = std::conditional_t<
        Production,
        ref::CorsikaOutputSink<Energy, Long, Prod, Obs, Interaction, Radio, Radio>,
        k::CorsikaOutputSink<Energy, Long, Prod, Obs, Interaction, Radio, Radio>>;
    Sink sink(cs, energy, longitudinal, production, observation, interaction, coreas, zhs,
              true, false);
    for (auto name :
         {"energy", "longitudinal", "production", "observation", "interaction"})
      boost::filesystem::create_directories(path / name);
    energy.startOfLibrary(path / "energy");
    longitudinal.startOfLibrary(path / "longitudinal");
    production.startOfLibrary(path / "production");
    observation.startOfLibrary(path / "observation");
    interaction.startOfLibrary(path / "interaction");
    for (unsigned event = 0; event < 2; ++event) {
      energy.startOfShower(event);
      longitudinal.startOfShower(event);
      production.startOfShower(event);
      observation.startOfShower(event);
      interaction.startOfShower(event);
      k::StepRecord step{};
      step.pid_ = 11;
      step.startPositionM_[2] = 950.;
      step.endPositionM_[2] = 750.;
      step.startTimeS_ = 1.e-6;
      step.endTimeS_ = 2.e-6;
      step.startEnergyGeV_ = 10.;
      step.endEnergyGeV_ = 8.;
      step.depositedEnergyGeV_ = 2.;
      step.cutDepositedEnergyGeV_ = 0.5;
      step.weight_ = 2.;
      sink.onStep(step);
      k::ProjectedStepRecord projected{};
      projected.pid_ = -11;
      projected.startGrammageGPerCm2_ = 25.;
      projected.endGrammageGPerCm2_ = 35.;
      projected.endEnergyGeV_ = 2.;
      projected.depositedEnergyGeV_ = 1.;
      projected.cutDepositedEnergyGeV_ = 0.25;
      projected.weight_ = 3.;
      sink.onProjectedStep(projected);
      k::StepRecord cut{};
      cut.pid_ = 13;
      cut.startEnergyGeV_ = cut.endEnergyGeV_ = 0.1;
      cut.startPositionM_[2] = cut.endPositionM_[2] = 900.;
      cut.depositedEnergyGeV_ = cut.cutDepositedEnergyGeV_ =
          0.1 - k::transport_mass_gev(13);
      cut.nativeMuonPairCut_ = 1;
      cut.weight_ = 4.;
      sink.onStep(cut);
      auto invalidStep = step;
      invalidStep.weight_ = -1.;
      require(throws([&] { sink.onStep(invalidStep); }), "negative step weight accepted");
      invalidStep = cut;
      invalidStep.nativeMuonPairCut_ = 0;
      require(throws([&] { sink.onStep(invalidStep); }),
              "untagged negative deposit accepted");
      invalidStep = cut;
      invalidStep.endPositionM_[0] = 1.;
      require(throws([&] { sink.onStep(invalidStep); }),
              "moving negative native cut accepted");
      auto invalidProjected = projected;
      invalidProjected.startGrammageGPerCm2_ = -1.;
      require(throws([&] { sink.onProjectedStep(invalidProjected); }),
              "negative projected depth accepted");
      k::ProfileResult profile{};
      for (auto* vector :
           {&profile.photons_, &profile.electrons_, &profile.positrons_,
            &profile.muonsMinus_, &profile.muonsPlus_, &profile.muonParentProductions_,
            &profile.energyLossGeV_, &profile.muonEnergyLossGeV_})
        vector->assign(16, 0.);
      profile.photons_[1] = 2.;
      profile.electrons_[2] = 3.;
      profile.positrons_[3] = 4.;
      profile.muonsMinus_[4] = 5.;
      profile.muonsPlus_[5] = 6.;
      profile.muonParentProductions_[2] = 7.;
      profile.energyLossGeV_[2] = 8.;
      profile.muonEnergyLossGeV_[2] = 3.;
      profile.steps_ = 9;
      profile.depositedSteps_ = 2;
      profile.weightedDepositedEnergyGeV_ = 8.;
      sink.onGpuProfile(profile);
      auto invalidProfile = profile;
      invalidProfile.fixedPointOverflows_ = 1;
      require(throws([&] { sink.onGpuProfile(invalidProfile); }),
              "overflowed profile accepted");
      invalidProfile = profile;
      invalidProfile.photons_.pop_back();
      require(throws([&] { sink.onGpuProfile(invalidProfile); }),
              "mismatched profile shape accepted");
      require(production.getBin(
                  2)[static_cast<int>(production_profile::ProjectileIndex::Muon)] == 7.,
              "muon parent production column missing");
      require(production.getBin(
                  2)[static_cast<int>(production_profile::ProjectileIndex::All)] == 7.,
              "muon parent all column missing");
      k::ObservationRecord observed{};
      observed.particle_.pid_ = 13;
      observed.particle_.energyGeV_ = 5.;
      observed.particle_.direction_[2] = -2.;
      observed.particle_.weight_ = 3.;
      observed.particle_.positionM_[0] = 2.;
      observed.particle_.timeS_ = 3.e-6;
      sink.onObservation(observed);
      observed.status_ = k::ObservationStatus::EscapedEnvironment;
      sink.onObservation(observed);
      auto badObservation = observed;
      badObservation.status_ = k::ObservationStatus::ReachedObservationSurface;
      badObservation.particle_.energyGeV_ = 0.01;
      require(throws([&] { sink.onObservation(badObservation); }),
              "sub-mass observation accepted");
      // The unchanged native continuous-process entry point shares the new
      // writer hook; exercise it as well as the device record entry point.
      k::output_detail::RadioTrackParticle nativeParticle(
          Code::MuMinus, 4_GeV, DirectionVector(cs, {0., 0., -1.}),
          Point(cs, 0_m, 0_m, 1_m), 0_ns, 2.);
      Line nativeLine(nativeParticle.getPosition(),
                      DirectionVector(cs, {0., 0., -1.}) * (1_m / 1_ns));
      StraightTrajectory nativeTrajectory(nativeLine, 1_ns);
      Step<k::output_detail::RadioTrackParticle> nativeStep(nativeParticle,
                                                            nativeTrajectory);
      require(
          observation.doContinuous(nativeStep, true) == ProcessReturn::ParticleAbsorbed,
          "native observation path changed absorption");
      k::FirstInteractionRecord first{};
      first.parentAtVertex_.pid_ = 22;
      first.parentAtVertex_.energyGeV_ = 10.;
      first.parentAtVertex_.positionM_[2] = 800.;
      first.parentAtVertex_.direction_[2] = -1.;
      first.secondaryCount_ = 2;
      for (unsigned child = 0; child < 2; ++child) {
        first.secondaries_[child].pid_ = child ? -11 : 11;
        first.secondaries_[child].energyGeV_ = child ? 4. : 6.;
        first.secondaries_[child].direction_[2] = -1.;
      }
      sink.onFirstInteraction(first);
      sink.onFirstInteraction(first);
      require(interaction.getInteractionCounter() == 2,
              "first-interaction postincrement contract");
      auto badFirst = first;
      badFirst.parentAtVertex_.generation_ = 1;
      require(throws([&] { sink.onFirstInteraction(badFirst); }),
              "non-primary first-interaction accepted");
      k::RadioTrackRecord radio{};
      radio.step_ = step;
      radio.startDirection_[2] = -1.;
      radio.endDirection_[2] = -1.;
      sink.onRadioTrack(radio);
      radio.step_.pid_ = 13;
      sink.onRadioTrack(radio);
      energy.endOfShower(event);
      longitudinal.endOfShower(event);
      production.endOfShower(event);
      observation.endOfShower(event);
      interaction.endOfShower(event);
    }
    require(observation.statistics().particles_ == 4,
            "observation counters must accumulate across showers");
    Sink deviceRadio(cs, energy, longitudinal, production, observation, interaction,
                     coreas, zhs, true, true);
    k::radio::RadioWaveforms waveforms;
    auto count = coreas.observers_.getObservers()[0].getWaveformX().size();
    k::radio::RadioWaveform waveform;
    waveform.x_.assign(count, 1.);
    waveform.y_.assign(count, 2.);
    waveform.z_.assign(count, -3.);
    waveforms.coreas_.push_back(waveform);
    waveforms.zhs_.push_back(waveform);
    deviceRadio.onGpuRadioWaveforms(waveforms, 12);
    deviceRadio.onGpuRadioWaveforms(waveforms, 3);
    require(throws([&] { sink.onGpuRadioWaveforms(waveforms, 1); }),
            "device waveforms accepted with host radio mode");
    auto badWaveforms = waveforms;
    badWaveforms.zhs_.clear();
    require(throws([&] { deviceRadio.onGpuRadioWaveforms(badWaveforms, 1); }),
            "missing ZHS observer accepted");
    for (unsigned component = 0; component < 3; ++component) {
      auto badGrid = waveforms;
      auto& broken = badGrid.coreas_[0];
      auto* values =
          component == 0 ? &broken.x_ : component == 1 ? &broken.y_ : &broken.z_;
      values->pop_back();
      require(throws([&] { deviceRadio.onGpuRadioWaveforms(badGrid, 1); }),
              "waveform bin count differs from native observer grid");
    }
    require(deviceRadio.statistics().radioTracks_ == 15, "device radio track count");
    Result result;
    auto const& stats = sink.statistics();
    result.counters_ = {double(stats.steps_),
                        double(stats.depositedSteps_),
                        double(stats.observations_),
                        double(stats.firstInteractions_),
                        double(stats.radioTracks_),
                        stats.weightedDepositedEnergyGeV_,
                        stats.weightedMuonParentProductions_,
                        observation.statistics().weightedTotalEnergyGeV_};
    require(stats.steps_ == 24 && stats.observations_ == 2 &&
                stats.firstInteractions_ == 2 && stats.radioTracks_ == 2,
            "output records lost or counted twice");
    result.summaries_ = {YAML::Dump(energy.getSummary()),
                         YAML::Dump(production.getSummary()),
                         YAML::Dump(interaction.getSummary())};
    for (auto const* process : {&coreas, &zhs}) {
      auto const& observer = process->observers_.getObservers()[0];
      for (auto value : observer.getWaveformX()) {
        require(value == 2., "waveform accumulation");
        result.waveform_.push_back(value);
      }
      for (auto value : observer.getWaveformY()) {
        require(value == 4., "Y waveform accumulation");
        result.waveform_.push_back(value);
      }
      for (auto value : observer.getWaveformZ()) {
        require(value == -6., "Z waveform accumulation");
        result.waveform_.push_back(value);
      }
    }
    result.radio_ = coreas.steps_;
    require(coreas.steps_ == zhs.steps_, "CoREAS/ZHS track handoff differs");
    energy.endOfLibrary();
    longitudinal.endOfLibrary();
    production.endOfLibrary();
    observation.endOfLibrary();
    interaction.endOfLibrary();
    result.rows_ = {
        profile_rows(path / "energy/dEdX.parquet", 1),
        profile_rows(path / "longitudinal/profile.parquet", number_profile::NColumns),
        profile_rows(path / "production/profile.parquet", production_profile::NColumns),
        particle_rows(path / "observation/particles.parquet"),
        interaction_rows(path / "interaction/interactions.parquet")};
    require(result.rows_[3].size() == 4 && result.rows_[4].size() == 4,
            "real observation/first-interaction Parquet row counts");
    return result;
  }
  void sink_outputs() {
    Directory directory;
    EnvironmentFixture fixture;
    auto got = exercise_sink<false>(fixture, directory.child("actual"));
    auto expected = exercise_sink<true>(fixture, directory.child("production"));
    require(got.rows_ == expected.rows_, "production Parquet rows differ");
    require(got.summaries_ == expected.summaries_, "production YAML summaries differ");
    require(got.counters_ == expected.counters_, "production output counters differ");
    require(got.waveform_ == expected.waveform_,
            "production native observer waveform differs");
    require(got.radio_ == expected.radio_, "production native Step handoff differs");
  }

  template <typename TOutput>
  std::string run_lifecycle(boost::filesystem::path const& path) {
    YAML::Node config;
    config["test"] = "completion";
    TOutput output(config);
    output.startOfLibrary(path);
    output.startOfShower(0);
    output.endOfShower(0);
    require(output.getSummary()["shower_0"]["status"].template as<std::string>() ==
                "in_progress",
            "endOfShower prematurely marked complete");
    YAML::Node counters;
    counters["tracks"] = 9;
    output.recordComplete(0, counters);
    output.startOfShower(1);
    output.recordIncomplete(1, "explicit failure", counters);
    output.startOfShower(2);
    output.endOfShower(2);
    output.writeSummary(path / "summary.yaml");
    auto before = output.getSummary();
    require(!before["shower_2"]["complete"].template as<bool>(),
            "late incomplete marker missing");
    output.endOfLibrary();
    auto result = output.getSummary();
    require(result.size() == 3, "per-event summaries overwritten");
    return YAML::Dump(result);
  }
  void completion_lifecycle() {
    Directory directory;
    auto got = run_lifecycle<k::EmRunOutput>(directory.child("actual"));
    auto expected = run_lifecycle<ref::EmRunOutput>(directory.child("production"));
    require(got == expected, "production completion lifecycle differs");
  }
#endif
  void writer_off() {
    Directory directory;
    EnvironmentFixture fixture;
    EnergyLossWriter<WriterOff> actual(*fixture.axis_);
    ref::EnergyLossWriter<WriterOff> expected(*fixture.axis_);
    auto missing = directory.path_ / "missing";
    require(throws([&] { actual.startOfLibrary(missing); }),
            "writer-off missing directory contract");
    require(throws([&] { expected.startOfLibrary(missing); }),
            "production missing directory contract");
    auto a = directory.child("actual"), b = directory.child("production");
    actual.startOfLibrary(a);
    expected.startOfLibrary(b);
    actual.startOfShower(0);
    expected.startOfShower(0);
    actual.write(fixture.point(5.), Code::Electron, 3_GeV);
    expected.write(fixture.point(5.), Code::Electron, 3_GeV);
    actual.endOfShower(0);
    expected.endOfShower(0);
    require(YAML::Dump(actual.getSummary()) == YAML::Dump(expected.getSummary()),
            "WriterOff summary lost");
    require(!boost::filesystem::exists(a / "dEdX.parquet"),
            "WriterOff wrote a Parquet stream");
    actual.endOfLibrary();
    expected.endOfLibrary();
  }
} // namespace

int main(int argc, char** argv) {
  try {
    logging::set_level(logging::level::warn);
    require(argc == 2, "one test mode is required");
    std::string mode = argv[1];
    if (mode == "writers") writer_conventions();
#ifndef CORSIKA_OUTPUT_BRIDGE_NATIVE_ONLY
    else if (mode == "sink")
      sink_outputs();
    else if (mode == "completion")
      completion_lifecycle();
#endif
    else if (mode == "writer-off")
      writer_off();
    else
      throw std::runtime_error("unknown mode");
    std::cout << "passed " << mode << " checks=" << checks << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
