/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <catch2/catch_all.hpp>

#include <corsika/modules/radio/RadioProcess.hpp>
#include <corsika/modules/radio/LimitedRadioProcess.hpp>
#include <corsika/modules/radio/ZHS.hpp>
#include <corsika/modules/radio/CoREAS.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/propagators/NumericalIntegratingPropagator.hpp>
#include <corsika/modules/radio/propagators/DummyTestPropagator.hpp>
#include <corsika/modules/radio/propagators/TabulatedFlatAtmospherePropagator.hpp>
#include <corsika/modules/radio/propagators/SignalPath.hpp>
#include <corsika/modules/radio/propagators/RadioPropagator.hpp>

#include <boost/filesystem.hpp>
#include <vector>
#include <istream>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <corsika/media/Environment.hpp>
#include <corsika/media/FlatExponential.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/LayeredSphericalAtmosphereBuilder.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/SlidingPlanarExponential.hpp>
#include <corsika/media/IMediumModel.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <corsika/media/ExponentialRefractiveIndex.hpp>
#include <corsika/media/VolumeTreeNode.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>

#include <corsika/framework/geometry/CoordinateSystem.hpp>
#include <corsika/framework/geometry/Line.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupTrajectory.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/PhysicalConstants.hpp>

#include <corsika/output/OutputManager.hpp>

using namespace corsika;
using Catch::Approx;

double constexpr absMargin = 1.0e-7;

template <typename TInterface>
using MyExtraEnv =
    UniformRefractiveIndex<MediumPropertyModel<UniformMagneticField<TInterface>>>;

template <typename TInterface2>
using MyExtraEnv2 =
    GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<TInterface2>>>;

// Dummy process for testing LimitedRadioProcess
// If it is run, returns ProcessReturn::Interacted
class TestRadioProcess {
public:
  TestRadioProcess() {}

  template <typename Particle>
  ProcessReturn doContinuous(Step<Particle> const&, bool const) {
    return ProcessReturn::Interacted;
  }

  template <typename Particle, typename Track>
  LengthType getMaxStepLength(Particle const&, Track const&) const {
    return 123_m;
  }
};

TEST_CASE("Radio", "[processes]") {

  logging::set_level(logging::level::debug);

  SECTION("CoREAS process") {

    // This serves as a compiler test for any changes in the CoREAS algorithm and
    // check the radio process output

    // Environment
    using IModelInterface =
        IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
    using AtmModel = UniformRefractiveIndex<
        MediumPropertyModel<UniformMagneticField<HomogeneousMedium<IModelInterface>>>>;
    using EnvType = Environment<AtmModel>;
    EnvType envCoREAS;
    CoordinateSystemPtr const& rootCS = envCoREAS.getCoordinateSystem();
    Point const center{rootCS, 0_m, 0_m, 0_m};

    const auto trackStart{Point(rootCS, 7_m, 8_m, 9_m)};
    const auto trackEnd{Point(rootCS, 5_m, 5_m, 10_m)};

    // create an electron
    const Code electron{Code::Electron};
    const auto pmass{get_mass(electron)};

    VelocityVector v0(rootCS, {5e+2_m / second, 5e+2_m / second, 5e+2_m / second});

    Vector B0(rootCS, 5_T, 5_T, 5_T);

    Line const line(trackStart, v0);

    auto const k{1_m * ((1_m) / ((1_s * 1_s) * 1_V))};

    auto const t = 1e-12_s;
    LeapFrogTrajectory base(trackEnd, v0, B0, k, t);

    // create a new stack for each trial
    setup::Stack<EnvType> stack;

    // construct an energy
    const HEPEnergyType E0{1_TeV};

    // compute the necessary momentumn
    const HEPMomentumType P0{sqrt(E0 * E0 - pmass * pmass)};
    const auto plab{MomentumVector(rootCS, {0_GeV, 0_GeV, P0})};

    // and create the location of the particle in this coordinate system
    const Point pos(rootCS, 50_m, 10_m, 80_m);

    // add the particle to the stack
    auto particle1{stack.addParticle(std::make_tuple(
        electron, calculate_kinetic_energy(plab.getNorm(), get_mass(electron)),
        plab.normalized(), pos, 0_ns))};

    Vector B1(rootCS, 0_T, 0_T, 1_T);
    NuclearComposition const protonComposition({Code::Proton}, {1.});
    const double refractiveIndex{1.000327};
    const auto density{1_g / cube(1_cm)};
    auto Medium = EnvType::createNode<Sphere>(
        center, 10_km * std::numeric_limits<double>::infinity());
    auto const props = Medium->setModelProperties<AtmModel>(
        refractiveIndex, Medium::AirDry1Atm, B1, density, protonComposition);
    particle1.setNode(Medium.get());
    envCoREAS.getUniverse()->addChild(std::move(Medium));

    // create the detector
    const auto obs1Loc{Point(rootCS, 100_m, 2_m, 3_m)};
    const auto obs2Loc{Point(rootCS, 4_m, 80_m, 6_m)};
    const TimeType t1{0_s};
    const TimeType t2{10_s};
    const InverseTimeType t3{1e+3_Hz};
    TimeDomainObserver obs1("observer_name", obs1Loc, rootCS, t1, t2, t3, t1);
    TimeDomainObserver obs2("observer_name2", obs2Loc, rootCS, t1, t2, t3, t1);
    ObserverCollection<TimeDomainObserver> detector;
    detector.addObserver(obs1);
    detector.addObserver(obs2);

    auto SP = make_numerical_integrating_radio_propagator(envCoREAS, 1_m);
    // create a radio process instance using CoREAS
    RadioProcess<decltype(detector), CoREAS<decltype(detector), decltype(SP)>,
                 decltype(SP)>
        coreas(detector, SP);

    Step step(particle1, base);
    // check doContinuous and simulate methods
    auto const result = coreas.doContinuous(step, true);
    REQUIRE(ProcessReturn::Ok == result);

    for (auto const& obs : detector.getObservers()) {
      // make sure something was put into the observer
      auto totalX = obs.getWaveformX()[0];
      auto totalY = obs.getWaveformY()[0];
      auto totalZ = obs.getWaveformZ()[0];
      for (size_t i = 0; i < obs.getWaveformX().size(); i++) {
        totalX += obs.getWaveformX()[i];
        totalY += obs.getWaveformY()[i];
        totalZ += obs.getWaveformZ()[i];
      }
      REQUIRE((totalX + totalY + totalZ) > (totalX * 0));
    }

    //////////////////////////////////////
    // reset everything for a new particle
    //////////////////////////////////////
    obs1.reset();
    obs2.reset();
    stack.purge();

    // add the particle to the stack that is VERY late
    auto const particle2{stack.addParticle(std::make_tuple(
        electron, calculate_kinetic_energy(plab.getNorm(), get_mass(electron)),
        plab.normalized(), pos, t1 + t2 * 100000))};
    Step step2(particle2, base);
    auto const result2 = coreas.doContinuous(step2, true);
    REQUIRE(ProcessReturn::Ok == result2);
    for (auto const& obs : detector.getObservers()) {
      // make sure something was put into the observer
      auto total = obs.getWaveformX()[0];
      for (size_t i = 0; i < obs.getWaveformX().size(); i++) {
        total += obs.getWaveformX()[i] * obs.getWaveformX()[i];
        total += obs.getWaveformY()[i] * obs.getWaveformY()[i];
        total += obs.getWaveformZ()[i] * obs.getWaveformZ()[i];
      }
      REQUIRE(total < (1e-12 * obs.getWaveformX().size()));
    }

    // coreas output check
    std::string const implemencoreas{"CoREAS"};

    boost::filesystem::path const tempPathC{boost::filesystem::temp_directory_path() /
                                            ("test_corsika_radio_" + implemencoreas)};
    if (boost::filesystem::exists(tempPathC)) {
      boost::filesystem::remove_all(tempPathC);
    }

    boost::filesystem::create_directory(tempPathC);
    coreas.startOfLibrary(tempPathC);
    auto const outputFileC = tempPathC / ("observers.parquet");
    CHECK(boost::filesystem::exists(outputFileC));
    // run end of shower and make sure that something extra was added
    auto const fileSizeC = boost::filesystem::file_size(outputFileC);
    coreas.endOfShower(0);
    CHECK(boost::filesystem::file_size(outputFileC) > fileSizeC);
    coreas.endOfLibrary();

  } // END: SECTION("CoREAS process")

  SECTION("CoREAS Edge Cases") {
    using IModelInterface =
        IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
    using AtmModel = UniformRefractiveIndex<
        MediumPropertyModel<UniformMagneticField<HomogeneousMedium<IModelInterface>>>>;
    using EnvType = Environment<AtmModel>;
    EnvType envCoREAS;
    CoordinateSystemPtr const& rootCS = envCoREAS.getCoordinateSystem();
    Point const center{rootCS, 0_m, 0_m, 0_m};

    const auto trackStart{Point(rootCS, 7_m, 8_m, 9_m)};
    const auto trackEnd{Point(rootCS, 5_m, 5_m, 10_m)};

    // create an electron
    const Code electron{Code::Electron};
    const auto pmass{get_mass(electron)};

    VelocityVector v0(rootCS, {1_m / second, 0_m / second, 0_m / second});

    Vector B0(rootCS, 5_T, 5_T, 5_T);

    Line const line(trackStart, v0);

    // create a new stack for each trial
    setup::Stack<EnvType> stack;

    // construct an energy
    const HEPEnergyType E0{1_TeV};

    // compute the necessary momentumn
    const HEPMomentumType P0{sqrt(E0 * E0 - pmass * pmass)};
    const auto plab{MomentumVector(rootCS, {0_GeV, 0_GeV, P0})};

    // and create the location of the particle in this coordinate system
    const Point pos(rootCS, 50_m, 10_m, 80_m);

    // add the particle to the stack
    auto particle1{stack.addParticle(std::make_tuple(
        electron, calculate_kinetic_energy(plab.getNorm(), get_mass(electron)),
        plab.normalized(), pos, 0_ns))};

    Vector B1(rootCS, 0_T, 0_T, 1_T);
    NuclearComposition const protonComposition({Code::Proton}, {1.});
    const double refractiveIndex{1.000327};
    const auto density{1_g / cube(1_cm)};
    auto Medium = EnvType::createNode<Sphere>(
        center, 10_km * std::numeric_limits<double>::infinity());
    auto const props = Medium->setModelProperties<AtmModel>(
        refractiveIndex, Medium::AirDry1Atm, B1, density, protonComposition);
    particle1.setNode(Medium.get());
    envCoREAS.getUniverse()->addChild(std::move(Medium));

    // create the detector
    const auto obs1Loc{Point(rootCS, 100_m, 2_m, 3_m)};
    const auto obs2Loc{Point(rootCS, 4_m, 80_m, 6_m)};
    const TimeType t1{0_s};
    const TimeType t2{10_s};
    const InverseTimeType t3{1e+3_Hz};
    TimeDomainObserver obs1("observer_name", obs1Loc, rootCS, t1, t2, t3, t1);
    TimeDomainObserver obs2("observer_name2", obs2Loc, rootCS, t1, t2, t3, t1);
    ObserverCollection<TimeDomainObserver> detector;
    detector.addObserver(obs1);
    detector.addObserver(obs2);

    auto SP = make_numerical_integrating_radio_propagator(envCoREAS, 1_m);

    // create a radio process instance using CoREAS
    RadioProcess<decltype(detector), CoREAS<decltype(detector), decltype(SP)>,
                 decltype(SP)>
        coreas(detector, SP);

    auto result = coreas.doContinuous(
        Step(particle1, StraightTrajectory(line, 0_ns, 0_ns, v0, v0)), true);
    REQUIRE(ProcessReturn::Ok == result);
    result = coreas.doContinuous(
        Step(particle1, StraightTrajectory(line, 0_ns, 1_ns, v0, v0)), true);
    REQUIRE(ProcessReturn::Ok == result);
    result = coreas.doContinuous(
        Step(particle1, StraightTrajectory(line, 0_ns, -1_ns, v0, v0)), true);
    REQUIRE(ProcessReturn::Ok == result);
    result = coreas.doContinuous(
        Step(particle1, StraightTrajectory(line, 1_ns, -1_ns, v0, v0)), true);
    REQUIRE(ProcessReturn::Ok == result);
    result = coreas.doContinuous(
        Step(particle1, StraightTrajectory(line, -1_ns, 1_ns, v0, v0)), true);
    REQUIRE(ProcessReturn::Ok == result);
    result = coreas.doContinuous(
        Step(particle1, StraightTrajectory(line, -1_ns, 1_ns, v0, -v0)), true);
    REQUIRE(ProcessReturn::Ok == result);

    // Use ZHS-like loop
    auto const vParallel =
        VelocityVector(rootCS, {0_m / second, 1_m / second, 0_m / second});
    result = coreas.doContinuous(
        Step(particle1, StraightTrajectory(Line(trackStart, vParallel), 0_ns, 1_ns,
                                           vParallel, vParallel)),
        true);
    REQUIRE(ProcessReturn::Ok == result);
  } // END: SECTION("CoREAS Edge Cases")

  SECTION("ZHS process") {

    // This section serves as a compiler test for any changes in the ZHS algorithm
    // Environment
    using IModelInterface =
        IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
    using AtmModel = UniformRefractiveIndex<
        MediumPropertyModel<UniformMagneticField<HomogeneousMedium<IModelInterface>>>>;
    using EnvType = Environment<AtmModel>;
    EnvType envZHS;
    CoordinateSystemPtr const& rootCS = envZHS.getCoordinateSystem();
    // get the center point
    Point const center{rootCS, 0_m, 0_m, 0_m};

    // create a particle
    auto const particle{Code::Electron};
    const auto pmass{get_mass(particle)};

    VelocityVector v0(rootCS, {5e+2_m / second, 5e+2_m / second, 5e+2_m / second});

    Vector B0(rootCS, 5_T, 5_T, 5_T);

    // the observers location
    const auto trackStart{Point(envZHS.getCoordinateSystem(), 7_m, 8_m, 9_m)};
    const auto trackEnd{Point(envZHS.getCoordinateSystem(), 5_m, 5_m, 10_m)};

    Line const line(trackStart, v0);

    auto const k{1_m * ((1_m) / ((1_s * 1_s) * 1_V))};

    auto const t = 1e-12_s;
    LeapFrogTrajectory base(trackEnd, v0, B0, k, t);

    // create a new stack for each trial
    setup::Stack<EnvType> stack;

    // construct an energy
    const HEPEnergyType E0{1_TeV};

    // compute the necessary momentum
    const HEPMomentumType P0{sqrt(E0 * E0 - pmass * pmass)};
    const auto plab{MomentumVector(rootCS, {0_GeV, 0_GeV, P0})};

    // and create the location of the particle in this coordinate system
    const Point pos(rootCS, 50_m, 10_m, 80_m);

    // add the particle to the stack
    auto particle1{stack.addParticle(std::make_tuple(
        particle, calculate_kinetic_energy(plab.getNorm(), get_mass(particle)),
        plab.normalized(), pos, 0_ns))};

    // a refractive index
    const double ri_{1.000327};

    // the constant density
    const auto density{19.2_g / cube(1_cm)};

    // the composition we use for the homogeneous medium
    NuclearComposition const protonComposition({Code::Proton}, {1.});

    // create magnetic field vector
    Vector B1(rootCS, 0_T, 0_T, 1_T);

    auto Medium = EnvType::createNode<Sphere>(
        center, 1_km * std::numeric_limits<double>::infinity());

    auto const props = Medium->setModelProperties<AtmModel>(ri_, Medium::AirDry1Atm, B1,
                                                            density, protonComposition);
    particle1.setNode(Medium.get());
    envZHS.getUniverse()->addChild(std::move(Medium));

    // create the detector
    const auto obs1Loc{Point(rootCS, 100_m, 2_m, 3_m)};
    const auto obs2Loc{Point(rootCS, 4_m, 80_m, 6_m)};
    const TimeType t1{0_s};
    const TimeType t2{10_s};
    const InverseTimeType t3{1e+3_Hz};
    TimeDomainObserver obs1("observer_name", obs1Loc, rootCS, t1, t2, t3, t1);
    TimeDomainObserver obs2("observer_name2", obs2Loc, rootCS, t1, t2, t3, t1);
    ObserverCollection<TimeDomainObserver> detector;
    detector.addObserver(obs1);
    detector.addObserver(obs2);

    auto const charge_{get_charge(particle1.getPID())};

    auto SP = make_numerical_integrating_radio_propagator(envZHS, 1_m);

    // create a radio process instance using ZHS
    RadioProcess<ObserverCollection<TimeDomainObserver>,
                 ZHS<ObserverCollection<TimeDomainObserver>, decltype(SP)>, decltype(SP)>
        zhs(detector, SP);

    Step step(particle1, base);
    // check doContinuous and simulate methods
    zhs.doContinuous(step, true);

    // zhs output check
    std::string const implemenzhs{"ZHS"};

    boost::filesystem::path const tempPathZ{boost::filesystem::temp_directory_path() /
                                            ("test_corsika_radio_" + implemenzhs)};
    if (boost::filesystem::exists(tempPathZ)) {
      boost::filesystem::remove_all(tempPathZ);
    }

    boost::filesystem::create_directory(tempPathZ);
    zhs.startOfLibrary(tempPathZ);
    auto const outputFileZ = tempPathZ / ("observers.parquet");
    CHECK(boost::filesystem::exists(outputFileZ));
    // run end of shower and make sure that something extra was added
    auto const fileSizeZ = boost::filesystem::file_size(outputFileZ);
    zhs.endOfShower(0);
    CHECK(boost::filesystem::file_size(outputFileZ) > fileSizeZ);
    zhs.endOfLibrary();

  } // END: SECTION("ZHS process")

  SECTION("Radio extreme cases") {

    // Environment
    using IModelInterface =
        IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
    using AtmModel = UniformRefractiveIndex<
        MediumPropertyModel<UniformMagneticField<HomogeneousMedium<IModelInterface>>>>;
    using EnvType = Environment<AtmModel>;
    EnvType envRadio;
    CoordinateSystemPtr const& rootCSRadio = envRadio.getCoordinateSystem();
    // get the center point
    Point const center{rootCSRadio, 0_m, 0_m, 0_m};

    Point const point_1(rootCSRadio, {1_m, 1_m, 0_m});
    // create a new stack for each trial
    setup::Stack<EnvType> stack;
    // create a particle
    const Code particle{Code::Electron};
    const Code particle2{Code::Proton};

    const auto pmass{get_mass(particle)};
    // construct an energy
    const HEPEnergyType E0{1_TeV};
    // compute the necessary momentum
    const HEPMomentumType P0{sqrt(E0 * E0 - pmass * pmass)};
    // and create the momentum vector
    const auto plab{MomentumVector(rootCSRadio, {P0, 0_GeV, 0_GeV})};
    // add the particle to the stack
    auto particle_stack{stack.addParticle(std::make_tuple(
        particle, calculate_kinetic_energy(plab.getNorm(), get_mass(particle)),
        plab.normalized(), point_1, 0_ns))};

    // particle stack with proton
    auto particle_stack_proton{stack.addParticle(std::make_tuple(
        particle2, calculate_kinetic_energy(plab.getNorm(), get_mass(particle2)),
        plab.normalized(), point_1, 0_ns))};

    // a refractive index
    const double ri_{1.415};

    // the constant density
    const auto density{19.2_g / cube(1_cm)};

    // the composition we use for the homogeneous medium
    NuclearComposition const protonComposition({Code::Proton}, {1.});

    // create magnetic field vector
    Vector B1(rootCSRadio, 0_T, 50_uT, 0_T);

    auto Medium = EnvType::createNode<Sphere>(
        center, 1_km * std::numeric_limits<double>::infinity());

    auto const props = Medium->setModelProperties<AtmModel>(ri_, Medium::AirDry1Atm, B1,
                                                            density, protonComposition);
    particle_stack.setNode(Medium.get());
    particle_stack_proton.setNode(Medium.get());
    envRadio.getUniverse()->addChild(std::move(Medium));

    // now create observers and detectors
    // the observers location
    const auto point1{Point(envRadio.getCoordinateSystem(), 0_m, 0_m, 0_m)};

    // track points
    Point const point_2(rootCSRadio, {2_km, 1_km, 0_m});
    Point const point_4(rootCSRadio, {0_m, 1_m, 0_m});
    const auto point_b{Point(rootCSRadio, 30000_m, 0_m, 0_m)};

    // create times for the observer
    const TimeType start{0_s};
    const TimeType duration{100_ns};
    const InverseTimeType sample{1e+12_Hz};
    const TimeType duration_dummy{2_s};
    const InverseTimeType sample_dummy{1_Hz};

    // create specific times for observer to do timebin check
    const TimeType start_b{0.994e-4_s};
    const TimeType duration_b{1.07e-4_s - 0.994e-4_s};
    const InverseTimeType sampleRate_b{5e+11_Hz};

    // check that I can create an observer at (1, 2, 3)
    TimeDomainObserver obs1("observer_name", point1, rootCSRadio, start, duration, sample,
                            start);
    TimeDomainObserver obs2("dummy", point1, rootCSRadio, start, duration_dummy,
                            sample_dummy, start);
    TimeDomainObserver obs_b("timebin", point_b, rootCSRadio, start_b, duration_b,
                             sampleRate_b, start_b);
    // construct a radio detector instance to store our observers
    ObserverCollection<TimeDomainObserver> detector;
    ObserverCollection<TimeDomainObserver> detector_dummy;
    ObserverCollection<TimeDomainObserver> detector_b;
    // add the observers to the detector
    detector.addObserver(obs1);
    detector_dummy.addObserver(obs2);
    detector_b.addObserver(obs_b);

    // feed radio with a proton track to check that it skips that track.
    TimeType tp{(point_2 - point_1).getNorm() / (0.999 * constants::c)};
    VelocityVector vp{(point_2 - point_1) / tp};
    Line lp{point_1, vp};
    StraightTrajectory track_p{lp, tp};
    Step step_proton(particle_stack_proton, track_p);

    // feed radio with a track that triggers zhs like approx in coreas and fraunhofer
    // limit check for zhs
    TimeType th{(point_4 - point1).getNorm() / constants::c};
    VelocityVector vh{(point_4 - point1) / th};
    Line lh{point1, vh};
    StraightTrajectory track_h{lh, th};
    StraightTrajectory track_h_neg_time{lh, -th};
    Step step_h(particle_stack, track_h);
    Step step_h_neg_time(particle_stack, track_h_neg_time);

    // feed radio with an electron track that ends in a different observer bin.
    Point const point_start(rootCSRadio, {100_m, 0_m, 0_m});
    Point const point_end(rootCSRadio, {100_m, 0.00628319_m, 0_m});
    TimeType tb{(point_end - point_start).getNorm() / (0.999 * constants::c)};
    VelocityVector vb{(point_end - point_start) / tb};
    Line lb{point_start, vb};
    StraightTrajectory track_b{lb, tb};
    Step step_b(particle_stack, track_b);

    auto SP = make_dummy_test_radio_propagator(envRadio);

    // create radio process instances
    RadioProcess<decltype(detector), CoREAS<decltype(detector), decltype(SP)>,
                 decltype(SP)>
        coreas(detector, SP);

    RadioProcess<decltype(detector), ZHS<decltype(detector), decltype(SP)>, decltype(SP)>
        zhs(detector, SP);

    coreas.doContinuous(step_proton, true);
    zhs.doContinuous(step_proton, true);
    coreas.doContinuous(step_h, true);
    zhs.doContinuous(step_h, true);
    zhs.doContinuous(step_h_neg_time, true);

    // create radio processes with "dummy" observer to trigger extreme time-binning
    RadioProcess<decltype(detector_dummy), CoREAS<decltype(detector_dummy), decltype(SP)>,
                 decltype(SP)>
        coreas_dummy(detector_dummy, SP);

    RadioProcess<decltype(detector_dummy), ZHS<decltype(detector_dummy), decltype(SP)>,
                 decltype(SP)>
        zhs_dummy(detector_dummy, SP);
    coreas_dummy.doContinuous(step_proton, true);
    zhs_dummy.doContinuous(step_proton, true);
    coreas_dummy.doContinuous(step_h, true);
    zhs_dummy.doContinuous(step_h, true);

    // create radio process instances
    RadioProcess<decltype(detector_b), CoREAS<decltype(detector_b), decltype(SP)>,
                 decltype(SP)>
        coreas_b(detector_b, SP);

    RadioProcess<decltype(detector_b), ZHS<decltype(detector_b), decltype(SP)>,
                 decltype(SP)>
        zhs_b(detector_b, SP);
    coreas_b.doContinuous(step_b, true);
    zhs_b.doContinuous(step_b, true);

  } // END: SECTION("Radio extreme cases")

  SECTION("Process Library") {
    using IModelInterface =
        IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
    using AtmModel = UniformRefractiveIndex<
        MediumPropertyModel<UniformMagneticField<HomogeneousMedium<IModelInterface>>>>;
    using EnvType = Environment<AtmModel>;
    EnvType envCoREAS;
    CoordinateSystemPtr const& rootCS = envCoREAS.getCoordinateSystem();
    Point const center{rootCS, 0_m, 0_m, 0_m};

    Vector B1(rootCS, 0_T, 0_T, 1_T);
    NuclearComposition const protonComposition({Code::Proton}, {1.});
    const double refractiveIndex{1.000327};
    const auto density{1_g / cube(1_cm)};
    auto Medium = EnvType::createNode<Sphere>(
        center, 10_km * std::numeric_limits<double>::infinity());
    auto const props = Medium->setModelProperties<AtmModel>(
        refractiveIndex, Medium::AirDry1Atm, B1, density, protonComposition);
    envCoREAS.getUniverse()->addChild(std::move(Medium));

    // create the detector
    const auto obs1Loc{Point(rootCS, 100_m, 2_m, 3_m)};
    const auto obs2Loc{Point(rootCS, 4_m, 80_m, 6_m)};
    const TimeType t1{0_s};
    const TimeType t2{10_s};
    const InverseTimeType t3{1e+3_Hz};
    TimeDomainObserver obs1("observer_name", obs1Loc, rootCS, t1, t2, t3, t1);
    TimeDomainObserver obs2("observer_name2", obs2Loc, rootCS, t1, t2, t3, t1);
    ObserverCollection<TimeDomainObserver> detector;
    detector.addObserver(obs1);
    detector.addObserver(obs2);

    const auto trackStart{Point(rootCS, 7_m, 8_m, 9_m)};
    const auto trackEnd{Point(rootCS, 5_m, 5_m, 10_m)};

    // create an electron
    const Code electron{Code::Electron};
    const auto pmass{get_mass(electron)};

    VelocityVector v0(rootCS, {1_m / second, 0_m / second, 0_m / second});

    Vector B0(rootCS, 5_T, 5_T, 5_T);

    Line const line(trackStart, v0);

    // create a new stack for each trial
    setup::Stack<EnvType> stack;

    // construct an energy
    const HEPEnergyType E0{1_TeV};

    // compute the necessary momentumn
    const HEPMomentumType P0{sqrt(E0 * E0 - pmass * pmass)};
    const auto plab{MomentumVector(rootCS, {0_GeV, 0_GeV, P0})};

    // and create the location of the particle in this coordinate system
    const Point pos(rootCS, 50_m, 10_m, 80_m);

    // add the particle to the stack
    auto const particle1{stack.addParticle(std::make_tuple(
        electron, calculate_kinetic_energy(plab.getNorm(), get_mass(electron)),
        plab.normalized(), pos, 0_ns))};

    auto SP = make_numerical_integrating_radio_propagator(envCoREAS, 1_m);

    // create a radio process instance using CoREAS
    RadioProcess<decltype(detector), CoREAS<decltype(detector), decltype(SP)>,
                 decltype(SP)>
        coreas(detector, SP);

    const auto config = coreas.getConfig();

    CHECK(config["type"].as<std::string>() == "RadioProcess");
    CHECK(config["algorithm"].as<std::string>() == "CoREAS");
    CHECK(config["units"]["time"].as<std::string>() == "ns");
    CHECK(config["units"]["frequency"].as<std::string>() == "GHz");
    CHECK(config["units"]["electric field"].as<std::string>() == "V/m");
    CHECK(config["units"]["distance"].as<std::string>() == "m");

    CHECK(config["observers"]["observer_name"]["location"][0].as<double>() == 100);
    CHECK(config["observers"]["observer_name"]["location"][1].as<double>() == 2);
    CHECK(config["observers"]["observer_name"]["location"][2].as<double>() == 3);

    CHECK(config["observers"]["observer_name2"]["location"][0].as<double>() == 4);
    CHECK(config["observers"]["observer_name2"]["location"][1].as<double>() == 80);
    CHECK(config["observers"]["observer_name2"]["location"][2].as<double>() == 6);
  } // END: SECTION("Process Library")

  SECTION("LimitedRadioProcess") {
    /// Make a limited selection where any particle further than 1_m from the origin will
    /// not be executed.

    TestRadioProcess wrappedRadioProcess;

    Environment<IEmpty> env;
    CoordinateSystemPtr const& rootCS = env.getCoordinateSystem();
    Point const origin(rootCS, {0_m, 0_m, 0_m});
    Point const outside(rootCS, {1000_m, 0_m, 0_m});

    // Fcn to check if the wrapped process should be run
    auto const RadioLimitedSelection = [origin](Point const& position) -> bool {
      return (position - origin).getNorm() > 1_m;
    };

    CHECK_FALSE(RadioLimitedSelection(origin)); // sanity check on the slection
    CHECK(RadioLimitedSelection(outside));      // sanity check on the slection, part2

    LimitedRadioProcess limitedProcess(std::move(wrappedRadioProcess),
                                       RadioLimitedSelection);

    // Check that the feed-through to the underlying process works
    double dummy = 0; // Doesn't matter what this is
    CHECK(limitedProcess.getMaxStepLength(dummy, dummy) == 123_m);
    // Does feed-through to underlying process
    CHECK(limitedProcess.getMaxStepLength(dummy, dummy) ==
          wrappedRadioProcess.getMaxStepLength(dummy, dummy));

    // Set up test for doContinuouse
    Line lineInside(origin, VelocityVector(rootCS, {1_m / 1_s, 1_m / 1_s, 1_m / 1_s}));
    Line lineOutside(outside, VelocityVector(rootCS, {1_m / 1_s, 1_m / 1_s, 1_m / 1_s}));

    StraightTrajectory const trajectoryInside(lineInside, 0_ns);
    StraightTrajectory const trajectoryOutside(lineOutside, 0_ns);

    setup::Stack<Environment<IEmpty>> stack;

    /////// Electron case //////

    auto electronInside = stack.addParticle(std::make_tuple(
        Code::Electron, 10_GeV, DirectionVector(rootCS, {1, 0, 0}), origin, 0_s));
    auto electronOutside = stack.addParticle(std::make_tuple(
        Code::Electron, 10_GeV, DirectionVector(rootCS, {1, 0, 0}), outside, 0_s));

    Step const stepElectronInside(electronInside, trajectoryInside);
    Step const stepElectronOutside(electronOutside, trajectoryOutside);

    CHECK(limitedProcess.doContinuous(stepElectronInside, true) == ProcessReturn::Ok);
    CHECK(limitedProcess.doContinuous(stepElectronOutside, true) ==
          ProcessReturn::Interacted);

    /////// Proton case //////

    auto protonInside = stack.addParticle(std::make_tuple(
        Code::Proton, 10_GeV, DirectionVector(rootCS, {1, 0, 0}), origin, 0_s));
    auto protonOutside = stack.addParticle(std::make_tuple(
        Code::Proton, 10_GeV, DirectionVector(rootCS, {1, 0, 0}), outside, 0_s));

    Step const stepProtonInside(protonInside, trajectoryInside);
    Step const stepProtonOutisde(protonOutside, trajectoryOutside);

    CHECK(limitedProcess.doContinuous(stepProtonInside, true) == ProcessReturn::Ok);
    CHECK(limitedProcess.doContinuous(stepProtonOutisde, true) == ProcessReturn::Ok);

  } // END: SECTION("LimitedRadioProcess")

} // END: TEST_CASE("Radio", "[processes]")

TEST_CASE("observers") {

  SECTION("TimeDomainObserver Constructor") {
    Environment<IRefractiveIndexModel<IMediumModel>> env;
    const auto rootCS = env.getCoordinateSystem();
    auto const obsPos = Point(rootCS, {0_m, 0_m, 0_m});
    TimeType const tStart(0_s);
    TimeType const duration(10_ns);
    InverseTimeType const sampleRate(1_GHz);
    TimeType const groundHitTime(1e3_ns);

    TimeDomainObserver const observer("observer", obsPos, rootCS, tStart, duration,
                                      sampleRate, groundHitTime);

    // All waveforms are of equal non-zero size
    CHECK(observer.getWaveformX().size() == observer.getWaveformY().size());
    CHECK(observer.getWaveformX().size() == observer.getWaveformZ().size());
    CHECK(observer.getWaveformX().size() > 0);

    // All waveform values are initialized to zero
    for (auto const& val : observer.getWaveformX()) { CHECK(val * 0 == val); }
    for (auto const& val : observer.getWaveformY()) { CHECK(val * 0 == val); }
    for (auto const& val : observer.getWaveformZ()) { CHECK(val * 0 == val); }

    // check that variables were set properly
    CHECK("observer" == observer.getName());
    CHECK(sampleRate == observer.getSampleRate());
    CHECK(tStart == observer.getStartTime());

    // and check that the observer is at the right location
    CHECK((observer.getLocation() - obsPos).getNorm() < 1e-12 * 1_m);
  } // END: SECTION("TimeDomainObserver Constructor")

  SECTION("TimeDomainObserver Bad Constructor") {
    Environment<IRefractiveIndexModel<IMediumModel>> env;
    const auto rootCS = env.getCoordinateSystem();
    auto const obsPos = Point(rootCS, {0_m, 0_m, 0_m});
    TimeType const tStart(0_s);
    TimeType const duration(1e3_ns);
    InverseTimeType const sampleRate(1_GHz);
    TimeType const groundHitTime(10_ns);

    // Giving zero or negative values for sampling rate and duration
    TimeDomainObserver const observer_bad1("bad_observer", obsPos, rootCS, tStart, -13_ns,
                                           sampleRate, groundHitTime);
    TimeDomainObserver const observer_bad2("bad_observer", obsPos, rootCS, tStart, 0_ns,
                                           sampleRate, groundHitTime);
    TimeDomainObserver const observer_bad3("bad_observer", obsPos, rootCS, tStart,
                                           duration, -1_GHz, groundHitTime);
  } // END: SECTION("TimeDomainObserver Bad Constructor")

  SECTION("TimeDomainObserver Receive Efield") {
    // Checks that the basic functionality of the receive function is working properly

    using EnvType = Environment<IRefractiveIndexModel<IMediumModel>>;
    EnvType env;

    const auto rootCS = env.getCoordinateSystem();

    auto const point1 = Point(rootCS, {1_m, 2_m, 3_m});
    auto const point2 = Point(rootCS, {4_m, 5_m, 6_m});

    // create times for the observer
    const TimeType t1{10_s};
    const TimeType t2{10_s};
    const InverseTimeType t3{1 / 1_s};
    const TimeType t4{11_s};

    // make the two observers with different start times
    TimeDomainObserver obs1("observer_name", point1, rootCS, t1, t2, t3, t1);
    TimeDomainObserver obs2("observer_name", point2, rootCS, t4, t2, t3, t4);

    Vector<dimensionless_d> receiveVec1(rootCS, {0, 0, 1});
    Vector<dimensionless_d> receiveVec2(rootCS, {0, 0, 1});
    Vector<dimensionless_d> emitVec1 = -1 * receiveVec1;
    Vector<dimensionless_d> emitVec2(rootCS, {0, 1, 0});

    Vector<ElectricFieldType::dimension_type> eField1(
        rootCS, {10_V / 1_m, 10_V / 1_m, 10_V / 1_m});
    Vector<ElectricFieldType::dimension_type> eField2(
        rootCS, {20_V / 1_m, 20_V / 1_m, 20_V / 1_m});

    // sources
    auto const source1 = point1 + 1_m * receiveVec1;
    auto const source2 = point2 + 1_m * receiveVec2;

    // Most of this isn't needed, so we don't need to worry about speed of light, etc.
    std::deque<Point> points1 = {source1, point1};
    SignalPath path1(1_ns, 1.0, 1.0, 1.0, emitVec1, receiveVec1,
                     (point1 - source1).getNorm(), points1);
    std::deque<Point> points2 = {source2, point2};
    SignalPath path2(1_ns, 1.0, 1.0, 1.0, emitVec2, receiveVec2,
                     (point2 - source2).getNorm(), points2);

    // inject efield into obs1
    obs1.receive(15_s, path1, eField1);
    REQUIRE(obs1.getWaveformX()[5] - 10 == 0);
    REQUIRE(obs1.getWaveformX()[5] == obs1.getWaveformY()[5]);
    REQUIRE(obs1.getWaveformX()[5] == obs1.getWaveformZ()[5]);

    // inject efield but with different receive vector into obs2
    obs2.receive(16_s, path2, eField1);
    REQUIRE(obs1.getWaveformX()[5] ==
            obs2.getWaveformX()[5]); // Currently receive vector does nothing
    obs2.reset();
    REQUIRE(obs2.getWaveformX()[5] == 0); // reset was successful

    // inject the other eField into obs2
    obs2.receive(16_s, path2, eField2);
    REQUIRE(obs2.getWaveformX()[5] - 20 == 0);
    REQUIRE(obs2.getWaveformX()[5] == obs2.getWaveformY()[5]);
    REQUIRE(obs2.getWaveformX()[5] == -1 * obs2.getWaveformZ()[5]);

    SignalPath pathCross(1_ns, 1.0, 1.0, 1.0, emitVec1, receiveVec2,
                         (point1 - source1).getNorm(), points1);
    // make sure the next one is empty before filling it
    REQUIRE(obs2.getWaveformX()[6] == 0);
    obs2.receive(17_s, pathCross, eField2);
    REQUIRE(obs2.getWaveformX()[6] - 20 == 0);

    // reset obs1 and then put values in out of range
    obs1.reset();
    obs1.receive(-1000_s, path1, eField1);
    for (auto const& val : obs1.getWaveformX()) { CHECK(val * 0 == val); }
    obs1.reset();
    obs1.receive(t1 + t2 + 1_s, path1, eField1);
    for (auto const& val : obs1.getWaveformX()) { CHECK(val * 0 == val); }
  } // END: SECTION("TimeDomainObserver Receive EField")

  SECTION("TimeDomainObserver Receive Vector Potential") {
    // Checks that the basic functionality of the receive function is working properly

    using EnvType = Environment<IRefractiveIndexModel<IMediumModel>>;
    EnvType env;

    const auto rootCS = env.getCoordinateSystem();

    auto const point1 = Point(rootCS, {1_m, 2_m, 3_m});
    auto const point2 = Point(rootCS, {4_m, 5_m, 6_m});

    // create times for the observer
    const TimeType t1{10_s};
    const TimeType t2{10_s};
    const InverseTimeType t3{1 / 1_s};
    const TimeType t4{11_s};

    // make the two observers with different start times
    TimeDomainObserver obs1("observer_name", point1, rootCS, t1, t2, t3, t1);
    TimeDomainObserver obs2("observer_name", point2, rootCS, t4, t2, t3, t4);

    Vector<dimensionless_d> receiveVec1(rootCS, {0, 0, 1});
    Vector<dimensionless_d> receiveVec2(rootCS, {0, 1, 0});
    Vector<dimensionless_d> emitVec1 = -1 * receiveVec1;
    Vector<dimensionless_d> emitVec2(rootCS, {0, 1, 0});

    Vector<VectorPotentialType::dimension_type> vectorPotential1(
        rootCS, {10_V * 1_s / 1_m, 10_V * 1_s / 1_m, 10_V * 1_s / 1_m});
    Vector<VectorPotentialType::dimension_type> vectorPotential2(
        rootCS, {20_V * 1_s / 1_m, 20_V * 1_s / 1_m, 20_V * 1_s / 1_m});

    // sources
    auto const source1 = point1 + 1_m * receiveVec1;
    auto const source2 = point2 + 1_m * receiveVec2;

    // Most of this isn't needed, so we don't need to worry about speed of light, etc.
    std::deque<Point> points1 = {source1, point1};
    SignalPath path1(1_ns, 1.0, 1.0, 1.0, emitVec1, receiveVec1,
                     (point1 - source1).getNorm(), points1);
    std::deque<Point> points2 = {source2, point2};
    SignalPath path2(1_ns, 1.0, 1.0, 1.0, emitVec2, receiveVec2,
                     (point2 - source2).getNorm(), points2);

    // inject efield into obs1
    obs1.receive(15_s, path1, vectorPotential1);
    REQUIRE(obs1.getWaveformX()[5] - 10 == 0);
    REQUIRE(obs1.getWaveformX()[5] == obs1.getWaveformY()[5]);
    REQUIRE(obs1.getWaveformX()[5] == obs1.getWaveformZ()[5]);

    // inject efield but with different receive vector into obs2
    obs2.receive(16_s, path2, vectorPotential1);
    REQUIRE(obs1.getWaveformX()[5] == -1 * obs2.getWaveformX()[5]);
    obs2.reset();
    REQUIRE(obs2.getWaveformX()[5] == 0); // reset was successful

    // inject the other eField into obs2
    obs2.receive(16_s, path2, vectorPotential2);
    REQUIRE(obs2.getWaveformX()[5] + 20 == 0);
    REQUIRE(obs2.getWaveformX()[5] == obs2.getWaveformY()[5]);
    REQUIRE(obs2.getWaveformX()[5] == obs2.getWaveformZ()[5]);

    // make sure the next one is empty before filling it
    REQUIRE(obs2.getWaveformX()[6] == 0);
    obs2.receive(17_s, path2, vectorPotential2);
    REQUIRE(obs2.getWaveformX()[6] + 20 == 0);

    // reset obs1 and then put values in out of range
    obs1.reset();
    obs1.receive(-1000_s, path1, vectorPotential1);
    for (auto const& val : obs1.getWaveformX()) { CHECK(val * 0 == val); }
    obs1.reset();
    obs1.receive(t1 + t2 + 1_s, path1, vectorPotential1);
    for (auto const& val : obs1.getWaveformX()) { CHECK(val * 0 == val); }
  } // END: SECTION("TimeDomainObserver Receive Vector Potential")

  SECTION("TimeDomainObserver ObserverCollection") {

    // create an environment so we can get a coordinate system
    using EnvType = Environment<IRefractiveIndexModel<IMediumModel>>;
    EnvType env6;

    using UniRIndex =
        UniformRefractiveIndex<HomogeneousMedium<IRefractiveIndexModel<IMediumModel>>>;

    // the observer location
    const auto point1{Point(env6.getCoordinateSystem(), 1_m, 2_m, 3_m)};
    const auto point2{Point(env6.getCoordinateSystem(), 4_m, 5_m, 6_m)};

    // get a coordinate system
    const CoordinateSystemPtr rootCS = env6.getCoordinateSystem();

    auto Medium6 = EnvType::createNode<Sphere>(
        Point{rootCS, 0_m, 0_m, 0_m}, 1_km * std::numeric_limits<double>::infinity());

    auto const props6 = Medium6->setModelProperties<UniRIndex>(
        1, 1_kg / (1_m * 1_m * 1_m), NuclearComposition({Code::Nitrogen}, {1.}));

    env6.getUniverse()->addChild(std::move(Medium6));

    // create times for the observer
    const TimeType t1{10_s};
    const TimeType t2{10_s};
    const InverseTimeType t3{1 / 1_s};
    const TimeType t4{11_s};

    // construct a radio detector instance to store our observers
    ObserverCollection<TimeDomainObserver> detector;

    // the following creates a star-shaped pattern of observers in the ground
    const auto point11{Point(env6.getCoordinateSystem(), 1000_m, 20_m, 30_m)};
    const TimeType t2222{1e-6_s};
    const InverseTimeType t3333{1e+9_Hz};

    std::vector<std::string> observer_names;
    std::vector<Point> observer_locations;
    for (auto radius = 100_m; radius <= 200_m; radius += 100_m) {
      for (auto phi = 0; phi <= 315; phi += 45) {
        auto phiRad = phi / 180. * M_PI;
        auto const point{Point(env6.getCoordinateSystem(), radius * cos(phiRad),
                               radius * sin(phiRad), 0_m)};
        observer_locations.push_back(point);
        auto time__{(point11 - point).getNorm() / constants::c};
        const int rr_ = static_cast<int>(radius / 1_m);
        std::string name = "observer_R=" + std::to_string(rr_) +
                           "_m-Phi=" + std::to_string(phi) + "degrees";
        observer_names.push_back(name);
        TimeDomainObserver obs(name, point, rootCS, time__, t2222, t3333, time__);
        detector.addObserver(obs);
      }
    }

    CHECK(detector.size() == 16);
    CHECK(detector.getObservers().size() == 16);
    int i = 0;
    // this prints out the observer names and locations
    for (auto const& observer : detector.getObservers()) {
      CHECK(observer.getName() == observer_names[i]);
      CHECK(distance(observer.getLocation(), observer_locations[i]) / 1_m == 0);
      i++;
    }

    // Check the .at() method for radio detectors
    for (int i = 0; i <= (detector.size() - 1); i++) {
      CHECK(detector.at(i).getName() == observer_names[i]);
      CHECK(distance(detector.at(i).getLocation(), observer_locations[i]) / 1_m == 0);
    }

  } // END: SECTION("TimeDomainObserver ObserverCollection")

  SECTION("TimeDomainObserver Config File") {
    // Runs checks that the file readers are working properly
    Environment<IRefractiveIndexModel<IMediumModel>> env;
    const auto rootCS = env.getCoordinateSystem();
    auto const obsPos = Point(rootCS, {0_m, 0_m, 0_m});
    TimeType const tStart(0_s);
    TimeType const duration(10_ns);
    InverseTimeType const sampleRate(1_GHz);
    TimeType const groundHitTime(1e3_ns);

    TimeDomainObserver observerC("test_observerCoREAS", obsPos, rootCS, tStart, duration,
                                 sampleRate, groundHitTime);
    TimeDomainObserver observerZ("test_observerZHS", obsPos, rootCS, tStart, duration,
                                 sampleRate, groundHitTime);

    // Check the YAML file output
    auto const configC = observerC.getConfig();
    CHECK(configC["type"].as<std::string>() == "TimeDomainObserver");
    CHECK(configC["start time"].as<double>() == tStart / 1_ns);
    CHECK(configC["duration"].as<double>() == duration / 1_ns);
    CHECK(configC["sampling frequency"].as<double>() == sampleRate / 1_GHz);

    auto const configZ = observerZ.getConfig();
    CHECK(configZ["type"].as<std::string>() == "TimeDomainObserver");
    CHECK(configZ["start time"].as<double>() == tStart / 1_ns);
    CHECK(configZ["duration"].as<double>() == duration / 1_ns);
    CHECK(configZ["sampling frequency"].as<double>() == sampleRate / 1_GHz);
  } // END: SECTION("TimeDomainObserver Config File")

} // END: TEST_CASE("observers")

TEST_CASE("Propagators") {

  SECTION("Dummy Test Propagator w/ Uniform Refractive Index") {

    // create a suitable environment
    using IModelInterface =
        IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
    using AtmModel = UniformRefractiveIndex<
        MediumPropertyModel<UniformMagneticField<HomogeneousMedium<IModelInterface>>>>;
    using EnvType = Environment<AtmModel>;
    EnvType env;
    CoordinateSystemPtr const& rootCS = env.getCoordinateSystem();

    // create a particle
    auto const particle{Code::Electron};
    const auto pmass{get_mass(particle)};

    // create a new stack for each trial
    setup::Stack<EnvType> stack;

    // construct an energy
    const HEPEnergyType E0{1_TeV};

    // compute the necessary momentum
    const HEPMomentumType P0{sqrt(E0 * E0 - pmass * pmass)};
    const auto plab{MomentumVector(rootCS, {0_GeV, 0_GeV, P0})};

    Point const p0(rootCS, {0_m, 0_m, 0_m});

    // add the particle to the stack
    auto particle1{stack.addParticle(std::make_tuple(
        particle, calculate_kinetic_energy(plab.getNorm(), get_mass(particle)),
        plab.normalized(), p0, 0_ns))};

    // get the center point
    Point const center{rootCS, 0_m, 0_m, 0_m};
    // a refractive index for the vacuum
    const double ri_{1};
    // the constobs density
    const auto density{19.2_g / cube(1_cm)};
    // the composition we use for the homogeneous medium
    NuclearComposition const Composition({Code::Nitrogen}, {1.});
    // create magnetic field vector
    Vector B1(rootCS, 0_T, 0_T, 0.3809_T);
    // create a Sphere for the medium
    auto Medium = EnvType::createNode<Sphere>(
        center, 1_km * std::numeric_limits<double>::infinity());
    // set the environment properties
    auto const props = Medium->setModelProperties<AtmModel>(ri_, Medium::AirDry1Atm, B1,
                                                            density, Composition);
    particle1.setNode(Medium.get());
    // bind things together
    env.getUniverse()->addChild(std::move(Medium));

    // get some points
    //    Point const p0(rootCS, {0_m, 0_m, 0_m});
    Point const p10(rootCS, {0_m, 0_m, 10_m});

    // get a unit vector
    Vector<dimensionless_d> const v1(rootCS, {0, 0, 1});
    Vector<dimensionless_d> const v2(rootCS, {0, 0, -1});

    // get a geometrical path of points
    Path const P1({p0, p10});

    // construct a Dummy Test Propagator given the uniform refractive index environment
    DummyTestPropagator SP(env);

    // store the outcome of the Propagate method to paths_
    auto const paths_ = SP.propagate(particle1, p0, p10);

    // perform checks to paths_ components
    for (auto const& path : paths_) {
      CHECK((path.propagation_time_ / 1_s) -
                (((p10 - p0).getNorm() / constants::c) / 1_s) ==
            Approx(0));
      CHECK(path.average_refractive_index_ == Approx(1));
      CHECK(path.refractive_index_source_ == Approx(1));
      CHECK(path.refractive_index_destination_ == Approx(1));
      CHECK(path.emit_.getComponents() == v1.getComponents());
      CHECK(path.receive_.getComponents() == v2.getComponents());
      CHECK(path.R_distance_ == 10_m);
      CHECK(std::equal(
          P1.begin(), P1.end(), path.begin(),
          [](Point const& a, Point const& b) { return (a - b).getNorm() / 1_m < 1e-5; }));
    }

  } // END: SECTION("Dummy Test Propagator w/ Uniform Refractive Index")

  // check that I can create working Numerical Integrating Propagators in different
  // environments
  SECTION("Numerical Integrating Propagator w/ Uniform Refractive Index") {

    // create a suitable environment
    using IModelInterface =
        IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
    using AtmModel = UniformRefractiveIndex<
        MediumPropertyModel<UniformMagneticField<HomogeneousMedium<IModelInterface>>>>;
    using EnvType = Environment<AtmModel>;
    EnvType env;
    CoordinateSystemPtr const& rootCS = env.getCoordinateSystem();

    // create a particle
    auto const particle{Code::Electron};
    const auto pmass{get_mass(particle)};

    // create a new stack for each trial
    setup::Stack<EnvType> stack;

    // construct an energy
    const HEPEnergyType E0{1_TeV};

    // compute the necessary momentum
    const HEPMomentumType P0{sqrt(E0 * E0 - pmass * pmass)};
    const auto plab{MomentumVector(rootCS, {0_GeV, 0_GeV, P0})};

    Point const p0(rootCS, {0_m, 0_m, 0_m});

    // add the particle to the stack
    auto particle1{stack.addParticle(std::make_tuple(
        particle, calculate_kinetic_energy(plab.getNorm(), get_mass(particle)),
        plab.normalized(), p0, 0_ns))};

    // get the center point
    Point const center{rootCS, 0_m, 0_m, 0_m};
    // a refractive index for the vacuum
    const double ri_{1};
    // the constobs density
    const auto density{19.2_g / cube(1_cm)};
    // the composition we use for the homogeneous medium
    NuclearComposition const Composition({Code::Nitrogen}, {1.});
    // create magnetic field vector
    Vector B1(rootCS, 0_T, 0_T, 0.3809_T);
    // create a Sphere for the medium
    auto Medium = EnvType::createNode<Sphere>(
        center, 1_km * std::numeric_limits<double>::infinity());
    // set the environment properties
    auto const props = Medium->setModelProperties<AtmModel>(ri_, Medium::AirDry1Atm, B1,
                                                            density, Composition);
    particle1.setNode(Medium.get());
    // bind things together
    env.getUniverse()->addChild(std::move(Medium));

    // get some points
    Point const p1(rootCS, {0_m, 0_m, 1_m});
    Point const p2(rootCS, {0_m, 0_m, 2_m});
    Point const p3(rootCS, {0_m, 0_m, 3_m});
    Point const p4(rootCS, {0_m, 0_m, 4_m});
    Point const p5(rootCS, {0_m, 0_m, 5_m});
    Point const p6(rootCS, {0_m, 0_m, 6_m});
    Point const p7(rootCS, {0_m, 0_m, 7_m});
    Point const p8(rootCS, {0_m, 0_m, 8_m});
    Point const p9(rootCS, {0_m, 0_m, 9_m});
    Point const p10(rootCS, {0_m, 0_m, 10_m});
    Point const p30(rootCS, {0_m, 0_m, 30000_m});

    // get a unit vector
    Vector<dimensionless_d> const v1(rootCS, {0, 0, 1});
    Vector<dimensionless_d> const v2(rootCS, {0, 0, -1});

    // get a geometrical path of points
    Path const P1({p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10});

    // construct a Numerical Integrating Propagator given the uniform refractive index
    // environment
    NumericalIntegratingPropagator const SP(env, 1_m);

    // store the outcome of the Propagate method to paths_
    auto const paths_ = SP.propagate(particle1, p0, p10);

    // perform checks to paths_ components
    for (auto const& path : paths_) {
      CHECK((path.propagation_time_ / 1_s) -
                (((p10 - p0).getNorm() / constants::c) / 1_s) ==
            Approx(0).margin(absMargin));
      CHECK(path.average_refractive_index_ == Approx(1));
      CHECK(path.refractive_index_source_ == Approx(1));
      CHECK(path.refractive_index_destination_ == Approx(1));
      CHECK(path.emit_.getComponents() == v1.getComponents());
      CHECK(path.receive_.getComponents() == v2.getComponents());
      CHECK(path.R_distance_ == 10_m);
      CHECK(std::equal(
          P1.begin(), P1.end(), path.begin(),
          [](Point const& a, Point const& b) { return (a - b).getNorm() / 1_m < 1e-5; }));
    }

    NumericalIntegratingPropagator const SP2(env, 909_m);
    // get another path to different points
    auto const paths2_{SP2.propagate(particle1, p0, p30)};

    for (auto const& path : paths2_) {
      CHECK((path.propagation_time_ / 1_s) -
                (((p30 - p0).getNorm() / constants::c) / 1_s) ==
            Approx(0).margin(absMargin));
      CHECK(path.average_refractive_index_ == Approx(1));
      CHECK(path.refractive_index_source_ == Approx(1));
      CHECK(path.refractive_index_destination_ == Approx(1));
      CHECK(path.R_distance_ == 30000_m);
    }

    NumericalIntegratingPropagator const SP3(env, 731.89_m);
    // get a third path using a weird stepsize
    auto const paths3_{SP3.propagate(particle1, p0, p30)};

    for (auto const& path : paths3_) {
      CHECK((path.propagation_time_ / 1_s) -
                (((p30 - p0).getNorm() / constants::c) / 1_s) ==
            Approx(0).margin(absMargin));
      CHECK(path.average_refractive_index_ == Approx(1));
      CHECK(path.refractive_index_source_ == Approx(1));
      CHECK(path.refractive_index_destination_ == Approx(1));
      CHECK(path.R_distance_ == 30000_m);
    }

    CHECK(paths_.size() == 1);
    CHECK(paths2_.size() == 1);
    CHECK(paths3_.size() == 1);
  } // END: SECTION("Numerical Integrating Propagator w/ Uniform Refractive Index")

  SECTION("Numerical Integrating Propagator w/ Exponential Refractive Index") {

    // create an environment with exponential refractive index (n_0 = 1 & lambda = 0)
    using ExpoRIndex = ExponentialRefractiveIndex<
        HomogeneousMedium<IRefractiveIndexModel<IMediumModel>>>;

    using EnvType = Environment<IRefractiveIndexModel<IMediumModel>>;
    EnvType env1;

    // get another coordinate system
    const CoordinateSystemPtr rootCS1 = env1.getCoordinateSystem();

    // create a particle
    auto const particle{Code::Electron};
    const auto pmass{get_mass(particle)};

    // create a new stack for each trial
    setup::Stack<EnvType> stack;

    // construct an energy
    const HEPEnergyType E0{1_TeV};

    // compute the necessary momentum
    const HEPMomentumType P0{sqrt(E0 * E0 - pmass * pmass)};
    const auto plab{MomentumVector(rootCS1, {0_GeV, 0_GeV, P0})};

    Point const pp0(rootCS1, {0_m, 0_m, 0_m});

    // add the particle to the stack
    auto particle1{stack.addParticle(std::make_tuple(
        particle, calculate_kinetic_energy(plab.getNorm(), get_mass(particle)),
        plab.normalized(), pp0, 0_ns))};

    // the center of the earth
    Point const center1_{rootCS1, 0_m, 0_m, 0_m};
    LengthType const radius_{0_m};

    auto Medium1 = EnvType::createNode<Sphere>(
        Point{rootCS1, 0_m, 0_m, 0_m}, 1_km * std::numeric_limits<double>::infinity());

    auto const props1 = Medium1->setModelProperties<ExpoRIndex>(
        1, 0 / 1_m, center1_, radius_, 1_kg / (1_m * 1_m * 1_m),
        NuclearComposition({Code::Nitrogen}, {1.}));
    particle1.setNode(Medium1.get());

    env1.getUniverse()->addChild(std::move(Medium1));

    // get some points
    Point const pp1(rootCS1, {0_m, 0_m, 1_m});
    Point const pp2(rootCS1, {0_m, 0_m, 2_m});
    Point const pp3(rootCS1, {0_m, 0_m, 3_m});
    Point const pp4(rootCS1, {0_m, 0_m, 4_m});
    Point const pp5(rootCS1, {0_m, 0_m, 5_m});
    Point const pp6(rootCS1, {0_m, 0_m, 6_m});
    Point const pp7(rootCS1, {0_m, 0_m, 7_m});
    Point const pp8(rootCS1, {0_m, 0_m, 8_m});
    Point const pp9(rootCS1, {0_m, 0_m, 9_m});
    Point const pp10(rootCS1, {0_m, 0_m, 10_m});

    // get a unit vector
    Vector<dimensionless_d> vv1(rootCS1, {0, 0, 1});
    Vector<dimensionless_d> vv2(rootCS1, {0, 0, -1});

    // get a geometrical path of points
    Path const PP1({pp0, pp1, pp2, pp3, pp4, pp5, pp6, pp7, pp8, pp9, pp10});

    // construct a Numerical Integrating Propagator given the exponential refractive index
    // environment
    NumericalIntegratingPropagator const SP1(env1, 1_m);

    // store the outcome of Propagate method to paths1_
    auto const paths1_ = SP1.propagate(particle1, pp0, pp10);

    // perform checks to paths1_ components (this is just a sketch for now)
    for (auto const& path : paths1_) {
      CHECK((path.propagation_time_ / 1_s) -
                (((pp10 - pp0).getNorm() / constants::c) / 1_s) ==
            Approx(0).margin(absMargin));
      CHECK(path.average_refractive_index_ == Approx(1));
      CHECK(path.refractive_index_source_ == Approx(1));
      CHECK(path.refractive_index_destination_ == Approx(1));
      CHECK(path.emit_.getComponents() == vv1.getComponents());
      CHECK(path.receive_.getComponents() == vv2.getComponents());
      CHECK(path.R_distance_ == 10_m);
      CHECK(std::equal(
          PP1.begin(), PP1.end(), path.begin(),
          [](Point const& a, Point const& b) { return (a - b).getNorm() / 1_m < 1e-5; }));
    }

    CHECK(paths1_.size() == 1);

    /*
     * A second environment with another exponential refractive index
     */

    // create an environment with exponential refractive index (n_0 = 2 & lambda = 2)
    using ExpoRIndex = ExponentialRefractiveIndex<
        HomogeneousMedium<IRefractiveIndexModel<IMediumModel>>>;

    using EnvType = Environment<IRefractiveIndexModel<IMediumModel>>;
    EnvType env2;

    // get another coordinate system
    const CoordinateSystemPtr rootCS2 = env2.getCoordinateSystem();

    // the center of the earth
    Point const center2_{rootCS2, 0_m, 0_m, 0_m};

    auto Medium2 = EnvType::createNode<Sphere>(
        Point{rootCS2, 0_m, 0_m, 0_m}, 1_km * std::numeric_limits<double>::infinity());

    auto const props2 = Medium2->setModelProperties<ExpoRIndex>(
        2, 2 / 1_m, center2_, radius_, 1_kg / (1_m * 1_m * 1_m),
        NuclearComposition({Code::Nitrogen}, {1.}));
    particle1.setNode(Medium2.get());

    env2.getUniverse()->addChild(std::move(Medium2));

    // get some points
    Point const ppp0(rootCS2, {0_m, 0_m, 0_m});
    Point const ppp10(rootCS2, {0_m, 0_m, 10_m});

    // get a unit vector
    Vector<dimensionless_d> const vvv1(rootCS2, {0, 0, 1});
    Vector<dimensionless_d> const vvv2(rootCS2, {0, 0, -1});

    // construct a Numerical Integrating Propagator given the exponential refractive index
    // environment
    NumericalIntegratingPropagator const SP2(env2, 1_m);

    // store the outcome of Propagate method to paths1_
    auto const paths2_ = SP2.propagate(particle1, ppp0, ppp10);

    // perform checks to paths1_ components (this is just a sketch for now)
    for (auto const& path : paths2_) {
      CHECK((path.propagation_time_ / 1_s) -
                ((3.177511688_m / (3 * constants::c)) / 1_s) ==
            Approx(0).margin(absMargin));
      CHECK(path.average_refractive_index_ == Approx(0.210275935));
      CHECK(path.refractive_index_source_ == Approx(2));
      CHECK(path.refractive_index_destination_ == Approx(4.12231e-09));
      CHECK(path.emit_.getComponents() == vvv1.getComponents());
      CHECK(path.receive_.getComponents() == vvv2.getComponents());
      CHECK(path.R_distance_ == 10_m);
    }

    CHECK(paths2_.size() == 1);

  } // END: SECTION("Numerical Integrating Propagator w/ Exponential Refractive Index")

  SECTION("Tabulated Flat Atmosphere Propagator w/ Uniform Refractive Index") {

    // get a CS
    CoordinateSystemPtr const& rootCS = get_root_CoordinateSystem();

    // the center of the earth
    Point const center_{rootCS, 0_m, 0_m, 0_m};

    // a point at the surface of the earth
    Point const surface_{rootCS, 0_m, 0_m, constants::EarthRadius::Mean};

    // the refractive index at sea level
    const double n0{1.000327};

    // setup a 5-layered environment
    using EnvironmentInterface =
        IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
    using EnvType = Environment<EnvironmentInterface>;
    EnvType env;

    create_5layer_atmosphere<EnvironmentInterface, MyExtraEnv>(
        env, AtmosphereId::LinsleyUSStd, center_, n0, Medium::AirDry1Atm,
        MagneticFieldVector{rootCS, 0_T, 50_uT, 0_T});

    // create a particle
    auto const particle{Code::Electron};
    const auto pmass{get_mass(particle)};

    // create a new stack for each trial
    setup::Stack<EnvType> stack;

    // construct an energy
    const HEPEnergyType E0{1_TeV};

    // compute the necessary momentum
    const HEPMomentumType P0{sqrt(E0 * E0 - pmass * pmass)};
    const auto plab{MomentumVector(rootCS, {0_GeV, 0_GeV, P0})};

    // and create the location of the particle in this coordinate system
    const Point pos(rootCS, 50_m, 10_m, 80_m);

    // add the particle to the stack
    auto const particle1{stack.addParticle(std::make_tuple(
        particle, calculate_kinetic_energy(plab.getNorm(), get_mass(particle)),
        plab.normalized(), pos, 0_ns))};

    // get some points
    Point const upperBoundary_(rootCS, {0_m, 0_m, constants::EarthRadius::Mean + 10_m});
    Point const p0(rootCS, {0_m, 0_m, constants::EarthRadius::Mean});
    Point const p10(rootCS, {0_m, 0_m, constants::EarthRadius::Mean + 10_m});

    // get a unit vector
    Vector<dimensionless_d> const v1(rootCS, {0, 0, 1});
    Vector<dimensionless_d> const v2(rootCS, {0, 0, -1});

    // get a geometrical path of points
    Path const P1({p0, p10});

    LengthType const step_{1_m};

    // construct a Tabulated Flat Atmosphere Propagator given the uniform refractive index
    // environment
    auto SP =
        make_tabulated_flat_atmosphere_radio_propagator(env, upperBoundary_, p0, step_);

    // store the outcome of the Propagate method to paths_
    auto paths_ = SP.propagate(particle1, p0, p10);

    // perform checks to paths_ components
    for (auto const& path : paths_) {
      CHECK((path.propagation_time_ / 1_s) -
                (((p10 - p0).getNorm() / constants::c) / 1_s) ==
            Approx(1.09075e-11));
      CHECK(path.average_refractive_index_ == n0);
      CHECK(path.refractive_index_source_ == n0);
      CHECK(path.refractive_index_destination_ == n0);
      CHECK(path.emit_.getComponents() == v1.getComponents());
      CHECK(path.receive_.getComponents() == v2.getComponents());
      CHECK(path.R_distance_ == 10_m);
      CHECK(std::equal(
          P1.begin(), P1.end(), path.begin(),
          [](Point const& a, Point const& b) { return (a - b).getNorm() / 1_m < 1e-5; }));
    }
  } // END: SECTION("Tabulated Flat Atmosphere Propagator w/ Uniform Refractive Index")

  SECTION("Tabulated Flat Atmosphere Propagator w/ Gladstone-Dale Refractive Index") {

    // get a CS
    CoordinateSystemPtr const& rootCS = get_root_CoordinateSystem();

    // the center of the earth
    Point const center_{rootCS, 0_m, 0_m, 0_m};

    // a point at the surface of the earth
    Point const surface_{rootCS, 0_m, 0_m, constants::EarthRadius::Mean};

    // the refractive index at sea level
    const double n0{1.000327};

    // setup a 5-layered environment
    using EnvironmentInterface =
        IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
    using EnvType = Environment<EnvironmentInterface>;
    EnvType env;

    create_5layer_atmosphere<EnvironmentInterface, MyExtraEnv2>(
        env, AtmosphereId::LinsleyUSStd, center_, n0, surface_, Medium::AirDry1Atm,
        MagneticFieldVector{rootCS, 0_T, 0_uT, 0_T});

    // create a particle
    auto const particle{Code::Electron};
    const auto pmass{get_mass(particle)};

    // create a new stack for each trial
    setup::Stack<EnvType> stack;

    // construct an energy
    const HEPEnergyType E0{1_TeV};

    // compute the necessary momentum
    const HEPMomentumType P0{sqrt(E0 * E0 - pmass * pmass)};
    const auto plab{MomentumVector(rootCS, {0_GeV, 0_GeV, P0})};

    // and create the location of the particle in this coordinate system
    const Point pos(rootCS, 50_m, 10_m, 80_m);

    // add the particle to the stack
    auto const particle1{stack.addParticle(std::make_tuple(
        particle, calculate_kinetic_energy(plab.getNorm(), get_mass(particle)),
        plab.normalized(), pos, 0_ns))};

    // get some points
    Point const upperBoundary_(rootCS, {0_m, 0_m, constants::EarthRadius::Mean + 10_m});
    Point const p0(rootCS, {0_m, 0_m, constants::EarthRadius::Mean});
    Point const p10(rootCS, {0_m, 0_m, constants::EarthRadius::Mean + 10_m});

    // get a unit vector
    Vector<dimensionless_d> const v1(rootCS, {0, 0, 1});
    Vector<dimensionless_d> const v2(rootCS, {0, 0, -1});

    // get a geometrical path of points
    Path const P1({p0, p10});

    LengthType const step_{1_m};

    // construct a Tabulated Flat Atmosphere Propagator given the uniform refractive index
    // environment
    TabulatedFlatAtmospherePropagator SP(env, upperBoundary_, p0, step_);

    // store the outcome of the Propagate method to paths_
    auto paths_ = SP.propagate(particle1, p0, p10);

    // perform checks to paths_ components
    for (auto const& path : paths_) {
      CHECK((path.propagation_time_ / 1_s) == Approx(3.33673e-08));
      CHECK(path.average_refractive_index_ == Approx(1.0003268356));
      CHECK(path.refractive_index_source_ == Approx(n0));
      CHECK(path.refractive_index_destination_ == Approx(1.0003266713));
      CHECK(path.emit_.getComponents() == v1.getComponents());
      CHECK(path.receive_.getComponents() == v2.getComponents());
      CHECK(path.R_distance_ == 10_m);
      CHECK(std::equal(
          P1.begin(), P1.end(), path.begin(),
          [](Point const& a, Point const& b) { return (a - b).getNorm() / 1_m < 1e-5; }));
    }
  } // END: SECTION("Tabulated Flat Atmosphere Propagator w/ Gladstone-Dale Refractive
    // Index")

} // END: TEST_CASE("Propagators")
