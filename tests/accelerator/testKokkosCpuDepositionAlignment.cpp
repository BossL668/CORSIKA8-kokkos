/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "KokkosCpuDepositionAlignmentDriver.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <PROPOSAL/Constants.h>
#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/particle/ParticleDef.h>
#include <PROPOSAL/secondaries/parametrization/photoeffect/PhotoeffectNoDeflection.h>

#include <corsika/accelerator/em/common/CorsikaOutputSink.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>
#include <corsika/accelerator/em/common/ProcessCapabilities.hpp>
#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <corsika/modules/writers/EnergyLossWriter.hpp>
#include <corsika/modules/writers/SubWriter.hpp>

namespace {
  using namespace corsika;
  namespace em = corsika::gpu::em;
  using namespace corsika::accelerator::em::testing;
  constexpr std::size_t Bins = 8;
  constexpr double Scale = 1.e15;

  void require(bool value, std::string const& message) {
    if (!value) throw std::runtime_error(message);
  }
  void near(double actual, double expected, std::string const& message) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > 2.e-11 * std::max(1., std::abs(expected)))
      throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                               ", expected=" + std::to_string(expected));
  }

  // This output stores rows emitted by the real CPU EnergyLossWriter. No copy
  // of the CPU binning formula is used as the expected-value implementation.
  struct CaptureOutput : WriterOff {
    using WriterOff::WriterOff;
    std::vector<double> rows;
    void write(unsigned int, GrammageType, dEdX_output::Profile const& values) {
      rows.push_back(values[0] / 1_GeV);
    }
  };
  using Writer = EnergyLossWriter<CaptureOutput>;
  struct UnusedOutput {
    template <class... Args> void write(Args const&...) {}
    template <class... Args> void writeProjected(Args const&...) {}
  };
  struct EndpointTrajectory {
    Point endpoint;
    DirectionVector direction;
    TimeType getDuration(double) const { return 1_ns; }
    Point getPosition(double) const { return endpoint; }
    DirectionVector getDirection(double) const { return direction; }
  };
  using Sink = em::CorsikaOutputSink<Writer, UnusedOutput, UnusedOutput,
                                    UnusedOutput, UnusedOutput, UnusedOutput,
                                    UnusedOutput>;

  struct Fixture {
    Environment<IMediumModel> environment;
    CoordinateSystemPtr cs{environment.getCoordinateSystem()};
    std::unique_ptr<ShowerAxis> axis;
    std::vector<double> grammage;
    double axis_step{};

    Fixture() {
      auto medium = Environment<IMediumModel>::createNode<Sphere>(
          Point{cs, 0_m, 0_m, 0_m},
          std::numeric_limits<double>::infinity() * 1_m);
      medium->setModelProperties<HomogeneousMedium<IMediumModel>>(
          10_kg / (1_m * 1_m * 1_m), NuclearComposition({Code::Nitrogen}, {1.}));
      environment.getUniverse()->addChild(std::move(medium));
      axis = std::make_unique<ShowerAxis>(Point{cs, 0_m, 0_m, 0_m},
          LengthVector{cs, {100_m, 0_m, 0_m}}, environment, false, 100);
      auto const& support = axis->getGrammageSupport();
      for (auto value : support) grammage.push_back(value / (1_g / square(1_cm)));
      axis_step = axis->getSteplength() / 1_m;
    }
  };

  void compareRows(std::vector<double> const& actual, std::vector<double> const& expected,
                   std::string const& path) {
    require(actual.size() == Bins && expected.size() == Bins, path + " row count");
    for (std::size_t i = 0; i < Bins; ++i)
      near(actual[i], expected[i], path + " bin " + std::to_string(i));
  }

  void checkCut(Fixture& f, Code code, double x0, double x1, double threshold,
                double weight, bool observed) {
    auto const photon = code == Code::Photon;
    auto const muon = is_muon(code);
    auto const cut_energy = muon ? .29995 : .00049995;
    auto const continuous_energy = photon || x0 == x1 ? 0. : .00004;
    auto const mass = get_mass(code) / 1_GeV;
    Point const start{f.cs, x0 * 1_m, 0_m, 0_m};
    Point const end{f.cs, x1 * 1_m, 0_m, 0_m};
    DirectionVector const direction{f.cs, {x1 >= x0 ? 1. : -1., 0., 0.}};
    // A real CPU Step followed by the real ParticleCut calls the scalar
    // point-writer overload; the continuous deposit is provided independently.
    Writer cpu(*f.axis, Bins, 10_g / square(1_cm), threshold * 1_g / square(1_cm));
    cpu.startOfShower(0);
    em::output_detail::RadioTrackParticle particle(code,
        (continuous_energy + cut_energy) * 1_GeV, direction, start, 0_ns, weight);
    EndpointTrajectory trajectory{end, direction};
    Step step(particle, trajectory);
    step.add_dEkin(-continuous_energy * 1_GeV);
    if (continuous_energy > 0.)
      cpu.write(start, end, code, continuous_energy * weight * 1_GeV);
    ParticleCut<SubWriter<Writer>> cut(.5_MeV, .5_MeV, .3_GeV, .3_GeV, .3_GeV,
                                      true, cpu);
    require(cut.doContinuous(step) == ProcessReturn::ParticleAbsorbed,
            "scalar ParticleCut must absorb fixture");
    cpu.endOfShower(0);

    em::LeptonTransportRecord lepton{};
    lepton.start.pid = static_cast<std::int32_t>(get_PDG(code));
    lepton.start.position_m[0] = x0;
    lepton.start.energy_GeV = mass + cut_energy + continuous_energy;
    lepton.start.weight = weight;
    lepton.end = lepton.start;
    lepton.end.position_m[0] = x1;
    lepton.end.energy_GeV = mass + cut_energy;
    lepton.end.time_s = 1.e-9;
    lepton.continuous_deposited_energy_GeV = continuous_energy;
    lepton.cut_deposited_energy_GeV = cut_energy;
    lepton.limit = em::LeptonTransportLimit::ParticleCut;
    lepton.observation_surface_reached_before_cut = observed ? 1U : 0U;
    auto const actual = runCpuDepositionDeviceStep(
        f.grammage, f.axis_step, lepton, photon, threshold);
    compareRows(actual.energy, cpu.rows, "resident scalar writer oracle");
    auto const counters = actual.counters;
    require(counters.fixed_point_overflows == 0 && counters.invalid_records == 0,
            "resident record or fixed-point error");
    near(static_cast<double>(counters.weighted_observation_cut_overlap_energy) / Scale,
         observed ? (cut_energy + mass) * weight : 0., "terminal overlap ledger");
    near(static_cast<double>(counters.weighted_observed_total_energy) / Scale,
         observed ? (cut_energy + mass) * weight : 0., "observed total ledger");

    Writer sink_writer(*f.axis, Bins, 10_g / square(1_cm),
                       threshold * 1_g / square(1_cm));
    UnusedOutput unused;
    Sink sink(f.cs, sink_writer, unused, unused, unused, unused, unused, unused, false);
    sink_writer.startOfShower(0);
    em::EmStepRecord full{};
    full.pid = lepton.start.pid;
    full.start_position_m[0] = x0;
    full.end_position_m[0] = x1;
    full.end_time_s = 1.e-9;
    full.start_energy_GeV = lepton.start.energy_GeV;
    full.end_energy_GeV = lepton.end.energy_GeV;
    full.deposited_energy_GeV = continuous_energy + cut_energy;
    full.cut_deposited_energy_GeV = cut_energy;
    full.weight = weight;
    sink.onStep(full);
    sink_writer.endOfShower(0);
    compareRows(sink_writer.rows, cpu.rows, "full-record sink scalar writer oracle");
    sink_writer.rows.clear();
    sink_writer.startOfShower(1);
    require(actual.projected.observation_surface_reached_before_cut == (observed ? 1U : 0U),
            "projected record lost observation/cut marker");
    sink.onProjectedStep(actual.projected);
    sink_writer.endOfShower(1);
    compareRows(sink_writer.rows, cpu.rows, "projected sink scalar writer oracle");
  }

  void checkPhotoelectric(Fixture& f) {
    PROPOSAL::GammaDef photon;
    PROPOSAL::Air air;
    PROPOSAL::secondaries::PhotoeffectNoDeflection oracle(photon, air);
    for (auto const& component : air.GetComponents()) {
      double const incident_GeV = .001;
      PROPOSAL::StochasticLoss loss(em::PhotoelectricProcessId, incident_GeV * 1000.,
          PROPOSAL::Cartesian3D(0., 0., 0.), PROPOSAL::Cartesian3D(1., 0., 0.),
          0., 0., incident_GeV * 1000., component.GetHash());
      std::vector<double> random;
      auto const secondaries = oracle.CalculateSecondaries(loss, component, random);
      require(secondaries.size() == 1, "photoelectric CPU oracle multiplicity");
      auto const cpu_electron_energy = secondaries[0].energy / 1000.;
      auto const binding = incident_GeV + PROPOSAL::ME / 1000. - cpu_electron_energy;
      em::PhotonPairLpmSnapshot snapshot{};
      snapshot.component_count = 1;
      snapshot.fine_structure_constant = PROPOSAL::ALPHA;
      snapshot.components[0].component_hash = component.GetHash();
      snapshot.components[0].nuclear_charge = component.GetNucCharge();
      em::PhotonTransportRecord transport{};
      transport.start.pid = 22;
      transport.start.weight = 17.;
      transport.start.energy_GeV = incident_GeV;
      transport.start.position_m[0] = 9.99;
      transport.start.history_id = 5;
      transport.end = transport.start;
      transport.end.position_m[0] = 10.01;
      transport.interaction.process_id = em::PhotoelectricProcessId;
      transport.limit = em::PhotonTransportLimit::Interaction;
      auto const actual = runCpuPhotoelectricDeviceStep(
          f.grammage, f.axis_step, transport, snapshot);
      near(actual.photoelectron_energy_GeV, cpu_electron_energy,
           "PROPOSAL photoelectric secondary energy");
      for (auto const value : actual.energy) near(value, 0., "legacy photoelectric dE/dX");
      auto const counters = actual.counters;
      near(static_cast<double>(counters.weighted_unwritten_photoelectric_binding_energy) / Scale,
           binding * 17., "independent PROPOSAL binding-energy ledger");
      near(static_cast<double>(counters.weighted_mass_convention_correction) / Scale,
           17. * (em::TransportElectronMassGeV - PROPOSAL::ME / 1000.),
           "explicit final-state mass-convention ledger");
      require(counters.fixed_point_overflows == 0 && counters.invalid_records == 0,
              "photoelectric accounting error");
    }
  }
} // namespace

int main() {
  try {
    corsika::accelerator::em::KokkosRuntimeConfig config;
#if defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
    config.threads = 2;
#endif
    corsika::accelerator::em::KokkosRuntime runtime(config);
    corsika::logging::set_level(corsika::logging::level::warn);
    Fixture fixture;
    std::size_t cases{};
    for (auto const code : {Code::Electron, Code::Positron, Code::MuMinus,
                            Code::MuPlus, Code::Photon})
      for (auto const endpoints : {std::array<double, 2>{9.99, 10.01},
              std::array<double, 2>{10.01, 9.99}, {9.99999, 10.00001},
              {10., 20.}, {75., 95.}, {5., 25.}, {10., 10.}})
        for (double const threshold : {0., 1.e-4})
          for (double const weight : {.25, 1., 17.})
            for (bool const observed : {false, true}) {
              checkCut(fixture, code, endpoints[0], endpoints[1], threshold, weight, observed);
              ++cases;
            }
    checkPhotoelectric(fixture);
    std::cout << "PASS: " << cases << " endpoint/segment cases; real scalar "
                 "EnergyLossWriter + ParticleCut, resident/full/projected outputs; "
                 "PROPOSAL photoelectric final-state oracle and separate binding ledger\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
