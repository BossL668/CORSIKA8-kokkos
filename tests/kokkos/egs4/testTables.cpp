/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/egs4/EmbeddedTables.hpp>
#include <corsika/kokkos/egs4/Session.hpp>
#include <corsika/kokkos/egs4/Tables.hpp>
#include <corsika/detail/kokkos/egs4/Egs4Transport.inl>
#include <catch2/catch_all.hpp>
#include <array>
#include <numeric>

namespace {
  namespace eg = corsika::kokkos::egs4;
  namespace app = eg::application;

  app::Configuration configuration() {
    app::Configuration c;
    c.earthRadiusM_ = 6.371315e6;
    c.seed_ = 1931;
    c.firstChildId_ = 100;
    c.maximumWaves_ = 100000;
    c.queueCapacity_ = 65536;
    auto& e = c.environment_;
    e.numberOfLayers_ = 5;
    double const heights[] = {0., 4000., 10000., 40000., 100000., 112800.};
    double const offsets[] = {1222.6562, 1144.9069, 1305.5948, 540.1778, 1.};
    double const scales[] = {994186.38, 878153.55, 636143.04, 772170.16, 1.e9};
    for (unsigned i = 0; i < 5; ++i) {
      auto& layer = e.atmosphereLayers_[i];
      layer.innerRadiusM_ = c.earthRadiusM_ + heights[i];
      layer.outerRadiusM_ = c.earthRadiusM_ + heights[i + 1];
      layer.densityParameterA_ = offsets[i] / scales[i];
      layer.densityParameterB_ = c.earthRadiusM_;
      layer.densityParameterC_ = -scales[i] * .01;
      layer.mediumId_ = 3;
      layer.densityModel_ =
          i < 4 ? eg::DensityModel::Exponential : eg::DensityModel::Homogeneous;
    }
    e.observationPlaneNormal_[2] = 1.;
    e.observationPlanePointM_[2] = c.earthRadiusM_ + 1100.;
    e.magneticFieldT_[0] = 2.e-5;
    e.magneticFieldT_[1] = 1.e-5;
    e.magneticFieldT_[2] = -4.e-5;
    return c;
  }
} // namespace

TEST_CASE("Embedded EGS4 table preserves transport and HATCH cut handling", "[egs4]") {
  auto const tables = eg::read_air_tables(CORSIKA_EGS4_TEST_TABLE);
  auto embedded = app::make_embedded_session();
  app::Session external(tables);
  REQUIRE(embedded->sternheimerCorrection() == external.sternheimerCorrection());
  REQUIRE(std::string(app::embedded_table_sha256()).size() == 64);
  auto cfg = configuration();
  cfg.sterncor_ = embedded->sternheimerCorrection();
  for (int pid : {11, -11, 22}) {
    corsika::kokkos::ParticleState particle;
    particle.pid_ = pid;
    particle.mediumId_ = 3;
    particle.energyGeV_ = .02;
    particle.weight_ = 1.25;
    particle.historyId_ = 9;
    particle.stepId_ = 4;
    particle.positionM_[2] = cfg.earthRadiusM_ + 4000.;
    particle.direction_[0] = .3;
    particle.direction_[1] = .4;
    particle.direction_[2] = -std::sqrt(.75);
    auto run = [&](app::Session& session) {
      std::vector<double> trace;
      app::OutputCallbacks output;
      output.step_ = [&](auto const& s) {
        trace.insert(trace.end(), {s.startEnergyGeV_, s.endEnergyGeV_,
                                   s.depositedEnergyGeV_, s.weight_});
      };
      output.radio_ = [&](auto const& r) { trace.push_back(r.step_.weight_); };
      output.observation_ = [&](auto const& r) {
        trace.push_back(r.particle_.energyGeV_);
      };
      output.discarded_ = [&](double energy, unsigned reason) {
        trace.insert(trace.end(), {energy, double(reason)});
      };
      auto const result = session.run(cfg, {particle}, output);
      REQUIRE(result.steps_ > 0);
      trace.insert(trace.end(), {double(result.steps_), double(result.children_)});
      return trace;
    };
    REQUIRE(run(*embedded) == run(external));
    auto const requested = cfg;
    cfg.electronTotalCutMeV_ = cfg.electronMassMeV_;
    cfg.photonCutMeV_ = 0.;
    auto const belowThreshold = run(*embedded);
    cfg.electronTotalCutMeV_ =
        std::max(tables.medium_.ae_, tables.medium_.ap_ + cfg.electronMassMeV_);
    cfg.photonCutMeV_ = tables.medium_.ap_;
    REQUIRE(belowThreshold == run(external));
    cfg = requested;
  }
  // C7 EGSIN2 ties its density correction to AP, rather than to the transport cut.
  constexpr std::array<std::pair<double, double>, 6> corrections{
      {{3., 0.}, {1., 6.}, {.4, 10.}, {.25, 11.}, {.15, 12.5}, {.05, 15.}}};
  for (auto const& pair : corrections) {
    auto metadata = tables;
    metadata.medium_.ap_ = pair.first;
    app::Session session(metadata);
    REQUIRE(session.sternheimerCorrection() == pair.second);
  }
  auto unsupported = tables;
  unsupported.medium_.ap_ = .123;
  app::Session session(unsupported);
  REQUIRE_THROWS_AS(session.sternheimerCorrection(), std::invalid_argument);
  auto const model = eg::model_view(tables, {.51099895, 152., 152., 422.});
  auto const cuts = eg::transport_settings(model, 0., 0.);
  REQUIRE(cuts.photonCutMeV_ == tables.medium_.ap_);
  REQUIRE(cuts.electronCutTotalMeV_ ==
          std::max(tables.medium_.ae_, tables.medium_.ap_ + .51099895));

  // Moving outputs to device accumulators must retain the same transported cascade.
  corsika::kokkos::ParticleState particle;
  particle.pid_ = 11;
  particle.mediumId_ = 3;
  particle.energyGeV_ = .02;
  particle.weight_ = 1.;
  particle.historyId_ = 9;
  particle.positionM_[2] = cfg.earthRadiusM_ + 4000.;
  particle.direction_[2] = -1.;
  app::OutputCallbacks output;
  double deposited = 0.;
  std::uint64_t radioTracks = 0;
  output.step_ = [&](auto const& s) { deposited += s.weight_ * s.depositedEnergyGeV_; };
  output.radio_ = [&](auto const&) { ++radioTracks; };
  output.observation_ = [](auto const&) {};
  output.discarded_ = [](double, unsigned) {};
  auto const reference = embedded->run(cfg, {particle}, output);
  REQUIRE(deposited > 0.);
  REQUIRE(radioTracks > 0);
  auto& profile = cfg.profile_;
  profile.enabled_ = profile.accumulateOnDevice_ = true;
  profile.axisStartPositionM_[2] = cfg.earthRadiusM_ + 5000.;
  profile.axisDirection_[2] = -1.;
  profile.axisStepLengthM_ = 5000.;
  profile.axisGrammageGPerCm2_ = {0., 500.};
  profile.outputBinCount_ = 100;
  profile.outputBinWidthGPerCm2_ = 10.;
  profile.fixedPointEnergyLimitGeV_ = 1.;
  profile.fixedPointWeightLimit_ = 1.e4;
  auto& radio = cfg.radio_;
  radio.enabled_ = true;
  radio.propagation_.homogeneousRefractiveIndex_ = 1.0003;
  corsika::kokkos::radio::RadioObserverSnapshot observer;
  observer.positionM_[0] = 50.;
  observer.positionM_[2] = cfg.earthRadiusM_ + 1100.;
  observer.startTimeS_ = 0.;
  observer.durationS_ = 1.e-4;
  observer.sampleRateHz_ = 1.e9;
  observer.numberOfBins_ = 100000;
  radio.coreasObservers_ = radio.zhsObservers_ = {observer};
  bool profileReceived = false, radioReceived = false;
  output.step_ = [](auto const&) { FAIL("Device profile leaked per-step callbacks"); };
  output.radio_ = [](auto const&) { FAIL("Device radio leaked per-track callbacks"); };
  output.profile_ = [&](auto const& result) {
    profileReceived = true;
    REQUIRE(result.steps_ == reference.steps_);
    REQUIRE(result.invalidRecords_ == 0);
    REQUIRE(result.fixedPointOverflows_ == 0);
    double const total =
        std::accumulate(result.energyLossGeV_.begin(), result.energyLossGeV_.end(), 0.);
    REQUIRE(std::abs(total - deposited) < 1.e-12);
  };
  output.radioWaveforms_ = [&](auto const& waveforms, auto count) {
    radioReceived = true;
    REQUIRE(count == radioTracks);
    REQUIRE(waveforms.coreas_.size() == 1);
    REQUIRE(waveforms.zhs_.size() == 1);
    for (auto const* waves : {&waveforms.coreas_, &waveforms.zhs_}) {
      double norm = 0.;
      for (auto const* component :
           {&waves->front().x_, &waves->front().y_, &waves->front().z_}) {
        REQUIRE(component->size() == observer.numberOfBins_);
        for (double x : *component) {
          REQUIRE(std::isfinite(x));
          norm += std::abs(x);
        }
      }
      REQUIRE(norm > 0.);
    }
  };
  auto const resident = embedded->run(cfg, {particle}, output);
  REQUIRE(resident.steps_ == reference.steps_);
  REQUIRE(resident.children_ == reference.children_);
  REQUIRE(resident.radio_.fixedPointOverflows_ == 0);
  REQUIRE(resident.profileExecutionSpace_ == resident.executionSpace_);
  REQUIRE(resident.radioExecutionSpace_ == resident.executionSpace_);
  REQUIRE(profileReceived);
  REQUIRE(radioReceived);
}
