/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/radio/firn/IGeometricalOpticsAccumulator.hpp>
#include <corsika/kokkos/radio/firn/CoherentStratifiedGOBackend.hpp>
#include <corsika/kokkos/radio/firn/ExponentialProfile.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <iostream>
#include <optional>

using namespace corsika::kokkos::radio::firn;

namespace {
  void check_endpoint_continuity() {
    // Independent 80-digit integration of n(z)=1.78-0.423 exp(z/77 m).
    // Rows straddle the old p cutoffs, including the exact endpoint junction.
    double const expected[][4] = {
        {565.3372087331841, 579.4786530917369, 3.298969656110204e-6,
         0.0021883925323786285},
        {566.425913718181, 580.5672149719704, 3.3050145864328036e-6,
         0.002187318440292072},
        {566.4481215033529, 580.5894226998048, 3.305137893057334e-6,
         0.0021872968254553287},
        {566.4703288566636, 580.611630110456, 3.305261197283997e-6, 0.0021872752228534},
        {567.5579546168541, 581.6993993455321, 3.3113001353145378e-6,
         0.0021862317056672792}};
    auto profile = std::make_shared<ExponentialProfile const>(1.78, .423, 77., 0.);
    CoherentStratifiedGOBackend cpu(profile, 1.);
    GeometricalOpticsConfig config;
    config.radio_.grid_ = {1.e-9, 64};
    std::vector<vector_type> observers;
    for (auto const& row : expected) observers.push_back({row[0], 0., -100.});
    vector_type const source{0., 0., -215.};
    auto device =
        make_geometrical_optics_accumulator(profile->parameters(), observers, config);
    auto paths = device->paths({source});
    for (std::size_t i = 0; i < observers.size(); ++i) {
      auto const response = cpu.evaluate(source, observers[i], config.radio_.grid_);
      REQUIRE(response.branches_.size() == 2);
      REQUIRE(paths[i].size() == 2);
      std::size_t ordinary = 0;
      for (auto const& path : paths[i]) {
        if (path.topology_ == "surface_reflected") continue;
        ++ordinary;
        CHECK(std::abs(path.geometricLengthM_ / expected[i][1] - 1.) < 1.e-10);
        CHECK(std::abs(path.travelTimeS_ / expected[i][2] - 1.) < 1.e-10);
        CHECK(std::abs(path.geometricSpreadingPerM_ / expected[i][3] - 1.) < 1.e-8);
        CHECK(std::abs(norm(path.emitDirection_) - 1.) < 1.e-12);
        CHECK(std::abs(norm(path.receiveDirection_) - 1.) < 1.e-12);
        CHECK(path.fermatRelativeResidual_ < 1.e-12);
        CHECK(path.phaseOffsetRad_ == 0.);
      }
      CHECK(ordinary == 1);
      for (auto const& branch : response.branches_) {
        auto const& path = branch.path_;
        if (path.topology_ == "surface_reflected") continue;
        CHECK(std::abs(path.geometricSpreadingPerM_ / expected[i][3] - 1.) < 1.e-8);
        CHECK(std::abs(path.travelTimeS_ / expected[i][2] - 1.) < 1.e-10);
      }
      // Reciprocity uses the same Green-function spreading, with reversed rays.
      auto const reverse = cpu.evaluate(observers[i], source, config.radio_.grid_);
      REQUIRE(reverse.branches_.size() == 2);
      for (auto const& branch : reverse.branches_) {
        if (branch.path_.topology_ == "surface_reflected") continue;
        CHECK(std::abs(branch.path_.geometricSpreadingPerM_ / expected[i][3] - 1.) <
              1.e-8);
      }
    }
  }

} // namespace

namespace {
  using spectra_type = GeometricalOpticsSpectra;
  double l2(std::vector<spectra_type::components_type> const& a,
            std::vector<spectra_type::components_type> const& b) {
    double error = 0., reference = 0.;
    for (std::size_t o = 0; o < a.size(); ++o)
      for (int c = 0; c < 3; ++c)
        for (std::size_t f = 0; f < a[o][c].size(); ++f) {
          error += std::norm(a[o][c][f] - b[o][c][f]);
          reference += std::norm(b[o][c][f]);
        }
    return std::sqrt(error / std::max(reference, 1.e-300));
  }
  std::vector<spectra_type::components_type> branch(spectra_type const& s, int b) {
    std::vector<spectra_type::components_type> out;
    for (auto const& o : s.branches_) out.push_back(o[b]);
    return out;
  }
  // Independent CPU oracle: the unchanged GreenResponse backend supplies all
  // branches/dyadics. Match VariableIndexRadioProcess's subdivision and phase.
  spectra_type reference(std::vector<GeometricalOpticsTrack> const& tracks,
                         std::vector<vector_type> const& observers,
                         CoherentStratifiedGOBackend const& backend,
                         OnlineRadioConfig const& config, std::size_t& childrenCount) {
    spectra_type result;
    result.total_.resize(observers.size());
    result.branches_.resize(observers.size());
    for (auto& o : result.total_)
      for (auto& c : o) c.resize(config.grid_.frequencyCount());
    for (auto& o : result.branches_)
      for (auto& b : o)
        for (auto& c : b) c.resize(config.grid_.frequencyCount());
    for (auto const& track : tracks) {
      auto displacement = subtract(track.endM_, track.startM_);
      double duration = track.endTimeS_ - track.startTimeS_, length = norm(displacement),
             curvature = 0.;
      auto midpoint = add(track.startM_, multiply(displacement, .5));
      std::vector<GreenResponseBatch> mid;
      for (auto const& o : observers) {
        mid.push_back(backend.evaluate(midpoint, o, config.grid_));
        for (auto const& b : mid.back().branches_) {
          double distance = 1. / std::max(b.path_.geometricSpreadingPerM_, 1.e-30);
          curvature = std::max(curvature, config.grid_.nyquistHz() *
                                              config.maximumRefractiveIndex_ * length *
                                              length / (8. * c_m_per_s * distance));
        }
      }
      std::size_t count = std::max(
          1., std::max(std::ceil(length / config.maximumLinearizationLengthM_),
                       std::ceil(std::sqrt(curvature / config.maximumPhaseCycles_))));
      childrenCount += count;
      for (std::size_t child = 0; child < count; ++child) {
        double f0 = double(child) / count, f1 = double(child + 1) / count,
               fm = .5 * (f0 + f1);
        auto start = add(track.startM_, multiply(displacement, f0));
        auto end = add(track.startM_, multiply(displacement, f1));
        auto delta = subtract(end, start);
        auto center = add(track.startM_, multiply(displacement, fm));
        double time = track.startTimeS_ + fm * duration;
        for (std::size_t o = 0; o < observers.size(); ++o) {
          auto response =
              count == 1 ? mid[o] : backend.evaluate(center, observers[o], config.grid_);
          for (auto const& b : response.branches_) {
            auto const& p = b.path_;
            int family = p.topology_ == "direct"
                             ? 0
                             : p.topology_ == "surface_reflected"
                                   ? 1
                                   : p.topology_ == "turning_short"
                                         ? 2
                                         : p.topology_ == "turning_long" ? 3 : 6;
            double retarded = duration / count - p.sourceRefractiveIndex_ *
                                                     dot(p.emitDirection_, delta) /
                                                     c_m_per_s;
            for (std::size_t f = 1; f < config.grid_.frequencyCount(); ++f) {
              double frequency = config.grid_.frequencyHz(f),
                     arg = pi * frequency * retarded;
              double sinc =
                  std::abs(arg) < 1.e-8 ? 1. - arg * arg / 6. : std::sin(arg) / arg;
              auto projected = contract(b.dyadicPerM_[f], delta);
              auto common = mu0_h_per_m * elementary_charge_c * track.chargeWeight_ /
                            (4. * pi) * std::complex<double>(0., -2. * pi * frequency) *
                            sinc *
                            std::exp(std::complex<double>(
                                0., -2. * pi * frequency * (time - config.timeOriginS_)));
              for (int c = 0; c < 3; ++c) {
                result.total_[o][c][f] += common * projected[c];
                result.branches_[o][family][c][f] += common * projected[c];
              }
            }
          }
        }
      }
    }
    return result;
  }
} // namespace

namespace {
  void check_branch_completeness() {
    auto profile = std::make_shared<ExponentialProfile const>(1.78, .423, 77., 0.);
    // Independent analytic root brackets; both roots lie inside one old scan cell.
    double const cases[][5] = {
        {-215., -100., 744.9455024359556, 1.594034870189314, 1.5942420138251752},
        {-335., -800., 7615.5059353029155, 1.7737867939041592, 1.774034027743452},
        {-100., -100., 504.83994519326643, 1.4893302855026818, 1.4896858238791608}};
    FrequencyGrid grid{1.e-9, 64};
    CoherentStratifiedGOBackend guarded(profile, 1.);
    CoherentStratifiedGOBackend unguarded(profile, 1., true, true, 0.);
    for (auto const& row : cases) {
      vector_type source{0., 0., row[0]}, observer{row[2], 0., row[1]};
      CHECK_THROWS(guarded.evaluate(source, observer, grid));
      GeometricalOpticsConfig config;
      config.radio_.grid_ = grid;
      auto guardedDevice =
          make_geometrical_optics_accumulator(profile->parameters(), {observer}, config);
      CHECK_THROWS(guardedDevice->paths({source}));
      // Disable the wave-model guard only to inspect ray-root completeness.
      config.foldGuardTimeS_ = 0.;
      auto device =
          make_geometrical_optics_accumulator(profile->parameters(), {observer}, config);
      auto paths = device->paths({source});
      auto response = unguarded.evaluate(source, observer, grid);
      REQUIRE(response.branches_.size() == 2);
      REQUIRE(paths[0].size() == 2);
      std::vector<double> roots;
      for (auto const& path : paths[0]) roots.push_back(path.invariantP_);
      std::sort(roots.begin(), roots.end());
      CHECK(std::abs(roots[0] - row[3]) < 1.e-10);
      CHECK(std::abs(roots[1] - row[4]) < 1.e-10);
    }
    GeometricalOpticsConfig config;
    config.radio_.grid_ = grid;
    auto device = make_geometrical_optics_accumulator(profile->parameters(),
                                                      {{0., 0., -100.}}, config);
    auto paths = device->paths({{0., 0., -215.}});
    REQUIRE(paths[0].size() == 2);
    for (auto const& path : paths[0]) {
      if (path.topology_ != "surface_reflected") continue;
      double const optical =
          1.78 * 315. - .423 * 77. * (2. - std::exp(-215. / 77.) - std::exp(-100. / 77.));
      CHECK(std::abs(path.opticalPathM_ / optical - 1.) < 1.e-12);
      CHECK(path.geometricLengthM_ == 315.);
    }
    CHECK_THROWS(device->paths({{0., 0., 1.}}));
    // Extending the exponential law above z=0 would create a false turn at
    // z=+0.123 mm. That path belongs to neither the firn nor the air model.
    vector_type surfaceSource{0., 0., -10.};
    vector_type surfaceObserver{71.0677166883896, 0., -1.e-6};
    CHECK_THROWS(guarded.evaluate(surfaceSource, surfaceObserver, grid));
    auto surfaceDevice = make_geometrical_optics_accumulator(profile->parameters(),
                                                             {surfaceObserver}, config);
    CHECK_THROWS(surfaceDevice->paths({surfaceSource}));
  }
} // namespace

namespace {
  void check_analytic_integrals() {
    ExponentialProfile profile(1.78, .423, 77., 0., .001);
    GeometricalOpticsConfig config;
    config.radio_.grid_ = {1.e-9, 64};
    // Independent 70-digit CPU quadrature. Test the complete device path,
    // including its root, without including CUDA implementation in this host test.
    double const monotonic[][7] = {
        {-335, -100, 0, 0, 235, 409.83194628730945, 134.79002030519106},
        {-335, -100, 9.9999999999999998e-13, 1.3479002030519107e-10, 235,
         409.83194628730945, 134.79002030519106},
        {-335, -100, 0.69999999999999996, 103.03217184172475, 256.6017261747055,
         447.47894354703368, 175.51898287043983},
        {-335, -100, 1.6000000000000001, 553.62425875004567, 602.06230369283105,
         1047.928112023671, 2381.3649258457622},
        {-200.00000000099999, -200, 1, 6.9719044308496063e-10, 1.2190378842511467e-09,
         2.1314884304264901e-09, 1.03608411665829e-09},
        {-5000, -4999, 1, 0.67909463693783312, 1.208788453749343, 2.1516434476738304,
         0.99227238870772472},
        {-335, 0, 1.3, 431.9631511594651, 550.23766057036482, 916.38091582454376,
         1052.4947634920156},
        {-335, 0, 1.3569998643000001, 532.74452373857071, 639.89089815380316,
         1051.0041928076762, 407721.90030952467},
    };
    for (auto const& row : monotonic) {
      auto device = make_geometrical_optics_accumulator(profile.parameters(),
                                                        {{row[3], 0., row[1]}}, config);
      auto const paths = device->paths({{0., 0., row[0]}});
      auto found = std::find_if(paths[0].begin(), paths[0].end(), [](auto const& path) {
        return path.topology_ == "direct";
      });
      INFO("monotonic p=" << row[2] << ", source z=" << row[0]);
      REQUIRE(found != paths[0].end());
      CHECK(std::abs(found->invariantP_ - row[2]) < 1.e-10);
      CHECK(std::abs(found->geometricLengthM_ / row[4] - 1.) < 1.e-9);
      CHECK(std::abs(found->opticalPathM_ / row[5] - 1.) < 1.e-9);
    }
    double const reflected[][5] = {
        {0, 0, 435, 718.46628351685683, 0.0021970917018277174},
        {0.29999999999999999, 80.832534626058518, 442.49154685295991, 730.69598417383622,
         0.00215556664071926},
        {1, 336.94515736152442, 551.29751478300511, 907.23180590897391,
         0.0016627518483684949},
        {1.3, 600.60754029463794, 747.47271053288057, 1217.3749338542914,
         0.001023054081181581},
        {1.3569998643000001, 773.61616970230546, 904.97052089663316, 1448.6628083255894,
         4.4189312947674089e-05},
    };
    std::vector<vector_type> observers;
    for (auto const& row : reflected) observers.push_back({row[1], 0., -100.});
    auto device =
        make_geometrical_optics_accumulator(profile.parameters(), observers, config);
    auto const paths = device->paths({{0., 0., -335.}});
    for (std::size_t i = 0; i < observers.size(); ++i) {
      auto const& row = reflected[i];
      auto found = std::find_if(paths[i].begin(), paths[i].end(), [](auto const& path) {
        return path.topology_ == "surface_reflected";
      });
      REQUIRE(found != paths[i].end());
      INFO("reflected p=" << row[0]);
      CHECK(std::abs(found->invariantP_ - row[0]) < 1.e-12);
      CHECK(std::abs(found->geometricLengthM_ / row[2] - 1.) < 1.e-10);
      CHECK(std::abs(found->opticalPathM_ / row[3] - 1.) < 1.e-10);
      CHECK(std::abs(found->geometricSpreadingPerM_ / row[4] - 1.) < 1.e-8);
      CHECK(std::abs(found->attenuationExponent_ / (.001 * row[2]) - 1.) < 1.e-10);
    }
    double const turning[]{903.46329253573026, 990.4582995812832, 1634.4398066861568};
    auto curved = make_geometrical_optics_accumulator(profile.parameters(),
                                                      {{turning[0], 0., -100.}}, config);
    auto const curvedPaths = curved->paths({{0., 0., -335.}});
    auto found =
        std::find_if(curvedPaths[0].begin(), curvedPaths[0].end(), [](auto const& path) {
          return path.topology_.find("turning_") == 0 &&
                 std::abs(path.invariantP_ - 1.5) < 1.e-10;
        });
    REQUIRE(found != curvedPaths[0].end());
    CHECK(std::abs(found->geometricLengthM_ / turning[1] - 1.) < 1.e-10);
    CHECK(std::abs(found->opticalPathM_ / turning[2] - 1.) < 1.e-10);
  }
} // namespace

TEST_CASE("Firn GO: device rays and ordered spectra preserve CPU physics",
          "[kokkos][firn][radio][go]") {
  std::optional<corsika::kokkos::Runtime> runtime;
  auto profile = std::make_shared<ExponentialProfile const>(1.78, .423, 77., 0., .001);
  CoherentStratifiedGOBackend cpu(profile, 1.);
  GeometricalOpticsConfig config;
  config.radio_.grid_ = {1.e-9, 64};
  config.radio_.timeOriginS_ = -1.e-6;
  config.radio_.writeBranchComponents_ = true;
  config.batchSegments_ = 2;
  std::vector<vector_type> observers{
      {400., 0., -100.}, {650., 0., -100.}, {0., 600., -100.}};
  auto device =
      make_geometrical_optics_accumulator(profile->parameters(), observers, config);
  auto empty = device->download();
  CHECK(empty.total_.size() == observers.size());
  runtime.emplace();
  CHECK(runtime->ownsInitialization());
  check_analytic_integrals();
  check_endpoint_continuity();
  check_branch_completeness();
  std::vector<vector_type> sources{{0., 0., -200.}, {.1, -.2, -201.}};
  auto paths = device->paths(sources);
  double maxTime = 0., maxAngle = 0., maxSpreading = 0.;
  std::size_t pathCount = 0;
  for (std::size_t s = 0; s < sources.size(); ++s)
    for (std::size_t o = 0; o < observers.size(); ++o) {
      auto r = cpu.evaluate(sources[s], observers[o], config.radio_.grid_);
      auto const& actual = paths[s * observers.size() + o];
      REQUIRE(actual.size() == r.branches_.size());
      for (std::size_t b = 0; b < actual.size(); ++b) {
        auto const& a = actual[b];
        auto const& e = r.branches_[b].path_;
        ++pathCount;
        CHECK(a.topology_ == e.topology_);
        if (s == 0 && o == 2 && a.topology_ == "surface_reflected") {
          // Independent 70-digit quadrature, including the ray-tube derivative.
          // A finite p stencil biased this grazing-reflection amplitude.
          double const spreading = .0003606446697898538;
          CHECK(std::abs(a.geometricSpreadingPerM_ / spreading - 1.) < 1.e-10);
          CHECK(std::abs(e.geometricSpreadingPerM_ / spreading - 1.) < 1.e-7);
        }
        maxTime = std::max(maxTime, std::abs(a.travelTimeS_ - e.travelTimeS_));
        maxAngle = std::max(maxAngle, norm(subtract(a.emitDirection_, e.emitDirection_)));
        maxAngle =
            std::max(maxAngle, norm(subtract(a.receiveDirection_, e.receiveDirection_)));
        maxSpreading = std::max(
            maxSpreading,
            std::abs(a.geometricSpreadingPerM_ / e.geometricSpreadingPerM_ - 1.));
      }
    }
  std::cout << "GO_RAYS count=" << pathCount << " maximum_time_s=" << maxTime
            << " maximum_direction=" << maxAngle
            << " maximum_spreading_relative=" << maxSpreading << '\n';
  // 0.1 ps bounds phase error at 500 MHz; no fitted shift/polarity correction.
  CHECK(maxTime < 1.e-13);
  CHECK(maxAngle < 1.e-7);
  CHECK(maxSpreading < 1.e-5);
  std::vector<GeometricalOpticsTrack> tracks{
      {{0., 0., -200.}, {.01, .02, -200.06}, 0., .3e-9, -1.},
      {{0., 0., -200.}, {.1, .05, -200.6}, .4e-9, 3.e-9, 2.},
      {{.1, -.2, -201.}, {.13, -.23, -201.03}, 4.e-9, 4.3e-9, -.5}};
  std::size_t childCount = 0;
  auto expected = reference(tracks, observers, cpu, config.radio_, childCount);
  device->reset();
  for (auto const& t : tracks) device->append(t);
  auto first = device->download();
  double error = l2(first.total_, expected.total_);
  std::cout << "GO_SPECTRUM execution=" << device->executionSpaceName()
            << " relative_l2=" << error << " children=" << childCount
            << " geometry_ms=" << device->statistics().geometryMs_ << '\n';
  CHECK(error < 1.e-5);
  for (int b = 0; b < 7; ++b) CHECK(l2(branch(first, b), branch(expected, b)) < 1.e-5);
  CHECK(device->statistics().inputSegments_ == tracks.size());
  CHECK(device->statistics().subdividedSegments_ == childCount);
  CHECK(device->statistics().failedQueries_ == 0);
  // Surface reflection is optional; turning rays and the direct field stay enabled.
  auto noReflectionConfig = config;
  noReflectionConfig.includeSurfaceReflection_ = false;
  auto noReflection = make_geometrical_optics_accumulator(profile->parameters(),
                                                          observers, noReflectionConfig);
  for (auto const& t : tracks) noReflection->append(t);
  auto directAndTurning = noReflection->download();
  auto expectedWithoutReflection = first.total_;
  for (std::size_t o = 0; o < observers.size(); ++o)
    for (int c = 0; c < 3; ++c)
      for (std::size_t f = 0; f < config.radio_.grid_.frequencyCount(); ++f)
        expectedWithoutReflection[o][c][f] -= first.branches_[o][1][c][f];
  CHECK(l2(directAndTurning.total_, expectedWithoutReflection) < 1.e-12);
  CHECK(noReflection->statistics().branchCounts_[1] == 0);
  CHECK(noReflection->statistics().branchCounts_[2] > 0);

  // Independent worker fields must add coherently, including their absolute phases.
  std::vector<spectra_type::components_type> joined;
  for (int worker = 0; worker < 2; ++worker) {
    auto shard =
        make_geometrical_optics_accumulator(profile->parameters(), observers, config);
    for (std::size_t i = worker; i < tracks.size(); i += 2) shard->append(tracks[i]);
    auto partial = shard->download().total_;
    if (worker == 0) {
      joined = std::move(partial);
    } else {
      for (std::size_t o = 0; o < joined.size(); ++o)
        for (int c = 0; c < 3; ++c)
          for (std::size_t f = 0; f < joined[o][c].size(); ++f)
            joined[o][c][f] += partial[o][c][f];
    }
  }
  CHECK(l2(joined, first.total_) < 1.e-12);
  config.batchSegments_ = 1;
  auto single =
      make_geometrical_optics_accumulator(profile->parameters(), observers, config);
  for (auto const& t : tracks) single->append(t);
  auto second = single->download();
  CHECK(l2(second.total_, first.total_) < 1.e-14);
  device->reset();
  for (auto const& t : tracks) device->append(t);
  CHECK(l2(device->download().total_, first.total_) == 0.);
  // Cover different observer depths, reversed depth order, a vertical ray,
  // a same-depth turning path, and the uniform-index limit of this model.
  for (bool uniform : {false, true}) {
    auto optical = std::make_shared<ExponentialProfile const>(1.78, uniform ? 0. : .423,
                                                              77., 0., .001);
    CoherentStratifiedGOBackend baseline(optical, 1.);
    std::vector<vector_type> targets{
        {0., 0., -100.}, {40., 30., -250.}, {30., 0., -200.}};
    auto varied =
        make_geometrical_optics_accumulator(optical->parameters(), targets, config);
    auto actual = varied->paths({{0., 0., -200.}});
    for (std::size_t o = 0; o < targets.size(); ++o) {
      auto expected = baseline.evaluate({0., 0., -200.}, targets[o], config.radio_.grid_);
      REQUIRE(actual[o].size() == expected.branches_.size());
      for (std::size_t b = 0; b < actual[o].size(); ++b) {
        auto const& a = actual[o][b];
        auto const& e = expected.branches_[b].path_;
        CHECK(a.topology_ == e.topology_);
        CHECK(std::abs(a.travelTimeS_ - e.travelTimeS_) < 1.e-13);
        CHECK(norm(subtract(a.emitDirection_, e.emitDirection_)) < 1.e-7);
      }
    }
  }
  // A previously biased near-horizontal branch: independent analytic oracle.
  auto nearHorizontal = make_geometrical_optics_accumulator(profile->parameters(),
                                                            {{568., 0., -100.}}, config);
  auto const nearPaths = nearHorizontal->paths({{0., 0., -215.}});
  bool checkedTurning = false;
  for (auto const& path : nearPaths[0]) {
    if (path.topology_ == "turning_short") {
      checkedTurning = true;
      CHECK(std::abs(path.geometricSpreadingPerM_ / .0021858157292539474 - 1.) < 1.e-7);
      CHECK(path.phaseOffsetRad_ == 0.);
    }
  }
  CHECK(checkedTurning);
  auto nonfinite = tracks[0];
  nonfinite.endTimeS_ = std::numeric_limits<double>::infinity();
  CHECK_THROWS(device->append(nonfinite));
  config.memoryLimitBytes_ = 1;
  CHECK_THROWS(
      make_geometrical_optics_accumulator(profile->parameters(), observers, config));
  config.memoryLimitBytes_ = 512 * 1024 * 1024;
  config.foldGuardTimeS_ = 1.;
  auto guarded = make_geometrical_optics_accumulator(profile->parameters(),
                                                     {{650., 0., -100.}}, config);
  CHECK_THROWS(guarded->paths({{0., 0., -200.}}));
  config.foldGuardTimeS_ = 2.e-9;
  auto shadow = make_geometrical_optics_accumulator(profile->parameters(),
                                                    {{100000., 0., -100.}}, config);
  CHECK_THROWS(shadow->paths({{0., 0., -200.}}));
  CHECK_THROWS(device->paths({{0., 0., 1.}}));
  config.radio_.maximumSubdivisions_ = 1;
  auto limit =
      make_geometrical_optics_accumulator(profile->parameters(), observers, config);
  CHECK_THROWS(limit->append(tracks[1]));
  CHECK_THROWS(limit->download());
  limit->reset();
  CHECK(limit->statistics().inputSegments_ == 0);

  // A finite event budget must fail explicitly and preserve its first error
  // during output finalization. Reset starts a clean, reproducible event.
  GeometricalOpticsConfig budgetConfig;
  budgetConfig.radio_.grid_ = {1.e-9, 64};
  budgetConfig.radio_.maximumResponseQueries_ = 1;
  budgetConfig.batchSegments_ = 1;
  auto budget = make_geometrical_optics_accumulator(profile->parameters(),
                                                    {{400., 0., -100.}}, budgetConfig);
  GeometricalOpticsTrack const shortTrack{
      {0., 0., -200.}, {0., 0., -200.001}, 0., 1.e-11, -1.};
  budget->append(shortTrack);
  auto const firstEvent = budget->download();
  CHECK(budget->statistics().responseQueries_ == 1);
  CHECK_THROWS_WITH(budget->append(shortTrack), "radio response-query budget exceeded");
  CHECK_THROWS_WITH(budget->download(), "radio response-query budget exceeded");
  CHECK_THROWS_WITH(budget->flush(), "radio response-query budget exceeded");
  CHECK_THROWS_WITH(budget->append(shortTrack), "radio response-query budget exceeded");
  CHECK(budget->statistics().inputSegments_ == 1);
  budget->reset();
  CHECK(budget->statistics().responseQueries_ == 0);
  budget->append(shortTrack);
  CHECK(budget->download().total_ == firstEvent.total_);
}
