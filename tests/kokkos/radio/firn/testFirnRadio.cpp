/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/radio/firn/GreenResponse.hpp>
#include <corsika/kokkos/radio/firn/Fft.hpp>
#include <catch2/catch_test_macros.hpp>
#include <corsika/kokkos/radio/firn/OnlineRadioConfig.hpp>
#include <cnpy.hpp>
#include <filesystem>
#include <fstream>

using namespace corsika::kokkos::radio::firn;
namespace {
  void require(bool condition, char const* message) {
    INFO(message);
    REQUIRE(condition);
  }
  double relative(double value, double reference) {
    return std::abs(value - reference) / std::max(std::abs(reference), 1.e-300);
  }
} // namespace

TEST_CASE("Firn radio: Sha256", "[kokkos][firn][radio]") {

  auto const path =
      std::filesystem::temp_directory_path() / "c8_variable_index_radio_sha256_abc.txt";
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << "abc";
  }
  auto const digest = sha256_file(path);
  std::filesystem::remove(path);
  require(digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 known vector");
}

TEST_CASE("Firn radio: Profiles", "[kokkos][firn][radio]") {

  UniformProfile uniform(1.78, 0.01);
  require(uniform.sample({0.0, 0.0, -10.0}, 100.0e6).refractiveIndex_ == 1.78,
          "uniform profile index");
  LinearProfile linear(-100.0, 0.0, 1.7, 1.3);
  require(relative(linear.refractiveIndexAtZ(-50.0), 1.5) < 1.0e-15,
          "linear profile interpolation");
  ExponentialProfile exponential(1.78, 0.423, 77.0, 0.0);
  require(relative(exponential.refractiveIndexAtZ(0.0), 1.357) < 1.0e-14,
          "exponential surface index");
  TabulatedProfile table({-100.0, -50.0, 0.0}, {1.7, 1.5, 1.3});
  require(relative(table.refractiveIndexAtZ(-75.0), 1.6) < 1.0e-15,
          "tabulated profile interpolation");
}

TEST_CASE("Firn radio: UniformGreen", "[kokkos][firn][radio]") {

  FrequencyGrid grid{1.0e-9, 4096};
  UniformAnalyticBackend backend(UniformProfile(1.5, 1.0 / 100.0));
  vector_type const source{0.0, 0.0, 0.0};
  vector_type const observer{100.0, 0.0, 0.0};
  auto const result = backend.evaluate(source, observer, grid);
  require(result.path_.valid_, "uniform path valid");
  require(relative(result.path_.travelTimeS_, 150.0 / c_m_per_s) < 1.0e-15,
          "uniform travel time");
  std::size_t const index = 400;
  double const expectedMagnitude = std::exp(-1.0) / 100.0;
  require(
      relative(std::abs(result.dyadicPerM_[index][1][1]), expectedMagnitude) < 1.0e-14,
      "uniform amplitude");
  require(std::abs(result.dyadicPerM_[index][0][0]) < 1.0e-16, "transverse projector");

  auto uniformPointer = std::make_shared<UniformProfile const>(1.5, 1.0 / 100.0);
  StraightAverageIndexBackend straight(uniformPointer);
  auto const straightResult = straight.evaluate(source, observer, grid);
  double squaredDifference = 0.0;
  double squaredReference = 0.0;
  for (std::size_t frequency = 0; frequency < grid.frequencyCount(); ++frequency) {
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t col = 0; col < 3; ++col) {
        squaredDifference += std::norm(straightResult.dyadicPerM_[frequency][row][col] -
                                       result.dyadicPerM_[frequency][row][col]);
        squaredReference += std::norm(result.dyadicPerM_[frequency][row][col]);
      }
    }
  }
  require(std::sqrt(squaredDifference / squaredReference) < 1.0e-13,
          "straight average-index backend has exact uniform limit");
}

TEST_CASE("Firn radio: StratifiedRay", "[kokkos][firn][radio]") {

  ExponentialProfile profile(1.78, 0.423, 77.0, 0.0);
  StratifiedRaySolver solver(profile);
  vector_type const source{0.0, 0.0, -200.0};
  vector_type const observer{500.0, 0.0, -100.0};
  auto const ray = solver.direct(source, observer, 100.0e6);
  require(ray.valid_, "exponential direct ray must exist");
  require(ray.fermatRelativeResidual_ < 1.0e-10, "Fermat horizontal residual");
  require(ray.snellResidual_ < 1.0e-12, "Snell invariant residual");
  require(ray.opticalPathM_ > ray.geometricLengthM_,
          "optical path must exceed geometric path");

  LinearProfile linear(-300.0, 0.0, 1.78, 1.35);
  auto pointer = std::make_shared<LinearProfile const>(linear);
  FrequencyGrid grid{2.0e-9, 256};
  StratifiedGOBackend backend(pointer);
  vector_type const linearObserver{300.0, 0.0, -100.0};
  auto const response = backend.evaluate(source, linearObserver, grid);
  require(response.path_.valid_ && response.dyadicPerM_.size() == 129,
          "stratified GO response");

  auto uniformPointer = std::make_shared<LinearProfile const>(-300.0, 0.0, 1.5, 1.5);
  StratifiedGOBackend uniformGo(uniformPointer);
  auto const uniformResponse = uniformGo.evaluate(source, linearObserver, grid);
  double const distance = norm(subtract(linearObserver, source));
  require(
      relative(uniformResponse.path_.geometricSpreadingPerM_, 1.0 / distance) < 1.0e-10,
      "stratified GO uniform limit is 1/R");

  vector_type const turningObserver{650.0, 0.0, -100.0};
  auto const falseDirectControl = solver.direct(source, turningObserver);
  require(!falseDirectControl.valid_ &&
              falseDirectControl.topology_ == "no_monotonic_direct_root",
          "turning-region endpoint must not acquire a spurious direct root");
  auto const turning = solver.turningBranches(source, turningObserver);
  require(turning.size() == 2, "firn control must have two turning branches");
  for (auto const& branch : turning) {
    require(branch.fermatRelativeResidual_ < 1.0e-10, "turning Fermat residual");
    require(branch.snellResidual_ < 1.0e-12, "turning Snell invariant residual");
    require(branch.geometricSpreadingPerM_ > 0.0, "turning ray-tube spreading");
  }
  vector_type const turningObserver3d{600.0, 250.0, -100.0};
  auto const turning3d = solver.turningBranches(source, turningObserver3d);
  require(turning3d.size() == turning.size(),
          "3D azimuth must preserve stratified branch count");
  require(relative(turning3d[0].travelTimeS_, turning[0].travelTimeS_) < 1.0e-10,
          "3D azimuthal rotation preserves travel time");

  auto const reflected = solver.surfaceReflected(source, {400.0, 0.0, -100.0});
  require(reflected.valid_, "firn surface-reflected path");
  require(reflected.fermatRelativeResidual_ < 1.0e-10,
          "surface-reflected Fermat residual");
  StratifiedSurfaceReflectionBackend reflectedBackend(
      std::make_shared<ExponentialProfile const>(profile), 1.0);
  auto const reflectedResponse =
      reflectedBackend.evaluate(source, {400.0, 0.0, -100.0}, grid);
  require(reflectedResponse.path_.valid_, "surface-reflected Fresnel Green response");

  CoherentStratifiedGOBackend coherent(
      std::make_shared<ExponentialProfile const>(profile), 1.0, true, true, 0.0);
  auto const coherentResponse = coherent.evaluate(source, turningObserver, grid);
  require(coherentResponse.path_.valid_ && coherentResponse.branches_.size() >= 2,
          "coherent GO enumerates multiple physical branches");
  bool hasShort = false;
  bool hasLong = false;
  double aggregateError = 0.0;
  double aggregateReference = 0.0;
  for (auto const& branch : coherentResponse.branches_) {
    hasShort = hasShort || branch.path_.topology_ == "turning_short";
    hasLong = hasLong || branch.path_.topology_ == "turning_long";
  }
  bool hasDirect = false;
  for (auto const& branch : coherentResponse.branches_) {
    hasDirect = hasDirect || branch.path_.topology_ == "direct";
  }
  require(!hasDirect,
          "coherent turning control must not double count a false direct ray");
  for (std::size_t frequency = 0; frequency < grid.frequencyCount(); ++frequency) {
    complex_matrix_type sum = zero_matrix();
    for (auto const& branch : coherentResponse.branches_) {
      for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
          sum[row][col] += branch.dyadicPerM_[frequency][row][col];
        }
      }
    }
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t col = 0; col < 3; ++col) {
        aggregateError +=
            std::norm(sum[row][col] - coherentResponse.dyadicPerM_[frequency][row][col]);
        aggregateReference += std::norm(sum[row][col]);
      }
    }
  }
  require(hasShort && hasLong, "coherent GO preserves short and long turning inventory");
  require(std::sqrt(aggregateError / std::max(aggregateReference, 1.0e-300)) < 1.0e-14,
          "coherent aggregate equals explicit complex branch sum");

  auto const oneTurning = coherent.evaluate(source, {575.0, 0.0, -100.0}, grid);
  std::size_t oneTurningCount = 0;
  for (auto const& branch : oneTurning.branches_) {
    if (branch.path_.topology_ == "turning_short" ||
        branch.path_.topology_ == "turning_long") {
      ++oneTurningCount;
    }
  }
  require(oneTurningCount == 1,
          "endpoint topology may contain one ordinary turning branch");
}

TEST_CASE("Firn radio: physical turning and evanescent phases", "[kokkos][firn][radio]") {
  auto profile = std::make_shared<ExponentialProfile const>(1.78, .423, 77., 0.);
  StratifiedRaySolver solver(*profile);
  auto const rays = solver.turningBranches({0., 0., -215.}, {568., 0., -100.});
  REQUIRE(rays.size() == 1);
  // Independent 70-digit analytic ray distance and derivative, not old CPU output.
  CHECK(relative(rays[0].invariantP_, 1.6645676108055380632) < 1.e-12);
  CHECK(relative(rays[0].geometricSpreadingPerM_, .0021858157292539474) < 1.e-7);
  CHECK(rays[0].phaseOffsetRad_ == 0.);
  auto const pair = solver.turningBranches({0., 0., -200.}, {650., 0., -100.});
  REQUIRE(pair.size() == 2);
  CHECK(pair[0].phaseOffsetRad_ == .5 * pi);
  CHECK(pair[1].phaseOffsetRad_ == 0.);
  CoherentStratifiedGOBackend coherent(profile, 1.);
  auto const response =
      coherent.evaluate({0., 0., -215.}, {568., 0., -100.}, {1.e-9, 64});
  for (auto const& b : response.branches_) {
    if (b.path_.topology_ == "turning_short") CHECK(b.path_.phaseOffsetRad_ == 0.);
  }

  // Maxwell interface check: n_ice=1.5, tan(theta_i)=2, n_air=1.
  // Outward decay for exp(+i omega t) gives rs=-.28+.96i, rp=-.8+.6i.
  auto uniform = std::make_shared<ExponentialProfile const>(1.5, 0., 77., 0.);
  StratifiedSurfaceReflectionBackend reflection(uniform, 1.);
  auto const reflected =
      reflection.evaluate({0., 0., -100.}, {400., 0., -100.}, {1.e-9, 64});
  auto const& path = reflected.path_;
  vector_type const s{0., 1., 0.};
  auto const ps = cross(s, path.emitDirection_);
  auto const po = cross(s, path.receiveDirection_);
  auto const& matrix = reflected.dyadicPerM_[0];
  auto const projected = contract(matrix, ps);
  auto const rp = (po[0] * projected[0] + po[1] * projected[1] + po[2] * projected[2]) /
                  path.geometricSpreadingPerM_;
  auto const rs = matrix[1][1] / path.geometricSpreadingPerM_;
  CHECK(std::abs(rs - std::complex<double>(-.28, .96)) < 1.e-10);
  CHECK(std::abs(rp - std::complex<double>(-.8, .6)) < 1.e-10);
  CoherentStratifiedGOBackend both(uniform, 1.);
  auto const all = both.evaluate({0., 0., -100.}, {400., 0., -100.}, {1.e-9, 64});
  for (auto const& branch : all.branches_) {
    if (branch.path_.topology_ == "surface_reflected")
      CHECK(std::abs(branch.dyadicPerM_[0][1][1] / branch.path_.geometricSpreadingPerM_ -
                     rs) < 1.e-12);
  }

  // The generic CPU profile path also keeps its stencil inside the branch.
  LinearProfile linear(-500., 0., 1.8, 1.3);
  double const p = 1.39999, ns = 1.5, no = 1.4, gradient = .001;
  double const x = p / gradient * (std::acosh(ns / p) + std::acosh(no / p));
  double const derivative =
      (std::acosh(ns / p) + std::acosh(no / p) - ns / std::sqrt(ns * ns - p * p) -
       no / std::sqrt(no * no - p * p)) /
      gradient;
  double const jacobian = x * std::sqrt(1. - p * p / (no * no)) *
                          std::abs(derivative * std::sqrt(ns * ns - p * p)) / (p / ns);
  auto const linearRays =
      StratifiedRaySolver(linear).turningBranches({0., 0., -200.}, {x, 0., -100.});
  REQUIRE(linearRays.size() == 1);
  CHECK(relative(linearRays[0].geometricSpreadingPerM_, std::sqrt(ns / (no * jacobian))) <
        1.e-4);
  CHECK(linearRays[0].phaseOffsetRad_ == 0.);
}

TEST_CASE("Firn radio: Fft", "[kokkos][firn][radio]") {

  FrequencyGrid grid{1.0e-9, 4096};
  std::vector<std::complex<double>> full(grid.sampleCount_);
  std::vector<double> original(grid.sampleCount_);
  for (std::size_t index = 0; index < grid.sampleCount_; ++index) {
    double const t = static_cast<double>(index) * grid.sampleIntervalS_;
    original[index] =
        2.0 * std::sin(2.0 * corsika::kokkos::radio::firn::pi * 25.0e6 * t) +
        0.5 * std::cos(2.0 * corsika::kokkos::radio::firn::pi * 75.0e6 * t);
    full[index] = original[index];
  }
  fft_in_place(full, false);
  std::vector<std::complex<double>> positive(grid.frequencyCount());
  for (std::size_t index = 0; index < positive.size(); ++index) {
    positive[index] = full[index] * grid.sampleIntervalS_;
  }
  double maximumImaginary = 0.0;
  auto const recovered = inverse_real_spectrum(positive, grid, &maximumImaginary);
  double maximumError = 0.0;
  for (std::size_t index = 0; index < original.size(); ++index) {
    maximumError = std::max(maximumError, std::abs(original[index] - recovered[index]));
  }
  require(maximumError < 2.0e-12, "FFT round trip");
  require(maximumImaginary < 2.0e-12, "Hermitian real waveform");
  require(parseval_relative(recovered, positive, grid) < 2.0e-13, "Parseval identity");
}

TEST_CASE("Firn radio: FullWaveTable", "[kokkos][firn][radio]") {

  auto const root =
      std::filesystem::temp_directory_path() / "c8_variable_index_radio_oracle";
  std::filesystem::create_directories(root);
  auto const npz = root / "response.npz";
  auto const manifest = root / "response.yaml";
  std::filesystem::remove(npz);
  std::vector<double> const frequency{0.0, 5.0e8};
  std::vector<double> const sourceZ{-210.0, -190.0};
  std::vector<double> const radius{500.0, 600.0};
  std::vector<double> const sourceIndex{1.5, 1.5};
  std::vector<double> path(sourceZ.size() * radius.size());
  std::vector<double> spreading(sourceZ.size() * radius.size());
  std::vector<double> real(frequency.size() * sourceZ.size() * radius.size() * 9, 0.0);
  std::vector<double> imag(real.size(), 0.0);
  auto flat = [&](std::size_t fi, std::size_t zi, std::size_t ri, std::size_t row,
                  std::size_t col) {
    return ((((fi * sourceZ.size() + zi) * radius.size() + ri) * 3 + row) * 3 + col);
  };
  for (std::size_t zi = 0; zi < sourceZ.size(); ++zi) {
    for (std::size_t ri = 0; ri < radius.size(); ++ri) {
      path[zi * radius.size() + ri] = 1.5 * radius[ri];
      spreading[zi * radius.size() + ri] = 1.0 / radius[ri];
      for (std::size_t fi = 0; fi < frequency.size(); ++fi) {
        double const phase = -2.0 * corsika::kokkos::radio::firn::pi * frequency[fi] *
                             path[zi * radius.size() + ri] / c_m_per_s;
        std::complex<double> const value = std::polar(1.0 / radius[ri], phase);
        for (std::size_t component = 0; component < 3; ++component) {
          real[flat(fi, zi, ri, component, component)] = value.real();
          imag[flat(fi, zi, ri, component, component)] = value.imag();
        }
      }
    }
  }
  cnpy::npz_save(npz.string(), "frequency_hz", frequency.data(), {frequency.size()}, "w");
  cnpy::npz_save(npz.string(), "source_z_m", sourceZ.data(), {sourceZ.size()}, "a");
  cnpy::npz_save(npz.string(), "source_refractive_index", sourceIndex.data(),
                 {sourceIndex.size()}, "a");
  cnpy::npz_save(npz.string(), "horizontal_radius_m", radius.data(), {radius.size()},
                 "a");
  cnpy::npz_save(npz.string(), "reduction_optical_path_m", path.data(),
                 {sourceZ.size(), radius.size()}, "a");
  cnpy::npz_save(npz.string(), "geometric_spreading_per_m", spreading.data(),
                 {sourceZ.size(), radius.size()}, "a");
  std::vector<std::size_t> const shape{frequency.size(), sourceZ.size(), radius.size(), 3,
                                       3};
  cnpy::npz_save(npz.string(), "green_cylindrical_real_per_m", real.data(), shape, "a");
  cnpy::npz_save(npz.string(), "green_cylindrical_imag_per_m", imag.data(), shape, "a");
  UniformProfile const profile(1.5);
  std::ofstream output(manifest);
  output << "schema_version: 1\n"
         << "profile_fingerprint: '" << profile.profileFingerprint() << "'\n"
         << "dyadic_basis: cylindrical_r_phi_z\n"
         << "observer_z_m: -100\n"
         << "observer_refractive_index: 1.5\n"
         << "npz: response.npz\n"
         << "npz_sha256: '" << sha256_file(npz) << "'\n";
  output.close();
  TabulatedFullWaveBackend backend(manifest, profile.profileFingerprint());
  FrequencyGrid grid{1.0e-9, 16};
  auto const response = backend.evaluate({0.0, 0.0, -200.0}, {550.0, 0.0, -100.0}, grid);
  require(response.path_.valid_ && response.dyadicPerM_.size() == 9,
          "full-wave table query");
  double squaredError = 0.0;
  double squaredReference = 0.0;
  double const reducedAmplitude = 0.5 * (1.0 / 500.0 + 1.0 / 600.0);
  for (std::size_t index = 1; index < response.dyadicPerM_.size(); ++index) {
    double const frequencyHz = grid.frequencyHz(index);
    std::complex<double> const expected =
        std::polar(reducedAmplitude, -2.0 * corsika::kokkos::radio::firn::pi *
                                         frequencyHz * 825.0 / c_m_per_s);
    squaredError += std::norm(response.dyadicPerM_[index][0][0] - expected);
    squaredReference += std::norm(expected);
  }
  require(std::sqrt(squaredError / squaredReference) < 1.0e-12,
          "C++ response-table interpolation agrees with schema-v1 oracle");
  bool rejected = false;
  try {
    TabulatedFullWaveBackend wrong(manifest, "wrong-profile");
  } catch (std::runtime_error const&) { rejected = true; }
  require(rejected, "profile fingerprint mismatch must fail");
}

TEST_CASE("Firn radio: CoherentFullWaveTableV2", "[kokkos][firn][radio]") {

  auto const root =
      std::filesystem::temp_directory_path() / "c8_variable_index_radio_oracle_v2";
  std::filesystem::create_directories(root);
  auto const npz = root / "response_v2.npz";
  auto const manifest = root / "response_v2.yaml";
  std::filesystem::remove(npz);
  std::vector<double> const sourceZ{-210.0, -190.0};
  std::vector<double> const radius{500.0, 600.0};
  std::vector<double> const sourceIndex{1.49, 1.51};
  std::vector<double> directPath{750.0, 850.0, 770.0, 870.0};
  std::vector<double> boundaryPath{900.0, 1000.0, 920.0, 1020.0};
  cnpy::npz_save(npz.string(), "source_z_m", sourceZ.data(), {sourceZ.size()}, "w");
  cnpy::npz_save(npz.string(), "horizontal_radius_m", radius.data(), {radius.size()},
                 "a");
  cnpy::npz_save(npz.string(), "source_refractive_index", sourceIndex.data(),
                 {sourceIndex.size()}, "a");
  cnpy::npz_save(npz.string(), "direct_reduction_optical_path_m", directPath.data(),
                 {sourceZ.size(), radius.size()}, "a");
  cnpy::npz_save(npz.string(), "boundary_reduction_optical_path_m", boundaryPath.data(),
                 {sourceZ.size(), radius.size()}, "a");

  std::array<std::array<double, 2>, 4> const frequency{
      {{0.0, 25.0e6}, {25.0e6, 62.5e6}, {62.5e6, 100.0e6}, {100.0e6, 500.0e6}}};
  std::array<double, 3> const directDiagonal{2.0e5, 3.0e5, 4.0e5};
  std::array<double, 3> const boundaryDiagonal{-0.5e5, 0.75e5, 1.0e5};
  std::vector<std::size_t> const shape{2, sourceZ.size(), radius.size(), 3, 3};
  for (std::size_t band = 0; band < frequency.size(); ++band) {
    std::string const prefix = "band" + std::to_string(band) + "_";
    cnpy::npz_save(npz.string(), prefix + "frequency_hz", frequency[band].data(), {2},
                   "a");
    for (auto const& [branch, diagonal] :
         std::array<std::pair<char const*, std::array<double, 3>>, 2>{
             {{"direct", directDiagonal}, {"boundary", boundaryDiagonal}}}) {
      std::vector<double> real(2 * sourceZ.size() * radius.size() * 9, 0.0);
      std::vector<double> imag(real.size(), 0.0);
      std::vector<double> second(real.size(), 0.0);
      auto flat = [&](std::size_t fi, std::size_t zi, std::size_t ri, std::size_t row,
                      std::size_t col) {
        return ((((fi * sourceZ.size() + zi) * radius.size() + ri) * 3 + row) * 3 + col);
      };
      for (std::size_t fi = 0; fi < 2; ++fi) {
        for (std::size_t zi = 0; zi < sourceZ.size(); ++zi) {
          for (std::size_t ri = 0; ri < radius.size(); ++ri) {
            double const spatialScale = 1.0 + 0.1 * zi + 0.2 * ri;
            for (std::size_t component = 0; component < 3; ++component) {
              real[flat(fi, zi, ri, component, component)] =
                  diagonal[component] * spatialScale;
            }
          }
        }
      }
      std::string const base = prefix + branch;
      cnpy::npz_save(npz.string(), base + "_reduced_fg_real_per_m_hz", real.data(), shape,
                     "a");
      cnpy::npz_save(npz.string(), base + "_reduced_fg_imag_per_m_hz", imag.data(), shape,
                     "a");
      cnpy::npz_save(npz.string(), base + "_second_real_per_m_per_hz", second.data(),
                     shape, "a");
      cnpy::npz_save(npz.string(), base + "_second_imag_per_m_per_hz", second.data(),
                     shape, "a");
    }
  }

  UniformProfile const profile(1.5);
  std::ofstream output(manifest);
  output << "schema_version: 2\n"
         << "profile_fingerprint: '" << profile.profileFingerprint() << "'\n"
         << "dyadic_basis: cylindrical_r_phi_z\n"
         << "frequency_interpolation: "
            "banded_not_a_knot_cubic_phase_reduced_fG\n"
         << "observer_z_m: -100\n"
         << "observer_refractive_index: 1.5\n"
         << "npz: response_v2.npz\n"
         << "npz_sha256: '" << sha256_file(npz) << "'\n";
  output.close();

  TabulatedCoherentFullWaveBackend backend(manifest, profile.profileFingerprint());
  FrequencyGrid grid{1.0e-9, 16};
  vector_type const source{0.0, 0.0, -200.0};
  vector_type const observer{550.0, 0.0, -100.0};
  auto const response = backend.evaluate(source, observer, grid);
  require(response.branches_.size() == 2 &&
              response.finiteTrackCoupling_ == FiniteTrackCoupling::PointCurrentDyadic,
          "schema-v2 response preserves two point-current branches");
  require(response.branches_[0].path_.topology_ == "fullwave_direct_matched" &&
              response.branches_[1].path_.topology_ == "fullwave_air_boundary",
          "schema-v2 response branch names");
  double const spatialScale = 1.0 + 0.1 * 0.5 + 0.2 * 0.5;
  double squaredError = 0.0;
  double squaredReference = 0.0;
  for (std::size_t fi = 1; fi < grid.frequencyCount(); ++fi) {
    double const f = grid.frequencyHz(fi);
    for (std::size_t component = 0; component < 3; ++component) {
      std::complex<double> const expectedDirect =
          directDiagonal[component] * spatialScale / f *
          std::exp(std::complex<double>(
              0.0, -2.0 * corsika::kokkos::radio::firn::pi * f * 810.0 / c_m_per_s));
      std::complex<double> const expectedBoundary =
          boundaryDiagonal[component] * spatialScale / f *
          std::exp(std::complex<double>(
              0.0, -2.0 * corsika::kokkos::radio::firn::pi * f * 960.0 / c_m_per_s));
      squaredError += std::norm(
          response.branches_[0].dyadicPerM_[fi][component][component] - expectedDirect);
      squaredError += std::norm(
          response.branches_[1].dyadicPerM_[fi][component][component] - expectedBoundary);
      squaredError += std::norm(response.dyadicPerM_[fi][component][component] -
                                expectedDirect - expectedBoundary);
      squaredReference += std::norm(expectedDirect) + std::norm(expectedBoundary) +
                          std::norm(expectedDirect + expectedBoundary);
    }
  }
  require(std::sqrt(squaredError / squaredReference) < 2.0e-13,
          "schema-v2 band/spatial/carrier interpolation oracle");
  require(!backend.covers(source, {700.0, 0.0, -100.0}, grid),
          "schema-v2 table forbids spatial extrapolation");
  std::filesystem::remove_all(root);
}

TEST_CASE("Firn GO: profile contracts and vertical reflection", "[kokkos][firn][radio]") {
  CHECK_THROWS(CoherentStratifiedGOBackend(nullptr, 1.));
  CHECK_THROWS(StratifiedGOBackend(nullptr));
  CHECK_THROWS(StratifiedSurfaceReflectionBackend(nullptr, 1.));
  CHECK_THROWS(StratifiedTurningGOBackend(nullptr, 0));
  double const infinity = std::numeric_limits<double>::infinity();
  CHECK_THROWS(ExponentialProfile(infinity, .423, 77., 0.));
  CHECK_THROWS(ExponentialProfile(1.78, .423, infinity, 0.));
  OnlineRadioConfig invalidConfig;
  invalidConfig.maximumPhaseCycles_ = infinity;
  CHECK_THROWS(invalidConfig.validate());
  auto exponential = std::make_shared<ExponentialProfile const>(1.78, .423, 77., 0.);
  CoherentStratifiedGOBackend cpu(exponential, 1.);
  CHECK_THROWS(CoherentStratifiedGOBackend(exponential, infinity));
  CHECK_THROWS(CoherentStratifiedGOBackend(exponential, 1., true, true, infinity));
  FrequencyGrid grid{1.e-9, 64};
  CHECK_THROWS(cpu.evaluate({0., 0., 1.}, {10., 0., -100.}, grid));
  StratifiedRaySolver solver(*exponential);
  auto ray = solver.surfaceReflected({0., 0., -215.}, {0., 0., -100.});
  REQUIRE(ray.valid_);
  CHECK(ray.geometricLengthM_ == 315.);
  double const optical =
      1.78 * 315. - .423 * 77. * (2. - std::exp(-215. / 77.) - std::exp(-100. / 77.));
  CHECK(std::abs(ray.opticalPathM_ / optical - 1.) < 1.e-12);
  CHECK(ray.emitDirection_[2] == 1.);
  CHECK(ray.receiveDirection_[2] == -1.);
  // Image-source solution at normal incidence: transverse E has r=(n-1)/(n+1).
  auto uniform = std::make_shared<ExponentialProfile const>(1.5, 0., 77., 0.);
  StratifiedSurfaceReflectionBackend mirror(uniform, 1.);
  auto field = mirror.evaluate({0., 0., -215.}, {0., 0., -100.}, grid);
  CHECK(std::abs(field.path_.geometricSpreadingPerM_ * 315. - 1.) < 1.e-12);
  CHECK(std::abs(field.dyadicPerM_[0][0][0] - .2 / 315.) < 1.e-14);
  CHECK(std::abs(field.dyadicPerM_[0][1][1] - .2 / 315.) < 1.e-14);
  CHECK(std::abs(field.dyadicPerM_[0][2][2]) < 1.e-14);
}
