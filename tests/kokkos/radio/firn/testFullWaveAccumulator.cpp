/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/radio/firn/IFullWaveAccumulator.hpp>
#include <corsika/kokkos/radio/firn/GreenResponse.hpp>
#include <corsika/kokkos/radio/firn/Fft.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cnpy.hpp>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <optional>

using namespace corsika::kokkos::radio::firn;
namespace {
  double relative_l2(std::vector<FullWaveSpectra::components_type> const& candidate,
                     std::vector<FullWaveSpectra::components_type> const& reference) {
    double difference = 0.0;
    double norm = 0.0;
    for (std::size_t observer = 0; observer < reference.size(); ++observer) {
      for (std::size_t component = 0; component < 3; ++component) {
        for (std::size_t frequency = 0; frequency < reference[observer][component].size();
             ++frequency) {
          difference += std::norm(candidate[observer][component][frequency] -
                                  reference[observer][component][frequency]);
          norm += std::norm(reference[observer][component][frequency]);
        }
      }
    }
    return std::sqrt(difference / std::max(norm, 1.0e-300));
  }

  void accumulate_cpu(TabulatedCoherentFullWaveBackend const& backend,
                      std::vector<vector_type> const& observers,
                      std::vector<FullWaveChildSegment> const& children,
                      FrequencyGrid const& grid, double timeOriginS,
                      FullWaveSpectra& output) {
    output.total_.resize(observers.size());
    output.direct_.resize(observers.size());
    output.boundary_.resize(observers.size());
    for (auto* family : {&output.total_, &output.direct_, &output.boundary_}) {
      for (auto& observer : *family)
        for (auto& component : observer)
          component.assign(grid.frequencyCount(), {0.0, 0.0});
    }
    for (auto const& child : children) {
      for (std::size_t observerIndex = 0; observerIndex < observers.size();
           ++observerIndex) {
        auto const response =
            backend.evaluate(child.midpointM_, observers[observerIndex], grid);
        if (response.branches_.size() != 2) {
          throw std::runtime_error("CPU schema-2 response lost branch inventory");
        }
        for (std::size_t branch = 0; branch < 2; ++branch) {
          auto& branchOutput = branch == 0 ? output.direct_ : output.boundary_;
          for (std::size_t frequencyIndex = 1; frequencyIndex < grid.frequencyCount();
               ++frequencyIndex) {
            double const frequency = grid.frequencyHz(frequencyIndex);
            complex_vector_type const projected =
                contract(response.branches_[branch].dyadicPerM_[frequencyIndex],
                         child.displacementM_);
            std::complex<double> const common =
                mu0_h_per_m * elementary_charge_c * child.chargeWeight_ / (4.0 * pi) *
                std::complex<double>(0.0, -2.0 * pi * frequency) *
                std::exp(std::complex<double>(
                    0.0, -2.0 * pi * frequency * (child.midpointTimeS_ - timeOriginS)));
            for (std::size_t component = 0; component < 3; ++component) {
              std::complex<double> const value = common * projected[component];
              branchOutput[observerIndex][component][frequencyIndex] += value;
              output.total_[observerIndex][component][frequencyIndex] += value;
            }
          }
        }
      }
    }
  }

  std::filesystem::path make_fixture() {
    auto root = std::filesystem::temp_directory_path() / "c8_firn_device_radio_oracle";
    std::filesystem::create_directories(root);
    auto npz = root / "response.npz";
    std::vector<double> z{-214., -213.5, -213.}, r{560., 565., 570.},
        index{1.7, 1.71, 1.72};
    std::vector<double> directPath, boundaryPath;
    for (auto zi : z)
      for (auto ri : r) {
        directPath.push_back(800. + ri - zi);
        boundaryPath.push_back(950. + ri + zi);
      }
    cnpy::npz_save(npz.string(), "source_z_m", z.data(), {z.size()}, "w");
    cnpy::npz_save(npz.string(), "horizontal_radius_m", r.data(), {r.size()}, "a");
    cnpy::npz_save(npz.string(), "source_refractive_index", index.data(), {index.size()},
                   "a");
    cnpy::npz_save(npz.string(), "direct_reduction_optical_path_m", directPath.data(),
                   {z.size(), r.size()}, "a");
    cnpy::npz_save(npz.string(), "boundary_reduction_optical_path_m", boundaryPath.data(),
                   {z.size(), r.size()}, "a");
    std::array<double, 5> edges{0., 25.e6, 62.5e6, 100.e6, 500.e6};
    for (std::size_t band = 0; band < 4; ++band) {
      std::string prefix = "band" + std::to_string(band) + "_";
      std::vector<double> frequency{edges[band], .5 * (edges[band] + edges[band + 1]),
                                    edges[band + 1]};
      cnpy::npz_save(npz.string(), prefix + "frequency_hz", frequency.data(), {3}, "a");
      for (std::string const branch : {"direct", "boundary"}) {
        std::vector<double> real, imag, secondReal, secondImag;
        for (auto f : frequency)
          for (std::size_t zi = 0; zi < z.size(); ++zi)
            for (std::size_t ri = 0; ri < r.size(); ++ri)
              for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) {
                  double const scale = (branch == "direct" ? 1. : -.3) *
                                       (1. + .1 * zi + .2 * ri) *
                                       (row == col ? 2. : .13 * (row + col + 1)) * 1.e5;
                  double const u = f / 1.e9;
                  real.push_back(scale * (1. + .2 * u + .3 * u * u + .1 * u * u * u));
                  imag.push_back(scale * (.1 - .2 * u + .4 * u * u));
                  secondReal.push_back(scale * (.6 + .6 * u) / 1.e18);
                  secondImag.push_back(scale * .8 / 1.e18);
                }
        std::vector<std::size_t> shape{3, z.size(), r.size(), 3, 3};
        auto save = [&](std::string suffix, std::vector<double> const& data) {
          cnpy::npz_save(npz.string(), prefix + branch + suffix, data.data(), shape, "a");
        };
        save("_reduced_fg_real_per_m_hz", real);
        save("_reduced_fg_imag_per_m_hz", imag);
        save("_second_real_per_m_per_hz", secondReal);
        save("_second_imag_per_m_per_hz", secondImag);
      }
    }
    auto manifest = root / "response.yaml";
    std::ofstream out(manifest);
    out << "schema_version: 2\nprofile_fingerprint: oracle\ndyadic_basis: "
           "cylindrical_r_phi_z\n"
        << "frequency_interpolation: banded_not_a_knot_cubic_phase_reduced_fG\n"
        << "observer_z_m: -100\nobserver_refractive_index: 1.5\nnpz: "
           "response.npz\nnpz_sha256: '"
        << sha256_file(npz) << "'\n";
    return manifest;
  }
} // namespace

TEST_CASE("Ordered firn full-wave projection matches CPU response",
          "[kokkos][firn][radio]") {
  std::optional<corsika::kokkos::Runtime> runtime;
  auto const fixture = make_fixture();
  auto const path = std::getenv("CORSIKA_FIRN_RADIO_TEST_TABLE");
  auto const manifest = path ? std::filesystem::path(path) : fixture;
  auto const fingerprint =
      YAML::LoadFile(manifest.string())["profile_fingerprint"].as<std::string>();
  FrequencyGrid const grid{1.e-9, 256};
  std::vector<vector_type> const observers{{560.5, 0., -100.},
                                           {563.5, 0., -100.},
                                           {0., 563.5, -100.},
                                           {-563.5, 0., -100.},
                                           {0., -563.5, -100.}};
  std::vector<FullWaveChildSegment> children;
  for (std::size_t i = 0; i < 37; ++i) {
    double phase = 2. * pi * i / 37.;
    children.push_back(
        {{.035 * std::cos(phase), .025 * std::sin(phase), -213.95 + .8 * i / 36.},
         {2.e-4 * std::cos(1.7 * phase), 1.5e-4 * std::sin(1.3 * phase),
          7.e-4 + 1.e-5 * i},
         3.e-9 + i * 1.1e-11,
         i % 3 == 0 ? 1.25 : -.75});
  }
  TabulatedCoherentFullWaveBackend cpu(manifest, fingerprint);
  FullWaveSpectra reference;
  accumulate_cpu(cpu, observers, children, grid, 0., reference);
  FullWaveDeviceConfig config;
  config.grid_ = grid;
  config.batchSegments_ = 13;
  auto device = make_full_wave_accumulator(manifest, fingerprint, observers, config);
  // The application constructs radio before the EM runtime. Metadata access,
  // empty events and resets must not take ownership of Kokkos initialization.
  CHECK(device->statistics().totalDeviceBytes_ == 0);
  CHECK(relative_l2(device->download().total_, reference.total_) == 1.);
  device->reset();
  CHECK(device->statistics().totalDeviceBytes_ == 0);
  runtime.emplace();
  CHECK(runtime->ownsInitialization());
  for (auto const& child : children) device->append(child);
  auto first = device->download();
  double const error = relative_l2(first.total_, reference.total_);
  std::cout << "FIRN_ORACLE execution=" << device->executionSpaceName()
            << " relative_l2=" << error << "\n";
  // Preserve the archived CUDA oracle gate. Phase restoration amplifies
  // roundoff in optical paths; the unchanged CUDA kernels have the same error.
  CHECK(error < 2.e-10);
  CHECK(relative_l2(first.direct_, reference.direct_) < 2.e-10);
  CHECK(relative_l2(first.boundary_, reference.boundary_) < 2.e-10);
  CHECK(device->statistics().inputSegments_ == 37);
  CHECK(device->statistics().batches_ == 3);
  CHECK(device->statistics().tableDomainFailures_ == 0);
  device->reset();
  CHECK(relative_l2(device->download().total_, first.total_) == 1.);
  for (auto const& child : children) device->append(child);
  CHECK(relative_l2(device->download().total_, first.total_) == 0.);
  config.batchSegments_ = 37;
  auto single = make_full_wave_accumulator(manifest, fingerprint, observers, config);
  for (auto const& child : children) single->append(child);
  CHECK(relative_l2(single->download().total_, first.total_) == 0.);
  config.retainBranchSpectra_ = false;
  auto totalOnly = make_full_wave_accumulator(manifest, fingerprint, observers, config);
  for (auto const& child : children) totalOnly->append(child);
  auto total = totalOnly->download();
  CHECK(total.direct_.empty());
  CHECK(relative_l2(total.total_, first.total_) == 0.);
  auto outside = children.front();
  outside.midpointM_[2] = -220.;
  device->reset();
  device->append(outside);
  CHECK_THROWS(device->flush());
  CHECK(device->statistics().tableDomainFailures_ == observers.size());
  auto nonfinite = children.front();
  nonfinite.midpointM_[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(device->append(nonfinite));
  config.memoryLimitBytes_ = 1;
  CHECK_THROWS(make_full_wave_accumulator(manifest, fingerprint, observers, config));
  CHECK_THROWS(make_full_wave_accumulator(manifest, "wrong", observers, config));
}
