/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline TabulatedCoherentFullWaveBackend::TabulatedCoherentFullWaveBackend(
      std::filesystem::path manifestPath, std::string expectedProfileFingerprint)
      : manifestPath_(std::move(manifestPath)) {
    YAML::Node const manifest = YAML::LoadFile(manifestPath_.string());
    if (manifest["schema_version"].as<int>() != 2 ||
        manifest["dyadic_basis"].as<std::string>() != "cylindrical_r_phi_z" ||
        manifest["frequency_interpolation"].as<std::string>() !=
            "banded_not_a_knot_cubic_phase_reduced_fG") {
      throw std::runtime_error("unsupported coherent full-wave response schema");
    }
    profileFingerprint_ = manifest["profile_fingerprint"].as<std::string>();
    if (profileFingerprint_ != expectedProfileFingerprint) {
      throw std::runtime_error(
          "coherent full-wave response profile fingerprint mismatch");
    }
    observerZM_ = manifest["observer_z_m"].as<double>();
    observerIndex_ = manifest["observer_refractive_index"].as<double>();
    if (!std::isfinite(observerZM_) || !(observerIndex_ >= 1.0)) {
      throw std::runtime_error("invalid coherent full-wave observer plane");
    }
    std::filesystem::path npz = manifest["npz"].as<std::string>();
    if (npz.is_relative()) npz = manifestPath_.parent_path() / npz;
    if (!manifest["npz_sha256"] ||
        sha256_file(npz) != manifest["npz_sha256"].as<std::string>()) {
      throw std::runtime_error("coherent full-wave response SHA-256 mismatch");
    }
    load(npz);
  }

  inline bool TabulatedCoherentFullWaveBackend::covers(vector_type const& source,
                                                       vector_type const& observer,
                                                       FrequencyGrid const& grid) const {
    grid.validate();
    double const radius = std::hypot(observer[0] - source[0], observer[1] - source[1]);
    return source[2] >= sourceZ_.front() && source[2] <= sourceZ_.back() &&
           radius >= radiusM_.front() && radius <= radiusM_.back() &&
           std::abs(observer[2] - observerZM_) <= 1.0e-9 &&
           grid.nyquistHz() <= bands_.back().frequencyHz_.back() + 1.0e-6;
  }

  inline GreenResponseBatch TabulatedCoherentFullWaveBackend::evaluate(
      vector_type const& source, vector_type const& observer,
      FrequencyGrid const& grid) const {
    if (!covers(source, observer, grid)) {
      throw std::runtime_error(
          "coherent full-wave response query requires extrapolation");
    }
    double const radius = std::hypot(observer[0] - source[0], observer[1] - source[1]);
    GreenResponseBranch direct = branch(source, observer, radius, grid, directPathM_,
                                        true, "fullwave_direct_matched");
    GreenResponseBranch boundary = branch(source, observer, radius, grid, boundaryPathM_,
                                          false, "fullwave_air_boundary");
    GreenResponseBatch result;
    result.branches_ = {std::move(direct), std::move(boundary)};
    result.dyadicPerM_.assign(grid.frequencyCount(), zero_matrix());
    for (auto const& item : result.branches_) {
      for (std::size_t frequency = 0; frequency < grid.frequencyCount(); ++frequency) {
        for (std::size_t row = 0; row < 3; ++row) {
          for (std::size_t col = 0; col < 3; ++col) {
            result.dyadicPerM_[frequency][row][col] +=
                item.dyadicPerM_[frequency][row][col];
          }
        }
      }
    }
    result.path_ = result.branches_.front().path_;
    result.path_.topology_ = "fullwave_coherent_total";
    result.finiteTrackCoupling_ = FiniteTrackCoupling::PointCurrentDyadic;
    return result;
  }

  inline std::string TabulatedCoherentFullWaveBackend::name() const {
    return "tabulated_coherent_fullwave";
  }

  inline void TabulatedCoherentFullWaveBackend::requireVector(cnpy::NpyArray const& array,
                                                              char const* name) {
    if (array.word_size != sizeof(double) || array.shape.size() != 1 ||
        array.shape[0] < 2 || array.fortran_order) {
      throw std::runtime_error(std::string("invalid NPZ vector: ") + name);
    }
  }

  inline void TabulatedCoherentFullWaveBackend::load(
      std::filesystem::path const& npzPath) {
    cnpy::npz_t const archive = cnpy::npz_load(npzPath.string());
    auto get = [&](std::string const& key) -> cnpy::NpyArray const& {
      auto const found = archive.find(key);
      if (found == archive.end()) {
        throw std::runtime_error("missing coherent full-wave NPZ array: " + key);
      }
      return found->second;
    };
    auto loadVector = [&](char const* key) {
      auto const& value = get(key);
      requireVector(value, key);
      return value.as_vec<double>();
    };
    sourceZ_ = loadVector("source_z_m");
    radiusM_ = loadVector("horizontal_radius_m");
    sourceIndex_ = loadVector("source_refractive_index");
    increasing(sourceZ_, "source z");
    increasing(radiusM_, "radius");
    if (sourceIndex_.size() != sourceZ_.size()) {
      throw std::runtime_error("coherent full-wave source-index shape differs");
    }
    directPathM_ = loadSurface(get("direct_reduction_optical_path_m"), "direct path");
    boundaryPathM_ =
        loadSurface(get("boundary_reduction_optical_path_m"), "boundary path");
    for (std::size_t bandIndex = 0; bandIndex < 4; ++bandIndex) {
      Band band;
      std::string const prefix = "band" + std::to_string(bandIndex) + "_";
      auto const& frequency = get(prefix + "frequency_hz");
      requireVector(frequency, (prefix + "frequency_hz").c_str());
      band.frequencyHz_ = frequency.as_vec<double>();
      increasing(band.frequencyHz_, (prefix + "frequency").c_str());
      band.direct_ = loadBranchBand(archive, prefix, "direct", band.frequencyHz_.size());
      band.boundary_ =
          loadBranchBand(archive, prefix, "boundary", band.frequencyHz_.size());
      bands_.push_back(std::move(band));
    }
    if (bands_.front().frequencyHz_.front() != 0.0 ||
        bands_.back().frequencyHz_.back() < 500.0e6) {
      throw std::runtime_error("coherent full-wave bands do not cover DC--500 MHz");
    }
  }

  inline TabulatedCoherentFullWaveBackend::BranchBand
  TabulatedCoherentFullWaveBackend::loadBranchBand(cnpy::npz_t const& archive,
                                                   std::string const& prefix,
                                                   std::string const& branch,
                                                   std::size_t frequencyCount) const {
    auto get = [&](std::string const& suffix) -> cnpy::NpyArray const& {
      std::string const key = prefix + branch + suffix;
      auto const found = archive.find(key);
      if (found == archive.end()) {
        throw std::runtime_error("missing coherent full-wave NPZ array: " + key);
      }
      return found->second;
    };
    std::vector<std::size_t> const expected{frequencyCount, sourceZ_.size(),
                                            radiusM_.size(), 3, 3};
    auto read = [&](std::string const& suffix) {
      auto const& value = get(suffix);
      if (value.shape != expected || value.word_size != sizeof(double) ||
          value.fortran_order) {
        throw std::runtime_error("coherent full-wave band shape differs");
      }
      return value.as_vec<double>();
    };
    return {read("_reduced_fg_real_per_m_hz"), read("_reduced_fg_imag_per_m_hz"),
            read("_second_real_per_m_per_hz"), read("_second_imag_per_m_per_hz")};
  }

  inline std::vector<double> TabulatedCoherentFullWaveBackend::loadSurface(
      cnpy::NpyArray const& value, char const* name) const {
    std::vector<std::size_t> const expected{sourceZ_.size(), radiusM_.size()};
    if (value.shape != expected || value.word_size != sizeof(double) ||
        value.fortran_order) {
      throw std::runtime_error(std::string("invalid coherent full-wave ") + name);
    }
    auto result = value.as_vec<double>();
    for (double item : result) {
      if (!(item > 0.0) || !std::isfinite(item)) {
        throw std::runtime_error(std::string("non-finite coherent full-wave ") + name);
      }
    }
    return result;
  }

  inline void TabulatedCoherentFullWaveBackend::increasing(
      std::vector<double> const& values, char const* name) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (!std::isfinite(values[index]) ||
          (index && !(values[index] > values[index - 1]))) {
        throw std::runtime_error(std::string("non-increasing table axis: ") + name);
      }
    }
  }

  inline std::pair<std::size_t, double> TabulatedCoherentFullWaveBackend::bracket(
      std::vector<double> const& axis, double value) {
    auto upper = std::upper_bound(axis.begin(), axis.end(), value);
    std::size_t hi = static_cast<std::size_t>(upper - axis.begin());
    hi = std::clamp<std::size_t>(hi, 1, axis.size() - 1);
    std::size_t const lo = hi - 1;
    return {lo, (value - axis[lo]) / (axis[hi] - axis[lo])};
  }

  inline std::size_t TabulatedCoherentFullWaveBackend::flat(std::size_t fi,
                                                            std::size_t zi,
                                                            std::size_t ri,
                                                            std::size_t row,
                                                            std::size_t col) const {
    return ((((fi * sourceZ_.size() + zi) * radiusM_.size() + ri) * 3 + row) * 3 + col);
  }

  inline double TabulatedCoherentFullWaveBackend::spatial(
      std::vector<double> const& values, std::size_t fi, double z, double radius,
      std::size_t row, std::size_t col) const {
    auto const [zi, wz] = bracket(sourceZ_, z);
    auto const [ri, wr] = bracket(radiusM_, radius);
    auto value = [&](std::size_t dz, std::size_t dr) {
      return values[flat(fi, zi + dz, ri + dr, row, col)];
    };
    return (1.0 - wz) * ((1.0 - wr) * value(0, 0) + wr * value(0, 1)) +
           wz * ((1.0 - wr) * value(1, 0) + wr * value(1, 1));
  }

  inline double TabulatedCoherentFullWaveBackend::surface(
      std::vector<double> const& values, double z, double radius) const {
    auto const [zi, wz] = bracket(sourceZ_, z);
    auto const [ri, wr] = bracket(radiusM_, radius);
    auto value = [&](std::size_t dz, std::size_t dr) {
      return values[(zi + dz) * radiusM_.size() + ri + dr];
    };
    return (1.0 - wz) * ((1.0 - wr) * value(0, 0) + wr * value(0, 1)) +
           wz * ((1.0 - wr) * value(1, 0) + wr * value(1, 1));
  }

  inline TabulatedCoherentFullWaveBackend::Band const&
  TabulatedCoherentFullWaveBackend::selectBand(double frequency) const {
    for (auto const& band : bands_) {
      if (frequency >= band.frequencyHz_.front() - 1.0e-6 &&
          frequency <= band.frequencyHz_.back() + 1.0e-6) {
        return band;
      }
    }
    throw std::runtime_error("full-wave frequency lies between table bands");
  }

  inline std::complex<double> TabulatedCoherentFullWaveBackend::interpolateReduced(
      Band const& band, BranchBand const& data, double frequency, double z, double radius,
      std::size_t row, std::size_t col) const {
    auto const [fi, weight] = bracket(band.frequencyHz_, frequency);
    double const x0 = band.frequencyHz_[fi];
    double const x1 = band.frequencyHz_[fi + 1];
    double const h = x1 - x0;
    double const a = 1.0 - weight;
    double const b = weight;
    auto cubic = [&](std::vector<double> const& value,
                     std::vector<double> const& second) {
      double const y0 = spatial(value, fi, z, radius, row, col);
      double const y1 = spatial(value, fi + 1, z, radius, row, col);
      double const m0 = spatial(second, fi, z, radius, row, col);
      double const m1 = spatial(second, fi + 1, z, radius, row, col);
      return a * y0 + b * y1 +
             ((a * a * a - a) * m0 + (b * b * b - b) * m1) * h * h / 6.0;
    };
    return {cubic(data.real_, data.secondReal_), cubic(data.imag_, data.secondImag_)};
  }

  inline vector_type TabulatedCoherentFullWaveBackend::emitDirection(
      std::vector<double> const& path, vector_type const& source,
      vector_type const& observer) const {
    double const radius = std::hypot(observer[0] - source[0], observer[1] - source[1]);
    auto const [zi, wz] = bracket(sourceZ_, source[2]);
    auto const [ri, wr] = bracket(radiusM_, radius);
    auto value = [&](std::size_t dz, std::size_t dr) {
      return path[(zi + dz) * radiusM_.size() + ri + dr];
    };
    double const dPathDr =
        ((1.0 - wz) * (value(0, 1) - value(0, 0)) + wz * (value(1, 1) - value(1, 0))) /
        (radiusM_[ri + 1] - radiusM_[ri]);
    double const dPathDz =
        ((1.0 - wr) * (value(1, 0) - value(0, 0)) + wr * (value(1, 1) - value(0, 1))) /
        (sourceZ_[zi + 1] - sourceZ_[zi]);
    vector_type const radial = radius > 0.0
                                   ? vector_type{(observer[0] - source[0]) / radius,
                                                 (observer[1] - source[1]) / radius, 0.0}
                                   : vector_type{1.0, 0.0, 0.0};
    return normalized(add(multiply(radial, dPathDr), vector_type{0.0, 0.0, -dPathDz}));
  }

  inline complex_matrix_type TabulatedCoherentFullWaveBackend::rotateFromCylindrical(
      complex_matrix_type const& local, vector_type const& source,
      vector_type const& observer) {
    double const dx = observer[0] - source[0];
    double const dy = observer[1] - source[1];
    double const radius = std::hypot(dx, dy);
    vector_type const radial = radius > 0.0 ? vector_type{dx / radius, dy / radius, 0.0}
                                            : vector_type{1.0, 0.0, 0.0};
    vector_type const azimuthal{-radial[1], radial[0], 0.0};
    std::array<vector_type, 3> const basis{radial, azimuthal, vector_type{0.0, 0.0, 1.0}};
    complex_matrix_type global = zero_matrix();
    for (std::size_t index = 0; index < 3; ++index) {
      for (std::size_t componentIndex = 0; componentIndex < 3; ++componentIndex) {
        for (std::size_t a = 0; a < 3; ++a) {
          for (std::size_t b = 0; b < 3; ++b) {
            global[index][componentIndex] +=
                basis[a][index] * local[a][b] * basis[b][componentIndex];
          }
        }
      }
    }
    return global;
  }

  inline GreenResponseBranch TabulatedCoherentFullWaveBackend::branch(
      vector_type const& source, vector_type const& observer, double radius,
      FrequencyGrid const& grid, std::vector<double> const& pathValues, bool direct,
      char const* topology) const {
    GreenResponseBranch result;
    result.finiteTrackCoupling_ = FiniteTrackCoupling::PointCurrentDyadic;
    result.path_.valid_ = true;
    result.path_.topology_ = topology;
    auto const [zi, wz] = bracket(sourceZ_, source[2]);
    result.path_.sourceRefractiveIndex_ =
        (1.0 - wz) * sourceIndex_[zi] + wz * sourceIndex_[zi + 1];
    result.path_.observerRefractiveIndex_ = observerIndex_;
    result.path_.geometricLengthM_ = norm(subtract(observer, source));
    result.path_.opticalPathM_ = surface(pathValues, source[2], radius);
    result.path_.travelTimeS_ = result.path_.opticalPathM_ / c_m_per_s;
    result.path_.geometricSpreadingPerM_ =
        1.0 / std::max(result.path_.geometricLengthM_, 1.0e-12);
    result.path_.emitDirection_ = emitDirection(pathValues, source, observer);
    result.path_.receiveDirection_ = normalized(subtract(observer, source));
    result.dyadicPerM_.assign(grid.frequencyCount(), zero_matrix());
    for (std::size_t index = 1; index < grid.frequencyCount(); ++index) {
      double const frequency = grid.frequencyHz(index);
      Band const& band = selectBand(frequency);
      BranchBand const& data = direct ? band.direct_ : band.boundary_;
      complex_matrix_type local = zero_matrix();
      std::complex<double> const restore = std::exp(std::complex<double>(
          0.0, -2.0 * pi * frequency * result.path_.opticalPathM_ / c_m_per_s));
      for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
          local[row][col] =
              interpolateReduced(band, data, frequency, source[2], radius, row, col) *
              restore / frequency;
        }
      }
      result.dyadicPerM_[index] = rotateFromCylindrical(local, source, observer);
    }
    return result;
  }
} // namespace corsika::kokkos::radio::firn
