/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline TabulatedFullWaveBackend::TabulatedFullWaveBackend(
      std::filesystem::path manifestPath, std::string expectedProfileFingerprint)
      : manifestPath_(std::move(manifestPath)) {
    YAML::Node const manifest = YAML::LoadFile(manifestPath_.string());
    if (manifest["schema_version"].as<int>() != 1) {
      throw std::runtime_error("unsupported full-wave response schema");
    }
    profileFingerprint_ = manifest["profile_fingerprint"].as<std::string>();
    if (profileFingerprint_ != expectedProfileFingerprint) {
      throw std::runtime_error("full-wave response profile fingerprint mismatch");
    }
    if (manifest["dyadic_basis"].as<std::string>() != "cylindrical_r_phi_z") {
      throw std::runtime_error("unsupported full-wave dyadic basis");
    }
    observerZM_ = manifest["observer_z_m"].as<double>();
    observerIndex_ = manifest["observer_refractive_index"].as<double>();
    if (!std::isfinite(observerZM_) || !(observerIndex_ >= 1.0)) {
      throw std::runtime_error("invalid fixed observer plane in manifest");
    }
    std::filesystem::path npz = manifest["npz"].as<std::string>();
    if (npz.is_relative()) npz = manifestPath_.parent_path() / npz;
    if (!manifest["npz_sha256"]) {
      throw std::runtime_error("full-wave manifest is missing npz_sha256");
    }
    std::string const expectedSha256 = manifest["npz_sha256"].as<std::string>();
    std::string const actualSha256 = sha256_file(npz);
    if (expectedSha256 != actualSha256) {
      throw std::runtime_error("full-wave response SHA-256 mismatch");
    }
    load(npz);
  }

  inline GreenResponseBatch TabulatedFullWaveBackend::evaluate(
      vector_type const& source, vector_type const& observer,
      FrequencyGrid const& grid) const {
    grid.validate();
    double const radius = std::hypot(observer[0] - source[0], observer[1] - source[1]);
    if (source[2] < sourceZ_.front() || source[2] > sourceZ_.back() ||
        radius < radiusM_.front() || radius > radiusM_.back() ||
        grid.nyquistHz() > frequencyHz_.back() + 1.0e-6 ||
        grid.spacingHz() < frequencyHz_.front() - 1.0e-6 ||
        std::abs(observer[2] - observerZM_) > 1.0e-9) {
      throw std::runtime_error("full-wave response query requires extrapolation");
    }
    GreenResponseBatch result;
    result.path_.valid_ = true;
    result.path_.topology_ = "tabulated_fullwave_total";
    result.path_.sourceRefractiveIndex_ = interpolateSourceIndex(source[2]);
    result.path_.observerRefractiveIndex_ = observerIndex_;
    result.path_.geometricLengthM_ = norm(subtract(observer, source));
    result.path_.opticalPathM_ = interpolatePath(source[2], radius);
    result.path_.travelTimeS_ = result.path_.opticalPathM_ / c_m_per_s;
    result.path_.geometricSpreadingPerM_ = interpolateSpreading(source[2], radius);
    result.path_.emitDirection_ = emitDirection(source, observer);
    result.path_.receiveDirection_ = normalized(subtract(observer, source));
    result.dyadicPerM_.resize(grid.frequencyCount(), zero_matrix());
    for (std::size_t out = 0; out < grid.frequencyCount(); ++out) {
      double const frequency = grid.frequencyHz(out);
      if (frequency == 0.0) continue;
      complex_matrix_type local = zero_matrix();
      interpolateDyadic(frequency, source[2], radius, result.path_.opticalPathM_, local);
      result.dyadicPerM_[out] = rotateFromCylindrical(local, source, observer);
    }
    return result;
  }

  inline std::string TabulatedFullWaveBackend::name() const {
    return "tabulated_fullwave";
  }

  inline void TabulatedFullWaveBackend::requireVector(cnpy::NpyArray const& array,
                                                      char const* name) {
    if (array.word_size != sizeof(double) || array.shape.size() != 1 ||
        array.shape[0] < 2 || array.fortran_order) {
      throw std::runtime_error(std::string("invalid NPZ vector: ") + name);
    }
  }

  inline void TabulatedFullWaveBackend::load(std::filesystem::path const& npzPath) {
    cnpy::npz_t const archive = cnpy::npz_load(npzPath.string());
    auto get = [&](char const* key) -> cnpy::NpyArray const& {
      auto const found = archive.find(key);
      if (found == archive.end()) {
        throw std::runtime_error(std::string("missing NPZ array: ") + key);
      }
      return found->second;
    };
    auto const& frequency = get("frequency_hz");
    auto const& sourceZ = get("source_z_m");
    auto const& radius = get("horizontal_radius_m");
    requireVector(frequency, "frequency_hz");
    requireVector(sourceZ, "source_z_m");
    requireVector(radius, "horizontal_radius_m");
    frequencyHz_ = frequency.as_vec<double>();
    sourceZ_ = sourceZ.as_vec<double>();
    radiusM_ = radius.as_vec<double>();
    auto const& sourceIndex = get("source_refractive_index");
    if (sourceIndex.shape != std::vector<std::size_t>{sourceZ_.size()} ||
        sourceIndex.word_size != sizeof(double) || sourceIndex.fortran_order) {
      throw std::runtime_error("source refractive-index array has wrong shape");
    }
    sourceIndex_ = sourceIndex.as_vec<double>();
    auto const& real = get("green_cylindrical_real_per_m");
    auto const& imag = get("green_cylindrical_imag_per_m");
    std::vector<std::size_t> const expected{frequencyHz_.size(), sourceZ_.size(),
                                            radiusM_.size(), 3, 3};
    if (real.shape != expected || imag.shape != expected ||
        real.word_size != sizeof(double) || imag.word_size != sizeof(double) ||
        real.fortran_order || imag.fortran_order) {
      throw std::runtime_error("full-wave dyadic array has wrong shape or type");
    }
    real_ = real.as_vec<double>();
    imag_ = imag.as_vec<double>();
    auto const& path = get("reduction_optical_path_m");
    if (path.shape != std::vector<std::size_t>{sourceZ_.size(), radiusM_.size()} ||
        path.word_size != sizeof(double) || path.fortran_order) {
      throw std::runtime_error("full-wave path array has wrong shape or type");
    }
    pathM_ = path.as_vec<double>();
    auto const& spreading = get("geometric_spreading_per_m");
    if (spreading.shape != std::vector<std::size_t>{sourceZ_.size(), radiusM_.size()} ||
        spreading.word_size != sizeof(double) || spreading.fortran_order) {
      throw std::runtime_error("full-wave spreading array has wrong shape");
    }
    spreadingPerM_ = spreading.as_vec<double>();
    increasing(frequencyHz_, "frequency");
    increasing(sourceZ_, "source z");
    increasing(radiusM_, "radius");
    for (double value : sourceIndex_) {
      if (!(value >= 1.0) || !std::isfinite(value)) {
        throw std::runtime_error("invalid source refractive index in table");
      }
    }
    for (double value : spreadingPerM_) {
      if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::runtime_error("invalid geometric spreading in table");
      }
    }
  }

  inline void TabulatedFullWaveBackend::increasing(std::vector<double> const& values,
                                                   char const* name) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (!std::isfinite(values[index]) ||
          (index && !(values[index] > values[index - 1]))) {
        throw std::runtime_error(std::string("non-increasing table axis: ") + name);
      }
    }
  }

  inline std::pair<std::size_t, double> TabulatedFullWaveBackend::bracket(
      std::vector<double> const& axis, double value) {
    auto const upper = std::upper_bound(axis.begin(), axis.end(), value);
    std::size_t hi = static_cast<std::size_t>(upper - axis.begin());
    hi = std::clamp<std::size_t>(hi, 1, axis.size() - 1);
    std::size_t const lo = hi - 1;
    return {lo, (value - axis[lo]) / (axis[hi] - axis[lo])};
  }

  inline std::size_t TabulatedFullWaveBackend::index(std::size_t fi, std::size_t zi,
                                                     std::size_t ri, std::size_t row,
                                                     std::size_t col) const {
    return ((((fi * sourceZ_.size() + zi) * radiusM_.size() + ri) * 3 + row) * 3 + col);
  }

  inline double TabulatedFullWaveBackend::interpolatePath(double z, double radius) const {
    auto const [zi, wz] = bracket(sourceZ_, z);
    auto const [ri, wr] = bracket(radiusM_, radius);
    auto value = [&](std::size_t dz, std::size_t dr) {
      return pathM_[(zi + dz) * radiusM_.size() + ri + dr];
    };
    return (1.0 - wz) * ((1.0 - wr) * value(0, 0) + wr * value(0, 1)) +
           wz * ((1.0 - wr) * value(1, 0) + wr * value(1, 1));
  }

  inline double TabulatedFullWaveBackend::interpolateSourceIndex(double z) const {
    auto const [zi, wz] = bracket(sourceZ_, z);
    return (1.0 - wz) * sourceIndex_[zi] + wz * sourceIndex_[zi + 1];
  }

  inline double TabulatedFullWaveBackend::interpolateSpreading(double z,
                                                               double radius) const {
    auto const [zi, wz] = bracket(sourceZ_, z);
    auto const [ri, wr] = bracket(radiusM_, radius);
    auto value = [&](std::size_t dz, std::size_t dr) {
      return spreadingPerM_[(zi + dz) * radiusM_.size() + ri + dr];
    };
    return (1.0 - wz) * ((1.0 - wr) * value(0, 0) + wr * value(0, 1)) +
           wz * ((1.0 - wr) * value(1, 0) + wr * value(1, 1));
  }

  inline vector_type TabulatedFullWaveBackend::emitDirection(
      vector_type const& source, vector_type const& observer) const {
    double const radius = std::hypot(observer[0] - source[0], observer[1] - source[1]);
    auto const [zi, wz] = bracket(sourceZ_, source[2]);
    auto const [ri, wr] = bracket(radiusM_, radius);
    auto path = [&](std::size_t dz, std::size_t dr) {
      return pathM_[(zi + dz) * radiusM_.size() + ri + dr];
    };
    double const dPathDr =
        ((1.0 - wz) * (path(0, 1) - path(0, 0)) + wz * (path(1, 1) - path(1, 0))) /
        (radiusM_[ri + 1] - radiusM_[ri]);
    double const dPathDz =
        ((1.0 - wr) * (path(1, 0) - path(0, 0)) + wr * (path(1, 1) - path(0, 1))) /
        (sourceZ_[zi + 1] - sourceZ_[zi]);
    vector_type const radial = radius > 0.0
                                   ? vector_type{(observer[0] - source[0]) / radius,
                                                 (observer[1] - source[1]) / radius, 0.0}
                                   : vector_type{1.0, 0.0, 0.0};
    // Hamilton-Jacobi endpoint relation: grad_source(S)=-n_s t_emit.
    return normalized(add(multiply(radial, dPathDr), vector_type{0.0, 0.0, -dPathDz}));
  }

  inline complex_matrix_type TabulatedFullWaveBackend::rotateFromCylindrical(
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

  inline void TabulatedFullWaveBackend::interpolateDyadic(
      double frequency, double z, double radius, double restorePath,
      complex_matrix_type& output) const {
    auto const [fi, wf] = bracket(frequencyHz_, frequency);
    auto const [zi, wz] = bracket(sourceZ_, z);
    auto const [ri, wr] = bracket(radiusM_, radius);
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t col = 0; col < 3; ++col) {
        std::complex<double> frequencyValues[2]{};
        for (std::size_t df = 0; df < 2; ++df) {
          double const nodeFrequency = frequencyHz_[fi + df];
          std::complex<double> spatial = 0.0;
          for (std::size_t dz = 0; dz < 2; ++dz) {
            for (std::size_t dr = 0; dr < 2; ++dr) {
              std::size_t const flat = index(fi + df, zi + dz, ri + dr, row, col);
              std::complex<double> value(real_[flat], imag_[flat]);
              double const nodePath = pathM_[(zi + dz) * radiusM_.size() + ri + dr];
              value *= std::exp(std::complex<double>(
                  0.0, 2.0 * pi * nodeFrequency * nodePath / c_m_per_s));
              double const weightZ = dz ? wz : 1.0 - wz;
              double const weightR = dr ? wr : 1.0 - wr;
              spatial += weightZ * weightR * value;
            }
          }
          frequencyValues[df] = spatial;
        }
        std::complex<double> const reduced =
            (1.0 - wf) * frequencyValues[0] + wf * frequencyValues[1];
        output[row][col] =
            reduced * std::exp(std::complex<double>(
                          0.0, -2.0 * pi * frequency * restorePath / c_m_per_s));
      }
    }
  }
} // namespace corsika::kokkos::radio::firn
