/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/egs4/Tables.hpp>
#include <fstream>
#include <sstream>

namespace corsika::kokkos::egs4 {
  namespace {
    void require(bool ok, char const* message) {
      if (!ok) throw std::runtime_error(message);
    }
    std::string line(std::istream& in) {
      std::string s;
      require(bool(std::getline(in, s)), "Truncated EGSDAT input");
      return s;
    }
    // Each HATCH READ starts a new record; unused columns of that record are skipped.
    std::vector<float> block(std::istream& in, std::size_t count) {
      std::vector<float> values;
      while (values.size() < count) {
        std::istringstream row(line(in));
        double x;
        std::size_t before = values.size();
        while (values.size() < count && row >> x) {
          require(std::isfinite(x), "Non-finite EGSDAT coefficient");
          auto f = static_cast<float>(x);
          require(std::isfinite(f), "EGSDAT coefficient overflows C7 REAL");
          values.push_back(f);
        }
        require(values.size() != before, "Malformed EGSDAT numeric record");
      }
      return values;
    }
  } // namespace
  Tables read_air_tables(std::string const& path) {
    std::ifstream in(path);
    require(bool(in), "Cannot open EGSDAT file");
    return read_air_tables(in);
  }
  Tables read_air_tables(std::istream& in) {
    Tables t;
    auto header = line(in);
    require(header.find(" MEDIUM=AIR-NTP") == 0,
            "Only C7 AIR-NTP is supported in this prototype");
    t.name_ = "AIR-NTP";
    auto mixture = line(in);
    auto r = mixture.find("RHO="), n = mixture.find("NE=");
    require(r != std::string::npos && n != std::string::npos,
            "Missing EGSDAT material fields");
    t.medium_.rho_ = static_cast<float>(std::stod(mixture.substr(r + 4)));
    int elements = std::stoi(mixture.substr(n + 3));
    require(elements > 0 && elements <= 100, "Invalid EGSDAT composition count");
    for (int i = 0; i < elements; ++i) line(in);
    auto constants = block(in, 5);
    auto& m = t.medium_;
    m.radiationLengthCm_ = constants[0];
    m.ae_ = constants[1];
    m.ap_ = constants[2];
    m.ue_ = constants[3];
    m.up_ = constants[4];
    require(m.rho_ > 0. && std::isfinite(m.rho_) && m.radiationLengthCm_ > 0. &&
                m.ap_ > 0. && m.ae_ > 0. && m.ue_ > m.ae_ && m.up_ > m.ap_,
            "Invalid EGSDAT medium");
    int sizes[8];
    std::istringstream sizes_in(line(in));
    for (auto& size : sizes) require(bool(sizes_in >> size), "Malformed EGSDAT sizes");
    require(sizes[0] == 0 && sizes[2] == 0 && sizes[4] == 0 && sizes[5] == 0 &&
                sizes[6] == 0 && sizes[7] == 0,
            "Unsupported EGSDAT auxiliary or Rayleigh tables");
    require(sizes[1] > 0 && sizes[1] <= 500 && sizes[3] > 0 && sizes[3] <= 500,
            "Invalid EGSDAT table sizes");
    t.bremsPairCoefficients_ = block(in, 36);
    auto b = block(in, 7);
    t.bremsPairCoefficients_.insert(t.bremsPairCoefficients_.end(), b.begin(), b.end());
    auto scatter = block(in, 4);
    double inverse_rlc = 1. / m.radiationLengthCm_;
    // Match HATCH's DOUBLE arithmetic followed by assignment into REAL tables.
    m.teffCm_ = static_cast<float>(double(scatter[1]) * m.radiationLengthCm_);
    m.blcc_ = static_cast<float>(double(scatter[2]) * inverse_rlc);
    m.xcc_ = static_cast<float>(double(scatter[3]) * std::sqrt(inverse_rlc));
    auto eindex = block(in, 2);
    m.eke0_ = eindex[0];
    m.eke1_ = eindex[1];
    auto e = block(in, std::size_t(sizes[3]) * 20);
    t.electrons_.resize(sizes[3]);
    for (std::size_t i = 0; i < t.electrons_.size(); ++i) {
      for (int j = 0; j < 20; ++j) {
        double value = e[i * 20 + j];
        if (j < 8) value *= inverse_rlc;
        if (j >= 18) value *= m.radiationLengthCm_;
        t.electrons_[i].c_[j] = static_cast<float>(value);
      }
    }
    auto pindex = block(in, 3);
    m.bindingEnergyMeV_ = pindex[0];
    m.ge0_ = pindex[1];
    m.ge1_ = pindex[2];
    require(m.bindingEnergyMeV_ >= 0., "Invalid EGSDAT binding energy");
    require(m.eke1_ > 0. && m.ge1_ > 0., "Invalid EGSDAT logarithmic grid");
    auto p = block(in, std::size_t(sizes[1]) * 10);
    t.photons_.resize(sizes[1]);
    for (std::size_t i = 0; i < t.photons_.size(); ++i)
      for (int j = 0; j < 10; ++j)
        t.photons_[i].c_[j] = static_cast<float>(double(p[i * 10 + j]) *
                                                 (j < 2 ? m.radiationLengthCm_ : 1.));
    return t;
  }
} // namespace corsika::kokkos::egs4
