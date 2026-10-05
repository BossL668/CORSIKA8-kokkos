#include "Egs4Tables.hpp"
#include <fstream>
#include <sstream>

namespace c7_egs4 {
namespace {
void require(bool ok, char const* message) { if (!ok) throw std::runtime_error(message); }
std::string line(std::istream& in) {
  std::string s; require(bool(std::getline(in, s)), "Truncated EGSDAT input"); return s;
}
// Each HATCH READ starts a new record; unused columns of that record are skipped.
std::vector<float> block(std::istream& in, std::size_t count) {
  std::vector<float> values;
  while (values.size() < count) {
    std::istringstream row(line(in)); double x; std::size_t before = values.size();
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
}
Tables readAirTables(std::string const& path) {
  std::ifstream in(path); require(bool(in), "Cannot open EGSDAT file");
  return readAirTables(in);
}
Tables readAirTables(std::istream& in) {
  Tables t;
  auto header = line(in);
  require(header.find(" MEDIUM=AIR-NTP") == 0, "Only C7 AIR-NTP is supported in this prototype");
  t.name = "AIR-NTP";
  auto mixture = line(in);
  auto r = mixture.find("RHO="), n = mixture.find("NE=");
  require(r != std::string::npos && n != std::string::npos, "Missing EGSDAT material fields");
  t.medium.rho = static_cast<float>(std::stod(mixture.substr(r + 4)));
  int elements = std::stoi(mixture.substr(n + 3));
  require(elements > 0 && elements <= 100, "Invalid EGSDAT composition count");
  for (int i = 0; i < elements; ++i) line(in);
  auto constants = block(in, 5);
  auto& m = t.medium;
  m.radiation_length_cm = constants[0]; m.ae = constants[1]; m.ap = constants[2];
  m.ue = constants[3]; m.up = constants[4];
  require(m.rho > 0. && std::isfinite(m.rho) && m.radiation_length_cm > 0. &&
          m.ap > 0. && m.ae > 0. && m.ue > m.ae && m.up > m.ap, "Invalid EGSDAT medium");
  int sizes[8]; std::istringstream sizes_in(line(in));
  for (auto& size : sizes) require(bool(sizes_in >> size), "Malformed EGSDAT sizes");
  require(sizes[0] == 0 && sizes[2] == 0 && sizes[4] == 0 && sizes[5] == 0 &&
          sizes[6] == 0 && sizes[7] == 0, "Unsupported EGSDAT auxiliary or Rayleigh tables");
  require(sizes[1] > 0 && sizes[1] <= 500 && sizes[3] > 0 && sizes[3] <= 500,
          "Invalid EGSDAT table sizes");
  t.brems_pair_coefficients = block(in, 36);
  auto b = block(in, 7);
  t.brems_pair_coefficients.insert(t.brems_pair_coefficients.end(), b.begin(), b.end());
  auto scatter = block(in, 4);
  double inverse_rlc = 1. / m.radiation_length_cm;
  // Match HATCH's DOUBLE arithmetic followed by assignment into REAL tables.
  m.teff_cm = static_cast<float>(double(scatter[1]) * m.radiation_length_cm);
  m.blcc = static_cast<float>(double(scatter[2]) * inverse_rlc);
  m.xcc = static_cast<float>(double(scatter[3]) * std::sqrt(inverse_rlc));
  auto eindex = block(in, 2); m.eke0 = eindex[0]; m.eke1 = eindex[1];
  auto e = block(in, std::size_t(sizes[3]) * 20);
  t.electrons.resize(sizes[3]);
  for (std::size_t i = 0; i < t.electrons.size(); ++i) {
    for (int j = 0; j < 20; ++j) {
      double value = e[i * 20 + j];
      if (j < 8) value *= inverse_rlc;
      if (j >= 18) value *= m.radiation_length_cm;
      t.electrons[i].c[j] = static_cast<float>(value);
    }
  }
  auto pindex = block(in, 3); m.binding_energy_MeV = pindex[0];
  m.ge0 = pindex[1]; m.ge1 = pindex[2];
  require(m.binding_energy_MeV >= 0., "Invalid EGSDAT binding energy");
  require(m.eke1 > 0. && m.ge1 > 0., "Invalid EGSDAT logarithmic grid");
  auto p = block(in, std::size_t(sizes[1]) * 10);
  t.photons.resize(sizes[1]);
  for (std::size_t i = 0; i < t.photons.size(); ++i)
    for (int j = 0; j < 10; ++j)
      t.photons[i].c[j] = static_cast<float>(double(p[i * 10 + j]) * (j < 2 ? m.radiation_length_cm : 1.));
  return t;
}
} // namespace c7_egs4
