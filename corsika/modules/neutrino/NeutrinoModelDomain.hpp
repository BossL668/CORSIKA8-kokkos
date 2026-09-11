// Admission/audit only: never substitutes the upstream arbitrary 4 nb for a
// missing weak cross section, extrapolates CTW, or deposits a neutrino's energy.
#pragma once
#include <corsika/framework/core/ParticleProperties.hpp>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <unordered_set>

namespace corsika::neutrino {
inline constexpr double ctwMinimumEnergyGeV = 1.e4;
inline constexpr double ctwMaximumEnergyGeV = 1.e12;
inline bool inCtwEnergyDomain(double energy) noexcept {
  return std::isfinite(energy) && energy >= ctwMinimumEnergyGeV && energy <= ctwMaximumEnergyGeV;
}

class NeutrinoModelDomain {
 public:
  explicit NeutrinoModelDomain(bool requireCoverage = false) : strict_(requireCoverage) {}
  void check(Code pid, double energy) const {
    if (!is_neutrino(pid)) return;
    if (!std::isfinite(energy) || energy < 0.) throw std::runtime_error("invalid neutrino energy");
    if (strict_ && !inCtwEnergyDomain(energy))
      throw std::runtime_error("neutrino outside CTW2011 [1e4,1e12] GeV: full weak transport unavailable");
  }
  void observe(Code pid, double energy, double weight, std::uint64_t history) {
    check(pid, energy);
    if (!is_neutrino(pid) || inCtwEnergyDomain(energy)) return;
    if (!std::isfinite(weight) || weight <= 0.) throw std::runtime_error("invalid neutrino audit weight");
    if (uncovered_.count(history)) return; // no multiple counting across boundaries
    if (uncovered_.size() >= 200000) throw std::runtime_error("neutrino domain audit history limit exceeded");
    uncovered_.insert(history);
    auto& row = species_[static_cast<int>(get_PDG(pid))];
    ++row.count;row.weightedEnergyGeV += weight*energy;
    weightedEnergyGeV_ += weight*energy;
  }
  struct Species { std::uint64_t count{}; double weightedEnergyGeV{}; };
  std::size_t count() const noexcept { return uncovered_.size(); }
  double weightedEnergyGeV() const noexcept { return weightedEnergyGeV_; }
  bool strict() const noexcept { return strict_; }
  std::map<int,Species> const& species() const noexcept { return species_; }
 private:
  bool strict_;
  std::unordered_set<std::uint64_t> uncovered_;
  std::map<int,Species> species_;
  double weightedEnergyGeV_{};
};
} // namespace corsika::neutrino
