/*
 * Read-only diagnostic used to compare the small physics snapshots embedded
 * in a legacy .c8emrt file with a proposal-native .c8emaux cache.  This is not
 * part of the shower executable and intentionally has no CUDA dependency.
 */

#include <corsika/gpu/em/BremsLpm.hpp>
#include <corsika/gpu/em/CudaPhotonPairFinalState.hpp>
#include <corsika/gpu/em/MoliereScattering.hpp>
#include <corsika/gpu/em/PhotonPairLpm.hpp>
#include <corsika/gpu/em/tables/ProposalNativeAux.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>

#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/crosssection/parametrization/PhotoPairProduction.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
  using namespace corsika::gpu::em;
  using namespace corsika::gpu::em::tables;

  struct AuxFileHeader {
    char magic[8]{};
    std::uint32_t format_version{};
    std::uint32_t payload_bytes{};
    Sha256Digest key_hash{};
    Sha256Digest content_hash{};
  };

  struct AuxPayloadPrefix {
    PhotonPairLpmSnapshot photon_pair_lpm{};
    BremsLpmSnapshot brems_lpm{};
    MoliereSnapshot electron_moliere{};
    MoliereSnapshot muon_moliere{};
    std::uint32_t has_muon_moliere{};
    std::uint32_t reserved[3]{};
  };

  static_assert(std::is_trivially_copyable_v<AuxFileHeader>);
  static_assert(std::is_trivially_copyable_v<AuxPayloadPrefix>);

  AuxPayloadPrefix readAux(std::filesystem::path const& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    AuxFileHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || std::memcmp(header.magic, "C8EMAUX", 7) != 0 ||
        header.format_version != ProposalNativeAuxFormatVersion ||
        header.payload_bytes != sizeof(AuxPayloadPrefix))
      throw std::runtime_error("unexpected c8emaux header or payload size");
    AuxPayloadPrefix payload{};
    input.read(reinterpret_cast<char*>(&payload), sizeof(payload));
    if (!input) throw std::runtime_error("truncated c8emaux payload");
    return payload;
  }

  double relative(double left, double right) {
    return std::abs(left - right) /
           std::max({std::abs(left), std::abs(right), 1.e-300});
  }

  template <typename Left, typename Right>
  void scalar(char const* name, Left left, Right right, double& maximum) {
    auto const difference = relative(static_cast<double>(left),
                                     static_cast<double>(right));
    maximum = std::max(maximum, difference);
    std::cout << name << " c8emrt=" << std::setprecision(17) << left
              << " native=" << right << " rel=" << difference << '\n';
  }
}

int main(int argc, char** argv) try {
  if (argc != 3) {
    std::cerr << "usage: inspect_c8emrt_native_aux TABLE.c8emrt AUX.c8emaux\n";
    return 2;
  }
  auto const table = corsika::gpu::em::tables::readRateTable(argv[1]);
  auto const aux = readAux(argv[2]);
  auto const table_photon =
      corsika::gpu::em::makePhotonPairLpmSnapshot(table.metadata.photon_pair_lpm);
  auto const table_brems =
      corsika::gpu::em::makeBremsLpmSnapshot(table.metadata.brems_lpm);
  auto const table_moliere =
      corsika::gpu::em::makeMoliereSnapshot(table.metadata.moliere);

  std::cout << "c8emrt metadata: proposal=" << table.metadata.proposal_version
            << " cut_MeV=" << std::setprecision(17)
            << table.metadata.energy_cut_MeV
            << " energy=[" << table.metadata.energy_min_MeV << ','
            << table.metadata.energy_max_MeV << "] requested_rate_tol="
            << table.metadata.requested_relative_tolerance
            << " measured_rate_error="
            << table.metadata.measured_max_relative_error
            << " requested_loss_tol="
            << table.metadata.requested_loss_relative_tolerance
            << " measured_loss_error="
            << table.metadata.measured_max_loss_relative_error << '\n';

  std::map<std::string, std::size_t> reference_modes;
  for (auto const& particle : table.particles)
    for (auto const& column : particle.columns)
      ++reference_modes[column.inverse_cdf.reference_mode];
  std::cout << "c8emrt inverse-CDF reference modes:\n";
  for (auto const& [mode, count] : reference_modes)
    std::cout << "  " << mode << ": " << count << '\n';
  for (auto const& particle : table.particles)
    for (auto const& column : particle.columns)
      if (!column.inverse_cdf.reference_mode.empty() &&
          column.inverse_cdf.reference_mode != "proposal_interpolated")
        std::cout << "  special column: pdg=" << particle.pdg_id
                  << " process=" << column.process_id
                  << " component=" << column.component_hash
                  << " name=" << column.process_name
                  << " target=" << column.target_name
                  << " mode=" << column.inverse_cdf.reference_mode << '\n';

  double maximum = 0.;
  scalar("photon.baseline_density", table_photon.baseline_mass_density_g_per_cm3,
         aux.photon_pair_lpm.baseline_mass_density_g_per_cm3, maximum);
  scalar("photon.molecular_density", table_photon.molecular_density_per_cm3,
         aux.photon_pair_lpm.molecular_density_per_cm3, maximum);
  scalar("photon.sum_charge", table_photon.sum_charge,
         aux.photon_pair_lpm.sum_charge, maximum);
  scalar("photon.e_lpm", table_photon.e_lpm_MeV,
         aux.photon_pair_lpm.e_lpm_MeV, maximum);
  scalar("brems.baseline_density", table_brems.baseline_mass_density_g_per_cm3,
         aux.brems_lpm.baseline_mass_density_g_per_cm3, maximum);
  scalar("brems.molecular_density", table_brems.molecular_density_per_cm3,
         aux.brems_lpm.molecular_density_per_cm3, maximum);
  scalar("brems.sum_charge", table_brems.sum_charge,
         aux.brems_lpm.sum_charge, maximum);
  scalar("brems.e_lpm", table_brems.e_lpm_MeV,
         aux.brems_lpm.e_lpm_MeV, maximum);
  scalar("moliere.mass", table_moliere.particle_mass_MeV,
         aux.electron_moliere.particle_mass_MeV, maximum);
  scalar("moliere.z_squared_over_a_average",
         table_moliere.z_squared_over_a_average,
         aux.electron_moliere.z_squared_over_a_average, maximum);
  scalar("moliere.inverse_weight_zz_sum",
         table_moliere.inverse_weight_zz_sum,
         aux.electron_moliere.inverse_weight_zz_sum, maximum);

  if (table_photon.component_count != aux.photon_pair_lpm.component_count ||
      table_brems.component_count != aux.brems_lpm.component_count ||
      table_moliere.component_count != aux.electron_moliere.component_count)
    throw std::runtime_error("component counts differ");
  for (std::uint32_t index = 0; index < table_photon.component_count; ++index) {
    if (table_photon.components[index].component_hash !=
            aux.photon_pair_lpm.components[index].component_hash ||
        table_brems.components[index].component_hash !=
            aux.brems_lpm.components[index].component_hash ||
        table_moliere.components[index].nuclear_charge !=
            aux.electron_moliere.components[index].nuclear_charge)
      throw std::runtime_error("component ordering or hashes differ");
    scalar("moliere.component.weight_zz",
           table_moliere.components[index].weight_zz,
           aux.electron_moliere.components[index].weight_zz, maximum);
    scalar("moliere.component.chi_0_squared",
           table_moliere.components[index].chi_0_squared_MeV2,
           aux.electron_moliere.components[index].chi_0_squared_MeV2,
           maximum);
    scalar("moliere.component.coulomb_correction",
           table_moliere.components[index].coulomb_correction,
           aux.electron_moliere.components[index].coulomb_correction,
           maximum);
  }
  auto compareArray = [&](char const* name, auto const& left,
                          auto const& right) {
    for (std::size_t index = 0; index < std::size(left); ++index)
      scalar((std::string(name) + "[" + std::to_string(index) + "]").c_str(),
             left[index], right[index], maximum);
  };
  compareArray("moliere.c1", table_moliere.c1, aux.electron_moliere.c1);
  compareArray("moliere.c2", table_moliere.c2, aux.electron_moliere.c2);
  compareArray("moliere.c2_large", table_moliere.c2_large,
               aux.electron_moliere.c2_large);
  compareArray("moliere.s2_large", table_moliere.s2_large,
               aux.electron_moliere.s2_large);
  compareArray("moliere.C1_large", table_moliere.C1_large,
               aux.electron_moliere.C1_large);

  // proposal-native has no artificial photon-pair split column and therefore
  // uses the analytic Koch--Motz rejection sampler at every energy.  Cover the
  // 10 GeV--1 TeV interval that the original unit test did not probe.
  PROPOSAL::crosssection::PhotoPairKochMotz pair_reference;
  PROPOSAL::GammaDef const photon;
  auto const air_components = PROPOSAL::Air().GetComponents();
  double maximum_pair_shape_error = 0.;
  for (auto const& component : air_components) {
    std::uint64_t component_hash = 0;
    for (std::uint32_t index = 0;
         index < aux.photon_pair_lpm.component_count; ++index)
      if (aux.photon_pair_lpm.components[index].nuclear_charge ==
          component.GetNucCharge())
        component_hash =
            aux.photon_pair_lpm.components[index].component_hash;
    if (component_hash == 0)
      throw std::runtime_error("cannot map high-energy Koch--Motz component");
    for (double const energy_MeV : {1.e4, 1.e5, 1.e6}) {
      auto const center_reference = pair_reference.DifferentialCrossSection(
          photon, component, energy_MeV, 0.5);
      auto const center_trial = photonPairFinalStateTrial(
          aux.photon_pair_lpm, component_hash, energy_MeV, 0.5, 0.5);
      if (!(center_reference > 0.) ||
          center_trial.status != PhotonPairFinalStateStatus::Success)
        throw std::runtime_error("high-energy Koch--Motz center query failed");
      auto const lower = 0.5109989461 / energy_MeV;
      auto const width = 1. - 2. * lower;
      for (double const coordinate :
           {0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999}) {
        auto const trial = photonPairFinalStateTrial(
            aux.photon_pair_lpm, component_hash, energy_MeV,
            coordinate, 0.5);
        if (trial.status != PhotonPairFinalStateStatus::Success)
          throw std::runtime_error("high-energy Koch--Motz query failed");
        auto const split = lower + width * coordinate;
        auto const expected = pair_reference.DifferentialCrossSection(
            photon, component, energy_MeV, split) / center_reference;
        auto const observed =
            trial.differential_weight / center_trial.differential_weight;
        maximum_pair_shape_error = std::max(
            maximum_pair_shape_error, relative(expected, observed));
      }
    }
  }
  std::cout << "high_energy_photon_pair_maximum_shape_error="
            << maximum_pair_shape_error << '\n';
  std::cout << "maximum_relative_difference=" << std::setprecision(17)
            << maximum << '\n';
  return 0;
} catch (std::exception const& error) {
  std::cerr << "inspect_c8emrt_native_aux failed: " << error.what() << '\n';
  return 1;
}
