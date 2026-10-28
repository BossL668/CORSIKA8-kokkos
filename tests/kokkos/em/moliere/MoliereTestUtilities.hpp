/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "MoliereReference.hpp"
#include <corsika/kokkos/em/tables/MoliereInterpolation.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace corsika::kokkos::test {
  /** Exact same-ABI adapter: every byte is copied only after layout checks. */
  inline moliere_reference::MoliereSnapshot moliere_reference_snapshot(
      MoliereSnapshot const& snapshot) {
    using old_type = moliere_reference::MoliereSnapshot;
    static_assert(sizeof(old_type) == sizeof(MoliereSnapshot));
    static_assert(alignof(old_type) == alignof(MoliereSnapshot));
#define CORSIKA_CHECK_MOLIERE_MEMBER(OLD, NEW) \
  static_assert(offsetof(old_type, OLD) == offsetof(MoliereSnapshot, NEW))
    CORSIKA_CHECK_MOLIERE_MEMBER(particle_mass_MeV, particleMassMeV_);
    CORSIKA_CHECK_MOLIERE_MEMBER(electron_mass_MeV, electronMassMeV_);
    CORSIKA_CHECK_MOLIERE_MEMBER(fine_structure_constant, fineStructureConstant_);
    CORSIKA_CHECK_MOLIERE_MEMBER(avogadro_per_mol, avogadroPerMol_);
    CORSIKA_CHECK_MOLIERE_MEMBER(hbar_MeV_s, hbarMeVS_);
    CORSIKA_CHECK_MOLIERE_MEMBER(speed_of_light_cm_per_s, speedOfLightCmPerS_);
    CORSIKA_CHECK_MOLIERE_MEMBER(z_squared_over_a_average, zSquaredOverAAverage_);
    CORSIKA_CHECK_MOLIERE_MEMBER(inverse_weight_zz_sum, inverseWeightZzSum_);
    CORSIKA_CHECK_MOLIERE_MEMBER(euler_mascheroni, eulerMascheroni_);
    CORSIKA_CHECK_MOLIERE_MEMBER(component_count, componentCount_);
    CORSIKA_CHECK_MOLIERE_MEMBER(maximum_weight_index, maximumWeightIndex_);
    CORSIKA_CHECK_MOLIERE_MEMBER(components, components_);
    CORSIKA_CHECK_MOLIERE_MEMBER(c1, c1_);
    CORSIKA_CHECK_MOLIERE_MEMBER(c2, c2_);
    CORSIKA_CHECK_MOLIERE_MEMBER(c2_large, c2Large_);
    CORSIKA_CHECK_MOLIERE_MEMBER(s2_large, s2Large_);
    CORSIKA_CHECK_MOLIERE_MEMBER(C1_large, c1IntegralLarge_);
#undef CORSIKA_CHECK_MOLIERE_MEMBER
    using old_component_type = moliere_reference::MoliereComponentSnapshot;
    static_assert(sizeof(old_component_type) == sizeof(MoliereComponentSnapshot));
    static_assert(offsetof(old_component_type, nuclear_charge) ==
                  offsetof(MoliereComponentSnapshot, nuclearCharge_));
    static_assert(offsetof(old_component_type, weight_zz) ==
                  offsetof(MoliereComponentSnapshot, weightZz_));
    static_assert(offsetof(old_component_type, chi_0_squared_MeV2) ==
                  offsetof(MoliereComponentSnapshot, chi0SquaredMeV2_));
    static_assert(offsetof(old_component_type, coulomb_correction) ==
                  offsetof(MoliereComponentSnapshot, coulombCorrection_));
    old_type result{};
    std::memcpy(static_cast<void*>(&result), &snapshot, sizeof(result));
    return result;
  }

  inline moliere_reference::MoliereInterpolationTable moliere_reference_table(
      MoliereInterpolationTable const& table) {
    using old_type = moliere_reference::MoliereInterpolationTable;
    static_assert(sizeof(old_type) == sizeof(MoliereInterpolationTable));
    static_assert(alignof(old_type) == alignof(MoliereInterpolationTable));
    static_assert(offsetof(old_type, polynomials) ==
                  offsetof(MoliereInterpolationTable, polynomials_));
    static_assert(offsetof(old_type, initial_guess_delta) ==
                  offsetof(MoliereInterpolationTable, initialGuessDelta_));
    static_assert(sizeof(moliere_reference::MoliereCubicPolynomial) ==
                  sizeof(MoliereCubicPolynomial));
    static_assert(offsetof(moliere_reference::MoliereCubicPolynomial, c) ==
                  offsetof(MoliereCubicPolynomial, c_));
    old_type result{};
    std::memcpy(static_cast<void*>(&result), &table, sizeof(result));
    return result;
  }

  /** Synthetic valid state for cache contracts; not a physical Air surrogate. */
  inline MoliereSnapshot small_moliere_snapshot() {
    MoliereSnapshot result{};
    result.particleMassMeV_ = 0.511;
    result.electronMassMeV_ = 0.511;
    result.fineStructureConstant_ = 1. / 137.;
    result.avogadroPerMol_ = 6.02214076e23;
    result.hbarMeVS_ = 6.582119569e-22;
    result.speedOfLightCmPerS_ = 2.99792458e10;
    result.zSquaredOverAAverage_ = 3.5;
    result.inverseWeightZzSum_ = 1.;
    result.eulerMascheroni_ = 0.5772156649015329;
    result.componentCount_ = 1;
    result.components_[0] = {7., 1., 1.e-4, 0.01};
    result.c1_[0] = 0.01;
    result.c2_[0] = 0.001;
    result.c1IntegralLarge_[0] = -0.01;
    return result;
  }

  inline std::filesystem::path moliere_test_directory(char const* label) {
    auto const* base = std::getenv("CORSIKA_MOLIERE_TEST_CACHE");
    if (!base || !*base) throw std::runtime_error("Missing Moliere test cache directory");
    auto const directory =
        std::filesystem::path(base) /
        (std::string(label) + "-" + std::to_string(::getpid()) + "-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    return directory;
  }

  inline std::vector<char> moliere_read_bytes(std::filesystem::path const& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read Moliere test evidence");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }

  inline void moliere_write_bytes(std::filesystem::path const& path,
                                  std::vector<char> const& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("Cannot write Moliere test evidence");
  }
} // namespace corsika::kokkos::test
