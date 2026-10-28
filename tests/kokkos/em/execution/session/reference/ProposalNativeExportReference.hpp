/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include "ProposalNativeReference.hpp"
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <array>
#include <string>
#include <vector>

namespace PROPOSAL {
  class Displacement;
  class Interaction;
  class Medium;
  namespace crosssection {
    class PhotoPairLPM;
    class BremsLPM;
  } // namespace crosssection
} // namespace PROPOSAL

// Independent fixed-production host oracle. Raw borrowed pointers below are test-only
// copies of the old API, not a second production interface.
namespace native_table_reference {
  using namespace corsika;
  using Sha256Digest = std::array<std::uint8_t, 32>;
  Sha256Digest sha256(std::uint8_t const* data, std::size_t size);
  inline Sha256Digest sha256(std::vector<std::uint8_t> const& data) {
    return sha256(data.data(), data.size());
  }
  std::string toHex(Sha256Digest const& digest);
  namespace proposal {
    /**
     * Non-owning, read-only access to one calculator already constructed by the
     * scalar InteractionModel.  The pointers are valid for the lifetime of that
     * model and are never exposed to device code.
     */
    struct NativeInteractionCalculatorView {
      Code projectile{Code::Unknown};
      std::size_t medium_hash{};
      PROPOSAL::Medium const* medium{};
      PROPOSAL::Interaction const* interaction{};
      PROPOSAL::crosssection::PhotoPairLPM const* photon_pair_lpm{};
      PROPOSAL::crosssection::BremsLPM const* brems_lpm{};
      HEPEnergyType stochastic_energy_cut{};
      double particle_mass_MeV{};
    };

    /** Read-only view of the scalar continuous-loss displacement calculator. */
    struct NativeContinuousCalculatorView {
      Code projectile{Code::Unknown};
      std::size_t medium_hash{};
      PROPOSAL::Medium const* medium{};
      PROPOSAL::Displacement const* displacement{};
      HEPEnergyType stochastic_energy_cut{};
      double particle_mass_MeV{};
    };

  } // namespace proposal
  struct ProposalNativeTableSet {
    std::uint32_t format_version{ProposalNativeTableFormatVersion};
    std::string proposal_version;
    std::string cubic_interpolation_version;
    NativePhysicalConstants constants{};
    double stochastic_cut_MeV{};
    Sha256Digest content_hash{};
    // Diagnostic state is deliberately excluded from content_hash: a cold
    // build and a warm PROPOSAL cache load must export identical physics.
    std::uint64_t proposal_cache_table_count{};
    std::uint64_t proposal_cache_hit_count{};
    bool proposal_cache_all_hit{};
    std::vector<NativeInteractionIdentity> interaction_identities;
    std::vector<NativeDndxColumn> dndx_columns;
    std::vector<NativeTotalRateColumn> total_rate_columns;
    // Index indirection that walks dndx_columns in Interaction::Rates order.
    // Entries remain grouped by PID and have one-to-one coverage.
    std::vector<std::uint32_t> selection_column_indices;
    std::vector<NativeDedxColumn> dedx_columns;
    // Index indirection that walks dedx_columns in scalar PROPOSAL's exact
    // cross-list/component summation order. Entries remain grouped by PID.
    std::vector<std::uint32_t> dedx_accumulation_indices;
    std::vector<NativeUtilityColumn> utility_columns;
    // All bicubic arrays use the same column-major offset and element count.
    std::vector<double> bicubic_values;
    std::vector<double> bicubic_derivative_energy;
    std::vector<double> bicubic_derivative_loss;
    std::vector<double> bicubic_mixed_derivative;
    // Sixteen row-major coefficients per cell.  Cells are ordered with the
    // energy interval varying fastest: energy + (rows - 1) * loss.
    std::vector<double> bicubic_polynomial_coefficients;
    // Every 1-D interval is represented exactly by node values and first
    // derivatives copied from the cardinal cubic B-spline.
    std::vector<double> cubic_values;
    std::vector<double> cubic_node_derivatives;
  };

  ProposalNativeDeviceView makeProposalNativeHostView(
      ProposalNativeTableSet const& table);
  std::size_t proposalNativeTableBytes(ProposalNativeTableSet const& table);
  void validateProposalNativeTable(ProposalNativeTableSet const& table);
  Sha256Digest calculateProposalNativeHash(ProposalNativeTableSet const& table);

  /**
   * Resolve the version-locked PROPOSAL process/PID/parametrization contract
   * used by the native exporter.  This host-only gate deliberately throws for
   * every tuple that is not explicitly supported; it is exposed so validation
   * tests can prove that an unknown parametrization cannot be uploaded as a
   * superficially valid rate column.
   */
  NativeKinematicModel classifyProposalNativeKinematicModel(
      std::int32_t process_id, std::string const& parametrization, std::int32_t pdg_id);

  ProposalNativeTableSet exportProposalNativeTables(
      std::vector<proposal::NativeInteractionCalculatorView> const& interactions,
      std::vector<proposal::NativeContinuousCalculatorView> const& continuous);
} // namespace native_table_reference
