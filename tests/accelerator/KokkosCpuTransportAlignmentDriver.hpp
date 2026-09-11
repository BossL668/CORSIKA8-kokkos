/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/accelerator/em/detail/LeptonContinuousStep.hpp>
#include <corsika/accelerator/em/common/ExternalTransportBoundary.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>

#include <vector>

namespace corsika::accelerator::em::testing {
  struct CpuTransportAlignmentInput {
    gpu::em::EmInteractionRecord interaction{};
    gpu::em::EnvironmentSnapshot environment{};
    double em_cut_MeV{0.5};
    double muon_cut_MeV{300.};
    bool propagate{};
    gpu::em::ExternalTransportBoundary external{};
  };
  struct CpuTransportAlignmentOutput {
    detail::LeptonContinuousPreparation preparation{};
    gpu::em::LeptonTransportRecord transport{};
    gpu::em::ProposalFallbackEvent fallback{};
    gpu::em::EmInteractionStatus selection_status{};
    std::uint32_t transport_status{};
  };

  std::vector<CpuTransportAlignmentOutput> runCpuTransportAlignmentOnDevice(
      gpu::em::tables::ProposalNativeTableSet const&,
      std::vector<CpuTransportAlignmentInput> const&);

  struct CpuSecondaryAlignmentInput {
    gpu::em::EmInteractionRecord interaction{};
    double split{};
    std::uint32_t keep_mask{3};
  };
  struct CpuSecondaryAlignmentOutput {
    gpu::em::EmParticleState children[3]{};
    std::uint32_t count{};
    std::uint32_t error{};
    double weighted_mass_correction_GeV{};
  };
  std::vector<CpuSecondaryAlignmentOutput> runCpuSecondaryAlignmentOnDevice(
      std::vector<CpuSecondaryAlignmentInput> const&);

  struct CpuRandomDomainAlignmentOutput {
    double first_moliere_uniform{};
    double proposal_selection_uniform{};
    double legacy_proposal_selection_uniform{};
    bool selection_observed{};
  };
  std::vector<CpuRandomDomainAlignmentOutput> runCpuRandomDomainAlignmentOnDevice(
      gpu::em::tables::ProposalNativeTableSet const&, std::size_t samples);
}
