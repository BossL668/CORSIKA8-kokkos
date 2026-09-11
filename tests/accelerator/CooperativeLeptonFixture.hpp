#pragma once
#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeAux.hpp>
struct CooperativeLeptonFixture {
  corsika::gpu::em::tables::ProposalNativeTableSet tables;
  corsika::gpu::em::tables::ProposalNativeAuxData auxiliary;
};
CooperativeLeptonFixture loadCooperativeLeptonFixture(char const* proposal_cache);

