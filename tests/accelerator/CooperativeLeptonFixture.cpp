/* Real CPU PROPOSAL calculators, isolated from NVCC. No resampled physics. */
#include "CooperativeLeptonFixture.hpp"
#include <corsika/accelerator/em/common/tables/ProposalNativeTableExporter.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>
#include <PROPOSAL/PROPOSAL.h>
#include <memory>
#include <iostream>
#include <vector>

CooperativeLeptonFixture loadCooperativeLeptonFixture(char const* proposal_cache) {
  using namespace corsika;
  using namespace corsika::gpu::em::tables;
  if (!std::filesystem::is_directory(proposal_cache))
    throw std::invalid_argument("existing PROPOSAL cache directory required");
  PROPOSAL::InterpolationSettings::TABLES_PATH = proposal_cache;
  struct Owner {
    PROPOSAL::Air medium;
    Code code;
    HEPEnergyType cut{0.4_MeV};
    PROPOSAL::crosssection_list_t cross;
    std::unique_ptr<PROPOSAL::Interaction> interaction;
    std::unique_ptr<PROPOSAL::Displacement> displacement;
    std::unique_ptr<PROPOSAL::crosssection::BremsLPM> brems;
    std::unique_ptr<PROPOSAL::crosssection::PhotoPairLPM> pair;
    Owner(Code c):code(c) {
      cross=proposal::make_cross_sections(c,medium,cut,true);
      interaction=PROPOSAL::make_interaction(cross,true,true);
      if(c!=Code::Photon) displacement=PROPOSAL::make_displacement(cross,true);
      if(c==Code::Photon)
        pair=std::make_unique<PROPOSAL::crosssection::PhotoPairLPM>(
            proposal::particle.at(c),medium,PROPOSAL::crosssection::PhotoPairKochMotz());
      if(c==Code::Electron || c==Code::Positron)
        brems=std::make_unique<PROPOSAL::crosssection::BremsLPM>(
            proposal::particle.at(c),medium,PROPOSAL::crosssection::BremsElectronScreening());
    }
  };
  std::vector<std::unique_ptr<Owner>> owners;
  std::vector<proposal::NativeInteractionCalculatorView> interactions;
  std::vector<proposal::NativeContinuousCalculatorView> continuous;
  for(auto code:{Code::Photon,Code::Electron,Code::Positron,Code::MuMinus,Code::MuPlus}) {
    owners.emplace_back(std::make_unique<Owner>(code));
    auto& o=*owners.back();
    auto mass=proposal::particle.at(code).mass;
    interactions.push_back({code,o.medium.GetHash(),&o.medium,o.interaction.get(),
                            o.pair.get(),o.brems.get(),o.cut,mass});
    if(o.displacement)
      continuous.push_back({code,o.medium.GetHash(),&o.medium,o.displacement.get(),o.cut,mass});
  }
  CooperativeLeptonFixture result;
  result.tables=exportProposalNativeTables(interactions,continuous);
  result.auxiliary=loadOrCreateProposalNativeAux(interactions);
  std::cout << "proposal_version=" << result.tables.proposal_version
            << " cubic_version=" << result.tables.cubic_interpolation_version
            << " native_sha256=" << toHex(result.tables.content_hash)
            << " auxiliary_sha256=" << toHex(result.auxiliary.content_hash)
            << '\n' << std::flush;
  return result;
}
