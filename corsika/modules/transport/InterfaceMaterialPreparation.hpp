/* Host-only bridge from actual CORSIKA calculators to immutable interface banks.
 * No scene/YAML, geographic origin, mesh shape or air/rock ordering is assumed. */
#pragma once
#include <corsika/modules/transport/InterfaceEmSession.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTableExporter.hpp>
#include <corsika/accelerator/em/ProposalNativeRequirements.hpp>
#include <algorithm>

namespace corsika::interfaces {
inline EmMaterial prepareEmMaterial(
    std::vector<proposal::NativeInteractionCalculatorView> interactions,
    std::vector<proposal::NativeContinuousCalculatorView> continuous,
    gpu::em::EnvironmentSnapshot const& environment,
    accelerator::em::AcceleratedPhysicsRequirements const& requirements,
    std::filesystem::path const& auxiliaryCache) {
  if(interactions.empty()||continuous.empty())
    throw std::invalid_argument("interface material requires actual calculators");
  auto hash=interactions.front().medium_hash;
  for(auto const& view:interactions)if(view.medium_hash!=hash)
    throw std::invalid_argument("mixed interaction materials in one interface bank");
  for(auto const& view:continuous)if(view.medium_hash!=hash)
    throw std::invalid_argument("interface continuous/interaction material mismatch");
  // Keep the original full admission check, including CPU-routed muons.
  {
    auto full=gpu::em::tables::exportProposalNativeTables(interactions,continuous);
    accelerator::em::validateProposalNativeRequirements(full,requirements);
  }
  auto emPid=[](Code p){return p==Code::Photon||p==Code::Electron||p==Code::Positron;};
  interactions.erase(std::remove_if(interactions.begin(),interactions.end(),
      [&](auto const& v){return !emPid(v.projectile);}),interactions.end());
  continuous.erase(std::remove_if(continuous.begin(),continuous.end(),
      [&](auto const& v){return !emPid(v.projectile);}),continuous.end());
  EmMaterial result;
  result.table=gpu::em::tables::exportProposalNativeTables(interactions,continuous);
  result.auxiliary=gpu::em::tables::loadOrCreateProposalNativeAux(interactions,auxiliaryCache);
  result.environment=environment;
  return result;
}
} // namespace corsika::interfaces
