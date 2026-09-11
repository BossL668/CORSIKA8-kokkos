// Application adapter: PROPOSAL exports, compatibility registry, selected-loss
// fallback and output routing. No CLI parsing, magnetic integration or kernels.
#pragma once
#include "TerrainEmRouter.hpp"
#include "TerrainEmPreparation.hpp"
#include <corsika/accelerator/ScalarPhysicalConstants.hpp>
#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/accelerator/em/common/ProposalCpuFallbackHandler.hpp>
#include <corsika/accelerator/em/common/ProcessSequenceCompatibility.hpp>
#include <corsika/modules/transport/InterfaceEmSession.hpp>
#include <corsika/modules/transport/InterfaceRegionMap.hpp>
namespace corsika::applications::terrain {
struct EmExecutionOptions {
  int threads=1,device=0;
  std::size_t batch=64,memoryMiB=128;
  double energy=1.,emcut=.0005,mucut=.3,emthin=1.e-6;
  long seed=67101;
  bool force=false;
  std::filesystem::path auxCache;
  double maxWeight{},maxStepM{HUGE_VAL},windowS{HUGE_VAL};
};
template<class Tracking,class Sequence,class TerrainStack,class Neutrinos,
         class Hadrons,class Decay,class Interaction,class Losses,class Continuous,
         class Diagnostics,class Thinning,class Cut,class Save>
void runAcceleratedTerrain(Scene const& scene,corsika::terrain::Environment& env,
    Tracking& tracking,Sequence& sequence,OutputManager& manager,TerrainStack& stack,
    Neutrinos&,Hadrons&,Decay&,Interaction& em,Losses&,
    Continuous& continuous,Diagnostics& diagnostics,Thinning&,Cut&,
    HEPEnergyType productionCut,EmExecutionOptions const& options,YAML::Node& summary,Save save) {
  auto const& [threads,device,batch,memoryMiB,energy,emcut,mucut,emthin,seed,force,auxCache,maxWeight,maxStepM,windowS]=options;
  auto cs=env.getCoordinateSystem();
  using Policy=gpu::em::GpuEmStepProcessPolicy;
  using Registry=gpu::em::GpuEmStepProcessRegistry<
      gpu::em::GpuEmStepProcessRegistration<Neutrinos,Policy::InapplicableToRoutedEm>,
      gpu::em::GpuEmStepProcessRegistration<Hadrons,Policy::InapplicableToRoutedEm>,
      gpu::em::GpuEmStepProcessRegistration<Decay,Policy::DeferredToCpu>,
      gpu::em::GpuEmStepProcessRegistration<Interaction,Policy::ReplacedOnDevice>,
      gpu::em::GpuEmStepProcessRegistration<Losses,Policy::ReplacedOnDevice>,
      gpu::em::GpuEmStepProcessRegistration<Diagnostics,Policy::ReplayedFromDeviceRecord>,
      gpu::em::GpuEmStepProcessRegistration<Thinning,Policy::ReplacedOnDevice>,
      gpu::em::GpuEmStepProcessRegistration<Cut,Policy::ReplacedOnDevice>>;
  Registry::template validateOrThrow<Sequence>();
  auto const& flat=tracking.flatTerrain();
  auto stochasticCut=proposal::optimized_proposal_energy_cut(productionCut);
  auto banks=prepareEmMaterials(scene,env,flat,em,continuous,
      {energy*1000.,emcut*1000.,mucut*1000.,stochasticCut/1_MeV},auxCache);
  corsika::terrain::TerrainEmConfig config;config.threads=threads;config.device=device;
  config.batch_size=batch;config.maximum_device_bytes=memoryMiB*1024u*1024u;config.emcut_MeV=emcut*1000.;config.seed=seed;
  config.thinning.enabled=emthin>0.;config.thinning.threshold_GeV=emthin*energy;
  config.thinning.maximum_weight=maxWeight;config.thinning.erase_zero_weight=1;
  config.maximum_step_m=maxStepM;config.transport_window_s=windowS;
  config.energy_ledger=diagnostics.ledger.enabled;
  summary["backend"]="kokkos-proposal-native";
  summary["accelerator_constants"]["version"]=accelerator::scalar_constants::Version;
  summary["accelerator_constants"]["magnetic_rigidity_GeV_per_T_m"]=
      accelerator::scalar_constants::MagneticRigidityGeVPerTeslaMeter;
  summary["accelerator_constants"]["boundary_mass_convention"]="corsika_transport";
  for(std::size_t i=0;i<banks.size();++i) {
    summary["material_tables"][i]["hash"]=gpu::em::tables::toHex(banks[i].table.content_hash);
    summary["material_tables"][i]["auxiliary_hash"]=gpu::em::tables::toHex(banks[i].auxiliary.content_hash);
    summary["material_tables"][i]["device_bytes"]=gpu::em::tables::proposalNativeTableBytes(banks[i].table);
  }
  save(summary);
  config.interface={0,1,0,1}; // Enclosing atmosphere / embedded material; not a physics rule.
  interfaces::InterfaceEmSession session(flat,banks,config);
  using Fallback=gpu::em::ProposalCpuFallbackHandler<TerrainStack,Interaction,Sequence,corsika::terrain::Environment>;
  Fallback fallback(em,sequence,env,cs,seed,0,stochasticCut);
  auto* embedded=env.getUniverse()->getContainingNode(point(scene.yaml["geometry"]["rock_reference_enu_m"],cs));
  interfaces::InterfaceRegionMap regionOf(*embedded,config.interface);
  EmRouter<TerrainStack,Fallback,Diagnostics,decltype(regionOf)> router(session,cs,fallback,diagnostics,batch,regionOf);
  HybridCascade<Tracking,Sequence,OutputManager,TerrainStack,decltype(router)>
      cascade(env,tracking,sequence,manager,stack,router);
  summary["backend"]="kokkos-proposal-native";save(summary);
  if(force)cascade.forceInteraction();
  cascade.run();summary["accelerator"]=router.summary();
}
} // namespace corsika::applications::terrain
