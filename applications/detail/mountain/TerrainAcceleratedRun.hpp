// Application adapter: PROPOSAL exports, compatibility registry, selected-loss
// fallback and output routing. No CLI parsing, magnetic integration or kernels.
#pragma once
#include "TerrainEmRouter.hpp"
#include "TerrainEmPreparation.hpp"
#include "TerrainRadioConfig.hpp"
#include <corsika/modules/radio/interface/Output.hpp>
#include <corsika/accelerator/ScalarPhysicalConstants.hpp>
#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/accelerator/em/common/ProposalCpuFallbackHandler.hpp>
#include <corsika/accelerator/em/common/ProcessSequenceCompatibility.hpp>
#include <corsika/modules/transport/InterfaceEmSession.hpp>
#include <corsika/modules/transport/InterfaceRegionMap.hpp>
#include <corsika/geometry/terrain/TerrainMagneticAccuracy.hpp>
#include <chrono>
namespace corsika::applications::terrain {
struct EmExecutionOptions {
  int threads=1,device=0;
  std::size_t batch=64,memoryMiB=128;
  double energy=1.,emcut=.0005,mucut=.3,emthin=1.e-6;
  long seed=67101;
  bool force=false;
  std::filesystem::path auxCache;
  double maxWeight{},maxStepM{HUGE_VAL},windowS{HUGE_VAL};
  bool resident{true};
  std::size_t residentCapacity{65536};
  bool radio{false};
  std::filesystem::path radioOutput;
  std::size_t residentWavefront{},residentRecords{},radioBatch{8192};
};
template<class Tracking,class Sequence,class TerrainStack,class Neutrinos,
         class Hadrons,class Decay,class Interaction,class Losses,class Continuous,
         class Diagnostics,class Thinning,class Cut,class Save>
void runAcceleratedTerrain(Scene const& scene,corsika::terrain::Environment& env,
    Tracking& tracking,Sequence& sequence,OutputManager& manager,TerrainStack& stack,
    Neutrinos&,Hadrons&,Decay&,Interaction& em,Losses&,
    Continuous& continuous,Diagnostics& diagnostics,Thinning&,Cut&,
    HEPEnergyType productionCut,EmExecutionOptions const& options,YAML::Node& summary,Save save) {
  auto const& [threads,device,batch,memoryMiB,energy,emcut,mucut,emthin,seed,force,auxCache,maxWeight,maxStepM,windowS,resident,residentCapacity,radioEnabled,radioOutput,residentWavefront,residentRecords,radioBatch]=options;
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
  config.coverage=scene.coverage;
  summary["transport_boundary"]["type"]=scene.dem_coverage_enabled?"dem_coverage":"none";
  summary["transport_boundary"]["perimeter_edges"]=scene.coverage.edges.size();
  summary["transport_boundary"]["device_geometry_bytes"]=scene.coverage.bytes();
  summary["transport_boundary"]["geometry_uploads"]=scene.dem_coverage_enabled?1:0;
  config.resident_capacity=resident?residentCapacity:0;
  config.resident_wavefront_capacity=resident?residentWavefront:0;
  auto front=std::max(batch,config.resident_wavefront_capacity);
  config.resident_record_capacity=residentRecords?residentRecords:std::max(std::size_t{4096},16*front);
  summary["resident_execution"]["cpu_staging_batch"]=batch;
  summary["resident_execution"]["wavefront_capacity"]=front;
  summary["resident_execution"]["record_capacity"]=config.resident_record_capacity;
  summary["resident_execution"]["openmp_radio_batch"]=radioBatch;
  if(radioEnabled){
    config.radio=makeTerrainRadioConfig(scene,env);config.radio.propagation.coverage=scene.coverage;
    config.radio.host_track_buffer_capacity=radioBatch;
  }
  summary["em_scheduler"]=resident?"resident":"batched-reference";
  summary["backend"]="kokkos-proposal-native";
  summary["accelerator_constants"]["version"]=accelerator::scalar_constants::Version;
  summary["accelerator_constants"]["magnetic_rigidity_GeV_per_T_m"]=
      accelerator::scalar_constants::MagneticRigidityGeVPerTeslaMeter;
  summary["accelerator_constants"]["boundary_mass_convention"]="corsika_transport";
  summary["magnetic_trajectory"]["policy"]="quadratic-step-accuracy-cap-v1";
  summary["magnetic_trajectory"]["maximum_deflection_parameter_rad"]=corsika::terrain::MaximumQuadraticMagneticDeflection;
  summary["magnetic_trajectory"]["maximum_local_chord_excess_before_roundoff"]=8.e-10;
  summary["magnetic_trajectory"]["scope"]="shared terrain CPU and interface Kokkos cap; original air tracker unchanged";
  for(std::size_t i=0;i<banks.size();++i) {
    summary["material_tables"][i]["hash"]=gpu::em::tables::toHex(banks[i].table.content_hash);
    summary["material_tables"][i]["auxiliary_hash"]=gpu::em::tables::toHex(banks[i].auxiliary.content_hash);
    summary["material_tables"][i]["device_bytes"]=gpu::em::tables::proposalNativeTableBytes(banks[i].table);
  }
  save(summary);
  config.interface={0,1,0,1}; // Enclosing atmosphere / embedded material; not a physics rule.
  interfaces::InterfaceEmSession session(flat,banks,config);
  std::vector<corsika::radio::interface::Track> cpuRadio;
  struct ClearRadioCallback {Diagnostics& diagnostics;~ClearRadioCallback(){diagnostics.radio_track={};}} clearRadio{diagnostics};
  if(radioEnabled){
    diagnostics.radio_enabled=true;
    cpuRadio.reserve(batch);
    diagnostics.radio_track=[&](auto const& track){cpuRadio.push_back(track);if(cpuRadio.size()==batch){session.accumulateRadioCpu(cpuRadio);cpuRadio.clear();}};
  }
  using Fallback=gpu::em::ProposalCpuFallbackHandler<TerrainStack,Interaction,Sequence,corsika::terrain::Environment>;
  Fallback fallback(em,sequence,env,cs,seed,0,stochasticCut);
  auto* embedded=env.getUniverse()->getContainingNode(point(scene.yaml["geometry"]["rock_reference_enu_m"],cs));
  interfaces::InterfaceRegionMap regionOf(*embedded,config.interface);
  auto run=[&](auto& router) {
    using Router=std::decay_t<decltype(router)>;
    HybridCascade<Tracking,Sequence,OutputManager,TerrainStack,Router>
        cascade(env,tracking,sequence,manager,stack,router);
    summary["backend"]="kokkos-proposal-native";save(summary);
    if(force)cascade.forceInteraction();
    cascade.run();summary["accelerator"]=router.summary();
    if(radioEnabled)summary["accelerator"]["scope"]="two-sided EM transport with resident interface CoREAS and ZHS radio; one shared source queue";
  };
  auto transportStart=std::chrono::steady_clock::now();
  if(resident) {
    ResidentEmRouter<TerrainStack,Fallback,Diagnostics,decltype(regionOf)>
        router(session,cs,fallback,diagnostics,batch,regionOf);
    run(router);
  } else {
    EmRouter<TerrainStack,Fallback,Diagnostics,decltype(regionOf)>
        router(session,cs,fallback,diagnostics,batch,regionOf);
    run(router);
  }
  auto radioStart=std::chrono::steady_clock::now();
  summary["execution_timing"]["transport_loop_seconds"]=std::chrono::duration<double>(radioStart-transportStart).count();
  if(radioEnabled){
    if(!cpuRadio.empty())session.accumulateRadioCpu(cpuRadio);
    auto result=session.finishRadio();auto n=summary["radio_result"];
    auto write=[&](char const* name,corsika::radio::interface::Result const& value){
      auto directory=radioOutput/name;
      corsika::radio::interface::writeResult(value,directory.string());
      auto row=n[name];auto const& s=value.statistics;
      row["algorithm"]=corsika::radio::interface::algorithmName(value.config.algorithm);
      row["device_tracks"]=s.device_tracks;row["cpu_tracks"]=s.cpu_tracks;
      row["roundoff_limited_tracks"]=s.roundoff_limited_tracks;
      row["track_observer_pairs"]=s.track_observer_pairs;row["paths"]=s.paths;
      row["direct_paths"]=s.direct_paths;row["transmitted_paths"]=s.transmitted_paths;
      row["outside_coverage_paths"]=s.outside_coverage_paths;
      row["blocked_paths"]=s.blocked_paths;row["leaves"]=s.leaves;
      row["downloads"]=s.downloads;row["wavefronts"]=s.wavefronts;
      row["out_of_window"]=s.out_of_window;row["errors"]=s.errors;
      row["moment_arrays"]=value.regularized_moments.empty()?1:2;
      row["moment_bytes"]=(value.moments.size()+value.regularized_moments.size())*sizeof(double);
      if(value.config.algorithm==corsika::radio::interface::Algorithm::CoREAS){
        row["endpoint_contributions"]=s.endpoint_contributions;row["regularized_pairs"]=s.regularized_pairs;
        row["boundary_endpoints"]=s.boundary_endpoints;
      }
      row["directory"]=directory.string();row["complete"]=true;
    };
    write("CoREAS",result.coreas);write("ZHS",result.zhs);
    auto const& s=result.coreas.statistics;
    n["algorithms"].push_back("CoREAS");n["algorithms"].push_back("ZHS");
    n["execution_space"]=session.executionSpace();
    n["device_tracks"]=s.device_tracks;n["cpu_tracks"]=s.cpu_tracks;n["track_observer_pairs"]=s.track_observer_pairs;
    n["downloads"]=1;n["moment_arrays"]=3;n["wavefronts"]=s.wavefronts;
    n["radio_kernel_launches"]=s.wavefronts+result.zhs.statistics.wavefronts;
    n["device_bytes"]=result.device_bytes;
    n["out_of_window"]=s.out_of_window+result.zhs.statistics.out_of_window;
    n["errors"]=s.errors+result.zhs.statistics.errors;
    n["directory"]=radioOutput.string();n["complete"]=true;
  }
  summary["execution_timing"]["radio_flush_download_write_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-radioStart).count();
  summary["execution_timing"]["scope"]="transport includes in-loop radio and output callbacks; final phase includes any buffered radio sources, download, reconstruction and file output";
}
} // namespace corsika::applications::terrain
