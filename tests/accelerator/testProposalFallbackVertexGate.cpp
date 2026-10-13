/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#include <corsika/accelerator/em/common/ProposalFallbackAdapter.hpp>
#include <iostream>
#include <stdexcept>
#include <utility>

int main() {
  using namespace corsika::gpu::em;
  for(int pid : {22,11,-11,13,-13}) {
    for(auto reason : {ProposalFallbackReason::NativeSelectionReplay,
                      ProposalFallbackReason::InverseCdfUnavailable,
                      ProposalFallbackReason::LossEnergyOutOfRange,
                      ProposalFallbackReason::LossQuantileOutOfRange}) {
      EmInteractionRecord record{};
      record.particle.pid=pid;record.particle.energy_GeV=1.;record.particle.direction[2]=1.;
      record.process_id=pid==22?PhotoproductionProcessId:BremsProcessId;
      record.component_hash=7;record.loss_quantile=.5;record.process_uniform=.5;
      auto event=makeProcessFallbackEvent(record,reason);
      event.medium_hash=1;event.interaction_hash=2;
      if(hasResolvableProposalSelectedLoss(event))
        throw std::runtime_error("CPU completion accepted a pre-transport state");
      record.interaction_vertex_reached=1;
      event=makeProcessFallbackEvent(record,reason);
      event.medium_hash=1;event.interaction_hash=2;
      if(!hasResolvableProposalSelectedLoss(event))
        throw std::runtime_error("CPU completion rejected a valid vertex");
    }
  }
  for(auto identity : {std::pair{22,PhotoproductionProcessId},
                       std::pair{13,PhotonuclearProcessId},std::pair{-13,PhotonuclearProcessId},
                       std::pair{11,ElectronPairProcessId},std::pair{-11,ElectronPairProcessId}}) {
    EmInteractionRecord record{};
    record.particle.pid=identity.first;record.particle.energy_GeV=1.;
    record.particle.direction[2]=1.;record.process_id=identity.second;
    record.component_hash=7;record.process_uniform=.5;record.energy_fraction=.25;
    auto reason=ProposalFallbackReason::CpuOnlyProcess;
    auto event=makeProcessFallbackEvent(record,reason);
    event.medium_hash=1;event.interaction_hash=2;
    if(permitsSpecifiedProposalCpuFinalState(event))
      throw std::runtime_error("CPU final state accepted a pre-transport state");
    record.interaction_vertex_reached=1;
    event=makeProcessFallbackEvent(record,reason);
    event.medium_hash=1;event.interaction_hash=2;
    if(!permitsSpecifiedProposalCpuFinalState(event))
      throw std::runtime_error("CPU final state rejected a valid vertex");
  }
  for(int pid:{11,-11,13,-13})for(auto reason:{ProposalFallbackReason::EpairRejectionTrialsExhausted,
                                               ProposalFallbackReason::EpairRejectionEnvelopeExceeded}){
    EmInteractionRecord r{};r.particle.pid=pid;r.particle.energy_GeV=1000.;r.particle.direction[2]=1.;
    r.process_id=ElectronPairProcessId;r.energy_fraction=.1;r.component_hash=7;r.process_uniform=.5;
    for(unsigned reached:{0u,1u}){r.interaction_vertex_reached=reached;
      auto event=makeProcessFallbackEvent(r,reason);event.medium_hash=1;event.interaction_hash=2;
      if(permitsSpecifiedProposalCpuFinalState(event)!=bool(reached))
        throw std::runtime_error("bounded rho fallback vertex gate failed");
      if(proposalFallbackRequiresSelectedLoss(event))throw std::runtime_error("rho fallback resamples v");
    }
  }
  for(int pid:{11,-11,13,-13}) {
    if(gpuProcessCapability(pid,ElectronPairProcessId)!=GpuProcessCapability::CpuOnlyFinalState ||
       !proposalNativeSelectionRequiresReplay(pid,ElectronPairProcessId,.5))
      throw std::runtime_error("direct pair did not select CPU native completion");
  }
  EmInteractionRecord photomu{};
  photomu.particle.pid=22;photomu.particle.energy_GeV=1.;photomu.particle.direction[2]=1.;
  photomu.interaction_vertex_reached=1;photomu.process_id=PhotonMuonPairProcessId;
  photomu.component_hash=7;photomu.energy_fraction=1.;photomu.process_uniform=.5;
  auto obsolete=makeProcessFallbackEvent(photomu,ProposalFallbackReason::CpuOnlyProcess);
  obsolete.medium_hash=1;obsolete.interaction_hash=2;
  if(permitsSpecifiedProposalCpuFinalState(obsolete))
    throw std::runtime_error("device photomu channel still authorized as CPU-only");
  std::cout<<"CPU vertex gate: 20 selected-loss + 5 final-state pairs; photomu device gate PASS\n";
}
