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
                       std::pair{22,PhotonMuonPairProcessId},
                       std::pair{13,BremsProcessId},std::pair{-13,ElectronPairProcessId},
                       std::pair{11,ElectronPairProcessId},std::pair{-11,ElectronPairProcessId}}) {
    EmInteractionRecord record{};
    record.particle.pid=identity.first;record.particle.energy_GeV=1.;
    record.particle.direction[2]=1.;record.process_id=identity.second;
    record.component_hash=7;record.process_uniform=.5;record.energy_fraction=.25;
    auto reason=isElectronOrPositronPid(identity.first)
                    ? ProposalFallbackReason::EpairRejectionEnvelopeExceeded
                    : ProposalFallbackReason::CpuOnlyProcess;
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
  std::cout<<"CPU vertex gate: 20 selected-loss + 6 final-state pre/at-vertex pairs PASS\n";
}
