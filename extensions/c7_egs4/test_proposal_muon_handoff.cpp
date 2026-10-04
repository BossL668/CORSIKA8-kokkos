#include "Egs4ProposalMuonBackend.hpp"
#include <cstring>
#include <iostream>
using namespace corsika::gpu::em;
using namespace corsika::accelerator::em;
using namespace c7_egs4::application;
void require(bool ok){if(!ok)throw std::runtime_error("muon handoff contract failed");}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(std::exception const&){failed=true;}require(failed);}
struct Fixture: IAcceleratedEmBackend {
  ResidentLeptonCascadeResult result;
  AcceleratedEmStatistics stats;
  std::size_t pending{};std::uint64_t first_seen{},limit_seen{};std::size_t waves_seen{};
  void beginShower(AcceleratedEmShowerConfig const&)override{}
  bool canTransport(EmParticleState const&)const override{return true;}
  bool hasProposalTable()const override{return true;}
  std::size_t minimumBatchSize()const override{return 1;}
  std::size_t maximumResidentPhotonBatchSize()const override{return 32;}
  std::size_t maximumResidentLeptonBatchSize()const override{return 32;}
  std::size_t maximumResidentInputBatchSize()const override{return 32;}
  std::size_t pendingPhotonCount()const noexcept override{return pending;}
  std::size_t pendingLeptonCount()const noexcept override{return 0;}
  ResidentPhotonCascadeResult runPhotonWavefront(std::vector<EmParticleState> const&,std::uint64_t,std::size_t,std::size_t)override{
    throw std::logic_error("must not call photon transport");
  }
  ResidentLeptonCascadeResult runLeptonWavefront(std::vector<EmParticleState> const&,std::uint64_t first,
      std::size_t waves,std::uint64_t limit,std::size_t)override{
    first_seen=first;waves_seen=waves;limit_seen=limit;return result;
  }
  BackendCapabilities capabilities()const override{BackendCapabilities c;c.muon_transport=true;return c;}
  AcceleratedEmStatistics const& statistics()const override{return stats;}
  bool gpuProfileEnabled()const noexcept override{return false;}
  GpuProfileResult downloadProfile()override{return {};}
  bool gpuRadioEnabled()const noexcept override{return false;}
  corsika::gpu::radio::GpuRadioWaveforms downloadRadioWaveforms()override{return {};}
  std::optional<GpuFirstInteractionSnapshot> downloadFirstInteractionSnapshot()override{return {};}
};
int main(){try{
  Fixture backend;ProposalMuonBackend adapter(backend);
  EmParticleState mu{},electron{},positron{},photon{};
  mu.pid=13;mu.history_id=7;mu.step_id=42;mu.energy_GeV=10.;mu.weight=2.;
  electron=mu;electron.pid=11;electron.history_id=100;electron.parent_history_id=7;electron.step_id=0;
  electron.energy_GeV=.01;electron.direction[2]=-1.;electron.medium_id=3;
  positron=electron;positron.pid=-11;positron.history_id=101;
  photon=electron;photon.pid=22;photon.history_id=102;
  require(adapter.canTransport(mu)&&!adapter.canTransport(electron)&&!adapter.canTransport(photon));
  backend.result.input_particles=1;backend.result.wavefronts=1;backend.result.transport_records=1;
  backend.result.remaining_leptons={mu,electron};backend.result.cpu_spill_particles={positron};
  backend.result.generated_photons={photon};backend.result.secondary_history_ids_used=3;
  backend.result.decay_candidates={mu};backend.result.fallback_events.resize(1);
  auto output=adapter.runLeptonWavefront({mu},100,1024,10000,1);
  require(backend.waves_seen==1&&backend.first_seen==100&&backend.limit_seen==10000);
  require(output.remaining_leptons.size()==1&&output.generated_photons.empty()&&output.cpu_spill_particles.empty());
  require(output.decay_candidates.size()==1&&output.fallback_events.size()==1&&output.secondary_history_ids_used==3);
  require(!output.completed&&adapter.muonSteps()==1&&adapter.emTransfers()==3);
  auto em=adapter.takeElectromagnetic();require(em.size()==3&&!adapter.hasElectromagnetic());
  require(std::memcmp(&em[0],&photon,sizeof photon)==0&&std::memcmp(&em[1],&electron,sizeof electron)==0&&
    std::memcmp(&em[2],&positron,sizeof positron)==0);
  rejects([&]{adapter.runLeptonWavefront({electron},100,2,10000,1);});
  rejects([&]{adapter.runPhotonWavefront({photon},100,2,1);});
  backend.pending=1;rejects([&]{adapter.runLeptonWavefront({mu},100,2,10000,1);});backend.pending=0;
  backend.result.remaining_leptons.back().step_id=1;
  rejects([&]{adapter.runLeptonWavefront({mu},100,2,10000,1);});
  std::cout<<"PASS: one-wave PROPOSAL muons, exact EGS4 daughter metadata, no EM leakage, fallback/decay retained\n";
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
