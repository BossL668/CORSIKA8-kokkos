#pragma once
#include "Egs4C8Rare.hpp"
#include "Egs4ResonanceDecay.hpp"

namespace c7_egs4::c8_adapter {
// Explicit local C7 decay axes. A global C8 direction alone is insufficient
// to reproduce ADDANG3's azimuth with the same random tape.
KOKKOS_INLINE_FUNCTION Frame decayFrame(AirRecord const& at_vertex,AirEnvironment const& env) {
  Frame local;
  if(!at_vertex.track.detector_local)
    local=radialFrame(at_vertex.track.observer_cm,env.curved.earth_radius_cm);
  Frame f;f.x=toGlobal(env.observer_frame,local.x);f.y=toGlobal(env.observer_frame,local.y);
  f.z=toGlobal(env.observer_frame,local.z);
  f.origin_m={at_vertex.metadata.position_m[0],at_vertex.metadata.position_m[1],at_vertex.metadata.position_m[2]};
  return f;
}
enum class DecayDestination {native_em,muon_transport,hadron_transport};
struct DecayProduct {
  DecayDestination destination{};
  C8Particle particle;
  double polarization_cosine{},polarization_azimuth{};
  CounterStream random,hadron_random;
};
struct ResonanceTransfer {
  bool valid{};
  InteractionStatus status{InteractionStatus::invalid_input};
  ResonanceBranch branch{ResonanceBranch::invalid};
  int count{};
  DecayProduct products[3];
  CounterStream parent_hadron_continuation;
};
// C8 carrier bridge for the validated native omega/phi kernels. The caller
// reserves globally unique IDs and supplies the recorded C7 vertex frame.
// No cuts or thinning here: photons explicitly return to the native EM queue;
// hadrons/muons require a host consumer. The transient parent is not an energy
// sink. Input request, RNG and ID allocator are untouched on every failure.
KOKKOS_INLINE_FUNCTION ResonanceTransfer resolveResonanceRequest(HostVertex const& request,
    Frame const& frame,ResonanceMasses const& masses,std::uint64_t first,std::uint64_t reserved) {
  ResonanceTransfer out;auto const& p=request.particle;
  if(request.kind!=HostVertexKind::c7_vector_meson_decay||(p.pid!=223&&p.pid!=333)||
     !validFrame(frame)||!finite(p.energy_GeV)||!finite(p.weight)||p.weight<0.||!finite(p.time_s)||
     p.history_id==0||p.generation==~std::uint32_t{0}||first==0||
     reserved>~std::uint64_t{0}-first||!reserved||
     (first<=p.history_id&&p.history_id<first+reserved))return out;
  for(int k=0;k<3;++k)if(!finite(p.position_m[k]))return out;
  Direction global{p.direction[0],p.direction[1],p.direction[2]};if(!validDirection(global))return out;
  auto rng=request.hadron_random;
  auto decay=sampleResonanceDecay({p.energy_GeV*1000.,p.pid,toLocal(frame,global)},masses,rng);
  out.status=decay.status;
  if(decay.status!=InteractionStatus::success)return out;
  if(std::uint64_t(decay.count)>reserved){out.status=InteractionStatus::invalid_input;return out;}
  for(int j=0;j<decay.count;++j) {
    auto& child=out.products[j];child.particle=p;auto& q=child.particle;auto generated=decay.particle[j];
    q.pid=generated.pdg;q.energy_GeV=generated.energy_MeV*.001;
    auto direction=toGlobal(frame,generated.direction);if(!validDirection(direction))return {};
    q.direction[0]=direction.x;q.direction[1]=direction.y;q.direction[2]=direction.z;
    q.history_id=first+j;q.parent_history_id=p.history_id;q.generation=p.generation+1;q.step_id=0;
    child.destination=q.pid==22?DecayDestination::native_em:
      q.pid==13||q.pid==-13?DecayDestination::muon_transport:DecayDestination::hadron_transport;
    child.polarization_cosine=decay.polarization_cosine[j];child.polarization_azimuth=decay.polarization_azimuth[j];
    child.random.key={request.random.key.seed,request.random.key.shower_id,q.history_id,0,NativeEgs4RandomDomain,0};
    child.hadron_random.key={rng.key.seed,rng.key.shower_id,q.history_id,0,NativeEgs4HadronRandomDomain,0};
  }
  out.parent_hadron_continuation=rng;out.branch=decay.branch;out.count=decay.count;out.valid=true;return out;
}
} // namespace c7_egs4::c8_adapter
