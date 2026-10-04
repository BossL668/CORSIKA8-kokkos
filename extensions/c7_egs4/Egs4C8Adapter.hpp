#pragma once
#include "Egs4History.hpp"
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/Philox.hpp>

namespace c7_egs4::c8_adapter {
using C8Particle=::corsika::gpu::em::EmParticleState;
// Axes of the local EGS frame expressed in the caller's C8 frame. They must
// form a right-handed orthonormal basis. This conversion does NOT implement
// the C7 curved-atmosphere frame rebasing or its time correction.
struct Frame {
  Point origin_m;
  Direction x{1.,0.,0.},y{0.,1.,0.},z{0.,0.,1.};
};
KOKKOS_INLINE_FUNCTION double dot(Direction a,Direction b){return a.x*b.x+a.y*b.y+a.z*b.z;}
KOKKOS_INLINE_FUNCTION Direction cross(Direction a,Direction b) {
  return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
KOKKOS_INLINE_FUNCTION bool validFrame(Frame const& f) {
  return finite(f.origin_m.x)&&finite(f.origin_m.y)&&finite(f.origin_m.z)&&
    ::fabs(dot(f.x,f.x)-1.)<1.e-12&&::fabs(dot(f.y,f.y)-1.)<1.e-12&&::fabs(dot(f.z,f.z)-1.)<1.e-12&&
    ::fabs(dot(f.x,f.y))<1.e-12&&::fabs(dot(f.x,f.z))<1.e-12&&::fabs(dot(f.y,f.z))<1.e-12&&
    ::fabs(dot(cross(f.x,f.y),f.z)-1.)<1.e-12;
}
KOKKOS_INLINE_FUNCTION Direction toLocal(Frame const& f,Direction d){return {dot(d,f.x),dot(d,f.y),dot(d,f.z)};}
KOKKOS_INLINE_FUNCTION Direction toGlobal(Frame const& f,Direction d) {
  return {f.x.x*d.x+f.y.x*d.y+f.z.x*d.z,f.x.y*d.x+f.y.y*d.y+f.z.y*d.z,f.x.z*d.x+f.y.z*d.y+f.z.z*d.z};
}
// C8 carrier energy_GeV is TOTAL energy. Never subtract/rest-add a mass here.
// Optical clocks are private native history state and cannot be reconstructed
// from EmParticleState: keep the NativeRecord across every batch continuation.
KOKKOS_INLINE_FUNCTION bool importParticle(C8Particle const& p,Frame f,TrackState& result) {
  if(!validFrame(f)||!finite(p.weight)||p.weight<0.)return false;
  Direction displacement{p.position_m[0]-f.origin_m.x,p.position_m[1]-f.origin_m.y,p.position_m[2]-f.origin_m.z};
  auto pos=toLocal(f,displacement);TrackState s;
  s.pdg=p.pid;s.energy_MeV=p.energy_GeV*1000.;s.time_s=p.time_s;
  s.position_cm={pos.x*100.,pos.y*100.,pos.z*100.};
  s.direction=toLocal(f,{p.direction[0],p.direction[1],p.direction[2]});
  if(!validTrack(s))return false;
  result=s;return true;
}
KOKKOS_INLINE_FUNCTION bool exportParticle(TrackState state,Frame f,C8Particle& result) {
  if(!validFrame(f)||!validTrack(state))return false;
  auto position=toGlobal(f,{state.position_cm.x*.01,state.position_cm.y*.01,state.position_cm.z*.01});
  auto direction=toGlobal(f,state.direction);
  result.pid=state.pdg;result.energy_GeV=state.energy_MeV*.001;result.time_s=state.time_s;
  result.position_m[0]=position.x+f.origin_m.x;result.position_m[1]=position.y+f.origin_m.y;
  result.position_m[2]=position.z+f.origin_m.z;
  result.direction[0]=direction.x;result.direction[1]=direction.y;result.direction[2]=direction.z;
  return true; // metadata, weight and history IDs remain caller-owned
}
struct CounterStream {
  ::corsika::gpu::em::RandomNumberKey key;
  KOKKOS_INLINE_FUNCTION bool next(double& u) {
    if(key.draw_id==~std::uint64_t{0})return false;
    u=::corsika::gpu::em::uniformOpen01(key);++key.draw_id;return true;
  }
};
// Scope-local preparation: no larger resident record, no changed lanes/draw IDs.
// The caller must not change seed/history/step/process during this one history step.
struct PreparedCounterStream {
  CounterStream& stream;bool enabled;
  ::corsika::gpu::em::PhiloxCounter counter;std::uint64_t prefix{};
  KOKKOS_INLINE_FUNCTION PreparedCounterStream(CounterStream& s,bool on):stream(s),enabled(on) {
    if(!on)return;
    auto const& k=s.key;
    using ::corsika::gpu::em::detail::splitmix64;
    prefix=splitmix64(splitmix64(splitmix64(k.seed)^k.shower_id)^std::uint64_t(k.process_id));
    counter={{std::uint32_t(k.history_id),std::uint32_t(k.history_id>>32),
              std::uint32_t(k.step_id),std::uint32_t(k.step_id>>32)}};
  }
  KOKKOS_INLINE_FUNCTION bool next(double& u) {
    if(!enabled)return stream.next(u);
    if(stream.key.draw_id==~std::uint64_t{0})return false;
    auto m=::corsika::gpu::em::detail::splitmix64(prefix^stream.key.draw_id);
    ::corsika::gpu::em::PhiloxKey k{{std::uint32_t(m),std::uint32_t(m>>32)}};
    auto words=::corsika::gpu::em::philox4x32_10(counter,k);
    u=(double(words.words[0])+.5)*(1./4294967296.);++stream.key.draw_id;return true;
  }
};
inline constexpr std::uint32_t NativeEgs4RandomDomain=0x43474534U;
// Per-history counterpart of C7 stream 1 (PTRANS/meson decays), separate
// from electromagnetic stream 2. Never a process-global mutable RNG.
inline constexpr std::uint32_t NativeEgs4HadronRandomDomain=0x43474831U;
struct NativeRecord {
  C8Particle metadata;
  TrackState state;
  CounterStream random;
};
KOKKOS_INLINE_FUNCTION bool importRecord(C8Particle const& p,Frame frame,std::uint64_t seed,
                                        std::uint64_t shower_id,NativeRecord& result) {
  NativeRecord candidate;
  if(!importParticle(p,frame,candidate.state))return false;
  candidate.metadata=p;
  candidate.random.key={seed,shower_id,p.history_id,p.step_id,NativeEgs4RandomDomain,0};
  result=candidate;return true;
}
// Host-assigned, non-overlapping IDs are supplied explicitly, rather than an
// atomic allocation whose result (and RNG stream) depends on GPU scheduling.
KOKKOS_INLINE_FUNCTION bool childRecord(NativeRecord const& parent,TrackState child,Frame frame,
                                      std::uint64_t id,NativeRecord& result) {
  if(id==parent.metadata.history_id||parent.metadata.generation==~std::uint32_t{0})return false;
  auto p=parent.metadata;
  if(!exportParticle(child,frame,p))return false;
  p.parent_history_id=parent.metadata.history_id;p.history_id=id;
  p.generation+=1;p.step_id=0;
  NativeRecord candidate;
  if(!importRecord(p,frame,parent.random.key.seed,parent.random.key.shower_id,candidate))return false;
  candidate.state=child;candidate.state.clock={};candidate.state.photon_clock={};candidate.state.pending_channel=Channel::invalid;
  result=candidate;return true;
}
// Adapter-level one-step integration. The geometry object is a device-callable
// functor BoundaryLimit(TrackState const&, double proposed_cm). It must use
// the same local frame as region/field; there is no hidden geometry fallback.
template<class Geometry>
KOKKOS_INLINE_FUNCTION HistoryOutcome advanceRecord(ModelView model,NativeRecord& record,Frame frame,
    AirRegion region,MagneticField field,TransportSettings settings,Geometry geometry) {
  HistoryOutcome invalid;invalid.particle=record.state;
  if(!validFrame(frame)||record.metadata.step_id==~std::uint64_t{0})return invalid;
  auto p=planHistory(model,record.state,region,field,settings,record.random);
  BoundaryLimit limit;
  if(p.action==PlanAction::transport)limit=geometry(p.particle,p.proposed_cm);
  auto r=finishHistory(model,p,region,field,settings,limit,record.random);
  if(r.action==HistoryAction::error)return r; // RNG may be consumed: fatal, never retry with another model
  record.state=r.particle;
  if(!exportParticle(record.state,frame,record.metadata)){r.action=HistoryAction::error;r.error=HistoryError::invalid_input;return r;}
  if(r.has_track)++record.metadata.step_id;
  return r;
}
static_assert(std::is_trivially_copyable_v<NativeRecord>);
} // namespace c7_egs4::c8_adapter
