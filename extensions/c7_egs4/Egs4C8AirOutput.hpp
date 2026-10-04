#pragma once
#include "Egs4C8Air.hpp"

namespace c7_egs4::c8_adapter {
struct AirOutput {
  bool valid{},has_step{},has_radio{},has_observation{};
  ::corsika::gpu::em::EmStepRecord step;
  ::corsika::gpu::em::RadioTrackRecord radio;
  ::corsika::gpu::em::ObservationRecord observation;
  AirDiscard discard{AirDiscard::none};
  double discarded_energy_GeV{}; // unweighted, separate from atmospheric deposition
  Channel channel{Channel::invalid};
};
KOKKOS_INLINE_FUNCTION Point observerToGlobal(Point observer,Frame const& frame) {
  auto d=toGlobal(frame,{observer.x*.01,observer.y*.01,observer.z*.01});
  return {frame.origin_m.x+d.x,frame.origin_m.y+d.y,frame.origin_m.z+d.z};
}
// Only a freshly exported position is reused. Imported Cartesian data can
// differ by roundoff from the native observer-coordinate representation.
KOKKOS_INLINE_FUNCTION Point outputPosition(AirRecord const& p,AirEnvironment const& e) {
  if((e.overhead_mask&overhead::coordinate_reuse)&&p.exported_position)
    return {p.metadata.position_m[0],p.metadata.position_m[1],p.metadata.position_m[2]};
  return observerToGlobal(p.track.observer_cm,e.observer_frame);
}
// C8's existing onStep/onRadioTrack/onObservation record schema, with explicit
// native channel metadata outside the PROPOSAL process-ID namespace.
// C7 CoREAS AddTrack derives velocity from its pre/post Cartesian endpoints;
// do not feed the final MSCAT-rotated continuation direction into radio.
KOKKOS_INLINE_FUNCTION AirOutput makeAirOutput(AirRecord const& before,AirRecord const& after,
    AirOutcome const& outcome,AirEnvironment const& e) {
  AirOutput output;if(outcome.action==AirAction::error)return output;
  auto const& h=outcome.history;
  if(!finite(before.metadata.weight)||before.metadata.weight<0.||
     !finite(h.continuous_deposit_MeV)||h.continuous_deposit_MeV<0.||
     !finite(h.vertex_deposit_MeV)||h.vertex_deposit_MeV<0.||
     !finite(outcome.discarded_energy_MeV)||outcome.discarded_energy_MeV<0.)return output;
  output.discard=outcome.discard;output.discarded_energy_GeV=outcome.discarded_energy_MeV*.001;output.channel=h.channel;
  output.has_step=h.has_track||h.vertex_deposit_MeV>0.;
  if(output.has_step) {
    auto start=outputPosition(before,e);
    auto end=h.has_track?outputPosition(after,e):start;
    auto& s=output.step;s.history_id=before.metadata.history_id;s.step_id=before.metadata.step_id;s.pid=before.metadata.pid;
    s.process_id=0; // geometric/deposition record, not a fictitious PROPOSAL interaction
    s.start_position_m[0]=start.x;s.start_position_m[1]=start.y;s.start_position_m[2]=start.z;
    s.end_position_m[0]=end.x;s.end_position_m[1]=end.y;s.end_position_m[2]=end.z;
    s.start_time_s=before.track.particle.time_s;s.end_time_s=after.track.particle.time_s;
    s.start_energy_GeV=before.track.particle.energy_MeV*.001;
    s.end_energy_GeV=h.particle.energy_MeV*.001;
    s.deposited_energy_GeV=(h.continuous_deposit_MeV+h.vertex_deposit_MeV)*.001;
    // The sink treats this field as endpoint deposition, included in total.
    s.cut_deposited_energy_GeV=h.vertex_deposit_MeV*.001;s.weight=before.metadata.weight;
    if(!validPoint(start)||!validPoint(end)||!finite(s.end_time_s)||s.end_time_s<s.start_time_s)return output;
    Direction chord{end.x-start.x,end.y-start.y,end.z-start.z};double distance=::sqrt(dot(chord,chord));
    output.has_radio=h.has_track&&s.pid!=22&&distance>0.&&s.end_time_s>s.start_time_s;
    if(output.has_radio) {
      output.radio.step=s;double u[]={chord.x/distance,chord.y/distance,chord.z/distance};
      for(int j=0;j<3;++j)output.radio.start_direction[j]=output.radio.end_direction[j]=u[j];
    }
  }
  output.has_observation=outcome.action==AirAction::observed;
  if(output.has_observation)output.observation.particle=after.metadata;
  output.valid=true;return output;
}
// Targets the existing CorsikaOutputSink record API. test_c8_output also runs
// the actual C8 writers and CoREAS/ZHS through the host-facing Session.
// The caller separately handles child production and angular/escape loss.
template<class Sink>void streamAirOutput(AirOutput const& output,Sink& sink) {
  if(!output.valid)throw std::runtime_error("Invalid native EGS4 output");
  if(output.has_step)sink.onStep(output.step);
  if(output.has_radio)sink.onRadioTrack(output.radio);
  if(output.has_observation)sink.onObservation(output.observation);
}
} // namespace c7_egs4::c8_adapter
