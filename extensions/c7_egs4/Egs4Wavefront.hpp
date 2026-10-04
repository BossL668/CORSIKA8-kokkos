#pragma once
#include "Egs4C8Adapter.hpp"
#include <limits>
#include <set>

namespace c7_egs4::c8_adapter {
// Isolated, single-material native C++ wavefront. Tables, optical clocks and
// RNG state stay resident. Only counts are returned to the host each wave.
// This is not the production C8 backend: geometry is supplied by the caller,
// and observations/rare host channels are explicitly returned, never hidden.
template<class Exec> class LocalWavefront {
public:
  using Records=Kokkos::View<NativeRecord*,typename Exec::memory_space>;
  using Outcomes=Kokkos::View<HistoryOutcome*,typename Exec::memory_space>;
  struct Batch {
    Records before; // immutable-by-contract input for track/deposition callbacks
    Records records; // particles AFTER the step; includes any handoff requests
    Outcomes outcomes; // caller must consume deposits/tracks/handoffs
    int next_count{},created_count{};
  };
  // Public kernel types are required by nvcc's generated launch stubs. State
  // and orchestration remain private; callers use load/advance/resume only.
  using Offsets=Kokkos::View<int*,typename Exec::memory_space>;
  template<class Geometry>struct AdvanceKernel {
    ModelView model;Records records;Outcomes outcomes;Frame frame;AirRegion region;
    MagneticField field;TransportSettings settings;Geometry geometry;
    KOKKOS_INLINE_FUNCTION void operator()(int i,int& count)const {
      auto r=advanceRecord(model,records(i),frame,region,field,settings,geometry);
      outcomes(i)=r;count+=r.action==HistoryAction::error;
    }
  };
  struct OffsetKernel {
    Outcomes outcomes;Offsets offsets;bool children_only;
    KOKKOS_INLINE_FUNCTION void operator()(int i,int& sum,bool final)const {
      if(final)offsets(i)=sum;
      auto r=outcomes(i);sum+=r.action==HistoryAction::replaced?r.child_count:
        (!children_only&&r.action==HistoryAction::active?1:0);
    }
  };
  struct EmitKernel {
    Records records,next;Outcomes outcomes;Offsets offsets,child_offsets;Frame frame;std::uint64_t first_id;
    KOKKOS_INLINE_FUNCTION void operator()(int i,int& count)const {
      auto r=outcomes(i);int offset=offsets(i);
      if(r.action==HistoryAction::active)next(offset)=records(i);
      else if(r.action==HistoryAction::replaced)for(int j=0;j<r.child_count;++j)
        count+=!childRecord(records(i),r.children[j],frame,first_id+child_offsets(i)+j,next(offset+j));
      // All boundary/rare-channel requests remain in Batch for the caller.
    }
  };
private:
  DeviceModel<Exec> model_;
  Frame frame_;
  AirRegion region_;
  MagneticField field_;
  TransportSettings settings_;
  std::int32_t medium_;
  int capacity_;
  Records active_;
  std::uint64_t next_id_{};
  std::uint64_t seed_{},shower_id_{};
public:
  LocalWavefront(Tables const& tables,ChannelThresholds thresholds,Frame frame,
      AirRegion region,MagneticField field,TransportSettings settings,
      std::int32_t medium,int capacity):model_(tables,thresholds),frame_(frame),region_(region),
      field_(field),settings_(settings),medium_(medium),capacity_(capacity) {
    if(capacity<1||capacity>std::numeric_limits<int>::max()/2||!validFrame(frame)||
       !validRegion(region)||!validField(field)||!validSettings(model_.view(),settings))
      throw std::runtime_error("Invalid native EGS4 wavefront configuration");
  }
  void load(std::vector<C8Particle> const& input,std::uint64_t seed,std::uint64_t shower_id,
            std::uint64_t first_child_id) {
    if(input.size()>std::size_t(capacity_))throw std::runtime_error("Native EGS4 input exceeds queue capacity");
    Records candidate("egs4_active",input.size());auto host=Kokkos::create_mirror_view(candidate);
    std::set<std::uint64_t> ids;
    for(std::size_t i=0;i<input.size();++i) {
      auto const& p=input[i];
      if(p.medium_id!=medium_||p.history_id>=first_child_id||!ids.insert(p.history_id).second||
         !importRecord(p,frame_,seed,shower_id,host(i)))
        throw std::runtime_error("Invalid native EGS4 input, medium or history-ID range");
    }
    Kokkos::deep_copy(candidate,host);active_=candidate;next_id_=first_child_id;
    seed_=seed;shower_id_=shower_id;
  }
  std::size_t activeCount()const{return active_.extent(0);}
  std::uint64_t nextHistoryId()const{return next_id_;}
  Records activeRecords()const{return active_;}
  // Resume a native handoff after the caller has handled an observation or a
  // same-model host vertex. Never re-import a bare C8 carrier here: that would
  // reset its optical clock and RNG. New host children need reserved IDs first.
  void resume(std::vector<NativeRecord> const& input) {
    if(input.size()>std::size_t(capacity_)-activeCount())
      throw std::runtime_error("Native EGS4 resumed records exceed queue capacity");
    auto current=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},active_);
    Records candidate("egs4_resumed",activeCount()+input.size());auto host=Kokkos::create_mirror_view(candidate);
    std::set<std::uint64_t> ids;
    for(std::size_t i=0;i<activeCount();++i){host(i)=current(i);ids.insert(current(i).metadata.history_id);}
    for(std::size_t i=0;i<input.size();++i) {
      auto r=input[i];auto const& key=r.random.key;
      if(r.metadata.medium_id!=medium_||r.metadata.history_id>=next_id_||!ids.insert(r.metadata.history_id).second||
         r.state.pending_channel!=Channel::invalid||key.history_id!=r.metadata.history_id||key.seed!=seed_||
         key.shower_id!=shower_id_||key.process_id!=NativeEgs4RandomDomain||
         !finite(r.metadata.weight)||r.metadata.weight<0.||!exportParticle(r.state,frame_,r.metadata))
        throw std::runtime_error("Invalid native EGS4 continuation or unresolved host channel");
      host(activeCount()+i)=r;
    }
    Kokkos::deep_copy(candidate,host);active_=candidate;
  }

  template<class Geometry> Batch advance(Geometry geometry) {
    int n=int(activeCount());Batch batch;
    batch.before=active_;
    batch.records=Records("egs4_after_step",n);batch.outcomes=Outcomes("egs4_outcomes",n);
    if(!n)return batch;
    Kokkos::deep_copy(batch.records,active_);
    auto records=batch.records;auto outcomes=batch.outcomes;
    auto model=model_.view();auto frame=frame_;auto region=region_;auto field=field_;auto settings=settings_;
    int errors=0;
    Kokkos::parallel_reduce("egs4_native_advance",Kokkos::RangePolicy<Exec>(0,n),
      AdvanceKernel<Geometry>{model,records,outcomes,frame,region,field,settings,geometry},errors);
    if(errors)throw std::runtime_error("Native EGS4 history failure; queue not committed, no alternate-model fallback");

    Offsets offsets("egs4_offsets",n),child_offsets("egs4_child_offsets",n);
    int next_count=0,created=0;
    Kokkos::parallel_scan("egs4_native_queue_offsets",Kokkos::RangePolicy<Exec>(0,n),
      OffsetKernel{outcomes,offsets,false},next_count);
    Kokkos::parallel_scan("egs4_native_child_ids",Kokkos::RangePolicy<Exec>(0,n),
      OffsetKernel{outcomes,child_offsets,true},created);
    if(next_count>capacity_)throw std::runtime_error("Native EGS4 queue overflow; no children discarded");
    if(std::uint64_t(created)>std::numeric_limits<std::uint64_t>::max()-next_id_)
      throw std::runtime_error("Native EGS4 history-ID exhaustion");
    Records next("egs4_next",next_count);auto first_id=next_id_;errors=0;
    Kokkos::parallel_reduce("egs4_native_emit",Kokkos::RangePolicy<Exec>(0,n),
      EmitKernel{records,next,outcomes,offsets,child_offsets,frame,first_id},errors);
    if(errors)throw std::runtime_error("Native EGS4 child metadata failure; queue not committed");
    Exec().fence();active_=next;next_id_+=std::uint64_t(created);
    batch.next_count=next_count;batch.created_count=created;return batch;
  }
};
} // namespace c7_egs4::c8_adapter
