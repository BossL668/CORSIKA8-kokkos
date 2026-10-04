#pragma once
#include "Egs4C8AirOutput.hpp"
#include "Egs4C8Rare.hpp"
#include "Egs4C8Thinning.hpp"
#include <limits>
#include <set>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace c7_egs4::c8_adapter {
// Native C++ B-backend building block for C8. Curved coordinates, optical
// clocks, RNG and all three species remain resident across atmospheric layers.
// No PROPOSAL fallback or false hasProposalTable() compatibility flag.
template<class Exec>class AirWavefront {
public:
  using Records=Kokkos::View<AirRecord*,typename Exec::memory_space>;
  using Outcomes=Kokkos::View<AirOutcome*,typename Exec::memory_space>;
  using Outputs=Kokkos::View<AirOutput*,typename Exec::memory_space>;
  using HostTransfers=Kokkos::View<RareAirTransfer*,typename Exec::memory_space>;
  using Offsets=Kokkos::View<int*,typename Exec::memory_space>;
  using ThinResults=Kokkos::View<ThinningResult*,typename Exec::memory_space>;
  struct Batch {Records before,after;Outcomes outcomes;Outputs outputs;HostTransfers host;int created{};ThinResults thinning;};
  KOKKOS_INLINE_FUNCTION static void clearTransfer(RareAirTransfer& h,unsigned mask) {
    if(!(mask&overhead::compact_clear)){h=RareAirTransfer{};return;}
    h.valid=false;h.status=InteractionStatus::invalid_input;h.count=0;h.created=0;
    h.target_rest_energy_MeV=0.;h.has_rho_decay=false;h.has_resonance_decay=false;h.resonance_children=0;
  }
  struct Advance {
    ModelView model;AirEnvironment env;TransportSettings settings;Records before,after;Outcomes outcomes;Outputs outputs;
    HostTransfers host;bool resolve_rare;RareMaterial rare;PhotonuclearPolicy nuclear;
    ThinningConfig thinning;ThinResults decisions;bool initialize_in_kernel{};
    KOKKOS_INLINE_FUNCTION void operator()(int i,int& errors)const {
      if(initialize_in_kernel) {
        after(i)=before(i);
        // Ordinary histories do not otherwise assign this slot. Clearing it
        // here avoids a separate deep-copy/fence without leaking old vertices.
        clearTransfer(host(i),env.overhead_mask);
      }
      auto r=advanceAirRecord(model,after(i),env,settings);
      // Freeze the transported segment BEFORE a stochastic vertex changes the
      // surviving lepton energy. It must not alter the preceding radio track.
      outputs(i)=makeAirOutput(before(i),after(i),r,env);
      if(resolve_rare&&r.action==AirAction::needs_host_channel) {
        host(i)=resolveRareAirRecord(model,after(i),r,env,settings,rare,nuclear);
        errors+=!host(i).valid;
      }
      outcomes(i)=r;
      decisions(i)=thinAirVertex(thinning,after(i),r);
      errors+=decisions(i).status==ThinningStatus::InvalidInput;
      errors+=r.action==AirAction::error||!outputs(i).valid;
    }
  };
  struct TransportCounts {int errors{},rare{};};
  // The main transport kernel must not inherit the register/stack footprint of
  // infrequent photonuclear/resonance final states. Compact their indices only;
  // the physical resolver below is unchanged and still writes original slots.
  struct TransportOnly {
    using value_type=TransportCounts;
    ModelView model;AirEnvironment env;TransportSettings settings;
    Records before,after;Outcomes outcomes;Outputs outputs;HostTransfers host;
    ThinningConfig thinning;ThinResults decisions;Offsets rare_indices,rare_cursor;
    KOKKOS_INLINE_FUNCTION void init(value_type& v)const {v={};}
    KOKKOS_INLINE_FUNCTION void join(value_type& a,value_type const& b)const {a.errors+=b.errors;a.rare+=b.rare;}
    KOKKOS_INLINE_FUNCTION void operator()(int i,value_type& count)const {
      after(i)=before(i);clearTransfer(host(i),env.overhead_mask);
      auto r=advanceAirRecord(model,after(i),env,settings);
      outputs(i)=makeAirOutput(before(i),after(i),r,env);outcomes(i)=r;
      decisions(i)=thinAirVertex(thinning,after(i),r);
      count.errors+=r.action==AirAction::error||!outputs(i).valid;
      count.errors+=decisions(i).status==ThinningStatus::InvalidInput;
      if(r.action==AirAction::needs_host_channel) {
        int const slot=Kokkos::atomic_fetch_add(&rare_cursor(0),1);
        rare_indices(slot)=i;++count.rare;
      }
    }
  };
  struct ResolveCompactedRare {
    ModelView model;AirEnvironment env;TransportSettings settings;
    Records after;Outcomes outcomes;HostTransfers host;RareMaterial rare;PhotonuclearPolicy nuclear;
    ThinningConfig thinning;ThinResults decisions;Offsets indices;
    KOKKOS_INLINE_FUNCTION void operator()(int slot,int& errors)const {
      int const i=indices(slot);auto r=outcomes(i);
      host(i)=resolveRareAirRecord(model,after(i),r,env,settings,rare,nuclear);
      outcomes(i)=r;decisions(i)=thinAirVertex(thinning,after(i),r);
      errors+=!host(i).valid||r.action==AirAction::error;
      errors+=decisions(i).status==ThinningStatus::InvalidInput;
    }
  };
  struct Offset {
    Outcomes outcomes;HostTransfers host;Offsets offsets;bool children;ThinResults decisions;
    KOKKOS_INLINE_FUNCTION void operator()(int i,int& sum,bool final)const {
      if(final)offsets(i)=sum;auto const& r=outcomes(i);
      if(r.action==AirAction::replaced)for(int j=0;j<r.history.child_count;++j)
        sum+=children||((decisions(i).keep_mask>>j)&1U);
      else sum+=!children&&r.action==AirAction::active?1:0;
      if(children)sum+=host(i).created;
      else for(int j=0;j<host(i).count;++j)sum+=host(i).vertices[j].kind==HostVertexKind::native_em;
    }
  };
  // The two stable prefixes are integer sums over the same input order.
  // Combining them does not change thinning, IDs, RNG or daughter ordering.
  struct PrefixCounts {int live{},created{};};
  struct CombinedOffset {
    using value_type=PrefixCounts;
    Outcomes outcomes;HostTransfers host;Offsets offsets,child_offsets;ThinResults decisions;
    KOKKOS_INLINE_FUNCTION void init(value_type& v)const {v.live=0;v.created=0;}
    KOKKOS_INLINE_FUNCTION void join(value_type& a,value_type const& b)const {a.live+=b.live;a.created+=b.created;}
    KOKKOS_INLINE_FUNCTION void operator()(int i,value_type& sum,bool final)const {
      if(final){offsets(i)=sum.live;child_offsets(i)=sum.created;}
      auto const& r=outcomes(i);
      if(r.action==AirAction::replaced)for(int j=0;j<r.history.child_count;++j) {
        ++sum.created;sum.live+=(decisions(i).keep_mask>>j)&1U;
      }
      else sum.live+=r.action==AirAction::active;
      sum.created+=host(i).created;
      for(int j=0;j<host(i).count;++j)sum.live+=host(i).vertices[j].kind==HostVertexKind::native_em;
    }
  };
  struct Emit {
    Records after,next;Outcomes outcomes;HostTransfers host;Offsets offsets,child_offsets;AirEnvironment env;std::uint64_t first;
    ThinResults decisions;
    KOKKOS_INLINE_FUNCTION void operator()(int i,int& errors)const {
      auto const& r=outcomes(i);auto const& parent=after(i);int offset=offsets(i);
      if(!assignHostHistoryIds(host(i),parent.metadata,first+child_offsets(i))){++errors;return;}
      if(r.action==AirAction::active)next(offset++)=parent;
      else if(r.action==AirAction::replaced)for(int j=0;j<r.history.child_count;++j) {
        if(!((decisions(i).keep_mask>>j)&1U))continue;
        AirRecord child=parent;child.track=r.children[j];auto& p=child.metadata;
        if(p.generation==~std::uint32_t{0}||!exportAirParticle(child.track,env,p)){++errors;continue;}
        child.exported_position=true;
        p.parent_history_id=parent.metadata.history_id;p.history_id=first+child_offsets(i)+j;p.step_id=0;++p.generation;
        p.weight=j?decisions(i).second_weight:decisions(i).first_weight;
        child.random.key={parent.random.key.seed,parent.random.key.shower_id,p.history_id,0,NativeEgs4RandomDomain,0};
        child.hadron_random.key={parent.hadron_random.key.seed,parent.hadron_random.key.shower_id,p.history_id,0,NativeEgs4HadronRandomDomain,0};
        next(offset++)=child;
      }
      for(int j=0;j<host(i).count;++j) {
        auto const& v=host(i).vertices[j];if(v.kind!=HostVertexKind::native_em)continue;
        // Preserve the C7 local/observer geometry of the vertex, not a fresh
        // Cartesian reimport. New daughters never inherit the nuclear clock.
        AirRecord child=parent;child.metadata=v.particle;auto& p=child.track.particle;
        p.pdg=v.native_secondary.pdg;p.energy_MeV=v.native_secondary.energy_MeV;
        p.direction=v.native_secondary.direction;p.clock={};p.photon_clock={};p.pending_channel=Channel::invalid;
        child.random=v.random;child.hadron_random=v.hadron_random;
        if(p.pdg!=22||!validTrack(p)){++errors;continue;}
        next(offset++)=child;
      }
    }
  };
private:
  DeviceModel<Exec> model_;AirEnvironment env_;TransportSettings settings_;int capacity_;
  bool resolve_rare_;RareMaterial rare_;PhotonuclearPolicy nuclear_;
  ThinningConfig thinning_;
  bool reuse_workspace_{};
  bool split_rare_kernels_{};
  Records active_storage_,after_storage_,next_storage_;
  Outcomes outcomes_storage_;Outputs outputs_storage_;HostTransfers host_storage_;
  ThinResults thinning_storage_;Offsets offsets_storage_,children_storage_;
  Offsets rare_indices_storage_,rare_cursor_storage_;
  std::uint64_t workspace_allocations_{},workspace_reuses_{};
  std::uint64_t processed_records_{},peak_active_{};
  double advance_wall_ms_{};
  std::uint64_t append_calls_{},append_records_{},append_copied_active_records_{};
  double append_wall_ms_{};
  std::uint64_t split_rare_waves_{},split_rare_records_{};
  template<class View>View workspace(View& storage,char const* name,int n) {
    if(!reuse_workspace_){++workspace_allocations_;return View(std::string(name),n);}
    if(storage.extent(0)<std::size_t(n)) {
      std::size_t grown=std::max<std::size_t>(1,storage.extent(0));
      while(grown<std::size_t(n))grown=std::min<std::size_t>(capacity_,grown*2);
      storage=View(std::string(name),grown);
      ++workspace_allocations_;
    }else ++workspace_reuses_;
    return Kokkos::subview(storage,std::make_pair(std::size_t(0),std::size_t(n)));
  }
  Records active_;std::uint64_t next_id_{};
  std::uint64_t seed_{},shower_{};
  std::set<std::uint64_t> imported_ids_;
  std::vector<std::pair<std::uint64_t,std::uint64_t>> child_ranges_;
public:
  AirWavefront(Tables const& tables,ChannelThresholds thresholds,AirEnvironment env,TransportSettings settings,int capacity,
      bool resolve_rare=false,RareMaterial rare={},PhotonuclearPolicy nuclear={},ThinningConfig thinning={},bool reuse_workspace=false,bool split_rare_kernels=false):
    model_(tables,thresholds),env_(env),settings_(settings),capacity_(capacity),resolve_rare_(resolve_rare),rare_(rare),nuclear_(nuclear),thinning_(thinning),reuse_workspace_(reuse_workspace),split_rare_kernels_(split_rare_kernels) {
    validateThinning(thinning);
    if(capacity<1||capacity>std::numeric_limits<int>::max()/5||!validSettings(model_.view(),settings))
      throw std::invalid_argument("Invalid native C8 EGS4 air queue settings");
    if(resolve_rare&&(!validRareMaterial(rare)||rare.electron_mass_MeV!=thresholds.mass_MeV))
      throw std::invalid_argument("Explicit C7 rare-channel material/masses required");
    if(!validPhotonuclearPolicy(nuclear,rare)||(nuclear.enabled&&!resolve_rare))
      throw std::invalid_argument("Native photonuclear vertices require explicit masses/tables and rare-vertex resolution");
  }
  void beginShower(std::vector<C8Particle> const& input,std::uint64_t seed,std::uint64_t shower,std::uint64_t first_child) {
    if(input.size()>std::size_t(capacity_))throw std::length_error("EGS4 input exceeds capacity");
    Records candidate("egs4_air_input",input.size());auto host=Kokkos::create_mirror_view(candidate);std::set<std::uint64_t> ids;
    for(std::size_t i=0;i<input.size();++i)if(input[i].history_id>=first_child||!ids.insert(input[i].history_id).second||
        !importAirRecord(input[i],env_,seed,shower,host(i)))throw std::invalid_argument("Invalid EGS4 C8 particle/ID/medium");
    Kokkos::deep_copy(candidate,host);active_=candidate;active_storage_=candidate;next_id_=first_child;
    seed_=seed;shower_=shower;imported_ids_=std::move(ids);child_ranges_.clear();
  }
  // A completed event releases history/RNG identity but retains buffer capacity.
  void resetForShower(std::uint64_t seed,std::uint64_t shower) {
    if(activeCount())throw std::logic_error("Cannot reset a nonempty EGS4 queue");
    seed_=seed;shower_=shower;next_id_=1;imported_ids_.clear();child_ranges_.clear();
    active_=Kokkos::subview(active_storage_,std::make_pair(std::size_t(0),std::size_t(0)));
    workspace_allocations_=workspace_reuses_=processed_records_=peak_active_=0;
    append_calls_=append_records_=append_copied_active_records_=0;
    split_rare_waves_=split_rare_records_=0;advance_wall_ms_=append_wall_ms_=0.;
  }
  // New C8 stack particles only. Existing native continuations never pass
  // through this import: their optical clocks and random streams stay resident.
  void append(std::vector<C8Particle> const& input) {
    if(input.empty())return;
    auto const started=std::chrono::steady_clock::now();
    if(input.size()>std::size_t(capacity_)-activeCount())throw std::length_error("Native air append exceeds capacity");
    // The reference keeps its owning copy; the resident path checks only the
    // incoming batch against the accumulated set. Nodes are merged only after
    // every import and device copy succeeds, preserving failed-append behavior.
    auto ids=reuse_workspace_?std::set<std::uint64_t>{}:imported_ids_;auto next_id=next_id_;
    Records added("egs4_air_added",input.size());auto host=Kokkos::create_mirror_view(added);
    for(std::size_t i=0;i<input.size();++i) {
      auto id=input[i].history_id;
      auto range=std::upper_bound(child_ranges_.begin(),child_ranges_.end(),id,
        [](auto key,auto const& r){return key<r.first;});
      bool native_id=range!=child_ranges_.begin()&&id<std::prev(range)->second;
      if(id==0||id==~std::uint64_t{0}||native_id||(reuse_workspace_&&imported_ids_.count(id))||!ids.insert(id).second||
          !importAirRecord(input[i],env_,seed_,shower_,host(i)))
        throw std::invalid_argument("Invalid/duplicate native air injection; continuation must not be reimported");
      next_id=std::max(next_id,id+1);
    }
    Kokkos::deep_copy(added,host);auto old=activeCount();
    Records joined=reuse_workspace_?workspace(active_storage_,"egs4_air_joined",int(old+input.size())):
      Records("egs4_air_joined",old+input.size());
    if(old&&joined.data()!=active_.data()) {
      Kokkos::deep_copy(Kokkos::subview(joined,std::make_pair(std::size_t(0),old)),active_);
      append_copied_active_records_+=old;
    }
    Kokkos::deep_copy(Kokkos::subview(joined,std::make_pair(old,old+input.size())),added);
    active_=joined;
    if(reuse_workspace_)imported_ids_.merge(ids);
    else {active_storage_=joined;imported_ids_=std::move(ids);}
    next_id_=next_id;++append_calls_;append_records_+=input.size();
    append_wall_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
  }
  std::size_t activeCount()const{return active_.extent(0);}
  std::uint64_t nextHistoryId()const{return next_id_;}
  std::uint64_t workspaceAllocations()const{return workspace_allocations_;}
  std::uint64_t workspaceReuses()const{return workspace_reuses_;}
  std::uint64_t processedRecords()const{return processed_records_;}
  std::uint64_t peakActive()const{return peak_active_;}
  double advanceWallMilliseconds()const{return advance_wall_ms_;}
  std::uint64_t appendCalls()const{return append_calls_;}
  std::uint64_t appendRecords()const{return append_records_;}
  std::uint64_t appendCopiedActiveRecords()const{return append_copied_active_records_;}
  double appendWallMilliseconds()const{return append_wall_ms_;}
  std::uint64_t splitRareWaves()const{return split_rare_waves_;}
  std::uint64_t splitRareRecords()const{return split_rare_records_;}
  Records activeRecords()const{return active_;}
  Batch advance(){return advance(next_id_,~std::uint64_t{0}-next_id_);}
  // In HybridCascade the CPU stack is the sole allocator. It reserves up to
  // five histories with omega/phi decay, four with prompt rho (including the transient parent),
  // three with native photonuclear generation, or two otherwise. Unused IDs are
  // harmless gaps, never reused or assigned concurrently by the GPU.
  Batch advance(std::uint64_t first_child,std::uint64_t reserved) {
    auto const start=std::chrono::steady_clock::now();
    if(first_child<next_id_||reserved>~std::uint64_t{0}-first_child)
      throw std::invalid_argument("Overlapping/overflowing native child ID reservation");
    // With reuse enabled, Batch is a borrowed view valid until the next
    // advance/beginShower. The session consumes it completely before advancing.
    // Default standalone API retains its original owning/snapshot semantics.
    int n=int(activeCount());Batch b;b.before=active_;b.after=workspace(after_storage_,"egs4_air_after",n);
    b.outcomes=workspace(outcomes_storage_,"egs4_air_outcomes",n);
    b.outputs=workspace(outputs_storage_,"egs4_air_outputs",n);
    b.host=workspace(host_storage_,"egs4_air_host",n);
    b.thinning=workspace(thinning_storage_,"egs4_air_thinning",n);
    if(!n)return b;
    bool const split=split_rare_kernels_&&resolve_rare_;
    int errors=0,rare_records=0;
    if(split) {
      auto indices=workspace(rare_indices_storage_,"egs4_air_rare_indices",n);
      auto cursor=workspace(rare_cursor_storage_,"egs4_air_rare_cursor",1);
      // Stream-ordered reset; the reduction below supplies the completion fence.
      Kokkos::deep_copy(Exec(),cursor,0);
      TransportCounts counts;
      Kokkos::parallel_reduce("egs4_air_transport_only",Kokkos::RangePolicy<Exec>(0,n),
        TransportOnly{model_.view(),env_,settings_,b.before,b.after,b.outcomes,b.outputs,b.host,
          thinning_,b.thinning,indices,cursor},counts);
      errors=counts.errors;rare_records=counts.rare;
      if(!errors&&rare_records)Kokkos::parallel_reduce("egs4_air_compacted_rare",Kokkos::RangePolicy<Exec>(0,rare_records),
        ResolveCompactedRare{model_.view(),env_,settings_,b.after,b.outcomes,b.host,rare_,nuclear_,thinning_,b.thinning,indices},errors);
    }else {
      if(!reuse_workspace_)Kokkos::deep_copy(b.after,b.before);
      Kokkos::parallel_reduce("egs4_air_advance",Kokkos::RangePolicy<Exec>(0,n),
        Advance{model_.view(),env_,settings_,b.before,b.after,b.outcomes,b.outputs,b.host,resolve_rare_,rare_,nuclear_,thinning_,b.thinning,reuse_workspace_},errors);
    }
    if(errors) {
      auto states=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},b.before);
      auto outcomes=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},b.outcomes);
      auto output=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},b.outputs);
      auto transfers=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},b.host);
      auto thin=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},b.thinning);
      std::ostringstream message;message<<std::setprecision(17)<<"Native air history failed; live queue/RNG not committed";
      for(int i=0;i<n;++i)if(outcomes(i).action==AirAction::error||!output(i).valid||
          thin(i).status==ThinningStatus::InvalidInput||
          (outcomes(i).action==AirAction::needs_host_channel&&!transfers(i).valid)) {
        auto const& s=states(i);auto const& p=s.track.particle;
        message<<"; history="<<s.metadata.history_id<<" step="<<s.metadata.step_id<<" pdg="<<p.pdg
          <<" E_MeV="<<p.energy_MeV<<" region="<<s.region<<" position_cm="<<p.position_cm.x<<","<<p.position_cm.y<<","<<p.position_cm.z
          <<" direction="<<p.direction.x<<","<<p.direction.y<<","<<p.direction.z
          <<" action="<<int(outcomes(i).action)<<" history_error="<<int(outcomes(i).history.error)
          <<" channel="<<int(outcomes(i).history.channel)<<" output_valid="<<output(i).valid
          <<" random_draw="<<s.random.key.draw_id;break;
      }
      throw std::runtime_error(message.str());
    }
    auto offsets=workspace(offsets_storage_,"egs4_air_offsets",n);
    auto children=workspace(children_storage_,"egs4_air_children",n);int count=0;
    if(reuse_workspace_) {
      PrefixCounts totals;
      Kokkos::parallel_scan("egs4_air_combined_count",Kokkos::RangePolicy<Exec>(0,n),
        CombinedOffset{b.outcomes,b.host,offsets,children,b.thinning},totals);
      count=totals.live;b.created=totals.created;
    }else {
      Kokkos::parallel_scan("egs4_air_count",Kokkos::RangePolicy<Exec>(0,n),Offset{b.outcomes,b.host,offsets,false,b.thinning},count);
      Kokkos::parallel_scan("egs4_air_child_count",Kokkos::RangePolicy<Exec>(0,n),Offset{b.outcomes,b.host,children,true,b.thinning},b.created);
    }
    if(count>capacity_)throw std::length_error("Native air queue overflow; no children discarded; active="+
      std::to_string(n)+" required="+std::to_string(count)+" capacity="+std::to_string(capacity_));
    if(std::uint64_t(b.created)>reserved)throw std::length_error("Native air history ID reservation exhausted");
    auto next=workspace(next_storage_,"egs4_air_next",count);errors=0;
    Kokkos::parallel_reduce("egs4_air_emit",Kokkos::RangePolicy<Exec>(0,n),
      Emit{b.after,next,b.outcomes,b.host,offsets,children,env_,first_child,b.thinning},errors);
    if(errors)throw std::runtime_error("Native air child export failed; queue not committed");
    Exec().fence();active_=next;
    if(reuse_workspace_)std::swap(active_storage_,next_storage_);
    else active_storage_=next;
    next_id_=first_child+b.created;
    if(b.created)child_ranges_.emplace_back(first_child,next_id_);
    processed_records_+=n;peak_active_=std::max(peak_active_,std::uint64_t(n));
    split_rare_waves_+=split;split_rare_records_+=rare_records;
    advance_wall_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return b;
  }
};
} // namespace c7_egs4::c8_adapter
