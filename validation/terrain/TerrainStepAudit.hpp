// Test-only shadow replay. Never included by a normal terrain/air build.
#pragma once
#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
#include <corsika/accelerator/em/detail/LeptonVertexSelection.hpp>
#include <corsika/accelerator/em/common/ExternalTransportBoundary.hpp>
#include <corsika/modules/terrain/TerrainEmSession.hpp>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cstdlib>

namespace corsika::terrain::audit {
struct Record {
  accelerator::em::detail::InteractionSelectionOutcome selection{};
  em::ExternalTransportBoundary boundary{};
  em::LeptonTransportRecord transport{}, scattering{};
  accelerator::em::detail::LeptonVertexSelectionOutcome vertex{};
  accelerator::em::detail::PhotonTransportOutcome photon{};
  em::PhotonFinalStateRecord photon_final{};
  em::BremsFinalStateRecord lepton_final{};
  std::uint32_t transport_state{},scattering_state{};
  std::uint32_t stages{};
};

// CSV stores differences only; counts and identical inputs remain auditable.
// Compare numeric fields, never padding bytes or pointer addresses.
class Writer {
  std::ofstream file_;
  std::uint64_t batch_{}, rows_{}, fields_{}, differences_{}, failures_{};
  std::string prefix_;
  em::EmParticleState input_{};
  std::string mode_;
  std::size_t index_{};
  static std::uint64_t bits(double x) {
    std::uint64_t u; std::memcpy(&u,&x,sizeof(u)); return u;
  }
  static std::uint64_t ordered(double x) {
    auto u=bits(x); return (u>>63) ? ~u : (u | (UINT64_C(1)<<63));
  }
  template<class T> void integer(char const* stage,std::string const& name,T a,T b) {
    ++fields_; if(a==b)return;
    ++differences_;++failures_;
    file_<<batch_<<','<<index_<<','<<input_.history_id<<','<<input_.step_id<<','
         <<input_.pid<<','<<input_.medium_id<<','<<mode_<<','<<stage<<','<<name<<','
         <<a<<','<<b<<",integer,0\n";
  }
  void number(char const* stage,std::string const& name,double a,double b) {
    ++fields_; if(bits(a)==bits(b))return;
    ++differences_;
    bool close=(a==b)||(std::isfinite(a)&&std::isfinite(b)&&
        std::abs(a-b)<=std::max(1.e-12,1.e-10*std::max(std::abs(a),std::abs(b))));
    failures_+=!close;
    auto ua=ordered(a),ub=ordered(b);
    file_<<batch_<<','<<index_<<','<<input_.history_id<<','<<input_.step_id<<','
         <<input_.pid<<','<<input_.medium_id<<','<<mode_<<','<<stage<<','<<name<<','
         <<a<<','<<b<<','<<(ua>ub?ua-ub:ub-ua)<<','<<close<<'\n';
  }
  void particle(char const* s,std::string const& p,em::EmParticleState const& a,
                em::EmParticleState const& b) {
    integer(s,p+"pid",a.pid,b.pid);integer(s,p+"medium",a.medium_id,b.medium_id);
    integer(s,p+"history",a.history_id,b.history_id);
    integer(s,p+"step",a.step_id,b.step_id);integer(s,p+"generation",a.generation,b.generation);
    number(s,p+"energy_GeV",a.energy_GeV,b.energy_GeV);
    number(s,p+"time_s",a.time_s,b.time_s);number(s,p+"weight",a.weight,b.weight);
    for(int i=0;i<3;++i) {
      number(s,p+"position_"+std::to_string(i),a.position_m[i],b.position_m[i]);
      number(s,p+"direction_"+std::to_string(i),a.direction[i],b.direction[i]);
    }
  }
  void interaction(char const* s,em::EmInteractionRecord const& a,em::EmInteractionRecord const& b) {
#define I(f) integer(s,#f,a.f,b.f)
#define D(f) number(s,#f,a.f,b.f)
    I(process_id);I(component_hash);I(input_index);
    integer(s,"status",int(a.status),int(b.status));
    I(distance_draw_id);I(process_draw_id);I(loss_draw_id);I(process_random_process_id);
    I(proposal_selection_random_process_id);I(proposal_selection_draw_id);
    D(total_rate_cm2_per_g);D(vertex_total_rate_cm2_per_g);D(interaction_grammage_g_per_cm2);
    D(mass_density_g_per_cm3);D(particle_mass_GeV);D(energy_fraction);
    D(distance_uniform);D(process_uniform);D(loss_quantile);D(proposal_selection_uniform);
    particle(s,"particle_",a.particle,b.particle);
#undef I
#undef D
  }
  void transport(char const* s,em::LeptonTransportRecord const& a,em::LeptonTransportRecord const& b) {
#define I(f) integer(s,#f,a.f,b.f)
#define D(f) number(s,#f,a.f,b.f)
    integer(s,"limit",int(a.limit),int(b.limit));I(start_layer_index);I(end_layer_index);
    D(distance_m);D(traversed_grammage_g_per_cm2);D(continuous_step_grammage_g_per_cm2);
    D(start_density_g_per_cm3);D(end_density_g_per_cm3);D(continuous_deposited_energy_GeV);
    D(cut_deposited_energy_GeV);I(magnetic_bending_applied);I(magnetic_step_status);
    D(magnetic_step_limit_m);D(magnetic_gyroradius_m);D(magnetic_bend_parameter);D(magnetic_chord_length_m);
    I(multiple_scattering_applied);I(multiple_scattering_status);I(multiple_scattering_iterations);
    D(multiple_scattering_angle_rad);D(multiple_scattering_first_uniform);
    D(multiple_scattering_second_uniform);D(multiple_scattering_azimuth_uniform);
    particle(s,"end_",a.end,b.end);
#undef I
#undef D
  }
 public:
  Writer() {
    auto p=std::getenv("C8_TERRAIN_AUDIT_PREFIX");
    if(!p||!*p)throw std::runtime_error("diagnostic requires C8_TERRAIN_AUDIT_PREFIX");
    prefix_=p;
    if(std::filesystem::exists(prefix_+".csv"))throw std::runtime_error("audit output exists");
    file_.open(prefix_+".csv");if(!file_)throw std::runtime_error("cannot open audit CSV");
    file_<<std::setprecision(17)<<"batch,index,history,step,pid,medium,mode,stage,field,host,device,ulp,within_gate\n";
  }
  void compare(char const* mode,std::size_t i,em::EmParticleState const& input,
               Record const& a,Record const& b,TerrainEmStep const& x,TerrainEmStep const& y) {
    mode_=mode;index_=i;input_=input;++rows_;
    integer("stages","mask",a.stages,b.stages);
    interaction("selection",a.selection.interaction,b.selection.interaction);
    integer("selection","fallback_flag",a.selection.fallback_flag,b.selection.fallback_flag);
    if((a.stages&b.stages)&2) {
      integer("boundary","enabled",a.boundary.enabled,b.boundary.enabled);
      number("boundary","distance_m",a.boundary.distance_m,b.boundary.distance_m);
    }
    if((a.stages&b.stages)&4) {
      integer("transport","state",a.transport_state,b.transport_state);
      transport("transport",a.transport,b.transport);
    }
    if((a.stages&b.stages)&8) {
      integer("scattering","state",a.scattering_state,b.scattering_state);
      transport("scattering",a.scattering,b.scattering);
    }
    if((a.stages&b.stages)&16) {
      interaction("vertex",a.vertex.record,b.vertex.record);
      integer("vertex","fallback_flag",a.vertex.fallback_flag,b.vertex.fallback_flag);
      integer("vertex","continuation_flag",a.vertex.continuation_flag,b.vertex.continuation_flag);
    }
    if((a.stages&b.stages)&32) {
      integer("photon_transport","fallback_flag",a.photon.fallback_flag,b.photon.fallback_flag);
      integer("photon_transport","limit",int(a.photon.record.limit),int(b.photon.record.limit));
      number("photon_transport","distance_m",a.photon.record.distance_m,b.photon.record.distance_m);
      number("photon_transport","grammage",a.photon.record.traversed_grammage_g_per_cm2,b.photon.record.traversed_grammage_g_per_cm2);
      particle("photon_transport","end_",a.photon.record.end,b.photon.record.end);
    }
    if((a.stages&b.stages)&64) {
#define D(f) number("photon_final",#f,a.photon_final.f,b.photon_final.f)
#define I(f) integer("photon_final",#f,a.photon_final.f,b.photon_final.f)
      I(process_id);I(thinning_keep_mask);I(thinning_status);I(secondary_count);
      D(energy_split_fraction);D(split_uniform);D(azimuth_uniform);D(electron_polar_uniform);
      D(positron_polar_uniform);D(lpm_uniform);D(lpm_survival_probability);
      D(thinning_first_uniform);D(thinning_second_uniform);
      I(split_draw_id);I(azimuth_draw_id);I(electron_polar_draw_id);I(positron_polar_draw_id);
      I(lpm_draw_id);I(thinning_first_draw_id);I(thinning_second_draw_id);
#undef D
#undef I
    }
    if((a.stages&b.stages)&128) {
#define D(f) number("lepton_final",#f,a.lepton_final.f,b.lepton_final.f)
#define I(f) integer("lepton_final",#f,a.lepton_final.f,b.lepton_final.f)
      I(process_id);I(thinning_keep_mask);I(thinning_status);I(secondary_count);
      D(photon_energy_fraction);D(final_state_uniform);D(azimuth_uniform);D(auxiliary_uniform);
      D(lpm_uniform);D(lpm_survival_probability);D(thinning_first_uniform);D(thinning_second_uniform);
      I(final_state_draw_id);I(azimuth_draw_id);I(auxiliary_draw_id);I(lpm_draw_id);
      I(thinning_first_draw_id);I(thinning_second_draw_id);
#undef D
#undef I
    }
    integer("output","outcome",int(x.outcome),int(y.outcome));
    integer("output","children",x.child_count,y.child_count);integer("output","process",x.process_id,y.process_id);
    integer("output","crossed_material",x.crossed_material,y.crossed_material);
    integer("output","has_track",x.has_track,y.has_track);integer("output","error",x.error,y.error);
    integer("output","fallback_reason",int(x.fallback.reason),int(y.fallback.reason));
    number("output","distance_m",x.distance_m,y.distance_m);
    number("output","grammage",x.grammage_g_cm2,y.grammage_g_cm2);
    number("output","deposited_GeV",x.deposited_GeV,y.deposited_GeV);
    number("output","binding_GeV",x.binding_energy_GeV,y.binding_energy_GeV);
    particle("output","end_",x.end,y.end);
    for(std::uint32_t c=0;c<std::min(x.child_count,y.child_count);++c)
      particle("output","child"+std::to_string(c)+"_",x.children[c],y.children[c]);
    if(!file_)throw std::runtime_error("audit CSV write failed");
  }
  void endBatch() {++batch_;file_.flush();}
  ~Writer() {
    std::ofstream out(prefix_+".counts.json");
    out<<"{\"batches\":"<<batch_<<",\"compared_step_pairs\":"<<rows_
       <<",\"fields\":"<<fields_<<",\"bit_differences\":"<<differences_
       <<",\"strict_failures\":"<<failures_<<"}\n";
  }
};
}
