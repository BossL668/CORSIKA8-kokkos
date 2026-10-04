#pragma once
#include <corsika/accelerator/em/common/Types.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace c7_egs4::application {
// Optional host-only diagnostic of the experiment's existing handoff records.
// Never draws random numbers or changes the records sent to either backend.
class MuonAudit {
  using Particle=::corsika::gpu::em::EmParticleState;
  std::ofstream out_;
  void vector(double const* v) {out_<<'['<<v[0]<<','<<v[1]<<','<<v[2]<<']';}
  void state(Particle const& p) {
    out_<<"{\"pid\":"<<p.pid<<",\"history\":"<<p.history_id<<",\"parent\":"<<p.parent_history_id
      <<",\"step\":"<<p.step_id<<",\"generation\":"<<p.generation<<",\"medium\":"<<p.medium_id
      <<",\"energy\":"<<p.energy_GeV<<",\"position\":";vector(p.position_m);
    out_<<",\"direction\":";vector(p.direction);
    out_<<",\"time\":"<<p.time_s<<",\"weight\":"<<p.weight<<'}';
  }
public:
  MuonAudit() {
    auto path=std::getenv("C8_EGS4_MUON_AUDIT");
    if(!path||!*path)return;
    if(std::filesystem::exists(path))throw std::runtime_error("Muon audit file already exists");
    out_.exceptions(std::ios::badbit|std::ios::failbit);out_.open(path);out_<<std::setprecision(17);
  }
  void particle(char const* kind,Particle const& p) {
    if(!out_.is_open())return;
    out_<<"{\"kind\":\""<<kind<<"\",\"state\":";state(p);out_<<"}\n";
  }
  void wave(::corsika::gpu::em::ResidentLeptonCascadeResult const& result) {
    if(!out_.is_open())return;
    for(auto const& r:result.step_records) {
      out_<<"{\"kind\":\"step\",\"start\":";state(r.start);out_<<",\"end\":";state(r.end);
      out_<<",\"limit\":"<<int(r.limit)<<",\"process\":"<<r.interaction.process_id
        <<",\"distance\":"<<r.distance_m<<",\"grammage\":"<<r.traversed_grammage_g_per_cm2
        <<",\"deposit\":"<<r.continuous_deposited_energy_GeV<<",\"scatter_angle\":"<<r.multiple_scattering_angle_rad
        <<",\"scatter_u\":["<<r.multiple_scattering_first_uniform<<','<<r.multiple_scattering_second_uniform
        <<','<<r.multiple_scattering_azimuth_uniform<<"],\"selection_u\":"<<r.interaction.proposal_selection_uniform
        <<",\"loss_quantile\":"<<r.interaction.loss_quantile<<"}\n";
    }
    for(auto const& e:result.fallback_events) {
      out_<<"{\"kind\":\"fallback\",\"state\":";state(e.particle);
      out_<<",\"reason\":"<<int(e.reason)<<",\"process\":"<<e.process_id
        <<",\"vertex_reached\":"<<e.interaction_vertex_reached<<",\"component_hash\":"<<e.component_hash
        <<",\"interaction_hash\":"<<e.interaction_hash<<",\"energy_fraction\":"<<e.energy_fraction
        <<",\"selection_u\":"<<e.selection_uniform<<",\"loss_quantile\":"<<e.loss_quantile
        <<",\"final_state_u\":"<<e.final_state_uniform<<",\"draw\":"<<e.random_draw_id
        <<",\"final_state_draw\":"<<e.final_state_draw_id<<"}\n";
    }
  }
};
} // namespace c7_egs4::application
