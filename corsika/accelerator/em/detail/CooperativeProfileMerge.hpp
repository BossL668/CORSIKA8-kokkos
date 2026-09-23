/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#pragma once
#include <corsika/accelerator/em/detail/CheckedFixedAccumulatorMerge.hpp>
#include <corsika/accelerator/em/detail/CooperativeScheduling.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/detail/ProfileProjectionData.hpp>
#include <algorithm>
#include <type_traits>
namespace corsika::accelerator::em::detail {
struct FixedProfileSnapshot {
  std::string shower_identity;
  CooperativeEndpoint endpoint{};
  gpu::em::GpuEmConfig::ProfileProjection config;
  double weight_units{}, energy_units{};
  std::vector<std::int64_t> histograms;
  gpu::em::detail::DeviceProfileCounters counters{};
};
inline bool sameProfileConfig(gpu::em::GpuEmConfig::ProfileProjection const& a,
                              gpu::em::GpuEmConfig::ProfileProjection const& b) {
#define SAME(f) if(a.f!=b.f) return false
  SAME(enabled); SAME(accumulate_on_device); SAME(axis_step_length_m);
  SAME(axis_grammage_g_per_cm2); SAME(output_bin_count); SAME(output_bin_width_g_per_cm2);
  SAME(energy_loss_threshold_g_per_cm2); SAME(fixed_point_weight_limit);
  SAME(fixed_point_energy_limit_GeV);
  SAME(crossing_mode);
  for(int d=0;d<3;++d){SAME(axis_start_position_m[d]);SAME(axis_direction[d]);}
#undef SAME
  return true;
}
inline void validateFixedProfile(FixedProfileSnapshot const& s) {
  if(s.counters.fixed_point_overflows || s.counters.invalid_records)
    throw std::runtime_error("cooperative profile rejected: overflows="+
        std::to_string(s.counters.fixed_point_overflows)+", invalid_records="+
        std::to_string(s.counters.invalid_records));
  if(s.shower_identity.empty() || static_cast<unsigned>(s.endpoint)>1 ||
     !s.config.enabled || !s.config.accumulate_on_device || !s.config.output_bin_count ||
     s.config.output_bin_count>std::numeric_limits<std::size_t>::max()/8 ||
     s.histograms.size()!=8*s.config.output_bin_count ||
     !std::isfinite(s.weight_units)||s.weight_units<=0. ||
     !std::isfinite(s.energy_units)||s.energy_units<=0.)
    throw std::invalid_argument("invalid/overflowed cooperative profile snapshot");
  if(s.weight_units!=1./(0x1p62/s.config.fixed_point_weight_limit) ||
     s.energy_units!=1./(0x1p62/s.config.fixed_point_energy_limit_GeV))
    throw std::invalid_argument("cooperative profile scale mismatch");
}
inline gpu::em::detail::DeviceProfileCounters mergeProfileCounters(
    gpu::em::detail::DeviceProfileCounters a,
    gpu::em::detail::DeviceProfileCounters const& b) {
  auto add=[](auto x,auto y) {
    using T=decltype(x);
    if constexpr(std::is_signed_v<T>) {
      if((y>0 && x>std::numeric_limits<T>::max()-y) ||
         (y<0 && x<std::numeric_limits<T>::min()-y))
        throw std::overflow_error("cooperative profile energy counter overflow");
    } else if(y>std::numeric_limits<T>::max()-x)
      throw std::overflow_error("cooperative profile counter overflow");
    return x+y;
  };
#define ADD(f) a.f=add(a.f,b.f)
  ADD(steps);ADD(deposited_steps);ADD(photon_cuts);
  for(unsigned i=0;i<8;++i) ADD(lepton_limits[i]);
  ADD(moliere_trials);ADD(moliere_deflections);ADD(moliere_zero_deflections);
  ADD(moliere_newton_iterations);ADD(thinning_hillas_vertices);
  ADD(thinning_statistical_vertices);ADD(thinning_particles_discarded);
  ADD(fixed_point_overflows);ADD(invalid_records);
  ADD(weighted_medium_rest_mass_input);ADD(weighted_cut_rest_mass_energy);
  ADD(weighted_observed_total_energy);ADD(weighted_escaped_total_energy);
  ADD(weighted_unwritten_photoelectric_binding_energy);
  ADD(weighted_observation_cut_overlap_energy);ADD(weighted_mass_convention_correction);
#undef ADD
  a.moliere_max_newton_iterations=std::max(a.moliere_max_newton_iterations,
                                         b.moliere_max_newton_iterations);
  return a;
}
inline gpu::em::GpuProfileResult decodeFixedProfile(FixedProfileSnapshot const& s) {
  validateFixedProfile(s);
  auto bins_=s.config.output_bin_count;
  gpu::em::detail::DeviceProfileAccumulator view_{};
  view_.inverse_weight_scale=s.weight_units;view_.inverse_energy_scale=s.energy_units;
  auto host_histograms_=[&](std::size_t i){return s.histograms.at(i);};
  auto const& counters=s.counters;
      gpu::em::GpuProfileResult result{};
      result.photons.resize(bins_);
      result.electrons.resize(bins_);
      result.positrons.resize(bins_);
      result.muons_minus.resize(bins_);
      result.muons_plus.resize(bins_);
      result.muon_parent_productions.resize(bins_);
      result.energy_loss_GeV.resize(bins_);
      result.muon_energy_loss_GeV.resize(bins_);
      for (std::size_t bin = 0; bin < bins_; ++bin) {
        result.photons[bin] = static_cast<double>(host_histograms_(bin)) *
                              view_.inverse_weight_scale;
        result.electrons[bin] =
            static_cast<double>(host_histograms_(bins_ + bin)) *
            view_.inverse_weight_scale;
        result.positrons[bin] =
            static_cast<double>(host_histograms_(2 * bins_ + bin)) *
            view_.inverse_weight_scale;
        result.muons_minus[bin] =
            static_cast<double>(host_histograms_(3 * bins_ + bin)) *
            view_.inverse_weight_scale;
        result.muons_plus[bin] =
            static_cast<double>(host_histograms_(4 * bins_ + bin)) *
            view_.inverse_weight_scale;
        result.muon_parent_productions[bin] =
            static_cast<double>(host_histograms_(5 * bins_ + bin)) *
            view_.inverse_weight_scale;
        auto const electromagnetic_loss =
            static_cast<double>(host_histograms_(6 * bins_ + bin)) *
            view_.inverse_energy_scale;
        result.muon_energy_loss_GeV[bin] =
            static_cast<double>(host_histograms_(7 * bins_ + bin)) *
            view_.inverse_energy_scale;
        result.energy_loss_GeV[bin] =
            electromagnetic_loss + result.muon_energy_loss_GeV[bin];
        result.weighted_deposited_energy_GeV += result.energy_loss_GeV[bin];
      }
      result.steps = counters.steps;
      result.deposited_steps = counters.deposited_steps;
      if (counters.lepton_limits[static_cast<std::size_t>(gpu::em::LeptonTransportLimit::ParticleCut)] >
          std::numeric_limits<unsigned long long>::max()-counters.photon_cuts)
        throw std::overflow_error("cooperative decoded particle cut counter overflow");
      result.particle_cuts =
          counters.photon_cuts +
          counters.lepton_limits[static_cast<std::size_t>(
              gpu::em::LeptonTransportLimit::ParticleCut)];
      result.fixed_point_overflows = counters.fixed_point_overflows;
      result.invalid_records = counters.invalid_records;
      result.weighted_medium_rest_mass_input_GeV =
          static_cast<double>(counters.weighted_medium_rest_mass_input) *
          view_.inverse_energy_scale;
      result.weighted_cut_rest_mass_energy_GeV =
          static_cast<double>(counters.weighted_cut_rest_mass_energy) *
          view_.inverse_energy_scale;
      result.weighted_observed_total_energy_GeV =
          static_cast<double>(counters.weighted_observed_total_energy) *
          view_.inverse_energy_scale;
      result.weighted_escaped_total_energy_GeV =
          static_cast<double>(counters.weighted_escaped_total_energy) *
          view_.inverse_energy_scale;
      result.weighted_unwritten_photoelectric_binding_energy_GeV =
          static_cast<double>(
              counters.weighted_unwritten_photoelectric_binding_energy) *
          view_.inverse_energy_scale;
      result.weighted_observation_cut_overlap_energy_GeV =
          static_cast<double>(counters.weighted_observation_cut_overlap_energy) *
          view_.inverse_energy_scale;
      result.weighted_mass_convention_correction_GeV =
          static_cast<double>(counters.weighted_mass_convention_correction) *
          view_.inverse_energy_scale;

  return result;
}
class CooperativeProfileMerge {
 public:
  CooperativeProfileMerge(std::string identity,gpu::em::GpuEmConfig::ProfileProjection config)
      :identity_(std::move(identity)),config_(std::move(config)) {}
  void commit(FixedProfileSnapshot const& s) {
    if(finished_) throw std::logic_error("cooperative profile already taken");
    validateFixedProfile(s);
    auto bit=1u<<static_cast<unsigned>(s.endpoint);
    if((seen_&bit)||s.shower_identity!=identity_||!sameProfileConfig(s.config,config_))
      throw std::invalid_argument("duplicate/incompatible cooperative profile commit");
    auto next=seen_?merged_:s;
    if(seen_) {
      if(next.weight_units!=s.weight_units || next.energy_units!=s.energy_units)
        throw std::invalid_argument("cooperative profile scales differ");
      FixedAccumulatorLayout layout{identity_+"/profile-eight-columns-v1",
                                     s.weight_units,s.histograms.size()};
      mergeFixedAccumulators(next.histograms,layout,s.histograms,layout);
      next.counters=mergeProfileCounters(next.counters,s.counters);
    }
    merged_=std::move(next);seen_|=bit;
  }
  FixedProfileSnapshot take() {
    if(seen_!=3||finished_) throw std::logic_error("profile endpoints not complete/already taken");
    finished_=true;return std::move(merged_);
  }
 private:
  std::string identity_;
  gpu::em::GpuEmConfig::ProfileProjection config_;
  FixedProfileSnapshot merged_;
  unsigned seen_{};
  bool finished_{};
};
} // namespace corsika::accelerator::em::detail
