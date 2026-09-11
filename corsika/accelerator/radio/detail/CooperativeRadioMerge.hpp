/*
 * (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license.
 */
#pragma once
#include <corsika/accelerator/em/detail/CheckedFixedAccumulatorMerge.hpp>
#include <corsika/accelerator/em/detail/CooperativeScheduling.hpp>
#include <corsika/accelerator/radio/common/Types.hpp>
#include <corsika/accelerator/radio/detail/RadioProjectionData.hpp>
#include <algorithm>
#include <array>
#include <string>

namespace corsika::accelerator::radio::detail {
using CooperativeEndpoint = em::detail::CooperativeEndpoint;

/** Final (not incremental) per-endpoint ledger. ZHS integers are potential,
 * not differentiated fields. Identity must include shower/config identity.
 */
struct FixedRadioSnapshot {
  std::string shower_identity;
  CooperativeEndpoint endpoint{};
  gpu::radio::GpuRadioConfig config;
  std::vector<std::int64_t> coreas, zhs;
  DeviceRadioCounters counters{};
};

inline bool sameObservers(std::vector<gpu::radio::RadioObserverSnapshot> const& a,
                          std::vector<gpu::radio::RadioObserverSnapshot> const& b) {
  if (a.size()!=b.size()) return false;
  for (std::size_t i=0;i<a.size();++i) {
    for(int d=0;d<3;++d) if(a[i].position_m[d]!=b[i].position_m[d]) return false;
    if(a[i].start_time_s!=b[i].start_time_s || a[i].duration_s!=b[i].duration_s ||
       a[i].sample_rate_Hz!=b[i].sample_rate_Hz ||
       a[i].number_of_bins!=b[i].number_of_bins) return false;
  }
  return true;
}
inline bool sameRadioPhysics(gpu::radio::GpuRadioConfig const& a,
                             gpu::radio::GpuRadioConfig const& b) {
#define SAME(f) if(a.f!=b.f) return false
  SAME(enabled); SAME(coreas_enabled); SAME(zhs_enabled); SAME(deterministic);
  SAME(fixed_point_field_limit_V_per_m); SAME(zhs_subtrack_refinement); SAME(track_diagnostics);
  SAME(propagation.homogeneous_refractive_index);
  SAME(propagation.minimum_height_m); SAME(propagation.maximum_height_m);
  SAME(propagation.step_m); SAME(propagation.inverse_step_per_m);
  SAME(propagation.slope_refractivity_lower); SAME(propagation.slope_refractivity_upper);
  SAME(propagation.slope_integrated_refractivity_lower);
  SAME(propagation.slope_integrated_refractivity_upper);
  SAME(propagation.refractivity); SAME(propagation.integrated_refractivity);
#undef SAME
  return sameObservers(a.coreas_observers,b.coreas_observers) &&
         sameObservers(a.zhs_observers,b.zhs_observers);
}
inline std::size_t radioBins(std::vector<gpu::radio::RadioObserverSnapshot> const& observers) {
  std::size_t n=0;
  for(auto const& o:observers) {
    if (!o.number_of_bins || o.number_of_bins > std::numeric_limits<std::size_t>::max()/3-n ||
        !std::isfinite(o.sample_rate_Hz) || o.sample_rate_Hz<=0. ||
        !std::isfinite(o.start_time_s) || !std::isfinite(o.duration_s) || o.duration_s<0.)
      throw std::invalid_argument("invalid cooperative radio observer layout");
    for(double x:o.position_m)
      if(!std::isfinite(x)) throw std::invalid_argument("nonfinite cooperative radio observer");
    n+=o.number_of_bins;
  }
  return n;
}
inline void validateFixedRadio(FixedRadioSnapshot const& s) {
  if (s.shower_identity.empty() || !s.config.enabled || !s.config.deterministic ||
      (!s.config.coreas_enabled && !s.config.zhs_enabled) ||
      !std::isfinite(s.config.fixed_point_field_limit_V_per_m) ||
      s.config.fixed_point_field_limit_V_per_m<=0. ||
      static_cast<unsigned>(s.endpoint)>1 ||
      s.counters.fixed_point_overflows)
    throw std::invalid_argument("invalid/overflowed cooperative fixed radio snapshot");
  if(s.coreas.size()!=3*radioBins(s.config.coreas_observers) ||
     s.zhs.size()!=3*radioBins(s.config.zhs_observers))
    throw std::invalid_argument("cooperative radio snapshot extent mismatch");
}
inline DeviceRadioCounters mergeRadioCounters(DeviceRadioCounters a,
                                               DeviceRadioCounters const& b) {
  auto add=[](unsigned long long x,unsigned long long y) {
    if(y>std::numeric_limits<unsigned long long>::max()-x)
      throw std::overflow_error("cooperative radio counter overflow");
    return x+y;
  };
#define COUNT(f) a.f=add(a.f,b.f)
  COUNT(coreas_contributions); COUNT(zhs_contributions); COUNT(zhs_subtracks);
  COUNT(valid_tracks); COUNT(fixed_point_overflows);
#undef COUNT
  auto sum=[](double x,double y) {
    double z=x+y;
    if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z))
      throw std::overflow_error("nonfinite cooperative radio diagnostics");
    return z;
  };
#define SUM(f) a.f=sum(a.f,b.f)
  SUM(weighted_segment_count); SUM(track_length_m); SUM(weighted_track_length_m);
  SUM(electron_weighted_track_length_m); SUM(positron_weighted_track_length_m);
  SUM(signed_charge_weighted_track_length_m); SUM(energy_weighted_track_length_GeV_m);
  SUM(weighted_direction_change_rad); SUM(weighted_direction_change_squared_rad2);
  SUM(weighted_beta_deficit_track_length_m); SUM(weighted_time_residual_s);
  for(int i=0;i<3;++i) SUM(signed_charge_weighted_direction_change[i]);
  for(int i=0;i<15;++i) SUM(weighted_track_length_by_kinetic_energy_m[i]);
#undef SUM
  // Maxima are not additive. Two finite maxima can have an overflowing sum
  // without their (valid) merged maximum overflowing.
  for(double x:{a.maximum_segment_length_m,b.maximum_segment_length_m,
                a.maximum_direction_change_rad,b.maximum_direction_change_rad})
    if(!std::isfinite(x))
      throw std::overflow_error("nonfinite cooperative radio maximum");
  a.maximum_segment_length_m=std::max(a.maximum_segment_length_m,b.maximum_segment_length_m);
  a.maximum_direction_change_rad=std::max(a.maximum_direction_change_rad,b.maximum_direction_change_rad);
  return a;
}

/** Convert only after the integer ledgers have been merged. This preserves
 * downloadSet's exact inverse-scale expression and SoA order. The existing
 * RadioProcess::endOfShower, not this function, differentiates ZHS potential.
 */
inline gpu::radio::GpuRadioWaveforms decodeFixedRadio(FixedRadioSnapshot const& s) {
  validateFixedRadio(s);
  gpu::radio::GpuRadioWaveforms result;
  auto decode=[&](auto const& observers, auto const& integers, bool zhs) {
    std::vector<gpu::radio::RadioWaveform> out;
    auto bins=radioBins(observers);
    std::size_t offset=0;
    for(auto const& observer:observers) {
      auto count=static_cast<std::size_t>(observer.number_of_bins);
      auto factor=zhs?observer.sample_rate_Hz:1.;
      auto inv=s.config.fixed_point_field_limit_V_per_m/(FixedPointHeadroom*factor);
      gpu::radio::RadioWaveform w;
      w.x.resize(count);w.y.resize(count);w.z.resize(count);
      for(std::size_t i=0;i<count;++i) {
        w.x[i]=static_cast<double>(integers[offset+i])*inv;
        w.y[i]=static_cast<double>(integers[bins+offset+i])*inv;
        w.z[i]=static_cast<double>(integers[2*bins+offset+i])*inv;
      }
      offset+=count;out.push_back(std::move(w));
    }
    return out;
  };
  result.coreas=decode(s.config.coreas_observers,s.coreas,false);
  result.zhs=decode(s.config.zhs_observers,s.zhs,true);
  return result;
}

/** Exactly once per endpoint, all-or-nothing across BOTH algorithms/counters.
 * Final snapshots only: accepting incremental snapshots here would double
 * count tracks. This object must be reset/replaced at each shower boundary.
 */
class CooperativeRadioMerge {
 public:
  CooperativeRadioMerge(std::string identity,gpu::radio::GpuRadioConfig config)
      : identity_(std::move(identity)), config_(std::move(config)) {
    if(identity_.empty()) throw std::invalid_argument("empty cooperative shower identity");
  }
  void commit(FixedRadioSnapshot const& input) {
    if(finished_) throw std::logic_error("cooperative radio output already taken");
    validateFixedRadio(input);
    auto bit=1u<<static_cast<unsigned>(input.endpoint);
    if((seen_&bit)||input.shower_identity!=identity_||!sameRadioPhysics(input.config,config_))
      throw std::invalid_argument("duplicate/incompatible cooperative radio commit");
    auto next=seen_ ? merged_ : input;
    if(seen_) {
      // Config equality above covers each observer's sample rate/scale and
      // propagation model. The layout label also disambiguates field/potential.
      auto merge=[&](auto& dst,auto const& src,char const* algorithm) {
        em::detail::FixedAccumulatorLayout layout{
            identity_+algorithm,config_.fixed_point_field_limit_V_per_m,dst.size()};
        em::detail::mergeFixedAccumulators(dst,layout,src,layout);
      };
      merge(next.coreas,input.coreas,"/coreas-field-soa-v1");
      merge(next.zhs,input.zhs,"/zhs-potential-soa-v1");
      next.counters=mergeRadioCounters(next.counters,input.counters);
    }
    merged_=std::move(next);seen_|=bit;
  }
  FixedRadioSnapshot take() {
    if(seen_!=3 || finished_)
      throw std::logic_error("cooperative radio endpoints not complete/already taken");
    finished_=true;
    return std::move(merged_);
  }
 private:
  std::string identity_;
  gpu::radio::GpuRadioConfig config_;
  FixedRadioSnapshot merged_;
  unsigned seen_{};
  bool finished_{};
};
} // namespace corsika::accelerator::radio::detail
