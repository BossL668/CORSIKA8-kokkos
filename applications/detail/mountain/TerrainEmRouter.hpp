// Application-only YAML presentation; transport and region routing live in core.
#pragma once
#include <corsika/modules/transport/InterfaceEmRouter.hpp>
#include <corsika/modules/transport/ResidentInterfaceEmRouter.hpp>
#include <yaml-cpp/yaml.h>
namespace corsika::applications::terrain {
template<class Stack,class Fallback,class Output,class RegionOf>
class EmRouter:public interfaces::InterfaceEmRouter<Stack,Fallback,Output,RegionOf> {
  using Base=interfaces::InterfaceEmRouter<Stack,Fallback,Output,RegionOf>;
 public:
  using Base::Base;
  YAML::Node summary()const {
    auto const& s=this->statistics();YAML::Node n;
    n["execution_space"]=this->session_.executionSpace();n["device_bytes"]=this->session_.deviceBytes();
    n["execution_concurrency"]=this->session_.executionConcurrency();
    n["batches"]=s.batches;n["particles_advanced"]=s.advanced;n["specified_cpu_fallbacks"]=s.fallbacks;
    n["cuts"]=s.cuts;n["world_escapes"]=s.escaped;n["peak_host_front_particles"]=s.peak;
    n["finite_window_survivors"]=s.windowEscaped;n["domain_exits"]=s.domainEscaped;
    n["cpu_direction_roundoff_canonicalizations"]=s.canonicalized;
    n["pending_particles"]=this->pendingParticles();
    n["scope"]="bounded synchronous two-sided EM interface; muons/taus/hadrons/neutrinos CPU; no radio";
    return n;
  }
};

template<class Stack,class Fallback,class Output,class RegionOf>
class ResidentEmRouter:public interfaces::ResidentInterfaceEmRouter<Stack,Fallback,Output,RegionOf> {
  using Base=interfaces::ResidentInterfaceEmRouter<Stack,Fallback,Output,RegionOf>;
 public:
  using Base::Base;
  YAML::Node summary()const {
    auto const& s=this->statistics();
    auto const& r=this->session_.residentStatistics();
    YAML::Node n;
    n["execution_space"]=this->session_.executionSpace();
    n["execution_concurrency"]=this->session_.executionConcurrency();
    n["device_bytes"]=this->session_.deviceBytes();
    n["projected_peak_device_bytes"]=this->session_.projectedPeakDeviceBytes();
    n["batches"]=s.batches;n["particles_advanced"]=s.advanced;
    n["specified_cpu_fallbacks"]=s.fallbacks;n["cuts"]=s.cuts;n["world_escapes"]=s.escaped;
    n["finite_window_survivors"]=s.windowEscaped;n["domain_exits"]=s.domainEscaped;
    n["cpu_direction_roundoff_canonicalizations"]=s.canonicalized;
    n["peak_host_staging_particles"]=s.peak;
    n["pending_particles"]=this->pendingParticles();
    n["resident_capacity"]=this->session_.residentCapacity();
    n["peak_resident_particles"]=r.peak_pending_particles;
    n["cpu_uploaded_particles"]=r.uploaded_particles;
    n["cpu_upload_batches"]=r.upload_batches;
    n["device_enqueued_particles"]=r.device_enqueued_particles;
    n["resident_cascade_calls"]=r.cascade_calls;
    n["maximum_call_wavefronts"]=r.maximum_call_wavefronts;
    n["control_downloads"]=r.control_downloads;
    n["record_downloads"]=r.record_downloads;
    n["downloaded_records"]=r.downloaded_records;
    n["peak_buffered_records"]=r.peak_buffered_records;
    char const* reasons[]{"drained","cpu_fallback","scalar_interleave","record_capacity","wavefront_limit"};
    for(std::size_t i=0;i<r.checkpoint_counts.size();++i)n["checkpoints"][reasons[i]]=r.checkpoint_counts[i];
    for(auto value:r.wavefront_size_log2)n["wavefront_size_floor_log2"].push_back(value);
    n["scope"]="multi-wavefront two-sided resident EM cascade; small front control; bounded output and specified CPU fallback checkpoints; no radio";
    return n;
  }
};
} // namespace corsika::applications::terrain
