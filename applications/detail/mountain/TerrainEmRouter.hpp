// Application-only YAML presentation; transport and region routing live in core.
#pragma once
#include <corsika/modules/transport/InterfaceEmRouter.hpp>
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
    n["batches"]=s.batches;n["particles_advanced"]=s.advanced;n["specified_cpu_fallbacks"]=s.fallbacks;
    n["cuts"]=s.cuts;n["world_escapes"]=s.escaped;n["peak_host_front_particles"]=s.peak;
    n["finite_window_survivors"]=s.windowEscaped;
    n["cpu_direction_roundoff_canonicalizations"]=s.canonicalized;
    n["pending_particles"]=this->pendingParticles();
    n["scope"]="bounded synchronous two-sided EM interface; muons/taus/hadrons/neutrinos CPU; no radio";
    return n;
  }
};
} // namespace corsika::applications::terrain
