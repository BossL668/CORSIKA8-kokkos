#pragma once
#include <Kokkos_Core.hpp>
#include <corsika/modules/transport/InterfaceEmTypes.hpp>

namespace corsika::interfaces::detail {
KOKKOS_INLINE_FUNCTION bool validSuccessor(em::EmParticleState const& p,
                                          MaterialInterface const& interface) {
  double norm = 0.;
  for (int k=0;k<3;++k) {
    if (!std::isfinite(p.position_m[k]) || !std::isfinite(p.direction[k])) return false;
    norm += p.direction[k]*p.direction[k];
  }
  return (p.pid==22 || p.pid==11 || p.pid==-11) && interface.containsRegion(p.medium_id)
      && p.history_id!=0 && p.step_id!=UINT64_MAX && std::isfinite(p.energy_GeV)
      && p.energy_GeV>=0. && std::isfinite(p.time_s) && std::isfinite(p.weight)
      && p.weight>=0. && std::abs(norm-1.)<=1.e-12;
}
KOKKOS_INLINE_FUNCTION std::uint32_t recordError(EmStep const& r,
                                               MaterialInterface const& interface) {
  if (r.error) return r.error;
  if (r.outcome==EmOutcome::Error || r.child_count>3 ||
      static_cast<std::uint32_t>(r.outcome)>static_cast<std::uint32_t>(EmOutcome::DomainEscape)) return 201;
  if(r.outcome==EmOutcome::DomainEscape&&(!validSuccessor(r.end,interface)||!r.has_track||r.crossed_material||r.child_count||r.domain_edge==UINT32_MAX))return 212;
  if (r.has_track && (!std::isfinite(r.distance_m) || r.distance_m<0. ||
      !std::isfinite(r.grammage_g_cm2) || r.grammage_g_cm2<0.)) return 202;
  if (r.outcome==EmOutcome::Continuation && !validSuccessor(r.end,interface)) return 203;
  if (r.outcome==EmOutcome::Children)
    for (std::uint32_t c=0;c<r.child_count;++c)
      if (!validSuccessor(r.children[c],interface)) return 204;
  return 0;
}
} // namespace corsika::interfaces::detail
