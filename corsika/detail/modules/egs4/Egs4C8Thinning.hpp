#pragma once
#include <corsika/detail/modules/egs4/Egs4C8Air.hpp>
#include <corsika/accelerator/em/common/EmThinning.hpp>

namespace c7_egs4::c8_adapter {
using ThinningConfig=::corsika::gpu::em::EmThinningConfig;
using ThinningResult=::corsika::gpu::em::EmThinningResult;
using ThinningStatus=::corsika::gpu::em::EmThinningStatus;
inline constexpr std::uint32_t NativeEgs4ThinningDomain=0x43475448U;
// Only ordinary two-EM-child vertices, exactly the C8 EMThinning scope.
// Cut positron annihilation (channel invalid) and nuclear decays are NOT
// silently thinned here. Output for the preceding segment keeps parent weight.
KOKKOS_INLINE_FUNCTION ThinningResult thinAirVertex(ThinningConfig config,
    AirRecord const& parent,AirOutcome const& result) {
  ThinningResult keep{ThinningStatus::NotApplied,3,parent.metadata.weight,parent.metadata.weight};
  if(!config.enabled||result.action!=AirAction::replaced||result.history.child_count!=2||
     result.history.channel==Channel::invalid)return keep;
  // Each physical vertex consumes EM draws. Its final draw counter is a
  // collision nonce in a separate domain, not another EM draw. No fixed draw
  // offset can collide with an arbitrarily long rejection-sampling stream.
  auto key=parent.random.key;key.process_id=NativeEgs4ThinningDomain;
  key.step_id=key.draw_id;key.draw_id=0;
  double u=::corsika::gpu::em::uniformOpen01(key);++key.draw_id;
  double v=::corsika::gpu::em::uniformOpen01(key);
  return ::corsika::gpu::em::applyEmThinning(config,parent.track.particle.energy_MeV*.001,
    parent.metadata.weight,result.children[0].particle.energy_MeV*.001,
    result.children[1].particle.energy_MeV*.001,u,v);
}
inline void validateThinning(ThinningConfig const& c) {
  if(c.enabled&&(!std::isfinite(c.threshold_GeV)||c.threshold_GeV<=0.||
     !std::isfinite(c.maximum_weight)||c.maximum_weight<=0.||!c.erase_zero_weight))
    throw std::invalid_argument("Native EGS4 thinning requires positive finite threshold/weight and removal of zero-weight histories; multithinning is not supported");
}
} // namespace c7_egs4::c8_adapter
