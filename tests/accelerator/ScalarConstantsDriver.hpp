#pragma once
#include <corsika/accelerator/em/common/UniformMagneticField.hpp>
#include <vector>

namespace scalar_constants_audit {
struct Input {
  corsika::gpu::em::EmParticleState particle{};
  double mass_GeV{}, charge{}, field_T[3]{}, distance_m{};
};
struct Output {
  corsika::gpu::em::MagneticAdvanceResult advance{};
  corsika::gpu::em::MagneticStepLimitResult limit{};
  double mass_GeV{}, c{}, e{}, epsilon{}, magnetic_factor{}, grammage{};
  double proposal_electron_GeV{}, proposal_muon_GeV{}, muon_lifetime_s{}, maximum_time_s{};
};
std::vector<Output> evaluate(std::vector<Input> const&);
}
