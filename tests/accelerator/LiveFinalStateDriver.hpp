#pragma once

#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/BremsLpm.hpp>
#include <vector>

namespace migration_audit {
struct Input {
  corsika::gpu::em::EmInteractionRecord interaction{};
  double rho_uniform{}, azimuth_uniform{};
};
struct Output {
  corsika::gpu::em::EmParticleState children[3]{};
  unsigned count{}, error{};
};
std::vector<Output> evaluate(std::vector<Input> const&);
struct RhoOutput { double rho{}; unsigned status{}, trials{}; };
std::vector<RhoOutput> evaluateRho(corsika::gpu::em::BremsLpmSnapshot const&,
    double energy_MeV, double v, unsigned samples, unsigned group);
}
