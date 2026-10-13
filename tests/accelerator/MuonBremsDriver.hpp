/* Muon bremsstrahlung and CPU pair-routing validation. BSD-3-Clause. */
#pragma once
#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>
#include <corsika/accelerator/em/common/BremsLpm.hpp>
#include <corsika/accelerator/em/common/ProposalFallback.hpp>
#include <vector>
namespace muon_test {
namespace em = corsika::gpu::em;
struct Input {
  int pid{13}, process{};
  double energy{}, v{}, density{1.e-3}, quantile{.5}; // MeV
  std::uint64_t component{}, history{};
  unsigned fault{}, thinning{};
};
struct Output {
  em::ProposalFallbackEvent fallback_event{};
  unsigned fallback{}, error{}, count{}, suppressed{}, trials{}, replay{}, loss_status{};
  double rho{}, lpm{}, azimuth{}, sign{}, loss{}, lpm_uniform{};
  double energies[3]{}, directions[3][3]{}, weights[3]{};
  int pids[3]{};
};
void initialize();
void finalize();
std::vector<Output> run(em::tables::ProposalNativeTableSet const&,
    em::BremsLpmSnapshot const&, std::vector<Input> const&, bool on_device=true);
}
