/* Verify every active native electromagnetic channel survives device export/routing. */
#include <corsika/accelerator/em/common/ProcessCapabilities.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTableExporter.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>
#include <PROPOSAL/PROPOSAL.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

using namespace corsika;
namespace em = corsika::gpu::em;
namespace nt = em::tables;
void require(bool ok, char const* message) { if (!ok) throw std::runtime_error(message); }

int main(int argc, char** argv) { try {
  if (argc > 1) PROPOSAL::InterpolationSettings::TABLES_PATH = argv[1];
  PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::err);
  logging::set_level(logging::level::warn);
  PROPOSAL::Air medium;
  std::size_t channels = 0, checks = 0;
  double max_relative_rate = 0.;
  std::map<Code,std::unique_ptr<PROPOSAL::Interaction>> interactions;
  std::map<Code,std::unique_ptr<PROPOSAL::Displacement>> displacements;
  std::vector<proposal::NativeInteractionCalculatorView> views;
  std::vector<proposal::NativeContinuousCalculatorView> continuous;
  for (auto code : {Code::Photon, Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus}) {
    auto cross = proposal::make_cross_sections(code, medium, .5_MeV, true);
    interactions[code] = PROPOSAL::make_interaction(cross, true, true);
    views.push_back({code, medium.GetHash(), &medium, interactions.at(code).get(),
                    nullptr, nullptr, .5_MeV, proposal::particle.at(code).mass});
    if (code != Code::Photon) {
      displacements[code] = PROPOSAL::make_displacement(cross, true);
      continuous.push_back({code, medium.GetHash(), &medium,
                            displacements.at(code).get(), .5_MeV,
                            proposal::particle.at(code).mass});
    }
  }
  // The exporter validates both charges together, as in the application.
  auto table = nt::exportProposalNativeTables(views, continuous);
  auto view = nt::makeProposalNativeHostView(table);
  for (auto code : {Code::Photon, Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus}) {
    int const pid = static_cast<int>(get_PDG(code));
    auto const& interaction = interactions.at(code);
    std::set<int> native, exported;
    std::map<int,bool> nonzero;
    for (auto const& c : interaction->GetCrossSections()) native.insert(int(c->GetInteractionType()));
    for (auto const& c : table.dndx_columns) if (c.pdg_id == pid) exported.insert(c.process_id);
    require(native == exported, "native process lost or extra process inserted during export");
    require(native.size() == (pid == 22 || pid == -11 ? 5U : 4U), "unexpected native process inventory");
    for (int process : native) {
      auto const capability = em::gpuProcessCapability(pid,process);
      require(capability != em::GpuProcessCapability::GpuFinalStateNotImplemented,
              "active native process has no supported device/CPU completion");
      if (process == em::ElectronPairProcessId)
        require(capability == em::GpuProcessCapability::CpuOnlyFinalState, "direct pair is not CPU-completed");
      nonzero[process] = false;
      std::cout << "CHANNEL pid=" << pid << " process=" << process << " name="
                << em::gpuEmProcessName(process) << " completion="
                << (capability == em::GpuProcessCapability::CpuOnlyFinalState ? "CPU" : "device") << '\n';
    }
    for (double E : {1.e4, 1.e5, 1.e6, 1.e8, 1.e11}) {
      for (auto const& rate : interaction->Rates(E)) {
        int const process = int(rate.crosssection->GetInteractionType());
        auto query = nt::queryProposalNativeRate(view,pid,process,rate.comp_hash,E);
        require(query.status == nt::NativeQueryStatus::Success, "native rate column query failed");
        double const scale = std::max({std::abs(rate.rate),std::abs(query.value),1.e-300});
        double const error = std::abs(rate.rate-query.value)/scale;
        require(std::isfinite(query.value) && error < 2.e-10, "exported rate differs from native C8");
        max_relative_rate = std::max(max_relative_rate,error);
        nonzero[process] = nonzero[process] || rate.rate > 0.;
        ++checks;
      }
    }
    for (auto const& entry : nonzero) require(entry.second, "channel never exercised above threshold");
    channels += native.size();
  }
  std::cout << "PASS native_channels=" << channels << " rate_checks=" << checks
            << " max_relative_rate=" << max_relative_rate << '\n';
  return 0;
} catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; } }
