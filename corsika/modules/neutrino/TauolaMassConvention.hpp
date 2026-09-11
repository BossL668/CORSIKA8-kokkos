// Mountain adapter only. TAUOLA calls remain serialized on the host, as the
// Fortran generator and RNG already require. Never change the stock air module.
#pragma once
#include <Tauola/Tauola.h>
#include <corsika/modules/tauola/TauolaInterfaceParticle.hpp>
#include <array>
#include <stdexcept>

namespace corsika::neutrino {
class ScopedTauolaMassConvention {
 public:
  ScopedTauolaMassConvention() {
    if (!Tauolapp::Tauola::getIsTauolaIni())
      throw std::logic_error("TAUOLA masses must be aligned after initialization");
    auto& m=Tauolapp::parmas_;
    slots_={&m.amtau,&m.amnuta,&m.amell,&m.amnue,&m.ammu,&m.amnumu,
            &m.ampiz,&m.ampi,&m.amk,&m.amkz};
    constexpr std::array<Code,10> codes{Code::TauMinus,Code::NuTau,Code::Electron,
      Code::NuE,Code::MuMinus,Code::NuMu,Code::Pi0,Code::PiPlus,Code::KPlus,Code::K0Long};
    for(std::size_t i=0;i<slots_.size();++i) {
      saved_[i]=*slots_[i];
      *slots_[i]=static_cast<float>(get_mass(codes[i])/1_GeV);
    }
  }
  ~ScopedTauolaMassConvention() noexcept {
    for(std::size_t i=0;i<slots_.size();++i)*slots_[i]=saved_[i];
  }
  ScopedTauolaMassConvention(ScopedTauolaMassConvention const&)=delete;
  ScopedTauolaMassConvention& operator=(ScopedTauolaMassConvention const&)=delete;
 private:
  std::array<float*,10> slots_{};
  std::array<float,10> saved_{};
};
} // namespace corsika::neutrino
