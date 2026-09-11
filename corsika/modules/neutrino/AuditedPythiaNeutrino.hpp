/*
 * Migrated from corsika8-mountain/cpp/AuditedPythiaNeutrino.hpp (2026-09-08).
 * Extended with selected NC vertices and beam/PDF lifetime isolation (2026-09-09).
 * This separate module leaves stock pythia8::NeutrinoInteraction untouched.
 *
 * Audited Pythia8 CC/NC neutrino--nucleon final states.
 *
 * The stock CORSIKA8 implementation samples a proton/neutron for the Pythia
 * beam mass but constructs its COMBoost with targetP4/A.  For a nuclear target
 * those masses are not identical.  This implementation states and enforces a
 * single impulse-approximation convention: one free stationary p/n is sampled
 * from Z/A and the same four-vector is used for sqrt(s), the boost, DIS
 * invariants and the four-momentum audit.  The residual nucleus is outside the
 * inclusive nu-N final-state model.
 */

#pragma once

#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/framework/utility/COMBoost.hpp>
#include <corsika/modules/pythia8/Pythia8.hpp>
#include <corsika/modules/pythia8/Random.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace corsika::neutrino {

using namespace corsika;
using namespace corsika::units::si;

enum class WeakCurrent { CC, NC };
inline char const* currentName(WeakCurrent current) {
  return current == WeakCurrent::CC ? "CC" : "NC";
}

struct GeneratedSecondary {
  Code pid;
  HEPEnergyType kineticEnergy;
  MomentumVector momentum;

  GeneratedSecondary(Code const code, HEPEnergyType const kinetic,
                     MomentumVector const& p)
      : pid(code), kineticEnergy(kinetic), momentum(p) {}
};

struct VertexAudit {
  WeakCurrent current{WeakCurrent::CC};
  int projectilePdg{0};
  int targetNucleusPdg{0};
  int selectedTargetNucleonPdg{0};
  int expectedOutgoingLeptonPdg{0};
  int selectedOutgoingLeptonPdg{0};
  std::size_t outgoingLeptonCandidateCount{0};
  std::size_t hardOutgoingLeptonCandidateCount{0};
  std::size_t finalStateCount{0};
  int initialChargeE{0};
  int finalChargeE{0};
  std::map<int, std::size_t> finalStatePdgCounts;

  // All four-vectors use [E, px, py, pz] in GeV in the shower lab frame.
  std::array<double, 4> projectileP4GeV{};
  std::array<double, 4> selectedTargetP4GeV{};
  std::array<double, 4> initialP4GeV{};
  std::array<double, 4> pythiaRawFinalP4GeV{};
  std::array<double, 4> corsikaOnShellFinalP4GeV{};
  std::array<double, 4> corsikaFinalMinusInitialP4GeV{};
  std::array<double, 4> outgoingLeptonP4GeV{};
  std::array<double, 4> hadronicFinalP4GeV{};
  std::array<double, 4> projectileComP4GeV{};
  std::array<double, 4> selectedTargetComP4GeV{};
  std::array<double, 4> outgoingLeptonComP4GeV{};
  std::array<double, 4> momentumTransferComP4GeV{};

  double sqrtSGeV{0.0};
  double selectedTargetNucleonMassGeV{0.0};
  double legacyNuclearMassPerNucleonGeV{0.0};
  double legacyVsSelectedTargetMassRelativeDifference{0.0};
  double pythiaComEnergyResidualGeV{0.0};
  double pythiaComMomentumResidualGeV{0.0};
  double pythiaComMaxComponentRelativeResidual{0.0};
  double corsikaEnergyRelativeResidual{0.0};
  double corsikaMomentumRelativeResidual{0.0};
  double corsikaMaxComponentRelativeResidual{0.0};

  double inelasticityY{0.0};
  double leptonicLabEnergyY{0.0};
  double invariantMinusLabY{0.0};
  double hadronicEnergyTransferY{0.0};
  double yEnergyBalanceResidual{0.0};
  double q2GeV2{0.0};
  double bjorkenX{0.0};
  double hadronicInvariantMass2GeV2{0.0};
};

struct GeneratedEvent {
  std::vector<GeneratedSecondary> secondaries;
  VertexAudit audit;
};

inline int expectedChargedLeptonPdg(int const neutrinoPdg) {
  int const magnitude = std::abs(neutrinoPdg);
  if (magnitude != 12 && magnitude != 14 && magnitude != 16) {
    throw std::invalid_argument("charged-current projectile is not a neutrino");
  }
  return neutrinoPdg > 0 ? magnitude - 1 : -(magnitude - 1);
}

inline std::array<double, 4> p4GeV(FourMomentum const& p4) {
  auto const components = p4.getSpaceLikeComponents().getComponents();
  return {p4.getTimeLikeComponent() / 1_GeV, components[0] / 1_GeV,
          components[1] / 1_GeV, components[2] / 1_GeV};
}

inline double minkowskiDotGeV2(std::array<double, 4> const& a,
                               std::array<double, 4> const& b) {
  return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
}

class AuditedPythiaNeutrinoFinalState {
 public:
  explicit AuditedPythiaNeutrinoFinalState(
      std::set<Code> const& stableParticles)
      : stableParticles_(stableParticles),
        pythiaStorage_(makeGenerator()) {
  }

  GeneratedEvent generate(Code const projectile, Code const target,
                          FourMomentum const& projectileP4,
                          WeakCurrent const current = WeakCurrent::CC) {
    if (!is_neutrino(projectile)) {
      throw std::invalid_argument("Pythia CC/NC projectile must be a neutrino");
    }
    if (!(is_nucleus(target) || target == Code::Proton ||
          target == Code::Neutron)) {
      throw std::invalid_argument("Pythia CC/NC target must be a nucleus or nucleon");
    }

    auto& rng = RNGManager<>::getInstance().getRandomStream("pythia");
    Code targetNucleon = target;
    if (is_nucleus(target)) {
      double const protonFraction =
          static_cast<double>(get_nucleus_Z(target)) /
          static_cast<double>(get_nucleus_A(target));
      std::bernoulli_distribution chooseProton(protonFraction);
      targetNucleon = chooseProton(rng) ? Code::Proton : Code::Neutron;
    }

    auto const& cs = projectileP4.getSpaceLikeComponents().getCoordinateSystem();
    MomentumVector const targetMomentum(cs, {0_GeV, 0_GeV, 0_GeV});
    FourMomentum const targetNucleonP4{get_mass(targetNucleon), targetMomentum};
    FourMomentum const initialP4 = projectileP4 + targetNucleonP4;
    double const sqrtS = std::sqrt(initialP4.getNormSqr() / (1_GeV * 1_GeV));
    if (!(sqrtS * sqrtS >= 25.0)) {
      throw std::invalid_argument("Pythia CC/NC vertex is below Q2 phase-space threshold");
    }

    // Pythia init() does NOT reset all beam/PDF state. Reusing its point-lepton
    // PDF after a nu -> anti-nu switch can produce a zero cross section.
    // Rebuild only on actual beam identity changes; NEVER reseed the C8 stream.
    std::pair<int,int> const beamKey{static_cast<int>(get_PDG(projectile)),
                                    static_cast<int>(get_PDG(targetNucleon))};
    if (beamKey_ != std::pair<int,int>{0,0} && beamKey_ != beamKey)
      pythiaStorage_ = makeGenerator();
    beamKey_ = beamKey;
    auto& pythia_ = *pythiaStorage_;
    configure(projectile, targetNucleon, sqrtS, current);
    if (!pythia_.next()) {
      throw std::runtime_error("Pythia neutrino collision failed");
    }

    COMBoost const boost{projectileP4, targetNucleonP4};
    auto const& rotatedCS = boost.getRotatedCS();
    Pythia8::Event const& event = pythia_.event;

    GeneratedEvent result;
    auto& audit = result.audit;
    audit.current = current;
    audit.projectilePdg = static_cast<int>(get_PDG(projectile));
    audit.targetNucleusPdg = static_cast<int>(get_PDG(target));
    audit.selectedTargetNucleonPdg = static_cast<int>(get_PDG(targetNucleon));
    audit.expectedOutgoingLeptonPdg = current == WeakCurrent::CC
        ? expectedChargedLeptonPdg(audit.projectilePdg) : audit.projectilePdg;
    audit.initialChargeE = get_charge_number(targetNucleon);
    audit.projectileP4GeV = p4GeV(projectileP4);
    audit.selectedTargetP4GeV = p4GeV(targetNucleonP4);
    audit.initialP4GeV = p4GeV(initialP4);
    audit.sqrtSGeV = sqrtS;
    audit.selectedTargetNucleonMassGeV = get_mass(targetNucleon) / 1_GeV;
    audit.legacyNuclearMassPerNucleonGeV =
        get_mass(target) / 1_GeV /
        (is_nucleus(target) ? static_cast<double>(get_nucleus_A(target)) : 1.0);
    audit.legacyVsSelectedTargetMassRelativeDifference =
        (audit.legacyNuclearMassPerNucleonGeV -
         audit.selectedTargetNucleonMassGeV) /
        audit.selectedTargetNucleonMassGeV;

    MomentumVector rawLabMomentum(cs, {0_GeV, 0_GeV, 0_GeV});
    MomentumVector c8LabMomentum(cs, {0_GeV, 0_GeV, 0_GeV});
    HEPEnergyType rawLabEnergy = HEPEnergyType::zero();
    HEPEnergyType c8LabEnergy = HEPEnergyType::zero();
    std::array<double, 4> pythiaComFinal{};

    bool leptonFound = false;
    HEPEnergyType selectedLeptonEnergy = HEPEnergyType::zero();
    FourMomentum selectedLeptonP4{
        HEPEnergyType::zero(), MomentumVector(cs, {0_GeV, 0_GeV, 0_GeV})};
    FourMomentum selectedLeptonComP4{
        HEPEnergyType::zero(),
        MomentumVector(rotatedCS, {0_GeV, 0_GeV, 0_GeV})};

    // Use the hard-process record for DIS kinematics.  The final event can
    // contain additional same-flavour leptons from hadron decays, and Pythia
    // can give the transported hard lepton a later shower status code even
    // with QED radiation off.
    Pythia8::Event const& process = pythia_.process;
    for (int index = 0; index < process.size(); ++index) {
      auto const& particle = process[index];
      if (particle.id() != audit.expectedOutgoingLeptonPdg ||
          particle.statusAbs() != 23) {
        continue;
      }
      ++audit.hardOutgoingLeptonCandidateCount;
      MomentumVector const pCom(
          rotatedCS, {particle.px() * 1_GeV, particle.py() * 1_GeV,
                      particle.pz() * 1_GeV});
      FourMomentum const rawLabP4 =
          boost.fromCoM(FourMomentum{particle.e() * 1_GeV, pCom});
      MomentumVector const pLab = rawLabP4.getSpaceLikeComponents();
      Code const pid = convert_from_PDG(static_cast<PDGCode>(particle.id()));
      HEPEnergyType const c8TotalEnergy =
          calculate_total_energy(pLab.getNorm(), get_mass(pid));
      if (!leptonFound || c8TotalEnergy > selectedLeptonEnergy) {
        leptonFound = true;
        selectedLeptonEnergy = c8TotalEnergy;
        selectedLeptonP4 = FourMomentum{c8TotalEnergy, pLab};
        selectedLeptonComP4 = FourMomentum{particle.e() * 1_GeV, pCom};
        audit.selectedOutgoingLeptonPdg = static_cast<int>(get_PDG(pid));
      }
    }

    for (int index = 0; index < event.size(); ++index) {
      auto const& particle = event[index];
      if (!particle.isFinal()) continue;

      Code const pid = convert_from_PDG(static_cast<PDGCode>(particle.id()));
      MomentumVector const pCom(
          rotatedCS, {particle.px() * 1_GeV, particle.py() * 1_GeV,
                      particle.pz() * 1_GeV});
      FourMomentum const rawLabP4 =
          boost.fromCoM(FourMomentum{particle.e() * 1_GeV, pCom});
      MomentumVector const pLab = rawLabP4.getSpaceLikeComponents();
      HEPEnergyType const c8TotalEnergy =
          calculate_total_energy(pLab.getNorm(), get_mass(pid));
      HEPEnergyType const kinetic = c8TotalEnergy - get_mass(pid);
      result.secondaries.emplace_back(pid, kinetic, pLab);

      rawLabEnergy += rawLabP4.getTimeLikeComponent();
      rawLabMomentum += pLab;
      c8LabEnergy += c8TotalEnergy;
      c8LabMomentum += pLab;
      pythiaComFinal[0] += particle.e();
      pythiaComFinal[1] += particle.px();
      pythiaComFinal[2] += particle.py();
      pythiaComFinal[3] += particle.pz();
      ++audit.finalStatePdgCounts[static_cast<int>(get_PDG(pid))];
      audit.finalChargeE += get_charge_number(pid);

      if (static_cast<int>(get_PDG(pid)) == audit.expectedOutgoingLeptonPdg) {
        ++audit.outgoingLeptonCandidateCount;
      }
    }

    if (!leptonFound) {
      throw std::runtime_error(
          "Pythia CC process record has no status-23 outgoing charged lepton");
    }

    audit.finalStateCount = result.secondaries.size();
    FourMomentum const rawFinalP4{rawLabEnergy, rawLabMomentum};
    FourMomentum const c8FinalP4{c8LabEnergy, c8LabMomentum};
    FourMomentum const residualP4 = c8FinalP4 - initialP4;
    FourMomentum const hadronicP4 = c8FinalP4 - selectedLeptonP4;
    audit.pythiaRawFinalP4GeV = p4GeV(rawFinalP4);
    audit.corsikaOnShellFinalP4GeV = p4GeV(c8FinalP4);
    audit.corsikaFinalMinusInitialP4GeV = p4GeV(residualP4);
    audit.outgoingLeptonP4GeV = p4GeV(selectedLeptonP4);
    audit.hadronicFinalP4GeV = p4GeV(hadronicP4);
    FourMomentum const projectileComP4 = boost.toCoM(projectileP4);
    FourMomentum const targetComP4 = boost.toCoM(targetNucleonP4);
    FourMomentum const qComP4 = projectileComP4 - selectedLeptonComP4;
    audit.projectileComP4GeV = p4GeV(projectileComP4);
    audit.selectedTargetComP4GeV = p4GeV(targetComP4);
    audit.outgoingLeptonComP4GeV = p4GeV(selectedLeptonComP4);
    audit.momentumTransferComP4GeV = p4GeV(qComP4);

    double const scale = std::max(std::abs(audit.initialP4GeV[0]), 1.0);
    audit.pythiaComEnergyResidualGeV = pythiaComFinal[0] - sqrtS;
    audit.pythiaComMomentumResidualGeV = std::sqrt(
        pythiaComFinal[1] * pythiaComFinal[1] +
        pythiaComFinal[2] * pythiaComFinal[2] +
        pythiaComFinal[3] * pythiaComFinal[3]);
    audit.pythiaComMaxComponentRelativeResidual =
        std::max({std::abs(audit.pythiaComEnergyResidualGeV),
                  std::abs(pythiaComFinal[1]), std::abs(pythiaComFinal[2]),
                  std::abs(pythiaComFinal[3])}) /
        std::max(sqrtS, 1.0);
    audit.corsikaEnergyRelativeResidual =
        std::abs(audit.corsikaFinalMinusInitialP4GeV[0]) / scale;
    audit.corsikaMomentumRelativeResidual =
        residualP4.getSpaceLikeComponents().getNorm() / 1_GeV / scale;
    audit.corsikaMaxComponentRelativeResidual =
        std::max({std::abs(audit.corsikaFinalMinusInitialP4GeV[0]),
                  std::abs(audit.corsikaFinalMinusInitialP4GeV[1]),
                  std::abs(audit.corsikaFinalMinusInitialP4GeV[2]),
                  std::abs(audit.corsikaFinalMinusInitialP4GeV[3])}) /
        scale;

    double const neutrinoEnergy = audit.projectileP4GeV[0];
    double const transferEnergy = neutrinoEnergy - audit.outgoingLeptonP4GeV[0];
    audit.leptonicLabEnergyY = transferEnergy / neutrinoEnergy;
    double const pDotQ =
        minkowskiDotGeV2(audit.selectedTargetComP4GeV,
                         audit.momentumTransferComP4GeV);
    double const pDotK =
        minkowskiDotGeV2(audit.selectedTargetComP4GeV,
                         audit.projectileComP4GeV);
    audit.inelasticityY = pDotQ / pDotK;
    audit.invariantMinusLabY =
        audit.inelasticityY - audit.leptonicLabEnergyY;
    audit.hadronicEnergyTransferY =
        (audit.hadronicFinalP4GeV[0] - audit.selectedTargetP4GeV[0]) /
        neutrinoEnergy;
    audit.yEnergyBalanceResidual =
        audit.hadronicEnergyTransferY - audit.inelasticityY;
    audit.q2GeV2 =
        -minkowskiDotGeV2(audit.momentumTransferComP4GeV,
                          audit.momentumTransferComP4GeV);
    audit.bjorkenX = audit.q2GeV2 / (2.0 * pDotQ);
    std::array<double, 4> hadronicSystemCom{};
    for (std::size_t index = 0; index < 4; ++index) {
      hadronicSystemCom[index] = audit.selectedTargetComP4GeV[index] +
                                 audit.momentumTransferComP4GeV[index];
    }
    audit.hadronicInvariantMass2GeV2 =
        minkowskiDotGeV2(hadronicSystemCom, hadronicSystemCom);
    return result;
  }

 private:
  static std::unique_ptr<Pythia8::Pythia> makeGenerator() {
    auto result = std::make_unique<Pythia8::Pythia>(
        corsika::pythia8::CORSIKA_Pythia8_XML_DIR,false);
    result->setRndmEnginePtr(std::make_shared<corsika::pythia8::Random>());
    return result;
  }
  void configure(Code const projectile, Code const targetNucleon,
                 double const sqrtSGeV, WeakCurrent const current) {
    auto& pythia_ = *pythiaStorage_;
    pythia_.readString("Beams:frameType = 1");
    pythia_.settings.parm("Beams:eCM", sqrtSGeV);
    pythia_.settings.mode("Beams:idA", static_cast<int>(get_PDG(projectile)));
    pythia_.settings.mode("Beams:idB", static_cast<int>(get_PDG(targetNucleon)));
    pythia_.settings.flag("WeakBosonExchange:ff2ff(t:gmZ)", current == WeakCurrent::NC);
    pythia_.settings.flag("WeakBosonExchange:ff2ff(t:W)", current == WeakCurrent::CC);
    pythia_.settings.parm("PhaseSpace:Q2Min", 25.0);
    pythia_.readString("SpaceShower:dipoleRecoil = on");
    pythia_.readString("SpaceShower:pTmaxMatch = 2");
    pythia_.readString("PDF:lepton = off");
    pythia_.readString("TimeShower:QEDshowerByL = off");
    if (!stableParticles_.empty()) {
      pythia_.readString("HadronLevel:Decay = on");
      for (Code const code : stableParticles_) {
        pythia_.particleData.mayDecay(static_cast<int>(get_PDG(code)), false);
      }
    } else {
      pythia_.readString("HadronLevel:Decay = off");
    }
    pythia_.readString("Stat:showProcessLevel = off");
    pythia_.readString("Stat:showPartonLevel = off");
    pythia_.readString("Print:quiet = on");
    pythia_.readString("Check:epTolErr = 0.1");
    pythia_.readString("Check:epTolWarn = 0.0001");
    pythia_.readString("Check:mTolErr = 0.01");
    if (!pythia_.init()) {
      pythia_.logger.errorStatistics();
      throw std::runtime_error("Pythia neutrino initialization failed");
    }
  }

  std::set<Code> stableParticles_;
  std::unique_ptr<Pythia8::Pythia> pythiaStorage_;
  std::pair<int,int> beamKey_{0,0};
};

}  // namespace corsika::neutrino
