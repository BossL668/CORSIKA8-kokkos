/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "AuxiliaryModelFixture.hpp"
#include "AuxiliaryModelConversion.hpp"
#include "AuxiliaryExportReference.hpp"
#include <corsika/kokkos/em/proposal/ProposalNativeAux.hpp>
#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/crosssection/parametrization/Bremsstrahlung.h>
#include <PROPOSAL/crosssection/parametrization/PhotoPairProduction.h>
#include <PROPOSAL/medium/Medium.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace corsika::kokkos::test {
  namespace {
    namespace original = auxiliary_reference::tables;
    using namespace corsika::units::si;
    using view_type = corsika::proposal::NativeInteractionCalculatorView;

    PROPOSAL::ParticleDef native_particle(Code code) {
      // The exact entries in C8's proposal::particle map, without instantiating
      // unrelated cross-section factories and interpolation-table machinery.
      switch (code) {
        case Code::Photon:
          return PROPOSAL::GammaDef();
        case Code::Electron:
          return PROPOSAL::EMinusDef();
        case Code::Positron:
          return PROPOSAL::EPlusDef();
        case Code::MuMinus:
          return PROPOSAL::MuMinusDef();
        case Code::MuPlus:
          return PROPOSAL::MuPlusDef();
        default:
          throw std::invalid_argument("Unsupported auxiliary fixture projectile");
      }
    }

    struct Owner {
      Code code_;
      PROPOSAL::Medium const& medium_;
      PROPOSAL::ParticleDef particle_;
      std::unique_ptr<PROPOSAL::crosssection::PhotoPairLPM> photon_;
      std::unique_ptr<PROPOSAL::crosssection::BremsLPM> brems_;

      Owner(Code code, PROPOSAL::Medium const& medium)
          : code_(code)
          , medium_(medium)
          , particle_(native_particle(code)) {
        if (code == Code::Photon) {
          photon_ = std::make_unique<PROPOSAL::crosssection::PhotoPairLPM>(
              particle_, medium, PROPOSAL::crosssection::PhotoPairKochMotz());
        } else {
          // The production LPM helper uses ElectronScreening also for muons;
          // it is distinct from their stochastic-rate parametrization.
          brems_ = std::make_unique<PROPOSAL::crosssection::BremsLPM>(
              particle_, medium, PROPOSAL::crosssection::BremsElectronScreening());
        }
      }

      view_type view(bool helpers = true, HEPEnergyType cut = .4_MeV) const {
        view_type::photon_pair_reference_type photon;
        view_type::brems_reference_type brems;
        if (helpers && photon_) photon = std::cref(*photon_);
        if (helpers && brems_) brems = std::cref(*brems_);
        // Aux only borrows these real helpers; it never uses an Interaction/CDF.
        return {code_, medium_.GetHash(), std::cref(medium_), std::nullopt, photon, brems,
                cut,   particle_.mass};
      }

      std::vector<double> ordinary() const {
        auto const exported =
            photon_ ? photon_->ExportParameters() : brems_->ExportParameters();
        std::vector<double> result{
            exported.particle_mass_MeV, exported.molecular_density_per_cm3,
            exported.mass_density_g_per_cm3, exported.sum_charge, exported.e_lpm_MeV};
        for (auto const& component : medium_.GetComponents()) {
          result.push_back(photon_ ? photon_->suppression_factor(1.e9, .3, component, .7)
                                   : brems_->suppression_factor(1.e9, .3, component, .7));
        }
        return result;
      }
    };

    struct Helpers {
      PROPOSAL::Air medium_;
      Owner photon_{Code::Photon, medium_};
      Owner electron_{Code::Electron, medium_};
      Owner positron_{Code::Positron, medium_};
      Owner minus_{Code::MuMinus, medium_};
      Owner plus_{Code::MuPlus, medium_};
      std::vector<view_type> views() const {
        return {photon_.view(), electron_.view(), positron_.view(), minus_.view(),
                plus_.view()};
      }
    };

    std::filesystem::path fresh_directory(char const* label) {
      char const* base = std::getenv("CORSIKA_AUXILIARY_TEST_CACHE");
      if (!base || !*base)
        throw std::runtime_error(
            "Set CORSIKA_AUXILIARY_TEST_CACHE to a dedicated test directory");
      auto directory =
          std::filesystem::path(base) /
          (std::string(label) + "-" + std::to_string(getpid()) + "-" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
      if (!std::filesystem::create_directories(directory))
        throw std::runtime_error("Auxiliary test directory already exists");
      return directory;
    }

    std::vector<std::uint8_t> read_file(std::filesystem::path const& path) {
      std::ifstream stream(path, std::ios::binary);
      if (!stream) throw std::runtime_error("Cannot read auxiliary test cache");
      return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    }

    void write_file(std::filesystem::path const& path,
                    std::vector<std::uint8_t> const& bytes) {
      std::filesystem::create_directories(path.parent_path());
      std::ofstream stream(path, std::ios::binary | std::ios::trunc);
      stream.write(reinterpret_cast<char const*>(bytes.data()), bytes.size());
      if (!stream)
        throw std::runtime_error("Cannot write dedicated auxiliary test cache");
    }

    template <typename TValue>
    bool same_bits(TValue const& first, TValue const& second) {
      static_assert(std::is_trivially_copyable_v<TValue>);
      return std::memcmp(&first, &second, sizeof(TValue)) == 0;
    }

    bool same_data(ProposalNativeAuxData const& actual,
                   original::ProposalNativeAuxData const& expected) {
      // All numerical records have complete field/offset checks in Conversion.hpp.
      return actual.formatVersion_ == expected.format_version &&
             same_bits(to_reference(actual.photonPairLpm_), expected.photon_pair_lpm) &&
             same_bits(to_reference(actual.bremsLpm_), expected.brems_lpm) &&
             same_bits(to_reference(actual.electronMoliere_),
                       expected.electron_moliere) &&
             same_bits(to_reference(actual.muonMoliere_), expected.muon_moliere) &&
             actual.hasMuonMoliere_ == expected.has_muon_moliere &&
             actual.keyHash_ == expected.key_hash &&
             actual.contentHash_ == expected.content_hash;
    }

    AuxiliarySnapshots snapshots(ProposalNativeAuxData const& aux) {
      AuxiliarySnapshots result{aux.photonPairLpm_, aux.bremsLpm_, aux.bremsLpm_,
                                aux.electronMoliere_, aux.muonMoliere_};
      // Exact production selection of the independently exported muon state.
      result.muon_.leptonMassMeV_ = aux.bremsLpm_.muonMassMeV_;
      result.muon_.baselineMassDensityGPerCm3_ =
          aux.bremsLpm_.muonBaselineMassDensityGPerCm3_;
      result.muon_.molecularDensityPerCm3_ = aux.bremsLpm_.muonMolecularDensityPerCm3_;
      result.muon_.sumCharge_ = aux.bremsLpm_.muonSumCharge_;
      result.muon_.eLpmMeV_ = aux.bremsLpm_.muonELpmMeV_;
      return result;
    }

    PROPOSAL::Medium modified_air(PROPOSAL::Medium const& air, double densityScale,
                                  double excitationScale = 1.) {
      return {"auxiliary-test-air",
              air.GetI() * excitationScale,
              air.GetC(),
              air.GetA(),
              air.GetM(),
              air.GetX0(),
              air.GetX1(),
              air.GetD0(),
              air.GetMassDensity() * densityScale,
              air.GetComponents()};
    }

    template <typename TFunction>
    bool rejects(TFunction function) {
      try {
        function();
      } catch (std::invalid_argument const&) { return true; }
      return false;
    }

    struct CacheHeader {
      char magic_[8];
      std::uint32_t version_;
      std::uint32_t bytes_;
      sha256_digest_type key_;
      sha256_digest_type content_;
    };
    static_assert(sizeof(CacheHeader) == 80);

    template <typename TValue>
    void overwrite(std::vector<std::uint8_t>& bytes, std::size_t offset,
                   TValue const& value) {
      if (offset + sizeof(value) > bytes.size())
        throw std::runtime_error("Invalid test corruption offset");
      std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    void rehash_payload(std::vector<std::uint8_t>& bytes) {
      auto const size = static_cast<std::uint32_t>(bytes.size() - sizeof(CacheHeader));
      overwrite(bytes, offsetof(CacheHeader, bytes_), size);
      auto const hash = sha256(
          std::vector<std::uint8_t>(bytes.begin() + sizeof(CacheHeader), bytes.end()));
      overwrite(bytes, offsetof(CacheHeader, content_), hash);
    }

    bool no_temporary_files(std::filesystem::path const& directory) {
      for (auto const& entry : std::filesystem::directory_iterator(directory))
        if (entry.path().filename().string().find(".tmp.") != std::string::npos)
          return false;
      return true;
    }
  } // namespace

  AuxiliaryAudit audit_auxiliary_models() {
    Helpers const helpers;
    auto const views = helpers.views();
    auto const directory = fresh_directory("cache-audit");
    AuxiliaryAudit audit;
    auto check = [&](std::string name, bool result) {
      audit.checks_.emplace_back(std::move(name), result);
    };
    auto const beforePhoton = helpers.photon_.ordinary();
    auto const beforeElectron = helpers.electron_.ordinary();
    auto const beforeMuon = helpers.minus_.ordinary();
    auto const expected = original::load_reference_aux(views, directory / "old");
    auto const actual = load_or_create_proposal_native_aux(views, directory / "new");
    check("Independent cold export agrees for every numerical record and hash",
          same_data(actual, expected));
    check("Both initial cache writes are cold", !actual.cacheHit_ && !expected.cache_hit);
    check("Same-ABI v4 serialized bytes are unchanged",
          read_file(actual.cacheFile_) == read_file(expected.cache_file));
    check("New exporter reads the old writer",
          load_or_create_proposal_native_aux(views, directory / "old").cacheHit_);
    check("Old exporter reads the new writer",
          original::load_reference_aux(views, directory / "new").cache_hit);
    auto const warm = load_or_create_proposal_native_aux(views, directory / "new");
    check("Warm key/content remain exact", warm.cacheHit_ && same_data(warm, expected));
    check("Native photon helper remains unchanged",
          beforePhoton == helpers.photon_.ordinary());
    check("Native electron helper remains unchanged",
          beforeElectron == helpers.electron_.ordinary());
    check("Native muon helper remains unchanged",
          beforeMuon == helpers.minus_.ordinary());
    auto const muon = helpers.minus_.brems_->ExportParameters();
    check("Muon eLpm comes from the actual muon helper",
          actual.bremsLpm_.muonELpmMeV_ == muon.e_lpm_MeV);
    check("Muon mass comes from the actual helper",
          actual.bremsLpm_.muonMassMeV_ == muon.particle_mass_MeV);
    check("Distinct muon/electron Moliere masses retained",
          actual.hasMuonMoliere_ == 1u &&
              actual.muonMoliere_.particleMassMeV_ == PROPOSAL::MMU &&
              actual.electronMoliere_.particleMassMeV_ == PROPOSAL::ME);
    check("Electron signs have equal live state",
          helpers.electron_.ordinary() == helpers.positron_.ordinary());
    check("Muon signs have equal live state",
          helpers.minus_.ordinary() == helpers.plus_.ordinary());

    auto reordered = views;
    std::reverse(reordered.begin(), reordered.end());
    reordered.push_back(views.front());
    reordered.emplace_back(Code::Proton, 0, std::nullopt, std::nullopt, std::nullopt,
                           std::nullopt, 0_MeV, 0.);
    auto const reorderResult =
        load_or_create_proposal_native_aux(reordered, directory / "new");
    check("Order duplicates and unsupported projectiles do not change cache identity",
          reorderResult.cacheHit_ && reorderResult.keyHash_ == actual.keyHash_);
    auto changedCut = views;
    changedCut[1] = helpers.electron_.view(true, .5_MeV);
    auto const changed =
        load_or_create_proposal_native_aux(changedCut, directory / "new");
    check("Stochastic cut changes the cache key",
          changed.keyHash_ != actual.keyHash_ && !changed.cacheHit_);
    check("Cut changes do not change the auxiliary physics payload",
          changed.contentHash_ == actual.contentHash_);

    auto noHelpers = views;
    for (std::size_t index = 0; index < noHelpers.size(); ++index) {
      auto const& input = views[index];
      noHelpers[index] =
          view_type(input.getProjectile(), input.getMediumHash(), input.getMedium(),
                    std::nullopt, std::nullopt, std::nullopt,
                    input.getStochasticEnergyCut(), input.getParticleMassMeV());
    }
    check("Valid hot cache precedes helper validation",
          load_or_create_proposal_native_aux(noHelpers, directory / "new").cacheHit_);
    check("Missing helper cold input is rejected", rejects([&] {
            (void)load_or_create_proposal_native_aux(noHelpers, directory / "missing");
          }));
    check("Old missing-helper cold behavior agrees", rejects([&] {
            original::load_reference_aux(noHelpers, directory / "missing-old");
          }));
    auto missingMuon = views;
    missingMuon[3] = helpers.minus_.view(false);
    missingMuon[4] = helpers.plus_.view(false);
    check("Muon state is never inferred from the electron helper", rejects([&] {
            (void)load_or_create_proposal_native_aux(missingMuon,
                                                     directory / "missing-muon");
          }));
    missingMuon[3] =
        view_type(Code::MuMinus, helpers.medium_.GetHash(), std::cref(helpers.medium_),
                  std::nullopt, std::nullopt, std::cref(*helpers.electron_.brems_),
                  .4_MeV, PROPOSAL::MMU);
    check("Incorrect actual muon helper mass is rejected", rejects([&] {
            (void)load_or_create_proposal_native_aux(missingMuon,
                                                     directory / "wrong-muon");
          }));
    check("Empty views are rejected", rejects([&] {
            (void)load_or_create_proposal_native_aux({}, directory / "empty");
          }));
    auto nullMedium = views;
    nullMedium[0] = view_type(Code::Photon, 0, std::nullopt, std::nullopt, std::nullopt,
                              std::nullopt, 0_MeV, 0.);
    check("Supported projectile without a medium is rejected", rejects([&] {
            (void)load_or_create_proposal_native_aux(nullMedium, directory / "null");
          }));

    auto noMuon = views;
    noMuon.resize(3);
    auto const withoutMuon =
        load_or_create_proposal_native_aux(noMuon, directory / "no-muon");
    auto const oldWithoutMuon =
        original::load_reference_aux(noMuon, directory / "no-muon-old");
    check("No-muon snapshot preserves the original absence contract",
          withoutMuon.hasMuonMoliere_ == 0 && same_data(withoutMuon, oldWithoutMuon));
    auto const thin = modified_air(helpers.medium_, .3);
    Owner const thinPhoton(Code::Photon, thin), thinElectron(Code::Electron, thin),
        thinMuon(Code::MuMinus, thin);
    auto layers = views;
    layers.push_back(thinPhoton.view());
    layers.push_back(thinElectron.view());
    layers.push_back(thinMuon.view());
    auto const layered = load_or_create_proposal_native_aux(layers, directory / "layers");
    auto const oldLayered =
        original::load_reference_aux(layers, directory / "layers-old");
    check("Density-only layers preserve old independent helper selection",
          same_data(layered, oldLayered));
    check("Additional density changes the key", layered.keyHash_ != actual.keyHash_);
    std::reverse(layers.begin(), layers.end());
    check("Density-layer order does not change the payload",
          same_data(
              load_or_create_proposal_native_aux(layers, directory / "layers-reversed"),
              oldLayered));
    auto const changedMaterial = modified_air(helpers.medium_, 1., 1.01);
    Owner const otherPhoton(Code::Photon, changedMaterial);
    auto incompatible = views;
    incompatible.push_back(otherPhoton.view());
    check("Non-density material changes are rejected", rejects([&] {
            (void)load_or_create_proposal_native_aux(incompatible,
                                                     directory / "material");
          }));
    PROPOSAL::Water const water;
    Owner const waterPhoton(Code::Photon, water);
    incompatible = views;
    incompatible.push_back(waterPhoton.view());
    check("Mixed chemical composition is rejected", rejects([&] {
            (void)load_or_create_proposal_native_aux(incompatible,
                                                     directory / "composition");
          }));

    auto const originalBytes = read_file(actual.cacheFile_);
    for (unsigned fault = 0; fault < 14; ++fault) {
      auto bytes = originalBytes;
      auto const headerSize = sizeof(CacheHeader);
      auto const photonSize = sizeof(PhotonPairLpmSnapshot);
      auto const bremsSize = sizeof(BremsLpmSnapshot);
      auto const moliereSize = sizeof(MoliereSnapshot);
      switch (fault) {
        case 0:
          bytes.resize(7);
          break;
        case 1:
          bytes.pop_back();
          break;
        case 2:
          bytes.push_back(0);
          break;
        case 3:
          bytes[0] ^= 1;
          break;
        case 4:
          overwrite(bytes, offsetof(CacheHeader, version_), std::uint32_t{3});
          break;
        case 5:
          bytes[offsetof(CacheHeader, key_)] ^= 1;
          break;
        case 6:
          bytes[offsetof(CacheHeader, content_)] ^= 1;
          break;
        case 7:
          overwrite(bytes, offsetof(CacheHeader, bytes_), std::uint32_t{0});
          break;
        case 8:
          overwrite(bytes, offsetof(CacheHeader, bytes_), std::uint32_t{1u << 29});
          break;
        case 9:
          overwrite(
              bytes,
              headerSize + offsetof(PhotonPairLpmSnapshot, baselineMassDensityGPerCm3_),
              std::numeric_limits<double>::quiet_NaN());
          rehash_payload(bytes);
          break;
        case 10:
          overwrite(bytes, headerSize + offsetof(PhotonPairLpmSnapshot, reserved_),
                    std::uint32_t{1});
          rehash_payload(bytes);
          break;
        case 11:
          overwrite(bytes, headerSize + photonSize + bremsSize + 2 * moliereSize,
                    std::uint32_t{2});
          rehash_payload(bytes);
          break;
        case 12:
          overwrite(bytes,
                    headerSize + photonSize + bremsSize + offsetof(MoliereSnapshot, c1_),
                    std::numeric_limits<double>::quiet_NaN());
          rehash_payload(bytes);
          break;
        case 13:
          bytes.push_back(0);
          rehash_payload(bytes);
          break;
      }
      auto const corruptDirectory = directory / ("corrupt-" + std::to_string(fault));
      auto const path = corruptDirectory / actual.cacheFile_.filename();
      write_file(path, bytes);
      auto const repaired = load_or_create_proposal_native_aux(views, corruptDirectory);
      check("Corrupt cache is rebuilt: " + std::to_string(fault),
            !repaired.cacheHit_ && same_data(repaired, expected) &&
                read_file(path) == originalBytes);
      check("Repaired file is readable by fixed production: " + std::to_string(fault),
            original::load_reference_aux(views, corruptDirectory).cache_hit &&
                no_temporary_files(corruptDirectory));
      ++audit.corruptionCases_;
    }

    auto const concurrentDirectory = directory / "concurrent";
    std::array<pid_t, 2> children{};
    for (auto& child : children) {
      child = fork();
      if (child == -1) throw std::runtime_error("Cannot fork auxiliary lock test");
      if (child == 0) {
        try {
          (void)load_or_create_proposal_native_aux(views, concurrentDirectory);
          _exit(0);
        } catch (...) { _exit(1); }
      }
    }
    for (auto child : children) {
      int status{};
      if (waitpid(child, &status, 0) != child)
        throw std::runtime_error("Cannot wait for auxiliary lock test");
      check("Concurrent cache writer exited successfully",
            WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    auto const concurrent =
        load_or_create_proposal_native_aux(views, concurrentDirectory);
    check("Concurrent publication is valid warm and leaves no temporary file",
          concurrent.cacheHit_ && same_data(concurrent, expected) &&
              no_temporary_files(concurrentDirectory));
    return audit;
  }

  namespace {
    struct BranchState {
      double s1_;
      double sp_;
      double s_;
    };

    BranchState branch_state(AuxiliarySnapshots const& snapshot,
                             AuxiliaryQuery const& query,
                             PROPOSAL::Component const& component) {
      auto const photon = query.operation_ == AuxiliaryOperation::Photon;
      auto const& brems = query.operation_ == AuxiliaryOperation::Muon
                              ? snapshot.muon_
                              : snapshot.electron_;
      auto const zThird = std::pow(component.GetNucCharge(), -1. / 3.);
      auto s1 = 1. / (zThird * component.GetLogConstant());
      auto density = query.densityGPerCm3_ / snapshot.photon_.baselineMassDensityGPerCm3_;
      auto eLpm = snapshot.photon_.eLpmMeV_;
      auto gamma = snapshot.photon_.classicalElectronRadiusCm_ /
                   (snapshot.photon_.fineStructureConstant_ * query.fraction_);
      auto charge = snapshot.photon_.sumCharge_;
      auto molecular = snapshot.photon_.molecularDensityPerCm3_;
      auto radius = snapshot.photon_.classicalElectronRadiusCm_;
      auto const sqrtTwo = 1.414213562373095048801688724209698079;
      if (photon) {
        s1 = s1 * s1 * sqrtTwo;
      } else {
        auto const nuclear = brems.leptonMassMeV_ / brems.muonMassMeV_ * 1.54 *
                             std::pow(component.GetAtomicNum(), .27);
        s1 = brems.electronMassMeV_ /
             (brems.leptonMassMeV_ * zThird * component.GetLogConstant());
        s1 = s1 * s1 * (1. + nuclear * nuclear) * sqrtTwo;
        density = query.densityGPerCm3_ / brems.baselineMassDensityGPerCm3_;
        eLpm = brems.eLpmMeV_;
        gamma = brems.classicalElectronRadiusCm_ * brems.electronMassMeV_ /
                (brems.fineStructureConstant_ * brems.leptonMassMeV_ * query.fraction_);
        charge = brems.sumCharge_;
        molecular = brems.molecularDensityPerCm3_;
        radius = brems.classicalElectronRadiusCm_;
      }
      auto const sp =
          photon
              ? .125 * std::sqrt(eLpm / (density * query.energyMeV_ * query.fraction_ *
                                         (1. - query.fraction_)))
              : .125 * std::sqrt(eLpm * query.fraction_ /
                                 (density * query.energyMeV_ * (1. - query.fraction_)));
      auto const h = std::log(sp) / std::log(s1);
      auto const xi =
          sp < s1
              ? 2.
              : sp < 1.
                    ? 1. + h - .08 * (1. - h) * (1. - (1. - h) * (1. - h)) / std::log(s1)
                    : 1.;
      gamma = 1. + 4. * 3.141592653589793238462643383279502884 * charge * radius * gamma *
                       gamma * molecular * density;
      return {s1, sp, sp / std::sqrt(xi) * gamma};
    }

    std::uint32_t branch_mask(BranchState state) {
      return (1u << (state.sp_ < state.s1_ ? 0 : state.sp_ < 1. ? 1 : 2)) |
             (1u << (state.s_ < 1.54954 ? 3 : 4)) |
             (1u << (state.s_ < .710390 ? 5 : state.s_ < .904912 ? 6 : 7));
    }
  } // namespace

  AuxiliaryDataset make_auxiliary_dataset() {
    Helpers const helpers;
    auto const aux =
        load_or_create_proposal_native_aux(helpers.views(), fresh_directory("query"));
    AuxiliaryDataset data;
    data.snapshots_.push_back(snapshots(aux));
    auto const baseline = data.snapshots_.front();
    auto const components = helpers.medium_.GetComponents();
    data.componentCount_ = components.size();
    auto append = [&](AuxiliaryQuery query, PROPOSAL::Component const& component) {
      auto const densityCorrection =
          query.densityGPerCm3_ / helpers.medium_.GetMassDensity();
      query.nativeExpected_ =
          query.operation_ == AuxiliaryOperation::Photon
              ? helpers.photon_.photon_->suppression_factor(
                    query.energyMeV_, query.fraction_, component, densityCorrection)
              : (query.operation_ == AuxiliaryOperation::Muon ? helpers.minus_.brems_
                                                              : helpers.electron_.brems_)
                    ->suppression_factor(query.energyMeV_, query.fraction_, component,
                                         densityCorrection);
      query.nativeComparable_ =
          std::isfinite(query.nativeExpected_) && query.nativeExpected_ >= 0. ? 1u : 0u;
      query.branchMask_ = branch_mask(branch_state(baseline, query, component));
      data.queries_.push_back(query);
    };
    for (auto const operation : {AuxiliaryOperation::Photon, AuxiliaryOperation::Electron,
                                 AuxiliaryOperation::Muon}) {
      for (auto const& component : components) {
        for (double energy : {1.e3, 1.e5, 1.e8, 1.e11, 1.e14, 1.e18, 1.e22})
          for (double fraction : {1.e-8, 1.e-4, .01, .3, .5, .99, 1. - 1.e-8})
            for (double density : {1.e-8, 1.e-6, 1.e-3, 1.e-2})
              append({operation, 0, component.GetHash(), energy, fraction, density},
                     component);
        AuxiliaryQuery target{operation, 0, component.GetHash(), 1.e10, .3, 1.e-3};
        // Construct inputs on both sides of all original piecewise boundaries.
        // This is test-input selection, never a replacement for the queried formula.
        auto const state = branch_state(baseline, target, component);
        for (double sp : {state.s1_ * .5, state.s1_, state.s1_ * 2., .5, 1., 2.}) {
          auto const energy = target.energyMeV_ * state.sp_ * state.sp_ / (sp * sp);
          for (double value :
               {std::nextafter(energy, 0.), energy,
                std::nextafter(energy, std::numeric_limits<double>::infinity())}) {
            auto query = target;
            query.energyMeV_ = value;
            append(query, component);
          }
        }
        for (double desired : {.710390, .904912, 1.54954}) {
          double low = 1.e-100, high = 1.e100;
          for (unsigned iteration = 0; iteration < 160; ++iteration) {
            auto const energy = std::sqrt(low) * std::sqrt(high);
            auto query = target;
            query.energyMeV_ = energy;
            if (branch_state(baseline, query, component).s_ > desired)
              low = energy;
            else
              high = energy;
          }
          for (double energy :
               {std::nextafter(high, 0.), high,
                std::nextafter(high, std::numeric_limits<double>::infinity())}) {
            auto query = target;
            query.energyMeV_ = energy;
            append(query, component);
          }
        }
      }
    }

    auto addFault = [&](AuxiliarySnapshots snapshot, AuxiliaryOperation operation) {
      auto const index = static_cast<std::uint32_t>(data.snapshots_.size());
      data.snapshots_.push_back(snapshot);
      data.queries_.push_back(
          {operation, index, components.front().GetHash(), 1.e9, .3, 1.e-3});
    };
    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const infinity = std::numeric_limits<double>::infinity();
    for (auto operation : {AuxiliaryOperation::Photon, AuxiliaryOperation::Electron,
                           AuxiliaryOperation::Muon}) {
      for (unsigned fault = 0; fault < 13; ++fault) {
        auto query =
            AuxiliaryQuery{operation, 0, components.front().GetHash(), 1.e9, .3, 1.e-3};
        switch (fault) {
          case 0:
            query.componentHash_ = std::numeric_limits<std::uint64_t>::max();
            break;
          case 1:
            query.energyMeV_ = 0.;
            break;
          case 2:
            query.energyMeV_ = -1.;
            break;
          case 3:
            query.energyMeV_ = nan;
            break;
          case 4:
            query.energyMeV_ = infinity;
            break;
          case 5:
            query.fraction_ = 0.;
            break;
          case 6:
            query.fraction_ = 1.;
            break;
          case 7:
            query.fraction_ = nan;
            break;
          case 8:
            query.densityGPerCm3_ = 0.;
            break;
          case 9:
            query.densityGPerCm3_ = -1.;
            break;
          case 10:
            query.densityGPerCm3_ = infinity;
            break;
          case 11:
            query.energyMeV_ =
                operation == AuxiliaryOperation::Muon ? PROPOSAL::MMU : PROPOSAL::ME;
            break;
          case 12:
            query.densityGPerCm3_ = std::numeric_limits<double>::max();
            break;
        }
        data.queries_.push_back(query);
      }
      for (unsigned fault = 0; fault < 8; ++fault) {
        auto changed = baseline;
        auto& brems =
            operation == AuxiliaryOperation::Muon ? changed.muon_ : changed.electron_;
        switch (fault) {
          case 0:
            changed.photon_.componentCount_ = 0;
            brems.componentCount_ = 0;
            break;
          case 1:
            changed.photon_.componentCount_ = 17;
            brems.componentCount_ = 17;
            break;
          case 2:
            changed.photon_.baselineMassDensityGPerCm3_ = nan;
            brems.baselineMassDensityGPerCm3_ = nan;
            break;
          case 3:
            changed.photon_.eLpmMeV_ = 0.;
            brems.eLpmMeV_ = 0.;
            break;
          case 4:
            changed.photon_.components_[0].nuclearCharge_ = 0.;
            brems.components_[0].nuclearCharge_ = 0.;
            break;
          case 5:
            changed.photon_.components_[0].radiationLogConstant_ = infinity;
            brems.components_[0].radiationLogConstant_ = infinity;
            break;
          case 6:
            changed.photon_.sumCharge_ = -1.;
            brems.sumCharge_ = -1.;
            break;
          case 7:
            changed.photon_.reserved_ = 9;
            brems.reserved_ = 9;
            break;
        }
        addFault(changed, operation);
      }
    }
    for (auto operation :
         {AuxiliaryOperation::ElectronMoliere, AuxiliaryOperation::MuonMoliere}) {
      data.queries_.push_back({operation});
      for (unsigned fault = 0; fault < 10; ++fault) {
        auto changed = baseline;
        auto& model = operation == AuxiliaryOperation::ElectronMoliere
                          ? changed.electronMoliere_
                          : changed.muonMoliere_;
        switch (fault) {
          case 0:
            model.componentCount_ = 0;
            break;
          case 1:
            model.maximumWeightIndex_ = model.componentCount_;
            break;
          case 2:
            model.particleMassMeV_ = 0.;
            break;
          case 3:
            model.components_[0].weightZz_ = -1.;
            break;
          case 4:
            model.components_[0].chi0SquaredMeV2_ = nan;
            break;
          case 5:
            model.c1_[0] = infinity;
            break;
          case 6:
            model.c2_[0] = nan;
            break;
          case 7:
            model.c2Large_[0] = nan;
            break;
          case 8:
            model.s2Large_[0] = infinity;
            break;
          case 9:
            model.c1IntegralLarge_[0] = nan;
            break;
        }
        addFault(changed, operation);
      }
    }
    return data;
  }
} // namespace corsika::kokkos::test
