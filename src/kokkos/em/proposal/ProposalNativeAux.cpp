/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/proposal/ProposalNativeAux.hpp>
#include <corsika/modules/proposal/NativeCalculatorView.hpp>

#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/crosssection/parametrization/Bremsstrahlung.h>
#include <PROPOSAL/crosssection/parametrization/PhotoPairProduction.h>
#include <PROPOSAL/scattering/multiple_scattering/Coefficients.h>
#include <PROPOSAL/version.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <thread>
#include <type_traits>
#include <vector>

#include <sys/file.h>
#include <unistd.h>

namespace corsika::kokkos {
  using namespace corsika::units::si;
  namespace {
    inline constexpr char AuxCacheMagic[8]{'C', '8', 'E', 'M', 'A', 'U', 'X', '\0'};

    template <typename TValue>
    TValue const* borrowed_pointer(
        std::optional<std::reference_wrapper<TValue const>> const& value) {
      return value ? &value->get() : nullptr;
    }

    struct MoliereComponentMetadata {
      std::uint64_t proposalHash_{};
      double nuclearCharge_{};
      double atomicMassNumber_{};
      double atomsInMolecule_{};
    };

    struct MoliereMetadata {
      bool enabled_{};
      std::string referenceMode_;
      double particleMassMeV_{};
      double electronMassMeV_{};
      double fineStructureConstant_{};
      double avogadroPerMol_{};
      double hbarMeVS_{};
      double speedOfLightCmPerS_{};
      double eulerMascheroni_{};
      std::vector<MoliereComponentMetadata> components_;
      std::vector<double> c1_;
      std::vector<double> c2_;
      std::vector<double> c2Large_;
      std::vector<double> s2Large_;
      std::vector<double> c1IntegralLarge_;
    };

    MoliereSnapshot make_moliere_snapshot(MoliereMetadata const& source) {
      if (!source.enabled_ || source.components_.empty() ||
          source.components_.size() > MaxMoliereComponents ||
          source.c1_.size() != MoliereSeriesCoefficientCount ||
          source.c2_.size() != MoliereSeriesCoefficientCount ||
          source.c2Large_.size() != MoliereLargeSeriesCoefficientCount ||
          source.s2Large_.size() != MoliereLargeSeriesCoefficientCount ||
          source.c1IntegralLarge_.size() != MoliereLargeIntegralCoefficientCount) {
        throw std::invalid_argument("Moliere metadata is unavailable or malformed");
      }
      MoliereSnapshot snapshot{};
      snapshot.particleMassMeV_ = source.particleMassMeV_;
      snapshot.electronMassMeV_ = source.electronMassMeV_;
      snapshot.fineStructureConstant_ = source.fineStructureConstant_;
      snapshot.avogadroPerMol_ = source.avogadroPerMol_;
      snapshot.hbarMeVS_ = source.hbarMeVS_;
      snapshot.speedOfLightCmPerS_ = source.speedOfLightCmPerS_;
      snapshot.eulerMascheroni_ = source.eulerMascheroni_;
      snapshot.componentCount_ = static_cast<std::uint32_t>(source.components_.size());

      double atomic_mass_sum = 0.;
      for (auto const& component : source.components_) {
        atomic_mass_sum += component.atomsInMolecule_ * component.atomicMassNumber_;
      }
      double mass_weights[MaxMoliereComponents]{};
      double average_atomic_mass = 0.;
      double average_charge_squared = 0.;
      double weight_zz_sum = 0.;
      for (std::size_t index = 0; index < source.components_.size(); ++index) {
        auto const& component = source.components_[index];
        auto const weight =
            component.atomsInMolecule_ * component.atomicMassNumber_ / atomic_mass_sum;
        mass_weights[index] = weight;
        average_atomic_mass += weight * component.atomicMassNumber_;
        auto const weight_zz =
            weight * component.nuclearCharge_ * component.nuclearCharge_;
        weight_zz_sum += weight_zz;
        average_charge_squared +=
            source.particleMassMeV_ == source.electronMassMeV_
                ? weight * component.nuclearCharge_ * (component.nuclearCharge_ + 1.)
                : weight_zz;
        auto const chi_0 =
            source.electronMassMeV_ * source.fineStructureConstant_ *
            std::pow(component.nuclearCharge_ * 128. /
                         (9. * moliere_snapshot_detail::Pi * moliere_snapshot_detail::Pi),
                     1. / 3.);
        snapshot.components_[index] = {
            component.nuclearCharge_, weight_zz, chi_0 * chi_0,
            3.76 * source.fineStructureConstant_ * source.fineStructureConstant_ *
                component.nuclearCharge_ * component.nuclearCharge_};
      }
      for (std::size_t index = 0; index + 1 < source.components_.size(); ++index) {
        if (mass_weights[index + 1] > mass_weights[index]) {
          snapshot.maximumWeightIndex_ = static_cast<std::uint32_t>(index + 1);
        }
      }
      snapshot.zSquaredOverAAverage_ = average_charge_squared / average_atomic_mass;
      snapshot.inverseWeightZzSum_ = 1. / weight_zz_sum;
      std::copy(source.c1_.begin(), source.c1_.end(), snapshot.c1_);
      std::copy(source.c2_.begin(), source.c2_.end(), snapshot.c2_);
      std::copy(source.c2Large_.begin(), source.c2Large_.end(), snapshot.c2Large_);
      std::copy(source.s2Large_.begin(), source.s2Large_.end(), snapshot.s2Large_);
      std::copy(source.c1IntegralLarge_.begin(), source.c1IntegralLarge_.end(),
                snapshot.c1IntegralLarge_);
      if (!valid_moliere_snapshot(snapshot)) {
        throw std::invalid_argument(
            "Moliere metadata produced an invalid device snapshot");
      }
      return snapshot;
    }

    using composition_identity_type =
        std::vector<std::pair<std::uint64_t, std::uint64_t>>;

    std::uint64_t double_bits(double value) {
      std::uint64_t result{};
      static_assert(sizeof(result) == sizeof(value));
      std::memcpy(&result, &value, sizeof(result));
      return result;
    }

    composition_identity_type composition_identity(PROPOSAL::Medium const& medium) {
      composition_identity_type result;
      result.reserve(medium.GetComponents().size() + 1);
      result.emplace_back(0u, double_bits(medium.GetSumNucleons()));
      for (auto const& component : medium.GetComponents()) {
        result.emplace_back(static_cast<std::uint64_t>(component.GetHash()),
                            double_bits(component.GetAtomInMolecule()));
      }
      std::sort(result.begin() + 1, result.end());
      return result;
    }

    bool supported_native_projectile(Code projectile) {
      switch (projectile) {
        case Code::Photon:
        case Code::Electron:
        case Code::Positron:
        case Code::MuMinus:
        case Code::MuPlus:
          return true;
        default:
          return false;
      }
    }

    struct AuxPayload {
      PhotonPairLpmSnapshot photonPairLpm_{};
      BremsLpmSnapshot bremsLpm_{};
      MoliereSnapshot electronMoliere_{};
      MoliereSnapshot muonMoliere_{};
      std::uint32_t hasMuonMoliere_{};
      std::uint32_t reserved_[3]{};
    };

    struct AuxFileHeader {
      char magic_[8]{'C', '8', 'E', 'M', 'A', 'U', 'X', '\0'};
      std::uint32_t formatVersion_{ProposalNativeAuxFormatVersion};
      std::uint32_t payloadBytes_{};
      sha256_digest_type keyHash_{};
      sha256_digest_type contentHash_{};
    };

    static_assert(std::is_trivially_copyable_v<AuxFileHeader>);

    template <typename T>
    void append_bytes(std::vector<std::uint8_t>& bytes, T const& value) {
      static_assert(std::is_trivially_copyable_v<T>);
      auto const* begin = reinterpret_cast<std::uint8_t const*>(&value);
      bytes.insert(bytes.end(), begin, begin + sizeof(T));
    }

    template <typename T>
    T read_bytes(std::vector<std::uint8_t> const& bytes, std::size_t& offset) {
      static_assert(std::is_trivially_copyable_v<T>);
      if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
        throw std::runtime_error("c8emaux payload is truncated");
      T value{};
      std::memcpy(&value, bytes.data() + offset, sizeof(T));
      offset += sizeof(T);
      return value;
    }

    std::vector<std::uint8_t> serialize_payload(AuxPayload const& payload) {
      std::vector<std::uint8_t> bytes;
      append_bytes(bytes, payload.photonPairLpm_);
      append_bytes(bytes, payload.bremsLpm_);
      append_bytes(bytes, payload.electronMoliere_);
      append_bytes(bytes, payload.muonMoliere_);
      append_bytes(bytes, payload.hasMuonMoliere_);
      append_bytes(bytes, payload.reserved_);
      // Direct-pair energy-asymmetry data are not part of the auxiliary cache.
      return bytes;
    }

    AuxPayload deserialize_payload(std::vector<std::uint8_t> const& bytes) {
      std::size_t offset = 0;
      AuxPayload payload;
      payload.photonPairLpm_ = read_bytes<PhotonPairLpmSnapshot>(bytes, offset);
      payload.bremsLpm_ = read_bytes<BremsLpmSnapshot>(bytes, offset);
      payload.electronMoliere_ = read_bytes<MoliereSnapshot>(bytes, offset);
      payload.muonMoliere_ = read_bytes<MoliereSnapshot>(bytes, offset);
      payload.hasMuonMoliere_ = read_bytes<std::uint32_t>(bytes, offset);
      auto const reserved = read_bytes<std::array<std::uint32_t, 3>>(bytes, offset);
      std::copy(reserved.begin(), reserved.end(), payload.reserved_);
      if (offset != bytes.size())
        throw std::runtime_error("c8emaux payload has trailing bytes");
      return payload;
    }

    sha256_digest_type medium_physics_key(PROPOSAL::Medium const& medium) {
      std::vector<std::uint8_t> bytes;
      auto const medium_hash = static_cast<std::uint64_t>(medium.GetHash());
      append_bytes(bytes, medium_hash);
      for (auto const value :
           {medium.GetI(), medium.GetC(), medium.GetA(), medium.GetM(), medium.GetX0(),
            medium.GetX1(), medium.GetD0(), medium.GetMassDensity(),
            medium.GetMolDensity(), medium.GetSumCharge(), medium.GetZA(), medium.GetMM(),
            medium.GetSumNucleons()})
        append_bytes(bytes, double_bits(value));
      for (auto const& entry : composition_identity(medium)) {
        append_bytes(bytes, entry.first);
        append_bytes(bytes, entry.second);
      }
      return sha256(bytes);
    }

    sha256_digest_type medium_static_physics_key(PROPOSAL::Medium const& medium) {
      std::vector<std::uint8_t> bytes;
      for (auto const value :
           {medium.GetI(), medium.GetC(), medium.GetA(), medium.GetM(), medium.GetX0(),
            medium.GetX1(), medium.GetD0(), medium.GetSumCharge(), medium.GetZA(),
            medium.GetMM(), medium.GetSumNucleons()})
        append_bytes(bytes, double_bits(value));
      for (auto const& entry : composition_identity(medium)) {
        append_bytes(bytes, entry.first);
        append_bytes(bytes, entry.second);
      }
      return sha256(bytes);
    }

    sha256_digest_type calculate_key(
        std::vector<proposal::NativeInteractionCalculatorView> const& views) {
      std::vector<std::uint8_t> bytes;
      append_bytes(bytes, ProposalNativeAuxFormatVersion);
      bytes.insert(bytes.end(), std::begin(ProposalNativeAuxAlgorithmVersion),
                   std::end(ProposalNativeAuxAlgorithmVersion));
      auto const proposal_version = getPROPOSALVersion();
      bytes.insert(bytes.end(), proposal_version.begin(), proposal_version.end());
      constexpr char cubic_version[] = "CubicInterpolation-0.1.5";
      bytes.insert(bytes.end(), std::begin(cubic_version), std::end(cubic_version));
      struct Identity {
        sha256_digest_type medium_{};
        std::int32_t pdgId_{};
        std::uint64_t cutBits_{};
      };
      std::vector<Identity> identities;
      identities.reserve(views.size());
      for (auto const& view : views) {
        if (!supported_native_projectile(view.getProjectile())) continue;
        if (!borrowed_pointer(view.getMedium()))
          throw std::invalid_argument(
              "proposal-native auxiliary key contains a null medium");
        auto const cut_MeV = static_cast<double>(view.getStochasticEnergyCut() / 1_MeV);
        std::uint64_t cut_bits = 0;
        std::memcpy(&cut_bits, &cut_MeV, sizeof(cut_bits));
        identities.push_back({medium_physics_key(*borrowed_pointer(view.getMedium())),
                              static_cast<std::int32_t>(get_PDG(view.getProjectile())),
                              cut_bits});
      }
      auto const less = [](Identity const& left, Identity const& right) {
        return std::tie(left.medium_, left.pdgId_, left.cutBits_) <
               std::tie(right.medium_, right.pdgId_, right.cutBits_);
      };
      std::sort(identities.begin(), identities.end(), less);
      identities.erase(std::unique(identities.begin(), identities.end(),
                                   [](Identity const& left, Identity const& right) {
                                     return left.medium_ == right.medium_ &&
                                            left.pdgId_ == right.pdgId_ &&
                                            left.cutBits_ == right.cutBits_;
                                   }),
                       identities.end());
      if (identities.empty())
        throw std::invalid_argument(
            "proposal-native auxiliary key has no supported projectile calculators");
      for (auto const& identity : identities) {
        bytes.insert(bytes.end(), identity.medium_.begin(), identity.medium_.end());
        append_bytes(bytes, identity.pdgId_);
        append_bytes(bytes, identity.cutBits_);
      }
      return sha256(bytes);
    }

    MoliereSnapshot make_moliere(PROPOSAL::Medium const& medium,
                                 double particle_mass_MeV) {
      MoliereMetadata metadata;
      metadata.enabled_ = true;
      metadata.referenceMode_ = "proposal_analytic";
      metadata.particleMassMeV_ = particle_mass_MeV;
      metadata.electronMassMeV_ = PROPOSAL::ME;
      metadata.fineStructureConstant_ = PROPOSAL::ALPHA;
      metadata.avogadroPerMol_ = PROPOSAL::NA;
      metadata.hbarMeVS_ = PROPOSAL::HBAR;
      metadata.speedOfLightCmPerS_ = PROPOSAL::SPEED;
      metadata.eulerMascheroni_ = PROPOSAL::EULER_MASCHERONI;
      for (auto const& component : medium.GetComponents()) {
        metadata.components_.push_back(
            {static_cast<std::uint64_t>(component.GetHash()), component.GetNucCharge(),
             component.GetAtomicNum(), component.GetAtomInMolecule()});
      }
      metadata.c1_.assign(PROPOSAL::c1, PROPOSAL::c1 + MoliereSeriesCoefficientCount);
      metadata.c2_.assign(PROPOSAL::c2, PROPOSAL::c2 + MoliereSeriesCoefficientCount);
      metadata.c2Large_.assign(PROPOSAL::c2large,
                               PROPOSAL::c2large + MoliereLargeSeriesCoefficientCount);
      metadata.s2Large_.assign(PROPOSAL::s2large,
                               PROPOSAL::s2large + MoliereLargeSeriesCoefficientCount);
      metadata.c1IntegralLarge_.assign(
          PROPOSAL::C1large, PROPOSAL::C1large + MoliereLargeIntegralCoefficientCount);
      return make_moliere_snapshot(metadata);
    }

    AuxPayload generate_payload(
        std::vector<proposal::NativeInteractionCalculatorView> const& views) {
      if (views.empty())
        throw std::invalid_argument(
            "proposal-native auxiliary export has no calculator views");
      PROPOSAL::Medium const* medium = nullptr;
      composition_identity_type composition;
      sha256_digest_type static_medium_physics{};
      PROPOSAL::crosssection::PhotoPairLPM const* photon_lpm = nullptr;
      PROPOSAL::crosssection::BremsLPM const* brems_lpm = nullptr;
      PROPOSAL::crosssection::BremsLPM const* muon_brems_lpm = nullptr;
      std::size_t muon_lpm_medium_hash = std::numeric_limits<std::size_t>::max();
      std::size_t photon_lpm_medium_hash = std::numeric_limits<std::size_t>::max();
      std::size_t brems_lpm_medium_hash = std::numeric_limits<std::size_t>::max();
      bool has_muon = false;
      for (auto const& view : views) {
        if (!supported_native_projectile(view.getProjectile())) continue;
        if (!borrowed_pointer(view.getMedium()))
          throw std::invalid_argument(
              "proposal-native auxiliary export contains a null medium");
        auto const candidate_composition =
            composition_identity(*borrowed_pointer(view.getMedium()));
        if (!medium) {
          medium = borrowed_pointer(view.getMedium());
          composition = candidate_composition;
          static_medium_physics =
              medium_static_physics_key(*borrowed_pointer(view.getMedium()));
        } else if (candidate_composition != composition) {
          throw std::invalid_argument(
              "proposal-native auxiliary cache supports one chemical "
              "composition; atmosphere layers may differ only in mass density");
        } else if (medium_static_physics_key(*borrowed_pointer(view.getMedium())) !=
                   static_medium_physics) {
          throw std::invalid_argument(
              "proposal-native auxiliary media differ in static material "
              "properties other than density");
        } else if (view.getMediumHash() < medium->GetHash())
          medium = borrowed_pointer(view.getMedium());
        if (borrowed_pointer(view.getPhotonPairLpm()) &&
            view.getMediumHash() < photon_lpm_medium_hash) {
          photon_lpm = borrowed_pointer(view.getPhotonPairLpm());
          photon_lpm_medium_hash = view.getMediumHash();
        }
        if (borrowed_pointer(view.getBremsLpm()) &&
            (view.getProjectile() == Code::Electron ||
             view.getProjectile() == Code::Positron) &&
            view.getMediumHash() < brems_lpm_medium_hash) {
          brems_lpm = borrowed_pointer(view.getBremsLpm());
          brems_lpm_medium_hash = view.getMediumHash();
        }
        if (view.getProjectile() == Code::MuMinus ||
            view.getProjectile() == Code::MuPlus) {
          has_muon = true;
          if (borrowed_pointer(view.getBremsLpm()) &&
              view.getMediumHash() < muon_lpm_medium_hash) {
            muon_brems_lpm = borrowed_pointer(view.getBremsLpm());
            muon_lpm_medium_hash = view.getMediumHash();
          }
        }
      }
      if (!photon_lpm || !brems_lpm)
        throw std::invalid_argument(
            "proposal-native auxiliary export requires photon-pair and electron brems "
            "LPM calculators");

      AuxPayload payload{};
      auto const photon = photon_lpm->ExportParameters();
      payload.photonPairLpm_.baselineMassDensityGPerCm3_ = photon.mass_density_g_per_cm3;
      payload.photonPairLpm_.molecularDensityPerCm3_ = photon.molecular_density_per_cm3;
      payload.photonPairLpm_.sumCharge_ = photon.sum_charge;
      payload.photonPairLpm_.eLpmMeV_ = photon.e_lpm_MeV;
      payload.photonPairLpm_.classicalElectronRadiusCm_ = PROPOSAL::RE;
      payload.photonPairLpm_.fineStructureConstant_ = PROPOSAL::ALPHA;

      auto const brems = brems_lpm->ExportParameters();
      payload.bremsLpm_.baselineMassDensityGPerCm3_ = brems.mass_density_g_per_cm3;
      payload.bremsLpm_.molecularDensityPerCm3_ = brems.molecular_density_per_cm3;
      payload.bremsLpm_.sumCharge_ = brems.sum_charge;
      payload.bremsLpm_.eLpmMeV_ = brems.e_lpm_MeV;
      payload.bremsLpm_.leptonMassMeV_ = brems.particle_mass_MeV;
      payload.bremsLpm_.electronMassMeV_ = PROPOSAL::ME;
      payload.bremsLpm_.muonMassMeV_ = PROPOSAL::MMU;
      payload.bremsLpm_.classicalElectronRadiusCm_ = PROPOSAL::RE;
      payload.bremsLpm_.fineStructureConstant_ = PROPOSAL::ALPHA;
      if (has_muon) {
        if (!muon_brems_lpm)
          throw std::invalid_argument(
              "muon device final states require native muon LPM parameters");
        auto const muon = muon_brems_lpm->ExportParameters();
        if (muon.particle_mass_MeV != PROPOSAL::MMU)
          throw std::invalid_argument("unexpected muon LPM mass");
        payload.bremsLpm_.muonBaselineMassDensityGPerCm3_ = muon.mass_density_g_per_cm3;
        payload.bremsLpm_.muonMolecularDensityPerCm3_ = muon.molecular_density_per_cm3;
        payload.bremsLpm_.muonSumCharge_ = muon.sum_charge;
        payload.bremsLpm_.muonELpmMeV_ = muon.e_lpm_MeV;
      }

      auto const components = medium->GetComponents();
      if (components.empty() || components.size() > MaxPhotonPairLpmComponents ||
          components.size() > MaxBremsLpmComponents)
        throw std::invalid_argument("proposal-native LPM component count is unsupported");
      payload.photonPairLpm_.componentCount_ = components.size();
      payload.bremsLpm_.componentCount_ = components.size();
      for (std::size_t index = 0; index < components.size(); ++index) {
        auto const& component = components[index];
        payload.photonPairLpm_.components_[index] = {
            static_cast<std::uint64_t>(component.GetHash()), component.GetNucCharge(),
            component.GetLogConstant()};
        payload.bremsLpm_.components_[index] = {
            static_cast<std::uint64_t>(component.GetHash()), component.GetNucCharge(),
            component.GetAtomicNum(), component.GetLogConstant()};
      }
      payload.electronMoliere_ = make_moliere(*medium, PROPOSAL::EMinusDef().mass);
      payload.hasMuonMoliere_ = has_muon ? 1u : 0u;
      if (has_muon)
        payload.muonMoliere_ = make_moliere(*medium, PROPOSAL::MuMinusDef().mass);
      return payload;
    }

    bool finite_positive(double value) { return std::isfinite(value) && value > 0.; }

    void validate_payload(AuxPayload const& payload) {
      auto const& photon = payload.photonPairLpm_;
      auto const& brems = payload.bremsLpm_;
      if (payload.hasMuonMoliere_ &&
          (!finite_positive(brems.muonBaselineMassDensityGPerCm3_) ||
           !finite_positive(brems.muonMolecularDensityPerCm3_) ||
           !finite_positive(brems.muonSumCharge_) ||
           !finite_positive(brems.muonELpmMeV_)))
        throw std::runtime_error("c8emaux muon LPM payload is invalid");
      if (!finite_positive(photon.baselineMassDensityGPerCm3_) ||
          !finite_positive(photon.molecularDensityPerCm3_) ||
          !finite_positive(photon.sumCharge_) || !finite_positive(photon.eLpmMeV_) ||
          !finite_positive(photon.classicalElectronRadiusCm_) ||
          !finite_positive(photon.fineStructureConstant_) ||
          photon.componentCount_ == 0 ||
          photon.componentCount_ > MaxPhotonPairLpmComponents || photon.reserved_ != 0)
        throw std::runtime_error("c8emaux photon-pair LPM payload is invalid");
      if (!finite_positive(brems.baselineMassDensityGPerCm3_) ||
          !finite_positive(brems.molecularDensityPerCm3_) ||
          !finite_positive(brems.sumCharge_) || !finite_positive(brems.eLpmMeV_) ||
          !finite_positive(brems.leptonMassMeV_) ||
          !finite_positive(brems.electronMassMeV_) ||
          !finite_positive(brems.muonMassMeV_) ||
          !finite_positive(brems.classicalElectronRadiusCm_) ||
          !finite_positive(brems.fineStructureConstant_) || brems.componentCount_ == 0 ||
          brems.componentCount_ > MaxBremsLpmComponents || brems.reserved_ != 0)
        throw std::runtime_error("c8emaux bremsstrahlung LPM payload is invalid");
      if (photon.componentCount_ != brems.componentCount_)
        throw std::runtime_error("c8emaux LPM component counts are inconsistent");
      for (std::uint32_t index = 0; index < photon.componentCount_; ++index) {
        auto const& photon_component = photon.components_[index];
        auto const& brems_component = brems.components_[index];
        if (photon_component.componentHash_ != brems_component.componentHash_ ||
            !finite_positive(photon_component.nuclearCharge_) ||
            !finite_positive(photon_component.radiationLogConstant_) ||
            !finite_positive(brems_component.nuclearCharge_) ||
            !finite_positive(brems_component.atomicMassNumber_) ||
            !finite_positive(brems_component.radiationLogConstant_))
          throw std::runtime_error("c8emaux LPM component payload is invalid");
      }
      if (!valid_moliere_snapshot(payload.electronMoliere_))
        throw std::runtime_error("c8emaux electron Moliere payload is invalid");
      if (payload.hasMuonMoliere_ > 1u)
        throw std::runtime_error("c8emaux muon Moliere presence flag is invalid");
      if (payload.hasMuonMoliere_ != 0u && !valid_moliere_snapshot(payload.muonMoliere_))
        throw std::runtime_error("c8emaux muon Moliere payload is invalid");
      for (auto const value : payload.reserved_)
        if (value != 0u) throw std::runtime_error("c8emaux reserved payload is nonzero");
    }

    class FileLock {
    public:
      explicit FileLock(std::filesystem::path const& path)
          : path_(path) {
        descriptor_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (descriptor_ < 0)
          throw std::runtime_error("cannot open c8emaux cache lock " + path.string() +
                                   ": " + std::strerror(errno));
        while (::flock(descriptor_, LOCK_EX) != 0) {
          if (errno == EINTR) continue;
          auto const message = std::string(std::strerror(errno));
          ::close(descriptor_);
          descriptor_ = -1;
          throw std::runtime_error("cannot acquire c8emaux cache lock " + path.string() +
                                   ": " + message);
        }
      }

      FileLock(FileLock const&) = delete;
      FileLock& operator=(FileLock const&) = delete;

      ~FileLock() {
        if (descriptor_ >= 0) {
          ::flock(descriptor_, LOCK_UN);
          ::close(descriptor_);
        }
      }

    private:
      std::filesystem::path path_;
      int descriptor_{-1};
    };

    struct TemporaryFileGuard {
      std::filesystem::path path_;
      ~TemporaryFileGuard() {
        if (!path_.empty()) {
          std::error_code ignored;
          std::filesystem::remove(path_, ignored);
        }
      }
    };

    ProposalNativeAuxData from_payload(AuxPayload const& payload,
                                       sha256_digest_type const& key,
                                       sha256_digest_type const& content,
                                       std::filesystem::path const& path, bool hit) {
      ProposalNativeAuxData result;
      result.photonPairLpm_ = payload.photonPairLpm_;
      result.bremsLpm_ = payload.bremsLpm_;
      result.electronMoliere_ = payload.electronMoliere_;
      result.muonMoliere_ = payload.muonMoliere_;
      result.hasMuonMoliere_ = payload.hasMuonMoliere_;
      result.keyHash_ = key;
      result.contentHash_ = content;
      result.cacheFile_ = path;
      result.cacheHit_ = hit;
      return result;
    }

    ProposalNativeAuxData read_aux(std::filesystem::path const& path,
                                   sha256_digest_type const& expected_key) {
      std::ifstream input(path, std::ios::binary);
      if (!input) throw std::runtime_error("cannot open c8emaux cache");
      AuxFileHeader header{};
      input.read(reinterpret_cast<char*>(&header), sizeof(header));
      if (!input || header.payloadBytes_ == 0 || header.payloadBytes_ > (1u << 28))
        throw std::runtime_error("c8emaux cache payload size is invalid");
      std::vector<std::uint8_t> bytes(header.payloadBytes_);
      input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
      if (!input || input.peek() != std::ifstream::traits_type::eof() ||
          std::memcmp(header.magic_, AuxCacheMagic, sizeof(AuxCacheMagic)) != 0 ||
          header.formatVersion_ != ProposalNativeAuxFormatVersion ||
          header.keyHash_ != expected_key)
        throw std::runtime_error("c8emaux cache metadata is incompatible");
      auto const actual = sha256(bytes);
      if (actual != header.contentHash_)
        throw std::runtime_error("c8emaux cache content hash mismatch");
      auto const payload = deserialize_payload(bytes);
      validate_payload(payload);
      return from_payload(payload, header.keyHash_, header.contentHash_, path, true);
    }

    std::optional<ProposalNativeAuxData> try_read_aux(
        std::filesystem::path const& path, sha256_digest_type const& expected_key) {
      if (!std::filesystem::exists(path)) return std::nullopt;
      try {
        return read_aux(path, expected_key);
      } catch (std::runtime_error const&) { return std::nullopt; }
    }
  } // namespace

  std::filesystem::path default_proposal_native_aux_cache_directory() {
    if (auto const* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
      auto const base = std::filesystem::path(xdg);
      if (!base.is_absolute())
        throw std::runtime_error("XDG_CACHE_HOME must be an absolute path");
      return base / "corsika8" / "gpu-em-aux";
    }
    if (auto const* home = std::getenv("HOME"); home && *home) {
      auto const base = std::filesystem::path(home);
      if (!base.is_absolute()) throw std::runtime_error("HOME must be an absolute path");
      return base / ".cache" / "corsika8" / "gpu-em-aux";
    }
    throw std::runtime_error(
        "cannot determine XDG cache directory for proposal-native auxiliary data");
  }

  ProposalNativeAuxData load_or_create_proposal_native_aux(
      std::vector<proposal::NativeInteractionCalculatorView> const& views,
      std::filesystem::path const& requested_directory) {
    auto const directory = requested_directory.empty()
                               ? default_proposal_native_aux_cache_directory()
                               : requested_directory;
    auto const key = calculate_key(views);
    auto const path = directory / (to_hex(key) + ".c8emaux");
    std::filesystem::create_directories(directory);
    if (auto cached = try_read_aux(path, key)) return *cached;

    auto lock_path = path;
    lock_path += ".lock.v2";
    FileLock lock(lock_path);
    if (auto cached = try_read_aux(path, key)) return *cached;

    auto const payload = generate_payload(views);
    validate_payload(payload);
    auto const bytes = serialize_payload(payload);
    auto const content = sha256(bytes);
    AuxFileHeader header;
    if (bytes.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("c8emaux payload exceeds format limit");
    header.payloadBytes_ = static_cast<std::uint32_t>(bytes.size());
    header.keyHash_ = key;
    header.contentHash_ = content;
    auto temporary = path;
    temporary +=
        ".tmp." + std::to_string(static_cast<long long>(getpid())) + "." +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    TemporaryFileGuard temporary_guard{temporary};
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      if (!output) throw std::runtime_error("cannot create c8emaux cache");
      output.write(reinterpret_cast<char const*>(&header), sizeof(header));
      output.write(reinterpret_cast<char const*>(bytes.data()), bytes.size());
      output.flush();
      if (!output) throw std::runtime_error("cannot write c8emaux cache");
    }
    auto const verified_temporary = read_aux(temporary, key);
    if (verified_temporary.contentHash_ != content)
      throw std::runtime_error("c8emaux write-after-read content hash mismatch");
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
      if (auto cached = try_read_aux(path, key)) return *cached;
      if (!std::filesystem::exists(path))
        throw std::runtime_error("cannot atomically install c8emaux cache: " +
                                 error.message());
      throw std::runtime_error(
          "c8emaux target exists but failed validation after rename error: " +
          error.message());
    }
    temporary_guard.path_.clear();
    auto installed = read_aux(path, key);
    installed.cacheHit_ = false;
    return installed;
  }

} // namespace corsika::kokkos
