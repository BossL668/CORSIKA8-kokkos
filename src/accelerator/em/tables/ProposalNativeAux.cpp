/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/common/tables/ProposalNativeAux.hpp>
#include <corsika/modules/proposal/NativeCalculatorView.hpp>

#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/crosssection/parametrization/Bremsstrahlung.h>
#include <PROPOSAL/crosssection/parametrization/PhotoPairProduction.h>
#include <PROPOSAL/scattering/multiple_scattering/Coefficients.h>
#include <PROPOSAL/version.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

#include <corsika/accelerator/em/common/tables/PhysicsConstants.hpp>

namespace corsika::gpu::em::tables {
  using namespace corsika::units::si;
  namespace {
    inline constexpr char AuxCacheMagic[8]{
        'C', '8', 'E', 'M', 'A', 'U', 'X', '\0'};

    using CompositionIdentity =
        std::vector<std::pair<std::uint64_t, std::uint64_t>>;

    std::uint64_t doubleBits(double value) {
      std::uint64_t result{};
      static_assert(sizeof(result) == sizeof(value));
      std::memcpy(&result, &value, sizeof(result));
      return result;
    }

    CompositionIdentity compositionIdentity(PROPOSAL::Medium const& medium) {
      CompositionIdentity result;
      result.reserve(medium.GetComponents().size() + 1);
      result.emplace_back(0u, doubleBits(medium.GetSumNucleons()));
      for (auto const& component : medium.GetComponents()) {
        result.emplace_back(
            static_cast<std::uint64_t>(component.GetHash()),
            doubleBits(component.GetAtomInMolecule()));
      }
      std::sort(result.begin() + 1, result.end());
      return result;
    }

    bool supportedNativeProjectile(Code projectile) {
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
      PhotonPairLpmSnapshot photon_pair_lpm{};
      BremsLpmSnapshot brems_lpm{};
      MoliereSnapshot electron_moliere{};
      MoliereSnapshot muon_moliere{};
      std::uint32_t has_muon_moliere{};
      std::uint32_t reserved[3]{};
    };

    struct AuxFileHeader {
      char magic[8]{'C', '8', 'E', 'M', 'A', 'U', 'X', '\0'};
      std::uint32_t format_version{ProposalNativeAuxFormatVersion};
      std::uint32_t payload_bytes{};
      Sha256Digest key_hash{};
      Sha256Digest content_hash{};
    };

    static_assert(std::is_trivially_copyable_v<AuxFileHeader>);

    template <typename T>
    void appendBytes(std::vector<std::uint8_t>& bytes, T const& value) {
      static_assert(std::is_trivially_copyable_v<T>);
      auto const* begin = reinterpret_cast<std::uint8_t const*>(&value);
      bytes.insert(bytes.end(), begin, begin + sizeof(T));
    }

    template <typename T>
    T readBytes(std::vector<std::uint8_t> const& bytes, std::size_t& offset) {
      static_assert(std::is_trivially_copyable_v<T>);
      if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
        throw std::runtime_error("c8emaux payload is truncated");
      T value{};
      std::memcpy(&value, bytes.data() + offset, sizeof(T));
      offset += sizeof(T);
      return value;
    }

    std::vector<std::uint8_t> serializePayload(AuxPayload const& payload) {
      std::vector<std::uint8_t> bytes;
      appendBytes(bytes, payload.photon_pair_lpm);
      appendBytes(bytes, payload.brems_lpm);
      appendBytes(bytes, payload.electron_moliere);
      appendBytes(bytes, payload.muon_moliere);
      appendBytes(bytes, payload.has_muon_moliere);
      appendBytes(bytes, payload.reserved);
/* Epair rho remains the device rejection sampler; no rho payload is cached. */
      return bytes;
    }

    AuxPayload deserializePayload(std::vector<std::uint8_t> const& bytes) {
      std::size_t offset = 0;
      AuxPayload payload;
      payload.photon_pair_lpm = readBytes<PhotonPairLpmSnapshot>(bytes, offset);
      payload.brems_lpm = readBytes<BremsLpmSnapshot>(bytes, offset);
      payload.electron_moliere = readBytes<MoliereSnapshot>(bytes, offset);
      payload.muon_moliere = readBytes<MoliereSnapshot>(bytes, offset);
      payload.has_muon_moliere = readBytes<std::uint32_t>(bytes, offset);
      auto const reserved = readBytes<std::array<std::uint32_t, 3>>(bytes, offset);
      std::copy(reserved.begin(), reserved.end(), payload.reserved);
      if (offset != bytes.size())
        throw std::runtime_error("c8emaux payload has trailing bytes");
      return payload;
    }

    Sha256Digest mediumPhysicsKey(PROPOSAL::Medium const& medium) {
      std::vector<std::uint8_t> bytes;
      auto const medium_hash =
          static_cast<std::uint64_t>(medium.GetHash());
      appendBytes(bytes, medium_hash);
      for (auto const value :
           {medium.GetI(), medium.GetC(), medium.GetA(), medium.GetM(),
            medium.GetX0(), medium.GetX1(), medium.GetD0(),
            medium.GetMassDensity(), medium.GetMolDensity(),
            medium.GetSumCharge(), medium.GetZA(), medium.GetMM(),
            medium.GetSumNucleons()})
        appendBytes(bytes, doubleBits(value));
      for (auto const& entry : compositionIdentity(medium)) {
        appendBytes(bytes, entry.first);
        appendBytes(bytes, entry.second);
      }
      return sha256(bytes);
    }

    Sha256Digest mediumStaticPhysicsKey(PROPOSAL::Medium const& medium) {
      std::vector<std::uint8_t> bytes;
      for (auto const value :
           {medium.GetI(), medium.GetC(), medium.GetA(), medium.GetM(),
            medium.GetX0(), medium.GetX1(), medium.GetD0(),
            medium.GetSumCharge(), medium.GetZA(), medium.GetMM(),
            medium.GetSumNucleons()})
        appendBytes(bytes, doubleBits(value));
      for (auto const& entry : compositionIdentity(medium)) {
        appendBytes(bytes, entry.first);
        appendBytes(bytes, entry.second);
      }
      return sha256(bytes);
    }

    Sha256Digest calculateKey(
        std::vector<proposal::NativeInteractionCalculatorView> const& views) {
      std::vector<std::uint8_t> bytes;
      appendBytes(bytes, ProposalNativeAuxFormatVersion);
      bytes.insert(bytes.end(), std::begin(ProposalNativeAuxAlgorithmVersion),
                   std::end(ProposalNativeAuxAlgorithmVersion));
      auto const proposal_version = getPROPOSALVersion();
      bytes.insert(bytes.end(), proposal_version.begin(),
                   proposal_version.end());
      constexpr char cubic_version[] = "CubicInterpolation-0.1.5";
      bytes.insert(bytes.end(), std::begin(cubic_version),
                   std::end(cubic_version));
      struct Identity {
        Sha256Digest medium{};
        std::int32_t pdg_id{};
        std::uint64_t cut_bits{};
      };
      std::vector<Identity> identities;
      identities.reserve(views.size());
      for (auto const& view : views) {
        if (!supportedNativeProjectile(view.projectile)) continue;
        if (!view.medium)
          throw std::invalid_argument(
              "proposal-native auxiliary key contains a null medium");
        auto const cut_MeV = static_cast<double>(view.stochastic_energy_cut / 1_MeV);
        std::uint64_t cut_bits = 0;
        std::memcpy(&cut_bits, &cut_MeV, sizeof(cut_bits));
        identities.push_back(
            {mediumPhysicsKey(*view.medium),
             static_cast<std::int32_t>(get_PDG(view.projectile)), cut_bits});
      }
      auto const less = [](Identity const& left, Identity const& right) {
        return std::tie(left.medium, left.pdg_id, left.cut_bits) <
               std::tie(right.medium, right.pdg_id, right.cut_bits);
      };
      std::sort(identities.begin(), identities.end(), less);
      identities.erase(
          std::unique(
              identities.begin(), identities.end(),
              [](Identity const& left, Identity const& right) {
                return left.medium == right.medium &&
                       left.pdg_id == right.pdg_id &&
                       left.cut_bits == right.cut_bits;
              }),
          identities.end());
      if (identities.empty())
        throw std::invalid_argument(
            "proposal-native auxiliary key has no supported projectile calculators");
      for (auto const& identity : identities) {
        bytes.insert(bytes.end(), identity.medium.begin(),
                     identity.medium.end());
        appendBytes(bytes, identity.pdg_id);
        appendBytes(bytes, identity.cut_bits);
      }
      return sha256(bytes);
    }

    MoliereSnapshot makeMoliere(PROPOSAL::Medium const& medium,
                                double particle_mass_MeV) {
      MoliereMetadata metadata;
      metadata.enabled = true;
      metadata.reference_mode = "proposal_analytic";
      metadata.particle_mass_MeV = particle_mass_MeV;
      metadata.electron_mass_MeV = PROPOSAL::ME;
      metadata.fine_structure_constant = PROPOSAL::ALPHA;
      metadata.avogadro_per_mol = PROPOSAL::NA;
      metadata.hbar_MeV_s = PROPOSAL::HBAR;
      metadata.speed_of_light_cm_per_s = PROPOSAL::SPEED;
      metadata.euler_mascheroni = PROPOSAL::EULER_MASCHERONI;
      for (auto const& component : medium.GetComponents()) {
        metadata.components.push_back(
            {static_cast<std::uint64_t>(component.GetHash()),
             component.GetNucCharge(), component.GetAtomicNum(),
             component.GetAtomInMolecule()});
      }
      metadata.c1.assign(PROPOSAL::c1,
                         PROPOSAL::c1 + MoliereSeriesCoefficientCount);
      metadata.c2.assign(PROPOSAL::c2,
                         PROPOSAL::c2 + MoliereSeriesCoefficientCount);
      metadata.c2_large.assign(
          PROPOSAL::c2large,
          PROPOSAL::c2large + MoliereLargeSeriesCoefficientCount);
      metadata.s2_large.assign(
          PROPOSAL::s2large,
          PROPOSAL::s2large + MoliereLargeSeriesCoefficientCount);
      metadata.C1_large.assign(
          PROPOSAL::C1large,
          PROPOSAL::C1large + MoliereLargeIntegralCoefficientCount);
      return makeMoliereSnapshot(metadata);
    }


    AuxPayload generatePayload(
        std::vector<proposal::NativeInteractionCalculatorView> const& views) {
      if (views.empty())
        throw std::invalid_argument(
            "proposal-native auxiliary export has no calculator views");
      PROPOSAL::Medium const* medium = nullptr;
      CompositionIdentity composition;
      Sha256Digest static_medium_physics{};
      PROPOSAL::crosssection::PhotoPairLPM const* photon_lpm = nullptr;
      PROPOSAL::crosssection::BremsLPM const* brems_lpm = nullptr;
      std::size_t photon_lpm_medium_hash =
          std::numeric_limits<std::size_t>::max();
      std::size_t brems_lpm_medium_hash =
          std::numeric_limits<std::size_t>::max();
      bool has_muon = false;
      for (auto const& view : views) {
        if (!supportedNativeProjectile(view.projectile)) continue;
        if (!view.medium)
          throw std::invalid_argument(
              "proposal-native auxiliary export contains a null medium");
        auto const candidate_composition = compositionIdentity(*view.medium);
        if (!medium) {
          medium = view.medium;
          composition = candidate_composition;
          static_medium_physics = mediumStaticPhysicsKey(*view.medium);
        } else if (candidate_composition != composition) {
          throw std::invalid_argument(
              "proposal-native auxiliary cache supports one chemical "
              "composition; atmosphere layers may differ only in mass density");
        } else if (mediumStaticPhysicsKey(*view.medium) !=
                   static_medium_physics) {
          throw std::invalid_argument(
              "proposal-native auxiliary media differ in static material "
              "properties other than density");
        } else if (view.medium_hash < medium->GetHash())
          medium = view.medium;
        if (view.photon_pair_lpm &&
            view.medium_hash < photon_lpm_medium_hash) {
          photon_lpm = view.photon_pair_lpm;
          photon_lpm_medium_hash = view.medium_hash;
        }
        if (view.brems_lpm &&
            (view.projectile == Code::Electron ||
             view.projectile == Code::Positron) &&
            view.medium_hash < brems_lpm_medium_hash) {
          brems_lpm = view.brems_lpm;
          brems_lpm_medium_hash = view.medium_hash;
        }
        if (view.projectile == Code::MuMinus || view.projectile == Code::MuPlus)
          has_muon = true;
      }
      if (!photon_lpm || !brems_lpm)
        throw std::invalid_argument(
            "proposal-native auxiliary export requires photon-pair and electron brems LPM calculators");

      AuxPayload payload{};
      auto const photon = photon_lpm->ExportParameters();
      payload.photon_pair_lpm.baseline_mass_density_g_per_cm3 =
          photon.mass_density_g_per_cm3;
      payload.photon_pair_lpm.molecular_density_per_cm3 =
          photon.molecular_density_per_cm3;
      payload.photon_pair_lpm.sum_charge = photon.sum_charge;
      payload.photon_pair_lpm.e_lpm_MeV = photon.e_lpm_MeV;
      payload.photon_pair_lpm.classical_electron_radius_cm = PROPOSAL::RE;
      payload.photon_pair_lpm.fine_structure_constant = PROPOSAL::ALPHA;

      auto const brems = brems_lpm->ExportParameters();
      payload.brems_lpm.baseline_mass_density_g_per_cm3 =
          brems.mass_density_g_per_cm3;
      payload.brems_lpm.molecular_density_per_cm3 =
          brems.molecular_density_per_cm3;
      payload.brems_lpm.sum_charge = brems.sum_charge;
      payload.brems_lpm.e_lpm_MeV = brems.e_lpm_MeV;
      payload.brems_lpm.lepton_mass_MeV = brems.particle_mass_MeV;
      payload.brems_lpm.electron_mass_MeV = PROPOSAL::ME;
      payload.brems_lpm.muon_mass_MeV = PROPOSAL::MMU;
      payload.brems_lpm.classical_electron_radius_cm = PROPOSAL::RE;
      payload.brems_lpm.fine_structure_constant = PROPOSAL::ALPHA;

      auto const components = medium->GetComponents();
      if (components.empty() ||
          components.size() > MaxPhotonPairLpmComponents ||
          components.size() > MaxBremsLpmComponents)
        throw std::invalid_argument(
            "proposal-native LPM component count is unsupported");
      payload.photon_pair_lpm.component_count = components.size();
      payload.brems_lpm.component_count = components.size();
      for (std::size_t index = 0; index < components.size(); ++index) {
        auto const& component = components[index];
        payload.photon_pair_lpm.components[index] =
            {static_cast<std::uint64_t>(component.GetHash()),
             component.GetNucCharge(), component.GetLogConstant()};
        payload.brems_lpm.components[index] =
            {static_cast<std::uint64_t>(component.GetHash()),
             component.GetNucCharge(), component.GetAtomicNum(),
             component.GetLogConstant()};
      }
      payload.electron_moliere =
          makeMoliere(*medium, PROPOSAL::EMinusDef().mass);
      payload.has_muon_moliere = has_muon ? 1u : 0u;
      if (has_muon)
        payload.muon_moliere = makeMoliere(*medium, PROPOSAL::MuMinusDef().mass);
      return payload;
    }

    bool finitePositive(double value) {
      return std::isfinite(value) && value > 0.;
    }

    void validatePayload(AuxPayload const& payload) {
      auto const& photon = payload.photon_pair_lpm;
      auto const& brems = payload.brems_lpm;
      if (!finitePositive(photon.baseline_mass_density_g_per_cm3) ||
          !finitePositive(photon.molecular_density_per_cm3) ||
          !finitePositive(photon.sum_charge) ||
          !finitePositive(photon.e_lpm_MeV) ||
          !finitePositive(photon.classical_electron_radius_cm) ||
          !finitePositive(photon.fine_structure_constant) ||
          photon.component_count == 0 ||
          photon.component_count > MaxPhotonPairLpmComponents ||
          photon.reserved != 0)
        throw std::runtime_error(
            "c8emaux photon-pair LPM payload is invalid");
      if (!finitePositive(brems.baseline_mass_density_g_per_cm3) ||
          !finitePositive(brems.molecular_density_per_cm3) ||
          !finitePositive(brems.sum_charge) ||
          !finitePositive(brems.e_lpm_MeV) ||
          !finitePositive(brems.lepton_mass_MeV) ||
          !finitePositive(brems.electron_mass_MeV) ||
          !finitePositive(brems.muon_mass_MeV) ||
          !finitePositive(brems.classical_electron_radius_cm) ||
          !finitePositive(brems.fine_structure_constant) ||
          brems.component_count == 0 ||
          brems.component_count > MaxBremsLpmComponents ||
          brems.reserved != 0)
        throw std::runtime_error(
            "c8emaux bremsstrahlung LPM payload is invalid");
      if (photon.component_count != brems.component_count)
        throw std::runtime_error(
            "c8emaux LPM component counts are inconsistent");
      for (std::uint32_t index = 0; index < photon.component_count; ++index) {
        auto const& photon_component = photon.components[index];
        auto const& brems_component = brems.components[index];
        if (photon_component.component_hash !=
                brems_component.component_hash ||
            !finitePositive(photon_component.nuclear_charge) ||
            !finitePositive(photon_component.radiation_log_constant) ||
            !finitePositive(brems_component.nuclear_charge) ||
            !finitePositive(brems_component.atomic_mass_number) ||
            !finitePositive(brems_component.radiation_log_constant))
          throw std::runtime_error(
              "c8emaux LPM component payload is invalid");
      }
      if (!moliere_detail::validSnapshot(payload.electron_moliere))
        throw std::runtime_error(
            "c8emaux electron Moliere payload is invalid");
      if (payload.has_muon_moliere > 1u)
        throw std::runtime_error(
            "c8emaux muon Moliere presence flag is invalid");
      if (payload.has_muon_moliere != 0u &&
          !moliere_detail::validSnapshot(payload.muon_moliere))
        throw std::runtime_error(
            "c8emaux muon Moliere payload is invalid");
      for (auto const value : payload.reserved)
        if (value != 0u)
          throw std::runtime_error("c8emaux reserved payload is nonzero");
    }

    class FileLock {
     public:
      explicit FileLock(std::filesystem::path const& path) : path_(path) {
        descriptor_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (descriptor_ < 0)
          throw std::runtime_error("cannot open c8emaux cache lock " +
                                   path.string() + ": " +
                                   std::strerror(errno));
        while (::flock(descriptor_, LOCK_EX) != 0) {
          if (errno == EINTR) continue;
          auto const message = std::string(std::strerror(errno));
          ::close(descriptor_);
          descriptor_ = -1;
          throw std::runtime_error("cannot acquire c8emaux cache lock " +
                                   path.string() + ": " + message);
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
      std::filesystem::path path;
      ~TemporaryFileGuard() {
        if (!path.empty()) {
          std::error_code ignored;
          std::filesystem::remove(path, ignored);
        }
      }
    };

    ProposalNativeAuxData fromPayload(
        AuxPayload const& payload, Sha256Digest const& key,
        Sha256Digest const& content, std::filesystem::path const& path,
        bool hit) {
      ProposalNativeAuxData result;
      result.photon_pair_lpm = payload.photon_pair_lpm;
      result.brems_lpm = payload.brems_lpm;
      result.electron_moliere = payload.electron_moliere;
      result.muon_moliere = payload.muon_moliere;
      result.has_muon_moliere = payload.has_muon_moliere;
      result.key_hash = key;
      result.content_hash = content;
      result.cache_file = path;
      result.cache_hit = hit;
      return result;
    }

    ProposalNativeAuxData readAux(std::filesystem::path const& path,
                                  Sha256Digest const& expected_key) {
      std::ifstream input(path, std::ios::binary);
      if (!input) throw std::runtime_error("cannot open c8emaux cache");
      AuxFileHeader header{};
      input.read(reinterpret_cast<char*>(&header), sizeof(header));
      if (!input || header.payload_bytes == 0 ||
          header.payload_bytes > (1u << 28))
        throw std::runtime_error("c8emaux cache payload size is invalid");
      std::vector<std::uint8_t> bytes(header.payload_bytes);
      input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
      if (!input || input.peek() != std::ifstream::traits_type::eof() ||
          std::memcmp(header.magic, AuxCacheMagic, sizeof(AuxCacheMagic)) != 0 ||
          header.format_version != ProposalNativeAuxFormatVersion ||
          header.key_hash != expected_key)
        throw std::runtime_error("c8emaux cache metadata is incompatible");
      auto const actual = sha256(bytes);
      if (actual != header.content_hash)
        throw std::runtime_error("c8emaux cache content hash mismatch");
      auto const payload = deserializePayload(bytes);
      validatePayload(payload);
      return fromPayload(payload, header.key_hash, header.content_hash, path,
                         true);
    }

    std::optional<ProposalNativeAuxData> tryReadAux(
        std::filesystem::path const& path, Sha256Digest const& expected_key) {
      if (!std::filesystem::exists(path)) return std::nullopt;
      try {
        return readAux(path, expected_key);
      } catch (std::runtime_error const&) {
        return std::nullopt;
      }
    }
  } // namespace

  std::filesystem::path defaultProposalNativeAuxCacheDirectory() {
    if (auto const* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
      auto const base = std::filesystem::path(xdg);
      if (!base.is_absolute())
        throw std::runtime_error("XDG_CACHE_HOME must be an absolute path");
      return base / "corsika8" / "gpu-em-aux";
    }
    if (auto const* home = std::getenv("HOME"); home && *home) {
      auto const base = std::filesystem::path(home);
      if (!base.is_absolute())
        throw std::runtime_error("HOME must be an absolute path");
      return base / ".cache" / "corsika8" / "gpu-em-aux";
    }
    throw std::runtime_error(
        "cannot determine XDG cache directory for proposal-native auxiliary data");
  }

  ProposalNativeAuxData loadOrCreateProposalNativeAux(
      std::vector<proposal::NativeInteractionCalculatorView> const& views,
      std::filesystem::path const& requested_directory) {
    auto const directory = requested_directory.empty()
                               ? defaultProposalNativeAuxCacheDirectory()
                               : requested_directory;
    auto const key = calculateKey(views);
    auto const path = directory / (toHex(key) + ".c8emaux");
    std::filesystem::create_directories(directory);
    if (auto cached = tryReadAux(path, key)) return *cached;

    auto lock_path = path;
    lock_path += ".lock.v2";
    FileLock lock(lock_path);
    if (auto cached = tryReadAux(path, key)) return *cached;

    auto const payload = generatePayload(views);
    validatePayload(payload);
    auto const bytes = serializePayload(payload);
    auto const content = sha256(bytes);
    AuxFileHeader header;
    if (bytes.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("c8emaux payload exceeds format limit");
    header.payload_bytes = static_cast<std::uint32_t>(bytes.size());
    header.key_hash = key;
    header.content_hash = content;
    auto temporary = path;
    temporary += ".tmp." + std::to_string(static_cast<long long>(getpid())) +
                 "." + std::to_string(
                           std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count());
    TemporaryFileGuard temporary_guard{temporary};
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      if (!output) throw std::runtime_error("cannot create c8emaux cache");
      output.write(reinterpret_cast<char const*>(&header), sizeof(header));
      output.write(reinterpret_cast<char const*>(bytes.data()), bytes.size());
      output.flush();
      if (!output) throw std::runtime_error("cannot write c8emaux cache");
    }
    auto const verified_temporary = readAux(temporary, key);
    if (verified_temporary.content_hash != content)
      throw std::runtime_error(
          "c8emaux write-after-read content hash mismatch");
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
      if (auto cached = tryReadAux(path, key)) return *cached;
      if (!std::filesystem::exists(path))
        throw std::runtime_error("cannot atomically install c8emaux cache: " +
                                 error.message());
      throw std::runtime_error(
          "c8emaux target exists but failed validation after rename error: " +
          error.message());
    }
    temporary_guard.path.clear();
    auto installed = readAux(path, key);
    installed.cache_hit = false;
    return installed;
  }

} // namespace corsika::gpu::em::tables
