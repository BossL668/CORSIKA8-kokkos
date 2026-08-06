/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/gpu/em/tables/RateTable.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>

#include <corsika/gpu/em/PhotonPairKinematics.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace corsika::gpu::em::tables {

  namespace {

    constexpr std::array<std::uint8_t, 8> Magic{'C', '8', 'E', 'M',
                                                'R', 'T', '1', '0'};
    constexpr std::array<std::uint8_t, 8> LegacyMagic{
        'C', '8', 'E', 'M', 'R', 'T', '0', '9'};
    constexpr std::uint32_t EndianMarker = 0x01020304U;
    constexpr std::uint64_t MaximumFileBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    constexpr std::uint64_t MaximumElements = 100000000ULL;
    constexpr std::uint64_t MaximumStringBytes = 1024ULL * 1024ULL;

    class Encoder {
    public:
      void u32(std::uint32_t value) {
        for (unsigned int shift = 0; shift < 32; shift += 8) {
          bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
        }
      }

      void i32(std::int32_t value) { u32(static_cast<std::uint32_t>(value)); }

      void u64(std::uint64_t value) {
        for (unsigned int shift = 0; shift < 64; shift += 8) {
          bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
        }
      }

      void f64(double value) {
        static_assert(sizeof(double) == sizeof(std::uint64_t));
        std::uint64_t bits{};
        std::memcpy(&bits, &value, sizeof(bits));
        u64(bits);
      }

      void string(std::string const& value) {
        if (value.size() > MaximumStringBytes) {
          throw std::length_error("rate-table string is too long");
        }
        u64(value.size());
        bytes_.insert(bytes_.end(), value.begin(), value.end());
      }

      template <typename T, typename Function>
      void vector(std::vector<T> const& values, Function encode) {
        if (values.size() > MaximumElements) {
          throw std::length_error("rate-table vector is too large");
        }
        u64(values.size());
        for (auto const& value : values) {
          encode(value);
        }
      }

      std::vector<std::uint8_t> take() { return std::move(bytes_); }

    private:
      std::vector<std::uint8_t> bytes_;
    };

    class Decoder {
    public:
      explicit Decoder(std::vector<std::uint8_t> const& bytes)
          : bytes_(bytes) {}

      std::uint32_t u32() {
        require(4);
        std::uint32_t value = 0;
        for (unsigned int i = 0; i < 4; ++i) {
          value |= static_cast<std::uint32_t>(bytes_[offset_++]) << (8U * i);
        }
        return value;
      }

      std::int32_t i32() { return static_cast<std::int32_t>(u32()); }

      std::uint64_t u64() {
        require(8);
        std::uint64_t value = 0;
        for (unsigned int i = 0; i < 8; ++i) {
          value |= static_cast<std::uint64_t>(bytes_[offset_++]) << (8U * i);
        }
        return value;
      }

      double f64() {
        auto const bits = u64();
        double value{};
        std::memcpy(&value, &bits, sizeof(value));
        return value;
      }

      std::string string() {
        auto const size = boundedCount(MaximumStringBytes, "string");
        require(size);
        std::string value(reinterpret_cast<char const*>(bytes_.data() + offset_),
                          static_cast<std::size_t>(size));
        offset_ += size;
        return value;
      }

      std::uint64_t count(char const* what) {
        return boundedCount(MaximumElements, what);
      }

      void expectEnd() const {
        if (offset_ != bytes_.size()) {
          throw std::runtime_error("rate-table payload contains trailing bytes");
        }
      }

    private:
      std::uint64_t boundedCount(std::uint64_t limit, char const* what) {
        auto const value = u64();
        if (value > limit) {
          throw std::runtime_error(std::string("rate-table ") + what +
                                   " count exceeds safety limit");
        }
        return value;
      }

      void require(std::uint64_t count) const {
        if (count > bytes_.size() - offset_) {
          throw std::runtime_error("truncated rate-table payload");
        }
      }

      std::vector<std::uint8_t> const& bytes_;
      std::size_t offset_{};
    };

    std::vector<std::uint8_t> encodePayload(RateTableSet const& table) {
      validateRateTable(table);
      Encoder encoder;
      auto const& metadata = table.metadata;
      encoder.string(metadata.proposal_version);
      encoder.string(metadata.generator_version);
      encoder.string(metadata.medium_name);
      encoder.u64(metadata.proposal_medium_hash);
      encoder.f64(metadata.energy_cut_MeV);
      encoder.f64(metadata.relative_v_cut);
      encoder.f64(metadata.energy_min_MeV);
      encoder.f64(metadata.energy_max_MeV);
      encoder.f64(metadata.requested_relative_tolerance);
      encoder.f64(metadata.measured_max_relative_error);
      encoder.f64(metadata.requested_loss_relative_tolerance);
      encoder.f64(metadata.measured_max_loss_relative_error);
      encoder.string(EnergyUnit);
      encoder.string(RateUnit);
      encoder.vector(metadata.components, [&](MediumComponent const& component) {
        encoder.i32(component.corsika_pid);
        encoder.u64(component.proposal_hash);
        encoder.f64(component.number_fraction);
        encoder.string(component.name);
      });
      auto const& lpm = metadata.photon_pair_lpm;
      encoder.f64(lpm.baseline_mass_density_g_per_cm3);
      encoder.f64(lpm.molecular_density_per_cm3);
      encoder.f64(lpm.sum_charge);
      encoder.f64(lpm.e_lpm_MeV);
      encoder.f64(lpm.classical_electron_radius_cm);
      encoder.f64(lpm.fine_structure_constant);
      encoder.vector(
          lpm.components,
          [&](PhotonPairLpmComponent const& component) {
            encoder.u64(component.proposal_hash);
            encoder.f64(component.nuclear_charge);
            encoder.f64(component.radiation_log_constant);
          });
      auto const& brems_lpm = metadata.brems_lpm;
      encoder.f64(
          brems_lpm.baseline_mass_density_g_per_cm3);
      encoder.f64(brems_lpm.molecular_density_per_cm3);
      encoder.f64(brems_lpm.sum_charge);
      encoder.f64(brems_lpm.e_lpm_MeV);
      encoder.f64(brems_lpm.lepton_mass_MeV);
      encoder.f64(brems_lpm.electron_mass_MeV);
      encoder.f64(brems_lpm.muon_mass_MeV);
      encoder.f64(
          brems_lpm.classical_electron_radius_cm);
      encoder.f64(brems_lpm.fine_structure_constant);
      encoder.vector(
          brems_lpm.components,
          [&](BremsLpmComponent const& component) {
            encoder.u64(component.proposal_hash);
            encoder.f64(component.nuclear_charge);
            encoder.f64(component.atomic_mass_number);
            encoder.f64(component.radiation_log_constant);
          });
      auto const& moliere = metadata.moliere;
      encoder.u32(moliere.enabled ? 1U : 0U);
      encoder.string(moliere.reference_mode);
      encoder.f64(moliere.particle_mass_MeV);
      encoder.f64(moliere.electron_mass_MeV);
      encoder.f64(moliere.fine_structure_constant);
      encoder.f64(moliere.avogadro_per_mol);
      encoder.f64(moliere.hbar_MeV_s);
      encoder.f64(moliere.speed_of_light_cm_per_s);
      encoder.f64(moliere.euler_mascheroni);
      encoder.vector(
          moliere.components,
          [&](MoliereComponentMetadata const& component) {
            encoder.u64(component.proposal_hash);
            encoder.f64(component.nuclear_charge);
            encoder.f64(component.atomic_mass_number);
            encoder.f64(component.atoms_in_molecule);
          });
      auto encodeCoefficients =
          [&](std::vector<double> const& coefficients) {
            encoder.vector(
                coefficients,
                [&](double value) { encoder.f64(value); });
          };
      encodeCoefficients(moliere.c1);
      encodeCoefficients(moliere.c2);
      encodeCoefficients(moliere.c2_large);
      encodeCoefficients(moliere.s2_large);
      encodeCoefficients(moliere.C1_large);
      auto const& epair_rho = table.epair_rho;
      encoder.u32(epair_rho.enabled ? 1U : 0U);
      encoder.string(epair_rho.reference_mode);
      encoder.f64(epair_rho.requested_max_normalized_error);
      encoder.f64(epair_rho.measured_max_normalized_error);
      encoder.vector(
          epair_rho.component_hashes,
          [&](std::uint64_t value) { encoder.u64(value); });
      encoder.vector(
          epair_rho.energies_MeV,
          [&](double value) { encoder.f64(value); });
      encoder.vector(
          epair_rho.v_coordinates,
          [&](double value) { encoder.f64(value); });
      encoder.vector(
          epair_rho.rho_quantiles,
          [&](double value) { encoder.f64(value); });
      encoder.vector(
          epair_rho.normalized_rho,
          [&](double value) { encoder.f64(value); });
      encoder.vector(
          table.continuous_energy_tables,
          [&](ContinuousEnergyTable const& continuous) {
            encoder.i32(continuous.pdg_id);
            encoder.string(continuous.particle_name);
            encoder.string(continuous.reference_mode);
            encoder.f64(continuous.mass_MeV);
            encoder.f64(
                continuous.minimum_total_energy_MeV);
            encoder.f64(
                continuous.measured_max_relative_error);
            encoder.vector(
                continuous.energies_MeV,
                [&](double value) { encoder.f64(value); });
            encoder.vector(
                continuous.dEdX_MeV_cm2_per_g,
                [&](double value) { encoder.f64(value); });
            encoder.vector(
                continuous.range_g_per_cm2,
                [&](double value) { encoder.f64(value); });
          });
      encoder.vector(table.particles, [&](ParticleRateTable const& particle) {
        encoder.i32(particle.pdg_id);
        encoder.string(particle.particle_name);
        encoder.u64(particle.interaction_hash);
        encoder.vector(particle.energies_MeV,
                       [&](double energy) { encoder.f64(energy); });
        encoder.vector(particle.columns, [&](RateColumn const& column) {
          encoder.i32(column.process_id);
          encoder.u64(column.component_hash);
          encoder.string(column.process_name);
          encoder.string(column.parameterization);
          encoder.string(column.target_name);
          encoder.vector(column.rates_cm2_per_g,
                         [&](double rate) { encoder.f64(rate); });
          encoder.string(column.inverse_cdf.reference_mode);
          encoder.vector(column.inverse_cdf.energies_MeV,
                         [&](double energy) { encoder.f64(energy); });
          encoder.vector(column.inverse_cdf.quantile_offsets,
                         [&](std::uint64_t offset) { encoder.u64(offset); });
          encoder.vector(column.inverse_cdf.quantiles,
                         [&](double quantile) { encoder.f64(quantile); });
          encoder.vector(column.inverse_cdf.v_loss,
                         [&](double loss) { encoder.f64(loss); });
        });
      });
      return encoder.take();
    }

    RateTableSet decodePayload(
        std::vector<std::uint8_t> const& payload,
        bool has_epair_rho) {
      Decoder decoder(payload);
      RateTableSet table;
      auto& metadata = table.metadata;
      metadata.proposal_version = decoder.string();
      metadata.generator_version = decoder.string();
      metadata.medium_name = decoder.string();
      metadata.proposal_medium_hash = decoder.u64();
      metadata.energy_cut_MeV = decoder.f64();
      metadata.relative_v_cut = decoder.f64();
      metadata.energy_min_MeV = decoder.f64();
      metadata.energy_max_MeV = decoder.f64();
      metadata.requested_relative_tolerance = decoder.f64();
      metadata.measured_max_relative_error = decoder.f64();
      metadata.requested_loss_relative_tolerance = decoder.f64();
      metadata.measured_max_loss_relative_error = decoder.f64();
      if (decoder.string() != EnergyUnit || decoder.string() != RateUnit) {
        throw std::runtime_error("rate-table unit metadata is unsupported");
      }

      auto const component_count = decoder.count("component");
      metadata.components.reserve(component_count);
      for (std::uint64_t i = 0; i < component_count; ++i) {
        metadata.components.push_back(
            {decoder.i32(), decoder.u64(), decoder.f64(), decoder.string()});
      }
      auto& lpm = metadata.photon_pair_lpm;
      lpm.baseline_mass_density_g_per_cm3 = decoder.f64();
      lpm.molecular_density_per_cm3 = decoder.f64();
      lpm.sum_charge = decoder.f64();
      lpm.e_lpm_MeV = decoder.f64();
      lpm.classical_electron_radius_cm = decoder.f64();
      lpm.fine_structure_constant = decoder.f64();
      auto const lpm_component_count =
          decoder.count("photon-pair LPM component");
      lpm.components.reserve(lpm_component_count);
      for (std::uint64_t i = 0; i < lpm_component_count; ++i) {
        lpm.components.push_back(
            {decoder.u64(), decoder.f64(), decoder.f64()});
      }
      auto& brems_lpm = metadata.brems_lpm;
      brems_lpm.baseline_mass_density_g_per_cm3 =
          decoder.f64();
      brems_lpm.molecular_density_per_cm3 = decoder.f64();
      brems_lpm.sum_charge = decoder.f64();
      brems_lpm.e_lpm_MeV = decoder.f64();
      brems_lpm.lepton_mass_MeV = decoder.f64();
      brems_lpm.electron_mass_MeV = decoder.f64();
      brems_lpm.muon_mass_MeV = decoder.f64();
      brems_lpm.classical_electron_radius_cm =
          decoder.f64();
      brems_lpm.fine_structure_constant = decoder.f64();
      auto const brems_lpm_component_count =
          decoder.count("bremsstrahlung LPM component");
      brems_lpm.components.reserve(
          brems_lpm_component_count);
      for (std::uint64_t i = 0;
           i < brems_lpm_component_count; ++i) {
        brems_lpm.components.push_back(
            {decoder.u64(), decoder.f64(), decoder.f64(),
             decoder.f64()});
      }
      auto& moliere = metadata.moliere;
      auto const moliere_enabled = decoder.u32();
      if (moliere_enabled > 1) {
        throw std::runtime_error(
            "invalid Moliere enabled flag");
      }
      moliere.enabled = moliere_enabled != 0;
      moliere.reference_mode = decoder.string();
      moliere.particle_mass_MeV = decoder.f64();
      moliere.electron_mass_MeV = decoder.f64();
      moliere.fine_structure_constant = decoder.f64();
      moliere.avogadro_per_mol = decoder.f64();
      moliere.hbar_MeV_s = decoder.f64();
      moliere.speed_of_light_cm_per_s = decoder.f64();
      moliere.euler_mascheroni = decoder.f64();
      auto const moliere_component_count =
          decoder.count("Moliere component");
      moliere.components.reserve(moliere_component_count);
      for (std::uint64_t i = 0;
           i < moliere_component_count; ++i) {
        moliere.components.push_back(
            {decoder.u64(), decoder.f64(), decoder.f64(),
             decoder.f64()});
      }
      auto decodeCoefficients =
          [&](std::vector<double>& coefficients,
              char const* name) {
            auto const count = decoder.count(name);
            coefficients.reserve(count);
            for (std::uint64_t i = 0; i < count; ++i) {
              coefficients.push_back(decoder.f64());
            }
          };
      decodeCoefficients(moliere.c1, "Moliere c1 coefficient");
      decodeCoefficients(moliere.c2, "Moliere c2 coefficient");
      decodeCoefficients(
          moliere.c2_large,
          "Moliere c2-large coefficient");
      decodeCoefficients(
          moliere.s2_large,
          "Moliere s2-large coefficient");
      decodeCoefficients(
          moliere.C1_large,
          "Moliere C1-large coefficient");

      if (has_epair_rho) {
        auto& epair_rho = table.epair_rho;
        auto const epair_rho_enabled = decoder.u32();
        if (epair_rho_enabled > 1) {
          throw std::runtime_error(
              "invalid Epair rho-table enabled flag");
        }
        epair_rho.enabled = epair_rho_enabled != 0;
        epair_rho.reference_mode = decoder.string();
        epair_rho.requested_max_normalized_error =
            decoder.f64();
        epair_rho.measured_max_normalized_error =
            decoder.f64();
        auto const epair_component_count =
            decoder.count("Epair rho component");
        epair_rho.component_hashes.reserve(
            epair_component_count);
        for (std::uint64_t i = 0; i < epair_component_count;
             ++i) {
          epair_rho.component_hashes.push_back(
              decoder.u64());
        }
        auto decodeEpairAxis =
            [&](std::vector<double>& values,
                char const* name) {
              auto const count = decoder.count(name);
              values.reserve(count);
              for (std::uint64_t i = 0; i < count; ++i) {
                values.push_back(decoder.f64());
              }
            };
        decodeEpairAxis(
            epair_rho.energies_MeV, "Epair rho energy");
        decodeEpairAxis(
            epair_rho.v_coordinates,
            "Epair rho v coordinate");
        decodeEpairAxis(
            epair_rho.rho_quantiles,
            "Epair rho quantile");
        decodeEpairAxis(
            epair_rho.normalized_rho,
            "Epair rho inverse-CDF value");
      }

      auto const continuous_count =
          decoder.count("continuous energy table");
      table.continuous_energy_tables.reserve(
          continuous_count);
      for (std::uint64_t i = 0; i < continuous_count; ++i) {
        ContinuousEnergyTable continuous;
        continuous.pdg_id = decoder.i32();
        continuous.particle_name = decoder.string();
        continuous.reference_mode = decoder.string();
        continuous.mass_MeV = decoder.f64();
        continuous.minimum_total_energy_MeV =
            decoder.f64();
        continuous.measured_max_relative_error =
            decoder.f64();
        auto const energy_count =
            decoder.count("continuous energy");
        continuous.energies_MeV.reserve(energy_count);
        for (std::uint64_t j = 0; j < energy_count; ++j) {
          continuous.energies_MeV.push_back(
              decoder.f64());
        }
        auto const dedx_count =
            decoder.count("continuous stopping power");
        continuous.dEdX_MeV_cm2_per_g.reserve(
            dedx_count);
        for (std::uint64_t j = 0; j < dedx_count; ++j) {
          continuous.dEdX_MeV_cm2_per_g.push_back(
              decoder.f64());
        }
        auto const range_count =
            decoder.count("continuous range");
        continuous.range_g_per_cm2.reserve(range_count);
        for (std::uint64_t j = 0; j < range_count; ++j) {
          continuous.range_g_per_cm2.push_back(
              decoder.f64());
        }
        table.continuous_energy_tables.push_back(
            std::move(continuous));
      }

      auto const particle_count = decoder.count("particle");
      table.particles.reserve(particle_count);
      for (std::uint64_t i = 0; i < particle_count; ++i) {
        ParticleRateTable particle;
        particle.pdg_id = decoder.i32();
        particle.particle_name = decoder.string();
        particle.interaction_hash = decoder.u64();
        auto const energy_count = decoder.count("energy");
        particle.energies_MeV.reserve(energy_count);
        for (std::uint64_t j = 0; j < energy_count; ++j) {
          particle.energies_MeV.push_back(decoder.f64());
        }
        auto const column_count = decoder.count("column");
        particle.columns.reserve(column_count);
        for (std::uint64_t j = 0; j < column_count; ++j) {
          RateColumn column;
          column.process_id = decoder.i32();
          column.component_hash = decoder.u64();
          column.process_name = decoder.string();
          column.parameterization = decoder.string();
          column.target_name = decoder.string();
          auto const rate_count = decoder.count("rate");
          column.rates_cm2_per_g.reserve(rate_count);
          for (std::uint64_t k = 0; k < rate_count; ++k) {
            column.rates_cm2_per_g.push_back(decoder.f64());
          }
          column.inverse_cdf.reference_mode = decoder.string();
          auto const loss_energy_count = decoder.count("loss energy");
          column.inverse_cdf.energies_MeV.reserve(loss_energy_count);
          for (std::uint64_t k = 0; k < loss_energy_count; ++k) {
            column.inverse_cdf.energies_MeV.push_back(decoder.f64());
          }
          auto const offset_count = decoder.count("loss quantile offset");
          column.inverse_cdf.quantile_offsets.reserve(offset_count);
          for (std::uint64_t k = 0; k < offset_count; ++k) {
            column.inverse_cdf.quantile_offsets.push_back(decoder.u64());
          }
          auto const quantile_count = decoder.count("loss quantile");
          column.inverse_cdf.quantiles.reserve(quantile_count);
          for (std::uint64_t k = 0; k < quantile_count; ++k) {
            column.inverse_cdf.quantiles.push_back(decoder.f64());
          }
          auto const loss_count = decoder.count("inverse-CDF value");
          column.inverse_cdf.v_loss.reserve(loss_count);
          for (std::uint64_t k = 0; k < loss_count; ++k) {
            column.inverse_cdf.v_loss.push_back(decoder.f64());
          }
          particle.columns.push_back(std::move(column));
        }
        table.particles.push_back(std::move(particle));
      }
      decoder.expectEnd();
      validateRateTable(table);
      return table;
    }

    void writeU32(std::ostream& output, std::uint32_t value) {
      std::array<char, 4> bytes{};
      for (unsigned int i = 0; i < 4; ++i) {
        bytes[i] = static_cast<char>(value >> (8U * i));
      }
      output.write(bytes.data(), bytes.size());
    }

    void writeU64(std::ostream& output, std::uint64_t value) {
      std::array<char, 8> bytes{};
      for (unsigned int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<char>(value >> (8U * i));
      }
      output.write(bytes.data(), bytes.size());
    }

    std::uint32_t readU32(std::istream& input) {
      std::array<std::uint8_t, 4> bytes{};
      input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
      if (!input) {
        throw std::runtime_error("truncated rate-table envelope");
      }
      std::uint32_t value = 0;
      for (unsigned int i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(bytes[i]) << (8U * i);
      }
      return value;
    }

    std::uint64_t readU64(std::istream& input) {
      std::array<std::uint8_t, 8> bytes{};
      input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
      if (!input) {
        throw std::runtime_error("truncated rate-table envelope");
      }
      std::uint64_t value = 0;
      for (unsigned int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
      }
      return value;
    }

    void requireFiniteNonnegative(double value, char const* name) {
      if (!std::isfinite(value) || value < 0.) {
        throw std::invalid_argument(std::string("invalid rate-table ") + name);
      }
    }

    double interpolateColumn(ParticleRateTable const& particle,
                             RateColumn const& column, double energy_MeV) {
      if (!std::isfinite(energy_MeV) ||
          energy_MeV < particle.energies_MeV.front() ||
          energy_MeV > particle.energies_MeV.back()) {
        throw std::out_of_range("rate-table energy is outside the stored domain");
      }
      auto const upper =
          std::lower_bound(particle.energies_MeV.begin(),
                           particle.energies_MeV.end(), energy_MeV);
      if (upper == particle.energies_MeV.begin()) {
        return column.rates_cm2_per_g.front();
      }
      if (upper == particle.energies_MeV.end()) {
        return column.rates_cm2_per_g.back();
      }
      auto const upper_index =
          static_cast<std::size_t>(upper - particle.energies_MeV.begin());
      if (*upper == energy_MeV) {
        return column.rates_cm2_per_g[upper_index];
      }
      auto const lower_index = upper_index - 1;
      auto const x0 = std::log(particle.energies_MeV[lower_index]);
      auto const x1 = std::log(particle.energies_MeV[upper_index]);
      auto const x = std::log(energy_MeV);
      auto const fraction = (x - x0) / (x1 - x0);
      auto const y0 = column.rates_cm2_per_g[lower_index];
      auto const y1 = column.rates_cm2_per_g[upper_index];
      if (y0 > 0. && y1 > 0.) {
        return std::exp(std::log(y0) + fraction * (std::log(y1) - std::log(y0)));
      }
      return y0 + fraction * (y1 - y0);
    }

    RateColumn const& findColumn(ParticleRateTable const& particle,
                                 std::int32_t process_id,
                                 std::uint64_t component_hash) {
      auto const column =
          std::find_if(particle.columns.begin(), particle.columns.end(),
                       [&](RateColumn const& candidate) {
                         return candidate.process_id == process_id &&
                                candidate.component_hash == component_hash;
                       });
      if (column == particle.columns.end()) {
        throw std::out_of_range(
            "rate-table process/component column was not found");
      }
      return *column;
    }

    enum class AxisTransform {
      Linear,
      Logarithmic,
      PhotonPairThresholdLogarithmic,
      Logit
    };

    double transformAxis(double value, AxisTransform transform) {
      if (transform == AxisTransform::Logarithmic) {
        return std::log(value);
      }
      if (transform ==
          AxisTransform::PhotonPairThresholdLogarithmic) {
        return std::log(value - PhotonPairThresholdMeV);
      }
      if (transform == AxisTransform::Logit) {
        return std::log(value / (1. - value));
      }
      return value;
    }

    std::pair<std::size_t, double> interpolationBracket(
        std::vector<double> const& grid, double value,
        AxisTransform transform, char const* name) {
      if (!std::isfinite(value) || value < grid.front() || value > grid.back()) {
        throw std::out_of_range(std::string(name) +
                                " is outside the stored inverse-CDF domain");
      }
      auto const upper = std::lower_bound(grid.begin(), grid.end(), value);
      if (upper == grid.begin()) {
        return {0, 0.};
      }
      if (upper == grid.end()) {
        return {grid.size() - 2, 1.};
      }
      auto const upper_index = static_cast<std::size_t>(upper - grid.begin());
      if (*upper == value) {
        if (upper_index == grid.size() - 1) {
          return {upper_index - 1, 1.};
        }
        return {upper_index, 0.};
      }
      auto const lower_index = upper_index - 1;
      auto const lower_value = transformAxis(grid[lower_index], transform);
      auto const upper_value = transformAxis(grid[upper_index], transform);
      auto const query = transformAxis(value, transform);
      return {lower_index,
              (query - lower_value) / (upper_value - lower_value)};
    }

    double interpolateLossPair(double lower, double upper, double fraction) {
      if (lower > 0. && upper > 0.) {
        return std::exp(std::log(lower) +
                        fraction * (std::log(upper) - std::log(lower)));
      }
      return lower + fraction * (upper - lower);
    }

    double interpolateInverseCdfRow(InverseCdfTable const& inverse,
                                    std::size_t energy_index,
                                    double quantile) {
      auto const begin_index = static_cast<std::size_t>(
          inverse.quantile_offsets[energy_index]);
      auto const end_index = static_cast<std::size_t>(
          inverse.quantile_offsets[energy_index + 1]);
      auto const begin = inverse.quantiles.begin() + begin_index;
      auto const end = inverse.quantiles.begin() + end_index;
      auto const upper = std::lower_bound(begin, end, quantile);
      if (upper == begin) {
        return inverse.v_loss[begin_index];
      }
      if (upper == end) {
        return inverse.v_loss[end_index - 1];
      }
      auto const upper_index =
          static_cast<std::size_t>(upper - inverse.quantiles.begin());
      if (*upper == quantile) {
        return inverse.v_loss[upper_index];
      }
      auto const lower_index = upper_index - 1;
      auto const lower_coordinate =
          transformAxis(inverse.quantiles[lower_index], AxisTransform::Logit);
      auto const upper_coordinate =
          transformAxis(inverse.quantiles[upper_index], AxisTransform::Logit);
      auto const query_coordinate =
          transformAxis(quantile, AxisTransform::Logit);
      auto const fraction =
          (query_coordinate - lower_coordinate) /
          (upper_coordinate - lower_coordinate);
      return interpolateLossPair(inverse.v_loss[lower_index],
                                 inverse.v_loss[upper_index], fraction);
    }

    double interpolateInverseCdf(
        InverseCdfTable const& inverse, double energy_MeV,
        double quantile, bool photon_pair_final_state) {
      auto const first_row_end =
          static_cast<std::size_t>(inverse.quantile_offsets[1]);
      if (quantile < inverse.quantiles.front() ||
          quantile > inverse.quantiles[first_row_end - 1]) {
        throw std::out_of_range(
            "inverse-CDF quantile is outside the stored capability domain");
      }
      auto const [energy_index, energy_fraction] =
          interpolationBracket(inverse.energies_MeV, energy_MeV,
                               photon_pair_final_state
                                   ? AxisTransform::
                                         PhotonPairThresholdLogarithmic
                                   : AxisTransform::Logarithmic,
                               "inverse-CDF energy");
      auto const lower =
          interpolateInverseCdfRow(inverse, energy_index, quantile);
      auto const upper =
          interpolateInverseCdfRow(inverse, energy_index + 1, quantile);
      auto const result =
          interpolateLossPair(lower, upper, energy_fraction);
      if (!std::isfinite(result) || result < -1.e-14 || result > 1. + 1.e-14) {
        throw std::runtime_error(
            "inverse-CDF interpolation produced an invalid loss fraction");
      }
      return std::clamp(result, 0., 1.);
    }

  } // namespace

  void validateRateTable(RateTableSet const& table) {
    auto const& metadata = table.metadata;
    if (metadata.proposal_version.empty() || metadata.generator_version.empty() ||
        metadata.medium_name.empty()) {
      throw std::invalid_argument("rate-table metadata strings must not be empty");
    }
    requireFiniteNonnegative(metadata.energy_cut_MeV, "energy cut");
    requireFiniteNonnegative(metadata.relative_v_cut, "relative v cut");
    requireFiniteNonnegative(metadata.measured_max_relative_error,
                             "measured interpolation error");
    requireFiniteNonnegative(metadata.measured_max_loss_relative_error,
                             "measured inverse-CDF interpolation error");
    if (!std::isfinite(metadata.energy_min_MeV) ||
        !std::isfinite(metadata.energy_max_MeV) ||
        !(metadata.energy_min_MeV > 0.) ||
        !(metadata.energy_max_MeV > metadata.energy_min_MeV)) {
      throw std::invalid_argument("invalid rate-table energy domain");
    }
    if (!std::isfinite(metadata.requested_relative_tolerance) ||
        !(metadata.requested_relative_tolerance > 0.) ||
        !(metadata.requested_relative_tolerance <= 1.)) {
      throw std::invalid_argument("invalid requested rate-table tolerance");
    }
    if (!std::isfinite(metadata.requested_loss_relative_tolerance) ||
        !(metadata.requested_loss_relative_tolerance > 0.) ||
        !(metadata.requested_loss_relative_tolerance <= 1.)) {
      throw std::invalid_argument(
          "invalid requested inverse-CDF table tolerance");
    }
    if (metadata.proposal_medium_hash == 0) {
      throw std::invalid_argument(
          "rate-table has no PROPOSAL medium hash");
    }
    if (metadata.components.empty() || table.particles.empty()) {
      throw std::invalid_argument("rate-table medium and particle lists must not be empty");
    }

    double fraction_sum = 0.;
    std::set<std::uint64_t> component_hashes;
    for (auto const& component : metadata.components) {
      if (component.corsika_pid == 0 || component.proposal_hash == 0 ||
          component.name.empty()) {
        throw std::invalid_argument("invalid rate-table medium component identity");
      }
      requireFiniteNonnegative(component.number_fraction, "component fraction");
      fraction_sum += component.number_fraction;
      if (!component_hashes.insert(component.proposal_hash).second) {
        throw std::invalid_argument("duplicate rate-table medium component hash");
      }
    }
    if (std::abs(fraction_sum - 1.) > 1.e-9) {
      throw std::invalid_argument("rate-table medium fractions do not sum to one");
    }

    auto const& lpm = metadata.photon_pair_lpm;
    auto requirePositive = [](double value, char const* name) {
      if (!std::isfinite(value) || !(value > 0.)) {
        throw std::invalid_argument(
            std::string("invalid photon-pair LPM ") + name);
      }
    };
    requirePositive(lpm.baseline_mass_density_g_per_cm3,
                    "baseline mass density");
    requirePositive(lpm.molecular_density_per_cm3,
                    "molecular density");
    requirePositive(lpm.sum_charge, "sum charge");
    requirePositive(lpm.e_lpm_MeV, "characteristic energy");
    requirePositive(lpm.classical_electron_radius_cm,
                    "classical electron radius");
    requirePositive(lpm.fine_structure_constant,
                    "fine-structure constant");
    if (lpm.components.size() != metadata.components.size()) {
      throw std::invalid_argument(
          "photon-pair LPM component list differs from medium");
    }
    std::set<std::uint64_t> lpm_component_hashes;
    for (auto const& component : lpm.components) {
      if (component.proposal_hash == 0 ||
          component_hashes.count(component.proposal_hash) == 0 ||
          !lpm_component_hashes.insert(component.proposal_hash).second) {
        throw std::invalid_argument(
            "invalid photon-pair LPM component identity");
      }
      requirePositive(component.nuclear_charge,
                      "component nuclear charge");
      requirePositive(component.radiation_log_constant,
                      "component radiation logarithm");
    }

    auto const& brems_lpm = metadata.brems_lpm;
    auto requireBremsPositive =
        [](double value, char const* name) {
          if (!std::isfinite(value) || !(value > 0.)) {
            throw std::invalid_argument(
                std::string("invalid bremsstrahlung LPM ") +
                name);
          }
        };
    requireBremsPositive(
        brems_lpm.baseline_mass_density_g_per_cm3,
        "baseline mass density");
    requireBremsPositive(
        brems_lpm.molecular_density_per_cm3,
        "molecular density");
    requireBremsPositive(brems_lpm.sum_charge, "sum charge");
    requireBremsPositive(
        brems_lpm.e_lpm_MeV, "characteristic energy");
    requireBremsPositive(
        brems_lpm.lepton_mass_MeV, "lepton mass");
    requireBremsPositive(
        brems_lpm.electron_mass_MeV, "electron mass");
    requireBremsPositive(
        brems_lpm.muon_mass_MeV, "muon mass");
    requireBremsPositive(
        brems_lpm.classical_electron_radius_cm,
        "classical electron radius");
    requireBremsPositive(
        brems_lpm.fine_structure_constant,
        "fine-structure constant");
    if (brems_lpm.components.size() !=
        metadata.components.size()) {
      throw std::invalid_argument(
          "bremsstrahlung LPM component list differs from medium");
    }
    std::set<std::uint64_t> brems_lpm_component_hashes;
    for (auto const& component : brems_lpm.components) {
      if (component.proposal_hash == 0 ||
          component_hashes.count(component.proposal_hash) == 0 ||
          !brems_lpm_component_hashes
               .insert(component.proposal_hash)
               .second) {
        throw std::invalid_argument(
            "invalid bremsstrahlung LPM component identity");
      }
      requireBremsPositive(
          component.nuclear_charge,
          "component nuclear charge");
      requireBremsPositive(
          component.atomic_mass_number,
          "component atomic mass number");
      requireBremsPositive(
          component.radiation_log_constant,
          "component radiation logarithm");
    }

    auto const& moliere = metadata.moliere;
    if (!moliere.enabled) {
      if (!moliere.reference_mode.empty() ||
          !moliere.components.empty() ||
          !moliere.c1.empty() || !moliere.c2.empty() ||
          !moliere.c2_large.empty() ||
          !moliere.s2_large.empty() ||
          !moliere.C1_large.empty()) {
        throw std::invalid_argument(
            "disabled Moliere metadata contains numerical data");
      }
    } else {
      auto requireMolierePositive =
          [](double value, char const* name) {
            if (!std::isfinite(value) || !(value > 0.)) {
              throw std::invalid_argument(
                  std::string("invalid Moliere ") + name);
            }
          };
      if (moliere.reference_mode != "proposal_analytic") {
        throw std::invalid_argument(
            "unsupported Moliere reference mode");
      }
      requireMolierePositive(
          moliere.particle_mass_MeV, "particle mass");
      requireMolierePositive(
          moliere.electron_mass_MeV, "electron mass");
      requireMolierePositive(
          moliere.fine_structure_constant,
          "fine-structure constant");
      requireMolierePositive(
          moliere.avogadro_per_mol, "Avogadro constant");
      requireMolierePositive(
          moliere.hbar_MeV_s, "reduced Planck constant");
      requireMolierePositive(
          moliere.speed_of_light_cm_per_s,
          "speed of light");
      if (!std::isfinite(moliere.euler_mascheroni)) {
        throw std::invalid_argument(
            "invalid Moliere Euler-Mascheroni constant");
      }
      if (moliere.components.size() !=
          metadata.components.size()) {
        throw std::invalid_argument(
            "Moliere component list differs from medium");
      }
      std::set<std::uint64_t> moliere_component_hashes;
      for (auto const& component : moliere.components) {
        if (component.proposal_hash == 0 ||
            component_hashes.count(
                component.proposal_hash) == 0 ||
            !moliere_component_hashes
                 .insert(component.proposal_hash)
                 .second) {
          throw std::invalid_argument(
              "invalid Moliere component identity");
        }
        requireMolierePositive(
            component.nuclear_charge,
            "component nuclear charge");
        requireMolierePositive(
            component.atomic_mass_number,
            "component atomic mass number");
        requireMolierePositive(
            component.atoms_in_molecule,
            "component atom count");
      }
      auto validateCoefficients =
          [](std::vector<double> const& values,
             std::size_t expected, char const* name) {
            if (values.size() != expected ||
                !std::all_of(
                    values.begin(), values.end(),
                    [](double value) {
                      return std::isfinite(value);
                    })) {
              throw std::invalid_argument(
                  std::string("invalid Moliere ") + name);
            }
          };
      validateCoefficients(
          moliere.c1, 70, "c1 coefficients");
      validateCoefficients(
          moliere.c2, 70, "c2 coefficients");
      validateCoefficients(
          moliere.c2_large, 13,
          "c2-large coefficients");
      validateCoefficients(
          moliere.s2_large, 13,
          "s2-large coefficients");
      validateCoefficients(
          moliere.C1_large, 15,
          "C1-large coefficients");
    }

    auto const& epair_rho = table.epair_rho;
    if (!epair_rho.enabled) {
      if (!epair_rho.reference_mode.empty() ||
          epair_rho.requested_max_normalized_error != 0. ||
          epair_rho.measured_max_normalized_error != 0. ||
          !epair_rho.component_hashes.empty() ||
          !epair_rho.energies_MeV.empty() ||
          !epair_rho.v_coordinates.empty() ||
          !epair_rho.rho_quantiles.empty() ||
          !epair_rho.normalized_rho.empty()) {
        throw std::invalid_argument(
            "disabled Epair rho table contains numerical data");
      }
    } else {
      if (epair_rho.reference_mode !=
          "proposal_compatible_kkp") {
        throw std::invalid_argument(
            "unsupported Epair rho-table reference mode");
      }
      if (!std::isfinite(
              epair_rho.requested_max_normalized_error) ||
          !(epair_rho.requested_max_normalized_error > 0.) ||
          epair_rho.requested_max_normalized_error >
              metadata.requested_loss_relative_tolerance ||
          !std::isfinite(
              epair_rho.measured_max_normalized_error) ||
          epair_rho.measured_max_normalized_error < 0. ||
          epair_rho.measured_max_normalized_error >
              epair_rho.requested_max_normalized_error) {
        throw std::invalid_argument(
            "invalid Epair rho-table interpolation error");
      }
      if (epair_rho.component_hashes.size() !=
              metadata.components.size() ||
          epair_rho.energies_MeV.size() < 2 ||
          epair_rho.v_coordinates.size() < 2 ||
          epair_rho.rho_quantiles.size() < 2) {
        throw std::invalid_argument(
            "Epair rho-table axes are incomplete");
      }
      std::set<std::uint64_t> epair_component_hashes;
      for (auto const hash : epair_rho.component_hashes) {
        if (hash == 0 || component_hashes.count(hash) == 0 ||
            !epair_component_hashes.insert(hash).second) {
          throw std::invalid_argument(
              "invalid Epair rho-table component identity");
        }
      }
      auto validateAxis =
          [](std::vector<double> const& axis,
             char const* name) {
            for (std::size_t index = 0;
                 index < axis.size(); ++index) {
              if (!std::isfinite(axis[index]) ||
                  (index != 0 &&
                   !(axis[index] > axis[index - 1]))) {
                throw std::invalid_argument(
                    std::string("invalid Epair rho-table ") +
                    name);
              }
            }
          };
      validateAxis(epair_rho.energies_MeV, "energy axis");
      validateAxis(
          epair_rho.v_coordinates, "v-coordinate axis");
      validateAxis(
          epair_rho.rho_quantiles, "rho-quantile axis");
      if (epair_rho.energies_MeV.front() <
              metadata.energy_min_MeV ||
          epair_rho.energies_MeV.back() >
              metadata.energy_max_MeV ||
          !(epair_rho.v_coordinates.front() > 0.) ||
          epair_rho.rho_quantiles.front() !=
              EpairRhoQuantileMinimum ||
          epair_rho.rho_quantiles.back() !=
              EpairRhoQuantileMaximum) {
        throw std::invalid_argument(
            "Epair rho-table domain is inconsistent");
      }
      std::uint64_t expected_values =
          epair_rho.component_hashes.size();
      for (auto const count :
           {epair_rho.energies_MeV.size(),
            epair_rho.v_coordinates.size(),
            epair_rho.rho_quantiles.size()}) {
        if (count != 0 &&
            expected_values >
                MaximumElements /
                    static_cast<std::uint64_t>(count)) {
          throw std::invalid_argument(
              "Epair rho-table dimensions overflow");
        }
        expected_values *= count;
      }
      if (expected_values !=
          epair_rho.normalized_rho.size()) {
        throw std::invalid_argument(
            "Epair rho-table value count is inconsistent");
      }
      auto const rho_count =
          epair_rho.rho_quantiles.size();
      for (std::size_t row = 0;
           row < epair_rho.normalized_rho.size() / rho_count;
           ++row) {
        double previous = -1.;
        for (std::size_t rho_index = 0;
             rho_index < rho_count; ++rho_index) {
          auto const value =
              epair_rho.normalized_rho[
                  row * rho_count + rho_index];
          if (!std::isfinite(value) || value < 0. ||
              value > 1. ||
              value + 1.e-14 < previous) {
            throw std::invalid_argument(
                "invalid or non-monotonic Epair rho-table row");
          }
          previous = value;
        }
      }
    }

    if (table.continuous_energy_tables.empty()) {
      throw std::invalid_argument(
          "rate-table has no continuous charged-particle data");
    }
    std::set<std::int32_t> continuous_particle_ids;
    for (auto const& continuous :
         table.continuous_energy_tables) {
      if ((continuous.pdg_id != 11 &&
           continuous.pdg_id != -11 &&
           continuous.pdg_id != 13 &&
           continuous.pdg_id != -13) ||
          continuous.particle_name.empty() ||
          continuous.reference_mode !=
              "proposal_interpolated" ||
          !continuous_particle_ids
               .insert(continuous.pdg_id)
               .second) {
        throw std::invalid_argument(
            "invalid or duplicate continuous energy table identity");
      }
      requireBremsPositive(
          continuous.mass_MeV,
          "continuous-table particle mass");
      requireBremsPositive(
          continuous.minimum_total_energy_MeV,
          "continuous-table minimum total energy");
      requireFiniteNonnegative(
          continuous.measured_max_relative_error,
          "continuous-table measured interpolation error");
      if (continuous.measured_max_relative_error >
          metadata.requested_relative_tolerance) {
        throw std::invalid_argument(
            "continuous energy table exceeds requested tolerance");
      }
      auto const count = continuous.energies_MeV.size();
      if (count < 2 ||
          continuous.dEdX_MeV_cm2_per_g.size() != count ||
          continuous.range_g_per_cm2.size() != count) {
        throw std::invalid_argument(
            "continuous energy table dimensions are inconsistent");
      }
      if (!(continuous.minimum_total_energy_MeV >
            continuous.mass_MeV) ||
          continuous.energies_MeV.front() !=
              continuous.minimum_total_energy_MeV ||
          continuous.energies_MeV.back() !=
              metadata.energy_max_MeV ||
          continuous.energies_MeV.front() <
              metadata.energy_min_MeV ||
          continuous.range_g_per_cm2.front() != 0.) {
        throw std::invalid_argument(
            "continuous energy table domain is inconsistent");
      }
      for (std::size_t index = 0; index < count; ++index) {
        auto const energy = continuous.energies_MeV[index];
        auto const dedx =
            continuous.dEdX_MeV_cm2_per_g[index];
        auto const range =
            continuous.range_g_per_cm2[index];
        if (!std::isfinite(energy) ||
            !std::isfinite(dedx) ||
            !std::isfinite(range) ||
            !(energy > continuous.mass_MeV) ||
            !(dedx > 0.) || range < 0. ||
            (index != 0 &&
             (!(energy >
                continuous.energies_MeV[index - 1]) ||
              !(range >
                continuous.range_g_per_cm2[index - 1])))) {
          throw std::invalid_argument(
              "continuous energy/range grid is not strictly increasing");
        }
      }
    }
    auto const electron_tables =
        continuous_particle_ids.count(11) != 0 ||
        continuous_particle_ids.count(-11) != 0;
    auto const muon_tables =
        continuous_particle_ids.count(13) != 0 ||
        continuous_particle_ids.count(-13) != 0;
    if ((electron_tables &&
         (continuous_particle_ids.count(11) == 0 ||
          continuous_particle_ids.count(-11) == 0)) ||
        (muon_tables &&
         (continuous_particle_ids.count(13) == 0 ||
          continuous_particle_ids.count(-13) == 0))) {
      throw std::invalid_argument(
          "continuous energy tables do not cover both signs of a charged-lepton "
          "family");
    }

    std::set<std::int32_t> particle_ids;
    for (auto const& particle : table.particles) {
      if (particle.pdg_id == 0 || particle.particle_name.empty() ||
          particle.interaction_hash == 0 ||
          !particle_ids.insert(particle.pdg_id).second) {
        throw std::invalid_argument("invalid or duplicate rate-table particle");
      }
      if (particle.energies_MeV.size() < 2 || particle.columns.empty()) {
        throw std::invalid_argument("rate-table particle has an empty grid or columns");
      }
      for (std::size_t i = 0; i < particle.energies_MeV.size(); ++i) {
        auto const energy = particle.energies_MeV[i];
        if (!std::isfinite(energy) || !(energy > 0.) ||
            (i != 0 && !(energy > particle.energies_MeV[i - 1]))) {
          throw std::invalid_argument(
              "rate-table particle energies are not strictly increasing");
        }
      }
      auto const muon_domain =
          (particle.pdg_id == 13 ||
           particle.pdg_id == -13) &&
          particle.energies_MeV.front() >=
              metadata.energy_min_MeV &&
          continuous_particle_ids.count(
              particle.pdg_id) != 0;
      if ((!muon_domain &&
           particle.energies_MeV.front() !=
               metadata.energy_min_MeV) ||
          particle.energies_MeV.back() != metadata.energy_max_MeV) {
        throw std::invalid_argument(
            "rate-table particle grid does not match metadata domain");
      }

      std::set<std::pair<std::int32_t, std::uint64_t>> keys;
      for (auto const& column : particle.columns) {
        if (column.process_id == 0 || column.component_hash == 0 ||
            column.process_name.empty() || column.parameterization.empty() ||
            column.target_name.empty()) {
          throw std::invalid_argument("invalid rate-table column identity");
        }
        if (!keys.emplace(column.process_id, column.component_hash).second) {
          throw std::invalid_argument("duplicate rate-table process/component column");
        }
        if (column.rates_cm2_per_g.size() != particle.energies_MeV.size()) {
          throw std::invalid_argument("rate-table column length differs from energy grid");
        }
        for (auto const rate : column.rates_cm2_per_g) {
          requireFiniteNonnegative(rate, "interaction rate");
        }

        auto const& inverse = column.inverse_cdf;
        auto const has_positive_rate =
            std::any_of(column.rates_cm2_per_g.begin(),
                        column.rates_cm2_per_g.end(),
                        [](double rate) { return rate > 0.; });
        auto const inverse_arrays_empty =
            inverse.energies_MeV.empty() &&
            inverse.quantile_offsets.empty() &&
            inverse.quantiles.empty() && inverse.v_loss.empty();
        if (inverse.reference_mode ==
            SelectedLossCpuFallbackReferenceMode) {
          if (!inverse_arrays_empty || !has_positive_rate) {
            throw std::invalid_argument(
                "selected-loss CPU fallback column must have a positive "
                "rate and no inverse-CDF arrays");
          }
          continue;
        }
        if (inverse.reference_mode.empty() &&
            inverse_arrays_empty) {
          auto const selected_process_cpu_fallback =
              (particle.pdg_id == 13 ||
               particle.pdg_id == -13) &&
              column.process_id !=
                  IonizationProcessId;
          if (has_positive_rate &&
              !selected_process_cpu_fallback) {
            throw std::invalid_argument(
                "positive rate column has no inverse-CDF table");
          }
          continue;
        }
        if (inverse.reference_mode != "proposal_interpolated" &&
            inverse.reference_mode != "proposal_direct" &&
            inverse.reference_mode !=
                "proposal_interpolated_monotone") {
          throw std::invalid_argument(
              "inverse-CDF reference mode is unsupported");
        }
        if (inverse.energies_MeV.size() < 2) {
          throw std::invalid_argument(
              "inverse-CDF energy grid needs at least two points");
        }
        for (std::size_t i = 0; i < inverse.energies_MeV.size(); ++i) {
          auto const energy = inverse.energies_MeV[i];
          if (!std::isfinite(energy) || energy < metadata.energy_min_MeV ||
              energy > metadata.energy_max_MeV ||
              (i != 0 && !(energy > inverse.energies_MeV[i - 1]))) {
            throw std::invalid_argument(
                "invalid inverse-CDF energy grid");
          }
        }
        if (inverse.quantile_offsets.size() !=
                inverse.energies_MeV.size() + 1 ||
            inverse.quantile_offsets.front() != 0 ||
            inverse.quantile_offsets.back() != inverse.quantiles.size() ||
            inverse.v_loss.size() != inverse.quantiles.size()) {
          throw std::invalid_argument(
              "inverse-CDF ragged offsets or value count are inconsistent");
        }
        for (std::size_t energy_index = 0;
             energy_index < inverse.energies_MeV.size(); ++energy_index) {
          auto const row_begin = static_cast<std::size_t>(
              inverse.quantile_offsets[energy_index]);
          auto const row_end = static_cast<std::size_t>(
              inverse.quantile_offsets[energy_index + 1]);
          if (row_end < row_begin || row_end - row_begin < 2 ||
              row_end > inverse.quantiles.size()) {
            throw std::invalid_argument(
                "inverse-CDF ragged row has invalid bounds");
          }
          auto const minimum_quantile = inverse.quantiles.front();
          auto const maximum_quantile =
              inverse.quantiles[static_cast<std::size_t>(
                  inverse.quantile_offsets[1]) -
                                1];
          if (inverse.quantiles[row_begin] != minimum_quantile ||
              inverse.quantiles[row_end - 1] != maximum_quantile ||
              minimum_quantile < PhiloxLossQuantileMinimum ||
              maximum_quantile > PhiloxLossQuantileMaximum ||
              !(maximum_quantile > minimum_quantile)) {
            throw std::invalid_argument(
                "inverse-CDF quantile row has invalid endpoints");
          }
          double previous = -1.;
          for (std::size_t quantile_index = row_begin;
               quantile_index < row_end; ++quantile_index) {
            auto const quantile = inverse.quantiles[quantile_index];
            if (!std::isfinite(quantile) ||
                quantile < PhiloxLossQuantileMinimum ||
                quantile > PhiloxLossQuantileMaximum ||
                (quantile_index != row_begin &&
                 !(quantile >
                   inverse.quantiles[quantile_index - 1]))) {
              throw std::invalid_argument(
                  "invalid inverse-CDF quantile row");
            }
            auto const loss = inverse.v_loss[quantile_index];
            if (!std::isfinite(loss) || loss < 0. || loss > 1. ||
                loss + 1.e-14 < previous) {
              throw std::invalid_argument(
                  "invalid or non-monotonic inverse-CDF loss row");
            }
            previous = loss;
          }
        }
      }
    }
  }

  void validateCompatibility(RateTableSet const& table,
                             RateTableRequirements const& requirements) {
    validateRateTable(table);
    if (requirements.proposal_version.empty() ||
        requirements.medium_name.empty() ||
        requirements.proposal_medium_hash == 0 ||
        requirements.components.empty()) {
      throw std::invalid_argument(
          "rate-table compatibility requirements are incomplete");
    }
    requireFiniteNonnegative(requirements.energy_cut_MeV,
                             "required energy cut");
    requireFiniteNonnegative(requirements.relative_v_cut,
                             "required relative v cut");
    if (!std::isfinite(requirements.required_energy_min_MeV) ||
        !std::isfinite(requirements.required_energy_max_MeV) ||
        !(requirements.required_energy_min_MeV > 0.) ||
        !(requirements.required_energy_max_MeV >
          requirements.required_energy_min_MeV) ||
        !std::isfinite(requirements.maximum_relative_error) ||
        !(requirements.maximum_relative_error > 0.) ||
        !(requirements.maximum_relative_error <= 1.) ||
        !std::isfinite(requirements.maximum_loss_relative_error) ||
        !(requirements.maximum_loss_relative_error > 0.) ||
        !(requirements.maximum_loss_relative_error <= 1.)) {
      throw std::invalid_argument(
          "invalid rate-table domain or error compatibility requirement");
    }

    auto const& metadata = table.metadata;
    if (metadata.proposal_version != requirements.proposal_version) {
      throw std::runtime_error("rate-table PROPOSAL version mismatch");
    }
    if (metadata.medium_name != requirements.medium_name) {
      throw std::runtime_error("rate-table medium mismatch");
    }
    if (metadata.proposal_medium_hash !=
        requirements.proposal_medium_hash) {
      throw std::runtime_error("rate-table PROPOSAL medium hash mismatch");
    }
    if (metadata.energy_cut_MeV != requirements.energy_cut_MeV ||
        metadata.relative_v_cut != requirements.relative_v_cut) {
      throw std::runtime_error("rate-table cut configuration mismatch");
    }
    if (metadata.energy_min_MeV > requirements.required_energy_min_MeV ||
        metadata.energy_max_MeV < requirements.required_energy_max_MeV) {
      throw std::runtime_error("rate-table energy domain is insufficient");
    }
    if (metadata.requested_relative_tolerance >
            requirements.maximum_relative_error ||
        metadata.measured_max_relative_error >
            requirements.maximum_relative_error) {
      throw std::runtime_error("rate-table interpolation tolerance is insufficient");
    }
    if (metadata.requested_loss_relative_tolerance >
            requirements.maximum_loss_relative_error ||
        metadata.measured_max_loss_relative_error >
            requirements.maximum_loss_relative_error) {
      throw std::runtime_error(
          "inverse-CDF interpolation tolerance is insufficient");
    }
    if (metadata.components.size() != requirements.components.size()) {
      throw std::runtime_error("rate-table medium composition size mismatch");
    }
    for (std::size_t i = 0; i < metadata.components.size(); ++i) {
      auto const& stored = metadata.components[i];
      auto const& required = requirements.components[i];
      if (stored.corsika_pid != required.corsika_pid ||
          stored.proposal_hash != required.proposal_hash ||
          stored.number_fraction != required.number_fraction ||
          stored.name != required.name) {
        throw std::runtime_error("rate-table medium composition mismatch");
      }
    }
  }

  double interpolateRate(ParticleRateTable const& particle,
                         std::int32_t process_id,
                         std::uint64_t component_hash,
                         double energy_MeV) {
    return interpolateColumn(
        particle, findColumn(particle, process_id, component_hash), energy_MeV);
  }

  double interpolateTotalRate(ParticleRateTable const& particle,
                              double energy_MeV) {
    double total = 0.;
    for (auto const& column : particle.columns) {
      total += interpolateColumn(particle, column, energy_MeV);
    }
    if (!std::isfinite(total)) {
      throw std::runtime_error("rate-table interpolation produced a non-finite total");
    }
    return total;
  }

  double interpolateLossFraction(ParticleRateTable const& particle,
                                 std::int32_t process_id,
                                 std::uint64_t component_hash,
                                 double energy_MeV, double quantile) {
    if (!std::isfinite(quantile) ||
        quantile < LossQuantileMinimum ||
        quantile > LossQuantileMaximum) {
      throw std::invalid_argument(
          "inverse-CDF quantile is outside the device Philox support");
    }
    auto const& column = findColumn(particle, process_id, component_hash);
    if (column.inverse_cdf.energies_MeV.empty()) {
      throw std::out_of_range(
          "rate-table process/component has no active inverse-CDF domain");
    }
    return interpolateInverseCdf(
        column.inverse_cdf, energy_MeV, quantile,
        process_id == PhotonPairFinalStateProcessId);
  }

  ParticleRateTable const& findParticle(RateTableSet const& table,
                                        std::int32_t pdg_id) {
    auto const particle =
        std::find_if(table.particles.begin(), table.particles.end(),
                     [&](ParticleRateTable const& candidate) {
                       return candidate.pdg_id == pdg_id;
                     });
    if (particle == table.particles.end()) {
      throw std::out_of_range("particle is not present in rate table");
    }
    return *particle;
  }

  ContinuousEnergyTable const& findContinuousEnergyTable(
      RateTableSet const& table, std::int32_t pdg_id) {
    auto const found = std::find_if(
        table.continuous_energy_tables.begin(),
        table.continuous_energy_tables.end(),
        [&](ContinuousEnergyTable const& candidate) {
          return candidate.pdg_id == pdg_id;
        });
    if (found == table.continuous_energy_tables.end()) {
      throw std::out_of_range(
          "particle is not present in continuous energy tables");
    }
    return *found;
  }

  double interpolateContinuousDedx(
      ContinuousEnergyTable const& table,
      double energy_MeV) {
    auto const [lower, fraction] = interpolationBracket(
        table.energies_MeV, energy_MeV,
        AxisTransform::Logarithmic, "continuous energy");
    return interpolateLossPair(
        table.dEdX_MeV_cm2_per_g[lower],
        table.dEdX_MeV_cm2_per_g[lower + 1],
        fraction);
  }

  double interpolateContinuousRange(
      ContinuousEnergyTable const& table,
      double energy_MeV) {
    auto const [lower, fraction] = interpolationBracket(
        table.energies_MeV, energy_MeV,
        AxisTransform::Logarithmic, "continuous energy");
    auto const result =
        table.range_g_per_cm2[lower] +
        fraction *
            (table.range_g_per_cm2[lower + 1] -
             table.range_g_per_cm2[lower]);
    if (!std::isfinite(result) || result < 0.) {
      throw std::runtime_error(
          "continuous range interpolation produced an invalid result");
    }
    return result;
  }

  double interpolateContinuousEnergy(
      ContinuousEnergyTable const& table,
      double range_g_per_cm2) {
    if (!std::isfinite(range_g_per_cm2) ||
        range_g_per_cm2 < table.range_g_per_cm2.front() ||
        range_g_per_cm2 > table.range_g_per_cm2.back()) {
      throw std::out_of_range(
          "continuous range is outside the stored domain");
    }
    auto const upper = std::lower_bound(
        table.range_g_per_cm2.begin(),
        table.range_g_per_cm2.end(), range_g_per_cm2);
    if (upper == table.range_g_per_cm2.begin()) {
      return table.energies_MeV.front();
    }
    if (upper == table.range_g_per_cm2.end()) {
      return table.energies_MeV.back();
    }
    auto const upper_index =
        static_cast<std::size_t>(
            upper - table.range_g_per_cm2.begin());
    if (*upper == range_g_per_cm2) {
      return table.energies_MeV[upper_index];
    }
    auto const lower_index = upper_index - 1;
    auto const fraction =
        (range_g_per_cm2 -
         table.range_g_per_cm2[lower_index]) /
        (table.range_g_per_cm2[upper_index] -
         table.range_g_per_cm2[lower_index]);
    return std::exp(
        std::log(table.energies_MeV[lower_index]) +
        fraction *
            (std::log(table.energies_MeV[upper_index]) -
             std::log(table.energies_MeV[lower_index])));
  }

  double energyAfterContinuousLoss(
      ContinuousEnergyTable const& table,
      double initial_energy_MeV,
      double grammage_g_per_cm2) {
    if (!std::isfinite(grammage_g_per_cm2) ||
        grammage_g_per_cm2 < 0.) {
      throw std::invalid_argument(
          "continuous-loss grammage must be finite and non-negative");
    }
    auto const initial_range =
        interpolateContinuousRange(table, initial_energy_MeV);
    if (grammage_g_per_cm2 >= initial_range &&
        grammage_g_per_cm2 != 0.) {
      throw std::out_of_range(
          "continuous loss reaches the configured transport cut");
    }
    return interpolateContinuousEnergy(
        table, initial_range - grammage_g_per_cm2);
  }

  Sha256Digest calculateContentHash(RateTableSet const& table) {
    return sha256(encodePayload(table));
  }

  Sha256Digest writeRateTable(std::filesystem::path const& path,
                              RateTableSet const& table) {
    auto const payload = encodePayload(table);
    auto const digest = sha256(payload);
    if (payload.size() > MaximumFileBytes) {
      throw std::length_error("rate-table payload exceeds the file size limit");
    }
    if (path.empty()) {
      throw std::invalid_argument("rate-table output path is empty");
    }
    if (!path.parent_path().empty()) {
      std::filesystem::create_directories(path.parent_path());
    }

    auto temporary = path;
    temporary += ".tmp";
    try {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      if (!output) {
        throw std::runtime_error("cannot open temporary rate-table output");
      }
      output.write(reinterpret_cast<char const*>(Magic.data()), Magic.size());
      writeU32(output, RateTableFormatVersion);
      writeU32(output, EndianMarker);
      writeU64(output, payload.size());
      output.write(reinterpret_cast<char const*>(digest.data()), digest.size());
      output.write(reinterpret_cast<char const*>(payload.data()), payload.size());
      output.close();
      if (!output) {
        throw std::runtime_error("failed while writing rate-table output");
      }
      std::filesystem::rename(temporary, path);
    } catch (...) {
      std::error_code error;
      std::filesystem::remove(temporary, error);
      throw;
    }
    return digest;
  }

  RateTableSet readRateTable(std::filesystem::path const& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("cannot open rate-table input: " + path.string());
    }
    std::array<std::uint8_t, Magic.size()> magic{};
    input.read(reinterpret_cast<char*>(magic.data()), magic.size());
    auto const current_format = magic == Magic;
    auto const legacy_format = magic == LegacyMagic;
    if (!input || (!current_format && !legacy_format)) {
      throw std::runtime_error("rate-table file has an invalid magic number");
    }
    auto const version = readU32(input);
    if ((current_format &&
         version != RateTableFormatVersion) ||
        (legacy_format && version != 9)) {
      throw std::runtime_error("rate-table file format version is unsupported");
    }
    if (readU32(input) != EndianMarker) {
      throw std::runtime_error("rate-table endian marker is invalid");
    }
    auto const payload_size = readU64(input);
    if (payload_size > MaximumFileBytes ||
        payload_size > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("rate-table payload size exceeds safety limit");
    }
    Sha256Digest expected_digest{};
    input.read(reinterpret_cast<char*>(expected_digest.data()), expected_digest.size());
    if (!input) {
      throw std::runtime_error("truncated rate-table digest");
    }
    std::vector<std::uint8_t> payload(static_cast<std::size_t>(payload_size));
    input.read(reinterpret_cast<char*>(payload.data()), payload.size());
    if (!input) {
      throw std::runtime_error("truncated rate-table payload");
    }
    if (input.peek() != std::char_traits<char>::eof()) {
      throw std::runtime_error("rate-table file contains trailing bytes");
    }
    auto const actual_digest = sha256(payload);
    if (actual_digest != expected_digest) {
      throw std::runtime_error("rate-table SHA-256 content hash mismatch");
    }
    auto table = decodePayload(payload, current_format);
    table.content_hash = actual_digest;
    return table;
  }

} // namespace corsika::gpu::em::tables
