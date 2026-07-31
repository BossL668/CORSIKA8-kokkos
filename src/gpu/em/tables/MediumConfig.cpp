/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/gpu/em/tables/MediumConfig.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace corsika::gpu::em::tables {

  namespace {

    constexpr double FractionTolerance = 1.e-8;

    void requireMap(YAML::Node const& node, std::string const& path) {
      if (!node || !node.IsMap()) {
        throw std::invalid_argument(path + " must be a YAML mapping");
      }
    }

    void rejectUnknownKeys(YAML::Node const& node,
                           std::set<std::string> const& allowed,
                           std::string const& path) {
      requireMap(node, path);
      for (auto const& entry : node) {
        if (!entry.first.IsScalar()) {
          throw std::invalid_argument(path + " contains a non-scalar key");
        }
        auto const key = entry.first.Scalar();
        if (allowed.count(key) == 0) {
          throw std::invalid_argument(path + " contains unknown key '" + key +
                                      "'");
        }
      }
    }

    YAML::Node requireField(YAML::Node const& node, char const* key,
                            std::string const& path) {
      auto const field = node[key];
      if (!field || field.IsNull()) {
        throw std::invalid_argument(path + " is missing required field '" +
                                    key + "'");
      }
      return field;
    }

    template <typename T>
    T parseField(YAML::Node const& node, char const* key,
                 std::string const& path) {
      try {
        return requireField(node, key, path).as<T>();
      } catch (YAML::Exception const& error) {
        throw std::invalid_argument(path + "." + key +
                                    " has an invalid value: " + error.what());
      }
    }

    bool safeName(std::string const& value) {
      static std::regex const pattern{
          R"(^[A-Za-z0-9][A-Za-z0-9_.+\-]{0,127}$)"};
      return std::regex_match(value, pattern);
    }

    void requireFinite(double value, char const* name) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
      }
    }

    std::string canonicalNumber(double value) {
      if (value == 0.) {
        value = 0.;
      }
      std::ostringstream stream;
      stream << std::scientific
             << std::setprecision(std::numeric_limits<double>::max_digits10)
             << value;
      return stream.str();
    }

  } // namespace

  MediumConfig standardDryAirMediumConfig() {
    MediumConfig config;
    config.name = "air_dry_1_atm";
    config.mean_excitation_energy_eV = 85.7;
    config.density_correction_C = -10.5961;
    config.density_correction_a = 0.10914;
    config.density_correction_m = 3.3994;
    config.density_correction_x0 = 1.7418;
    config.density_correction_x1 = 4.2759;
    config.density_correction_delta0 = 0.;
    config.reference_mass_density_g_per_cm3 = 0.0012048;
    config.components = {
        {"nucleus", 1000070140, 7., 14., 0.78479},
        {"nucleus", 1000080160, 8., 16., 0.21052},
        {"nucleus", 1000180400, 18., 40., 0.00469}};
    return normalizeMediumConfig(std::move(config));
  }

  MediumConfig normalizeMediumConfig(MediumConfig config) {
    if (config.schema_version != MediumConfigSchemaVersion) {
      throw std::invalid_argument("unsupported medium YAML schema_version");
    }
    if (!safeName(config.name)) {
      throw std::invalid_argument(
          "medium name must match [A-Za-z0-9][A-Za-z0-9_.+-]{0,127}");
    }

    requireFinite(config.mean_excitation_energy_eV,
                  "mean_excitation_energy_eV");
    requireFinite(config.density_correction_C, "density_correction_C");
    requireFinite(config.density_correction_a, "density_correction_a");
    requireFinite(config.density_correction_m, "density_correction_m");
    requireFinite(config.density_correction_x0, "density_correction_x0");
    requireFinite(config.density_correction_x1, "density_correction_x1");
    requireFinite(config.density_correction_delta0,
                  "density_correction_delta0");
    requireFinite(config.reference_mass_density_g_per_cm3,
                  "reference_mass_density_g_per_cm3");
    if (!(config.mean_excitation_energy_eV > 0.) ||
        !(config.density_correction_a > 0.) ||
        !(config.density_correction_m > 0.) ||
        !(config.density_correction_x1 > config.density_correction_x0) ||
        config.density_correction_delta0 < 0. ||
        !(config.reference_mass_density_g_per_cm3 > 0.)) {
      throw std::invalid_argument(
          "medium PROPOSAL properties violate their physical domains");
    }
    if (config.components.empty() || config.components.size() > 64) {
      throw std::invalid_argument(
          "medium must contain between one and 64 components");
    }

    double fraction_sum = 0.;
    std::set<std::int32_t> pids;
    for (auto const& component : config.components) {
      if (!safeName(component.name)) {
        throw std::invalid_argument(
            "component name must match [A-Za-z0-9][A-Za-z0-9_.+-]{0,127}");
      }
      requireFinite(component.nuclear_charge, "component nuclear_charge");
      requireFinite(component.atomic_mass_g_per_mol,
                    "component atomic_mass_g_per_mol");
      requireFinite(component.number_fraction, "component number_fraction");
      if (component.corsika_pid <= 0 ||
          !(component.nuclear_charge >= 1.) ||
          !(component.nuclear_charge <= 118.) ||
          std::abs(component.nuclear_charge -
                   std::round(component.nuclear_charge)) >
              1.e-12 ||
          !(component.atomic_mass_g_per_mol >=
            component.nuclear_charge) ||
          !(component.number_fraction > 0.)) {
        throw std::invalid_argument(
            "medium component violates PID, Z, mass or fraction constraints");
      }
      if (!pids.insert(component.corsika_pid).second) {
        throw std::invalid_argument(
            "medium contains a duplicate corsika_pid");
      }
      fraction_sum += component.number_fraction;
    }
    if (!std::isfinite(fraction_sum) ||
        std::abs(fraction_sum - 1.) > FractionTolerance) {
      throw std::invalid_argument(
          "medium component number fractions must sum to one within 1e-8");
    }
    for (auto& component : config.components) {
      component.number_fraction /= fraction_sum;
    }
    std::sort(
        config.components.begin(), config.components.end(),
        [](MediumComponentConfig const& left,
           MediumComponentConfig const& right) {
          return std::tie(left.corsika_pid, left.nuclear_charge,
                          left.atomic_mass_g_per_mol, left.name) <
                 std::tie(right.corsika_pid, right.nuclear_charge,
                          right.atomic_mass_g_per_mol, right.name);
        });
    return config;
  }

  MediumConfig loadMediumConfig(std::filesystem::path const& path) {
    if (!std::filesystem::is_regular_file(path)) {
      throw std::invalid_argument("medium YAML is not a regular file: " +
                                  path.string());
    }
    YAML::Node root;
    try {
      root = YAML::LoadFile(path.string());
    } catch (YAML::Exception const& error) {
      throw std::invalid_argument("cannot parse medium YAML: " +
                                  std::string(error.what()));
    }
    rejectUnknownKeys(root, {"schema_version", "name", "proposal",
                             "components"},
                      "medium");

    MediumConfig config;
    config.schema_version =
        parseField<std::uint32_t>(root, "schema_version", "medium");
    config.name = parseField<std::string>(root, "name", "medium");

    auto const proposal = requireField(root, "proposal", "medium");
    rejectUnknownKeys(
        proposal,
        {"mean_excitation_energy_eV", "density_correction_C",
         "density_correction_a", "density_correction_m",
         "density_correction_x0", "density_correction_x1",
         "density_correction_delta0",
         "reference_mass_density_g_per_cm3"},
        "medium.proposal");
    config.mean_excitation_energy_eV = parseField<double>(
        proposal, "mean_excitation_energy_eV", "medium.proposal");
    config.density_correction_C = parseField<double>(
        proposal, "density_correction_C", "medium.proposal");
    config.density_correction_a = parseField<double>(
        proposal, "density_correction_a", "medium.proposal");
    config.density_correction_m = parseField<double>(
        proposal, "density_correction_m", "medium.proposal");
    config.density_correction_x0 = parseField<double>(
        proposal, "density_correction_x0", "medium.proposal");
    config.density_correction_x1 = parseField<double>(
        proposal, "density_correction_x1", "medium.proposal");
    config.density_correction_delta0 = parseField<double>(
        proposal, "density_correction_delta0", "medium.proposal");
    config.reference_mass_density_g_per_cm3 = parseField<double>(
        proposal, "reference_mass_density_g_per_cm3", "medium.proposal");

    auto const components = requireField(root, "components", "medium");
    if (!components.IsSequence()) {
      throw std::invalid_argument(
          "medium.components must be a YAML sequence");
    }
    for (std::size_t index = 0; index < components.size(); ++index) {
      auto const component = components[index];
      auto const path_name =
          "medium.components[" + std::to_string(index) + "]";
      rejectUnknownKeys(component,
                        {"name", "corsika_pid", "nuclear_charge",
                         "atomic_mass_g_per_mol", "number_fraction"},
                        path_name);
      config.components.push_back(
          {parseField<std::string>(component, "name", path_name),
           parseField<std::int32_t>(component, "corsika_pid", path_name),
           parseField<double>(component, "nuclear_charge", path_name),
           parseField<double>(component, "atomic_mass_g_per_mol", path_name),
           parseField<double>(component, "number_fraction", path_name)});
    }
    return normalizeMediumConfig(std::move(config));
  }

  std::string canonicalMediumYaml(MediumConfig const& input) {
    auto const config = normalizeMediumConfig(input);
    std::ostringstream stream;
    stream << "schema_version: " << config.schema_version << '\n'
           << "name: " << config.name << '\n'
           << "proposal:\n"
           << "  mean_excitation_energy_eV: "
           << canonicalNumber(config.mean_excitation_energy_eV) << '\n'
           << "  density_correction_C: "
           << canonicalNumber(config.density_correction_C) << '\n'
           << "  density_correction_a: "
           << canonicalNumber(config.density_correction_a) << '\n'
           << "  density_correction_m: "
           << canonicalNumber(config.density_correction_m) << '\n'
           << "  density_correction_x0: "
           << canonicalNumber(config.density_correction_x0) << '\n'
           << "  density_correction_x1: "
           << canonicalNumber(config.density_correction_x1) << '\n'
           << "  density_correction_delta0: "
           << canonicalNumber(config.density_correction_delta0) << '\n'
           << "  reference_mass_density_g_per_cm3: "
           << canonicalNumber(config.reference_mass_density_g_per_cm3) << '\n'
           << "components:\n";
    for (auto const& component : config.components) {
      stream << "  - name: " << component.name << '\n'
             << "    corsika_pid: " << component.corsika_pid << '\n'
             << "    nuclear_charge: "
             << canonicalNumber(component.nuclear_charge) << '\n'
             << "    atomic_mass_g_per_mol: "
             << canonicalNumber(component.atomic_mass_g_per_mol) << '\n'
             << "    number_fraction: "
             << canonicalNumber(component.number_fraction) << '\n';
    }
    return stream.str();
  }

  Sha256Digest mediumConfigHash(MediumConfig const& config) {
    auto const canonical = canonicalMediumYaml(config);
    return sha256(
        reinterpret_cast<std::uint8_t const*>(canonical.data()),
        canonical.size());
  }

  std::string mediumConfigHashHex(MediumConfig const& config) {
    return toHex(mediumConfigHash(config));
  }

  void writeCanonicalMediumYaml(std::filesystem::path const& path,
                                MediumConfig const& config) {
    if (path.empty()) {
      throw std::invalid_argument(
          "canonical medium YAML output path is empty");
    }
    if (!path.parent_path().empty()) {
      std::filesystem::create_directories(path.parent_path());
    }
    static std::atomic<std::uint64_t> temporary_counter{0};
    auto temporary = path;
    temporary +=
        ".tmp." + std::to_string(std::random_device{}()) + "." +
        std::to_string(temporary_counter.fetch_add(
            1, std::memory_order_relaxed));
    try {
      std::ofstream output(temporary, std::ios::trunc);
      if (!output) {
        throw std::runtime_error(
            "cannot open temporary canonical medium YAML");
      }
      output << canonicalMediumYaml(config);
      output.close();
      if (!output) {
        throw std::runtime_error(
            "failed while writing canonical medium YAML");
      }
      std::filesystem::rename(temporary, path);
    } catch (...) {
      std::error_code error;
      std::filesystem::remove(temporary, error);
      throw;
    }
  }

} // namespace corsika::gpu::em::tables
