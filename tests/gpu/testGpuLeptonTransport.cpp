/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <PROPOSAL/Constants.h>
#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/particle/Particle.h>
#include <PROPOSAL/scattering/multiple_scattering/Coefficients.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <GpuEmFlatRateTableFixture.hpp>
#include <testCascade.hpp>

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/utility/QuadraticSolver.hpp>
#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/CudaInteractionSelector.hpp>
#include <corsika/gpu/em/CudaLeptonTransport.hpp>
#include <corsika/gpu/em/EnvironmentSnapshotBuilder.hpp>
#include <corsika/gpu/em/MoliereScattering.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/PhysicalCudaEmRouter.hpp>
#include <corsika/gpu/em/ProposalFallbackAdapter.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/SphericalAtmosphere.hpp>
#include <corsika/gpu/em/UniformMagneticField.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>

namespace {

  using namespace corsika;
  using namespace corsika::gpu::em;
  using namespace corsika::gpu::em::tables;

  std::size_t checks = 0;

  struct SpecifiedFallbackSink {
    std::uint64_t handled{};

    bool canHandle(
        ProposalFallbackEvent const& event) const {
      return hasSpecifiedProposalFinalState(event);
    }

    template <typename TStack>
    void handle(
        TStack&, ProposalFallbackEvent const&) {
      ++handled;
    }
  };

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void requireNear(double actual, double expected,
                   double relative_tolerance,
                   std::string const& message) {
    ++checks;
    auto const scale =
        std::max({1.e-300, std::abs(actual), std::abs(expected)});
    if (std::abs(actual - expected) >
        relative_tolerance * scale) {
      std::ostringstream detail;
      detail << message << ": actual=" << actual
             << ", expected=" << expected;
      throw std::runtime_error(detail.str());
    }
  }

  template <typename Function>
  void requireThrows(Function function, std::string const& message) {
    ++checks;
    try {
      function();
    } catch (std::exception const&) {
      return;
    }
    throw std::runtime_error(message);
  }

  std::filesystem::path temporaryPath() {
    auto const stamp =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();
    return std::filesystem::temp_directory_path() /
           ("c8_gpu_lepton_transport_" +
            std::to_string(stamp) + ".c8emrt");
  }

  std::size_t processCount(RateTableSet const& table) {
    std::size_t count = 0;
    for (auto const& particle : table.particles) {
      count += particle.columns.size();
    }
    return count;
  }

  MoliereMetadata makeMoliereMetadata() {
    PROPOSAL::Air const medium;
    MoliereMetadata metadata;
    metadata.enabled = true;
    metadata.reference_mode = "proposal_analytic";
    metadata.particle_mass_MeV =
        PROPOSAL::EMinusDef().mass;
    metadata.electron_mass_MeV = PROPOSAL::ME;
    metadata.fine_structure_constant = PROPOSAL::ALPHA;
    metadata.avogadro_per_mol = PROPOSAL::NA;
    metadata.hbar_MeV_s = PROPOSAL::HBAR;
    metadata.speed_of_light_cm_per_s = PROPOSAL::SPEED;
    metadata.euler_mascheroni =
        PROPOSAL::EULER_MASCHERONI;
    auto const components = medium.GetComponents();
    std::array<std::uint64_t, 3> const fixture_hashes{
        101, 102, 103};
    if (components.size() != fixture_hashes.size()) {
      throw std::runtime_error(
          "PROPOSAL Air composition differs from the flat fixture");
    }
    for (std::size_t index = 0; index < components.size();
         ++index) {
      auto const& component = components[index];
      metadata.components.push_back(
          {fixture_hashes[index],
           component.GetNucCharge(),
           component.GetAtomicNum(),
           component.GetAtomInMolecule()});
    }
    metadata.c1.assign(
        PROPOSAL::c1,
        PROPOSAL::c1 + MoliereSeriesCoefficientCount);
    metadata.c2.assign(
        PROPOSAL::c2,
        PROPOSAL::c2 + MoliereSeriesCoefficientCount);
    metadata.c2_large.assign(
        PROPOSAL::c2large,
        PROPOSAL::c2large +
            MoliereLargeSeriesCoefficientCount);
    metadata.s2_large.assign(
        PROPOSAL::s2large,
        PROPOSAL::s2large +
            MoliereLargeSeriesCoefficientCount);
    metadata.C1_large.assign(
        PROPOSAL::C1large,
        PROPOSAL::C1large +
            MoliereLargeIntegralCoefficientCount);
    return metadata;
  }

  void addLeptonRateTables(RateTableSet& source) {
    auto const makeParticle =
        [&](std::int32_t pdg_id, std::string name) {
          ParticleRateTable particle;
          particle.pdg_id = pdg_id;
          particle.particle_name = std::move(name);
          particle.interaction_hash =
              pdg_id == 11 ? 9102 : 9103;
          particle.energies_MeV = {1., 10., 100.};

          auto brems = source.particles.front().columns.front();
          brems.process_id = BremsProcessId;
          brems.component_hash = 101;
          brems.process_name = "Brems";
          brems.parameterization = "BremsKelnerKokoulinPetrukhin";
          brems.target_name = "nitrogen";
          brems.rates_cm2_per_g = {2., 2., 2.};

          auto ionization = brems;
          ionization.process_id = IonizationProcessId;
          ionization.component_hash = 102;
          ionization.process_name = "Ioniz";
          ionization.parameterization = "IonizBetheBlochRossi";
          ionization.target_name = "oxygen";
          ionization.rates_cm2_per_g = {1., 1., 1.};
          particle.columns = {
              std::move(brems), std::move(ionization)};
          if (pdg_id == -11) {
            auto annihilation =
                source.particles.front().columns.at(3);
            annihilation.process_id =
                AnnihilationProcessId;
            annihilation.component_hash = 101;
            annihilation.process_name = "Annihilation";
            annihilation.parameterization =
                "AnnihilationHeitler";
            annihilation.target_name = "nitrogen";
            annihilation.rates_cm2_per_g =
                {2., 2., 2.};
            particle.columns.push_back(
                std::move(annihilation));
          }
          return particle;
        };
    source.particles.push_back(makeParticle(11, "electron"));
    source.particles.push_back(makeParticle(-11, "positron"));
  }

  EmInteractionRecord makeCandidate(
      std::int32_t pid, double energy_MeV,
      double earth_radius_m, double altitude_m,
      std::array<double, 3> direction,
      std::uint64_t history_id, std::size_t input_index,
      double interaction_grammage =
          std::numeric_limits<double>::infinity()) {
    auto const norm = std::sqrt(
        direction[0] * direction[0] +
        direction[1] * direction[1] +
        direction[2] * direction[2]);
    EmInteractionRecord interaction{};
    auto& particle = interaction.particle;
    particle.pid = pid;
    particle.medium_id = 17;
    particle.energy_GeV = energy_MeV / 1000.;
    particle.position_m[2] = earth_radius_m + altitude_m;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      particle.direction[axis] = direction[axis] / norm;
    }
    particle.time_s = 2.e-6;
    particle.weight = 1.;
    particle.history_id = history_id;
    particle.step_id = 7;
    interaction.input_index = input_index;
    interaction.process_id = BremsProcessId;
    interaction.status =
        std::isfinite(interaction_grammage)
            ? EmInteractionStatus::Selected
            : EmInteractionStatus::NoDiscreteInteraction;
    interaction.component_hash = 101;
    interaction.total_rate_cm2_per_g = 2.;
    interaction.interaction_grammage_g_per_cm2 =
        interaction_grammage;
    interaction.process_uniform = 0.25;
    interaction.process_draw_id = 1;
    return interaction;
  }

  void compareRecords(
      std::vector<LeptonTransportRecord> const& left,
      std::vector<LeptonTransportRecord> const& right) {
    require(left.size() == right.size(),
            "repeat transport record count differs");
    for (std::size_t index = 0; index < left.size(); ++index) {
      auto const& a = left[index];
      auto const& b = right[index];
      require(
          a.input_index == b.input_index &&
              a.limit == b.limit &&
              a.end.energy_GeV == b.end.energy_GeV &&
              a.end.time_s == b.end.time_s &&
              a.distance_m == b.distance_m &&
              a.traversed_grammage_g_per_cm2 ==
                  b.traversed_grammage_g_per_cm2 &&
              a.continuous_deposited_energy_GeV ==
                  b.continuous_deposited_energy_GeV &&
              a.observation_surface_reached_before_cut ==
                  b.observation_surface_reached_before_cut,
          "straight lepton transport is not deterministic");
    }
  }

  void compareInteractionRecords(
      std::vector<EmInteractionRecord> const& left,
      std::vector<EmInteractionRecord> const& right,
      std::string const& label) {
    require(left.size() == right.size(),
            label + " count differs");
    for (std::size_t index = 0; index < left.size(); ++index) {
      auto const& a = left[index];
      auto const& b = right[index];
      require(
          a.input_index == b.input_index &&
              a.status == b.status &&
              a.process_id == b.process_id &&
              a.component_hash == b.component_hash &&
              a.particle.history_id == b.particle.history_id &&
              a.particle.step_id == b.particle.step_id &&
              a.particle.energy_GeV == b.particle.energy_GeV &&
              a.energy_fraction == b.energy_fraction &&
              a.loss_quantile == b.loss_quantile,
          label + " differs at index " +
              std::to_string(index));
    }
  }

  void compareFallbacks(
      std::vector<ProposalFallbackEvent> const& left,
      std::vector<ProposalFallbackEvent> const& right,
      std::string const& label) {
    require(left.size() == right.size(),
            label + " count differs");
    for (std::size_t index = 0; index < left.size(); ++index) {
      auto const& a = left[index];
      auto const& b = right[index];
      require(
          a.input_index == b.input_index &&
              a.reason == b.reason &&
              a.process_id == b.process_id &&
              a.component_hash == b.component_hash &&
              a.particle.history_id == b.particle.history_id,
          label + " differs at index " +
              std::to_string(index));
    }
  }

  void compareBremsResults(
      BremsFinalStateBatchResult const& left,
      BremsFinalStateBatchResult const& right) {
    require(
        left.input_interactions == right.input_interactions &&
            left.gpu_interactions == right.gpu_interactions &&
            left.brems_interactions ==
                right.brems_interactions &&
            left.annihilation_interactions ==
                right.annihilation_interactions &&
            left.ionization_interactions ==
                right.ionization_interactions &&
            left.electron_pair_interactions ==
                right.electron_pair_interactions &&
            left.brems_lpm_trials ==
                right.brems_lpm_trials &&
            left.brems_lpm_suppressions ==
                right.brems_lpm_suppressions &&
            left.electron_pair_lpm_trials ==
                right.electron_pair_lpm_trials &&
            left.electron_pair_lpm_suppressions ==
                right.electron_pair_lpm_suppressions &&
            left.final_state_records.size() ==
                right.final_state_records.size() &&
            left.secondaries.size() == right.secondaries.size() &&
            left.lpm_suppressed.size() ==
                right.lpm_suppressed.size(),
        "device-chained bremsstrahlung counts differ");
    compareFallbacks(
        left.fallback_events, right.fallback_events,
        "device-chained final-state fallbacks");
    compareInteractionRecords(
        left.continuations, right.continuations,
        "device-chained final-state continuations");
    for (std::size_t index = 0;
         index < left.final_state_records.size(); ++index) {
      auto const& a = left.final_state_records[index];
      auto const& b = right.final_state_records[index];
      require(
          a.input_index == b.input_index &&
              a.parent_history_id == b.parent_history_id &&
              a.secondary_offset == b.secondary_offset &&
              a.process_id == b.process_id &&
              a.photon_energy_fraction ==
                  b.photon_energy_fraction &&
              a.final_state_uniform ==
                  b.final_state_uniform &&
              a.azimuth_uniform == b.azimuth_uniform &&
              a.auxiliary_uniform == b.auxiliary_uniform &&
              a.lpm_uniform == b.lpm_uniform,
          "device-chained bremsstrahlung record differs");
    }
    for (std::size_t index = 0;
         index < left.secondaries.size(); ++index) {
      auto const& a = left.secondaries[index];
      auto const& b = right.secondaries[index];
      require(
          a.pid == b.pid &&
              a.energy_GeV == b.energy_GeV &&
              a.history_id == b.history_id &&
              a.parent_history_id == b.parent_history_id &&
              a.step_id == b.step_id &&
              a.direction[0] == b.direction[0] &&
              a.direction[1] == b.direction[1] &&
              a.direction[2] == b.direction[2],
          "device-chained bremsstrahlung secondary differs");
    }
    for (std::size_t index = 0;
         index < left.lpm_suppressed.size(); ++index) {
      auto const& a = left.lpm_suppressed[index];
      auto const& b = right.lpm_suppressed[index];
      require(
          a.input_index == b.input_index &&
              a.particle.history_id == b.particle.history_id &&
              a.particle.step_id == b.particle.step_id &&
              a.survival_probability ==
                  b.survival_probability &&
              a.uniform == b.uniform,
          "device-chained bremsstrahlung suppression differs");
    }
  }

  bool sameParticle(
      EmParticleState const& left,
      EmParticleState const& right) {
    if (left.pid != right.pid ||
        left.medium_id != right.medium_id ||
        left.generation != right.generation ||
        left.energy_GeV != right.energy_GeV ||
        left.time_s != right.time_s ||
        left.weight != right.weight ||
        left.history_id != right.history_id ||
        left.parent_history_id != right.parent_history_id ||
        left.step_id != right.step_id) {
      return false;
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (left.position_m[axis] != right.position_m[axis] ||
          left.direction[axis] != right.direction[axis]) {
        return false;
      }
    }
    return true;
  }

  void compareParticles(
      std::vector<EmParticleState> const& left,
      std::vector<EmParticleState> const& right,
      std::string const& label) {
    require(left.size() == right.size(),
            label + " count differs");
    for (std::size_t index = 0; index < left.size(); ++index) {
      require(
          sameParticle(left[index], right[index]),
          label + " differs at index " +
              std::to_string(index));
    }
  }

  struct ExpectedLeptonEndpoints {
    std::vector<EmParticleState> next_leptons;
    std::vector<EmParticleState> generated_photons;
    std::vector<ObservationRecord> observations;
    std::vector<EmParticleState> decay_candidates;
  };

  ExpectedLeptonEndpoints reconstructLeptonEndpoints(
      LeptonDevicePipelineBatchResult const& pipeline,
      std::size_t source_count) {
    std::vector<std::pair<std::size_t, EmParticleState>>
        next;
    std::vector<std::pair<std::size_t, EmParticleState>>
        photons;
    std::vector<std::pair<std::size_t, ObservationRecord>>
        observations;
    std::vector<std::pair<std::size_t, EmParticleState>>
        decays;
    for (auto const& record : pipeline.transport_records) {
      auto const source =
          static_cast<std::size_t>(record.input_index);
      require(source < source_count,
              "transport endpoint source is invalid");
      switch (record.limit) {
      case LeptonTransportLimit::ContinuousStep:
      case LeptonTransportLimit::LayerBoundary:
      case LeptonTransportLimit::MagneticStep:
        next.emplace_back(3 * source, record.end);
        break;
      case LeptonTransportLimit::ObservationSurface:
      case LeptonTransportLimit::EscapedEnvironment:
        observations.emplace_back(
            source,
            ObservationRecord{
                record.end,
                record.limit ==
                        LeptonTransportLimit::
                            ObservationSurface
                    ? ObservationStatus::
                          ReachedObservationSurface
                    : ObservationStatus::
                          EscapedEnvironment,
                0});
        break;
      case LeptonTransportLimit::ParticleCut:
        if (record.observation_surface_reached_before_cut != 0U) {
          observations.emplace_back(
              source,
              ObservationRecord{
                  record.end,
                  ObservationStatus::ReachedObservationSurface,
                  0});
        }
        break;
      case LeptonTransportLimit::InteractionCandidate:
        break;
      case LeptonTransportLimit::DecayCandidate:
        decays.emplace_back(source, record.end);
        break;
      }
    }
    for (auto const& continuation :
         pipeline.vertex_continuations) {
      next.emplace_back(
          3 * continuation.input_index,
          continuation.particle);
    }
    for (auto const& suppression :
         pipeline.final_states.lpm_suppressed) {
      next.emplace_back(
          3 * suppression.input_index,
          suppression.particle);
    }
    for (auto const& continuation :
         pipeline.final_states.continuations) {
      next.emplace_back(
          3 * continuation.input_index,
          continuation.particle);
    }
    for (auto const& record :
         pipeline.final_states.final_state_records) {
      std::size_t lepton_child = 0;
      std::size_t photon_child = 0;
      for (std::size_t child = 0;
           child < record.secondary_count; ++child) {
        auto const& particle =
            pipeline.final_states.secondaries[
                record.secondary_offset + child];
        if (particle.pid ==
                static_cast<std::int32_t>(EmPid::Photon)) {
          photons.emplace_back(
              3 * record.input_index + photon_child++,
              particle);
        } else {
          next.emplace_back(
              3 * record.input_index + lepton_child++,
              particle);
        }
      }
    }
    auto bySlot = [](auto const& left, auto const& right) {
      return left.first < right.first;
    };
    std::sort(next.begin(), next.end(), bySlot);
    std::sort(photons.begin(), photons.end(), bySlot);
    std::sort(observations.begin(), observations.end(), bySlot);
    std::sort(decays.begin(), decays.end(), bySlot);
    ExpectedLeptonEndpoints result;
    for (auto const& entry : next) {
      result.next_leptons.push_back(entry.second);
    }
    for (auto const& entry : photons) {
      result.generated_photons.push_back(entry.second);
    }
    for (auto const& entry : observations) {
      result.observations.push_back(entry.second);
    }
    for (auto const& entry : decays) {
      result.decay_candidates.push_back(entry.second);
    }
    return result;
  }

  int validateExternalMuonTable(
      std::filesystem::path const& table_path) {
    auto const source = readRateTable(table_path);
    std::vector<ParticleRateTable const*> muon_tables;
    for (auto const& particle : source.particles) {
      if (isMuonPid(particle.pdg_id)) {
        muon_tables.push_back(&particle);
      }
    }
    require(muon_tables.size() == 2,
            "muon validation table must contain both charge states");
    for (auto const* particle : muon_tables) {
      require(!particle->energies_MeV.empty(),
              "muon rate grid is empty");
      findContinuousEnergyTable(source, particle->pdg_id);
    }

    ProposalTableSet descriptor{};
    descriptor.process_count =
        static_cast<std::uint32_t>(processCount(source));
    descriptor.content_hash = source.content_hash;
    auto const earth_radius_m =
        constants::EarthRadius::Mean / 1_m;
    auto environment = makeCorsika7AtmosphereSnapshot(
        AtmosphereId::USStdBK, {0., 0., 0.}, 17,
        earth_radius_m + 100.);
    GpuEmConfig config{};
    config.device = 0;
    config.min_batch_size = 1;
    config.memory_fraction = 0.10;
    config.table_tolerance =
        source.metadata.requested_relative_tolerance;
    config.random_seed = 0x4d554f4e43554441ULL;
    config.shower_id = 20260730;
    config.table_cache = table_path;

    CudaEmBackend backend;
    backend.initialize(environment, descriptor, config);
    EmParticleState capability_probe{};
    capability_probe.pid =
        static_cast<std::int32_t>(EmPid::MuonMinus);
    capability_probe.energy_GeV = 100.;
    require(backend.canTransport(capability_probe),
            "validated muon table did not enable the production capability gate");

    // The continuous table anchor is deliberately 0.9999 times the
    // configured kinetic cut.  Exercise the narrow interval between that
    // anchor and the true per-PID ParticleCut: beta2 incorrectly compared
    // muons against the electron cut here and could create a zero-progress
    // resident wavefront at the 300 MeV muon threshold.
    std::vector<EmParticleState> muons_inside_cut_safety_interval;
    for (auto const pdg : {13, -13}) {
      auto const& continuous =
          findContinuousEnergyTable(source, pdg);
      auto const kinetic_anchor =
          continuous.minimum_total_energy_MeV -
          continuous.mass_MeV;
      auto const kinetic_cut =
          kinetic_anchor / ContinuousCutSafetyFactor;
      EmParticleState particle{};
      particle.pid = pdg;
      particle.medium_id = 17;
      particle.energy_GeV =
          (continuous.mass_MeV +
           0.5 * (kinetic_anchor + kinetic_cut)) /
          1000.;
      particle.position_m[2] = earth_radius_m + 50000.;
      particle.direction[2] = -1.;
      particle.weight = 1.;
      particle.history_id =
          900000 + muons_inside_cut_safety_interval.size();
      particle.step_id = 4;
      muons_inside_cut_safety_interval.push_back(particle);
    }
    auto const cut_interval_selection =
        backend.selectInteractionsForValidation(
            muons_inside_cut_safety_interval);
    require(
        cut_interval_selection.fallback_events.empty() &&
            cut_interval_selection.interactions.size() == 2,
        "per-PID muon cut selection lost a safety-interval particle");
    for (auto const& interaction :
         cut_interval_selection.interactions) {
      require(
          interaction.status ==
                  EmInteractionStatus::ParticleCut &&
              std::isinf(
                  interaction.interaction_grammage_g_per_cm2),
          "muon inside its ParticleCut safety interval entered transport");
    }
    auto const cut_interval_pipeline =
        backend.runLeptonDevicePipelineForValidation(
            muons_inside_cut_safety_interval, 9050000);
    require(
        cut_interval_pipeline.selection_fallback_events.empty() &&
            cut_interval_pipeline.transport_fallback_events.empty() &&
            cut_interval_pipeline.transport_records.size() == 2 &&
            cut_interval_pipeline.next_leptons.empty(),
        "per-PID muon cut did not terminate on the GPU");
    for (auto const& record :
         cut_interval_pipeline.transport_records) {
      require(
          record.limit == LeptonTransportLimit::ParticleCut &&
              record.distance_m == 0. &&
              record.traversed_grammage_g_per_cm2 == 0.,
          "per-PID muon cut produced a nonzero transport step");
    }

    constexpr std::size_t ParticleCount = 8192;
    std::vector<EmParticleState> particles;
    particles.reserve(ParticleCount);
    for (std::size_t index = 0; index < ParticleCount; ++index) {
      auto const& particle_table =
          *muon_tables[index % muon_tables.size()];
      auto const fraction = std::fmod(
          (static_cast<double>(index) + 0.5) *
              0.6180339887498948482,
          1.);
      auto const log_min =
          std::log(particle_table.energies_MeV.front());
      auto const log_max =
          std::log(particle_table.energies_MeV.back());
      EmParticleState particle{};
      particle.pid = particle_table.pdg_id;
      particle.medium_id = 17;
      particle.energy_GeV =
          std::exp(log_min + fraction * (log_max - log_min)) /
          1000.;
      particle.position_m[2] = earth_radius_m + 50000.;
      particle.direction[2] = -1.;
      particle.weight = 1.;
      particle.history_id = index + 1;
      particle.step_id = index % 11;
      particles.push_back(particle);
    }

    auto const selection =
        backend.selectInteractionsForValidation(particles);
    require(selection.fallback_events.empty() &&
                selection.interactions.size() == particles.size(),
            "muon distance selector lost or rejected a particle");
    for (auto const& interaction : selection.interactions) {
      require(interaction.status ==
                  EmInteractionStatus::DistanceSampled,
              "muon selector sampled a process before continuous loss");
    }

    auto const transported =
        backend.transportLeptonsStraightForValidation(
            selection.interactions);
    auto const repeated =
        backend.transportLeptonsStraightForValidation(
            selection.interactions);
    require(transported.fallback_events.empty() &&
                transported.records.size() == particles.size(),
            "muon straight transport lost or rejected a particle");
    compareRecords(transported.records, repeated.records);

    std::vector<EmInteractionRecord> vertex_candidates;
    std::size_t decay_candidates = 0;
    for (auto const& record : transported.records) {
      require(isMuonPid(record.start.pid) &&
                  record.end.pid == record.start.pid &&
                  std::isfinite(record.end.energy_GeV) &&
                  record.end.energy_GeV <=
                      record.start.energy_GeV &&
                  record.continuous_deposited_energy_GeV >= 0.,
              "muon transport produced an invalid endpoint");
      require(
          std::abs(
              record.start.energy_GeV - record.end.energy_GeV -
              record.continuous_deposited_energy_GeV) <=
              5.e-13 *
                  std::max(1., record.start.energy_GeV),
          "muon continuous energy balance differs");
      auto const& continuous =
          findContinuousEnergyTable(source, record.start.pid);
      auto const start_range = interpolateContinuousRange(
          continuous, record.start.energy_GeV * 1000.);
      auto const end_range = interpolateContinuousRange(
          continuous, record.end.energy_GeV * 1000.);
      auto const target_range = std::max(
          0., start_range -
                  record.traversed_grammage_g_per_cm2);
      auto const expected_end_energy =
          interpolateContinuousEnergy(continuous, target_range);
      requireNear(
          record.end.energy_GeV * 1000.,
          expected_end_energy, 3.e-12,
          "GPU muon range inversion differs from the CPU table");
      auto const range_scale = std::max(
          {1., std::abs(start_range), std::abs(end_range),
           record.traversed_grammage_g_per_cm2});
      require(
          std::abs(
              start_range - end_range -
              record.traversed_grammage_g_per_cm2) <=
              3. * source.metadata.requested_relative_tolerance *
                  range_scale,
          "muon range/grammage closure exceeds the table tolerance");
      if (record.limit ==
          LeptonTransportLimit::InteractionCandidate) {
        vertex_candidates.push_back(record.interaction);
      } else if (
          record.limit ==
          LeptonTransportLimit::DecayCandidate) {
        ++decay_candidates;
        auto const& selected =
            selection.interactions[record.input_index];
        auto const mass_GeV =
            selected.particle_mass_GeV;
        auto const momentum_GeV = std::sqrt(
            (selected.particle.energy_GeV - mass_GeV) *
            (selected.particle.energy_GeV + mass_GeV));
        auto const expected_distance =
            -std::log(selected.decay_uniform) *
            MuonDecaySpeedOfLightMPerS *
            MuonMeanLifetimeS * momentum_GeV / mass_GeV;
        requireNear(
            record.distance_m, expected_distance, 3.e-13,
            "muon decay distance differs from its Philox lifetime draw");
        require(
            record.interaction.process_id == DecayProcessId &&
                record.interaction.particle.history_id ==
                    record.end.history_id &&
                record.interaction.particle.energy_GeV ==
                    record.end.energy_GeV,
            "muon decay candidate did not preserve its GPU vertex");
      }
    }
    require(!vertex_candidates.empty(),
            "muon transport produced no interaction vertices");
    require(decay_candidates > 0,
            "muon transport produced no decay candidates");

    auto const vertices =
        backend.reselectLeptonInteractionsAtVertexForValidation(
            vertex_candidates);
    auto const repeated_vertices =
        backend.reselectLeptonInteractionsAtVertexForValidation(
            vertex_candidates);
    require(
        vertices.interactions.size() +
                vertices.continuations.size() +
                vertices.fallback_events.size() ==
            vertex_candidates.size(),
        "muon vertex selector lost or duplicated a candidate");
    compareInteractionRecords(
        vertices.interactions, repeated_vertices.interactions,
        "muon vertex interactions");
    compareInteractionRecords(
        vertices.continuations, repeated_vertices.continuations,
        "muon vertex continuations");
    compareFallbacks(
        vertices.fallback_events,
        repeated_vertices.fallback_events,
        "muon vertex fallbacks");
    for (auto const& interaction : vertices.interactions) {
      require(
          isMuonPid(interaction.particle.pid) &&
              interaction.process_id == IonizationProcessId,
          "a non-ionization muon process entered the GPU final-state path");
    }
    for (auto const& fallback : vertices.fallback_events) {
      require(
          isMuonPid(fallback.particle.pid) &&
              fallback.process_id != 0 &&
              fallback.component_hash != 0 &&
              hasResolvableProposalSelectedLoss(fallback),
          "muon CPU fallback did not preserve its selected process/component/loss draw");
    }

    auto const ionization_final_states =
        backend.generateBremsFinalStatesForValidation(
            vertices.interactions, 30000000);
    auto const repeated_final_states =
        backend.generateBremsFinalStatesForValidation(
            vertices.interactions, 30000000);
    require(
        ionization_final_states.gpu_interactions ==
                vertices.interactions.size() &&
            ionization_final_states.ionization_interactions ==
                vertices.interactions.size() &&
            ionization_final_states.secondaries.size() ==
                2 * vertices.interactions.size() &&
            ionization_final_states.fallback_events.empty(),
        "selected muon ionization did not complete on the GPU");
    compareBremsResults(
        ionization_final_states, repeated_final_states);
    for (std::size_t index = 0;
         index < vertices.interactions.size(); ++index) {
      auto const& parent = vertices.interactions[index].particle;
      auto const& outgoing =
          ionization_final_states.secondaries[2 * index];
      auto const& delta =
          ionization_final_states.secondaries[2 * index + 1];
      require(
          outgoing.pid == parent.pid &&
              isMuonPid(outgoing.pid) &&
              delta.pid ==
                  static_cast<std::int32_t>(EmPid::Electron),
          "GPU muon ionization produced the wrong child species");
    }

    auto const device_pipeline =
        backend.runLeptonDevicePipelineForValidation(
            particles, 40000000);
    auto const repeated_pipeline =
        backend.runLeptonDevicePipelineForValidation(
            particles, 40000000);
    require(
        device_pipeline.multiple_scattering_enabled &&
            device_pipeline.selection_fallback_events.empty() &&
            device_pipeline.transport_fallback_events.empty() &&
            device_pipeline.transport_records.size() ==
                particles.size(),
        "physical muon pipeline did not transport every input with Moliere enabled");
    compareRecords(
        device_pipeline.transport_records,
        repeated_pipeline.transport_records);
    compareFallbacks(
        device_pipeline.vertex_fallback_events,
        repeated_pipeline.vertex_fallback_events,
        "physical muon pipeline vertex fallbacks");
    compareBremsResults(
        device_pipeline.final_states,
        repeated_pipeline.final_states);
    std::size_t pipeline_decay_candidates = 0;
    for (auto const& record :
         device_pipeline.transport_records) {
      require(
          record.multiple_scattering_status ==
                  static_cast<std::uint32_t>(
                      MoliereStatus::Success) ||
              record.multiple_scattering_status ==
                  static_cast<std::uint32_t>(
                      MoliereStatus::NoDeflection),
          "muon Moliere sampling failed in the physical pipeline");
      pipeline_decay_candidates +=
          record.limit ==
          LeptonTransportLimit::DecayCandidate;
    }
    require(
        pipeline_decay_candidates == decay_candidates,
        "Moliere-enabled pipeline changed the sampled muon decay count");
    require(
        device_pipeline.decay_candidates.size() ==
            decay_candidates,
        "device endpoint compaction lost a muon decay candidate");
    compareParticles(
        device_pipeline.decay_candidates,
        repeated_pipeline.decay_candidates,
        "deterministic muon decay endpoint queue");

    auto const& coordinate_system =
        get_root_CoordinateSystem();
    TestCascadeIdentityStack decay_stack;
    auto decay_muon = decay_stack.addParticle(
        std::make_tuple(
            Code::MuMinus, 310_MeV,
            DirectionVector{
                coordinate_system, {0., 0., -1.}},
            Point{
                coordinate_system, 0_m, 0_m,
                (earth_radius_m + 50000.) * meter},
            0_s));
    auto const decay_history =
        decay_muon.getHistoryId();
    auto const decay_step =
        decay_muon.beginTransportStep();
    PhysicalCudaEmRouter<TestCascadeIdentityStack>
        decay_router{
            backend, coordinate_system, environment};
    require(
        decay_router.canRoute(decay_muon, decay_step),
        "production router rejected a table-covered muon");
    decay_router.stage(
        decay_muon, decay_history,
        decay_muon.getParentHistoryId(),
        decay_muon.getGeneration(), decay_step);
    decay_muon.erase();
    std::size_t forced_decay_markers = 0;
    std::size_t decay_router_rounds = 0;
    while (decay_router.pending() ||
           !decay_stack.isEmpty()) {
      require(
          ++decay_router_rounds < 2048,
          "production muon decay router did not terminate");
      if (decay_router.pending()) {
        decay_router.advanceOneWavefrontAndReturn(
            decay_stack);
      }
      while (!decay_stack.isEmpty()) {
        auto returned = decay_stack.getNextParticle();
        auto const returned_step =
            returned.beginTransportStep();
        auto const blocked_from_reroute =
            !decay_router.canRoute(
                returned, returned_step);
        auto const forced =
            decay_router.consumeForcedDecay(
                returned.getHistoryId(), returned_step);
        if (forced) {
          require(
              blocked_from_reroute,
              "consumed forced-decay particle was rerouted");
          ++forced_decay_markers;
        }
        returned.erase();
      }
    }
    auto const& decay_router_statistics =
        decay_router.statistics();
    require(
        forced_decay_markers == 1 &&
            decay_router_statistics
                    .particles_returned_for_cpu_decay ==
                1 &&
            decay_router_statistics
                    .forced_cpu_decays_executed ==
                1,
        "production router did not preserve exactly one forced CPU muon decay");

    std::cout
        << "CUDA muon selection/straight-transport validation passed: "
        << checks << " checks, " << transported.records.size()
        << " transported, " << vertex_candidates.size()
        << " vertices, " << vertices.interactions.size()
        << " GPU ionization, " << vertices.fallback_events.size()
        << " selected-process CPU fallback, "
        << decay_candidates << " forced CPU decay\n";
    return 0;
  }

} // namespace

int main(int argc, char** argv) {
  int device_count = 0;
  auto const cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no usable CUDA device\n";
    return 77;
  }

  if (argc == 2) {
    try {
      return validateExternalMuonTable(argv[1]);
    } catch (std::exception const& error) {
      std::cerr
          << "CUDA muon selection/straight-transport validation failed after "
          << checks << " checks: " << error.what() << '\n';
      return 1;
    }
  }
  if (argc != 1) {
    std::cerr << "usage: " << argv[0]
              << " [muon-table.c8emrt]\n";
    return 2;
  }

  auto const temporary = temporaryPath();
  try {
    auto source = testing::makeFlatRateTableFixture();
    addLeptonRateTables(source);
    auto const digest = writeRateTable(temporary, source);
    ProposalTableSet descriptor{};
    descriptor.process_count =
        static_cast<std::uint32_t>(processCount(source));
    descriptor.content_hash = digest;
    auto const earth_radius_m =
        constants::EarthRadius::Mean / 1_m;
    auto environment = makeCorsika7AtmosphereSnapshot(
        AtmosphereId::USStdBK, {0., 0., 0.}, 17,
        earth_radius_m + 100.);
    auto const interior_boundary_altitude_m =
        environment.atmosphere_layers[2].outer_radius_m -
        earth_radius_m;
    auto const atmosphere_top_altitude_m =
        environment
            .atmosphere_layers[
                environment.number_of_layers - 1]
            .outer_radius_m -
        earth_radius_m;
    GpuEmConfig config{};
    config.device = 0;
    config.min_batch_size = 16;
    config.memory_fraction = 0.10;
    config.table_tolerance = 1.e-3;
    config.table_cache = temporary;

    auto const& electron =
        findContinuousEnergyTable(source, 11);
    auto const continuous_x =
        interpolateContinuousRange(electron, 100.) -
        interpolateContinuousRange(electron, 90.);
    std::vector<EmInteractionRecord> interactions;
    interactions.push_back(makeCandidate(
        11, 100., earth_radius_m, 50000., {0., 0., -1.},
        1, 0, 0.25 * continuous_x));
    interactions.push_back(makeCandidate(
        -11, 100., earth_radius_m, 50000., {0., 0., -1.},
        2, 1));
    interactions.push_back(makeCandidate(
        11, 1.05, earth_radius_m, 20000., {0., 0., -1.},
        3, 2));
    interactions.push_back(makeCandidate(
        -11, 100., earth_radius_m,
        interior_boundary_altitude_m + 1.,
        {0., 0., -1.},
        4, 3));
    interactions.push_back(makeCandidate(
        11, 100., earth_radius_m, 100.1, {0., 0., -1.},
        5, 4));
    interactions.push_back(makeCandidate(
        -11, 100., earth_radius_m,
        atmosphere_top_altitude_m - 0.1,
        {0., 0., 1.},
        6, 5));
    auto unsupported = makeCandidate(
        11, 100., earth_radius_m, 50000., {0., 0., -1.},
        7, 6);
    unsupported.particle.pid = 22;
    interactions.push_back(unsupported);

    CudaEmBackend backend;
    backend.initialize(environment, descriptor, config);
    auto const distance_only =
        backend.selectInteractionsForValidation(
            {interactions.front().particle});
    require(distance_only.interactions.size() == 1 &&
                distance_only.fallback_events.empty(),
            "charged distance selection failed");
    require(
        distance_only.interactions[0].status ==
                EmInteractionStatus::DistanceSampled &&
            distance_only.interactions[0].process_id == 0 &&
            distance_only.interactions[0].component_hash == 0,
        "charged selector prematurely sampled a process before loss");

    std::vector<EmParticleState> below_table_leptons;
    for (std::size_t index = 0; index < 2; ++index) {
      auto candidate = makeCandidate(
          index == 0 ? 11 : -11,
          index == 0 ? 0.9 : 0.75, earth_radius_m, 20000.,
          {0., 0., -1.}, 80 + index, index);
      candidate.particle.step_id = 13;
      below_table_leptons.push_back(candidate.particle);
    }
    CudaEmBackend below_table_backend;
    below_table_backend.initialize(
        environment, descriptor, config);
    auto const below_table_selection =
        below_table_backend.selectInteractionsForValidation(
            below_table_leptons);
    require(
        below_table_selection.fallback_events.empty() &&
            below_table_selection.interactions.size() ==
                below_table_leptons.size(),
        "sub-table charged states incorrectly fell back during rate selection");
    for (auto const& selected :
         below_table_selection.interactions) {
      require(
          selected.status ==
                  EmInteractionStatus::ParticleCut &&
              std::isinf(
                  selected.interaction_grammage_g_per_cm2),
          "sub-table charged state sampled a discrete interaction");
    }
    auto const below_table_pipeline =
        below_table_backend
            .runLeptonDevicePipelineForValidation(
                below_table_leptons, 9000000);
    require(
        below_table_pipeline.selection_fallback_events.empty() &&
            below_table_pipeline.transport_fallback_events.empty() &&
            below_table_pipeline.vertex_fallback_events.empty() &&
            below_table_pipeline.final_states.fallback_events.empty() &&
            below_table_pipeline.transport_records.size() ==
                below_table_leptons.size() &&
            below_table_pipeline.next_leptons.empty() &&
            below_table_pipeline.generated_photons.empty() &&
            below_table_pipeline.observations.empty(),
        "sub-table charged states did not terminate entirely on the GPU");
    for (std::size_t index = 0;
         index < below_table_pipeline.transport_records.size();
         ++index) {
      auto const& record =
          below_table_pipeline.transport_records[index];
      require(
          record.limit == LeptonTransportLimit::ParticleCut &&
              record.distance_m == 0. &&
              record.traversed_grammage_g_per_cm2 == 0. &&
              record.continuous_deposited_energy_GeV == 0. &&
              record.end.step_id == record.start.step_id + 1 &&
              sameParticle(
                  record.start, below_table_leptons[index]),
          "sub-table charged ParticleCut record differs");
      requireNear(
          record.cut_deposited_energy_GeV,
          below_table_leptons[index].energy_GeV -
              electron.mass_MeV / 1000.,
          2.e-13,
          "sub-table charged ParticleCut did not deposit kinetic energy");
    }
    auto const below_table_resident =
        below_table_backend
            .runResidentLeptonCascadeForValidation(
                below_table_leptons, 9100000, 4);
    require(
        below_table_resident.completed &&
            below_table_resident.wavefronts == 1 &&
            below_table_resident.fallback_events.empty() &&
            below_table_resident.remaining_leptons.empty() &&
            below_table_resident.generated_photons.empty() &&
            below_table_resident.step_records.size() ==
                below_table_leptons.size(),
        "resident sub-table charged cascade did not finish in one GPU wavefront");

    auto const first =
        backend.transportLeptonsStraightForValidation(interactions);
    auto const second =
        backend.transportLeptonsStraightForValidation(interactions);
    require(first.input_interactions == interactions.size(),
            "lepton transport input count differs");
    if (first.records.size() != 6) {
      std::cerr << "diagnostic: straight transport records="
                << first.records.size() << " fallbacks="
                << first.fallback_events.size() << '\n';
      for (auto const& fallback : first.fallback_events) {
        std::cerr << "  input=" << fallback.input_index
                  << " reason="
                  << static_cast<std::uint32_t>(fallback.reason)
                  << " status=" << fallback.diagnostic_status
                  << " values=" << fallback.diagnostic_value0 << ','
                  << fallback.diagnostic_value1 << ','
                  << fallback.diagnostic_value2 << '\n';
      }
    }
    require(first.records.size() == 6,
            "six charged inputs must produce transport records");
    require(first.fallback_events.size() == 1,
            "photon must produce one explicit fallback");
    require(first.fallback_events[0].input_index == 6 &&
                first.fallback_events[0].reason ==
                    ProposalFallbackReason::UnsupportedParticle &&
                first.fallback_events[0].medium_hash == 9001 &&
                first.fallback_events[0].interaction_hash == 9101,
            "unsupported lepton transport fallback differs");
    compareRecords(first.records, second.records);

    auto const near_observation = makeCandidate(
        11, 100., earth_radius_m,
        100. + 0.5 * AtmosphereBoundaryGuardM,
        {0., 0., -1.}, 70, 0);
    CudaEmBackend near_observation_backend;
    near_observation_backend.initialize(
        environment, descriptor, config);
    auto const near_observation_straight =
        near_observation_backend
            .transportLeptonsStraightForValidation(
                {near_observation});
    require(
        near_observation_straight.fallback_events.empty() &&
            near_observation_straight.records.size() == 1,
        "near-observation straight lepton must not fall back");
    require(
        near_observation_straight.records[0].limit ==
                LeptonTransportLimit::ObservationSurface &&
            near_observation_straight.records[0].distance_m >
                0. &&
            near_observation_straight.records[0].distance_m <
                AtmosphereBoundaryGuardM,
        "near-observation straight lepton must retain its terminal root");
    auto late_at_observation = makeCandidate(
        11, 100., earth_radius_m, 100.1,
        {0., 0., -1.}, 70000001, 0);
    late_at_observation.particle.time_s =
        ParticleCutMaximumTimeS - 0.05 / 299792458.;
    auto const late_observation_straight =
        near_observation_backend
            .transportLeptonsStraightForValidation(
                {late_at_observation});
    require(
        late_observation_straight.fallback_events.empty() &&
            late_observation_straight.records.size() == 1 &&
            late_observation_straight.records[0].limit ==
                LeptonTransportLimit::ParticleCut &&
            late_observation_straight.records[0]
                    .observation_surface_reached_before_cut ==
                1U &&
            late_observation_straight.records[0].end.time_s >
                ParticleCutMaximumTimeS &&
            late_observation_straight.records[0]
                    .cut_deposited_energy_GeV >
                0.,
        "post-step lepton time cut does not retain the scalar observation order");
    auto const late_observation_pipeline =
        near_observation_backend
            .runLeptonDevicePipelineForValidation(
                {late_at_observation.particle}, 70000100);
    require(
        late_observation_pipeline.selection_fallback_events.empty() &&
            late_observation_pipeline.transport_fallback_events.empty() &&
            late_observation_pipeline.vertex_fallback_events.empty() &&
            late_observation_pipeline.final_states.fallback_events.empty() &&
            late_observation_pipeline.transport_records.size() == 1 &&
            late_observation_pipeline.transport_records[0].limit ==
                LeptonTransportLimit::ParticleCut &&
            late_observation_pipeline.next_leptons.empty() &&
            late_observation_pipeline.observations.size() == 1 &&
            late_observation_pipeline.observations[0].status ==
                ObservationStatus::ReachedObservationSurface,
        "lepton endpoint compaction lost observation-before-time-cut semantics");
    auto late_profile_config = config;
    auto& late_projection =
        late_profile_config.profile_projection;
    late_projection.enabled = true;
    late_projection.accumulate_on_device = true;
    late_projection.axis_start_position_m[2] =
        earth_radius_m + 120000.;
    late_projection.axis_direction[2] = -1.;
    late_projection.axis_step_length_m = 1000.;
    late_projection.axis_grammage_g_per_cm2.resize(130);
    for (std::size_t index = 0;
         index <
         late_projection.axis_grammage_g_per_cm2.size();
         ++index) {
      late_projection.axis_grammage_g_per_cm2[index] =
          10. * static_cast<double>(index);
    }
    late_projection.output_bin_count = 130;
    late_projection.output_bin_width_g_per_cm2 = 10.;
    late_projection.energy_loss_threshold_g_per_cm2 =
        1.e-4;
    late_projection.fixed_point_weight_limit = 1.e6;
    late_projection.fixed_point_energy_limit_GeV = 1.e6;
    CudaEmBackend late_profile_backend;
    late_profile_backend.initialize(
        environment, descriptor, late_profile_config);
    auto const late_profile_cascade =
        late_profile_backend
            .runResidentLeptonCascadeForValidation(
                {late_at_observation.particle}, 70000200,
                4,
                std::numeric_limits<std::uint64_t>::max(),
                1);
    auto const late_profile =
        late_profile_backend.downloadProfile();
    auto const late_ledger_tolerance_GeV = 2.e-6;
    require(
        late_profile_cascade.completed &&
            late_profile_cascade.observations.size() == 1 &&
            late_profile_cascade.observations[0].status ==
                ObservationStatus::ReachedObservationSurface &&
            late_profile.particle_cuts == 1 &&
            std::abs(
                late_profile.weighted_deposited_energy_GeV -
                (late_at_observation.particle.energy_GeV -
                 ElectronMassGeV)) <
                late_ledger_tolerance_GeV &&
            std::abs(
                late_profile
                        .weighted_cut_rest_mass_energy_GeV -
                    ElectronMassGeV) <
                late_ledger_tolerance_GeV &&
            std::abs(
                late_profile
                        .weighted_observed_total_energy_GeV -
                    late_profile_cascade.observations[0]
                        .particle.energy_GeV) <
                late_ledger_tolerance_GeV,
        "resident lepton observation-before-time-cut ledger is incomplete");
    auto off_axis_inclined = makeCandidate(
        11, 100., earth_radius_m, 100.1,
        {std::sin(80. * 3.141592653589793 / 180.),
         0.,
         -std::cos(80. * 3.141592653589793 / 180.)},
        71, 0);
    off_axis_inclined.particle.position_m[0] = 600.;
    auto const off_axis_straight =
        near_observation_backend
            .transportLeptonsStraightForValidation(
                {off_axis_inclined});
    require(
        off_axis_straight.fallback_events.empty() &&
            off_axis_straight.records.size() == 1 &&
            off_axis_straight.records[0].limit ==
                LeptonTransportLimit::ObservationSurface,
        "inclined off-axis straight lepton did not reach the plane");
    requireNear(
        off_axis_straight.records[0].end.position_m[2],
        environment.observation_plane_point_m[2], 2.e-15,
        "inclined straight lepton endpoint is not on the plane");
    require(
        off_axis_straight.records[0].limiting_radius_m >
            environment.observation_radius_m + 0.01,
        "inclined straight lepton still terminated on the legacy sphere");

    std::array<LeptonTransportLimit, 6> const expected_limits{
        LeptonTransportLimit::InteractionCandidate,
        LeptonTransportLimit::ContinuousStep,
        LeptonTransportLimit::ParticleCut,
        LeptonTransportLimit::LayerBoundary,
        LeptonTransportLimit::ObservationSurface,
        LeptonTransportLimit::EscapedEnvironment};
    for (std::size_t index = 0; index < first.records.size(); ++index) {
      auto const& record = first.records[index];
      require(record.input_index == index,
              "lepton record order is not stable");
      require(record.limit == expected_limits[index],
              "lepton nearest-limit classification differs at input " +
                  std::to_string(index) + ": actual=" +
                  std::to_string(
                      static_cast<int>(record.limit)) +
                  ", expected=" +
                  std::to_string(
                      static_cast<int>(expected_limits[index])));
      require(record.distance_m >= 0. &&
                  record.traversed_grammage_g_per_cm2 >= 0.,
              "lepton transport returned negative path data");
      require(
          std::abs(
              (record.start.energy_GeV - record.end.energy_GeV) -
              record.continuous_deposited_energy_GeV) <=
              2.e-15 * record.start.energy_GeV,
          "continuous energy deposit does not close");
      require(record.end.energy_GeV <= record.start.energy_GeV,
              "continuous transport increased particle energy");
    }

    auto const& interaction = first.records[0];
    require(interaction.interaction.status ==
                EmInteractionStatus::RequiresReselection,
            "charged interaction was not marked for vertex reselection");
    require(interaction.interaction.process_id == 0 &&
                interaction.interaction.component_hash == 0,
            "stale pre-transport process identity survived");
    require(interaction.interaction.total_rate_cm2_per_g == 2. &&
                interaction.interaction.process_uniform == 0.25,
            "interaction reselection lost its pre-step threshold");
    requireNear(
        interaction.end.energy_GeV * 1000.,
        energyAfterContinuousLoss(
            electron, 100.,
            interaction.traversed_grammage_g_per_cm2),
        2.e-13, "interaction vertex energy differs");
    require(interaction.end.step_id ==
                interaction.start.step_id,
            "interaction candidate advanced the Philox step");

    auto const& continuous = first.records[1];
    requireNear(continuous.end.energy_GeV, 0.09, 2.e-13,
                "continuous step did not enforce 10% loss");
    require(continuous.end.step_id ==
                continuous.start.step_id + 1,
            "continuous step did not advance Philox identity");

    auto const& cut = first.records[2];
    requireNear(
        cut.end.energy_GeV * 1000.,
        electron.minimum_total_energy_MeV, 2.e-13,
        "particle-cut endpoint differs");
    requireNear(
        cut.continuous_deposited_energy_GeV +
            cut.cut_deposited_energy_GeV,
        cut.start.energy_GeV - electron.mass_MeV / 1000.,
        2.e-13, "particle-cut energy accounting differs");

    for (std::size_t index = 3; index < 6; ++index) {
      require(first.records[index].end.step_id ==
                  first.records[index].start.step_id + 1,
              "geometric endpoint did not advance Philox identity");
    }
    auto const beta = std::sqrt(
        1. - std::pow(electron.mass_MeV / 100., 2));
    requireNear(
        interaction.end.time_s - interaction.start.time_s,
        interaction.distance_m / (beta * SpeedOfLightMPerS),
        2.e-13, "massive-particle flight time differs");

    std::vector<EmInteractionRecord> vertex_candidates;
    vertex_candidates.push_back(interaction.interaction);
    auto no_interaction = interaction.interaction;
    no_interaction.input_index = 10;
    no_interaction.particle.history_id = 10;
    no_interaction.total_rate_cm2_per_g = 8.;
    no_interaction.process_uniform = 0.75;
    vertex_candidates.push_back(no_interaction);
    auto ionization = interaction.interaction;
    ionization.input_index = 11;
    ionization.particle.history_id = 11;
    ionization.total_rate_cm2_per_g = 3.;
    ionization.process_uniform = 0.9;
    vertex_candidates.push_back(ionization);
    auto vertex_unsupported = interaction.interaction;
    vertex_unsupported.input_index = 12;
    vertex_unsupported.particle.history_id = 12;
    vertex_unsupported.particle.pid = 22;
    vertex_candidates.push_back(vertex_unsupported);

    auto const selected =
        backend.reselectLeptonInteractionsAtVertexForValidation(
            vertex_candidates);
    auto const selected_repeat =
        backend.reselectLeptonInteractionsAtVertexForValidation(
            vertex_candidates);
    require(selected.input_candidates == 4 &&
                selected.interactions.size() == 2 &&
                selected.continuations.size() == 1 &&
                selected.fallback_events.size() == 1,
            "lepton vertex classification counts differ");
    require(selected.interactions[0].input_index == 0 &&
                selected.interactions[0].process_id ==
                    BremsProcessId &&
                selected.interactions[1].input_index == 11 &&
                selected.interactions[1].process_id ==
                    IonizationProcessId,
            "lepton vertex threshold selected the wrong process");
    for (auto const& selected_interaction :
         selected.interactions) {
      require(selected_interaction.status ==
                  EmInteractionStatus::Selected &&
                  selected_interaction.vertex_total_rate_cm2_per_g ==
                      3.,
              "lepton vertex interaction metadata differs");
      RandomNumberKey const loss_key{
          config.random_seed, config.shower_id,
          selected_interaction.particle.history_id,
          selected_interaction.particle.step_id,
          static_cast<std::uint32_t>(
              selected_interaction.process_id),
          InteractionLossDrawId};
      require(
          selected_interaction.loss_quantile ==
              uniformOpen01(loss_key),
          "lepton vertex loss Philox key differs");
    }
    require(
        selected.continuations[0].input_index == 10 &&
            selected.continuations[0].status ==
                EmInteractionStatus::NoDiscreteInteraction &&
            selected.continuations[0].particle.step_id ==
                no_interaction.particle.step_id + 1,
        "reduced vertex rate did not create a fresh continuation");
    require(
        selected.fallback_events[0].input_index == 12 &&
            selected.fallback_events[0].reason ==
                ProposalFallbackReason::UnsupportedParticle,
        "unsupported vertex candidate fallback differs");
    require(
        selected_repeat.interactions.size() ==
                selected.interactions.size() &&
            selected_repeat.continuations.size() ==
                selected.continuations.size() &&
            selected_repeat.interactions[0].energy_fraction ==
                selected.interactions[0].energy_fraction &&
            selected_repeat.interactions[1].loss_quantile ==
                selected.interactions[1].loss_quantile,
        "lepton vertex reselection is not deterministic");

    auto dispatch_inputs = selected.interactions;
    auto explicit_cpu_fallback = selected.interactions.back();
    explicit_cpu_fallback.input_index = 13;
    explicit_cpu_fallback.process_id =
        PhotonuclearProcessId;
    dispatch_inputs.push_back(explicit_cpu_fallback);
    auto const dispatched =
        backend.generateBremsFinalStatesForValidation(
            dispatch_inputs, 1000);
    require(
        dispatched.gpu_interactions +
                dispatched.lpm_suppressed.size() ==
            2,
        "selected bremsstrahlung/ionization did not reach the GPU final state");
    require(
        dispatched.fallback_events.size() == 1 &&
            dispatched.fallback_events[0].process_id ==
                PhotonuclearProcessId &&
            dispatched.fallback_events[0].medium_hash == 9001 &&
            dispatched.fallback_events[0].interaction_hash == 9102,
        "unimplemented selected process did not fall back explicitly");
    auto const& specified_fallback =
        dispatched.fallback_events.front();
    require(
        hasSpecifiedProposalFinalState(specified_fallback),
        "selected CPU fallback is not a fully specified PROPOSAL interaction");
    auto const proposal_record =
        makeProposalInteractionRecord(
            specified_fallback, config.random_seed,
            config.shower_id);
    require(
        proposal_record.context.projectile_id ==
                Code::Electron &&
            proposal_record.context.medium_hash == 9001 &&
            proposal_record.interaction_hash == 9102 &&
            proposal_record.type ==
                PROPOSAL::InteractionType::Photonuclear &&
            proposal_record.component_hash ==
                specified_fallback.component_hash &&
            proposal_record.v_loss ==
                specified_fallback.energy_fraction &&
            proposal_record.selection_uniform ==
                specified_fallback.selection_uniform &&
            proposal_record.random_key &&
            proposal_record.random_key->history_id ==
                specified_fallback.particle.history_id,
        "specified fallback to PROPOSAL record conversion differs");
    auto const proposal_randoms =
        makeProposalFinalStateRandomNumbers(
            specified_fallback, 7, config.random_seed,
            config.shower_id);
    auto const repeated_proposal_randoms =
        makeProposalFinalStateRandomNumbers(
            specified_fallback, 7, config.random_seed,
            config.shower_id);
    require(
        proposal_randoms == repeated_proposal_randoms &&
            std::all_of(
                proposal_randoms.begin(),
                proposal_randoms.end(),
                [](double value) {
                  return value > 0. && value < 1.;
                }),
        "specified CPU PROPOSAL final-state Philox draws are not deterministic");

    auto const& statistics = backend.statistics();
    require(statistics.lepton_transport_batches == 2 &&
                statistics.lepton_transport_interaction_candidates == 2 &&
                statistics.lepton_transport_continuous_steps == 2 &&
                statistics.lepton_transport_cuts == 2 &&
                statistics.lepton_transport_boundaries == 2 &&
                statistics.lepton_transport_observations == 2 &&
                statistics.lepton_transport_escapes == 2 &&
                statistics.lepton_vertex_selection_batches == 2 &&
                statistics.lepton_vertex_interactions_selected == 4 &&
                statistics
                        .lepton_vertex_no_interaction_continuations ==
                    2,
            "lepton transport statistics differ");

    std::vector<EmParticleState> pipeline_particles;
    for (std::uint64_t index = 0; index < 64; ++index) {
      auto particle = interactions.front().particle;
      particle.pid =
          index % 2 == 0
              ? static_cast<std::int32_t>(EmPid::Electron)
              : static_cast<std::int32_t>(EmPid::Positron);
      particle.history_id = 100 + index;
      particle.parent_history_id = 0;
      particle.step_id = 3;
      pipeline_particles.push_back(particle);
    }

    CudaEmBackend staged_backend;
    staged_backend.initialize(environment, descriptor, config);
    auto const staged_selection =
        staged_backend.selectInteractionsForValidation(
            pipeline_particles);
    auto const staged_transport =
        staged_backend.transportLeptonsStraightForValidation(
            staged_selection.interactions);
    std::vector<EmInteractionRecord> staged_candidates;
    for (auto const& record : staged_transport.records) {
      if (record.limit ==
          LeptonTransportLimit::InteractionCandidate) {
        staged_candidates.push_back(record.interaction);
      }
    }
    auto const staged_vertex =
        staged_backend
            .reselectLeptonInteractionsAtVertexForValidation(
                staged_candidates);
    auto const staged_final =
        staged_backend.generateBremsFinalStatesForValidation(
            staged_vertex.interactions, 5000);

    CudaEmBackend pipeline_backend;
    pipeline_backend.initialize(environment, descriptor, config);
    auto const pipeline =
        pipeline_backend.runLeptonDevicePipelineForValidation(
            pipeline_particles, 5000);
    require(
        pipeline.input_particles == pipeline_particles.size() &&
            pipeline.interaction_candidates ==
                staged_candidates.size() &&
            pipeline.vertex_interactions ==
                staged_vertex.interactions.size(),
        "device-chained lepton pipeline stage counts differ");
    compareRecords(
        pipeline.transport_records,
        staged_transport.records);
    compareFallbacks(
        pipeline.selection_fallback_events,
        staged_selection.fallback_events,
        "device-chained selection fallbacks");
    compareFallbacks(
        pipeline.transport_fallback_events,
        staged_transport.fallback_events,
        "device-chained transport fallbacks");
    compareInteractionRecords(
        pipeline.vertex_continuations,
        staged_vertex.continuations,
        "device-chained vertex continuations");
    compareFallbacks(
        pipeline.vertex_fallback_events,
        staged_vertex.fallback_events,
        "device-chained vertex fallbacks");
    compareBremsResults(
        pipeline.final_states, staged_final);
    auto const expected_endpoints =
        reconstructLeptonEndpoints(
            pipeline, pipeline_particles.size());
    compareParticles(
        pipeline.next_leptons,
        expected_endpoints.next_leptons,
        "device lepton endpoint queue");
    compareParticles(
        pipeline.generated_photons,
        expected_endpoints.generated_photons,
        "device generated-photon queue");
    compareParticles(
        pipeline.decay_candidates,
        expected_endpoints.decay_candidates,
        "device decay-candidate queue");
    require(
        pipeline.observations.size() ==
            expected_endpoints.observations.size(),
        "device lepton observation count differs");
    for (std::size_t index = 0;
         index < pipeline.observations.size(); ++index) {
      require(
          pipeline.observations[index].status ==
                  expected_endpoints.observations[index]
                      .status &&
              sameParticle(
                  pipeline.observations[index].particle,
                  expected_endpoints.observations[index]
                      .particle),
          "device lepton observation differs");
    }
    require(
        pipeline.interaction_candidates > 0 &&
            pipeline.vertex_interactions > 0 &&
            (pipeline.final_states.gpu_interactions +
             pipeline.final_states.lpm_suppressed.size()) >
                0,
        "device-chained validation batch did not exercise bremsstrahlung");
    require(
        pipeline.final_states.annihilation_interactions > 0,
        "device-chained validation batch did not exercise positron annihilation");

    auto mixed_selection_particles = pipeline_particles;
    mixed_selection_particles[5].energy_GeV = 0.5;
    mixed_selection_particles[42].energy_GeV = 0.75;
    CudaEmBackend mixed_staged_backend;
    mixed_staged_backend.initialize(
        environment, descriptor, config);
    auto const mixed_selection =
        mixed_staged_backend
            .selectInteractionsForValidation(
                mixed_selection_particles);
    auto const mixed_transport =
        mixed_staged_backend
            .transportLeptonsStraightForValidation(
                mixed_selection.interactions);
    std::vector<EmInteractionRecord> mixed_candidates;
    for (auto const& record : mixed_transport.records) {
      if (record.limit ==
          LeptonTransportLimit::InteractionCandidate) {
        mixed_candidates.push_back(record.interaction);
      }
    }
    auto const mixed_vertex =
        mixed_staged_backend
            .reselectLeptonInteractionsAtVertexForValidation(
                mixed_candidates);
    auto const mixed_final =
        mixed_staged_backend
            .generateBremsFinalStatesForValidation(
                mixed_vertex.interactions, 7200000);
    CudaEmBackend mixed_pipeline_backend;
    mixed_pipeline_backend.initialize(
        environment, descriptor, config);
    auto const mixed_pipeline =
        mixed_pipeline_backend
            .runLeptonDevicePipelineForValidation(
                mixed_selection_particles, 7200000);
    require(
        mixed_selection.fallback_events.size() == 2 &&
            mixed_selection.interactions.size() ==
                mixed_selection_particles.size() - 2,
        "mixed deferred lepton selection did not exercise both branches");
    compareFallbacks(
        mixed_pipeline.selection_fallback_events,
        mixed_selection.fallback_events,
        "mixed deferred lepton selection");
    compareRecords(
        mixed_pipeline.transport_records,
        mixed_transport.records);
    compareFallbacks(
        mixed_pipeline.transport_fallback_events,
        mixed_transport.fallback_events,
        "mixed deferred lepton transport");
    compareInteractionRecords(
        mixed_pipeline.vertex_continuations,
        mixed_vertex.continuations,
        "mixed deferred lepton vertex continuations");
    compareFallbacks(
        mixed_pipeline.vertex_fallback_events,
        mixed_vertex.fallback_events,
        "mixed deferred lepton vertex fallbacks");
    compareBremsResults(
        mixed_pipeline.final_states, mixed_final);

    auto thinned_config = config;
    thinned_config.thinning = {
        1.e20, 1.e20, 1, 1};
    CudaEmBackend thinned_pipeline_backend;
    thinned_pipeline_backend.initialize(
        environment, descriptor, thinned_config);
    auto const thinned_pipeline =
        thinned_pipeline_backend
            .runLeptonDevicePipelineForValidation(
                pipeline_particles, 7000000);
    auto thinned_two_body_vertices = std::size_t{0};
    for (auto const& record :
         thinned_pipeline.final_states.final_state_records) {
      if (record.process_id == ElectronPairProcessId) {
        require(
            record.secondary_count == 3 &&
                record.thinning_status ==
                    static_cast<std::uint32_t>(
                        EmThinningStatus::NotApplied),
            "existing EMThinning semantics changed a 1->3 Epair state");
        continue;
      }
      ++thinned_two_body_vertices;
      require(
          record.secondary_count == 1 &&
              record.thinning_status ==
                  static_cast<std::uint32_t>(
                      EmThinningStatus::Hillas) &&
              (record.thinning_keep_mask == 0x1U ||
               record.thinning_keep_mask == 0x2U),
          "charged device pipeline did not Hillas-thin a 1->2 state");
    }
    require(
        thinned_two_body_vertices > 0,
        "charged thinning pipeline exercised no accepted 1->2 vertex");
    auto const thinned_endpoints =
        reconstructLeptonEndpoints(
            thinned_pipeline, pipeline_particles.size());
    compareParticles(
        thinned_pipeline.next_leptons,
        thinned_endpoints.next_leptons,
        "thinned device lepton endpoint queue");
    compareParticles(
        thinned_pipeline.generated_photons,
        thinned_endpoints.generated_photons,
        "thinned device generated-photon queue");
    compareParticles(
        thinned_pipeline.decay_candidates,
        thinned_endpoints.decay_candidates,
        "thinned device decay-candidate queue");

    for (auto const& final_record :
         pipeline.final_states.final_state_records) {
      auto const transport_record = std::find_if(
          pipeline.transport_records.begin(),
          pipeline.transport_records.end(),
          [&](LeptonTransportRecord const& record) {
            return record.input_index ==
                   final_record.input_index;
          });
      require(
          transport_record !=
              pipeline.transport_records.end(),
          "bremsstrahlung output lost its transport parent");
      auto const child_offset =
          static_cast<std::size_t>(
              final_record.secondary_offset);
      require(
          child_offset + 1 <
              pipeline.final_states.secondaries.size(),
          "bremsstrahlung secondary offset is invalid");
      auto const children_energy =
          pipeline.final_states
              .secondaries[child_offset]
              .energy_GeV +
          pipeline.final_states
              .secondaries[child_offset + 1]
              .energy_GeV;
      requireNear(
          transport_record->continuous_deposited_energy_GeV +
              children_energy,
          transport_record->start.energy_GeV +
              ((final_record.process_id ==
                        AnnihilationProcessId ||
                final_record.process_id ==
                        IonizationProcessId)
                   ? 0.0005109989461
                   : 0.),
          2.e-13,
          "continuous-loss plus lepton final-state energy does not close");
    }
    auto const& staged_statistics =
        staged_backend.statistics();
    auto const& pipeline_statistics =
        pipeline_backend.statistics();
    require(
        pipeline_statistics.physical_lepton_wavefronts == 1 &&
            pipeline_statistics
                    .lepton_selection_transport_summary_fusions ==
                1 &&
            pipeline_statistics
                    .lepton_transport_vertex_summary_fusions ==
                1 &&
            pipeline_statistics
                    .lepton_vertex_final_state_summary_fusions ==
                1 &&
            pipeline_statistics
                    .lepton_final_state_endpoint_summary_fusions ==
                1 &&
            pipeline_statistics
                    .pipeline_host_synchronizations_eliminated ==
                4 &&
            pipeline_statistics
                    .pipeline_device_to_host_bytes_eliminated ==
                20 &&
            pipeline_statistics
                    .physical_host_to_device_bytes <
                staged_statistics
                    .physical_host_to_device_bytes &&
            pipeline_statistics
                    .physical_device_to_host_bytes <
                staged_statistics
                    .physical_device_to_host_bytes,
        "device chaining did not reduce host/device traffic");

    std::vector<EmParticleState> resident_input(
        pipeline_particles.begin(),
        pipeline_particles.begin() + 8);
    std::uint64_t constexpr ResidentFirstHistory = 8000000;
    std::size_t constexpr ResidentWavefrontLimit = 128;
    CudaEmBackend host_driven_backend;
    host_driven_backend.initialize(
        environment, descriptor, config);
    std::vector<EmParticleState> host_current =
        resident_input;
    std::vector<EmParticleState> host_photons;
    std::vector<ProposalFallbackEvent> host_fallbacks;
    std::vector<ObservationRecord> host_observations;
    std::vector<LeptonTransportRecord> host_steps;
    std::vector<BremsFinalStateRecord> host_final_records;
    auto host_next_history = ResidentFirstHistory;
    std::size_t host_wavefronts = 0;
    while (!host_current.empty() &&
           host_wavefronts < ResidentWavefrontLimit) {
      auto const wave =
          host_driven_backend
              .runLeptonDevicePipelineForValidation(
                  host_current, host_next_history);
      ++host_wavefronts;
      host_steps.insert(
          host_steps.end(), wave.transport_records.begin(),
          wave.transport_records.end());
      host_photons.insert(
          host_photons.end(),
          wave.generated_photons.begin(),
          wave.generated_photons.end());
      host_observations.insert(
          host_observations.end(),
          wave.observations.begin(),
          wave.observations.end());
      host_final_records.insert(
          host_final_records.end(),
          wave.final_states.final_state_records.begin(),
          wave.final_states.final_state_records.end());
      auto appendFallbacks =
          [&](auto const& events) {
            host_fallbacks.insert(
                host_fallbacks.end(),
                events.begin(), events.end());
          };
      appendFallbacks(wave.selection_fallback_events);
      appendFallbacks(wave.transport_fallback_events);
      appendFallbacks(wave.vertex_fallback_events);
      appendFallbacks(
          wave.final_states.fallback_events);
      require(
          wave.final_states.secondaries.size() <=
              std::numeric_limits<std::uint64_t>::max() -
                  host_next_history,
          "host-driven resident oracle history overflow");
      host_next_history +=
          wave.final_states.secondaries.size();
      host_current = wave.next_leptons;
    }

    CudaEmBackend resident_backend;
    resident_backend.initialize(
        environment, descriptor, config);
    auto const resident =
        resident_backend
            .runResidentLeptonCascadeForValidation(
                resident_input, ResidentFirstHistory,
                ResidentWavefrontLimit);
    require(
        resident.wavefronts == host_wavefronts &&
            resident.completed == host_current.empty() &&
            resident.input_particles ==
                resident_input.size(),
        "resident lepton cascade lifecycle differs");
    compareParticles(
        resident.generated_photons, host_photons,
        "resident generated photons");
    compareParticles(
        resident.remaining_leptons, host_current,
        "resident lepton checkpoint");
    compareFallbacks(
        resident.fallback_events, host_fallbacks,
        "resident lepton fallbacks");
    require(
        resident.observations.size() ==
                host_observations.size() &&
            resident.step_records.size() ==
                host_steps.size() &&
            resident.final_state_records.size() ==
                host_final_records.size(),
        "resident lepton host-output counts differ");
    for (std::size_t index = 0;
         index < resident.observations.size(); ++index) {
      require(
          resident.observations[index].status ==
                  host_observations[index].status &&
              sameParticle(
                  resident.observations[index].particle,
                  host_observations[index].particle),
          "resident lepton observation differs");
    }
    for (std::size_t index = 0;
         index < resident.step_records.size(); ++index) {
      auto const& left = resident.step_records[index];
      auto const& right = host_steps[index];
      require(
          left.input_index == right.input_index &&
              left.limit == right.limit &&
              sameParticle(left.start, right.start) &&
              sameParticle(left.end, right.end),
          "resident lepton transport record differs");
    }
    for (std::size_t index = 0;
         index < resident.final_state_records.size();
         ++index) {
      auto const& left =
          resident.final_state_records[index];
      auto const& right = host_final_records[index];
      require(
          left.input_index == right.input_index &&
              left.parent_history_id ==
                  right.parent_history_id &&
              left.process_id == right.process_id &&
              left.secondary_offset ==
                  right.secondary_offset &&
              left.photon_energy_fraction ==
                  right.photon_energy_fraction,
          "resident lepton final-state record differs");
    }
    require(
        resident_backend.statistics()
                .physical_host_to_device_bytes <
            host_driven_backend.statistics()
                .physical_host_to_device_bytes,
        "resident lepton cascade did not reduce host-to-device traffic");

    {
      std::uint64_t constexpr CrossFirstHistory =
          9000000;
      CudaEmBackend cross_oracle;
      cross_oracle.initialize(
          environment, descriptor, config);
      auto const oracle =
          cross_oracle
              .runResidentLeptonCascadeForValidation(
                  pipeline_particles, CrossFirstHistory,
                  ResidentWavefrontLimit,
                  std::numeric_limits<std::uint64_t>::max(),
                  1);
      require(
          !oracle.generated_photons.empty(),
          "cross-species fixture did not generate bremsstrahlung photons");

      auto cross_config = config;
      cross_config.resident_cross_species = true;
      CudaEmBackend cross_backend;
      cross_backend.initialize(
          environment, descriptor, cross_config);
      auto const device_leptons =
          cross_backend
              .runResidentLeptonCascadeForValidation(
                  pipeline_particles, CrossFirstHistory,
                  ResidentWavefrontLimit,
                  std::numeric_limits<std::uint64_t>::max(),
                  1);
      auto const pending_photons =
          cross_backend.pendingPhotonCount();
      require(
          device_leptons.generated_photons.empty() &&
              pending_photons ==
                  oracle.generated_photons.size() &&
              cross_backend.pendingLeptonCount() == 0,
          "lepton photons did not remain in the resident device queue");

      auto const device_photons =
          cross_backend
              .runResidentPhotonCascadeForValidation(
                  {}, 10000000, 128, 1);
      auto const& cross_statistics =
          cross_backend.statistics();
      require(
          device_photons.input_particles ==
                  pending_photons &&
              cross_backend.pendingPhotonCount() == 0 &&
              cross_backend.pendingLeptonCount() > 0 &&
              device_photons
                  .electromagnetic_secondaries.empty() &&
              cross_statistics.cross_species_host_spills ==
                  0 &&
              cross_statistics
                      .cross_species_particles_kept_on_device >
                  pending_photons,
          "bidirectional resident cross-species routing did not close");
    }

    CudaEmBackend range_limited_backend;
    range_limited_backend.initialize(
        environment, descriptor, config);
    auto const history_limit =
        ResidentFirstHistory +
        3 * resident_input.size();
    auto const range_limited =
        range_limited_backend
            .runResidentLeptonCascadeForValidation(
                resident_input, ResidentFirstHistory,
                ResidentWavefrontLimit, history_limit);
    require(
        range_limited.history_range_exhausted &&
            !range_limited.completed &&
            !range_limited.remaining_leptons.empty() &&
            range_limited.secondary_history_ids_used <=
                3 * resident_input.size(),
        "resident lepton history range did not create a safe checkpoint");
    for (auto const& photon :
         range_limited.generated_photons) {
      require(
          photon.history_id < history_limit,
          "resident photon exceeded its reserved history range");
    }
    for (auto const& lepton :
         range_limited.remaining_leptons) {
      require(
          lepton.history_id < history_limit ||
              lepton.history_id <
                  ResidentFirstHistory,
          "resident checkpoint exceeded its reserved history range");
    }

    environment.magnetic_field_T[0] = 1.e-5;
    environment.maximum_magnetic_deflection_rad = 0.001;
    CudaEmBackend magnetic;
    magnetic.initialize(environment, descriptor, config);
    requireThrows(
        [&] {
          magnetic.transportLeptonsStraightForValidation(
              {interactions.front()});
        },
        "straight validation silently ignored a magnetic field");
    std::vector<EmParticleState> magnetic_particles;
    for (std::uint64_t index = 0; index < 16; ++index) {
      auto particle = interactions.front().particle;
      particle.pid =
          index % 2 == 0
              ? static_cast<std::int32_t>(EmPid::Electron)
              : static_cast<std::int32_t>(EmPid::Positron);
      particle.history_id = 10000 + index;
      particle.step_id = 9;
      magnetic_particles.push_back(particle);
    }
    auto const magnetic_pipeline =
        magnetic.runLeptonDevicePipelineForValidation(
            magnetic_particles, 20000);
    auto const magnetic_repeat =
        magnetic.runLeptonDevicePipelineForValidation(
            magnetic_particles, 20000);
    require(
        magnetic_pipeline.transport_fallback_events.empty() &&
            magnetic_pipeline.transport_records.size() ==
                magnetic_particles.size(),
        "uniform-field device pipeline did not transport every lepton");
    require(
        magnetic_repeat.transport_records.size() ==
            magnetic_pipeline.transport_records.size(),
        "uniform-field repeat changed the transport count");
    std::size_t chord_grammage_checks = 0;
    std::size_t tangent_grammage_differences = 0;
    for (std::size_t index = 0;
         index < magnetic_pipeline.transport_records.size();
         ++index) {
      auto const& record =
          magnetic_pipeline.transport_records[index];
      auto const& repeated =
          magnetic_repeat.transport_records[index];
      auto const charge =
          record.start.pid ==
                  static_cast<std::int32_t>(EmPid::Electron)
              ? -1.
              : 1.;
      auto const expected = advanceUniformMagneticField(
          record.start, electron.mass_MeV / 1000., charge,
          environment.magnetic_field_T, record.distance_m);
      auto const expected_limit = maximumUniformMagneticStep(
          record.start, electron.mass_MeV / 1000., charge,
          environment.magnetic_field_T,
          environment.maximum_magnetic_deflection_rad);
      require(
          record.magnetic_bending_applied == 1 &&
              record.magnetic_step_status ==
                  static_cast<std::uint32_t>(
                      MagneticStepStatus::Success) &&
              expected.status == MagneticStepStatus::Success,
          "uniform-field pipeline omitted magnetic bending");
      requireNear(
          record.magnetic_step_limit_m,
          expected_limit.distance_m, 2.e-13,
          "uniform-field pipeline ignored configured magnetic deflection");
      double chord_direction[3]{};
      auto chord_length_squared = 0.;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        chord_direction[axis] =
            record.end.position_m[axis] -
            record.start.position_m[axis];
        chord_length_squared +=
            chord_direction[axis] * chord_direction[axis];
      }
      auto const chord_length =
          std::sqrt(chord_length_squared);
      require(chord_length > 0.,
              "uniform-field pipeline produced a zero chord");
      for (auto& component : chord_direction) {
        component /= chord_length;
      }
      auto const chord_grammage = atmosphereGrammage(
          environment, record.start_layer_index,
          record.start.position_m, chord_direction,
          chord_length);
      auto const tangent_grammage = atmosphereGrammage(
          environment, record.start_layer_index,
          record.start.position_m, record.start.direction,
          record.distance_m);
      require(
          chord_grammage.status == AtmosphereStatus::Success &&
              tangent_grammage.status ==
                  AtmosphereStatus::Success,
          "uniform-field grammage oracle failed");
      requireNear(
          record.traversed_grammage_g_per_cm2,
          chord_grammage.value, 3.e-13,
          "uniform-field transport did not use the CPU Step chord grammage");
      if (record.limit != LeptonTransportLimit::ParticleCut) {
        requireNear(
            record.end.energy_GeV * 1000.,
            energyAfterContinuousLoss(
                findContinuousEnergyTable(
                    source, record.start.pid),
                record.start.energy_GeV * 1000.,
                chord_grammage.value),
            3.e-13,
            "uniform-field continuous loss did not use the CPU Step chord");
      }
      ++chord_grammage_checks;
      auto const grammage_scale = std::max(
          {1.e-300, std::abs(chord_grammage.value),
           std::abs(tangent_grammage.value)});
      if (std::abs(
              chord_grammage.value -
              tangent_grammage.value) >
          1.e-12 * grammage_scale) {
        ++tangent_grammage_differences;
      }
      for (std::size_t axis = 0; axis < 3; ++axis) {
        requireNear(
            record.end.position_m[axis],
            expected.particle.position_m[axis], 2.e-13,
            "uniform-field pipeline position differs");
        requireNear(
            record.end.direction[axis],
            expected.particle.direction[axis], 2.e-13,
            "uniform-field pipeline direction differs");
        require(
            repeated.end.position_m[axis] ==
                    record.end.position_m[axis] &&
                repeated.end.direction[axis] ==
                    record.end.direction[axis],
            "uniform-field pipeline is not deterministic");
      }
      auto const radius = std::sqrt(
          std::pow(
              record.end.position_m[0] -
                  environment.earth_center_m[0],
              2) +
          std::pow(
              record.end.position_m[1] -
                  environment.earth_center_m[1],
              2) +
          std::pow(
              record.end.position_m[2] -
                  environment.earth_center_m[2],
              2));
      if (record.limit ==
          LeptonTransportLimit::ObservationSurface) {
        auto plane_residual = 0.;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          plane_residual +=
              (record.end.position_m[axis] -
               environment.observation_plane_point_m[axis]) *
              environment.observation_plane_normal[axis];
        }
        require(
            std::abs(plane_residual) < 1.e-8,
            "curved lepton endpoint is off the observation plane");
      } else if (record.limit ==
                     LeptonTransportLimit::LayerBoundary ||
                 record.limit ==
                     LeptonTransportLimit::EscapedEnvironment) {
        requireNear(
            radius, record.limiting_radius_m, 2.e-12,
            "curved lepton endpoint is off its spherical boundary");
      }
    }
    require(
        chord_grammage_checks ==
                magnetic_pipeline.transport_records.size() &&
            tangent_grammage_differences > 0,
        "uniform-field test did not distinguish the CPU Step chord from the "
        "legacy start-tangent grammage");
    auto near_observation_magnetic =
        near_observation.particle;
    near_observation_magnetic.history_id = 10020;
    near_observation_magnetic.step_id = 9;
    auto const magnetic_near_observation =
        magnetic.runLeptonDevicePipelineForValidation(
            {near_observation_magnetic}, 20030);
    require(
        magnetic_near_observation
                .transport_fallback_events.empty() &&
            magnetic_near_observation
                    .transport_records.size() == 1 &&
            magnetic_near_observation
                    .transport_records[0]
                    .limit ==
                LeptonTransportLimit::ObservationSurface &&
            magnetic_near_observation
                    .transport_records[0]
                    .distance_m <
                AtmosphereBoundaryGuardM &&
            magnetic_near_observation
                    .observations.size() == 1,
        "near-observation magnetic lepton must terminate on the GPU");

    auto inclined_magnetic_environment = environment;
    inclined_magnetic_environment.magnetic_field_T[0] = 0.;
    inclined_magnetic_environment.magnetic_field_T[1] = 2.e-5;
    inclined_magnetic_environment.magnetic_field_T[2] = 0.;
    CudaEmBackend inclined_magnetic_backend;
    inclined_magnetic_backend.initialize(
        inclined_magnetic_environment, descriptor, config);
    auto inclined_magnetic_particle = off_axis_inclined.particle;
    inclined_magnetic_particle.history_id = 10021;
    inclined_magnetic_particle.step_id = 9;
    auto const inclined_magnetic_step = maximumUniformMagneticStep(
        inclined_magnetic_particle, electron.mass_MeV / 1000.,
        -1., inclined_magnetic_environment.magnetic_field_T,
        inclined_magnetic_environment.maximum_magnetic_deflection_rad);
    auto const plane_limit = intersectUniformMagneticPlane(
        inclined_magnetic_particle, electron.mass_MeV / 1000.,
        -1., inclined_magnetic_environment.magnetic_field_T,
        inclined_magnetic_environment,
        inclined_magnetic_step.distance_m,
        inclined_magnetic_environment.maximum_magnetic_deflection_rad);
    require(
        plane_limit.status == MagneticIntersectionStatus::Success,
        "inclined magnetic plane oracle found no crossing");
    auto const momentum_GeV = std::sqrt(
        (inclined_magnetic_particle.energy_GeV -
         electron.mass_MeV / 1000.) *
        (inclined_magnetic_particle.energy_GeV +
         electron.mass_MeV / 1000.));
    std::array<double, 3> direction_cross_field{
        inclined_magnetic_particle.direction[1] *
                inclined_magnetic_environment.magnetic_field_T[2] -
            inclined_magnetic_particle.direction[2] *
                inclined_magnetic_environment.magnetic_field_T[1],
        inclined_magnetic_particle.direction[2] *
                inclined_magnetic_environment.magnetic_field_T[0] -
            inclined_magnetic_particle.direction[0] *
                inclined_magnetic_environment.magnetic_field_T[2],
        inclined_magnetic_particle.direction[0] *
                inclined_magnetic_environment.magnetic_field_T[1] -
            inclined_magnetic_particle.direction[1] *
                inclined_magnetic_environment.magnetic_field_T[0]};
    auto scalar_a = 0.;
    auto scalar_b = 0.;
    auto scalar_c = 0.;
    auto const scalar_curvature =
        -GeVPerCToTeslaMeter / momentum_GeV;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      auto const normal = inclined_magnetic_environment
                              .observation_plane_normal[axis];
      scalar_a += 0.5 * scalar_curvature * normal *
                  direction_cross_field[axis];
      scalar_b += normal * inclined_magnetic_particle.direction[axis];
      scalar_c +=
          normal *
          (inclined_magnetic_particle.position_m[axis] -
           inclined_magnetic_environment
               .observation_plane_point_m[axis]);
    }
    auto const scalar_roots =
        solve_quadratic_real(scalar_a, scalar_b, scalar_c);
    auto scalar_distance =
        std::numeric_limits<double>::infinity();
    for (auto const root : scalar_roots) {
      if (root > 0. && root < scalar_distance) {
        scalar_distance = root;
      }
    }
    requireNear(
        plane_limit.distance_m, scalar_distance, 2.e-13,
        "CUDA magnetic plane root differs from the scalar quadratic solver");
    auto const plane_endpoint = advanceUniformMagneticField(
        inclined_magnetic_particle, electron.mass_MeV / 1000.,
        -1., inclined_magnetic_environment.magnetic_field_T,
        plane_limit.distance_m);
    auto plane_residual = 0.;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      plane_residual +=
          (plane_endpoint.particle.position_m[axis] -
           inclined_magnetic_environment
               .observation_plane_point_m[axis]) *
          inclined_magnetic_environment
              .observation_plane_normal[axis];
    }
    require(
        plane_endpoint.status == MagneticStepStatus::Success &&
            std::abs(plane_residual) < 1.e-8,
        "curved plane root does not land on the CPU observation plane");
    auto const inclined_magnetic_pipeline =
        inclined_magnetic_backend
            .runLeptonDevicePipelineForValidation(
                {inclined_magnetic_particle}, 20040);
    require(
        inclined_magnetic_pipeline.transport_fallback_events.empty() &&
            inclined_magnetic_pipeline.transport_records.size() == 1 &&
            inclined_magnetic_pipeline.transport_records[0].limit ==
                LeptonTransportLimit::ObservationSurface &&
            inclined_magnetic_pipeline.observations.size() == 1,
        "inclined off-axis magnetic lepton did not terminate on the plane");
    auto const& inclined_record =
        inclined_magnetic_pipeline.transport_records[0];
    auto inclined_residual = 0.;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      inclined_residual +=
          (inclined_record.end.position_m[axis] -
           inclined_magnetic_environment
               .observation_plane_point_m[axis]) *
          inclined_magnetic_environment
              .observation_plane_normal[axis];
    }
    require(
        std::abs(inclined_residual) < 1.e-8 &&
            inclined_record.limiting_radius_m >
                inclined_magnetic_environment.observation_radius_m +
                    0.01,
        "inclined magnetic transport still terminated on the sphere");

    // Production regression from proton seed 10200251, shower 22.  The
    // maximum-deflection step ends about 43 micrometres inside the USStdBK
    // 11.4 km boundary.  Direction-aware layer ownership already assigns
    // that guarded endpoint to the outward layer, so it is a valid layer
    // transition rather than a magnetic transport failure.
    auto boundary_guard_environment = environment;
    boundary_guard_environment.magnetic_field_T[0] =
        2.5067327193094893e-05;
    boundary_guard_environment.magnetic_field_T[1] =
        -1.10186120465756e-06;
    boundary_guard_environment.magnetic_field_T[2] =
        -5.1031311292104371e-05;
    boundary_guard_environment.maximum_magnetic_deflection_rad = 0.2;
    CudaEmBackend boundary_guard_magnetic;
    boundary_guard_magnetic.initialize(
        boundary_guard_environment, descriptor, config);
    EmParticleState boundary_guard_particle{};
    boundary_guard_particle.pid =
        static_cast<std::int32_t>(EmPid::Electron);
    boundary_guard_particle.medium_id = 17;
    boundary_guard_particle.energy_GeV =
        0.0011808222196131866;
    boundary_guard_particle.position_m[0] =
        -32.82159051860207;
    boundary_guard_particle.position_m[1] =
        9.385832804803458;
    boundary_guard_particle.position_m[2] =
        6382399.0942817135;
    boundary_guard_particle.direction[0] =
        -0.4188050119591029;
    boundary_guard_particle.direction[1] =
        0.05284253648793534;
    boundary_guard_particle.direction[2] =
        0.9065373838377857;
    boundary_guard_particle.weight = 1.;
    boundary_guard_particle.history_id = 10021;
    boundary_guard_particle.step_id = 13;
    std::vector<EmParticleState> boundary_guard_particles;
    for (std::uint64_t index = 0; index < 32; ++index) {
      auto particle = boundary_guard_particle;
      particle.history_id += index;
      boundary_guard_particles.push_back(particle);
    }
    auto const boundary_guard_result =
        boundary_guard_magnetic.runLeptonDevicePipelineForValidation(
            boundary_guard_particles, 30031);
    auto const boundary_guard_record_count =
        boundary_guard_result.transport_records.size();
    auto const boundary_guard_fallback_count =
        boundary_guard_result.transport_fallback_events.size();
    auto const accepted_guard_transition = std::any_of(
        boundary_guard_result.transport_records.begin(),
        boundary_guard_result.transport_records.end(),
        [](LeptonTransportRecord const& record) {
          return record.limit ==
                     LeptonTransportLimit::LayerBoundary &&
                 record.start_layer_index == 1 &&
                 record.end_layer_index == 2 &&
                 record.limiting_radius_m == 6382400.;
        });
    require(
        boundary_guard_result.transport_fallback_events.empty() &&
            boundary_guard_result.transport_records.size() ==
                boundary_guard_particles.size() &&
            accepted_guard_transition,
        "guard-owned adjacent layer endpoint was not accepted as a transition: "
        "records=" + std::to_string(boundary_guard_record_count) +
            ", fallbacks=" +
            std::to_string(boundary_guard_fallback_count));

    auto moliere_source = source;
    moliere_source.metadata.moliere =
        makeMoliereMetadata();
    auto moliere_path = temporary;
    moliere_path += ".moliere";
    auto const moliere_digest =
        writeRateTable(moliere_path, moliere_source);
    ProposalTableSet moliere_descriptor{};
    moliere_descriptor.process_count =
        static_cast<std::uint32_t>(
            processCount(moliere_source));
    moliere_descriptor.content_hash = moliere_digest;
    auto moliere_config = config;
    moliere_config.table_cache = moliere_path;
    // The physical-router section below intentionally starts from one
    // particle and tests the GPU path itself, not scalar small-front
    // expansion.
    moliere_config.min_batch_size = 1;
    CudaEmBackend moliere_backend;
    moliere_backend.initialize(
        environment, moliere_descriptor, moliere_config);
    auto const scattered =
        moliere_backend.runLeptonDevicePipelineForValidation(
            magnetic_particles, 30000);
    auto const scattered_repeat =
        moliere_backend.runLeptonDevicePipelineForValidation(
            magnetic_particles, 30000);
    require(
        scattered.multiple_scattering_enabled &&
            scattered.transport_fallback_events.empty() &&
            scattered.transport_records.size() ==
                magnetic_particles.size(),
        "versioned Moliere metadata did not enable scattering");
    std::size_t deflections = 0;
    std::size_t cpu_step_composition_differences = 0;
    for (std::size_t index = 0;
         index < scattered.transport_records.size(); ++index) {
      auto const& record =
          scattered.transport_records[index];
      auto const& repeated =
          scattered_repeat.transport_records[index];
      require(
          record.multiple_scattering_status ==
                  static_cast<std::uint32_t>(
                      MoliereStatus::Success) ||
              record.multiple_scattering_status ==
                  static_cast<std::uint32_t>(
                      MoliereStatus::NoDeflection),
          "integrated Moliere sampling failed");
      require(
          record.multiple_scattering_first_uniform ==
                  repeated.multiple_scattering_first_uniform &&
              record.multiple_scattering_second_uniform ==
                  repeated.multiple_scattering_second_uniform &&
              record.multiple_scattering_azimuth_uniform ==
                  repeated.multiple_scattering_azimuth_uniform,
          "integrated Moliere Philox draws are not deterministic");
      if (record.multiple_scattering_applied == 0) {
        continue;
      }
      ++deflections;
      auto const charge =
          record.start.pid ==
                  static_cast<std::int32_t>(EmPid::Electron)
              ? -1.
              : 1.;
      auto const magnetic_endpoint =
          advanceUniformMagneticField(
              record.start, electron.mass_MeV / 1000.,
              charge, environment.magnetic_field_T,
              record.distance_m);
      auto const scattered_from_start =
          applyMoliereDirection(
              record.start.direction,
              record.multiple_scattering_angle_rad,
              record.multiple_scattering_azimuth_uniform);
      require(
          scattered_from_start.status ==
              MoliereStatus::Success,
          "independent start-direction Moliere rotation failed");
      double expected_direction[3]{};
      auto expected_norm_squared = 0.;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        expected_direction[axis] =
            magnetic_endpoint.particle.direction[axis] +
            scattered_from_start.direction[axis] -
            record.start.direction[axis];
        expected_norm_squared +=
            expected_direction[axis] *
            expected_direction[axis];
      }
      auto const expected_norm =
          std::sqrt(expected_norm_squared);
      require(
          std::isfinite(expected_norm) &&
              expected_norm > 0.,
          "independent CPU Step direction is invalid");
      auto const sequential_direction =
          applyMoliereDirection(
              magnetic_endpoint.particle.direction,
              record.multiple_scattering_angle_rad,
              record.multiple_scattering_azimuth_uniform);
      require(
          sequential_direction.status ==
              MoliereStatus::Success,
          "sequential comparison rotation failed");
      auto direction_norm = 0.;
      auto sequential_difference_squared = 0.;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        expected_direction[axis] /= expected_norm;
        direction_norm += record.end.direction[axis] *
                          record.end.direction[axis];
        requireNear(
            record.end.direction[axis],
            expected_direction[axis],
            2.e-13,
            "Moliere/magnetic composition differs from CPU Step");
        auto const sequential_difference =
            record.end.direction[axis] -
            sequential_direction.direction[axis];
        sequential_difference_squared +=
            sequential_difference *
            sequential_difference;
        require(
            record.end.direction[axis] ==
                repeated.end.direction[axis],
            "integrated Moliere direction is not deterministic");
      }
      if (sequential_difference_squared > 1.e-28) {
        ++cpu_step_composition_differences;
      }
      requireNear(
          direction_norm, 1., 2.e-13,
          "combined magnetic/Moliere direction is not normalized");
    }
    require(deflections > 0,
            "integrated Moliere test sampled no deflection");
    require(
        cpu_step_composition_differences > 0,
        "integrated test did not distinguish CPU Step composition "
        "from sequential endpoint rotation");
    auto const& moliere_statistics =
        moliere_backend.statistics();
    require(
        moliere_statistics.moliere_trials ==
                2 * scattered.transport_records.size() &&
            moliere_statistics.moliere_deflections >=
                2 * deflections,
        "integrated Moliere statistics differ");

    auto const& coordinate_system =
        get_root_CoordinateSystem();
    TestCascadeIdentityStack routed_stack;
    auto routed_particle = routed_stack.addParticle(
        std::make_tuple(
            Code::Electron,
            100_MeV - get_mass(Code::Electron),
            DirectionVector{
                coordinate_system, {0., 0., -1.}},
            Point{
                coordinate_system, 0_m, 0_m,
                (earth_radius_m + 50000.) * meter},
            0_s));
    auto const routed_history =
        routed_particle.getHistoryId();
    auto const routed_step =
        routed_particle.beginTransportStep();
    SpecifiedFallbackSink specified_fallback_sink;
    PhysicalCudaEmRouter<
        TestCascadeIdentityStack,
        SpecifiedFallbackSink>
        router{
            moliere_backend, coordinate_system,
            environment, specified_fallback_sink};
    require(
        router.canRoute(routed_particle, routed_step),
        "physical HybridCascade router rejected an in-atmosphere electron");
    router.stage(
        routed_particle, routed_history,
        routed_particle.getParentHistoryId(),
        routed_particle.getGeneration(), routed_step);
    routed_particle.erase();

    std::size_t router_wavefronts = 0;
    std::size_t scalar_fallbacks = 0;
    while (router.pending() || !routed_stack.isEmpty()) {
      require(++router_wavefronts < 2048,
              "physical HybridCascade router did not terminate");
      if (router.pending()) {
        router.advanceOneWavefrontAndReturn(routed_stack);
      }
      while (!routed_stack.isEmpty()) {
        auto fallback = routed_stack.getNextParticle();
        auto const fallback_step =
            fallback.beginTransportStep();
        require(
            !router.canRoute(fallback, fallback_step),
            "explicit GPU fallback was immediately routed back to the GPU");
        ++scalar_fallbacks;
        fallback.erase();
      }
    }
    auto const& router_statistics = router.statistics();
    require(
        router_statistics.particles_staged == 1 &&
            router_statistics.leptons_advanced > 0 &&
            router_statistics.wavefronts > 0 &&
            router_statistics.reserved_history_ids > 0,
        "physical HybridCascade router statistics are incomplete: staged=" +
            std::to_string(router_statistics.particles_staged) +
            ", leptons=" +
            std::to_string(router_statistics.leptons_advanced) +
            ", wavefronts=" +
            std::to_string(router_statistics.wavefronts) +
            ", reserved=" +
            std::to_string(router_statistics.reserved_history_ids));
    // The sampled 100 MeV route is not required to hit a rare specified
    // final state. Dedicated tests above exercise that path directly; here
    // we require exact ownership accounting whenever it does occur.
    require(
        router_statistics.cpu_fallback_steps_executed ==
                scalar_fallbacks &&
            router_statistics
                    .particles_returned_for_cpu_fallback ==
                scalar_fallbacks &&
            router_statistics.specified_cpu_final_states ==
                specified_fallback_sink.handled,
        "physical HybridCascade CPU fallback ownership differs: "
        "executed=" +
            std::to_string(
                router_statistics.cpu_fallback_steps_executed) +
            ", returned=" +
            std::to_string(
                router_statistics
                    .particles_returned_for_cpu_fallback) +
            ", scalar=" +
            std::to_string(scalar_fallbacks) +
            ", specified=" +
            std::to_string(
                router_statistics.specified_cpu_final_states) +
            ", handled=" +
            std::to_string(specified_fallback_sink.handled));
    require(
        !router.stepRecords().empty() &&
            !router.radioTracks().empty() &&
            router.radioTracks().size() <=
                router.stepRecords().size(),
        "physical HybridCascade router lost its lepton track records");
    for (auto const& radio : router.radioTracks()) {
      auto const step = std::find_if(
          router.stepRecords().begin(),
          router.stepRecords().end(),
          [&](EmStepRecord const& candidate) {
            return candidate.history_id ==
                       radio.step.history_id &&
                   candidate.step_id ==
                       radio.step.step_id;
          });
      require(
          step != router.stepRecords().end() &&
              radio.step.pid == step->pid &&
              step->deposited_energy_GeV >= 0.,
          "physical HybridCascade radio/step record identity differs");
    }
    auto fresh_primary = routed_stack.addParticle(
        std::make_tuple(
            Code::Proton, 1_GeV,
            DirectionVector{
                coordinate_system, {0., 0., -1.}},
            Point{
                coordinate_system, 0_m, 0_m,
                (earth_radius_m + 1000.) * meter},
            0_s));
    require(
        fresh_primary.getHistoryId() >
            routed_history +
                router_statistics.reserved_history_ids,
        "CPU history allocator reused a GPU-reserved identity");
    fresh_primary.erase();
    std::filesystem::remove(moliere_path);

    std::filesystem::remove(temporary);
    std::cout << "GPU straight lepton transport validation passed: "
              << checks << " checks, " << first.records.size()
              << " records, " << first.fallback_events.size()
              << " fallback\n";
    return 0;
  } catch (std::exception const& error) {
    std::filesystem::remove(temporary);
    std::cerr << "GPU straight lepton transport validation failed after "
              << checks << " checks: " << error.what() << '\n';
    return 1;
  }
}
