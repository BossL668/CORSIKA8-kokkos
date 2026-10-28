/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "ContinuousDataset.hpp"
#include <corsika/kokkos/em/proposal/NativeTableExporter.hpp>
#include <corsika/kokkos/em/tables/ProposalNativeQueries.hpp>
#include <corsika/modules/proposal/ContinuousProcess.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <PROPOSAL/propagation_utility/InteractionBuilder.h>
#include <PROPOSAL/propagation_utility/DisplacementBuilder.h>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <map>
#include <stdexcept>

namespace continuous_test {
  namespace {
    using namespace corsika;
    using model_type = IMediumPropertyModel<IMediumModel>;
    using environment_type = Environment<model_type>;
    using medium_type = MediumPropertyModel<HomogeneousMedium<model_type>>;

    class Settings {
    public:
      explicit Settings(std::string const& directory)
          : path_(PROPOSAL::InterpolationSettings::TABLES_PATH)
          , upper_(PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM)
          , nodes_{PROPOSAL::InterpolationSettings::NODES_DNDX_E,
                   PROPOSAL::InterpolationSettings::NODES_DNDX_V,
                   PROPOSAL::InterpolationSettings::NODES_DEDX,
                   PROPOSAL::InterpolationSettings::NODES_UTILITY,
                   PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT} {
        auto const* previous = std::getenv("CORSIKA_DATA");
        existed_ = previous != nullptr;
        if (previous) data_ = previous;
        std::filesystem::create_directories(std::filesystem::path(directory) /
                                            "PROPOSAL");
        if (setenv("CORSIKA_DATA", directory.c_str(), 1) != 0)
          throw std::runtime_error("Cannot isolate C8 data directory");
        PROPOSAL::InterpolationSettings::TABLES_PATH = directory + "/PROPOSAL";
        PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM = 1.e8;
        PROPOSAL::InterpolationSettings::NODES_DNDX_E = 32;
        PROPOSAL::InterpolationSettings::NODES_DNDX_V = 32;
        PROPOSAL::InterpolationSettings::NODES_DEDX = 64;
        PROPOSAL::InterpolationSettings::NODES_UTILITY = 64;
        PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT = 64;
        for (auto code : proposal::tracked) {
          production_.push_back(get_energy_production_threshold(code));
          propagation_.push_back(get_kinetic_energy_propagation_threshold(code));
          set_energy_production_threshold(code, .4_MeV);
          set_kinetic_energy_propagation_threshold(code, .4_MeV);
        }
      }
      ~Settings() {
        PROPOSAL::InterpolationSettings::TABLES_PATH = path_;
        PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM = upper_;
        PROPOSAL::InterpolationSettings::NODES_DNDX_E = nodes_[0];
        PROPOSAL::InterpolationSettings::NODES_DNDX_V = nodes_[1];
        PROPOSAL::InterpolationSettings::NODES_DEDX = nodes_[2];
        PROPOSAL::InterpolationSettings::NODES_UTILITY = nodes_[3];
        PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT = nodes_[4];
        for (std::size_t index = 0; index < proposal::tracked.size(); ++index) {
          set_energy_production_threshold(proposal::tracked[index], production_[index]);
          set_kinetic_energy_propagation_threshold(proposal::tracked[index],
                                                   propagation_[index]);
        }
        if (existed_)
          setenv("CORSIKA_DATA", data_.c_str(), 1);
        else
          unsetenv("CORSIKA_DATA");
      }

    private:
      std::string path_, data_;
      bool existed_{};
      double upper_;
      std::array<unsigned, 5> nodes_;
      std::vector<HEPEnergyType> production_, propagation_;
    };
    // A minimal particle interface, not a replacement continuous-loss algorithm.
    // The actual C8 module executes its own getMaxStepLength on this input.
    struct Particle {
      Code code_;
      HEPEnergyType energy_;
      environment_type::BaseNodeType const* node_;
      Code getPID() const { return code_; }
      HEPEnergyType getEnergy() const { return energy_; }
      auto getNode() const { return node_; }
    };
  } // namespace

  std::string cache_directory() {
    auto const* directory = std::getenv("CORSIKA_CONTINUOUS_TEST_CACHE");
    if (!directory || !*directory)
      throw std::runtime_error("Set a dedicated CORSIKA_CONTINUOUS_TEST_CACHE");
    return directory;
  }

  Dataset make_dataset() {
    using namespace corsika;
    Settings settings(cache_directory());
    environment_type environment;
    auto const frame = environment.getCoordinateSystem();
    auto world =
        environment_type::createNode<Sphere>(Point(frame, 0_m, 0_m, 0_m), 100_km);
    world->setModelProperties<medium_type>(
        Medium::AirDry1Atm, 1.225_kg / cube(1_m),
        NuclearComposition({Code::Nitrogen, Code::Oxygen}, {.78, .22}));
    auto const* node = world.get();
    environment.getUniverse()->addChild(std::move(world));
    RNGManager<>::getInstance().registerRandomStream("proposal");
    proposal::ContinuousProcess<> process(environment);
    auto const all = process.nativeCalculatorViews();
    std::vector<proposal::NativeContinuousCalculatorView> continuous;
    std::vector<std::unique_ptr<PROPOSAL::Interaction>> owners;
    std::map<Code, std::unique_ptr<PROPOSAL::Displacement>> diagnostics;
    std::vector<proposal::NativeInteractionCalculatorView> interactions;
    for (auto const& view : all) {
      auto const code = view.getProjectile();
      if (code != Code::Electron && code != Code::Positron && code != Code::MuMinus &&
          code != Code::MuPlus)
        continue;
      continuous.push_back(view);
      auto medium = view.getMedium()->get();
      auto crosses = proposal::cross.at(code)(medium, .4_MeV);
      // These ordinary methods are non-const in PROPOSAL. Keep an independent
      // owner instead of casting away the read-only native module view.
      diagnostics.emplace(code, PROPOSAL::make_displacement(crosses, true));
      owners.push_back(PROPOSAL::make_interaction(crosses, true, true));
      interactions.emplace_back(code, view.getMediumHash(),
                                std::cref(view.getMedium()->get()),
                                std::cref(*owners.back()), std::nullopt, std::nullopt,
                                .4_MeV, view.getParticleMassMeV());
    }
    if (continuous.size() != 4)
      throw std::logic_error("Expected all four charged leptons");
    Dataset result;
    result.table_ = kokkos::export_proposal_native_tables(interactions, continuous);
    StraightTrajectory const track(
        Line(Point(frame, 0_m, 0_m, 0_m),
             VelocityVector(frame, {1_m / 1_s, 0_m / 1_s, 0_m / 1_s})),
        1_s);
    for (auto const& view : continuous) {
      auto const code = view.getProjectile();
      auto const pid = static_cast<std::int32_t>(get_PDG(code));
      auto& displacement = *diagnostics.at(code);
      for (double const energy : {1.e3, 3.e3, 1.e4, 1.e5, 1.e6, 1.e7}) {
        Query base;
        base.interaction_.particle_.pid_ = pid;
        base.interaction_.particle_.energyGeV_ = energy / 1000.;
        auto const effective = base.interaction_.particle_.energyGeV_ * 1000.;
        for (auto op : {Operation::Mass, Operation::TransportMass, Operation::Cut,
                        Operation::Minimum, Operation::Range}) {
          auto query = base;
          query.operation_ = op;
          result.queries_.push_back(query);
        }
        auto dedx = base;
        dedx.operation_ = Operation::Dedx;
        dedx.nativeKind_ = 1;
        auto const integrand = displacement.FunctionToIntegral(effective);
        dedx.nativeExpected_ = integrand != 0. ? -1. / integrand : 0.;
        result.queries_.push_back(dedx);
        auto prepare = base;
        prepare.operation_ = Operation::Prepare;
        prepare.nativeKind_ = 3;
        Particle const particle{code, effective * 1_MeV, node};
        prepare.nativeExpected_ =
            process.getMaxStepLength(particle, track) / 1_m * .001225 * 100.;
        result.queries_.push_back(prepare);
        auto const maxX = displacement.SolveTrackIntegral(effective, .9 * effective);
        for (double fraction : {0., .001, .1, .5, 1.}) {
          auto query = base;
          query.operation_ = Operation::AfterLoss;
          query.argument_ = maxX * fraction;
          query.nativeKind_ = 2;
          query.nativeExpected_ =
              displacement.UpperLimitTrackIntegral(effective, query.argument_);
          result.queries_.push_back(query);
          query.nativeKind_ = 0;
          query.operation_ = Operation::Loss;
          result.queries_.push_back(query);
        }
        // Exercise both outcomes of the continuous-limit guard inside the
        // shared host/device query path, not only in host contract assertions.
        for (double fraction : {0., .5}) {
          auto query = base;
          query.operation_ = Operation::Loss;
          query.argument_ = maxX * fraction;
          query.continuousLimitWon_ = 1;
          result.queries_.push_back(query);
        }
        auto inverse = base;
        inverse.operation_ = Operation::Energy;
        auto const native = kokkos::make_proposal_native_host_view(result.table_);
        inverse.argument_ =
            kokkos::query_proposal_native_range(native, pid, effective).value_;
        result.queries_.push_back(inverse);
        for (double length : {.1, 1., 10.}) {
          auto chain = base;
          chain.operation_ = Operation::Chain;
          chain.argument_ = length;
          result.queries_.push_back(chain);
        }
      }
    }
    return result;
  }
} // namespace continuous_test
