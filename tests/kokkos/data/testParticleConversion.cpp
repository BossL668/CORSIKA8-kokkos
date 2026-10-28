/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/data/ParticleConversion.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/stack/CombinedStack.hpp>
#include <corsika/stack/VectorStack.hpp>
#include <corsika/stack/WeightStackExtension.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace {

  using namespace corsika;
  using Catch::Approx;
  using state_type = kokkos::ParticleState;

  template <typename TStackIterator>
  using weighted_interface_type =
      CombinedParticleInterface<VectorStack::pi_type, weights::WeightDataInterface,
                                TStackIterator>;
  using weighted_stack_type = CombinedStack<VectorStack::stack_data_type,
                                            weights::WeightData, weighted_interface_type>;

  state_type sentinel_state() {
    state_type state{};
    state.mediumId_ = 17;
    state.environmentNode_ = 29;
    state.generation_ = 31;
    state.reserved_ = 0x80000001u;
    state.historyId_ = (std::uint64_t{1} << 61) + 7;
    state.parentHistoryId_ = (std::uint64_t{1} << 60) + 3;
    state.stepId_ = (std::uint64_t{1} << 59) + 11;
    return state;
  }

  void require_metadata(state_type const& actual, state_type const& expected) {
    REQUIRE(actual.mediumId_ == expected.mediumId_);
    REQUIRE(actual.environmentNode_ == expected.environmentNode_);
    REQUIRE(actual.generation_ == expected.generation_);
    REQUIRE(actual.reserved_ == expected.reserved_);
    REQUIRE(actual.historyId_ == expected.historyId_);
    REQUIRE(actual.parentHistoryId_ == expected.parentHistoryId_);
    REQUIRE(actual.stepId_ == expected.stepId_);
  }

  void require_values(state_type const& actual, state_type const& expected) {
    REQUIRE(actual.pid_ == expected.pid_);
    REQUIRE(actual.energyGeV_ == expected.energyGeV_);
    REQUIRE(actual.timeS_ == expected.timeS_);
    REQUIRE(actual.weight_ == expected.weight_);
    for (int axis = 0; axis < 3; ++axis) {
      REQUIRE(actual.positionM_[axis] == expected.positionM_[axis]);
      REQUIRE(actual.direction_[axis] == expected.direction_[axis]);
    }
  }

  // The fixed Cp8 value expressions, independently evaluated through C8's API.
  // Source: 7c24233b, em/common/RouterParticleConversion.hpp::toDeviceState.
  // Identity and node conversion are deliberately outside this value-only test.
  template <typename TParticle>
  state_type reference_values(TParticle const& particle,
                              CoordinateSystemPtr const& coordinateSystem,
                              double const weight) {
    state_type state{};
    state.pid_ = static_cast<std::int32_t>(particle.getPDG());
    state.energyGeV_ = particle.getEnergy() / 1_GeV;
    auto const position = particle.getPosition().getCoordinates(coordinateSystem);
    auto const direction = particle.getDirection().getComponents(coordinateSystem);
    for (int axis = 0; axis < 3; ++axis) {
      state.positionM_[axis] = position[axis] / meter;
      state.direction_[axis] = direction[axis].magnitude();
    }
    state.timeS_ = particle.getTime() / second;
    state.weight_ = weight;
    return state;
  }

  auto initial_data(Code const code, CoordinateSystemPtr const& coordinateSystem,
                    HEPEnergyType const kineticEnergy = 5_GeV) {
    return std::make_tuple(code, kineticEnergy,
                           DirectionVector(coordinateSystem, {0.3, -0.4, 0.5}),
                           Point(coordinateSystem, 123_cm, -250_cm, 30_cm), 17_ns);
  }

} // namespace

TEST_CASE("Particle values preserve units, PDG codes and metadata",
          "[kokkos_conversion]") {
  auto const root = get_root_CoordinateSystem();
  std::array<Code, 8> const codes{
      Code::Electron, Code::Positron, Code::Photon,  Code::MuMinus,
      Code::MuPlus,   Code::Proton,   Code::Neutron, get_nucleus_code(56, 26)};
  for (auto const code : codes) {
    CAPTURE(static_cast<int>(code));
    VectorStack stack;
    auto const source = stack.addParticle(initial_data(code, root, 125_MeV));
    auto state = sentinel_state();
    auto const metadata = state;
    kokkos::export_particle_values(source, root, state);
    require_metadata(state, metadata);
    require_values(state, reference_values(source, root, 1.));
    REQUIRE(state.pid_ == static_cast<std::int32_t>(get_PDG(code)));
    REQUIRE(state.energyGeV_ == Approx(0.125 + get_mass(code) / 1_GeV));
    REQUIRE(state.positionM_[0] == Approx(1.23));
    REQUIRE(state.positionM_[1] == Approx(-2.5));
    REQUIRE(state.timeS_ == Approx(17.e-9));
    REQUIRE(state.weight_ == 1.);

    VectorStack targetStack;
    auto target = targetStack.addParticle(initial_data(Code::Photon, root));
    kokkos::import_particle_values(state, root, target);
    REQUIRE(targetStack.getEntries() == 1);
    REQUIRE(target.getPID() == code);
    REQUIRE(target.getKineticEnergy() / 1_GeV ==
            (state.energyGeV_ * 1_GeV - get_mass(code)) / 1_GeV);
    REQUIRE(target.getKineticEnergy() / 1_GeV == Approx(0.125).margin(2e-14));
    REQUIRE(target.getDirection().getNorm().magnitude() == Approx(std::sqrt(0.5)));
    REQUIRE(target.getTime() / 1_ns == Approx(17.));
    require_metadata(state, metadata);
  }
}

TEST_CASE("Particle coordinate export does not rebase its source",
          "[kokkos_conversion]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {1_m, -2_m, 3_m});
  auto const rotated = make_rotation(root, QuantityVector<dimensionless_d>{0., 0., 1.},
                                     std::acos(-1.) / 2.);
  auto const nested = make_translation(rotated, {4_m, 5_m, -6_m});
  std::vector<CoordinateSystemPtr> const frames{root, translated, rotated, nested};
  VectorStack stack;
  auto const source = stack.addParticle(initial_data(Code::MuPlus, root));
  auto const original = reference_values(source, root, 1.);
  for (auto const& frame : frames) {
    auto state = sentinel_state();
    kokkos::export_particle_values(source, frame, state);
    require_values(state, reference_values(source, frame, 1.));
    REQUIRE(source.getPosition().getCoordinateSystem() == root);
    REQUIRE(source.getDirection().getCoordinateSystem() == root);
    require_values(reference_values(source, root, 1.), original);
    VectorStack targetStack;
    auto target = targetStack.addParticle(initial_data(Code::Photon, root));
    kokkos::import_particle_values(state, frame, target);
    REQUIRE(target.getPosition().getCoordinateSystem() == frame);
    REQUIRE(target.getDirection().getCoordinateSystem() == frame);
    REQUIRE(distance(target.getPosition(), source.getPosition()) / 1_m < 3e-14);
    REQUIRE((target.getDirection() - source.getDirection()).getNorm().magnitude() <
            3e-14);
  }
  state_type translatedState{};
  kokkos::export_particle_values(source, translated, translatedState);
  REQUIRE(translatedState.positionM_[0] == Approx(0.23));
  REQUIRE(translatedState.positionM_[1] == Approx(-0.5));
  REQUIRE(translatedState.positionM_[2] == Approx(-2.7));
  REQUIRE(translatedState.direction_[0] == 0.3);
  REQUIRE(translatedState.direction_[1] == -0.4);
  REQUIRE(translatedState.direction_[2] == 0.5);
}

TEST_CASE("Particle values preserve supported weights and reject weight loss",
          "[kokkos_conversion]") {
  auto const root = get_root_CoordinateSystem();
  weighted_stack_type stack;
  auto source = stack.addParticle(initial_data(Code::Electron, root));
  source.setWeight(7.25);
  auto state = sentinel_state();
  kokkos::export_particle_values(source, root, state);
  REQUIRE(state.weight_ == 7.25);
  auto target = stack.addParticle(initial_data(Code::Photon, root));
  kokkos::import_particle_values(state, root, target);
  REQUIRE(target.getWeight() == 7.25);
  REQUIRE(target.getPID() == Code::Electron);
  REQUIRE(stack.getEntries() == 2);

  VectorStack plainStack;
  auto plain = plainStack.addParticle(initial_data(Code::MuPlus, root));
  auto const before = reference_values(plain, root, 1.);
  REQUIRE_THROWS_AS(kokkos::import_particle_values(state, root, plain),
                    std::runtime_error);
  require_values(reference_values(plain, root, 1.), before);
  REQUIRE(plainStack.getEntries() == 1);
}

TEST_CASE("Particle value import keeps the exact C8 mass boundary",
          "[kokkos_conversion]") {
  auto const root = get_root_CoordinateSystem();
  VectorStack stack;
  auto target = stack.addParticle(initial_data(Code::Proton, root));
  for (auto const code : {Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus}) {
    auto state = sentinel_state();
    state.pid_ = static_cast<std::int32_t>(get_PDG(code));
    double const massGeV = get_mass(code) / 1_GeV;
    for (double const energy :
         {std::nextafter(massGeV, 0.), massGeV,
          std::nextafter(massGeV, std::numeric_limits<double>::infinity()),
          massGeV + 1e-12}) {
      state.energyGeV_ = energy;
      auto const before = reference_values(target, root, 1.);
      bool const belowMass = energy * 1_GeV < get_mass(code);
      if (belowMass) {
        REQUIRE_THROWS_AS(kokkos::import_particle_values(state, root, target),
                          std::runtime_error);
        require_values(reference_values(target, root, 1.), before);
      } else {
        kokkos::import_particle_values(state, root, target);
        REQUIRE(target.getKineticEnergy() == energy * 1_GeV - get_mass(code));
      }
    }
  }
  auto state = sentinel_state();
  state.pid_ = 22;
  state.energyGeV_ = 0.;
  REQUIRE_NOTHROW(kokkos::import_particle_values(state, root, target));
  REQUIRE(target.getKineticEnergy() == 0_GeV);
  state.energyGeV_ = -1.;
  REQUIRE_THROWS_AS(kokkos::import_particle_values(state, root, target),
                    std::runtime_error);
  REQUIRE(target.getKineticEnergy() == 0_GeV);
}

TEST_CASE("Invalid particle conversion arguments do not change values",
          "[kokkos_conversion]") {
  auto const root = get_root_CoordinateSystem();
  VectorStack stack;
  auto particle = stack.addParticle(initial_data(Code::MuMinus, root));
  auto state = sentinel_state();
  auto const beforeState = state;
  REQUIRE_THROWS_AS(kokkos::export_particle_values(particle, {}, state),
                    std::invalid_argument);
  require_values(state, beforeState);
  require_metadata(state, beforeState);
  auto const before = reference_values(particle, root, 1.);
  REQUIRE_THROWS_AS(kokkos::import_particle_values(state, {}, particle),
                    std::invalid_argument);
  require_values(reference_values(particle, root, 1.), before);
  state.energyGeV_ = 1.;
  for (std::int32_t const pid :
       {std::int32_t{0}, std::numeric_limits<std::int32_t>::min()}) {
    state.pid_ = pid;
    REQUIRE_THROWS_AS(kokkos::import_particle_values(state, root, particle),
                      std::invalid_argument);
    require_values(reference_values(particle, root, 1.), before);
  }
}
