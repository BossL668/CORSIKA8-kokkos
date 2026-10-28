/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "Check.hpp"
#include <corsika/setup/SetupStack.hpp>
#include "ProductionSetupStack.hpp"
#include "ProductionReservation.hpp"
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <tuple>
#include <type_traits>

using namespace corsika;
using EnvironmentInterface = IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>;
using EnvironmentType = Environment<EnvironmentInterface>;
using CandidateHybridStack = setup::HybridStack<EnvironmentType>;
using ReferenceStack = production_setup::HybridStack<EnvironmentType>;
using NativeStack = setup::Stack<EnvironmentType>;
template <class T, class = void>
struct HasHistoryReservation : std::false_type {};
template <class T>
struct HasHistoryReservation<
    T, std::void_t<decltype(std::declval<T&>().reserveTransportHistoryIds(
           std::uint64_t{1}))>> : std::true_type {};
static_assert(HasHistoryReservation<CandidateHybridStack>::value);
static_assert(!HasHistoryReservation<NativeStack>::value);
static_assert(std::is_same_v<decltype(std::declval<CandidateHybridStack&>()
                                          .reserveTransportHistoryIds(1)),
                             transport::HistoryId>);
static_assert(
    transport::HasTransportIdentity<CandidateHybridStack::particle_type>::value);
static_assert(!transport::HasTransportIdentity<NativeStack::particle_type>::value);
static_assert(std::is_same_v<setup::HybridStackView<EnvironmentType>,
                             CandidateHybridStack::stack_view_type>);
static_assert(
    std::is_same_v<setup::StackView<EnvironmentType>, NativeStack::stack_view_type>);
#ifdef WITH_HISTORY
static_assert(
    std::is_same_v<NativeStack,
                   setup::detail::StackGenerator<EnvironmentType>::StackWithHistory>);
#else
static_assert(
    std::is_same_v<NativeStack,
                   setup::detail::StackGenerator<EnvironmentType>::StackWithWeight>);
#endif

void reservation(CoordinateSystemPtr const& cs) {
  CandidateHybridStack stack;
  ProductionReservation reference;
  auto reserve = [&](std::uint64_t count) {
    auto actual = stack.reserveTransportHistoryIds(count);
    auto expected = reference.reserveTransportHistoryIds(count);
    CHECK(actual == expected);
    return actual;
  };
  CHECK(reserve(5) == 1);
  CHECK(stack.getEntries() == 0);
  auto p = stack.addParticle(std::make_tuple(Code::Electron, 1_GeV,
                                             DirectionVector(cs, {0., 0., 1.}),
                                             Point(cs, 0_m, 0_m, 0_m), 0_s));
  CHECK(p.getHistoryId() == reference.reserveTransportHistoryIds(1));
  CHECK(p.getHistoryId() == 6);
  CHECK(outcome([&] { stack.reserveTransportHistoryIds(0); }) ==
        outcome([&] { reference.reserveTransportHistoryIds(0); }));
  CHECK(reserve(17) == 7);
  typename CandidateHybridStack::stack_view_type view(p);
  auto child = view.addSecondary(std::make_tuple(Code::Photon, .1_GeV, p.getDirection()));
  CHECK(child.getHistoryId() == reference.reserveTransportHistoryIds(1));
  CHECK(child.getHistoryId() == 24);
  CHECK(child.getParentHistoryId() == 6);
  stack.clear();
  reference.data_.clear();
  auto maximum = std::numeric_limits<transport::HistoryId>::max();
  CHECK(reserve(maximum - 1) == 1);
  CHECK(outcome([&] { stack.reserveTransportHistoryIds(2); }) ==
        outcome([&] { reference.reserveTransportHistoryIds(2); }));
  CHECK(reserve(1) == maximum);
  CHECK(outcome([&] { stack.reserveTransportHistoryIds(1); }) ==
        outcome([&] { reference.reserveTransportHistoryIds(1); }));
  stack.clear();
  reference.data_.clear();
  CHECK(reserve(1) == 1);
}

template <class TStack>
std::string exercise(CoordinateSystemPtr const& cs, EnvironmentType::BaseNodeType* node) {
  TStack stack;
  std::ostringstream result;
  result << std::setprecision(17);
  auto add = [&] {
    return stack.addParticle(std::make_tuple(Code::Electron, 1_GeV,
                                             DirectionVector(cs, {0., 0., 1.}),
                                             Point(cs, 1_m, 2_m, 3_m), 4_ns));
  };
  auto p = add();
  p.setNode(node);
  p.setWeight(3.);
  CHECK(p.getHistoryId() == 1);
  CHECK(p.getParentHistoryId() == 0);
  CHECK(p.getGeneration() == 0);
  CHECK(p.beginTransportStep() == 0);
  CHECK(p.beginTransportStep() == 1);
#ifdef WITH_HISTORY
  CHECK(p.getEvent() == nullptr);
#endif
  typename TStack::stack_view_type view(p);
  for (unsigned i = 0; i < 7; ++i) {
    auto child =
        view.addSecondary(std::make_tuple(Code::Photon, .1_GeV, p.getDirection()));
    CHECK(child.getHistoryId() == i + 2);
    CHECK(child.getParentHistoryId() == 1);
    CHECK(child.getGeneration() == 1);
    CHECK(child.getStepId() == 0);
    CHECK(child.getNode() == node);
    CHECK(child.getWeight() == 3.);
#ifdef WITH_HISTORY
    CHECK(child.getEvent() != nullptr);
    CHECK(child.getParentEventIndex() == int(i));
    CHECK(child.getEvent()->parentEvent() == nullptr);
#endif
  }
  auto another = add();
  another.setNode(node);
  CHECK(another.getHistoryId() == 9);
  stack.swap(stack.begin(), stack.begin() + 8);
  CHECK(stack.begin().getHistoryId() == 9);
  CHECK((stack.begin() + 8).getHistoryId() == 1);
  stack.copy(stack.begin() + 8, stack.begin());
  CHECK(stack.begin().getStepId() == 2);
  CHECK(stack.begin().getWeight() == 3.);
  (stack.begin() + 7).erase();
  stack.swap(stack.begin() + 7, stack.begin() + 8);
  CHECK(stack.purgeLastIfDeleted());
  CHECK(stack.getEntries() == 8);
  for (auto it = stack.begin(); it != stack.end(); ++it) {
    auto coordinates = it.getPosition().getCoordinates(cs);
    result << it.getHistoryId() << ',' << it.getParentHistoryId() << ','
           << it.getGeneration() << ',' << it.getStepId() << ','
           << static_cast<int>(it.getPID()) << ',' << it.getEnergy() / 1_GeV << ','
           << it.getWeight() << ',' << it.getTime() / 1_s << ','
           << coordinates.getX() / 1_m << ',' << coordinates.getY() / 1_m << ','
           << coordinates.getZ() / 1_m << ',' << (it.getNode() == node);
#ifdef WITH_HISTORY
    result << ',' << bool(it.getEvent()) << ',' << it.getParentEventIndex();
#endif
    result << '\n';
  }
  stack.clear();
  CHECK(add().getHistoryId() == 1);
  return result.str();
}
int main() {
  try {
    EnvironmentType env;
    auto const& cs = env.getCoordinateSystem();
    auto node = EnvironmentType::createNode<Sphere>(Point(cs, 0_m, 0_m, 0_m), 100_km);
    using Model = MediumPropertyModel<
        UniformMagneticField<HomogeneousMedium<EnvironmentInterface>>>;
    node->setModelProperties<Model>(Medium::AirDry1Atm, Vector(cs, 0_T, 0_T, 0_T),
                                    1_kg / (1_m * 1_m * 1_m),
                                    NuclearComposition({Code::Oxygen}, {1.}));
    auto* nodePointer = node.get();
    env.getUniverse()->addChild(std::move(node));
    CHECK(exercise<CandidateHybridStack>(cs, nodePointer) ==
          exercise<ReferenceStack>(cs, nodePointer));
    reservation(cs);
    NativeStack native;
    auto p = native.addParticle(std::make_tuple(Code::Electron, 1_GeV,
                                                DirectionVector(cs, {0., 0., 1.}),
                                                Point(cs, 0_m, 0_m, 0_m), 0_s));
    p.setNode(nodePointer);
    p.setWeight(2.);
    CHECK(p.getWeight() == 2.);
    CHECK(p.getNode() == nodePointer);
#ifdef WITH_HISTORY
    CHECK(p.getEvent() == nullptr);
    std::cout << "STACK_MODE=history\n";
#else
    std::cout << "STACK_MODE=plain\n";
#endif
    std::cout << "hybrid-stack CHECKS=" << checks << '\n';
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
