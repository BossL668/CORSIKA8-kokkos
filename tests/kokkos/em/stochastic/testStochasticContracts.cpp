/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "StochasticChecks.hpp"

TEST_CASE("Production random identities and fallback routing survive migration",
          "[stochastic]") {
  using namespace stochastic_test;
  current::Runtime runtime;
  REQUIRE(current::RandomDomainVersion == reference::RandomDomainVersion);
  REQUIRE(current::random_process_domains_are_unique());
  REQUIRE(current::NativeSelectionReplayEndpointWidth == 0.);
  REQUIRE(current::MuonMeanLifetimeS == reference::MuonMeanLifetimeS);
  REQUIRE(current::MuonDecaySpeedOfLightMPerS == reference::MuonDecaySpeedOfLightMPerS);
  std::uint32_t const domains[]{current::InteractionDistanceRandomProcessId,
                                current::InteractionColumnRandomProcessId,
                                current::ProposalSelectionRandomProcessId,
                                current::ContinuousScatteringRandomProcessId,
                                current::MuonDecayRandomProcessId};
  std::uint32_t const oldDomains[]{reference::InteractionDistanceRandomProcessId,
                                   reference::InteractionColumnRandomProcessId,
                                   reference::ProposalSelectionRandomProcessId,
                                   reference::ContinuousScatteringRandomProcessId,
                                   reference::MuonDecayRandomProcessId};
  for (unsigned index = 0; index < 5; ++index)
    REQUIRE(domains[index] == oldDomains[index]);
  std::uint64_t const draws[]{
      current::InteractionDistanceDrawId, current::InteractionColumnDrawId,
      current::ProposalSelectionDrawId,   current::InteractionLossDrawId,
      current::MoliereFirstAngleDrawId,   current::MoliereSecondAngleDrawId,
      current::MoliereAzimuthDrawId,      current::MuonDecayDrawId};
  std::uint64_t const oldDraws[]{
      reference::InteractionDistanceDrawId, reference::InteractionColumnDrawId,
      reference::ProposalSelectionDrawId,   reference::InteractionLossDrawId,
      reference::MoliereFirstAngleDrawId,   reference::MoliereSecondAngleDrawId,
      reference::MoliereAzimuthDrawId,      reference::MuonDecayDrawId};
  for (unsigned index = 0; index < 8; ++index) REQUIRE(draws[index] == oldDraws[index]);
  auto const zero = current::philox4x32_10({}, {});
  std::uint32_t const golden[]{0x6627e8d5U, 0xe169c58dU, 0xbc57ac4cU, 0x9b00dbd8U};
  for (unsigned lane = 0; lane < 4; ++lane) REQUIRE(zero.words_[lane] == golden[lane]);
  std::vector<current::RandomNumberKey> keys(1);
  // Exercise every bit of every key field, not only small shower/history IDs.
  for (unsigned field = 0; field < 6; ++field)
    for (unsigned bit = 0; bit < (field == 4 ? 32u : 64u); ++bit) {
      current::RandomNumberKey key;
      auto value = std::uint64_t{1} << bit;
      switch (field) {
        case 0:
          key.seed_ = value;
          break;
        case 1:
          key.showerId_ = value;
          break;
        case 2:
          key.historyId_ = value;
          break;
        case 3:
          key.stepId_ = value;
          break;
        case 4:
          key.processId_ = static_cast<std::uint32_t>(value);
          break;
        case 5:
          key.drawId_ = value;
          break;
      }
      keys.push_back(key);
    }
  keys.push_back({~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0U, ~0ULL});
  for (std::uint64_t index = 0; index < 2048; ++index)
    for (auto domain : domains)
      keys.push_back({0xabcdef0000000000ULL + index, ~index,
                      index * 0x9e3779b97f4a7c15ULL, 0xffffffffffffffffULL - index,
                      domain, index % 8});
  auto deviceKeys = upload(keys);
  Kokkos::View<Outcome * [2], Kokkos::LayoutRight, memory_space_type> randomOutput(
      "random_results", keys.size());
  Kokkos::parallel_for(
      "production_rng", Kokkos::RangePolicy<execution_space_type>(0, keys.size()),
      KOKKOS_LAMBDA(std::size_t index) {
        // Reverse traversal ensures no dependence on launch/queue order.
        auto const reversed = deviceKeys.extent(0) - 1 - index;
        randomOutput(reversed, 0) = random_result(deviceKeys(reversed));
        randomOutput(reversed, 1) = random_reference(deviceKeys(reversed));
      });
  auto hostRandom =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, randomOutput);
  for (std::size_t index = 0; index < keys.size(); ++index) {
    INFO("RNG key " << index);
    check(random_result(keys[index]), random_reference(keys[index]));
    check(hostRandom(index, 0), hostRandom(index, 1));
    check(hostRandom(index, 0), random_result(keys[index]));
    for (unsigned lane = 0; lane < 9; ++lane) {
      REQUIRE(hostRandom(index, 0).values_[lane] > 0.);
      REQUIRE(hostRandom(index, 0).values_[lane] < 1.);
    }
  }
  std::vector<PolicyQuery> policies;
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  auto const inf = std::numeric_limits<double>::infinity();
  std::vector<std::int32_t> processes{-1, 0, 2000000013, 2000000015};
  for (int process = 1000000002; process <= 1000000016; ++process)
    processes.push_back(process);
  for (auto pid : {11, -11, 13, -13, 22, 2112, 2212, 0})
    for (auto process : processes)
      for (auto quantile : {-inf, -1., std::nextafter(0., -1.), -0., 0., .5, 1.,
                            std::nextafter(1., 2.), inf, nan})
        policies.push_back({pid, process, 28, 6, quantile, .01});
  for (int reason = -1; reason <= 30; ++reason)
    for (int status = -1; status <= 14; ++status)
      for (double time :
           {std::nextafter(.01, 0.), .01, std::nextafter(.01, 1.), nan, inf, -inf})
        policies.push_back({11, 1000000004, reason, status, .42, time});
  auto devicePolicies = upload(policies);
  Kokkos::View<Outcome * [2], Kokkos::LayoutRight, memory_space_type> policyOutput(
      "policy_results", policies.size());
  Kokkos::parallel_for(
      "production_policies",
      Kokkos::RangePolicy<execution_space_type>(0, policies.size()),
      KOKKOS_LAMBDA(std::size_t index) {
        policyOutput(index, 0) = policy(devicePolicies(index));
        policyOutput(index, 1) = policy_reference(devicePolicies(index));
      });
  auto hostPolicy =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, policyOutput);
  for (std::size_t index = 0; index < policies.size(); ++index) {
    INFO("Policy case " << index);
    check(policy(policies[index]), policy_reference(policies[index]));
    check(hostPolicy(index, 0), hostPolicy(index, 1));
  }
  for (auto pid : {11, -11, 13, -13})
    REQUIRE(current::process_capability(pid, current::ElectronPairProcessId) ==
            current::ProcessCapability::CpuOnlyFinalState);
  REQUIRE(current::process_capability(22, current::PhotonPairProcessId) ==
          current::ProcessCapability::PhotonPair);
  REQUIRE(current::process_capability(22, current::PhotonMuonPairProcessId) ==
          current::ProcessCapability::PhotonMuonPair);
  REQUIRE_FALSE(current::proposal_native_selection_requires_replay(
      22, current::PhotonPairProcessId, 0.));
  REQUIRE_FALSE(current::proposal_native_selection_requires_replay(
      22, current::PhotonPairProcessId, 1.));
  REQUIRE_FALSE(current::proposal_native_selection_requires_replay(22, -1, -1.));
  for (auto process : processes)
    REQUIRE(std::strcmp(current::em_process_name(process),
                        reference::gpuEmProcessName(process)) == 0);
  for (int reason = -1; reason <= 30; ++reason)
    REQUIRE(std::strcmp(current::proposal_fallback_reason_name(
                            static_cast<current::FallbackReason>(reason)),
                        reference::proposalFallbackReasonName(
                            static_cast<reference::ProposalFallbackReason>(reason))) ==
            0);
  std::ofstream report("stochastic-counts.txt");
  report << "random_keys " << keys.size() << "\npolicy_cases " << policies.size() << "\n";
}
