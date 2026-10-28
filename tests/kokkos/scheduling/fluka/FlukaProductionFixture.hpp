/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/scheduling/FlukaWorker.hpp>
#include <corsika/kokkos/scheduling/HadronicProcessPool.hpp>
#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <array>
namespace corsika::kokkos::fluka_worker_test {
  inline HadronicInteractionRequest make_self_test_request(std::uint64_t const sequenceId,
                                                           Code const projectile,
                                                           Code const target,
                                                           double const momentumGeV) {
    HadronicInteractionRequest request;
    request.sequenceId_ = sequenceId;
    request.randomKey_ = HadronicRandomKey{0xC08A2026ULL, 7, 1000 + sequenceId, 3, 1, 0};
    request.projectilePdg_ = static_cast<std::int32_t>(get_PDG(projectile));
    request.targetPdg_ = static_cast<std::int32_t>(get_PDG(target));
    auto const momentum = momentumGeV * 1_GeV;
    request.projectileFourMomentumGeV_ = {
        calculate_total_energy(momentum, get_mass(projectile)) / 1_GeV, 0., 0.,
        momentumGeV};
    request.targetFourMomentumGeV_ = {get_mass(target) / 1_GeV, 0., 0., 0.};
    return request;
  }
  inline bool equal_self_test_result(FlukaWorkerResult const& lhs,
                                     FlukaWorkerResult const& rhs) {
    if (lhs.header_.protocolVersion_ != rhs.header_.protocolVersion_ ||
        lhs.header_.status_ != rhs.header_.status_ ||
        lhs.header_.sequenceId_ != rhs.header_.sequenceId_ ||
        lhs.header_.secondaryCount_ != rhs.header_.secondaryCount_ ||
        lhs.header_.randomValuesSupplied_ != rhs.header_.randomValuesSupplied_ ||
        lhs.secondaries_.size() != rhs.secondaries_.size()) {
      return false;
    }
    if (!std::isfinite(lhs.header_.finalStateTimeMs_) ||
        !std::isfinite(rhs.header_.finalStateTimeMs_) ||
        lhs.header_.finalStateTimeMs_ < 0. || rhs.header_.finalStateTimeMs_ < 0.) {
      return false;
    }
    for (std::size_t i = 0; i < lhs.secondaries_.size(); ++i) {
      auto const& a = lhs.secondaries_[i];
      auto const& b = rhs.secondaries_[i];
      if (a.pdg_ != b.pdg_ || a.kineticEnergyGeV_ != b.kineticEnergyGeV_ ||
          a.direction_ != b.direction_) {
        return false;
      }
    }
    return true;
  }
  inline bool equal_self_test_response(HadronicInteractionResponse const& lhs,
                                       HadronicInteractionResponse const& rhs) {
    FlukaWorkerResult left{lhs.header_, lhs.secondaries_};
    FlukaWorkerResult right{rhs.header_, rhs.secondaries_};
    return equal_self_test_result(left, right);
  }
  inline auto make_self_test_requests(std::size_t const count) {
    std::array<Code, 4> const selfTestProjectiles{Code::Proton, Code::PiPlus,
                                                  Code::PiMinus, Code::KPlus};
    std::array<Code, 3> const selfTestTargets{Code::Nitrogen, Code::Oxygen, Code::Argon};

    std::vector<HadronicInteractionRequest> requests;
    requests.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      requests.push_back(make_self_test_request(
          static_cast<std::uint64_t>(i),
          selfTestProjectiles[i % selfTestProjectiles.size()],
          selfTestTargets[i % selfTestTargets.size()], 1. + static_cast<double>(i % 80)));
    }
    return requests;
  }
  inline auto make_self_test_batches(
      std::vector<HadronicInteractionRequest> const& requests,
      std::size_t const poolWorkers) {
    auto const batchCount =
        std::min<std::size_t>(std::max<std::size_t>(poolWorkers * 4, 1),
                              std::max<std::size_t>(requests.size(), 1));
    std::vector<HadronicRequestBatch> batches(batchCount);
    for (std::size_t i = 0; i < batchCount; ++i) {
      batches[i].batchId_ = 0xC08A000000000000ULL + i;
    }
    for (std::size_t i = 0; i < requests.size(); ++i) {
      batches[i % batchCount].requests_.push_back(requests[i]);
    }
    batches.erase(
        std::remove_if(batches.begin(), batches.end(),
                       [](auto const& batch) { return batch.requests_.empty(); }),
        batches.end());

    return batches;
  }
} // namespace corsika::kokkos::fluka_worker_test
