/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/Logging.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <FLUKA.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#if defined(__unix__) || defined(__linux__)
#include <cerrno>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace corsika::kokkos {
  namespace fluka_worker_detail {
#if defined(__unix__) || defined(__linux__)
    inline bool read_all_from_descriptor(int const descriptor, void* destination,
                                         std::size_t const bytes,
                                         bool const allowCleanEof) {
      auto* cursor = static_cast<unsigned char*>(destination);
      std::size_t remaining = bytes;
      while (remaining > 0) {
        auto const received = ::recv(descriptor, cursor, remaining, 0);
        if (received < 0 && errno == EINTR) { continue; }
        if (received == 0 && allowCleanEof && remaining == bytes) { return false; }
        if (received <= 0) {
          throw std::runtime_error(
              "Could not read complete record from persistent worker socket");
        }
        cursor += static_cast<std::size_t>(received);
        remaining -= static_cast<std::size_t>(received);
      }
      return true;
    }
    inline void write_all_to_descriptor(int const descriptor, void const* source,
                                        std::size_t const bytes) {
      auto const* cursor = static_cast<unsigned char const*>(source);
      std::size_t remaining = bytes;
      while (remaining > 0) {
        auto const sent = ::send(descriptor, cursor, remaining, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) { continue; }
        if (sent <= 0) {
          throw std::runtime_error(
              "Could not write complete record to persistent worker socket");
        }
        cursor += static_cast<std::size_t>(sent);
        remaining -= static_cast<std::size_t>(sent);
      }
    }
    template <typename TValue>
    inline bool read_descriptor_record(int const descriptor, TValue& value,
                                       bool const allowCleanEof = false) {
      static_assert(std::is_trivially_copyable_v<TValue>);
      return read_all_from_descriptor(descriptor, &value, sizeof(value), allowCleanEof);
    }
    template <typename TValue>
    inline void write_descriptor_record(int const descriptor, TValue const& value) {
      static_assert(std::is_trivially_copyable_v<TValue>);
      write_all_to_descriptor(descriptor, &value, sizeof(value));
    }
#endif
    inline FourMomentum decode_four_momentum(
        std::array<double, 4> const& values,
        CoordinateSystemPtr const& coordinateSystem) {
      return FourMomentum{
          values[0] * 1_GeV,
          MomentumVector{coordinateSystem,
                         {values[1] * 1_GeV, values[2] * 1_GeV, values[3] * 1_GeV}}};
    }
  } // namespace fluka_worker_detail
  inline FlukaWorkerResult execute_fluka_request(
      corsika::fluka::InteractionModel& model, HadronicInteractionRequest const& request,
      CoordinateSystemPtr const& coordinateSystem) {
    FlukaWorkerResult result;
    result.header_.sequenceId_ = request.sequenceId_;

    try {
      validate_hadronic_interaction_request(request);
      if (request.model_ != HadronicWorkerModel::Fluka) {
        result.header_.status_ = HadronicWorkerStatus::InvalidRequest;
        return result;
      }

      auto const projectile =
          convert_from_PDG(static_cast<PDGCode>(request.projectilePdg_));
      auto const target = convert_from_PDG(static_cast<PDGCode>(request.targetPdg_));
      auto const projectileP4 = fluka_worker_detail::decode_four_momentum(
          request.projectileFourMomentumGeV_, coordinateSystem);
      auto const targetP4 = fluka_worker_detail::decode_four_momentum(
          request.targetFourMomentumGeV_, coordinateSystem);
      if (!model.isValid(projectile, target, (projectileP4 + targetP4).getNorm())) {
        result.header_.status_ = model.getMaterialIndex(target) < 0
                                     ? HadronicWorkerStatus::UnsupportedTarget
                                     : HadronicWorkerStatus::UnsupportedProjectile;
        return result;
      }

      auto keyedRng =
          default_prng_type{derive_hadronic_worker_seed(request.randomKey_), 0};
      std::uint64_t supplied = 0;
      ::fluka::set_rng_function(
          [rng = std::move(keyedRng), &supplied](double* destination,
                                                 std::size_t const count) mutable {
            std::uniform_real_distribution<double> uniform{0., 1.};
            std::generate_n(destination, count, [&]() { return uniform(rng); });
            supplied += count;
          });

      auto const finalStateStart = std::chrono::steady_clock::now();
      auto const finalState =
          model.generateFinalState(projectile, target, projectileP4, targetP4);
      result.header_.finalStateTimeMs_ =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                    finalStateStart)
              .count();
      if (finalState.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("FLUKA final state exceeds protocol count range");
      }
      result.secondaries_.reserve(finalState.size());
      for (auto const& [pid, kineticEnergy, direction] : finalState) {
        HadronicSecondaryRecord record{};
        record.pdg_ = static_cast<std::int32_t>(get_PDG(pid));
        record.kineticEnergyGeV_ = kineticEnergy / 1_GeV;
        auto const components =
            direction.getComponents(coordinateSystem).getEigenVector();
        record.direction_ = {components[0], components[1], components[2]};
        if (!std::isfinite(record.kineticEnergyGeV_) || record.kineticEnergyGeV_ < 0. ||
            !std::all_of(record.direction_.begin(), record.direction_.end(),
                         [](double const value) { return std::isfinite(value); })) {
          throw std::runtime_error("FLUKA generated an invalid secondary record");
        }
        result.secondaries_.push_back(record);
      }
      result.header_.status_ = HadronicWorkerStatus::Success;
      result.header_.secondaryCount_ =
          static_cast<std::uint32_t>(result.secondaries_.size());
      result.header_.randomValuesSupplied_ =
          supplied > std::numeric_limits<std::uint32_t>::max()
              ? std::numeric_limits<std::uint32_t>::max()
              : static_cast<std::uint32_t>(supplied);
    } catch (std::exception const& error) {
      CORSIKA_LOG_ERROR("FLUKA worker request {} failed: {}", request.sequenceId_,
                        error.what());
      result.header_.status_ = HadronicWorkerStatus::GeneratorFailure;
      result.header_.secondaryCount_ = 0;
      result.secondaries_.clear();
    }
    return result;
  }
  inline int serve_fluka_worker(fluka::InteractionModel& model,
                                CoordinateSystemPtr const& coordinateSystem,
                                int serverDescriptor, std::size_t workerId) {
#if defined(__unix__) || defined(__linux__)

    constexpr std::uint32_t maximumRequests = 1'000'000;
    std::uint64_t batchesProcessed = 0;
    std::uint64_t requestsProcessed = 0;
    while (true) {
      HadronicBatchHeader header;
      if (!fluka_worker_detail::read_descriptor_record(serverDescriptor, header, true)) {
        break;
      }
      validate_hadronic_batch_header(header);
      if (header.requestCount_ > maximumRequests) {
        throw std::runtime_error(
            "Persistent hadronic batch exceeds request safety limit");
      }
      std::vector<HadronicInteractionRequest> requests(header.requestCount_);
      if (!requests.empty()) {
        fluka_worker_detail::read_all_from_descriptor(
            serverDescriptor, requests.data(), requests.size() * sizeof(requests.front()),
            false);
      }

      std::vector<FlukaWorkerResult> results;
      results.reserve(requests.size());
      std::size_t totalSecondaries = 0;
      for (auto const& request : requests) {
        results.push_back(execute_fluka_request(model, request, coordinateSystem));
        if (results.back().secondaries_.size() >
            std::numeric_limits<std::size_t>::max() - totalSecondaries) {
          throw std::overflow_error(
              "Persistent hadronic response secondary count overflow");
        }
        totalSecondaries += results.back().secondaries_.size();
      }
      std::vector<HadronicInteractionResultHeader> resultHeaders;
      resultHeaders.reserve(results.size());
      std::vector<HadronicSecondaryRecord> flattenedSecondaries;
      flattenedSecondaries.reserve(totalSecondaries);
      for (auto& result : results) {
        resultHeaders.push_back(result.header_);
        flattenedSecondaries.insert(flattenedSecondaries.end(),
                                    result.secondaries_.begin(),
                                    result.secondaries_.end());
      }

      fluka_worker_detail::write_descriptor_record(serverDescriptor, header);
      if (!resultHeaders.empty()) {
        fluka_worker_detail::write_all_to_descriptor(
            serverDescriptor, resultHeaders.data(),
            resultHeaders.size() * sizeof(resultHeaders.front()));
      }
      if (!flattenedSecondaries.empty()) {
        fluka_worker_detail::write_all_to_descriptor(
            serverDescriptor, flattenedSecondaries.data(),
            flattenedSecondaries.size() * sizeof(flattenedSecondaries.front()));
      }
      ++batchesProcessed;
      requestsProcessed += requests.size();
    }
    CORSIKA_LOG_INFO(
        "Persistent FLUKA worker {} stopped cleanly after {} batches "
        "and {} requests",
        workerId, batchesProcessed, requestsProcessed);
    return EXIT_SUCCESS;

#else
    (void)model;
    (void)coordinateSystem;
    (void)serverDescriptor;
    (void)workerId;
    throw std::runtime_error("--server-fd requires a POSIX process environment");
#endif
  }
} // namespace corsika::kokkos
