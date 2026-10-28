/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "MoliereReference.hpp"
#include <corsika/kokkos/em/models/MoliereScattering.hpp>
#include <array>
#include <filesystem>
#include <vector>

namespace corsika::kokkos::test {
  enum class MoliereOperation : std::uint32_t {
    Root,
    Normal,
    Functions,
    InitialGuess,
    Angle,
    Projection,
    Direction,
    TrackedDirection,
    SmallCapacity
  };

  struct MoliereQuery {
    MoliereOperation operation_{};
    std::uint32_t snapshotIndex_{};
    std::uint32_t viewIndex_{};
    std::uint32_t validate_{1};
    double values_[6]{};
    double initial_[3]{1., 0., 0.};
    double tracked_[3]{0.8, 0.6, 0.};
    std::uint32_t nativeComparable_{};
    double nativeExpected_{};
  };

  struct MoliereOutcome {
    std::uint32_t status_{};
    std::uint32_t iterations_{};
    std::uint32_t count_{};
    double values_[3 * MaxMoliereComponents + 5]{};
  };

  struct MoliereDataset {
    std::vector<MoliereSnapshot> snapshots_;
    std::vector<MoliereQuery> queries_;
    MoliereInterpolationTable table_;
    std::filesystem::path cacheFile_;
  };

  /** Build real Air electron/muon snapshots; both use the electron-owned table. */
  MoliereDataset make_moliere_dataset();

  CORSIKA_HOST_DEVICE inline MoliereInterpolationView moliere_view_variant(
      MoliereInterpolationView const& full, std::uint32_t index) {
    if (index == 0) return {};
    MoliereInterpolationParameters parameters{
        full.getIntervalCount(),
        full.getInitialGuessBNodeCount(),
        full.getInitialGuessBetaSquaredNodeCount(),
        full.getInitialGuessCoordinateNodeCount(),
        full.getReserved(),
        full.getLower(),
        full.getUpper(),
        full.getInverseStep(),
        full.getInitialGuessBLower(),
        full.getInitialGuessBUpper(),
        full.getInitialGuessInverseBStep(),
        full.getInitialGuessBetaSquaredLower(),
        full.getInitialGuessBetaSquaredUpper(),
        full.getInitialGuessInverseBetaSquaredStep(),
        full.getInitialGuessCoordinateLower(),
        full.getInitialGuessCoordinateUpper(),
        full.getInitialGuessInverseCoordinateStep()};
    auto seed = full.getInitialGuessDelta();
    auto first = full.getF1();
    if (index == 1) {
      seed = {};
      parameters.initialGuessBNodeCount_ = 0;
      parameters.initialGuessBetaSquaredNodeCount_ = 0;
      parameters.initialGuessCoordinateNodeCount_ = 0;
    } else if (index == 3) {
      parameters.intervalCount_ = 0;
    } else if (index == 4) {
      first = {};
    } else if (index == 5) {
      parameters.initialGuessBNodeCount_ = 0;
    } else if (index == 6) {
      parameters.inverseStep_ = 0.;
    }
    return {first, full.getF2(), full.getIntegralF1(), full.getIntegralF2(),
            seed,  parameters};
  }

  CORSIKA_HOST_DEVICE inline moliere_reference::MoliereInterpolationView
  moliere_reference_view_variant(moliere_reference::MoliereInterpolationView full,
                                 std::uint32_t index) {
    if (index == 0) return {};
    if (index == 1) {
      full.initial_guess_delta = nullptr;
      full.initial_guess_b_node_count = 0;
      full.initial_guess_beta_squared_node_count = 0;
      full.initial_guess_coordinate_node_count = 0;
    } else if (index == 3) {
      full.interval_count = 0;
    } else if (index == 4) {
      full.f1 = nullptr;
    } else if (index == 5) {
      full.initial_guess_b_node_count = 0;
    } else if (index == 6) {
      full.inverse_step = 0.;
    }
    return full;
  }

  CORSIKA_HOST_DEVICE inline MoliereOutcome evaluate_moliere(
      MoliereSnapshot const& snapshot, MoliereInterpolationView const& view,
      MoliereQuery const& query) {
    MoliereOutcome result{};
    auto const& value = query.values_;
    using namespace moliere_detail;
    switch (query.operation_) {
      case MoliereOperation::Root:
        result.count_ = 1;
        result.values_[0] = solve_moliere_b(value[0]);
        break;
      case MoliereOperation::Normal:
        result.count_ = 1;
        result.values_[0] = normal_ppf(value[0]);
        break;
      case MoliereOperation::Functions:
        result.count_ = 8;
        result.values_[0] = f1_m(snapshot, value[0]);
        result.values_[1] = f2_m(snapshot, value[0]);
        result.values_[2] = integral_f1_m(snapshot, value[0]);
        result.values_[3] = integral_f2_m(snapshot, value[0]);
        result.values_[4] = f1_m(snapshot, view, value[0]);
        result.values_[5] = f2_m(snapshot, view, value[0]);
        result.values_[6] = integral_f1_m(snapshot, view, value[0]);
        result.values_[7] = integral_f2_m(snapshot, view, value[0]);
        break;
      case MoliereOperation::InitialGuess:
        result.count_ = 1;
        result.values_[0] =
            improve_moliere_initial_guess(view, value[0], value[1], value[2]);
        break;
      case MoliereOperation::Angle:
      case MoliereOperation::SmallCapacity: {
        auto const angle =
            query.operation_ == MoliereOperation::SmallCapacity
                ? (value[5] == 4. ? sample_moliere_scattering_angle_2d_for_capacity<4>(
                                        snapshot, view, value[0], value[1], value[2],
                                        value[3], value[4], query.validate_ != 0)
                                  : sample_moliere_scattering_angle_2d_for_capacity<1>(
                                        snapshot, view, value[0], value[1], value[2],
                                        value[3], value[4], query.validate_ != 0))
                : sample_moliere_scattering_angle_2d(snapshot, view, value[0], value[1],
                                                     value[2], value[3], value[4],
                                                     query.validate_ != 0);
        result.status_ = static_cast<std::uint32_t>(angle.status_);
        result.iterations_ = angle.iterations_;
        result.count_ = 1;
        result.values_[0] = angle.angleRad_;
        break;
      }
      case MoliereOperation::Projection: {
        distribution_state_type state{};
        auto const status =
            build_distribution(snapshot, value[0], value[1], state, query.validate_ != 0);
        result.status_ = static_cast<std::uint32_t>(status);
        if (status != MoliereStatus::Success) break;
        for (std::uint32_t index = 0; index < snapshot.componentCount_; ++index) {
          result.values_[result.count_++] = state.inverseScale_[index];
          result.values_[result.count_++] = state.inverseB_[index];
          result.values_[result.count_++] = state.densityWeight_[index];
        }
        result.values_[result.count_++] = state.prefactor_;
        result.values_[result.count_++] = state.betaSquared_;
        auto const evaluation = evaluate_distribution(snapshot, view, state, value[2]);
        result.values_[result.count_++] = evaluation.density_;
        result.values_[result.count_++] = evaluation.cumulative_;
        auto const angle = sample_one_dimensional(snapshot, view, state, value[3]);
        result.status_ = static_cast<std::uint32_t>(angle.status_);
        result.iterations_ = angle.iterations_;
        result.values_[result.count_++] = angle.angleRad_;
        break;
      }
      case MoliereOperation::Direction:
      case MoliereOperation::TrackedDirection: {
        auto const direction =
            query.operation_ == MoliereOperation::Direction
                ? apply_moliere_direction(query.initial_, value[0], value[1])
                : apply_moliere_direction_cpu_step_compatible(
                      query.initial_, query.tracked_, value[0], value[1]);
        result.status_ = static_cast<std::uint32_t>(direction.status_);
        result.iterations_ = direction.reserved_;
        result.count_ = 3;
        for (unsigned axis = 0; axis < 3; ++axis)
          result.values_[axis] = direction.direction_[axis];
        break;
      }
    }
    return result;
  }

  CORSIKA_HOST_DEVICE inline MoliereOutcome reference_moliere(
      moliere_reference::MoliereSnapshot const& snapshot,
      moliere_reference::MoliereInterpolationView const& view,
      MoliereQuery const& query) {
    MoliereOutcome result{};
    auto const& value = query.values_;
    namespace old = moliere_reference;
    using namespace old::moliere_detail;
    switch (query.operation_) {
      case MoliereOperation::Root:
        result.count_ = 1;
        result.values_[0] = solveMoliereB(value[0]);
        break;
      case MoliereOperation::Normal:
        result.count_ = 1;
        result.values_[0] = normalPpf(value[0]);
        break;
      case MoliereOperation::Functions:
        result.count_ = 8;
        result.values_[0] = f1M(snapshot, value[0]);
        result.values_[1] = f2M(snapshot, value[0]);
        result.values_[2] = F1M(snapshot, value[0]);
        result.values_[3] = F2M(snapshot, value[0]);
        result.values_[4] = f1M(snapshot, view, value[0]);
        result.values_[5] = f2M(snapshot, view, value[0]);
        result.values_[6] = F1M(snapshot, view, value[0]);
        result.values_[7] = F2M(snapshot, view, value[0]);
        break;
      case MoliereOperation::InitialGuess:
        result.count_ = 1;
        result.values_[0] =
            improveMoliereInitialGuess(view, value[0], value[1], value[2]);
        break;
      case MoliereOperation::Angle:
      case MoliereOperation::SmallCapacity: {
        auto const angle =
            query.operation_ == MoliereOperation::SmallCapacity
                ? (value[5] == 4. ? old::sampleMoliereScatteringAngle2DForCapacity<4>(
                                        snapshot, view, value[0], value[1], value[2],
                                        value[3], value[4], query.validate_ != 0)
                                  : old::sampleMoliereScatteringAngle2DForCapacity<1>(
                                        snapshot, view, value[0], value[1], value[2],
                                        value[3], value[4], query.validate_ != 0))
                : old::sampleMoliereScatteringAngle2D(snapshot, view, value[0], value[1],
                                                      value[2], value[3], value[4],
                                                      query.validate_ != 0);
        result.status_ = static_cast<std::uint32_t>(angle.status);
        result.iterations_ = angle.iterations;
        result.count_ = 1;
        result.values_[0] = angle.angle_rad;
        break;
      }
      case MoliereOperation::Projection: {
        old::moliere_detail::DistributionState state{};
        auto const status =
            buildDistribution(snapshot, value[0], value[1], state, query.validate_ != 0);
        result.status_ = static_cast<std::uint32_t>(status);
        if (status != old::MoliereStatus::Success) break;
        for (std::uint32_t index = 0; index < snapshot.component_count; ++index) {
          result.values_[result.count_++] = state.inverse_scale[index];
          result.values_[result.count_++] = state.inverse_B[index];
          result.values_[result.count_++] = state.density_weight[index];
        }
        result.values_[result.count_++] = state.prefactor;
        result.values_[result.count_++] = state.beta_squared;
        auto const evaluation = evaluateDistribution(snapshot, view, state, value[2]);
        result.values_[result.count_++] = evaluation.density;
        result.values_[result.count_++] = evaluation.cumulative;
        auto const angle = sampleOneDimensional(snapshot, view, state, value[3]);
        result.status_ = static_cast<std::uint32_t>(angle.status);
        result.iterations_ = angle.iterations;
        result.values_[result.count_++] = angle.angle_rad;
        break;
      }
      case MoliereOperation::Direction:
      case MoliereOperation::TrackedDirection: {
        auto const direction =
            query.operation_ == MoliereOperation::Direction
                ? old::applyMoliereDirection(query.initial_, value[0], value[1])
                : old::applyMoliereDirectionCpuStepCompatible(
                      query.initial_, query.tracked_, value[0], value[1]);
        result.status_ = static_cast<std::uint32_t>(direction.status);
        result.iterations_ = direction.reserved;
        result.count_ = 3;
        for (unsigned axis = 0; axis < 3; ++axis)
          result.values_[axis] = direction.direction[axis];
        break;
      }
    }
    return result;
  }
} // namespace corsika::kokkos::test
