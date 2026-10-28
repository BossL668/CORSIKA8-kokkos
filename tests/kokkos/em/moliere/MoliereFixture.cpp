/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "MoliereQueryTestData.hpp"
#include "MoliereTestUtilities.hpp"
#include "../models/AuxiliaryModelFixture.hpp"
#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/scattering/multiple_scattering/Moliere.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace corsika::kokkos::test {
  MoliereDataset make_moliere_dataset() {
    auto const auxiliary = make_auxiliary_dataset();
    MoliereDataset data;
    data.snapshots_ = {auxiliary.snapshots_.front().electronMoliere_,
                       auxiliary.snapshots_.front().muonMoliere_};
    data.cacheFile_ = moliere_test_directory("real-air") / "electron-shared.cache";
    // Preserve the actual production wiring: a single electron-owned table.
    data.table_ =
        load_or_make_moliere_interpolation_table(data.snapshots_[0], data.cacheFile_);
    auto const oldSnapshot = moliere_reference_snapshot(data.snapshots_[0]);
    auto const oldTable = moliere_reference::makeMoliereInterpolationTable(oldSnapshot);
    auto const adaptedTable = moliere_reference_table(data.table_);
    if (std::memcmp(&oldTable, &adaptedTable, sizeof(oldTable)) != 0)
      throw std::runtime_error(
          "Real Air table differs from independent production generation");

    auto append = [&](MoliereOperation operation, std::uint32_t snapshot,
                      std::uint32_t view, std::array<double, 6> values) {
      MoliereQuery query{};
      query.operation_ = operation;
      query.snapshotIndex_ = snapshot;
      query.viewIndex_ = view;
      std::copy(values.begin(), values.end(), query.values_);
      data.queries_.push_back(query);
    };
    auto const infinity = std::numeric_limits<double>::infinity();
    auto const nan = std::numeric_limits<double>::quiet_NaN();
    for (double rhs : {0., 1., std::nextafter(MoliereMinimumBEquationRightHandSide, 0.),
                       MoliereMinimumBEquationRightHandSide,
                       std::nextafter(MoliereMinimumBEquationRightHandSide, infinity),
                       MoliereMinimumBEquationRightHandSide + 1.e-9, 3., 4., 8., 16., 64.,
                       1.e3, 1.e8, infinity, nan})
      append(MoliereOperation::Root, 0, 0, {rhs});
    for (double uniform : {std::numeric_limits<double>::denorm_min(), 1.e-15, 1.e-8,
                           std::nextafter(.02425, 0.), .02425, std::nextafter(.02425, 1.),
                           .1, std::nextafter(.5, 0.), .5, std::nextafter(.5, 1.), .9,
                           std::nextafter(.97575, 0.), .97575, std::nextafter(.97575, 1.),
                           std::nextafter(1., 0.)})
      append(MoliereOperation::Normal, 0, 0, {uniform});

    for (std::uint32_t snapshot = 0; snapshot < 2; ++snapshot) {
      auto const mass = data.snapshots_[snapshot].particleMassMeV_;
      for (std::uint32_t view = 0; view < 3; ++view) {
        for (double x :
             {0., .01, 1., std::nextafter(12., 0.), 12., std::nextafter(12., infinity),
              std::nextafter(18.0625, 0.), 18.0625, std::nextafter(18.0625, infinity),
              std::nextafter(20., 0.), 20., std::nextafter(20., infinity),
              std::nextafter(42.25, 0.), 42.25, std::nextafter(42.25, infinity), 100.,
              1.e6})
          append(MoliereOperation::Functions, snapshot, view, {x});
        for (double energy : {mass * 1.01, mass * 2., 1.e3, 1.e6}) {
          for (double grammage : {1.e-10, 1.e-4, .1, 10.}) {
            for (double uniform : {1.e-10, .02425, .1, std::nextafter(.5, 0.), .5,
                                   std::nextafter(.5, 1.), .8, 1. - 1.e-10}) {
              append(MoliereOperation::Angle, snapshot, view,
                     {grammage, energy, energy * .9, uniform, .713});
              append(MoliereOperation::SmallCapacity, snapshot, view,
                     {grammage, energy, energy * .9, uniform, .713, 4.});
            }
          }
        }
        for (double theta : {-.1, -.001, -0., 0., .001, .1, 2.}) {
          for (double uniform : {1.e-8, .25, .5, .75, 1. - 1.e-8})
            append(MoliereOperation::Projection, snapshot, view,
                   {1., 1.e3, theta, uniform});
        }
      }
      for (std::uint32_t view = 0; view < 7; ++view) {
        append(MoliereOperation::Angle, snapshot, view, {1., 1.e3, nan, .2, .8});
        append(MoliereOperation::SmallCapacity, snapshot, view, {1., 1.e3, 900., .2, .8});
      }
      for (double grammage : {-1., 0., nan, infinity, std::numeric_limits<double>::max()})
        append(MoliereOperation::Angle, snapshot, 2, {grammage, 1.e3, 900., .2, .8});
      for (double energy :
           {0., mass, std::nextafter(mass, infinity), nan, infinity, 1.e200})
        append(MoliereOperation::Angle, snapshot, 2, {1., energy, energy, .2, .8});
      for (double uniform : {-1., 0., 1., infinity, nan}) {
        append(MoliereOperation::Angle, snapshot, 2, {1., 1.e3, 900., uniform, .8});
        append(MoliereOperation::Angle, snapshot, 2, {1., 1.e3, 900., .2, uniform});
      }
    }
    // Values on and outside each seed-grid boundary, including both signs.
    for (double referenceB : {4.49, 4.5, 8., 64., std::nextafter(64., infinity), nan})
      for (double betaSquared : {.249, .25, .75, 1., 1.01})
        for (double coordinate : {-5.01, -5., -1., 0., 1., 5., 5.01})
          append(MoliereOperation::InitialGuess, 0, 2,
                 {referenceB, betaSquared, coordinate});
    append(MoliereOperation::InitialGuess, 0, 1, {8., .75, 1.});
    append(MoliereOperation::InitialGuess, 0, 2, {8., .75, nan});

    for (auto operation :
         {MoliereOperation::Direction, MoliereOperation::TrackedDirection}) {
      for (double x : {0., std::nextafter(.1, 0.), .1, std::nextafter(.1, 1.), .8, 1.}) {
        for (double angle : {0., 1.e-8, .1, 3.}) {
          append(operation, 0, 0, {angle, .37});
          auto& query = data.queries_.back();
          query.initial_[0] = x;
          query.initial_[1] = std::sqrt(1. - x * x);
        }
      }
      for (double value : {-1., nan, infinity}) append(operation, 0, 0, {value, .37});
      for (double value : {0., 1., nan}) append(operation, 0, 0, {.1, value});
      for (unsigned axis = 0; axis < 3; ++axis) {
        append(operation, 0, 0, {.1, .37});
        data.queries_.back().initial_[axis] = 2.;
        append(operation, 0, 0, {.1, .37});
        data.queries_.back().tracked_[axis] = 2.;
      }
      append(operation, 0, 0, {.1, .37});
      data.queries_.back().initial_[0] = -1.;
      append(operation, 0, 0, {.1, .37});
      data.queries_.back().initial_[0] = 2. / 3.;
      data.queries_.back().initial_[1] = 2. / 3.;
      data.queries_.back().initial_[2] = 1. / 3.;
    }
    // A host-materialized cancellation input; both device paths consume it
    // unchanged. Exact cancellation is not promised across different backends.
    append(MoliereOperation::TrackedDirection, 0, 0, {moliere_detail::Pi / 3., .25});
    auto& cancellation = data.queries_.back();
    auto const scattered = moliere_reference::applyMoliereDirection(
        cancellation.initial_, cancellation.values_[0], cancellation.values_[1]);
    for (unsigned axis = 0; axis < 3; ++axis)
      cancellation.tracked_[axis] =
          cancellation.initial_[axis] - scattered.direction[axis];
    // Distinct invalid snapshots cover early exits without reaching unsafe array
    // accesses.
    for (unsigned mutation = 0; mutation < 5; ++mutation) {
      auto changed = data.snapshots_[0];
      switch (mutation) {
        case 0:
          changed.componentCount_ = 0;
          break;
        case 1:
          changed.componentCount_ = MaxMoliereComponents + 1;
          break;
        case 2:
          changed.maximumWeightIndex_ = changed.componentCount_;
          break;
        case 3:
          changed.components_[0].weightZz_ = nan;
          break;
        case 4:
          changed.c1_[0] = nan;
          break;
      }
      auto const index = static_cast<std::uint32_t>(data.snapshots_.size());
      data.snapshots_.push_back(changed);
      append(MoliereOperation::Angle, index, 2, {1., 1.e3, 900., .2, .8});
    }
    // Validation can be skipped only with a genuinely prevalidated input.
    append(MoliereOperation::Angle, 0, 2, {1., 1.e3, 900., .2, .8});
    data.queries_.back().validate_ = 0;

    PROPOSAL::Air medium;
    std::array<PROPOSAL::ParticleDef, 2> particles{PROPOSAL::EMinusDef(),
                                                   PROPOSAL::MuMinusDef()};
    std::array<PROPOSAL::ParticleDef, 2> antiparticles{PROPOSAL::EPlusDef(),
                                                       PROPOSAL::MuPlusDef()};
    for (std::uint32_t snapshot = 0; snapshot < 2; ++snapshot) {
      PROPOSAL::multiple_scattering::Moliere analytic(particles[snapshot], medium);
      PROPOSAL::multiple_scattering::MoliereInterpol interpolated(particles[snapshot],
                                                                  medium);
      PROPOSAL::multiple_scattering::Moliere positiveAnalytic(antiparticles[snapshot],
                                                              medium);
      PROPOSAL::multiple_scattering::MoliereInterpol positiveInterpolated(
          antiparticles[snapshot], medium);
      for (std::uint32_t view : {0u, 1u, 2u})
        for (double energy : {1.e3, 1.e5})
          for (double grammage : {.1, 1., 10.})
            for (double uniform : {.1, .25, .75, .9}) {
              append(MoliereOperation::Angle, snapshot, view,
                     {grammage, energy, energy * .9, uniform, .713});
              auto& query = data.queries_.back();
              query.nativeComparable_ = 1;
              query.nativeExpected_ =
                  view == 0 ? analytic.CalculateScatteringAngle2D(
                                  grammage, energy, energy * .9, uniform, .713)
                            : interpolated.CalculateScatteringAngle2D(
                                  grammage, energy, energy * .9, uniform, .713);
              auto const positive =
                  view == 0 ? positiveAnalytic.CalculateScatteringAngle2D(
                                  grammage, energy, energy * .9, uniform, .713)
                            : positiveInterpolated.CalculateScatteringAngle2D(
                                  grammage, energy, energy * .9, uniform, .713);
              if (!std::isfinite(query.nativeExpected_) ||
                  positive != query.nativeExpected_)
                throw std::runtime_error(
                    "Real PROPOSAL Moliere charge-pair comparison failed");
            }
    }
    return data;
  }
} // namespace corsika::kokkos::test
