/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "UniformMagneticFieldReference.hpp"
#include <corsika/kokkos/environment/UniformMagneticField.hpp>
#include <cmath>
#include <limits>
#include <vector>

namespace uniform_magnetic_test {

  using particle_type = corsika::kokkos::ParticleState;
  using status_type = corsika::kokkos::MagneticStepStatus;
  enum class Category {
    Regular,
    Analytic,
    Neutral,
    ZeroField,
    Parallel,
    NearParallel,
    PerpendicularThreshold,
    RadiusThreshold,
    Angle,
    InvalidParticle,
    InvalidField,
    InvalidDistance,
    ZeroDistance,
    SpecialMetadata,
    NonFiniteMomentum,
    UnderflowMomentum,
    OverflowField,
    OverflowChord,
    Scan
  };

  struct Query {
    particle_type particle_;
    double massGeV_{3.};
    double chargeNumber_{1.};
    double fieldT_[3]{0., 0., 1.};
    double distanceM_{1.};
    double maximumDeflection_{.2};
    Category category_{Category::Regular};
    // Only ordinary well-conditioned queries have a cross-device accuracy gate.
    // Threshold and IEEE stress queries still require every same-device output
    // to equal the immutable production oracle, without changing FMA settings.
    bool crossDevice_{true};
    int side_{};
  };

  struct Limit {
    std::uint32_t status_{};
    std::uint32_t reserved_{};
    double distanceM_{};
    double gyroradiusM_{};
  };

  struct Advance {
    std::uint32_t status_{};
    std::uint32_t reserved_{};
    particle_type particle_;
    double bendParameter_{};
    double chordLengthM_{};
  };

  struct Outcome {
    Limit limit_;
    Advance advance_;
    Advance callerAdvance_;
    bool callerRan_{};
    bool callerUsedZeroField_{};
  };

  struct Witness {
    double perpendicularMomentumGeV_{};
    double gyroradiusM_{};
  };

  inline Query base_query() {
    Query query;
    query.particle_.pid_ = 13;
    query.particle_.mediumId_ = 7;
    query.particle_.generation_ = 2;
    query.particle_.reserved_ = 19;
    query.particle_.energyGeV_ = 5.;
    query.particle_.positionM_[0] = .25;
    query.particle_.positionM_[1] = -.5;
    query.particle_.positionM_[2] = .75;
    query.particle_.direction_[0] = 1.;
    query.particle_.timeS_ = 2.e-7;
    query.particle_.weight_ = 3.;
    query.particle_.historyId_ = 123;
    query.particle_.parentHistoryId_ = 45;
    query.particle_.stepId_ = 6;
    query.particle_.environmentNode_ = 1;
    return query;
  }

  inline std::vector<Query> queries() {
    std::vector<Query> result;
    auto query = base_query();
    query.category_ = Category::Analytic;
    query.distanceM_ = 4. / uniform_magnetic_reference::GeVPerCToTeslaMeter;
    result.push_back(query);
    query = base_query();
    result.push_back(query);
    query.chargeNumber_ = 0.;
    query.category_ = Category::Neutral;
    result.push_back(query);
    query = base_query();
    query.fieldT_[2] = 0.;
    query.category_ = Category::ZeroField;
    result.push_back(query);
    query.fieldT_[0] = 1.;
    query.category_ = Category::Parallel;
    result.push_back(query);

    for (double transverse : {.5e-9, 1.e-9, 2.e-9}) {
      query = base_query();
      query.fieldT_[0] = 1.;
      query.fieldT_[2] = 0.;
      query.particle_.direction_[1] = transverse / 4.;
      query.particle_.direction_[0] =
          std::sqrt(1. - query.particle_.direction_[1] * query.particle_.direction_[1]);
      query.category_ = Category::PerpendicularThreshold;
      query.crossDevice_ = false;
      query.side_ = transverse < 1.e-9 ? -1 : (transverse > 1.e-9 ? 1 : 0);
      result.push_back(query);
    }
    for (double scale : {.5, 1., 2.}) {
      query = base_query();
      query.fieldT_[2] =
          4. / (uniform_magnetic_reference::GeVPerCToTeslaMeter * scale * 1.e9);
      query.category_ = Category::RadiusThreshold;
      query.crossDevice_ = false;
      query.side_ = scale < 1. ? -1 : (scale > 1. ? 1 : 0);
      result.push_back(query);
    }
    for (double perturbation : {0., 1.e-14, 1.e-10}) {
      query = base_query();
      query.particle_.energyGeV_ = 1.e8;
      query.massGeV_ = .000511;
      query.particle_.direction_[0] = 1. / 3.;
      query.particle_.direction_[1] = 2. / 3.;
      query.particle_.direction_[2] = 2. / 3.;
      for (int axis = 0; axis < 3; ++axis) {
        query.fieldT_[axis] = 5.e-5 * query.particle_.direction_[axis];
      }
      query.fieldT_[1] += perturbation;
      query.category_ = Category::NearParallel;
      query.crossDevice_ = false;
      result.push_back(query);
    }
    double const halfPi = .5 * 3.141592653589793;
    double const infinity = std::numeric_limits<double>::infinity();
    double const nan = std::numeric_limits<double>::quiet_NaN();
    for (double angle :
         {0., -.2, halfPi, std::nextafter(halfPi, 0.), std::nextafter(halfPi, infinity),
          infinity, nan, std::numeric_limits<double>::denorm_min()}) {
      query = base_query();
      query.maximumDeflection_ = angle;
      query.category_ = Category::Angle;
      query.crossDevice_ = false;
      result.push_back(query);
    }
    for (double energy : {3., 2., infinity, nan}) {
      query = base_query();
      query.particle_.energyGeV_ = energy;
      query.category_ = Category::InvalidParticle;
      query.crossDevice_ = false;
      result.push_back(query);
    }
    for (double mass : {0., -1., infinity, nan}) {
      query = base_query();
      query.massGeV_ = mass;
      query.category_ = Category::InvalidParticle;
      query.crossDevice_ = false;
      result.push_back(query);
    }
    for (double charge : {infinity, nan}) {
      query = base_query();
      query.chargeNumber_ = charge;
      query.category_ = Category::InvalidParticle;
      query.crossDevice_ = false;
      result.push_back(query);
    }
    for (int axis = 0; axis < 3; ++axis) {
      query = base_query();
      query.particle_.positionM_[axis] = infinity;
      query.category_ = Category::InvalidParticle;
      query.crossDevice_ = false;
      result.push_back(query);
      query = base_query();
      query.particle_.direction_[axis] = nan;
      query.category_ = Category::InvalidParticle;
      query.crossDevice_ = false;
      result.push_back(query);
      query = base_query();
      query.fieldT_[axis] = infinity;
      query.category_ = Category::InvalidField;
      query.crossDevice_ = false;
      result.push_back(query);
    }
    for (double norm : {0., 1. + 2.e-12, 1. + 2.e-13}) {
      query = base_query();
      query.particle_.direction_[0] = norm;
      query.category_ = Category::InvalidParticle;
      query.crossDevice_ = false;
      query.side_ = norm == 1. + 2.e-13 ? 1 : -1;
      result.push_back(query);
    }
    for (double distance : {-1., infinity, nan}) {
      query = base_query();
      query.distanceM_ = distance;
      query.category_ = Category::InvalidDistance;
      query.crossDevice_ = false;
      result.push_back(query);
    }
    for (double distance : {0., -0.}) {
      query = base_query();
      query.distanceM_ = distance;
      query.category_ = Category::ZeroDistance;
      result.push_back(query);
    }
    for (double time : {infinity, nan, -1.}) {
      query = base_query();
      query.particle_.timeS_ = time;
      query.particle_.weight_ = nan;
      query.category_ = Category::SpecialMetadata;
      query.crossDevice_ = false;
      result.push_back(query);
    }
    query = base_query();
    query.particle_.energyGeV_ = 1.e308;
    query.category_ = Category::NonFiniteMomentum;
    query.crossDevice_ = false;
    result.push_back(query);
    query.particle_.energyGeV_ = 1.e-200;
    query.massGeV_ = .5e-200;
    query.category_ = Category::UnderflowMomentum;
    result.push_back(query);
    query = base_query();
    query.fieldT_[2] = 1.e308;
    query.category_ = Category::OverflowField;
    query.crossDevice_ = false;
    result.push_back(query);
    query = base_query();
    query.fieldT_[2] = 0.;
    query.distanceM_ = 1.e308;
    query.particle_.positionM_[0] = 1.e308;
    query.category_ = Category::OverflowChord;
    query.crossDevice_ = false;
    result.push_back(query);

    for (double energy : {.001, 1., 1.e3, 1.e8}) {
      for (double charge : {-1., 1., 2.}) {
        for (double distance : {.001, .1, 1., 10.}) {
          query = base_query();
          query.particle_.energyGeV_ = energy;
          query.massGeV_ = .000511;
          query.chargeNumber_ = charge;
          query.distanceM_ = distance;
          query.particle_.direction_[0] = 1. / 3.;
          query.particle_.direction_[1] = 2. / 3.;
          query.particle_.direction_[2] = 2. / 3.;
          query.fieldT_[0] = 2.e-5;
          query.fieldT_[1] = -3.e-5;
          query.fieldT_[2] = 4.e-5;
          query.category_ = Category::Scan;
          result.push_back(query);
        }
      }
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION Witness witness(Query const& query) {
    using namespace uniform_magnetic_reference::magnetic_detail;
    double const momentumGeV = squareRoot((query.particle_.energyGeV_ - query.massGeV_) *
                                          (query.particle_.energyGeV_ + query.massGeV_));
    double momentum[3];
    for (int axis = 0; axis < 3; ++axis) {
      momentum[axis] = momentumGeV * query.particle_.direction_[axis];
    }
    double const fieldSquared = dot(query.fieldT_, query.fieldT_);
    double const projection = dot(momentum, query.fieldT_) / fieldSquared;
    double perpendicular[3];
    for (int axis = 0; axis < 3; ++axis) {
      perpendicular[axis] = momentum[axis] - projection * query.fieldT_[axis];
    }
    double const transverse = squareRoot(dot(perpendicular, perpendicular));
    return {transverse,
            transverse / (uniform_magnetic_reference::GeVPerCToTeslaMeter *
                          absolute(query.chargeNumber_) * squareRoot(fieldSquared))};
  }

  KOKKOS_INLINE_FUNCTION Limit
  convert(uniform_magnetic_reference::MagneticStepLimitResult value) {
    return {static_cast<std::uint32_t>(value.status), value.reserved, value.distance_m,
            value.gyroradius_m};
  }
  KOKKOS_INLINE_FUNCTION Advance
  convert(uniform_magnetic_reference::MagneticAdvanceResult value) {
    return {static_cast<std::uint32_t>(value.status), value.reserved, value.particle,
            value.bend_parameter, value.chord_length_m};
  }
  KOKKOS_INLINE_FUNCTION Limit convert(corsika::kokkos::MagneticStepLimitResult value) {
    return {static_cast<std::uint32_t>(value.status_), value.reserved_, value.distanceM_,
            value.gyroradiusM_};
  }
  KOKKOS_INLINE_FUNCTION Advance convert(corsika::kokkos::MagneticAdvanceResult value) {
    return {static_cast<std::uint32_t>(value.status_), value.reserved_, value.particle_,
            value.bendParameter_, value.chordLengthM_};
  }

  KOKKOS_INLINE_FUNCTION Outcome expected(Query const& query) {
    using namespace uniform_magnetic_reference;
    Outcome result;
    result.limit_ = convert(maximumUniformMagneticStep(query.particle_, query.massGeV_,
                                                       query.chargeNumber_, query.fieldT_,
                                                       query.maximumDeflection_));
    result.advance_ = convert(
        advanceUniformMagneticField(query.particle_, query.massGeV_, query.chargeNumber_,
                                    query.fieldT_, query.distanceM_));
    result.callerRan_ = result.limit_.status_ == 0 || result.limit_.status_ == 1;
    result.callerUsedZeroField_ = result.limit_.status_ == 1;
    if (result.callerRan_) {
      double field[3];
      for (int axis = 0; axis < 3; ++axis) {
        field[axis] = result.callerUsedZeroField_ ? 0. : query.fieldT_[axis];
      }
      result.callerAdvance_ = convert(advanceUniformMagneticField(
          query.particle_, query.massGeV_, query.chargeNumber_, field, query.distanceM_));
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION Outcome actual(Query const& query) {
    using namespace corsika::kokkos;
    Outcome result;
    result.limit_ = convert(maximum_uniform_magnetic_step(
        query.particle_, query.massGeV_, query.chargeNumber_, query.fieldT_,
        query.maximumDeflection_));
    result.advance_ = convert(advance_uniform_magnetic_field(
        query.particle_, query.massGeV_, query.chargeNumber_, query.fieldT_,
        query.distanceM_));
    result.callerRan_ = result.limit_.status_ == 0 || result.limit_.status_ == 1;
    result.callerUsedZeroField_ = result.limit_.status_ == 1;
    if (result.callerRan_) {
      double field[3];
      for (int axis = 0; axis < 3; ++axis) {
        field[axis] = result.callerUsedZeroField_ ? 0. : query.fieldT_[axis];
      }
      result.callerAdvance_ = convert(advance_uniform_magnetic_field(
          query.particle_, query.massGeV_, query.chargeNumber_, field, query.distanceM_));
    }
    return result;
  }

} // namespace uniform_magnetic_test
