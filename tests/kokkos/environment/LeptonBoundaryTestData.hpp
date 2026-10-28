/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include "LeptonBoundaryReference.hpp"

#include <corsika/kokkos/environment/EnvironmentLeptonBoundary.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace lepton_boundary_test {

  using particle_type = corsika::kokkos::ParticleState;
  using limit_type = corsika::kokkos::MagneticStepLimitResult;
  using status_type = corsika::kokkos::MagneticStepStatus;
  using shape_type = corsika::kokkos::EnvironmentShape;
  using tree_fixture_type = straight_boundary_reference::TreeFixture;

  enum class TreeKind { Ordered, Spatial, Unsupported, Incomplete, Both };
  enum class Category {
    Limit,
    TimeTie,
    Outside,
    NearRest,
    Mass,
    Charge,
    ZeroField,
    Parallel,
    ObservationZero,
    ObservationBehind,
    ObservationParallel,
    ObservationBeyond,
    ObservationEqual,
    ForcedCurved,
    NoRealPlaneRoots,
    Diagnostics,
    Spatial,
    Scan
  };

  struct Query {
    particle_type particle_;
    double massGeV_{.000511};
    double chargeNumber_{1.};
    double fieldT_[3]{};
    limit_type limit_;
    double pointM_[3]{3., 0., 0.};
    double normal_[3]{1., 0., 0.};
    Category category_{Category::Limit};
  };

  struct Outcome {
    double speedMPerS_{};
    double distanceM_{};
    double observationM_{};
    double emptyObservationM_{};
    std::int32_t nextNode_{};
    bool failed_{};
    bool unsupported_{};
    bool incomplete_{};
    bool freshFlagsClear_{};
    bool wrapperAgrees_{};
  };

  inline Query base_query() {
    Query query;
    auto& particle = query.particle_;
    particle.pid_ = 11;
    particle.mediumId_ = 7;
    particle.generation_ = 2;
    particle.reserved_ = 17;
    particle.energyGeV_ = 1.;
    particle.direction_[0] = 1.;
    particle.timeS_ = 2.e-7;
    particle.weight_ = 3.;
    particle.historyId_ = 123;
    particle.parentHistoryId_ = 45;
    particle.stepId_ = 6;
    particle.environmentNode_ = 1;
    query.limit_ = {status_type::Linear, 19, 1., std::numeric_limits<double>::infinity()};
    return query;
  }

  inline tree_fixture_type tree(TreeKind kind) {
    tree_fixture_type result;
    std::size_t const size =
        kind == TreeKind::Ordered
            ? 4
            : kind == TreeKind::Incomplete || kind == TreeKind::Spatial ? 2 : 3;
    result.nodes_.resize(size);
    result.shapes_.resize(size);
    result.shapes_[0].kind_ = shape_type::Kind::Universe;
    result.shapes_[1].kind_ = shape_type::Kind::Sphere;
    result.shapes_[1].radiusM_ =
        kind == TreeKind::Incomplete || kind == TreeKind::Both ? 5.e-5 : 10.;
    result.nodes_[0].childCount_ = 1;
    result.nodes_[1].parentIndex_ = 0;
    result.children_.push_back(1);
    if (kind == TreeKind::Ordered) {
      result.nodes_[1].childOffset_ = 1;
      result.nodes_[1].childCount_ = 1;
      result.nodes_[1].excludedCount_ = 1;
      result.children_.push_back(2);
      result.exclusions_.push_back(3);
      for (std::size_t node = 2; node < 4; ++node) {
        result.nodes_[node].parentIndex_ = 1;
        result.shapes_[node].kind_ = shape_type::Kind::SeparationPlane;
        result.shapes_[node].centerM_[0] = 2.;
        result.shapes_[node].normal_[0] = -1.;
      }
    } else if (size == 3) {
      result.nodes_[1].childOffset_ = 1;
      result.nodes_[1].childCount_ = 1;
      result.children_.push_back(2);
      result.nodes_[2].parentIndex_ = 1;
      result.shapes_[2] = straight_boundary_reference::box(2.e-6);
      result.shapes_[2].centerM_[0] = 1.e-5;
    }
    return result;
  }

  inline std::vector<Query> queries(TreeKind kind) {
    std::vector<Query> result;
    if (kind == TreeKind::Spatial) {
      for (double charge : {-1., 1., 2.}) {
        auto query = base_query();
        query.category_ = Category::Spatial;
        query.chargeNumber_ = charge;
        query.particle_.positionM_[0] = .25;
        query.particle_.positionM_[1] = -.5;
        query.particle_.positionM_[2] = .75;
        query.particle_.direction_[0] = 1. / 3.;
        query.particle_.direction_[1] = 2. / 3.;
        query.particle_.direction_[2] = 2. / 3.;
        query.fieldT_[0] = .01;
        query.fieldT_[1] = -.02;
        query.fieldT_[2] = .03;
        query.pointM_[0] = 2.;
        query.pointM_[1] = 3.;
        query.pointM_[2] = 4.;
        query.normal_[0] = 2. / 3.;
        query.normal_[1] = 1. / 3.;
        query.normal_[2] = 2. / 3.;
        double const energy = query.particle_.energyGeV_;
        double const momentum =
            std::sqrt((energy - query.massGeV_) * (energy + query.massGeV_));
        double fieldSquared = 0.;
        double projected = 0.;
        for (int axis = 0; axis < 3; ++axis) {
          fieldSquared += query.fieldT_[axis] * query.fieldT_[axis];
          projected += query.particle_.direction_[axis] * query.fieldT_[axis];
        }
        double const perpendicularMomentum =
            momentum * std::sqrt(1. - projected * projected / fieldSquared);
        query.limit_.gyroradiusM_ =
            perpendicularMomentum /
            (lepton_boundary_reference::MagneticRigidityGeVPerTeslaMeter *
             std::abs(charge) * std::sqrt(fieldSquared));
        query.limit_.distanceM_ = 15.;
        query.limit_.status_ = status_type::Success;
        result.push_back(query);
      }
      return result;
    }
    if (kind != TreeKind::Ordered) {
      auto query = base_query();
      query.category_ = Category::Diagnostics;
      double const momentumGeV =
          .01 * lepton_boundary_reference::MagneticRigidityGeVPerTeslaMeter;
      query.particle_.energyGeV_ =
          std::sqrt(query.massGeV_ * query.massGeV_ + momentumGeV * momentumGeV);
      query.fieldT_[2] = 1.;
      query.limit_.status_ = status_type::Success;
      query.limit_.distanceM_ = 1.e-3;
      double const recovered = std::sqrt((query.particle_.energyGeV_ - query.massGeV_) *
                                         (query.particle_.energyGeV_ + query.massGeV_));
      query.limit_.gyroradiusM_ =
          recovered / lepton_boundary_reference::MagneticRigidityGeVPerTeslaMeter;
      result.push_back(query);
      // An adjacent linear lane must not inherit the preceding lane's flags.
      query.limit_.status_ = status_type::Linear;
      query.limit_.gyroradiusM_ = std::numeric_limits<double>::infinity();
      query.fieldT_[2] = 0.;
      result.push_back(query);
      return result;
    }

    auto query = base_query();
    result.push_back(query);
    query.limit_.distanceM_ = 2.;
    query.category_ = Category::TimeTie;
    result.push_back(query);
    query.limit_.distanceM_ = std::numeric_limits<double>::infinity();
    query.category_ = Category::Limit;
    result.push_back(query);
    query = base_query();
    query.particle_.positionM_[0] = -20.;
    query.particle_.environmentNode_ = 0;
    query.limit_.distanceM_ = 50.;
    query.category_ = Category::Outside;
    result.push_back(query);
    query = base_query();
    query.particle_.energyGeV_ = std::nextafter(query.massGeV_, 1.);
    query.category_ = Category::NearRest;
    result.push_back(query);
    for (double mass : {3., 4.}) {
      query = base_query();
      query.particle_.energyGeV_ = 5.;
      query.massGeV_ = mass;
      query.category_ = Category::Mass;
      // Analytic kinematics probes of the supplied mass argument, not PID lookup.
      result.push_back(query);
    }
    for (double charge : {-1., 0., 2.}) {
      query = base_query();
      query.chargeNumber_ = charge;
      query.category_ = Category::Charge;
      result.push_back(query);
    }
    query = base_query();
    query.category_ = Category::ZeroField;
    result.push_back(query);
    query.fieldT_[0] = 5.e-5;
    query.category_ = Category::Parallel;
    result.push_back(query);
    query = base_query();
    query.pointM_[0] = 0.;
    query.category_ = Category::ObservationZero;
    result.push_back(query);
    query.pointM_[0] = -3.;
    query.category_ = Category::ObservationBehind;
    result.push_back(query);
    query = base_query();
    query.pointM_[0] = 0.;
    query.pointM_[1] = 1.;
    query.normal_[0] = 0.;
    query.normal_[1] = 1.;
    query.chargeNumber_ = 0.;
    query.category_ = Category::ObservationParallel;
    result.push_back(query);
    query = base_query();
    query.category_ = Category::ObservationBeyond;
    result.push_back(query);
    query.pointM_[0] = query.limit_.distanceM_;
    query.category_ = Category::ObservationEqual;
    result.push_back(query);
    query = base_query();
    query.particle_.energyGeV_ = 1.e18;
    query.fieldT_[2] = 1.e8;
    double const momentumGeV = std::sqrt((query.particle_.energyGeV_ - query.massGeV_) *
                                         (query.particle_.energyGeV_ + query.massGeV_));
    query.limit_.gyroradiusM_ =
        momentumGeV /
        (lepton_boundary_reference::MagneticRigidityGeVPerTeslaMeter * query.fieldT_[2]);
    query.pointM_[0] = 1.;
    query.normal_[0] = query.normal_[1] = std::sqrt(.5);
    query.category_ = Category::ForcedCurved;
    // Finite numerical stress, not a physical atmospheric field. The native
    // large-radius limit is linear, but the independent observation query is not.
    result.push_back(query);
    query.pointM_[0] = 0.;
    query.pointM_[1] = 1.;
    query.normal_[0] = 0.;
    query.normal_[1] = 1.;
    query.category_ = Category::NoRealPlaneRoots;
    result.push_back(query);

    for (double energy : {.001, .1, 10., 1.e6}) {
      for (double charge : {-1., 0., 1.}) {
        for (double directionY : {0., .8}) {
          for (double limit : {.1, .3, 1.}) {
            query = base_query();
            query.particle_.energyGeV_ = energy;
            query.particle_.direction_[0] = directionY == 0. ? 1. : .6;
            query.particle_.direction_[1] = directionY;
            query.chargeNumber_ = charge;
            query.fieldT_[2] = 5.e-5;
            query.limit_.distanceM_ = limit;
            if (charge != 0.) {
              double const momentum =
                  std::sqrt((energy - query.massGeV_) * (energy + query.massGeV_));
              query.limit_.gyroradiusM_ =
                  momentum /
                  (lepton_boundary_reference::MagneticRigidityGeVPerTeslaMeter *
                   std::abs(charge) * query.fieldT_[2]);
              query.limit_.status_ = query.limit_.gyroradiusM_ > 1.e9
                                         ? status_type::Linear
                                         : status_type::Success;
            }
            query.category_ = Category::Scan;
            result.push_back(query);
          }
        }
      }
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION bool same_input(Query const& first, Query const& second) {
    auto const& a = first.particle_;
    auto const& b = second.particle_;
    bool same = a.pid_ == b.pid_ && a.mediumId_ == b.mediumId_ &&
                a.generation_ == b.generation_ && a.reserved_ == b.reserved_ &&
                a.energyGeV_ == b.energyGeV_ && a.timeS_ == b.timeS_ &&
                a.weight_ == b.weight_ && a.historyId_ == b.historyId_ &&
                a.parentHistoryId_ == b.parentHistoryId_ && a.stepId_ == b.stepId_ &&
                a.environmentNode_ == b.environmentNode_ &&
                first.massGeV_ == second.massGeV_ &&
                first.chargeNumber_ == second.chargeNumber_ &&
                first.limit_.status_ == second.limit_.status_ &&
                first.limit_.reserved_ == second.limit_.reserved_ &&
                first.limit_.distanceM_ == second.limit_.distanceM_ &&
                first.limit_.gyroradiusM_ == second.limit_.gyroradiusM_ &&
                first.category_ == second.category_;
    for (int axis = 0; axis < 3; ++axis) {
      same = same && a.positionM_[axis] == b.positionM_[axis] &&
             a.direction_[axis] == b.direction_[axis] &&
             first.fieldT_[axis] == second.fieldT_[axis] &&
             first.pointM_[axis] == second.pointM_[axis] &&
             first.normal_[axis] == second.normal_[axis];
    }
    return same;
  }

  KOKKOS_INLINE_FUNCTION Outcome expected(straight_boundary_reference::Tree tree,
                                          Query const& query) {
    auto access = lepton_boundary_reference::lepton_access(
        tree, query.particle_, query.massGeV_, query.chargeNumber_, query.fieldT_,
        query.limit_);
    bool const fresh = !access.unsupportedShape_ && !access.incompleteRoots_;
    auto const selection =
        boundary_reference::next(access, query.particle_.environmentNode_,
                                 query.limit_.distanceM_ / access.speed_);
    auto const boundary = lepton_boundary_reference::lepton_boundary(
        tree, query.particle_, query.massGeV_, query.chargeNumber_, query.fieldT_,
        query.limit_);
    auto const observation = lepton_boundary_reference::lepton_observation_distance(
        tree, query.particle_, query.massGeV_, query.chargeNumber_, query.fieldT_,
        query.limit_, query.pointM_, query.normal_);
    auto unassignedParticle = query.particle_;
    unassignedParticle.environmentNode_ = -3;
    straight_boundary_reference::Tree const empty;
    auto const emptyObservation = lepton_boundary_reference::lepton_observation_distance(
        empty, unassignedParticle, query.massGeV_, query.chargeNumber_, query.fieldT_,
        query.limit_, query.pointM_, query.normal_);
    return {
        access.speed_,
        boundary.distanceM_,
        observation,
        emptyObservation,
        boundary.nextNode_,
        boundary.failed_,
        access.unsupportedShape_,
        access.incompleteRoots_,
        fresh,
        boundary.distanceM_ == selection.parameter_ * access.speed_ &&
            boundary.nextNode_ == selection.node_ &&
            boundary.failed_ == (access.unsupportedShape_ || access.incompleteRoots_)};
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION Outcome
  actual(corsika::kokkos::EnvironmentTreeView<TMemorySpace> const& tree,
         Query const& query) {
    auto access =
        corsika::kokkos::lepton_access(tree, query.particle_, query.massGeV_,
                                       query.chargeNumber_, query.fieldT_, query.limit_);
    bool const fresh = !access.hasUnsupportedShape() && !access.hasIncompleteRoots();
    auto const selection = corsika::kokkos::next_environment_boundary(
        access, query.particle_.environmentNode_,
        query.limit_.distanceM_ / access.getSpeedMPerS());
    auto const boundary = corsika::kokkos::lepton_boundary(
        tree, query.particle_, query.massGeV_, query.chargeNumber_, query.fieldT_,
        query.limit_);
    auto const observation = corsika::kokkos::lepton_observation_distance(
        tree, query.particle_, query.massGeV_, query.chargeNumber_, query.fieldT_,
        query.limit_, query.pointM_, query.normal_);
    auto unassignedParticle = query.particle_;
    unassignedParticle.environmentNode_ = -3;
    corsika::kokkos::EnvironmentTreeView<TMemorySpace> const empty;
    auto const emptyObservation = corsika::kokkos::lepton_observation_distance(
        empty, unassignedParticle, query.massGeV_, query.chargeNumber_, query.fieldT_,
        query.limit_, query.pointM_, query.normal_);
    return {access.getSpeedMPerS(),
            boundary.distanceM_,
            observation,
            emptyObservation,
            boundary.nextNode_,
            boundary.failed_,
            access.hasUnsupportedShape(),
            access.hasIncompleteRoots(),
            fresh,
            boundary.distanceM_ == selection.getParameter() * access.getSpeedMPerS() &&
                boundary.nextNode_ == selection.getNode() &&
                boundary.failed_ ==
                    (access.hasUnsupportedShape() || access.hasIncompleteRoots())};
  }

} // namespace lepton_boundary_test
