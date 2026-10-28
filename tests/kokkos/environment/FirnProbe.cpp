/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "FirnProbe.hpp"
#include <corsika/kokkos/environment/EnvironmentStorage.hpp>
#include <corsika/kokkos/environment/EnvironmentBorrowedView.hpp>
#include <corsika/kokkos/environment/EnvironmentQuery.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

namespace firn_test {
  static_assert(static_cast<int>(Status::Success) ==
                static_cast<int>(corsika::kokkos::EnvironmentStatus::Success));
  static_assert(static_cast<int>(Status::InvalidSnapshot) ==
                static_cast<int>(corsika::kokkos::EnvironmentStatus::InvalidSnapshot));
  static_assert(
      static_cast<int>(Status::NoForwardIntersection) ==
      static_cast<int>(corsika::kokkos::EnvironmentStatus::NoForwardIntersection));
  static_assert(static_cast<int>(Status::GrammageOutOfRange) ==
                static_cast<int>(corsika::kokkos::EnvironmentStatus::GrammageOutOfRange));

  char const* execution_name() { return Kokkos::DefaultExecutionSpace::name(); }
  std::vector<Result> probe(corsika::kokkos::EnvironmentDataView data,
                            std::vector<Query> const& queries) {
    namespace cp = corsika::kokkos;
    static cp::Runtime runtime;
    using execution_type = Kokkos::DefaultExecutionSpace;
    cp::EnvironmentStorage<execution_type> storage(data);
    cp::EnvironmentBorrowedView const environment(storage.getDataView());
    Kokkos::View<Query*> input("firn queries", queries.size());
    auto host = Kokkos::create_mirror_view(input);
    for (std::size_t index = 0; index < queries.size(); ++index) {
      host(index) = queries[index];
    }
    Kokkos::deep_copy(input, host);
    Kokkos::View<Result*> output("firn results", queries.size());
    Kokkos::parallel_for(
        "firn snapshot queries", queries.size(), KOKKOS_LAMBDA(int index) {
          auto const& query = input(index);
          auto const& model = *environment.getModel(1);
          auto& result = output(index);
          result.values_[0] = cp::density(model, query.positionM_);
          result.values_[1] = cp::refractive_index(model, query.positionM_);
          result.values_[2] =
              cp::grammage(model, query.positionM_, query.direction_, query.lengthM_);
          result.values_[3] = cp::arclength(model, query.positionM_, query.direction_,
                                            query.grammageGPerCm2_);
          result.values_[4] = cp::locate_node(environment.getTree(), query.positionM_);
          auto const boundary = cp::distance_to_environment_boundary(
              environment, {}, query.positionM_, query.direction_);
          result.values_[5] = boundary.distanceM_;
          result.values_[6] = model.mediumId_;
          result.statuses_[0] =
              static_cast<int>(cp::query_environment(environment, {}, query.positionM_,
                                                     query.direction_, 1)
                                   .status_);
          result.statuses_[1] = static_cast<int>(
              cp::environment_grammage(environment, {}, 1, query.positionM_,
                                       query.direction_, query.lengthM_)
                  .status_);
          result.statuses_[2] = static_cast<int>(
              cp::environment_arclength(environment, {}, 1, query.positionM_,
                                        query.direction_, query.grammageGPerCm2_)
                  .status_);
          result.statuses_[3] = static_cast<int>(boundary.status_);
        });
    auto results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
    std::vector<Result> result(queries.size());
    for (std::size_t index = 0; index < result.size(); ++index) {
      result[index] = results(index);
    }
    return result;
  }
} // namespace firn_test
