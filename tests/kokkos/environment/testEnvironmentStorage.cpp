/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentStorage.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <tests/kokkos/environment/EnvironmentQueryTestData.hpp>

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <utility>
#include <vector>

namespace {
  namespace cp = corsika::kokkos;
  namespace fixture = environment_query_test;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using storage_type = cp::EnvironmentStorage<execution_space_type>;

  void ensure_runtime() { static cp::Runtime runtime; }

  void exact(double a, double b) {
    if (std::isnan(b)) {
      REQUIRE(std::isnan(a));
    } else {
      REQUIRE(a == b);
      if (b == 0.) { REQUIRE(std::signbit(a) == std::signbit(b)); }
    }
  }

  cp::EnvironmentDataView borrow(fixture::Dataset const& data) {
    return {data.nodes_.data(),
            data.shapes_.data(),
            data.children_.data(),
            data.exclusions_.data(),
            data.models_.data(),
            data.components_.data(),
            static_cast<std::uint32_t>(data.nodes_.size()),
            static_cast<std::uint32_t>(data.children_.size()),
            static_cast<std::uint32_t>(data.exclusions_.size()),
            static_cast<std::uint32_t>(data.models_.size()),
            static_cast<std::uint32_t>(data.components_.size())};
  }

  template <typename T>
  std::vector<T> download(T const* data, std::uint32_t count) {
    using borrowed_view_type = Kokkos::View<T const*, memory_space_type,
                                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                    borrowed_view_type(data, count));
    std::vector<T> result;
    for (std::uint32_t i = 0; i < count; ++i) { result.push_back(host(i)); }
    return result;
  }

  void check_records(fixture::Dataset const& input, storage_type const& owner) {
    auto const raw = owner.getDataView();
    REQUIRE(raw.nodeCount_ == input.nodes_.size());
    REQUIRE(raw.childCount_ == input.children_.size());
    REQUIRE(raw.exclusionCount_ == input.exclusions_.size());
    REQUIRE(raw.modelCount_ == input.models_.size());
    REQUIRE(raw.componentCount_ == input.components_.size());
    fixture::Dataset recovered;
    recovered.nodes_ = download(raw.nodes_, raw.nodeCount_);
    recovered.shapes_ = download(raw.shapes_, raw.nodeCount_);
    recovered.children_ = download(raw.children_, raw.childCount_);
    recovered.exclusions_ = download(raw.exclusions_, raw.exclusionCount_);
    recovered.models_ = download(raw.models_, raw.modelCount_);
    recovered.components_ = download(raw.composition_, raw.componentCount_);
    // Reuse the accepted field-by-field checker, including unused model/shape
    // fields and signed zero. Padding bytes are deliberately not compared.
    fixture::Uploaded<Kokkos::HostSpace> recorded(recovered);
    fixture::check_storage(input, recorded, exact,
                           [](auto a, auto b) { REQUIRE(a == b); });
    auto expectedBytes =
        input.nodes_.size() * sizeof(cp::EnvironmentNode) +
        input.shapes_.size() * sizeof(cp::EnvironmentShape) +
        (input.children_.size() + input.exclusions_.size()) * sizeof(std::int32_t) +
        input.models_.size() * sizeof(cp::EnvironmentModel) +
        input.components_.size() * sizeof(cp::EnvironmentComponent);
    REQUIRE(owner.getDeviceBytes() == expectedBytes);
  }
} // namespace

TEST_CASE("Environment storage preserves exported records and production queries",
          "[kokkos][environment][storage]") {
  ensure_runtime();
  std::size_t queriesChecked{}, datasetsChecked{};
  for (auto const& data : fixture::datasets()) {
    INFO(data.name_);
    storage_type owner(borrow(data));
    check_records(data, owner);
    auto view = owner.getView();
    fixture::Uploaded<memory_space_type> oracle(data);
    auto reference = oracle.referenceView();
    auto queries = fixture::Uploaded<memory_space_type>::upload(data.queries_);
    Kokkos::View<fixture::Outcome * [2], Kokkos::LayoutRight, memory_space_type> results(
        "storage query results", data.queries_.size());
    Kokkos::parallel_for(
        "storage production query comparison",
        Kokkos::RangePolicy<execution_space_type>(0, data.queries_.size()),
        KOKKOS_LAMBDA(int i) {
          results(i, 0) = fixture::actual(view, queries(i));
          results(i, 1) = fixture::expected(reference, queries(i));
        });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    for (std::size_t i = 0; i < data.queries_.size(); ++i) {
      fixture::compare(host(i, 0), host(i, 1), exact,
                       [](auto a, auto b) { REQUIRE(a == b); });
      fixture::check_semantics(data.queries_[i], host(i, 0),
                               [](bool value) { REQUIRE(value); });
      ++queriesChecked;
    }
    ++datasetsChecked;
  }
  CAPTURE(datasetsChecked, queriesChecked);
  REQUIRE(queriesChecked == 145);
  REQUIRE(datasetsChecked > 10);
}

TEST_CASE("Environment storage lifetime and borrowed context records",
          "[kokkos][environment][storage]") {
  ensure_runtime();
  storage_type empty;
  REQUIRE(empty.getDeviceBytes() == 0);
  REQUIRE(empty.getDataView().nodeCount_ == 0);
  REQUIRE_FALSE(empty.getView().getTree().hasNodeStorage());
  auto data = fixture::simple("storage lifetime");
  auto host = borrow(data);
  for (int array = 0; array < 6; ++array) {
    auto broken = host;
    switch (array) {
      case 0:
        broken.nodes_ = nullptr;
        break;
      case 1:
        broken.shapes_ = nullptr;
        break;
      case 2:
        broken.children_ = nullptr;
        break;
      case 3:
        broken.exclusions_ = nullptr;
        broken.exclusionCount_ = 1;
        break;
      case 4:
        broken.models_ = nullptr;
        break;
      case 5:
        broken.composition_ = nullptr;
        break;
    }
    REQUIRE_THROWS_AS(storage_type(broken), std::invalid_argument);
  }
  storage_type owner(host);
  auto shared = owner;
  auto managed = owner.getView();
  auto raw = owner.getDataView();
  owner = {};
  check_records(data, shared);
  REQUIRE(shared.getDataView().nodes_ == raw.nodes_);
  auto moved = std::move(shared);
  check_records(data, moved);

  // Exercise rank-zero context copying and real device dereferencing. The
  // managed view keeps arrays alive after the explicit storage is released.
  Kokkos::View<cp::EnvironmentDataView, memory_space_type> context("environment context");
  Kokkos::deep_copy(context, raw);
  moved = {};
  Kokkos::View<double*, memory_space_type> result("borrowed environment read", 2);
  Kokkos::parallel_for(
      "read environment context", Kokkos::RangePolicy<execution_space_type>(0, 1),
      KOKKOS_LAMBDA(int) {
        auto record = context();
        auto id = record.nodes_[1].modelIndex_;
        result(0) = record.models_[id].referenceDensityGPerCm3_;
        result(1) = managed.getModel(1)->referenceDensityGPerCm3_;
      });
  auto resultHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  exact(resultHost(0), data.models_[0].referenceDensityGPerCm3_);
  exact(resultHost(1), resultHost(0));
}
