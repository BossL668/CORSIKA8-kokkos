/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <tests/kokkos/environment/LeptonBorrowedTransportChecks.hpp>
#include <tests/kokkos/environment/PhotonBorrowedTransportChecks.hpp>
#include <corsika/kokkos/environment/EnvironmentBorrowedView.hpp>
#include <corsika/kokkos/environment/EnvironmentStorage.hpp>

namespace {
  namespace cp = corsika::kokkos;
  namespace fixture = environment_query_test;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;

  void ensure_runtime() { static cp::Runtime runtime; }

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
} // namespace

TEST_CASE("Borrowed atmosphere retains native-table lepton transport",
          "[kokkos][environment][borrowed][transport]") {
  ensure_runtime();
  namespace lt = lepton_transport_test;
  auto data = fixture::atmosphere();
  auto tables = continuous_test::make_dataset();
  cp::NativeTableStorage<execution_space_type> tableStorage;
  tableStorage.initialize(tables.table_);
  native_query_test::UploadedReference<memory_space_type> oldTables(tables.table_);
  auto native = tableStorage.getDeviceView();
  auto oldNative = oldTables.view();
  cp::EnvironmentStorage<execution_space_type> owner(borrow(data));
  auto managed = owner.getView();
  Kokkos::View<cp::EnvironmentBorrowedView, memory_space_type> context(
      "environment context");
  Kokkos::deep_copy(context, cp::EnvironmentBorrowedView(owner.getDataView()));
  fixture::Uploaded<memory_space_type> referenceOwner(data);
  auto reference = referenceOwner.referenceView();

  std::vector<lt::Input> inputs;
  for (auto const& query : data.queries_) {
    for (int pid : {11, -11, 13, -13}) {
      for (double energy : {1000., 10000., 100000.}) {
        auto input = lt::input(query, pid, energy);
        // Use real loss/geometry paths, plus a distinct external limit.
        inputs.push_back(input);
        input.external_.enabled_ = true;
        input.external_.distanceM_ = .1;
        inputs.push_back(input);
      }
    }
  }
  auto requests = transport_test::upload(inputs);
  Kokkos::View<lt::Result * [2], Kokkos::LayoutRight, memory_space_type> results(
      "borrowed lepton steps", inputs.size());
  Kokkos::View<lt::Expected*, memory_space_type> expected("production lepton steps",
                                                          inputs.size());
  Kokkos::parallel_for(
      "borrowed lepton transport",
      Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
      KOKKOS_LAMBDA(std::size_t i) {
        results(i, 0) = lt::actual(native, context(), requests(i));
        results(i, 1) = lt::actual(native, managed, requests(i));
        expected(i) = lt::expected(oldNative, reference, requests(i));
      });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
  auto oracle = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected);
  std::size_t advanced{};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    CAPTURE(i, inputs[i].interaction_.particle_.pid_);
    lt::compare(host(i, 0), oracle(i));
    lt::compare(host(i, 1), oracle(i));
    if (host(i, 0).state_ == 0 && host(i, 0).record_.distanceM_ > 0.) {
      ++advanced;
      REQUIRE(std::isfinite(host(i, 0).record_.end_.energyGeV_));
    }
  }
  REQUIRE(inputs.size() == 360);
  REQUIRE(advanced > 0);
  std::cout << "borrowed lepton cases=" << inputs.size() << " advanced=" << advanced
            << '\n';
}

TEST_CASE("Borrowed atmosphere retains photon transport",
          "[kokkos][environment][borrowed][transport]") {
  ensure_runtime();
  namespace pt = photon_transport_test;
  auto data = fixture::atmosphere();
  cp::EnvironmentStorage<execution_space_type> owner(borrow(data));
  auto managed = owner.getView();
  Kokkos::View<cp::EnvironmentBorrowedView, memory_space_type> context("photon context");
  Kokkos::deep_copy(context, cp::EnvironmentBorrowedView(owner.getDataView()));
  fixture::Uploaded<memory_space_type> referenceOwner(data);
  auto reference = referenceOwner.referenceView();
  std::vector<pt::Input> inputs;
  for (auto const& query : data.queries_) {
    auto input = pt::input(query);
    inputs.push_back(input);
    input.external_.enabled_ = true;
    input.external_.distanceM_ = .1;
    inputs.push_back(input);
  }
  auto requests = transport_test::upload(inputs);
  Kokkos::View<cp::PhotonTransportOutcome * [2], Kokkos::LayoutRight, memory_space_type>
      results("borrowed photon steps", inputs.size());
  Kokkos::View<pt::Result*, memory_space_type> expected("production photon steps",
                                                        inputs.size());
  Kokkos::parallel_for(
      "borrowed photon transport",
      Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
      KOKKOS_LAMBDA(std::size_t i) {
        auto const& request = requests(i);
        results(i, 0) = cp::transport_photon(context(), request.config_,
                                             request.interaction_, request.external_);
        results(i, 1) = cp::transport_photon(managed, request.config_,
                                             request.interaction_, request.external_);
        expected(i) = pt::capture(pt::expected(reference, request));
      });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
  auto oracle = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected);
  std::size_t advanced{};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    CAPTURE(i);
    pt::compare(pt::capture(host(i, 0)), oracle(i));
    pt::compare(pt::capture(host(i, 1)), oracle(i));
    if (!host(i, 0).fallbackFlag_ && host(i, 0).record_.distanceM_ > 0.) ++advanced;
  }
  REQUIRE(inputs.size() == 30);
  REQUIRE(advanced > 0);
  std::cout << "borrowed photon cases=" << inputs.size() << " advanced=" << advanced
            << '\n';
}
