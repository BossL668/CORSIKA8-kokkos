/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/application/ExecutionSelection.hpp>
#include <corsika/kokkos/application/cascade/CascadeSupport.hpp>
#include "Check.hpp"
#include <iostream>

namespace {
  namespace application = corsika::kokkos::application;
  struct Arguments {
    std::vector<std::string> values_;
    std::vector<char*> pointers_;
    explicit Arguments(std::vector<std::string> values)
        : values_(std::move(values)) {
      values_.insert(values_.begin(), "c8_air_shower");
      for (auto& value : values_) pointers_.push_back(value.data());
    }
    int size() const { return static_cast<int>(pointers_.size()); }
    char** data() { return pointers_.data(); }
  };

  int directCalls = 0;
  std::vector<std::string> directDevices;
  int direct_run(int, char**, application::DeviceSelection const& devices) {
    ++directCalls;
    directDevices = devices.ids_;
    return 17;
  }

  void direct_case(std::vector<std::string> values) {
    Arguments arguments(std::move(values));
    auto const selected =
        application::select_execution(arguments.size(), arguments.data());
    CHECK(!selected.isCoordinated());
    CHECK(application::dispatch_cascade(arguments.size(), arguments.data(), direct_run) ==
          17);
    CHECK(directDevices == selected.devices().ids_);
    bool rejected = false;
    try {
      selected.runCoordinator(arguments.size(), arguments.data());
    } catch (std::logic_error const&) { rejected = true; }
    CHECK(rejected);
  }
} // namespace

int main() {
  try {
    direct_case({});
    direct_case({"--em-backend", "proposal"});
    direct_case({"--kokkos-execution", "openmp"});
    direct_case({"--device", "0"});
    direct_case({"--help", "--device", "invalid"});
    CHECK(directCalls == 5);
    for (auto const& values : {std::vector<std::string>{"--device"},
                               std::vector<std::string>{"--device=bad"}}) {
      Arguments arguments(values);
      bool rejected = false;
      try {
        application::select_execution(arguments.size(), arguments.data());
      } catch (application::CascadeAborted const&) { rejected = true; }
      CHECK(rejected);
      CHECK(application::dispatch_cascade(arguments.size(), arguments.data(),
                                          direct_run) == EXIT_FAILURE);
      CHECK(directCalls == 5);
    }
#ifdef CORSIKA_KOKKOS_COORDINATOR
    for (auto const& values : {std::vector<std::string>{"--device=0,1"},
                               std::vector<std::string>{"--prefix-workers=16"}}) {
      Arguments arguments(values);
      auto const selected =
          application::select_execution(arguments.size(), arguments.data());
      CHECK(selected.isCoordinated());
    }
#endif
    std::cout << "Execution selection checks: " << checks << '\n';
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
