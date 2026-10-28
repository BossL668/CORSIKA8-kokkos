/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/application/cascade/CascadeBindings.hpp>
#include <corsika/kokkos/application/cascade/CascadeContext.hpp>
#include "applications/detail/air_shower_kokkos/KokkosAirShowerBindings.hpp"
#include "applications/detail/air_shower_kokkos/KokkosAirShowerContext.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
  namespace current = corsika::kokkos::application;
  namespace original = corsika::applications::air_shower;
  unsigned checks{};
  void require(bool value) {
    ++checks;
    if (!value) throw std::runtime_error("Application binding assertion failed");
  }

  struct Marker {
    int id_{};
    Marker() = default;
    Marker(Marker const&) = delete;
    Marker& operator=(Marker const&) = delete;
  };
  struct Counter {
    std::string name_;
    std::vector<std::string>& calls_;
    unsigned value_;
    std::vector<int> timings_;
    unsigned getCount() const {
      calls_.push_back(name_ + ".count");
      return value_;
    }
    std::vector<int> const& getTimingSamples() const {
      calls_.push_back(name_ + ".timings");
      return timings_;
    }
  };
  struct Pool {
    std::vector<std::string>& calls_;
    unsigned value_;
    unsigned statistics() const {
      calls_.push_back("pool");
      return value_;
    }
  };

  void bindings() {
    Marker role;
    current::CascadeModels models{role, role, role, role, role, role,
                                  role, role, role, role, role};
    auto refs = std::tie(
        models.coreas_, models.zhs_, models.emCascade_, models.emContinuousProposal_,
        models.neutrinoPrimary_, models.hadronSequence_, models.decaySequence_,
        models.emContinuous_, models.longitudinal_, models.production_, models.cut_);
    std::apply([&](auto&... items) { (require(&items == &role), ...); }, refs);
    std::vector<std::string> calls;
    Counter high{"high", calls, 17, {1, 2, 3}}, low{"low", calls, 23, {1, 2}};
    Pool pool{calls, 29};
    unsigned photoHigh = 31, photoLow = 37, beforeHigh = 41, beforeLow = 43;
    Marker fallback;
    original::KokkosAirShowerMonitors before{photoHigh, photoLow, high,
                                             low,       &pool,    fallback};
    current::CascadeMonitors after{
        photoHigh, photoLow, high, low, boost::optional<Pool&>(pool), fallback};
    auto a = before.capture(beforeHigh, beforeLow);
    auto const previousCalls = calls;
    calls.clear();
    auto b = after.capture(beforeHigh, beforeLow);
    require(calls == previousCalls);
    require(calls == std::vector<std::string>{"low.count", "high.count", "low.timings",
                                              "high.timings", "pool"});
    require(std::tie(a.photo_high, a.photo_low, a.low_interactions, a.high_interactions,
                     a.low_timings, a.high_timings, a.pool) ==
            std::tie(b.photoHigh_, b.photoLow_, b.lowInteractions_, b.highInteractions_,
                     b.lowTimings_, b.highTimings_, b.pool_));
    beforeHigh = beforeLow = high.value_ = low.value_ = pool.value_ = 99;
    require(b.photoHigh_ == 41 && b.photoLow_ == 43 && b.lowInteractions_ == 23 &&
            b.highInteractions_ == 17 && b.pool_ == 29);
    after.pool_ = boost::none;
    before.pool = nullptr;
    calls.clear();
    auto c = before.capture(beforeHigh, beforeLow);
    auto const nullCalls = calls;
    calls.clear();
    auto d = after.capture(beforeHigh, beforeLow);
    require(c.pool == 0 && d.pool_ == 0 && calls == nullCalls && calls.size() == 4);
    std::cout << "capture low/high count/timing order preserved\n";
  }

  struct Cascade {
    std::string calls_;
    void forceInteraction() { calls_ += "interaction,"; }
    void forceDecay() { calls_ += "decay,"; }
  };
  void cascade() {
    for (unsigned flags = 0; flags < 4; ++flags) {
      Cascade oldCascade, newCascade;
      auto oldRun = [&](auto& configure) {
        configure(oldCascade);
        oldCascade.calls_ += "run;";
      };
      auto newRun = [&](auto& configure) {
        configure(newCascade);
        newCascade.calls_ += "run;";
      };
      original::KokkosCascade before{oldRun};
      current::CascadeAdapter after{newRun};
      if (flags & 1) {
        before.forceInteraction();
        after.forceInteraction();
      }
      if (flags & 2) {
        before.forceDecay();
        after.forceDecay();
      }
      for (int repeat = 0; repeat < 3; ++repeat) {
        before.run();
        after.run();
        require(oldCascade.calls_ == newCascade.calls_);
      }
      std::cout << flags << ' ' << newCascade.calls_ << '\n';
    }
    auto failure = [](auto&) { throw std::domain_error("event failure"); };
    original::KokkosCascade before{failure};
    current::CascadeAdapter after{failure};
    for (auto* name : {"original", "current"}) {
      try {
        if (std::string(name) == "original")
          before.run();
        else
          after.run();
        throw std::runtime_error("Exception was swallowed");
      } catch (std::domain_error const& error) {
        require(std::string(error.what()) == "event failure");
      }
    }
  }

  std::string describe(Marker const& marker) { return "M" + std::to_string(marker.id_); }
  std::string describe(Marker* marker) { return "P" + std::to_string(marker->id_); }
  std::string describe(std::array<double, 3> const& value) {
    return "array:" + std::to_string(value[0]) + "," + std::to_string(value[1]) + "," +
           std::to_string(value[2]);
  }
  template <typename T>
  std::string describe(T const& value) {
    std::ostringstream text;
    text << value;
    return text.str();
  }
  struct ReferenceResult {
    template <typename T, typename... TRest>
    T& operator()(T& first, TRest&&...) && {
      return first;
    }
    ReferenceResult() = default;
    ReferenceResult(ReferenceResult const&) = delete;
  };
  void context() {
    std::array<Marker, 41> m;
    for (int i = 0; i < 41; ++i) m[i].id_ = i;
    std::array<double, 3> magnetic{{.1, .2, .3}};
    current::CascadeContext after{
        current::CascadeGeometry{m[0], m[1], m[2], m[3], m[4], m[5], m[6], 1.25, 2.5,
                                 3.75, magnetic},
        current::CascadeOutputs{m[7], m[8], m[9], m[10], m[11], m[12], m[13]},
        current::CascadeProcesses{m[14], m[15], m[16], m[17], m[18], m[19], m[20], m[21],
                                  m[22], m[23], m[24], m[25], m[26]},
        current::CascadeExecution{m[27], m[28], m[29], m[30],
                                  boost::optional<Marker&>(m[31]), m[32]},
        current::CascadeDiagnostics{m[33], m[34], m[35], m[36], m[37], m[38], 701, 702,
                                    703, 704, m[39], m[40]}};
    original::KokkosAirShowerContext before{
        original::KokkosAirShowerGeometry{m[0], m[1], m[2], m[3], m[4], m[5], m[6], 1.25,
                                          2.5, 3.75, magnetic},
        original::KokkosAirShowerOutputs{m[7], m[8], m[9], m[10], m[11], m[12], m[13]},
        original::KokkosAirShowerProcesses{m[14], m[15], m[16], m[17], m[18], m[19],
                                           m[20], m[21], m[22], m[23], m[24], m[25],
                                           m[26]},
        original::KokkosAirShowerExecution{m[27], m[28], m[29], m[30], &m[31], m[32]},
        original::KokkosAirShowerDiagnostics{m[33], m[34], m[35], m[36], m[37], m[38],
                                             701, 702, 703, 704, m[39], m[40]}};
    auto serialize = [](auto&&... values) {
      return std::vector<std::string>{describe(values)...};
    };
    auto a = original::detail::withKokkosAirShowerArguments(before, serialize);
    auto b = current::detail::with_cascade_arguments(after, serialize);
    require(a == b && b.size() == 49);
    std::vector<std::string> expected{"M0",
                                      "M1",
                                      "M2",
                                      "M3",
                                      "M4",
                                      "M12",
                                      "M13",
                                      "M5",
                                      "M6",
                                      "1.25",
                                      "2.5",
                                      "3.75",
                                      "array:0.100000,0.200000,0.300000",
                                      "M7",
                                      "M8",
                                      "M9",
                                      "M10",
                                      "M11",
                                      "M14",
                                      "M15",
                                      "M16",
                                      "M17",
                                      "M18",
                                      "M19",
                                      "M20",
                                      "M21",
                                      "M22",
                                      "M23",
                                      "M24",
                                      "M25",
                                      "M26",
                                      "M27",
                                      "M28",
                                      "M29",
                                      "M30",
                                      "P31",
                                      "M32",
                                      "M33",
                                      "M34",
                                      "M35",
                                      "M36",
                                      "M37",
                                      "M38",
                                      "701",
                                      "702",
                                      "703",
                                      "704",
                                      "M39",
                                      "M40"};
    for (unsigned i = 0; i < b.size(); ++i) {
      require(b[i] == expected[i]);
      std::cout << i << ' ' << b[i] << '\n';
    }
    auto& oldResult =
        original::detail::withKokkosAirShowerArguments(before, ReferenceResult{});
    auto& newResult = current::detail::with_cascade_arguments(after, ReferenceResult{});
    require(&oldResult == &m[0] && &newResult == &m[0]);
    newResult.id_ = 88;
    require(before.geometry.environment.id_ == 88 &&
            after.geometry_.environment_.id_ == 88);
    magnetic[0] = 999.;
    require(after.geometry_.magneticFieldT_[0] == .1 &&
            before.geometry.magnetic_field_T[0] == .1);
    auto failure = [](auto&&...) -> int {
      throw std::domain_error("forwarding failure");
    };
    try {
      current::detail::with_cascade_arguments(after, failure);
      throw std::runtime_error("Swallowed failure");
    } catch (std::domain_error const& error) {
      require(std::string(error.what()) == "forwarding failure");
    }
  }
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2);
    if (std::string(argv[1]) == "bindings")
      bindings();
    else if (std::string(argv[1]) == "cascade")
      cascade();
    else if (std::string(argv[1]) == "context")
      context();
    else
      throw std::invalid_argument("Unknown scenario");
    std::cout << "CHECKS=" << checks << '\n';
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
