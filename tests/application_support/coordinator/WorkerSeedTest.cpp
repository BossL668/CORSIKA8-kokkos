/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/scheduling/AirShowerCoordinator.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <CLI/CLI.hpp>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
  using corsika::kokkos::coordinator::worker_seed;

  void require(bool condition, char const* message) {
    if (!condition) throw std::runtime_error(message);
  }

  struct Input {
    std::uint64_t seed_;
    unsigned index_;
  };

  void test_known_values() {
    static_assert(std::is_same_v<decltype(worker_seed(1, 0)), std::uint64_t>);
    // Independently calculated SHA256 vectors, including master seeds above 32 bits.
    std::array<Input, 5> const inputs{
        {{1, 0}, {1, 1}, {91, 1024}, {4294967299ULL, 15}, {9223372036854775806ULL, 254}}};
    std::array<std::uint64_t, 5> const expected{{8969997700718094502ULL,
                                               5438946774738908630ULL,
                                               8407057043092802244ULL,
                                               547898566085934871ULL,
                                               6178570000020906985ULL}};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto const value = worker_seed(inputs[i].seed_, inputs[i].index_);
      require(value == expected[i], "Worker seed reference vector changed");

      // Match the native application's string option followed by as<long>().
      CLI::App app;
      app.add_option("-s,--seed", "The random number seed.")
          ->default_val(0)
          ->check(CLI::NonNegativeNumber);
      std::array<std::string, 3> args{{"worker", "--seed", std::to_string(value)}};
      std::array<char*, 3> argv{{args[0].data(), args[1].data(), args[2].data()}};
      app.parse(argv.size(), argv.data());
      auto const parsed = app["--seed"]->as<long>();
      require(parsed > 0 && std::uint64_t(parsed) == value,
              "Native CLI truncated the worker seed");
      auto const json = nlohmann::json::parse(nlohmann::json(value).dump());
      require(json.get<std::uint64_t>() == value, "JSON truncated the worker seed");
    }
    for (std::size_t i = inputs.size(); i-- > 0;)
      require(worker_seed(inputs[i].seed_, inputs[i].index_) == expected[i],
              "Worker seeds depend on evaluation order");
    require(worker_seed(1, 0) != worker_seed(1 + (1ULL << 32), 0),
            "Master seed high bits were discarded");
  }

  void test_previous_collisions() {
    // All six collisions found in seeds 1..10000 with 16 prefix and 4 GPU workers.
    std::array<std::array<Input, 2>, 6> const collisions{{
        {{{2328, 1025}, {3550, 1036}}},
        {{{352, 1031}, {4517, 1024}}},
        {{{3988, 1029}, {6311, 1025}}},
        {{{7504, 1033}, {8037, 1034}}},
        {{{3527, 2}, {8150, 1032}}},
        {{{667, 1029}, {8909, 1032}}},
    }};
    for (auto const& pair : collisions) {
      auto const a = worker_seed(pair[0].seed_, pair[0].index_);
      auto const b = worker_seed(pair[1].seed_, pair[1].index_);
      require(std::uint32_t(a) == std::uint32_t(b), "Invalid old collision fixture");
      require(a != b, "A known 32-bit worker seed collision remains");
      // The actual native CPU engine must preserve the widened seed, not just the CLI.
      corsika::RNGManager<>::prng_type first(a, 0), second(b, 0), repeat(a, 0);
      unsigned different = 0;
      for (unsigned i = 0; i < 4096; ++i) {
        auto const value = first();
        require(value == repeat(), "Native stream is not reproducible");
        different += value != second();
      }
      require(different > 0, "Native CPU streams still repeat after widening seeds");
    }
  }

  void test_campaign() {
    std::set<std::uint64_t> seen;
    for (std::uint64_t seed = 1; seed <= 10000; ++seed) {
      for (unsigned worker = 0; worker < 20; ++worker) {
        unsigned const index = worker < 16 ? 1024 + worker : worker - 16;
        auto const value = worker_seed(seed, index);
        require(value > 0 && value <= std::uint64_t(std::numeric_limits<long>::max()),
                "Worker seed is outside the nonzero signed CLI range");
        require(seen.insert(value).second, "Repeated seed in regression campaign");
      }
    }
    require(seen.size() == 200000, "Incomplete seed regression campaign");
    // With these fixed inputs no worker seed may equal any bootstrap master seed.
    require(*seen.begin() > 10000, "Worker seed overlaps a bootstrap master seed");
  }
} // namespace

int main() {
  try {
    test_known_values();
    test_previous_collisions();
    test_campaign();
    std::cout << "Worker seeds: reference vectors, signed CLI/JSON round-trip, "
                 "six former collisions, native streams and 200000-seed scan passed.\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
