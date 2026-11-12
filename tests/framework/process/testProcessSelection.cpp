/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/process/ProcessSelection.hpp>
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/framework/process/SwitchProcessSequence.hpp>
#include <corsika/modules/writers/SubWriter.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>

namespace {
  template <int Index>
  struct Process : corsika::NullModel {
    std::unique_ptr<int> value_ = std::make_unique<int>(Index);
  };

  template <typename T>
  struct First : std::is_same<T, Process<1>> {};
  template <typename T>
  struct Second : std::is_same<T, Process<2>> {};
  template <typename T>
  struct AnyProcess : std::is_base_of<corsika::NullModel, T> {};
  template <typename T>
  struct Switch : std::bool_constant<T::is_switch_process_sequence> {};

  struct Condition {
    int& calls_;
    template <typename T>
    bool operator()(T const&) const {
      ++calls_;
      return false;
    }
  };
} // namespace

TEST_CASE("Process selection borrows noncopyable models in sequence order",
          "[process_selection]") {
  using namespace corsika;
  Process<1> first;
  Process<2> second;
  Process<3> third;
  auto nested = make_sequence(first, second);
  auto sequence = make_sequence(third, nested);
  auto matches = matching_processes<AnyProcess>(sequence);
  STATIC_REQUIRE(std::tuple_size_v<decltype(matches)> == 3);
  CHECK(&std::get<0>(matches) == &third);
  CHECK(&std::get<1>(matches) == &first);
  CHECK(&std::get<2>(matches) == &second);
  CHECK(&select_process<First>(sequence) == &first);
  CHECK(&select_process<Second>(sequence) == &second);
  *select_process<First>(sequence).value_ = 42;
  CHECK(*first.value_ == 42);

  auto reordered = make_sequence(second, third, first);
  CHECK(&select_process<First>(reordered) == &first);
  CHECK(&select_process<Second>(reordered) == &second);
  CHECK(&std::get<1>(sequence.getProcesses()) == &nested);
}

TEST_CASE("Inspecting a switch does not execute or flatten it", "[process_selection]") {
  using namespace corsika;
  Process<1> first;
  Process<2> second;
  Process<3> third;
  int calls = 0;
  auto branch = make_select(Condition{calls}, first, second);
  auto sequence = make_sequence(branch, third);
  CHECK(&select_process<Switch>(sequence) == &branch);
  CHECK(&select_process<First>(sequence) == &first);
  CHECK(&select_process<Second>(sequence) == &second);
  CHECK(calls == 0);
}

TEST_CASE("Missing and ambiguous process roles remain distinguishable",
          "[process_selection]") {
  using namespace corsika;
  Process<1> first, duplicate;
  Process<3> third;
  auto sequence = make_sequence(first, third, duplicate);
  STATIC_REQUIRE(contains_process_v<First, decltype(sequence)>);
  STATIC_REQUIRE_FALSE(contains_process_v<Second, decltype(sequence)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(matching_processes<First>(sequence))> == 2);
  STATIC_REQUIRE(std::tuple_size_v<decltype(matching_processes<Second>(sequence))> == 0);
}

TEST_CASE("SubWriter exposes its existing writer without copying it",
          "[process_selection]") {
  struct Writer {
    std::unique_ptr<int> value_ = std::make_unique<int>(0);
    void write(int value) { *value_ = value; }
  } writer;
  struct WrittenProcess : corsika::SubWriter<Writer> {
    using corsika::SubWriter<Writer>::SubWriter;
    YAML::Node getConfig() const override { return {}; }
  } process(writer);
  CHECK(&process.getWriter() == &writer);
  auto const& constant = process;
  STATIC_REQUIRE(std::is_same_v<decltype(constant.getWriter()), Writer const&>);
  CHECK(&constant.getWriter() == &writer);
  process.getWriter().write(17);
  CHECK(*writer.value_ == 17);
}
