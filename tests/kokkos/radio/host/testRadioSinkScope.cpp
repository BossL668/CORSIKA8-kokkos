/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/radio/ScopedRadioTrackSink.hpp>
#include <catch2/catch_all.hpp>
#include <vector>
namespace {
  struct Sink : corsika::radio::RadioTrackSink {
    std::size_t count_{};
    void append(corsika::radio::RadioTrackSegment const&) override { ++count_; }
  };
  struct Process {
    corsika::radio::RadioTrackSink* sink_{};
    bool capture_{};
    bool throwOnAttach_{};
    std::vector<int> observers_;
    bool hasTrackSink() const { return sink_ != nullptr; }
    auto const& observerCollection() const { return observers_; }
    void setTrackSink(corsika::radio::RadioTrackSink* sink, int, bool capture = true) {
      if (sink && throwOnAttach_) throw std::runtime_error("second attach failed");
      sink_ = sink;
      capture_ = capture;
    }
    void step() {
      if (sink_ && capture_ && !observers_.empty()) sink_->append({});
    }
  };
} // namespace
TEST_CASE("Radio scope handles all model combinations and failed second attachment",
          "[kokkos][radio][host]") {
  using corsika::kokkos::radio::ScopedRadioTrackSink;
  for (int models = 0; models < 4; ++models) {
    Process a, b;
    Sink sink;
    a.observers_.resize(models & 1);
    b.observers_.resize((models >> 1) & 1);
    {
      ScopedRadioTrackSink scope(a, b, sink, 0);
      a.step();
      b.step();
      REQUIRE(sink.count_ == static_cast<std::size_t>(models != 0));
      REQUIRE_THROWS_AS((ScopedRadioTrackSink(a, b, sink, 0)), std::logic_error);
    }
    REQUIRE_FALSE(a.hasTrackSink());
    REQUIRE_FALSE(b.hasTrackSink());
  }
  Process a, b;
  Sink sink;
  b.throwOnAttach_ = true;
  REQUIRE_THROWS_AS((ScopedRadioTrackSink(a, b, sink, 0)), std::runtime_error);
  REQUIRE_FALSE(a.hasTrackSink());
  REQUIRE_FALSE(b.hasTrackSink());
}
