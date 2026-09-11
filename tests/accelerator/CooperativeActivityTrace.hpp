#pragma once
#include <cstdint>
#include <utility>
#include <vector>

// Optional diagnostic instrumentation only. Never linked into air/mountain.
namespace c8_overlap_trace {
  void start(char const* kernel_name_substring = "Arithmetic");
  std::uint64_t timestamp();
  void finish(std::uint64_t host_start, std::uint64_t host_end);
  void finish(std::vector<std::pair<std::uint64_t,std::uint64_t>> const& host_windows);
}
