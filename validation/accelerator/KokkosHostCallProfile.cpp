// Diagnostic-only Kokkos Tools plugin; never linked into a production target.
// Measures inclusive HOST callback intervals, not CUDA device kernel duration.
#include <impl/Kokkos_Profiling_C_Interface.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t maxRows = 2048, maxActive = 1024, maxCopyDepth = 64;
constexpr std::uint32_t noDevice = std::numeric_limits<std::uint32_t>::max();
using Key = std::tuple<std::string, std::uint32_t, std::string>;
struct Row {
  std::uint64_t calls = 0, bytes = 0;
  double seconds = 0, secondsSquared = 0, maximum = 0;
};
struct Active {
  Key key;
  Clock::time_point begin;
  std::uint64_t bytes;
};
std::mutex mutex;
std::map<Key, Row> rows;
std::map<std::uint64_t, Active> active;
std::atomic<std::uint64_t> errors{0};
std::uint64_t nextId = 1, runtimeVersion = 0;
bool initialized = false, enabled = false, settingsRequested = false;
unsigned sampleLog2 = 0;
std::uint64_t sampleMask = 0;
Clock::time_point started;
thread_local std::vector<std::uint64_t> copies;
thread_local std::size_t discardedCopyDepth = 0;
// Tool-only selection, independent of all shower RNGs/history/physics state.
// Each host caller owns its state; no shared hot counter/cache line.
thread_local std::uint64_t sampleRandom = 0x8f3ab65491c2e7d1ULL;

bool selected() noexcept {
  if (!sampleMask) return true;
  sampleRandom ^= sampleRandom << 13;
  sampleRandom ^= sampleRandom >> 7;
  sampleRandom ^= sampleRandom << 17;
  return (sampleRandom & sampleMask) == 0;
}

template<class F> void guarded(F&& f) noexcept {
  try { f(); } catch (...) { ++errors; }
}
std::string label(char const* value, std::size_t capacity = 512) {
  if (!value) return "<null>";
  auto n = strnlen(value, capacity);
  if (n == capacity) ++errors; // bounded memory; truncation invalidates the report
  return std::string(value, n);
}
void add(std::uint64_t& target, std::uint64_t amount) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target) ++errors;
  else target += amount;
}
Row* row(Key const& key) {
  auto it = rows.find(key);
  if (it != rows.end()) return &it->second;
  if (rows.size() == maxRows) { ++errors; return nullptr; }
  return &rows.emplace(key, Row{}).first->second;
}
void begin(char const* kind, char const* name, std::uint32_t device,
           std::uint64_t* id, std::uint64_t bytes = 0, bool preselected = false) noexcept {
  if (!id) { ++errors; return; }
  *id = 0;
  if (!preselected && !selected()) return;
  guarded([&] {
    std::lock_guard<std::mutex> lock(mutex);
    if (!enabled) return;
    Key key{kind, device, label(name)};
    if (active.size() == maxActive || nextId == std::numeric_limits<std::uint64_t>::max()) {
      ++errors; return;
    }
    if (!row(key)) return;
    auto assigned = nextId++;
    active.emplace(assigned, Active{std::move(key), Clock::now(), bytes});
    *id = assigned;
  });
}
void end(std::uint64_t id) noexcept {
  if (!id) return; // a dropped begin has already invalidated this report
  auto now = Clock::now();
  guarded([&] {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = active.find(id);
    if (it == active.end()) { ++errors; return; }
    auto& r = rows.at(it->second.key);
    auto seconds = std::chrono::duration<double>(now-it->second.begin).count();
    add(r.calls, 1); add(r.bytes, it->second.bytes);
    r.seconds += seconds;
    r.secondsSquared += seconds*seconds;
    if (seconds > r.maximum) r.maximum = seconds;
    active.erase(it);
  });
}
std::string quote(std::string const& input) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : input) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  out << '"'; return out.str();
}
char const* deviceType(std::uint32_t id) {
  if (id == noDevice) return "unspecified";
  // Kokkos 4.7.03 identifier_from_devid: type 8, device 7, instance 17 bits.
  switch (id >> 24) {
    case 0: return "Serial";
    case 1: return "OpenMP";
    case 2: return "Cuda";
    case 3: return "HIP";
    case 7: return "SYCL";
    default: return "other";
  }
}
void allocation(char const* kind, Kokkos_Profiling_SpaceHandle space,
                char const* name, std::uint64_t bytes) noexcept {
  guarded([&] {
    std::lock_guard<std::mutex> lock(mutex);
    if (!enabled) return;
    auto r = row({kind, noDevice, label(space.name, 64)+":"+label(name)});
    if (r) { add(r->calls, 1); add(r->bytes, bytes); }
  });
}
} // namespace

extern "C" {
void kokkosp_request_tool_settings(std::uint32_t, Kokkos_Tools_ToolSettings* settings) {
  // CRITICAL: Kokkos otherwise defaults to global fencing around tool calls,
  // which would serialize CUDA/OpenMP and invalidate overlap diagnostics.
  if (!settings) { ++errors; return; }
  settings->requires_global_fencing = false;
  settingsRequested = true;
}
void kokkosp_init_library(int, std::uint64_t version, std::uint32_t,
                          Kokkos_Profiling_KokkosPDeviceInfo*) {
  guarded([&] {
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) { ++errors; return; }
    initialized = true; runtimeVersion = version; started = Clock::now();
    if (auto value = std::getenv("C8_KOKKOS_HOST_PROFILE_SAMPLE_LOG2")) {
      char* last = nullptr;
      long setting = std::strtol(value, &last, 10);
      if (!*value || *last || setting < 0 || setting > 10) ++errors;
      else sampleLog2 = static_cast<unsigned>(setting);
    }
    sampleMask = (std::uint64_t{1} << sampleLog2)-1;
    enabled = version == KOKKOSP_INTERFACE_VERSION;
    if (!enabled) ++errors;
  });
}
void kokkosp_finalize_library() {
  guarded([&] {
    std::lock_guard<std::mutex> lock(mutex);
    enabled = false;
    auto path = std::getenv("C8_KOKKOS_HOST_PROFILE");
    if (!path || !*path) {
      ++errors; std::fprintf(stderr, "Host profile requires C8_KOKKOS_HOST_PROFILE\n"); return;
    }
    std::ostringstream out;
    out << std::setprecision(17) << "{\n  \"schema\": 1,\n  \"complete\": "
        << (initialized && settingsRequested && errors == 0 && active.empty() ? "true" : "false")
        << ",\n  \"runtime_interface_version\": " << runtimeVersion
        << ",\n  \"compiled_interface_version\": " << KOKKOSP_INTERFACE_VERSION
        << ",\n  \"requires_global_fencing\": false,\n  \"settings_callback_seen\": "
        << (settingsRequested ? "true" : "false")
        << ",\n  \"errors\": " << errors << ",\n  \"active_at_finalize\": " << active.size()
        << ",\n  \"sample_log2\": " << sampleLog2
        << ",\n  \"timing_sample_probability\": " << 1./(sampleMask+1)
        << ",\n  \"host_elapsed_s\": " << (initialized ? std::chrono::duration<double>(Clock::now()-started).count() : 0)
        << ",\n  \"semantics\": \"Timing rows contain sampled inclusive host API intervals, calls/bytes/max/squares describe ONLY selected calls; allocation/deallocation counts are exact. Sampling uses independent tool-only state, not shower RNG. CUDA values are NOT device kernel times. Nested calls and overlapping threads must not be summed as wall time. Allocation bytes are cumulative, NOT a leak estimate. Profiling perturbs scheduling; not a production speed benchmark.\",\n  \"rows\": [";
    bool first = true;
    for (auto const& entry : rows) {
      auto const& key = entry.first; auto const& r = entry.second;
      if (!first) out << ',';
      first = false;
      out << "\n    {\"kind\":" << quote(std::get<0>(key))
          << ",\"device_id\":" << std::get<1>(key)
          << ",\"execution_type\":" << quote(deviceType(std::get<1>(key)))
          << ",\"label\":" << quote(std::get<2>(key)) << ",\"calls\":" << r.calls
          << ",\"host_inclusive_s\":" << r.seconds << ",\"host_max_s\":" << r.maximum
          << ",\"host_seconds_squared_sum\":" << r.secondsSquared
          << ",\"bytes\":" << r.bytes << '}';
    }
    out << "\n  ]\n}\n";
    auto data = out.str();
    // Never replace a previous diagnosis. Incomplete writes are rejected by JSON validation.
    int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) { ++errors; std::perror("create host profile"); return; }
    auto written = ::write(fd, data.data(), data.size());
    if (written < 0 || static_cast<std::size_t>(written) != data.size()) std::perror("write host profile");
    if (::close(fd) != 0) std::perror("close host profile");
  });
}
#define C8_HOST_PROFILE_CALLBACK(kind) \
  void kokkosp_begin_##kind(char const* n, std::uint32_t d, std::uint64_t* id) { begin(#kind,n,d,id); } \
  void kokkosp_end_##kind(std::uint64_t id) { end(id); }
C8_HOST_PROFILE_CALLBACK(parallel_for)
C8_HOST_PROFILE_CALLBACK(parallel_scan)
C8_HOST_PROFILE_CALLBACK(parallel_reduce)
C8_HOST_PROFILE_CALLBACK(fence)
#undef C8_HOST_PROFILE_CALLBACK
void kokkosp_begin_deep_copy(Kokkos_Profiling_SpaceHandle dst, char const* dl, void const*,
                             Kokkos_Profiling_SpaceHandle src, char const* sl, void const*,
                             std::uint64_t bytes) {
  guarded([&] {
    if (discardedCopyDepth || copies.size() == maxCopyDepth) {
      ++discardedCopyDepth; ++errors; return;
    }
    if (!selected()) { copies.push_back(0); return; }
    auto n = label(src.name,64)+":"+label(sl)+" -> "+label(dst.name,64)+":"+label(dl);
    std::uint64_t id = 0;
    begin("deep_copy", n.c_str(), noDevice, &id, bytes, true);
    copies.push_back(id);
  });
}
void kokkosp_end_deep_copy() {
  if (discardedCopyDepth) { --discardedCopyDepth; return; }
  if (copies.empty()) { ++errors; return; }
  auto id = copies.back(); copies.pop_back(); end(id);
}
void kokkosp_allocate_data(Kokkos_Profiling_SpaceHandle s, char const* n, void const*, std::uint64_t b) {
  allocation("allocate",s,n,b);
}
void kokkosp_deallocate_data(Kokkos_Profiling_SpaceHandle s, char const* n, void const*, std::uint64_t b) {
  allocation("deallocate",s,n,b);
}
} // extern "C"
