// Optional Kokkos Tools plugin for CUDA 12.6. Never linked into production.
// CUPTI supplies actual device execution/copy timestamps; Kokkos labels identify
// allocations and the purpose of each copy. No extra fences are requested.
#include <impl/Kokkos_Profiling_C_Interface.h>
#include <cupti.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>

namespace {
using Json = nlohmann::json;
struct State {
  std::ofstream output;
  std::mutex mutex;
  std::atomic<std::uint64_t> sequence{0}, dropped{0};
  std::atomic<bool> failed{false};
};
State& state() { static State instance; return instance; }
void emit(Json row) {
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.output.is_open()) s.output << row.dump() << '\n';
}
void check(CUptiResult result) {
  if (result == CUPTI_SUCCESS) return;
  state().failed = true;
  char const* message = "unknown CUPTI error";
  cuptiGetResultString(result, &message);
  emit({{"event", "error"}, {"message", message}});
}
std::uint64_t timestamp() {
  std::uint64_t result{};
  check(cuptiGetTimestamp(&result));
  return result;
}
void CUPTIAPI request(std::uint8_t** buffer, std::size_t* bytes, std::size_t* records) {
  *bytes = 1024 * 1024;
  *records = 0;
  *buffer = static_cast<std::uint8_t*>(std::malloc(*bytes));
  if (!*buffer) { *bytes = 0; state().failed = true; }
}
void CUPTIAPI complete(CUcontext context, std::uint32_t stream,
                        std::uint8_t* buffer, std::size_t, std::size_t valid) {
  if (valid) {
    CUpti_Activity* activity{};
    CUptiResult status;
    while ((status = cuptiActivityGetNextRecord(buffer, valid, &activity)) == CUPTI_SUCCESS) {
      if (activity->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        auto const* k = reinterpret_cast<CUpti_ActivityKernel9 const*>(activity);
        emit({{"event", "gpu_kernel"}, {"name", k->name ? k->name : ""},
              {"start", k->start}, {"end", k->end}, {"device", k->deviceId},
              {"stream", k->streamId}, {"correlation", k->correlationId},
              {"grid", {k->gridX, k->gridY, k->gridZ}},
              {"block", {k->blockX, k->blockY, k->blockZ}}});
      } else if (activity->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
        auto const* m = reinterpret_cast<CUpti_ActivityMemcpy5 const*>(activity);
        emit({{"event", "gpu_copy"}, {"kind", m->copyKind}, {"bytes", m->bytes},
              {"start", m->start}, {"end", m->end}, {"device", m->deviceId},
              {"stream", m->streamId}, {"correlation", m->correlationId},
              {"src_kind", m->srcKind}, {"dst_kind", m->dstKind}});
      } else if (activity->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
        auto const* a = reinterpret_cast<CUpti_ActivityAPI const*>(activity);
        char const* name{};
        check(cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, a->cbid, &name));
        emit({{"event", "cuda_api"}, {"name", name ? name : ""},
              {"start", a->start}, {"end", a->end}, {"correlation", a->correlationId}});
      }
    }
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) check(status);
  }
  std::size_t lost{};
  check(cuptiActivityGetNumDroppedRecords(context, stream, &lost));
  state().dropped += lost;
  std::free(buffer);
}
void begin(char const* kind, char const* name, std::uint32_t device, std::uint64_t* id) {
  *id = ++state().sequence;
  emit({{"event", kind}, {"name", name}, {"id", *id}, {"device", device},
        {"time", timestamp()}});
}
void end(char const* kind, std::uint64_t id) {
  emit({{"event", kind}, {"id", id}, {"time", timestamp()}});
}
void allocation(char const* kind, Kokkos_Profiling_SpaceHandle space,
                char const* label, void const* pointer, std::uint64_t bytes) {
  emit({{"event", kind}, {"space", space.name}, {"label", label},
        {"pointer", reinterpret_cast<std::uintptr_t>(pointer)}, {"bytes", bytes},
        {"time", timestamp()}});
}
} // namespace

extern "C" {
void kokkosp_request_tool_settings(std::uint32_t, Kokkos_Tools_ToolSettings* settings) {
  settings->requires_global_fencing = false;
}
void kokkosp_init_library(int, std::uint64_t version, std::uint32_t,
                          Kokkos_Profiling_KokkosPDeviceInfo*) {
  auto path = std::getenv("C8_INTERFACE_TRACE_FILE");
  if (!path) std::abort();
  state().output.open(path);
  if (!state().output) std::abort();
  emit({{"event", "init"}, {"kokkos_tools_version", version},
        {"cupti_header_version", CUPTI_API_VERSION}, {"extra_global_fences", false}});
  check(cuptiActivityRegisterCallbacks(request, complete));
  check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
}
void kokkosp_finalize_library() {
  check(cuptiActivityFlushAll(0));
  check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  emit({{"event", "finalize"}, {"failed", state().failed.load()},
        {"dropped_records", state().dropped.load()}});
  state().output.flush();
}
void kokkosp_allocate_data(Kokkos_Profiling_SpaceHandle s, char const* n,
                            void const* p, std::uint64_t b) { allocation("allocate", s, n, p, b); }
void kokkosp_deallocate_data(Kokkos_Profiling_SpaceHandle s, char const* n,
                              void const* p, std::uint64_t b) { allocation("deallocate", s, n, p, b); }
void kokkosp_begin_deep_copy(Kokkos_Profiling_SpaceHandle dst, char const* dl, void const* dp,
                            Kokkos_Profiling_SpaceHandle src, char const* sl, void const* sp,
                            std::uint64_t bytes) {
  emit({{"event", "copy_begin"}, {"dst_space", dst.name}, {"dst_label", dl},
        {"dst", reinterpret_cast<std::uintptr_t>(dp)}, {"src_space", src.name},
        {"src_label", sl}, {"src", reinterpret_cast<std::uintptr_t>(sp)},
        {"bytes", bytes}, {"time", timestamp()}});
}
void kokkosp_end_deep_copy() { emit({{"event", "copy_end"}, {"time", timestamp()}}); }
void kokkosp_begin_parallel_for(char const* n, std::uint32_t d, std::uint64_t* i) { begin("for_begin",n,d,i); }
void kokkosp_end_parallel_for(std::uint64_t i) { end("for_end",i); }
void kokkosp_begin_parallel_scan(char const* n, std::uint32_t d, std::uint64_t* i) { begin("scan_begin",n,d,i); }
void kokkosp_end_parallel_scan(std::uint64_t i) { end("scan_end",i); }
void kokkosp_begin_parallel_reduce(char const* n, std::uint32_t d, std::uint64_t* i) { begin("reduce_begin",n,d,i); }
void kokkosp_end_parallel_reduce(std::uint64_t i) { end("reduce_end",i); }
void kokkosp_begin_fence(char const* n, std::uint32_t d, std::uint64_t* i) { begin("fence_begin",n,d,i); }
void kokkosp_end_fence(std::uint64_t i) { end("fence_end",i); }
}
