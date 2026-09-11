#include "CooperativeActivityTrace.hpp"
#include <cupti.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace c8_overlap_trace {
  namespace {
    struct Kernel { std::uint64_t start, end; };
    std::array<Kernel, 16384> kernels;
    char const* selected_kernel = "Arithmetic";
    std::size_t count{};
    std::mutex mutex;
    bool failed{};
    std::size_t dropped{};
    void check(CUptiResult status) {
      if (status == CUPTI_SUCCESS) return;
      char const* message = "unknown CUPTI error";
      cuptiGetResultString(status, &message);
      throw std::runtime_error(message);
    }
    void CUPTIAPI request(std::uint8_t** buffer, std::size_t* size, std::size_t* maximum_records) {
      *size = 1024 * 1024;
      *maximum_records = 0;
      *buffer = static_cast<std::uint8_t*>(std::malloc(*size));
      if (!*buffer) { *size = 0; std::lock_guard<std::mutex> lock(mutex); failed = true; }
    }
    void CUPTIAPI complete(CUcontext context, std::uint32_t stream,
                           std::uint8_t* buffer, std::size_t, std::size_t valid) {
      std::lock_guard<std::mutex> lock(mutex);
      if (valid) {
        CUpti_Activity* activity = nullptr;
        CUptiResult status;
        while ((status = cuptiActivityGetNextRecord(buffer, valid, &activity)) == CUPTI_SUCCESS) {
          if (activity->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
          auto const* kernel = reinterpret_cast<CUpti_ActivityKernel9 const*>(activity);
          // A bounded set of selected diagnostic kernels, never production
          // activity from another process. The filter has static lifetime.
          if (!kernel->name || !std::strstr(kernel->name, selected_kernel)) continue;
          if (count == kernels.size()) { failed = true; continue; }
          kernels[count++] = {kernel->start, kernel->end};
        }
        if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) failed = true;
      }
      std::size_t lost{};
      if (cuptiActivityGetNumDroppedRecords(context, stream, &lost) != CUPTI_SUCCESS) failed = true;
      dropped += lost;
      std::free(buffer);
    }
  }
  void start(char const* kernel_name_substring) {
    if (!kernel_name_substring) throw std::invalid_argument("null kernel filter");
    selected_kernel = kernel_name_substring;
    check(cuptiActivityRegisterCallbacks(request, complete));
    // CONCURRENT_KERNEL preserves concurrency; KIND_KERNEL would serialize it.
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  }
  std::uint64_t timestamp() {
    std::uint64_t result{};
    check(cuptiGetTimestamp(&result));
    return result;
  }
  void finish(std::uint64_t host_start, std::uint64_t host_end) {
    finish(std::vector<std::pair<std::uint64_t,std::uint64_t>>{{host_start,host_end}});
  }
  void finish(std::vector<std::pair<std::uint64_t,std::uint64_t>> const& host_windows) {
    check(cuptiActivityFlushAll(0));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    std::lock_guard<std::mutex> lock(mutex);
    if (failed || dropped || host_windows.empty())
      throw std::runtime_error("incomplete CUPTI overlap timeline");
    std::uint64_t last=0;
    for(auto const& window:host_windows) {
      if(window.second<=window.first || window.first<last)
        throw std::runtime_error("invalid/overlapping host activity windows");
      last=window.second;
    }
    auto host_start=host_windows.front().first,host_end=host_windows.back().second;
    std::uint64_t overlap{};
    std::cout << "{\"timeline\":\"CUPTI concurrent kernel, common nanosecond clock\",\"host_start_ns\":"
              << host_start << ",\"host_end_ns\":" << host_end << ",\"host_windows\":[";
    for(std::size_t i=0;i<host_windows.size();++i) {
      if(i)std::cout<<',';
      std::cout<<'['<<host_windows[i].first<<','<<host_windows[i].second<<']';
    }
    std::cout<<"],\"cuda_kernels\":[";
    for (std::size_t i = 0; i < count; ++i) {
      auto const& kernel = kernels[i];
      if (!kernel.start || kernel.end <= kernel.start)
        throw std::runtime_error("invalid CUPTI kernel timestamps");
      if (i) std::cout << ',';
      std::cout << "{\"start_ns\":" << kernel.start << ",\"end_ns\":" << kernel.end << '}';
      for(auto const& window:host_windows) {
        auto const a=std::max(window.first,kernel.start), b=std::min(window.second,kernel.end);
        if(b>a)overlap+=b-a;
      }
    }
    std::cout << "],\"overlap_ns\":" << overlap << ",\"dropped_records\":" << dropped << "}\n";
    if (!overlap) throw std::runtime_error("no actual host/device kernel overlap in CUPTI timeline");
  }
}
