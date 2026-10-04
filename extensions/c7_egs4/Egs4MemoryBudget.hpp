#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace c7_egs4::application {
// Capacity planning only: no empty buffer is allocated to inflate GPU usage.
// The queue still allocates actual live histories and fails without dropping
// particles if its bounded capacity is exceeded. External allocations after
// this snapshot can still make CUDA allocation fail.
struct GpuMemoryBudget {
  double fraction{};
  std::size_t total_bytes{},free_bytes{},requested_bytes{},working_bytes{};
  std::size_t reserve_bytes{},bytes_per_history{};
  int queue_capacity{};
};
inline GpuMemoryBudget planGpuMemory(double fraction,std::size_t total,std::size_t free,
    std::size_t bytes_per_history,std::size_t reserve=512ULL*1024*1024) {
  if(!std::isfinite(fraction)||fraction<=0.||fraction>1.||!total||free>total||!bytes_per_history)
    throw std::invalid_argument("Invalid native EGS4 GPU memory budget");
  auto requested=fraction==1.?total:static_cast<std::size_t>(static_cast<long double>(total)*fraction);
  auto available=std::min(requested,free);
  if(available<=reserve)throw std::length_error("Insufficient GPU memory after the native EGS4 reserve");
  auto working=available-reserve;
  auto capacity=std::min(working/bytes_per_history,std::size_t(std::numeric_limits<int>::max()/5));
  if(capacity<8192)throw std::length_error("Insufficient GPU memory for the minimum native EGS4 queue");
  return {fraction,total,free,requested,working,reserve,bytes_per_history,int(capacity)};
}
} // namespace c7_egs4::application
