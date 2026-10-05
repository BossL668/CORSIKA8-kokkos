#pragma once
#include <cstdlib>
#include <stdexcept>
namespace c7_egs4::overhead {
enum : unsigned { column_cache=1, compact_clear=2, prepared_rng=4, empty_radio=8, sparse_host=16, reuse_query=32, scatter_pair=64, curved_azimuth=128, coordinate_reuse=256 };
// Test-only, sampled once at construction; no process environment access on device.
inline unsigned options() {
  auto p=std::getenv("EGS4_OVERHEAD_MASK");if(!p)return 0;
  char* end=nullptr;auto value=std::strtoul(p,&end,0);
  if(!end||*end||value>511)throw std::invalid_argument("EGS4_OVERHEAD_MASK must be 0..511");
  return unsigned(value);
}
inline bool noAntennas(){auto p=std::getenv("EGS4_AUDIT_NO_ANTENNAS");return p&&p[0]=='1';}
}
