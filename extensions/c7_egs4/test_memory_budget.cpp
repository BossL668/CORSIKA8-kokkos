#include "Egs4MemoryBudget.hpp"
#include <iostream>

namespace {
void check(bool value,char const* what){if(!value)throw std::runtime_error(what);}
template<class F>void rejects(F action){bool failed=false;try{action();}catch(std::exception const&){failed=true;}
  check(failed,"Invalid budget was accepted");}
}
int main(){try {
  using c7_egs4::application::planGpuMemory;
  constexpr std::size_t GiB=1024ULL*1024*1024;
  auto idle=planGpuMemory(.9,48*GiB,47*GiB,16384);
  check(idle.requested_bytes<=48*GiB*.9&&idle.working_bytes+idle.reserve_bytes==idle.requested_bytes,
    "Requested fraction not applied");
  check(std::size_t(idle.queue_capacity)*idle.bytes_per_history<=idle.working_bytes,"Capacity exceeds budget");
  auto shared=planGpuMemory(.9,48*GiB,39*GiB,16384);
  check(shared.working_bytes+shared.reserve_bytes==39*GiB&&shared.queue_capacity<idle.queue_capacity,
    "Other GPU processes not respected");
  auto full=planGpuMemory(1.,48*GiB,48*GiB,16384);
  check(full.requested_bytes==48*GiB,"Full fraction rounded incorrectly");
  auto large=planGpuMemory(1.,std::numeric_limits<std::size_t>::max(),
    std::numeric_limits<std::size_t>::max(),1,0);
  check(large.queue_capacity==std::numeric_limits<int>::max()/5,"Scan integer limit lost");
  rejects([&]{planGpuMemory(0.,48*GiB,47*GiB,16384);});
  rejects([&]{planGpuMemory(1.1,48*GiB,47*GiB,16384);});
  rejects([&]{planGpuMemory(std::numeric_limits<double>::quiet_NaN(),48*GiB,47*GiB,16384);});
  rejects([&]{planGpuMemory(.9,48*GiB,49*GiB,16384);});
  rejects([&]{planGpuMemory(.9,48*GiB,47*GiB,0);});
  rejects([&]{planGpuMemory(.9,48*GiB,0,16384);});
  rejects([&]{planGpuMemory(.9,48*GiB,GiB,GiB);});
  std::cout<<"Native EGS4 GPU budget contracts passed; demand allocation only\n";return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
