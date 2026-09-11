#include <corsika/accelerator/em/detail/AdaptiveSubshowerControl.hpp>
#include <iostream>
#include <stdexcept>
using corsika::accelerator::em::detail::AdaptiveSubshowerControl;
void require(bool value,char const* message) {
  if(!value) throw std::runtime_error(message);
}
int main() {
  try {
    for(unsigned fast:{0U,1U}) {
      AdaptiveSubshowerControl c;
      require(c.waves(0,1,65536)==1 && c.waves(1,1,65536)==1,
              "unmeasured endpoint received a long lease");
      for(unsigned k:{0U,1U}) for(unsigned e:{0U,1U}) {
        std::size_t requested=1;
        for(int i=0;i<12;++i) {
          c.observe(e,k,4096,requested,requested,4096*requested,
                    (e==fast?.01:5.)*requested);
          requested=c.waves(e,k,4096);
        }
      }
      require(c.unitCost(fast,1)*100<c.unitCost(1-fast,1),
              "throughput estimate biased by endpoint identity");
      require(c.waves(fast,1,4096)==1024,"fast CPU/GPU wave budget restricted");
      require(c.waves(1-fast,1,4096)<64,"slow endpoint lease not adapted");
      require(c.waves(fast,0,4096)<=16,"photon history lease limit broken");
      auto before=c.sample(fast,1).observations;
      c.observe(fast,1,4096,0,8,4096,10.);
      require(c.sample(fast,1).observations==before,"allocation refusal polluted estimate");
      require(c.waves(fast,1,100000000)<c.waves(fast,1,4096),
              "lease prediction ignores input size");
    }
    AdaptiveSubshowerControl service;
    require(service.inputLimit(1,1,65536,5.)==2048,"cold CPU quantum fills full arena");
    service.observe(1,1,65536,8,8,65536*8,200.);
    auto budget=service.inputLimit(1,1,65536,5.);
    require(budget>=256 && budget<2048,"large CPU batch not shortened for GPU service");
    require(service.waves(1,1,budget,5.)<=8,"CPU time budget ignored by wave lease");
    require(service.inputLimit(1,1,128,5.)==128,"explicit small arena cap exceeded");
    service.observeGpuDuration(20.);
    require(service.hostTargetMs(true)<5.,"fast GPU does not shorten host quantum");
    for(int i=0;i<20;++i)service.observeGpuDuration(1000.);
    require(service.hostTargetMs(true)==10.,"slow GPU does not relax host quantum");
    require(service.hostTargetMs(false)==200.,"idle GPU unnecessarily limits CPU work");
    require(service.inputLimit(1,1,65536,200.)>budget,"full CPU budget cannot recover");
    std::cout<<"PASS: symmetric fast-CPU/fast-GPU estimates, cold probes, native wave ceilings, invalid observations\n";
  } catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
}
